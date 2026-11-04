import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, Any, Tuple


BEST_RE = re.compile(r"^BEST_VAL_SMAPE\s+([0-9]*\.?[0-9]+)\s*$", re.MULTILINE)
TUNE_RE = re.compile(r"^TUNE_REPORT\s+(\d+)\s+([0-9]*\.?[0-9]+)\s*$")


def run_trial_streaming(model_py: str, trial_args: Dict[str, Any], trial=None, report_every: int = 10) -> float:
    """
    Run train_od_mlp.py, stream its output, and optionally report intermediate values to an Optuna trial.
    If trial is provided, may raise optuna.TrialPruned when the pruner decides to stop.
    """
    cmd = [sys.executable, model_py]
    cmd += ["--skip_save", "--print_best_val", "--no-excel_output", "--tune_report_every", str(report_every)]

    for k, v in trial_args.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            cmd.append(flag if v else f"--no-{k}")
        else:
            cmd += [flag, str(v)]

    # Merge stderr into stdout so we can parse progress reliably
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None

    best_val = None
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        s = line.strip()
        m = TUNE_RE.match(s)
        if m and trial is not None:
            step = int(m.group(1))
            val = float(m.group(2))
            trial.report(val, step=step)
            if trial.should_prune():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                # Lazy import to avoid hard dependency in type-checking contexts
                import optuna
                raise optuna.TrialPruned()
        m2 = BEST_RE.match(s)
        if m2:
            best_val = float(m2.group(1))

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Model failed (exit {rc}). Last best={best_val}")
    if best_val is None:
        raise RuntimeError("Could not find BEST_VAL_SMAPE in train_od_mlp.py output.")
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_py", default="train_od_mlp.py")
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--timeout_min", type=int, default=0, help="0 = no timeout")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", default="Results/best_hparams_optuna.json")
    ap.add_argument("--pruner", type=str, default="hyperband", choices=["none", "median", "hyperband"],
                    help="Prune bad trials early to run more configs in same time.")
    ap.add_argument("--report_every", type=int, default=10, help="Epoch interval for pruning reports.")

    # Allow overriding dataset globs here if needed
    ap.add_argument("--counts_glob", default=None)
    ap.add_argument("--od_glob", default=None)
    ap.add_argument("--K", type=int, default=None)

    args = ap.parse_args()

    try:
        import optuna
    except Exception as e:
        raise SystemExit(
            "Optuna is not installed. Install it with:\n"
            "  pip install optuna\n"
            f"\nOriginal error: {e}"
        )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    # TPE (Bayesian) is a strong default for noisy, non-convex objectives like NN training.
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    if args.pruner == "none":
        pruner = optuna.pruners.NopPruner()
    elif args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(50, args.report_every * 3))
    else:
        # ASHA/Hyperband-like behavior
        pruner = optuna.pruners.HyperbandPruner()

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    base: Dict[str, Any] = {"seed": args.seed}
    if args.counts_glob is not None:
        base["counts_glob"] = args.counts_glob
    if args.od_glob is not None:
        base["od_glob"] = args.od_glob
    if args.K is not None:
        base["K"] = args.K

    def objective(trial: "optuna.Trial") -> float:
        # Search space: focuses on the knobs most likely to move sMAPE in your current code.
        trial_args: Dict[str, Any] = dict(base)

        trial_args["lr"] = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        trial_args["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 3e-2, log=True)
        trial_args["batch_size"] = trial.suggest_categorical("batch_size", [16, 24, 32, 48, 64])
        trial_args["dropout"] = trial.suggest_float("dropout", 0.0, 0.45)

        trial_args["hidden"] = trial.suggest_categorical(
            "hidden",
            [
                "512,256,128",
                "512,512,256",
                "256,128,64",
                "384,192,96",
                "768,384,192",
                "512,256",
                "256,128",
            ],
        )

        # Early stopping makes huge epoch counts unnecessary; keep a reasonable max and tune patience.
        trial_args["epochs"] = trial.suggest_int("epochs", 400, 1600, step=100)
        trial_args["patience"] = trial.suggest_int("patience", 60, 260, step=20)

        # Loss weights (these often matter more than epochs/batch size for OD structure fidelity)
        trial_args["lambda_marg"] = trial.suggest_float("lambda_marg", 0.0, 3.5)
        trial_args["lambda_diag"] = trial.suggest_float("lambda_diag", 0.0, 1.0)
        trial_args["lambda_pct"] = trial.suggest_float("lambda_pct", 0.0, 6.0)
        trial_args["lambda_log"] = trial.suggest_float("lambda_log", 0.0, 1.2)
        trial_args["lambda_total"] = trial.suggest_float("lambda_total", 0.0, 20.0)

        trial_args["use_total_head"] = trial.suggest_categorical("use_total_head", [True, False])
        trial_args["use_magnitude_weighting"] = trial.suggest_categorical("use_magnitude_weighting", [True, False])
        trial_args["use_huber"] = trial.suggest_categorical("use_huber", [True, False])
        if trial_args["use_huber"]:
            trial_args["huber_delta"] = trial.suggest_float("huber_delta", 0.3, 1.2)

        # Stabilization in your sMAPE computation and loss
        trial_args["pct_eps"] = trial.suggest_float("pct_eps", 0.0, 40.0)

        # IPF (post-correction) can be a big lever on sMAPE for OD matrices.
        trial_args["use_ipf"] = trial.suggest_categorical("use_ipf", [True, False])
        if trial_args["use_ipf"]:
            trial_args["ipf_rho"] = trial.suggest_float("ipf_rho", 0.92, 1.0)
            trial_args["ipf_beta"] = trial.suggest_float("ipf_beta", 0.02, 0.35)
            trial_args["ipf_iter"] = trial.suggest_int("ipf_iter", 20, 80, step=10)
            trial_args["ipf_blend_alpha"] = trial.suggest_float("ipf_blend_alpha", 0.0, 0.6)
            trial_args["ipf_clip_min"] = trial.suggest_float("ipf_clip_min", 0.01, 0.2)
            trial_args["ipf_clip_max"] = trial.suggest_float("ipf_clip_max", 3.0, 12.0)

        # Give each trial a deterministic but different seed by default
        trial_args["seed"] = int(args.seed + trial.number)

        # A few additional high-leverage toggles to explore
        trial_args["log_x"] = trial.suggest_categorical("log_x", [True, False])
        trial_args["x_only_summaries"] = trial.suggest_categorical("x_only_summaries", [False, True])
        trial_args["global_rescale"] = trial.suggest_categorical("global_rescale", [True, False])
        trial_args["gamma_sharpen"] = trial.suggest_float("gamma_sharpen", 1.0, 1.6)

        val_smape = run_trial_streaming(args.model_py, trial_args, trial=trial, report_every=args.report_every)
        return val_smape

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=None if args.timeout_min <= 0 else args.timeout_min * 60,
        show_progress_bar=True,
    )

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "note": "Run train_od_mlp.py once with these params (and without --skip_save) to generate final artifacts.",
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\nBest validation sMAPE:", study.best_value)
    print("Saved:", args.out_json)
    print("\nSuggested final run (example):")
    cmd = [sys.executable, args.model_py]
    for k, v in study.best_params.items():
        if isinstance(v, bool):
            cmd.append(f"--{k}" if v else f"--no-{k}")
        else:
            cmd += [f"--{k}", str(v)]
    print(" ".join(cmd))


if __name__ == "__main__":
    main()

