# Reproducing the modeling workflow

The repository provides the complete supplied Python source but **does not bundle the original synthetic traffic dataset or trained checkpoints**. Executing model training requires reconstructing the input directories documented in [Data and input formats](data.md).

## Environment

Python **3.10–3.12** is recommended; the original environment used Python 3.12. The dependency list is provided at the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd pipeline
```

## Data preparation

All commands below run from `pipeline/` and expect the original CityFlow Manhattan input files under `pipeline/Manhattan Network/MPLight/data/manhattan/`.

```bash
python create_zones.py --help
python create_zones.py
python generate_link_od.py
python extract_link_counts.py
python convert_zonal_od.py --help
python convert_zonal_od.py
python aggregate_zonal_counts.py
```

The OD-conversion and traffic-aggregation scripts have their own naming and argument conventions. Their output layout should be examined after each preprocessing stage before proceeding to the next one.

## Training and tuning

```bash
python train_od_mlp.py --K 20
python tune_hyperparameters.py --n_trials 40 --pruner hyperband --report_every 10
python train_od_mlp.py --K 20 --hparams_json Results/best_hparams_optuna.json
```

A separate implementation variant is retained as `train_od_mlp_variant.py`; it is not silently substituted for the primary training script. The tuner invokes the renamed training script; this is the only executable-code adjustment beyond filename normalization.

## Reduced-data sensitivity

```bash
python aggregate_zonal_counts_masked.py --help
python train_od_mlp.py --K 20 --counts_glob 'Refined Data/Traffic counts_zones_deletion/manhattan_*_traffic-counts-zone-*.csv'
```

The exact mask percentage, seed, dataset state, and training configuration must be aligned to compare trials. The original experimental metrics have **not** been independently replicated with the packaged files.
