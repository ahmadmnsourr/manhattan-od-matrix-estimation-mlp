#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train MLP: zone counts (X) -> zone OD (Y)

OPTION A (aka "Option One"):
- Enforce a fixed number of slots per level (e.g., 12 five‑minute windows).
- Split exactly 8 slots to TRAIN, 2 slots to VAL, 2 slots to TEST within each level.
- Optionally aggregate consecutive slots into chunks (e.g., chunk_size=2 => 10‑minute samples).
- Rest of the training/eval pipeline remains identical (scaler fit on TRAIN only, etc.).

Why: ensures train/val/test are comparable in duration and scale, avoiding
situations where one split looks better/worse purely due to having more minutes.
"""

import os, re, glob, argparse, json
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For Excel output
import openpyxl
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# ---------------------------
# Filename helpers (level, start, end)
# ---------------------------
# Support both old and new filename patterns, e.g.:
#  - manhattan_<level>_traffic-counts-<start>_<end>_zone.csv
#  - manhattan_<level>_traffic-counts-zone-<start>_<end>.csv
#  - manhattan_<level>_zone-od-<start>_<end>_K20.csv
FN_RE = re.compile(
    r"manhattan_(\d+)_"
    r"(?:traffic[-_]?counts(?:[-_]zone)?|zone[-_]?od(?:-matrix)?)"
    r"[-_](\d+)_(\d+)"
    r"(?:_zone)?(?:_K\d+)?\.csv$",
    re.IGNORECASE,
)

def parse_triplet(path: str) -> Tuple[str, int, int]:
    """Extract (level, start, end) from filename."""
    m = FN_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Unexpected filename: {os.path.basename(path)}")
    return m.group(1), int(m.group(2)), int(m.group(3))

# ---------------------------
# Loaders
# ---------------------------
def load_edge_counts_csv(path: str, K: int) -> np.ndarray:
    """
    Load edge-based traffic counts for one window.
    Expect columns: ['zone_o','zone_d','traffic_count']
    Returns: (K, K) float32 matrix with counts aggregated per (o,d).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"zone_o", "zone_d", "traffic_count"}
    if not required <= set(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    M = np.zeros((K, K), dtype=np.float32)
    for oz, dz, c in df[["zone_o", "zone_d", "traffic_count"]].itertuples(index=False, name=None):
        oz_i = int(oz); dz_i = int(dz)
        if 0 <= oz_i < K and 0 <= dz_i < K:
            M[oz_i, dz_i] += float(c)
    # Sanity check on internal rows presence
    if (df["zone_o"] == df["zone_d"]).any():
        pass  # internal circulation provided
    return M

def load_zone_od(path: str, K: int) -> np.ndarray:
    """
    Load zone OD for one window.
    Accepts:
      - Long: columns ['oz','dz','count'] -> returns KxK dense
      - Dense: KxK CSV with first column index 'oz' or unnamed; columns are zones
    Returns: (K, K) float32
    """
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    # Long format
    if {"oz","dz","count"} <= set(cols):
        M = np.zeros((K, K), dtype=np.float32)
        for oz, dz, c in df[["oz","dz","count"]].itertuples(index=False, name=None):
            if 0 <= int(oz) < K and 0 <= int(dz) < K:
                M[int(oz), int(dz)] = float(c)
        return M
    # Dense format
    # try to detect if first col is index
    if "oz" in cols:
        df = df.set_index("oz")
    # ensure we have at least K columns with 0..K-1 names or can coerce
    try:
        # try convert col names to int zones
        df.columns = [int(str(c)) for c in df.columns]
        if df.index.dtype != np.int64 and df.index.dtype != np.int32:
            df.index = df.index.astype(int)
        # reindex to full [0..K-1]
        df = df.reindex(index=range(K), columns=range(K), fill_value=0.0)
        return df.values.astype(np.float32)
    except Exception:
        raise ValueError(f"{path}: could not parse OD format (need long oz,dz,count or dense KxK).")

# ---------------------------
# Dataset builder
# ---------------------------
def discover_pairs(counts_glob: str, od_glob: str) -> List[Tuple[str, str, str, int, int]]:
    """
    Find matching (level,start,end) triples across counts & od files.
    Returns list of tuples: (level, counts_path, od_path, start, end)
    """
    # Resolve globs relative to this script's directory so CLI cwd does not matter
    script_dir = os.path.dirname(os.path.abspath(__file__))
    counts_glob_abs = os.path.normpath(os.path.join(script_dir, counts_glob))
    od_glob_abs = os.path.normpath(os.path.join(script_dir, od_glob))
    counts_files = glob.glob(counts_glob_abs)
    od_files     = glob.glob(od_glob_abs)
    if not counts_files: raise FileNotFoundError(f"No counts files match {counts_glob}")
    if not od_files:     raise FileNotFoundError(f"No OD files match {od_glob}")
    idx_counts: Dict[Tuple[str,int,int], str] = {}
    idx_od: Dict[Tuple[str,int,int], str] = {}
    for p in counts_files:
        lvl, st, en = parse_triplet(p)
        idx_counts[(lvl,st,en)] = p
    for p in od_files:
        lvl, st, en = parse_triplet(p)
        idx_od[(lvl,st,en)] = p
    keys = sorted(set(idx_counts.keys()) & set(idx_od.keys()), key=lambda x:(x[0],x[1],x[2]))
    pairs = [(lvl, idx_counts[(lvl,st,en)], idx_od[(lvl,st,en)], st, en) for (lvl,st,en) in keys]
    if not pairs:
        raise RuntimeError("No overlapping windows between counts and OD files.")
    return pairs

def build_matrices(pairs, K: int, summaries_only: bool = False):
    """
    Build X (zone counts) and Y (zone OD flattened) across all windows.
    X shape: (N, F)   (edge feats)
    Y shape: (N, K*K) (flattened row-major)
    Also returns: levels list aligned with rows, (start,end) list, and marginals from input
    """
    X_rows, Y_rows, levels, windows, marginals_from_input = [], [], [], [], []
    for (lvl, p_counts, p_od, st, en) in pairs:
        # New input: edge-based counts per (zone_o, zone_d)
        Mc = load_edge_counts_csv(p_counts, K=K)
        x_vec = edge_feats(Mc.reshape(-1), K, summaries_only)
        X_rows.append(x_vec.astype(np.float32))
        M = load_zone_od(p_od, K=K)
        Y_rows.append(M.reshape(-1).astype(np.float32))
        levels.append(lvl)
        windows.append((st,en))
        # Store marginals from input edge counts for realistic IPF
        dep_in = Mc.sum(axis=1)  # departures per origin zone
        arr_in = Mc.sum(axis=0)  # arrivals per destination zone
        marginals_from_input.append((dep_in, arr_in))
    X = np.vstack(X_rows)
    Y = np.vstack(Y_rows)
    return X, Y, levels, windows, marginals_from_input

# ---------------------------
# NEW: Fixed‑slot split & optional chunk aggregation
# ---------------------------
def _chunk(list_idx: List[int], chunk_size: int) -> Tuple[List[List[int]], int]:
    """Split list of indices into consecutive chunks of equal size; drop remainder.
    Returns: (chunks, dropped_count)"""
    if chunk_size <= 1:
        return [[i] for i in list_idx], 0
    n_full = (len(list_idx) // chunk_size) * chunk_size
    dropped_count = len(list_idx) - n_full
    list_idx = list_idx[:n_full]
    chunks = [list_idx[i:i+chunk_size] for i in range(0, n_full, chunk_size)]
    return chunks, dropped_count

def _aggregate_groups(X, Y, levels, windows, marginals, groups: List[List[int]]):
    """Aggregate rows in each group by simple sum; propagate level and window span."""
    X_out, Y_out, L_out, W_out, M_out = [], [], [], [], []
    for g in groups:
        Xg = np.sum([X[i] for i in g], axis=0)
        Yg = np.sum([Y[i] for i in g], axis=0)
        lvl = levels[g[0]]
        st0, _ = windows[g[0]]
        _, en1 = windows[g[-1]]
        dep_sum = np.sum([marginals[i][0] for i in g], axis=0)
        arr_sum = np.sum([marginals[i][1] for i in g], axis=0)
        X_out.append(Xg); Y_out.append(Yg); L_out.append(lvl); W_out.append((st0, en1)); M_out.append((dep_sum, arr_sum))
    return np.vstack(X_out), np.vstack(Y_out), L_out, W_out, M_out

def aggregate_predictions_for_comparison(Y_pred_individual, groups, levels, windows, marginals, K, original_to_split_mapping=None):
    """
    Aggregate individual predictions to match the chunked structure for fair comparison.
    This ensures we compare aggregated predictions vs aggregated true values.
    """
    if not groups:
        return np.empty((0, K*K)), [], [], []
    
    Y_agg, L_agg, W_agg, M_agg = [], [], [], []
    for g in groups:
        if not g:  # Skip empty groups
            continue
            
        # Map original indices to split indices if mapping provided
        if original_to_split_mapping is not None:
            g_mapped = [original_to_split_mapping[i] for i in g if i in original_to_split_mapping]
        else:
            g_mapped = g
        
        if not g_mapped:
            continue
            
        # Aggregate predictions for this group
        Yg = np.sum([Y_pred_individual[i] for i in g_mapped], axis=0)
        
        # Use original indices for metadata (levels, windows, marginals)
        # but ensure we have valid indices
        if g[0] < len(levels):
            lvl = levels[g[0]]
        else:
            lvl = "unknown"
        if g[0] < len(windows):
            st0, _ = windows[g[0]]
        else:
            st0 = 0
        if g[-1] < len(windows):
            _, en1 = windows[g[-1]]
        else:
            en1 = 0
            
        # For marginals, use the original indices but check bounds
        dep_sum = np.sum([marginals[i][0] for i in g if i < len(marginals)], axis=0)
        arr_sum = np.sum([marginals[i][1] for i in g if i < len(marginals)], axis=0)
        Y_agg.append(Yg)
        L_agg.append(lvl)
        W_agg.append((st0, en1))
        M_agg.append((dep_sum, arr_sum))
    
    return np.vstack(Y_agg), L_agg, W_agg, M_agg

def fixedslot_split_and_chunk(
    X, Y, levels, windows, marginals,
    keep_first_slots: int = 12,
    split_slots: Tuple[int,int,int] = (8,2,2),
    chunk_size: int = 2
):
    """
    Enforce exactly keep_first_slots per level. Within those, allocate
    the first split_slots=(tr, va, te) SLOTS to TRAIN/VAL/TEST.
    Keep individual time slots separate for fair prediction comparison.
    Returns split arrays for X/Y/levels/windows/marginals and chunk groups.
    """
    assert sum(split_slots) <= keep_first_slots, "split_slots must fit within keep_first_slots"
    levels_unique = sorted(set(levels))
    by_level: Dict[str, List[int]] = {lv: [] for lv in levels_unique}
    for i, lv in enumerate(levels):
        by_level[lv].append(i)
    # sort each level by window start time
    for lv in levels_unique:
        by_level[lv] = sorted(by_level[lv], key=lambda i: windows[i][0])

    TR, VA, TE = split_slots
    # Collect per-split groups (of indices)
    groups_tr, groups_va, groups_te = [], [], []
    total_dropped = 0
    levels_processed = 0
    
    for lv in levels_unique:
        rows = by_level[lv]
        # limit to the first keep_first_slots (and drop any partial tail when chunking)
        rows = rows[:keep_first_slots]
        if len(rows) < sum(split_slots):
            # skip levels that don't have enough slots
            print(f"Warning: Level {lv} has only {len(rows)} slots, skipping (need {sum(split_slots)})")
            continue
        levels_processed += 1
        
        # train rows [0:TR), val [TR:TR+VA), test [TR+VA:TR+VA+TE)
        rows_tr = rows[:TR]
        rows_va = rows[TR:TR+VA]
        rows_te = rows[TR+VA:TR+VA+TE]

        # chunk within each split
        chunks_tr, dropped_tr = _chunk(rows_tr, chunk_size)
        chunks_va, dropped_va = _chunk(rows_va, chunk_size)
        chunks_te, dropped_te = _chunk(rows_te, chunk_size)
        
        level_dropped = dropped_tr + dropped_va + dropped_te
        total_dropped += level_dropped
        if level_dropped > 0:
            print(f"Level {lv}: dropped {level_dropped} slots due to chunking (chunk_size={chunk_size})")
        
        groups_tr += chunks_tr
        groups_va += chunks_va
        groups_te += chunks_te

    if total_dropped > 0:
        print(f"Total dropped samples across all levels: {total_dropped} (chunk_size={chunk_size})")
    print(f"Processed {levels_processed} levels with sufficient data")

    # Extract individual time slots (no aggregation yet)
    def extract_individual_slots(groups):
        if not groups:
            return np.empty((0, X.shape[1])), np.empty((0, Y.shape[1])), [], [], [], {}
        all_indices = []
        for group in groups:
            all_indices.extend(group)
        all_indices = sorted(all_indices)
        # Create mapping from original indices to split indices
        original_to_split = {orig_idx: split_idx for split_idx, orig_idx in enumerate(all_indices)}
        return X[all_indices], Y[all_indices], [levels[i] for i in all_indices], [windows[i] for i in all_indices], [marginals[i] for i in all_indices], original_to_split

    # Get individual time slots for each split
    Xtr, Ytr, Ltr, Wtr, Mtr, mapping_tr = extract_individual_slots(groups_tr)
    Xva, Yva, Lva, Wva, Mva, mapping_va = extract_individual_slots(groups_va)
    Xte, Yte, Lte, Wte, Mte, mapping_te = extract_individual_slots(groups_te)

    return (Xtr, Ytr, Ltr, Wtr, Mtr), (Xva, Yva, Lva, Wva, Mva), (Xte, Yte, Lte, Wte, Mte), (groups_tr, groups_va, groups_te), (mapping_tr, mapping_va, mapping_te)

# ---------------------------
# Splitting (time-ordered per level) – original utility
# ---------------------------
def split_per_level(levels: List[str], windows: List[Tuple[int,int]], tr=0.7, va=0.15):
    levels_unique = sorted(set(levels))
    idx_tr, idx_va, idx_te = [], [], []
    # group rows by level, in time order
    by_level: Dict[str, List[int]] = {lv: [] for lv in levels_unique}
    for i, lv in enumerate(levels):
        by_level[lv].append(i)
    for lv in levels_unique:
        rows = by_level[lv]
        # sort by window start
        rows = sorted(rows, key=lambda i: windows[i][0])
        T = len(rows)
        n_tr = max(1, int(tr*T))
        n_va = max(1, int(va*T))
        idx_tr += rows[:n_tr]
        idx_va += rows[n_tr:n_tr+n_va]
        idx_te += rows[n_tr+n_va:]
    return np.array(idx_tr), np.array(idx_va), np.array(idx_te), levels_unique

def split_across_levels(levels_unique: List[str], tr=0.7, va=0.15):
    """
    Split indices across unique levels (no time ordering). Uses deterministic
    ordering by level name/string to keep reproducible splits.
    Returns idx_tr, idx_va, idx_te over range(len(levels_unique)).
    """
    L = len(levels_unique)
    n_tr = max(1, int(tr * L))
    n_va = max(1, int(va * L))
    order = list(range(L))  # already sorted by caller typically
    idx_tr = np.array(order[:n_tr])
    idx_va = np.array(order[n_tr:n_tr+n_va])
    idx_te = np.array(order[n_tr+n_va:])
    return idx_tr, idx_va, idx_te

# ---------------------------
# Model
# ---------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(64,32), dropout=0.1, use_total_head=False):
        super().__init__()
        feats = []
        prev = in_dim
        
        # Enhanced architecture with residual connections and batch normalization
        for h in hidden:
            feats += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        
        self.feats = nn.Sequential(*feats)
        self.out = nn.Linear(prev, out_dim)
        
        # Add residual connection for the last hidden layer
        self.residual_proj = nn.Linear(in_dim, hidden[-1]) if len(hidden) > 0 and in_dim != hidden[-1] else None
        
        # Optional total-flow head for better total prediction
        self.use_total_head = use_total_head
        if use_total_head:
            self.total_head = nn.Sequential(
                nn.Linear(in_dim, max(32, in_dim // 2)),
                nn.BatchNorm1d(max(32, in_dim // 2)),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(max(32, in_dim // 2), max(16, in_dim // 4)),
                nn.ReLU(),
                nn.Linear(max(16, in_dim // 4), 1)  # outputs log_total
            )
    
    def forward(self, x):
        h = self.feats(x) if len(self.feats) > 0 else x
        if self.residual_proj is not None:
            h = h + self.residual_proj(x)
        y = self.out(h)
        
        if self.use_total_head:
            log_total = self.total_head(x)
            return y, log_total
        return y

# ---------------------------
# Loss helpers (unchanged)
# ---------------------------
def log1p_(A: np.ndarray) -> np.ndarray: return np.log1p(A)
def expm1_(A: np.ndarray) -> np.ndarray: return np.expm1(A)

def marginals(M: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """Row and column sums for a batch of matrices [N,K*K] or [N,K,K]."""
    if M.ndim == 2:
        # reshape
        N, KK = M.shape
        K = int(np.sqrt(KK))
        M = M.reshape(N, K, K)
    row = M.sum(axis=2)  # departures per zone
    col = M.sum(axis=1)  # arrivals per zone
    return row, col

def mae(a,b): return float(np.mean(np.abs(a-b)))
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def mape(a,b): 
    """Mean Absolute Percentage Error: mean(|a-b|/|b|) * 100"""
    mask = np.abs(b) > 1e-9  # Avoid division by zero
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(a[mask] - b[mask]) / np.abs(b[mask])) * 100)

def smape_np(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 0.0) -> float:
    num = 2.0 * np.abs(y_pred - y_true)
    den = np.abs(y_true) + np.abs(y_pred)
    if eps > 0:
        den = np.maximum(den, eps)
    return float(np.mean(num / den) * 100.0)

def smape_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    num = 2.0 * torch.abs(y_pred - y_true)
    den = torch.abs(y_true) + torch.abs(y_pred)
    if eps > 0:
        den = torch.maximum(den, torch.tensor(eps))
    return (num / den).mean()

def pct_loss_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 5.0) -> torch.Tensor:
    return (torch.abs(y_pred - y_true) / (torch.abs(y_true) + eps)).mean()

def magnitude_weights(y_true: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute magnitude weights: sqrt(y_true) / mean(sqrt(y_true))
    This prevents big cells from being washed out by the loss.
    """
    sqrt_y = torch.sqrt(torch.clamp(y_true, min=eps))
    mean_sqrt = sqrt_y.mean()
    weights = sqrt_y / (mean_sqrt + eps)
    return weights

def log_space_loss(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Log-space MSE loss to preserve dynamic range.
    """
    log_true = torch.log1p(torch.clamp(y_true, min=0.0))
    log_pred = torch.log1p(torch.clamp(y_pred, min=0.0))
    return torch.nn.functional.mse_loss(log_pred, log_true)

def apply_global_rescale(Y_pred, input_marginals, estimate_total_from_counts=None, clip=(0.2, 5.0), eps=1e-9):
    Y_pred_rescaled = Y_pred.copy()
    for i in range(len(Y_pred)):
        dep_in, arr_in = input_marginals[i]
        if estimate_total_from_counts is not None:
            target_total = estimate_total_from_counts(dep_in, arr_in, Y_pred[i].sum())
        else:
            target_total = 0.5 * (dep_in.sum() + arr_in.sum())
        current_total = Y_pred[i].sum()
        if current_total > eps and target_total > eps:
            scale_factor = np.clip(target_total / current_total, clip[0], clip[1])
            Y_pred_rescaled[i] = Y_pred[i] * scale_factor
    return Y_pred_rescaled

def sharpen_gamma(Y, gamma=1.25, eps=1e-12):
    Yg = np.power(np.maximum(Y, eps), gamma)
    s = Yg.sum(axis=1, keepdims=True) + eps
    return Yg * (Y.sum(axis=1, keepdims=True) / s)

def make_consistent_marginals(dep_in, arr_in, target_total, eps=1e-9):
    dep = np.maximum(dep_in.astype(np.float64), 0.0)
    arr = np.maximum(arr_in.astype(np.float64), 0.0)
    Sd = dep.sum(); Sa = arr.sum()
    if Sd < eps or Sa < eps:
        K = dep.shape[-1]
        dep = np.ones_like(dep) * (target_total / K)
        arr = np.ones_like(arr) * (target_total / K)
        return dep, arr
    scale = target_total / (0.5 * (Sd + Sa))
    dep *= scale; arr *= scale
    Sa2 = arr.sum()
    if Sa2 > eps:
        arr *= (dep.sum() / Sa2)
    dep = np.maximum(dep, eps); arr = np.maximum(arr, eps)
    return dep, arr

def fit_counts_to_od_calibration(tr_marginals, Ytr_raw):
    train_counts_tot = np.array([0.5*(d.sum()+a.sum()) for (d,a) in tr_marginals], dtype=np.float64)
    train_od_tot = Ytr_raw.sum(axis=1).astype(np.float64)
    a, b = np.polyfit(train_counts_tot, train_od_tot, 1)
    def estimate_total_from_counts(dep, arr, model_total):
        ctot = 0.5*(dep.sum()+arr.sum())
        est = a*ctot + b
        return float(est)
    return estimate_total_from_counts

def ipf_adjust(Y_pred, row_targets, col_targets, n_iter=10, clip=(0.1, 10.0), target_total=None):
    M = np.array(Y_pred, copy=True)
    M[M < 1e-12] = 1e-12
    batch_one = False
    if M.ndim == 1:
        K = int(np.sqrt(M.shape[0]))
        M = M.reshape(1, K, K)
        batch_one = True
    elif M.ndim == 2 and M.shape[0] == M.shape[1]:
        K = M.shape[0]
        M = M.reshape(1, K, K)
        batch_one = True
    elif M.ndim == 2:
        N, KK = M.shape
        K = int(np.sqrt(KK))
        M = M.reshape(N, K, K)
    else:
        N, K, _ = M.shape
    rt = np.array(row_targets, dtype=np.float64)
    ct = np.array(col_targets, dtype=np.float64)
    if rt.ndim == 1:
        rt = rt.reshape(1, K)
    if ct.ndim == 1:
        ct = ct.reshape(1, K)
    lo, hi = clip
    for _ in range(n_iter):
        r = M.sum(axis=2, keepdims=True); r = np.maximum(r, 1e-12)
        M *= np.clip(rt.reshape(M.shape[0], K, 1) / r, lo, hi)
        c = M.sum(axis=1, keepdims=True); c = np.maximum(c, 1e-12)
        M *= np.clip(ct.reshape(M.shape[0], 1, K) / c, lo, hi)
        if target_total is not None:
            tt = np.array(target_total, ndmin=1, dtype=np.float64)
            if tt.size == 1:
                M *= (tt[0] / np.maximum(M.sum(), 1e-12))
            else:
                cur = M.reshape(M.shape[0], -1).sum(axis=1, keepdims=True)
                scale = tt.reshape(-1,1) / np.maximum(cur, 1e-12)
                M *= scale.reshape(-1,1,1)
        np.nan_to_num(M, copy=False, nan=1e-12, posinf=1e12, neginf=1e-12)
        M[M < 1e-12] = 1e-12
    return M[0] if batch_one else M

def edge_feats(Mc_flat: np.ndarray, K: int, summaries_only: bool = False) -> np.ndarray:
    Mc = Mc_flat.reshape(K, K)
    dep  = Mc.sum(axis=1)
    arr  = Mc.sum(axis=0)
    diag = np.diag(Mc)
    if summaries_only:
        return np.concatenate([dep, arr, diag], axis=0)
    else:
        return np.concatenate([Mc_flat, dep, arr, diag], axis=0)

def create_excel_output(Y_true, Y_pred, Y_pred_ipf, levels, windows, K, save_dir, split_name, pct_eps=10.0):
    # Ensure we have the right shape
    if Y_true.ndim == 2:
        Y_true = Y_true.reshape(-1, K, K)
    if Y_pred.ndim == 2:
        Y_pred = Y_pred.reshape(-1, K, K)
    if Y_pred_ipf is not None and Y_pred_ipf.ndim == 2:
        Y_pred_ipf = Y_pred_ipf.reshape(-1, K, K)
    # Create zone labels
    zone_labels = [f"Zone_{i}" for i in range(K)]
    # ----- Split-level summary (same as console) -----
    # Y_pred is already the final data (after IPF if enabled), so use it directly
    summary = summarize_levelwise(Y_true, Y_pred, levels, K, pct_eps=pct_eps)
    # Group data by level
    level_data = {}
    for i in range(len(Y_true)):
        level = levels[i]
        if level not in level_data:
            level_data[level] = {'true_matrices': [], 'pred_matrices': [], 'pred_ipf_matrices': [], 'windows': []}
        level_data[level]['true_matrices'].append(Y_true[i])
        level_data[level]['pred_matrices'].append(Y_pred[i])
        if Y_pred_ipf is not None:
            level_data[level]['pred_ipf_matrices'].append(Y_pred_ipf[i])
        level_data[level]['windows'].append(windows[i])
    excel_filename = os.path.join(save_dir, f"OD_matrices_{split_name.lower()}_tunned.xlsx")
    wb = Workbook()
    wb.remove(wb.active)
    ws_summary = wb.create_sheet(title="Summary")
    ws_summary['A1'] = f"== {split_name} =="
    ws_summary['A2'] = f"sMAPE(all): {summary['smape_all']:.2f}% (PRIMARY)"
    ws_summary['A3'] = f"MAE(all): {summary['mae_all']:.6f} | RMSE(all): {summary['rmse_all']:.6f}"
    ws_summary['A4'] = f"MAE(nz) : {summary['mae_nz']:.6f} | RMSE(nz) : {summary['rmse_nz']:.6f} | sMAPE(nz): {summary['smape_nz']:.2f}%"
    ws_summary['A5'] = f"Marginal MAE (rows/cols avg): {summary['marg_mae']:.6f}"
    ws_summary['A6'] = f"Total Traffic - True: {summary['total_true']:.2f} | Pred: {summary['total_pred']:.2f} | Error: {summary['total_error']:.2f} ({summary['total_error_pct']:.2f}%) | Avg per-level: {summary['total_error_pct_avg']:.2f}%"
    ws_summary['A7'] = f"Mean - True: {summary['mean_true']:.6f} | Pred: {summary['mean_pred']:.6f}"
    ws_summary['A8'] = f"Variance - True: {summary['var_true']:.6f} | Pred: {summary['var_pred']:.6f}"
    ws_summary['A9'] = f"Variance Δ (avg per-level): {summary['var_diff_pct']:.2f}%"
    if Y_pred_ipf is not None:
        sum_ipf = summarize_levelwise(Y_true, Y_pred_ipf, levels, K, pct_eps=pct_eps)
        ws_summary['A10'] = f"== {split_name} (IPF-corrected) =="
        ws_summary['A11'] = f"sMAPE(all, IPF): {sum_ipf['smape_all']:.2f}%"
        ws_summary['A12'] = f"MAE(all): {sum_ipf['mae_all']:.6f} | RMSE(all): {sum_ipf['rmse_all']:.6f}"
        ws_summary['A13'] = f"MAE(nz): {sum_ipf['mae_nz']:.6f} | RMSE(nz): {sum_ipf['rmse_nz']:.6f} | sMAPE(nz, IPF): {sum_ipf['smape_nz']:.2f}%"
        ws_summary['A14'] = f"Marginal MAE: {sum_ipf['marg_mae']:.6f}"
        ws_summary['A15'] = f"Total Flow - True: {sum_ipf['total_true']:.2f} | IPF: {sum_ipf['total_pred']:.2f} | Error: {sum_ipf['total_error']:.2f} ({sum_ipf['total_error_pct']:.2f}%) | Avg per-level: {sum_ipf['total_error_pct_avg']:.2f}%"
        ws_summary['A16'] = f"Mean - Pred (IPF): {sum_ipf['mean_pred']:.6f}"
        ws_summary['A17'] = f"Variance - Pred (IPF): {sum_ipf['var_pred']:.6f} | Var Δ: {sum_ipf['var_diff_pct']:.2f}% (avg per-level)"
        ws_summary['A18'] = f"Total flow Δ (avg per-level): {sum_ipf['total_error_pct_avg']:.2f}%"
    sorted_levels = sorted(level_data.keys())
    for level in sorted_levels:
        data = level_data[level]
        # Data is already aggregated, so use it directly instead of summing again
        true_total = data['true_matrices'][0] if len(data['true_matrices']) == 1 else np.sum(data['true_matrices'], axis=0)
        pred_total = data['pred_matrices'][0] if len(data['pred_matrices']) == 1 else np.sum(data['pred_matrices'], axis=0)
        pred_ipf_total = data['pred_ipf_matrices'][0] if data['pred_ipf_matrices'] and len(data['pred_ipf_matrices']) == 1 else (np.sum(data['pred_ipf_matrices'], axis=0) if data['pred_ipf_matrices'] else None)
        ws_real = wb.create_sheet(title=f"L{level}_Real")
        ws_real['A1'] = f"Level {level} - Real OD Matrix"
        ws_real['A2'] = f"Total across {len(data['windows'])} windows"
        real_df = pd.DataFrame(true_total, index=zone_labels, columns=zone_labels)
        for r in dataframe_to_rows(real_df, index=True, header=True):
            ws_real.append(r)
        ws_pred = wb.create_sheet(title=f"L{level}_Pred")
        ws_pred['A1'] = f"Level {level} - Predicted OD Matrix"
        ws_pred['A2'] = f"Total across {len(data['windows'])} windows"
        pred_df = pd.DataFrame(pred_total, index=zone_labels, columns=zone_labels)
        for r in dataframe_to_rows(pred_df, index=True, header=True):
            ws_pred.append(r)
        if pred_ipf_total is not None:
            ws_ipf = wb.create_sheet(title=f"L{level}_IPF")
            ws_ipf['A1'] = f"Level {level} - IPF-Corrected Predicted OD Matrix"
            ws_ipf['A2'] = f"Total across {len(data['windows'])} windows"
            ipf_df = pd.DataFrame(pred_ipf_total, index=zone_labels, columns=zone_labels)
            for r in dataframe_to_rows(ipf_df, index=True, header=True):
                ws_ipf.append(r)
        ws_diff = wb.create_sheet(title=f"L{level}_Diff")
        ws_diff['A1'] = f"Level {level} - Difference (Predicted - Real)"
        ws_diff['A2'] = f"Total across {len(data['windows'])} windows"
        diff_matrix = pred_total - true_total
        diff_df = pd.DataFrame(diff_matrix, index=zone_labels, columns=zone_labels)
        for r in dataframe_to_rows(diff_df, index=True, header=True):
            ws_diff.append(r)
        current_row = ws_diff.max_row + 3
        ws_diff[f'A{current_row}'] = "SUMMARY STATISTICS"; current_row += 1
        
        # Calculate comprehensive metrics for this level
        mae_all = np.mean(np.abs(diff_matrix))
        rmse_all = np.sqrt(np.mean(diff_matrix**2))
        mae_nz = np.mean(np.abs(diff_matrix[true_total > 0])) if np.any(true_total > 0) else 0
        rmse_nz = np.sqrt(np.mean(diff_matrix[true_total > 0]**2)) if np.any(true_total > 0) else 0
        
        # Calculate sMAPE (symmetric Mean Absolute Percentage Error)
        smape_all = smape_np(true_total.reshape(-1), pred_total.reshape(-1), eps=pct_eps)
        smape_nz = smape_np(true_total[true_total > 0].reshape(-1), pred_total[true_total > 0].reshape(-1), eps=pct_eps) if np.any(true_total > 0) else np.nan
        
        # Calculate MAPE
        mape_all = np.mean(np.abs(diff_matrix[true_total > 1e-9]) / np.abs(true_total[true_total > 1e-9])) * 100 if np.any(true_total > 1e-9) else 0
        
        # Calculate marginal errors
        true_dep = true_total.sum(axis=1)
        true_arr = true_total.sum(axis=0)
        pred_dep = pred_total.sum(axis=1)
        pred_arr = pred_total.sum(axis=0)
        marg_mae = (mae(true_dep, pred_dep) + mae(true_arr, pred_arr)) / 2.0
        
        # Calculate variance metrics
        var_true = float(true_total.reshape(-1).var())
        var_pred = float(pred_total.reshape(-1).var())
        var_diff_pct = 0.0 if var_true < 1e-12 else abs(var_pred - var_true) / var_true * 100.0
        
        # Calculate mean metrics
        mean_true = float(true_total.reshape(-1).mean())
        mean_pred = float(pred_total.reshape(-1).mean())
        
        # Total flow metrics
        total_true = np.sum(true_total)
        total_pred = np.sum(pred_total)
        total_error = abs(total_true - total_pred)
        total_error_pct = (total_error / max(total_true, 1e-9)) * 100
        
        # Write metrics matching the global summary format
        ws_diff[f'A{current_row}'] = f"sMAPE(all): {smape_all:.2f}% (PRIMARY)"; current_row += 1
        ws_diff[f'A{current_row}'] = f"MAE(all): {mae_all:.6f} | RMSE(all): {rmse_all:.6f}"; current_row += 1
        ws_diff[f'A{current_row}'] = f"MAE(nz) : {mae_nz:.6f} | RMSE(nz) : {rmse_nz:.6f} | sMAPE(nz): {smape_nz:.2f}%"; current_row += 1
        ws_diff[f'A{current_row}'] = f"Marginal MAE (rows/cols avg): {marg_mae:.6f}"; current_row += 1
        ws_diff[f'A{current_row}'] = f"Total Traffic - True: {total_true:.2f} | Pred: {total_pred:.2f} | Error: {total_error:.2f} ({total_error_pct:.2f}%)"; current_row += 1
        ws_diff[f'A{current_row}'] = f"Mean - True: {mean_true:.6f} | Pred: {mean_pred:.6f}"; current_row += 1
        ws_diff[f'A{current_row}'] = f"Variance - True: {var_true:.6f} | Pred: {var_pred:.6f}"; current_row += 1
        ws_diff[f'A{current_row}'] = f"Variance Δ (avg per-level): {var_diff_pct:.2f}%"
        
        # Add IPF difference sheet if IPF is active
        if pred_ipf_total is not None:
            ws_diff_ipf = wb.create_sheet(title=f"L{level}_Diff_IPF")
            ws_diff_ipf['A1'] = f"Level {level} - Difference (IPF-Corrected Predicted - Real)"
            ws_diff_ipf['A2'] = f"Total across {len(data['windows'])} windows"
            diff_ipf_matrix = pred_ipf_total - true_total
            diff_ipf_df = pd.DataFrame(diff_ipf_matrix, index=zone_labels, columns=zone_labels)
            for r in dataframe_to_rows(diff_ipf_df, index=True, header=True):
                ws_diff_ipf.append(r)
            
            # Calculate IPF metrics for this level
            current_row_ipf = ws_diff_ipf.max_row + 3
            ws_diff_ipf[f'A{current_row_ipf}'] = "SUMMARY STATISTICS (IPF-CORRECTED)"; current_row_ipf += 1
            
            mae_all_ipf = np.mean(np.abs(diff_ipf_matrix))
            rmse_all_ipf = np.sqrt(np.mean(diff_ipf_matrix**2))
            mae_nz_ipf = np.mean(np.abs(diff_ipf_matrix[true_total > 0])) if np.any(true_total > 0) else 0
            rmse_nz_ipf = np.sqrt(np.mean(diff_ipf_matrix[true_total > 0]**2)) if np.any(true_total > 0) else 0
            
            # Calculate sMAPE for IPF
            smape_all_ipf = smape_np(true_total.reshape(-1), pred_ipf_total.reshape(-1), eps=pct_eps)
            smape_nz_ipf = smape_np(true_total[true_total > 0].reshape(-1), pred_ipf_total[true_total > 0].reshape(-1), eps=pct_eps) if np.any(true_total > 0) else np.nan
            
            # Calculate marginal errors for IPF
            pred_ipf_dep = pred_ipf_total.sum(axis=1)
            pred_ipf_arr = pred_ipf_total.sum(axis=0)
            marg_mae_ipf = (mae(true_dep, pred_ipf_dep) + mae(true_arr, pred_ipf_arr)) / 2.0
            
            # Calculate variance metrics for IPF
            var_ipf = float(pred_ipf_total.reshape(-1).var())
            var_diff_pct_ipf = 0.0 if var_true < 1e-12 else abs(var_ipf - var_true) / var_true * 100.0
            
            # Calculate mean metrics for IPF
            mean_ipf = float(pred_ipf_total.reshape(-1).mean())
            
            # Total flow metrics for IPF
            total_ipf = np.sum(pred_ipf_total)
            total_error_ipf = abs(total_true - total_ipf)
            total_error_pct_ipf = (total_error_ipf / max(total_true, 1e-9)) * 100
            
            # Write IPF metrics matching the global summary format
            ws_diff_ipf[f'A{current_row_ipf}'] = f"sMAPE(all, IPF): {smape_all_ipf:.2f}% (PRIMARY)"; current_row_ipf += 1
            ws_diff_ipf[f'A{current_row_ipf}'] = f"MAE(all): {mae_all_ipf:.6f} | RMSE(all): {rmse_all_ipf:.6f}"; current_row_ipf += 1
            ws_diff_ipf[f'A{current_row_ipf}'] = f"MAE(nz) : {mae_nz_ipf:.6f} | RMSE(nz) : {rmse_nz_ipf:.6f} | sMAPE(nz, IPF): {smape_nz_ipf:.2f}%"; current_row_ipf += 1
            ws_diff_ipf[f'A{current_row_ipf}'] = f"Marginal MAE: {marg_mae_ipf:.6f}"; current_row_ipf += 1
            ws_diff_ipf[f'A{current_row_ipf}'] = f"Total Flow - True: {total_true:.2f} | IPF: {total_ipf:.2f} | Error: {total_error_ipf:.2f} ({total_error_pct_ipf:.2f}%)"; current_row_ipf += 1
            ws_diff_ipf[f'A{current_row_ipf}'] = f"Mean - Pred (IPF): {mean_ipf:.6f}"; current_row_ipf += 1
            ws_diff_ipf[f'A{current_row_ipf}'] = f"Variance - Pred (IPF): {var_ipf:.6f} | Var Δ: {var_diff_pct_ipf:.2f}% (avg per-level)"
    wb.save(excel_filename)
    print(f"Excel file saved: {excel_filename}")
    create_summary_excel(Y_true, Y_pred, Y_pred_ipf, levels, windows, K, save_dir, split_name)

def create_summary_excel(Y_true, Y_pred, Y_pred_ipf, levels, windows, K, save_dir, split_name):
    if Y_true.ndim == 2:
        Y_true = Y_true.reshape(-1, K, K)
    if Y_pred.ndim == 2:
        Y_pred = Y_pred.reshape(-1, K, K)
    summary_filename = os.path.join(save_dir, f"OD_summary_{split_name.lower()}_tunned.xlsx")
    wb = Workbook(); ws = wb.active; ws.title = f"Summary_{split_name}"
    zone_labels = [f"Zone_{i}" for i in range(K)]
    current_row = 1
    def excel_col_name(n: int) -> str:
        n = n + 1
        name = ""
        while n > 0:
            n, rem = divmod(n - 1, 26)
            name = chr(ord('A') + rem) + name
        return name
    for i in range(len(Y_true)):
        level = levels[i] if i < len(levels) else "Unknown"
        start, end = windows[i] if i < len(windows) else (0, 0)
        ws[f'A{current_row}'] = f"Sample {i+1}: Level {level}, Window {start}-{end}"; current_row += 1
        ws[f'A{current_row}'] = "TRUE OD Matrix"; current_row += 1
        ws[f'A{current_row}'] = "Origin\\Dest"
        for j, zone in enumerate(zone_labels):
            col_letter = excel_col_name(j + 1)
            ws[f'{col_letter}{current_row}'] = zone
        current_row += 1
        for row_idx, zone in enumerate(zone_labels):
            ws[f'A{current_row}'] = zone
            for col_idx in range(K):
                col_letter = excel_col_name(col_idx + 1)
                ws[f'{col_letter}{current_row}'] = Y_true[i, row_idx, col_idx]
            current_row += 1
        current_row += 1
        ws[f'A{current_row}'] = "PREDICTED OD Matrix"; current_row += 1
        ws[f'A{current_row}'] = "Origin\\Dest"
        for j, zone in enumerate(zone_labels):
            col_letter = excel_col_name(j + 1)
            ws[f'{col_letter}{current_row}'] = zone
        current_row += 1
        for row_idx, zone in enumerate(zone_labels):
            ws[f'A{current_row}'] = zone
            for col_idx in range(K):
                col_letter = excel_col_name(col_idx + 1)
                ws[f'{col_letter}{current_row}'] = Y_pred[i, row_idx, col_idx]
            current_row += 1
        current_row += 1
        diff_matrix = Y_pred[i] - Y_true[i]
        ws[f'A{current_row}'] = "DIFFERENCE (Predicted - True)"; current_row += 1
        ws[f'A{current_row}'] = "Origin\\Dest"
        for j, zone in enumerate(zone_labels):
            col_letter = excel_col_name(j + 1)
            ws[f'{col_letter}{current_row}'] = zone
        current_row += 1
        for row_idx, zone in enumerate(zone_labels):
            ws[f'A{current_row}'] = zone
            for col_idx in range(K):
                col_letter = excel_col_name(col_idx + 1)
                ws[f'{col_letter}{current_row}'] = diff_matrix[row_idx, col_idx]
            current_row += 1
        if Y_pred_ipf is not None:
            current_row += 1
            ws[f'A{current_row}'] = "IPF-CORRECTED PREDICTED OD Matrix"; current_row += 1
            ws[f'A{current_row}'] = "Origin\\Dest"
            for j, zone in enumerate(zone_labels):
                col_letter = excel_col_name(j + 1)
                ws[f'{col_letter}{current_row}'] = zone
            current_row += 1
            for row_idx, zone in enumerate(zone_labels):
                ws[f'A{current_row}'] = zone
                for col_idx in range(K):
                    col_letter = excel_col_name(col_idx + 1)
                    ws[f'{col_letter}{current_row}'] = Y_pred_ipf[i, row_idx, col_idx]
                current_row += 1
            current_row += 1
            ws[f'A{current_row}'] = "DIFFERENCE IPF (IPF-Predicted - True)"; current_row += 1
            ws[f'A{current_row}'] = "Origin\\Dest"
            for j, zone in enumerate(zone_labels):
                col_letter = excel_col_name(j + 1)
                ws[f'{col_letter}{current_row}'] = zone
            current_row += 1
            diff_ipf_matrix = Y_pred_ipf[i] - Y_true[i]
            for row_idx, zone in enumerate(zone_labels):
                ws[f'A{current_row}'] = zone
                for col_idx in range(K):
                    col_letter = excel_col_name(col_idx + 1)
                    ws[f'{col_letter}{current_row}'] = diff_ipf_matrix[row_idx, col_idx]
                current_row += 1
        current_row += 3
    wb.save(summary_filename)
    print(f"Summary Excel file saved: {summary_filename}")

# ---------------------------
# Level-wise summary helper
# ---------------------------
def summarize_levelwise(Y_true, Y_pred, levels, K, pct_eps=10.0):
    if Y_true.ndim == 2: Y_true = Y_true.reshape(-1, K, K)
    if Y_pred.ndim == 2: Y_pred = Y_pred.reshape(-1, K, K)
    level_data = {}
    for i, lv in enumerate(levels):
        d = level_data.setdefault(lv, {"true": [], "pred": []})
        d["true"].append(Y_true[i])
        d["pred"].append(Y_pred[i])
    per_level = []
    for lv in sorted(level_data.keys()):
        T = np.sum(level_data[lv]["true"], axis=0)
        P = np.sum(level_data[lv]["pred"], axis=0)
        t = T.reshape(-1); p = P.reshape(-1)
        nz = t > 0
        lv_mae   = float(np.mean(np.abs(p - t)))
        lv_rmse  = float(np.sqrt(np.mean((p - t) ** 2)))
        lv_smape = smape_np(t, p, eps=pct_eps)
        lv_smape_stab = smape_np(t, p, eps=5.0)  # Stabilized sMAPE for debugging
        lv_mae_nz   = float(np.mean(np.abs(p[nz] - t[nz]))) if nz.any() else np.nan
        lv_rmse_nz  = float(np.sqrt(np.mean((p[nz] - t[nz]) ** 2))) if nz.any() else np.nan
        lv_smape_nz = smape_np(t[nz], p[nz], eps=pct_eps) if nz.any() else np.nan
        lv_smape_nz_stab = smape_np(t[nz], p[nz], eps=5.0) if nz.any() else np.nan
        r_t, c_t = marginals(T.reshape(1, -1))
        r_p, c_p = marginals(P.reshape(1, -1))
        lv_marg_mae = (mae(r_t, r_p) + mae(c_t, c_p)) / 2.0
        v_true = float(t.var()); v_pred = float(p.var())
        lv_var_diff_pct = 0.0 if v_true < 1e-12 else abs(v_pred - v_true) / v_true * 100.0
        tt = float(t.sum()); tp = float(p.sum())
        lv_tot_err = abs(tt - tp)
        lv_tot_err_pct = 0.0 if tt < 1e-12 else lv_tot_err / tt * 100.0
        per_level.append(dict(
            mae_all=lv_mae, rmse_all=lv_rmse, smape_all=lv_smape, smape_all_stab=lv_smape_stab,
            mae_nz=lv_mae_nz, rmse_nz=lv_rmse_nz, smape_nz=lv_smape_nz, smape_nz_stab=lv_smape_nz_stab,
            marg_mae=lv_marg_mae,
            mean_true=float(t.mean()), mean_pred=float(p.mean()),
            var_true=v_true, var_pred=v_pred, var_diff_pct=lv_var_diff_pct,
            total_true=tt, total_pred=tp, total_err=lv_tot_err, total_err_pct=lv_tot_err_pct
        ))
    def avg(key):
        vals = [m[key] for m in per_level if not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else np.nan
    total_true = float(np.sum([m["total_true"] for m in per_level]))
    total_pred = float(np.sum([m["total_pred"] for m in per_level]))
    total_err  = abs(total_true - total_pred)
    total_err_pct = (total_err / max(total_true, 1e-9)) * 100.0
    return dict(
        smape_all = avg("smape_all"),
        smape_all_stab = avg("smape_all_stab"),
        mae_all   = avg("mae_all"),
        rmse_all  = avg("rmse_all"),
        mae_nz    = avg("mae_nz"),
        rmse_nz   = avg("rmse_nz"),
        smape_nz  = avg("smape_nz"),
        smape_nz_stab = avg("smape_nz_stab"),
        marg_mae  = avg("marg_mae"),
        mean_true = avg("mean_true"),
        mean_pred = avg("mean_pred"),
        var_true  = avg("var_true"),
        var_pred  = avg("var_pred"),
        var_diff_pct = avg("var_diff_pct"),
        total_true = total_true,
        total_pred = total_pred,
        total_error = total_err,
        total_error_pct = total_err_pct,
        total_error_pct_avg = avg("total_err_pct")
    )

# ---------------------------
# Training script
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts_glob", default="Refined Data/Traffic counts/manhattan_*_traffic-counts-*.csv", help="Edge-based counts CSVs in 'Traffic counts' (zone_o, zone_d, traffic_count)")
    ap.add_argument("--od_glob", default="Refined Data/OD matrice/manhattan_*_zone-od-*_K20.csv", help="e.g. '../Refined Data/OD matrice/manhattan_*_zone-od-*_K20.csv'")
    ap.add_argument("--K", type=int, help="Number of zones (auto-detected if not provided)")
    ap.add_argument("--use_svd", action=argparse.BooleanOptionalAction, default=False, help="Compress Y with SVD")
    ap.add_argument("--ry", type=int, default=32, help="Latent dim for Y if --use_svd")
    ap.add_argument("--hidden", type=str, default="512,256,128", help="Hidden sizes, e.g. '256,128,64' (use ' ' for linear head)")
    ap.add_argument("--log_x", action=argparse.BooleanOptionalAction, default=True, help="Apply log1p to X before scaling (default: on)")
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight_decay", type=float, default=3e-3)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=1400)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--min_epochs", type=int, default=0,
                    help="Do not decrement early-stop patience before this epoch (default: 0).")
    ap.add_argument("--min_delta", type=float, default=0.0,
                    help="Minimum required improvement in val_sMAPE to reset patience (default: 0.0).")
    ap.add_argument("--lambda_marg", type=float, default=1.0, help="weight for marginal consistency loss during training")
    ap.add_argument("--x_only_summaries", action=argparse.BooleanOptionalAction, default=False, help="Use only departures/arrivals/diagonal as X (no edges)")
    ap.add_argument("--loss_type", type=str, default="hybrid", choices=["mse", "l1", "hybrid", "poisson"], help="Loss function type")
    ap.add_argument("--lambda_diag", type=float, default=0.3, help="weight for internal (diagonal) consistency loss")
    ap.add_argument("--lambda_pct", type=float, default=3.0, help="weight for percentage loss (SMAPE-like) on original scale")
    ap.add_argument("--lambda_log", type=float, default=0.4, help="weight for log-space loss to preserve dynamic range")
    ap.add_argument("--pct_eps", type=float, default=25.0, help="epsilon floor in denominator to avoid tiny-target blowups (0.0 for standard sMAPE)")
    ap.add_argument("--lambda_level_smape", type=float, default=0.0,
                    help="Weight for per-level aggregated sMAPE loss in training (default: 0.0).")
    ap.add_argument("--use_ipf", action=argparse.BooleanOptionalAction, default=False, help="Apply IPF post-correction for perfect marginals (default: off)")
    ap.add_argument("--use_magnitude_weighting", action=argparse.BooleanOptionalAction, default=True, help="Use magnitude weighting to prevent big cells from being washed out")
    ap.add_argument("--global_rescale", action=argparse.BooleanOptionalAction, default=True, help="Apply post-hoc global mass rescale at inference")
    ap.add_argument("--ipf_iter", type=int, default=20, help="Number of IPF iterations")
    ap.add_argument("--ipf_blend_alpha", type=float, default=0.0, help="Blend factor for model marginals in IPF (0=use only input, 1=use only model)")
    ap.add_argument("--ipf_clip_min", type=float, default=0.05, help="Lower clip for IPF row/col multipliers")
    ap.add_argument("--ipf_clip_max", type=float, default=8.0, help="Upper clip for IPF row/col multipliers")
    ap.add_argument("--ipf_rho", type=float, default=0.97, help="Target total scaling factor for IPF (default: 0.97)")
    ap.add_argument("--ipf_beta", type=float, default=0.15, help="Baseline blending factor for IPF seed (default: 0.15)")
    ap.add_argument("--save_dir", default="Results")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits/training (default: 42)")
    ap.add_argument("--skip_save", action=argparse.BooleanOptionalAction, default=False,
                    help="Skip evaluation + saving artifacts (useful for hyperparameter tuning)")
    ap.add_argument("--print_best_val", action=argparse.BooleanOptionalAction, default=False,
                    help="Print a machine-readable BEST_VAL_SMAPE line and exit early if --skip_save is set")
    ap.add_argument("--tune_report_every", type=int, default=0,
                    help="If >0, print TUNE_REPORT <epoch> <val_smape> every N epochs (for pruners).")
    ap.add_argument("--hparams_json", type=str, default=None,
                    help="Path to a JSON file containing hyperparameters. If it contains 'best_params', they will be applied automatically.")
    ap.add_argument("--auto_hparams", action=argparse.BooleanOptionalAction, default=True,
                    help="If enabled and --hparams_json is not provided, auto-load Results/best_hparams_optuna.json when it exists.")

    # ---- OPTION A parameters ----
    ap.add_argument("--optionA", action=argparse.BooleanOptionalAction, default=True,
                    help="Use fixed-slot split per level (keep_first_slots, split 8/2/2, optional chunking).")
    ap.add_argument("--keep_first_slots", type=int, default=12,
                    help="How many 5-min slots to keep from the start of each level (drop the rest).")
    ap.add_argument("--split_slots", type=str, default="8,2,2",
                    help="How many 5-min slots to allocate to train,val,test per level (e.g., '8,2,2').")
    ap.add_argument("--chunk_size", type=int, default=1,
                    help="Aggregate consecutive slots into chunks of this size (2 => 10-minute samples).")

    # ---- legacy toggles kept for completeness ----
    ap.add_argument("--aggregate_per_level", action="store_true",
                    help="If set, build one sample per level by summing all windows within that level")
    ap.add_argument("--excel_output", action=argparse.BooleanOptionalAction, default=True,
                    help="Generate Excel files with OD matrices and differences (default: on)")
    ap.add_argument("--predict_fractions", action=argparse.BooleanOptionalAction, default=False,
                    help="Predict fractions (shape) and scale by observed totals")
    ap.add_argument("--frac_scale", type=float, default=1e5,
                    help="Scaling factor for fraction prediction (default: 1e5)")
    ap.add_argument("--use_total_head", action=argparse.BooleanOptionalAction, default=True,
                    help="Use separate total-flow head for better total prediction")
    ap.add_argument("--lambda_total", type=float, default=12.0,
                    help="Weight for total flow loss (default: 1.0)")
    ap.add_argument("--use_huber", action=argparse.BooleanOptionalAction, default=True,
                    help="Use Huber loss in hybrid loss instead of MSE")
    ap.add_argument("--huber_delta", type=float, default=0.8,
                    help="Delta parameter for Huber loss (default: 1.0)")
    ap.add_argument("--lambda_entropy", type=float, default=0.0,
                    help="Weight for negative entropy penalty to encourage contrast (default: 0.001)")
    ap.add_argument("--lambda_variance", type=float, default=0.0,
                    help="Weight for variance-preserving loss (default: 0.2)")
    ap.add_argument("--lambda_level_var", type=float, default=0.0,
                    help="Weight for per-level aggregated variance loss (default: 1.0)")
    ap.add_argument("--gamma_sharpen", type=float, default=1.2,
                    help="Gamma sharpening factor for post-hoc variance increase (default: 1.0 for debugging)")
    
    args = ap.parse_args()

    # Optional: auto-load best hyperparameters if present
    if args.auto_hparams and not args.hparams_json:
        default_hp = os.path.join("Results", "best_hparams_optuna.json")
        if os.path.exists(default_hp):
            args.hparams_json = default_hp

    # Optional: load hyperparameters from a JSON file (e.g., Results/best_hparams_optuna.json)
    if args.hparams_json:
        try:
            with open(args.hparams_json, "r", encoding="utf-8") as f:
                hp = json.load(f)
            # Accept either {"best_params": {...}} or a flat dict of params
            if isinstance(hp, dict) and "best_params" in hp and isinstance(hp["best_params"], dict):
                hp = hp["best_params"]
            if not isinstance(hp, dict):
                raise ValueError("Hyperparameter JSON must be a dict or contain a 'best_params' dict.")

            unknown = []
            for k, v in hp.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                else:
                    unknown.append(k)
            if unknown:
                print(f"[hparams_json] Ignored unknown keys: {unknown}")
            print(f"[hparams_json] Loaded hyperparameters from: {args.hparams_json}")
        except Exception as e:
            raise RuntimeError(f"Failed to load --hparams_json '{args.hparams_json}': {e}") from e

    # Reproducibility (best-effort; CUDA nondeterminism may still apply if you later add GPU usage)
    try:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    os.makedirs(args.save_dir, exist_ok=True)
    hidden = tuple(int(x) for x in args.hidden.split(",")) if args.hidden and args.hidden.strip() else ()

    # Discover files & build matrices
    pairs = discover_pairs(args.counts_glob, args.od_glob)
    # Auto-detect K if not provided
    if args.K is None:
        K = 0
        # detect from counts
        max_zone = 0
        for (lvl, p_counts, p_od, st, en) in pairs:
            df = pd.read_csv(p_counts)
            df.columns = [c.strip().lower() for c in df.columns]
            if {"zone_o", "zone_d"} <= set(df.columns) and len(df) > 0:
                max_zone = max(max_zone, int(df["zone_o"].max()), int(df["zone_d"].max()))
        K = max_zone + 1
        print(f"Detected K = {K} zones from traffic counts data")
    else:
        K = args.K
        print(f"Using provided K = {K} zones")

    # Always build per-window matrices first
    X, Y_raw, levels, windows, marginals_from_input = build_matrices(pairs, K=K, summaries_only=args.x_only_summaries)

    # OPTION A path: fixed slot split + chunking
    if args.optionA and not args.aggregate_per_level:
        # parse split slots
        try:
            tr_s, va_s, te_s = [int(x) for x in args.split_slots.split(",")]
        except Exception as e:
            raise ValueError("--split_slots must be like '8,2,2'") from e
        (Xtr, Ytr_raw, tr_levels, tr_windows, tr_marginals), \
        (Xva, Yva_raw, va_levels, va_windows, va_marginals), \
        (Xte, Yte_raw, te_levels, te_windows, te_marginals), \
        (groups_tr, groups_va, groups_te), \
        (mapping_tr, mapping_va, mapping_te) = fixedslot_split_and_chunk(
            X, Y_raw, levels, windows, marginals_from_input,
            keep_first_slots=args.keep_first_slots,
            split_slots=(tr_s, va_s, te_s),
            chunk_size=args.chunk_size
        )
        if len(Xtr)==0 or len(Xva)==0 or len(Xte)==0:
            raise RuntimeError("After fixed-slot split, one of the splits is empty. "
                               "Check keep_first_slots / split_slots / chunk_size.")
    elif args.aggregate_per_level:
        # Legacy per-level aggregation path
        def aggregate_by_level(pairs, K: int, summaries_only: bool = False):
            level_to_sum_x: Dict[str, np.ndarray] = {}
            level_to_sum_y: Dict[str, np.ndarray] = {}
            level_to_sum_m: Dict[str, Tuple[np.ndarray,np.ndarray]] = {}
            for (lvl, p_counts, p_od, st, en) in pairs:
                Mc = load_edge_counts_csv(p_counts, K=K)
                M = load_zone_od(p_od, K=K)
                if lvl not in level_to_sum_x:
                    level_to_sum_x[lvl] = Mc.astype(np.float32)
                    level_to_sum_y[lvl] = M.astype(np.float32)
                    level_to_sum_m[lvl] = (Mc.sum(axis=1), Mc.sum(axis=0))
                else:
                    level_to_sum_x[lvl] = level_to_sum_x[lvl] + Mc.astype(np.float32)
                    level_to_sum_y[lvl] = level_to_sum_y[lvl] + M.astype(np.float32)
                    d,a = level_to_sum_m[lvl]
                    level_to_sum_m[lvl] = (d + Mc.sum(axis=1), a + Mc.sum(axis=0))
            levels_unique = sorted(level_to_sum_x.keys())
            X = np.vstack([edge_feats(level_to_sum_x[lv].reshape(-1), K, summaries_only) for lv in levels_unique]).astype(np.float32)
            Y = np.vstack([level_to_sum_y[lv].reshape(-1) for lv in levels_unique]).astype(np.float32)
            marginals_from_input = [level_to_sum_m[lv] for lv in levels_unique]
            return X, Y, levels_unique, marginals_from_input
        Xagg, Yagg, levels_unique, marginals_agg = aggregate_by_level(pairs, K=K, summaries_only=args.x_only_summaries)
        idx_tr_lv, idx_va_lv, idx_te_lv = split_across_levels(levels_unique)
        Xtr, Xva, Xte = Xagg[idx_tr_lv], Xagg[idx_va_lv], Xagg[idx_te_lv]
        Ytr_raw, Yva_raw, Yte_raw = Yagg[idx_tr_lv], Yagg[idx_va_lv], Yagg[idx_te_lv]
        tr_marginals = [marginals_agg[i] for i in idx_tr_lv]
        va_marginals = [marginals_agg[i] for i in idx_va_lv]
        te_marginals = [marginals_agg[i] for i in idx_te_lv]
        tr_windows = [(0,0) for _ in tr_marginals]; va_windows = [(0,0) for _ in va_marginals]; te_windows = [(0,0) for _ in te_marginals]
        tr_levels = [levels_unique[i] for i in idx_tr_lv]
        va_levels = [levels_unique[i] for i in idx_va_lv]
        te_levels = [levels_unique[i] for i in idx_te_lv]
    else:
        # Original per-window split (not optionA)
        idx_tr, idx_va, idx_te, levels_unique = split_per_level(levels, windows)
        Xtr, Xva, Xte = X[idx_tr], X[idx_va], X[idx_te]
        Ytr_raw, Yva_raw, Yte_raw = Y_raw[idx_tr], Y_raw[idx_va], Y_raw[idx_te]
        tr_marginals = [marginals_from_input[i] for i in idx_tr]
        va_marginals = [marginals_from_input[i] for i in idx_va]
        te_marginals = [marginals_from_input[i] for i in idx_te]
        tr_windows = [windows[i] for i in idx_tr]
        va_windows = [windows[i] for i in idx_va]
        te_windows = [windows[i] for i in idx_te]
        tr_levels = [levels[i] for i in idx_tr]
        va_levels = [levels[i] for i in idx_va]
        te_levels = [levels[i] for i in idx_te]

    # Optional log-scale X before scaling
    if args.log_x:
        Xtr = np.log1p(Xtr); Xva = np.log1p(Xva); Xte = np.log1p(Xte)

    # Transforms: X scaler
    x_scaler = StandardScaler()
    Xtr_s = x_scaler.fit_transform(Xtr)
    Xva_s = x_scaler.transform(Xva)
    Xte_s = x_scaler.transform(Xte)

    # Transform Y: log1p always; then optional SVD on TRAIN ONLY
    if args.predict_fractions:
        totals_tr = Ytr_raw.sum(axis=1, keepdims=True) + 1e-9
        totals_va = Yva_raw.sum(axis=1, keepdims=True) + 1e-9
        totals_te = Yte_raw.sum(axis=1, keepdims=True) + 1e-9
        Ytr_frac = Ytr_raw / totals_tr; Yva_frac = Yva_raw / totals_va; Yte_frac = Yte_raw / totals_te
        Ytr_log = log1p_(args.frac_scale * Ytr_frac)
        Yva_log = log1p_(args.frac_scale * Yva_frac)
        Yte_log = log1p_(args.frac_scale * Yte_frac)
        tr_totals = totals_tr.flatten(); va_totals = totals_va.flatten(); te_totals = totals_te.flatten()
    else:
        Ytr_log = log1p_(Ytr_raw); Yva_log = log1p_(Yva_raw); Yte_log = log1p_(Yte_raw)
        tr_totals = va_totals = te_totals = None

    if args.use_svd:
        n_comp = min(args.ry, Ytr_log.shape[1]-1, Xtr_s.shape[0]-1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        Ztr = svd.fit_transform(Ytr_log)
        Zva = svd.transform(Yva_log)
        Zte = svd.transform(Yte_log)
        Dout = Ztr.shape[1]
        print(f"SVD: requested {args.ry} components, using {n_comp} (capped by n_train={Xtr_s.shape[0]}, n_features={Ytr_log.shape[1]})")
    else:
        svd = None
        Ztr, Zva, Zte = Ytr_log, Yva_log, Yte_log
        Dout = Ztr.shape[1]
    Din = Xtr_s.shape[1]

    # Use lambda_marg as specified - IPF should be a light post-process, not a replacement for marginal training
    effective_lambda_marg = args.lambda_marg
    # Fit calibration for counts-to-OD total mapping if using IPF or global_rescale
    if args.use_ipf or args.global_rescale:
        estimate_total_from_counts = fit_counts_to_od_calibration(tr_marginals, Ytr_raw)
    else:
        estimate_total_from_counts = None

    # Calculate proper initialization for total head bias
    if args.use_total_head:
        init_total = float(Ytr_raw.sum(axis=1).mean())
        init_bias = float(np.log(max(init_total, 1e-6)))
    else:
        init_bias = None

    print(f"""
=== Training Configuration ===
Input dim: {Din}, Output dim: {Dout}
Hidden layers: {hidden}
Loss type: {args.loss_type}
Lambda marginal: {effective_lambda_marg}
Learning rate: {args.lr}
Use SVD: {args.use_svd} (ry={args.ry}) if args.use_svd else False
Use IPF: {args.use_ipf}
""".rstrip())
    if args.use_ipf:
        print(f"IPF parameters: rho={args.ipf_rho}, beta={args.ipf_beta}, iterations={args.ipf_iter}")
        print(f"IPF clips: [{args.ipf_clip_min}, {args.ipf_clip_max}], blend_alpha={args.ipf_blend_alpha}")
    if args.optionA and not args.aggregate_per_level:
        print(f"OPTION A: keep_first_slots={args.keep_first_slots}, split_slots={args.split_slots}, chunk_size={args.chunk_size}")
    print("===============================")

    # Create level ID mappings for level-aggregated variance loss
    all_levels = sorted(set(tr_levels + va_levels + te_levels))
    level_to_idx = {lv: i for i, lv in enumerate(all_levels)}
    tr_level_ids = np.array([level_to_idx[lv] for lv in tr_levels], dtype=np.int64)
    va_level_ids = np.array([level_to_idx[lv] for lv in va_levels], dtype=np.int64)
    te_level_ids = np.array([level_to_idx[lv] for lv in te_levels], dtype=np.int64)

    # PyTorch datasets
    class XY(Dataset):
        def __init__(self, X, Z, totals=None, level_ids=None):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.Z = torch.from_numpy(Z.astype(np.float32))
            self.totals = None if totals is None else torch.from_numpy(np.asarray(totals, dtype=np.float32))
            self.level_ids = None if level_ids is None else torch.from_numpy(np.asarray(level_ids, dtype=np.int64))
        def __len__(self): return self.X.shape[0]
        def __getitem__(self, i):
            if self.totals is None and self.level_ids is None:
                return self.X[i], self.Z[i]
            if self.totals is None:
                return self.X[i], self.Z[i], self.level_ids[i]
            if self.level_ids is None:
                return self.X[i], self.Z[i], self.totals[i]
            # both present
            return self.X[i], self.Z[i], self.totals[i], self.level_ids[i]

    if args.predict_fractions:
        dl_tr = DataLoader(XY(Xtr_s, Ztr, tr_totals, tr_level_ids), batch_size=args.batch_size, shuffle=True)
        dl_va = DataLoader(XY(Xva_s, Zva, va_totals, va_level_ids), batch_size=args.batch_size, shuffle=False)
        dl_te = DataLoader(XY(Xte_s, Zte, te_totals, te_level_ids), batch_size=args.batch_size, shuffle=False)
    else:
        dl_tr = DataLoader(XY(Xtr_s, Ztr, None, tr_level_ids), batch_size=args.batch_size, shuffle=True)
        dl_va = DataLoader(XY(Xva_s, Zva, None, va_level_ids), batch_size=args.batch_size, shuffle=False)
        dl_te = DataLoader(XY(Xte_s, Zte, None, te_level_ids), batch_size=args.batch_size, shuffle=False)

    if hidden:
        model = MLP(Din, Dout, hidden=hidden, dropout=args.dropout, use_total_head=args.use_total_head)
    else:
        if args.use_total_head:
            class LinearWithTotalHead(nn.Module):
                def __init__(self, in_dim, out_dim):
                    super().__init__()
                    self.pattern_head = nn.Linear(in_dim, out_dim)
                    self.total_head = nn.Sequential(
                        nn.Linear(in_dim, max(16, in_dim // 4)),
                        nn.ReLU(),
                        nn.Linear(max(16, in_dim // 4), 1)
                    )
                def forward(self, x):
                    pattern = self.pattern_head(x)
                    log_total = self.total_head(x)
                    return pattern, log_total
            model = LinearWithTotalHead(Din, Dout)
        else:
            model = nn.Linear(Din, Dout)

    if args.use_total_head and hasattr(model, 'total_head'):
        with torch.no_grad():
            model.total_head[-1].bias.fill_(init_bias)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.7, patience=10)
    
    if args.loss_type == "mse":
        primary_loss = nn.MSELoss()
    elif args.loss_type == "l1":
        primary_loss = nn.L1Loss()
    elif args.loss_type == "hybrid":
        primary_loss = None
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    if args.use_huber:
        huber_loss = nn.SmoothL1Loss(beta=args.huber_delta)

    best_val = 1e9
    best_state = None
    patience = args.patience

    def decode_to_OD(batch_Z: np.ndarray, totals: np.ndarray = None) -> np.ndarray:
        if svd is None:
            Y_log = batch_Z
        else:
            Y_log = svd.inverse_transform(batch_Z)
        if args.predict_fractions:
            Y_frac = expm1_(Y_log) / args.frac_scale
            Y_frac = np.maximum(Y_frac, 0.0)
            if totals is not None:
                Y = Y_frac * totals.reshape(-1, 1)
            else:
                Y = Y_frac
        else:
            Y = expm1_(Y_log)
            Y = np.maximum(Y, 0.0)
        return Y

    # Training
    for epoch in range(args.epochs):
        model.train()
        tr_losses = []
        for batch in dl_tr:
            opt.zero_grad(set_to_none=True)
            # unpack batch based on predict_fractions & presence of level_ids
            if args.predict_fractions:
                if len(batch) == 4:
                    xb, zb, batch_totals, batch_levels = batch
                else:
                    xb, zb, batch_totals = batch
                    batch_levels = None
            else:
                if len(batch) == 3:
                    xb, zb, batch_levels = batch
                    batch_totals = None
                else:
                    xb, zb = batch
                    batch_totals = None
                    batch_levels = None
            if args.use_total_head:
                pred_lat, log_total = model(xb)
                pred_scale = torch.exp(log_total).squeeze(-1)  # (B,)
            else:
                pred_lat = model(xb); pred_scale = None
            if svd is None:
                Yb_true_t = torch.expm1(zb).view(-1, K, K)
                pred_lat_clamped = torch.clamp(pred_lat, min=-16.0, max=12.0)
                Yb_pred_t = torch.expm1(pred_lat_clamped).view(-1, K, K)
            else:
                Yb_true_t = None; Yb_pred_t = None
            if svd is not None:
                V_tensor = torch.from_numpy(svd.components_).to(zb.device).float()
                pred_lat_clamped = torch.clamp(pred_lat, min=-16.0, max=12.0)
                Yb_pred_cnt = torch.expm1(torch.matmul(pred_lat_clamped, V_tensor)).view(-1, K, K)
                Yb_true_cnt = torch.expm1(torch.matmul(zb, V_tensor)).view(-1, K, K)
                eps = 1e-9
                shp_pred = Yb_pred_cnt / (Yb_pred_cnt.sum(dim=(1,2), keepdim=True) + eps)
                shp_true = Yb_true_cnt / (Yb_true_cnt.sum(dim=(1,2), keepdim=True) + eps)
                L1_loss = torch.nn.functional.l1_loss(shp_pred, shp_true)
                L2_loss = huber_loss(shp_pred, shp_true) if args.use_huber else torch.nn.functional.mse_loss(shp_pred, shp_true)
                L_shape = 0.7 * L1_loss + 0.3 * L2_loss
                if args.use_total_head and batch_totals is not None:
                    L_total = (torch.log(batch_totals.to(pred_scale.dtype) + 1e-6) - log_total.squeeze(-1)).pow(2).mean()
                else:
                    L_total = 0.0
                # Initialize all loss components to zero for SVD branch
                L_marg = L_diag = L_pct = L_log = L_entropy = L_variance = 0.0
                L_level_var = 0.0  # SVD branch: no level-aggregated variance loss
            else:
                if args.use_total_head and pred_scale is not None:
                    t_pred = pred_scale.squeeze()
                    t_true = Yb_true_t.sum(dim=(1,2))
                    pattern_mat = torch.clamp(Yb_pred_t, min=1e-12)
                    denom = pattern_mat.sum(dim=(1,2), keepdim=True).clamp_min(1e-9)
                    shape_pred = pattern_mat / denom
                    shape_true = Yb_true_t / (t_true.view(-1,1,1) + 1e-9)
                    if args.loss_type == "poisson":
                        shape_pred_clamped = torch.clamp(shape_pred, min=1e-6)
                        L_shape = torch.nn.functional.poisson_nll_loss(shape_pred_clamped, shape_true, log_input=False, full=True)
                    elif args.loss_type == "hybrid":
                        if args.use_magnitude_weighting:
                            weights = magnitude_weights(shape_true)
                            L1_loss = torch.nn.functional.l1_loss(shape_pred * weights, shape_true * weights)
                            MSE_loss = (huber_loss if args.use_huber else torch.nn.functional.mse_loss)(shape_pred * weights, shape_true * weights)
                        else:
                            L1_loss = torch.nn.functional.l1_loss(shape_pred, shape_true)
                            MSE_loss = (huber_loss if args.use_huber else torch.nn.functional.mse_loss)(shape_pred, shape_true)
                        L_shape = 0.7 * L1_loss + 0.3 * MSE_loss
                    else:
                        if args.loss_type == "mse":
                            L_shape = nn.MSELoss()(pred_lat, zb)
                        elif args.loss_type == "l1":
                            L_shape = nn.L1Loss()(pred_lat, zb)
                        else:
                            L_shape = primary_loss(pred_lat, zb)
                    L_total = (torch.log(t_true + 1e-6) - log_total.squeeze(-1)).pow(2).mean()
                    r_true_shape = shape_true.sum(dim=2)
                    c_true_shape = shape_true.sum(dim=1)
                    r_pred_shape = shape_pred.sum(dim=2)
                    c_pred_shape = shape_pred.sum(dim=1)
                    L_marg = ((r_true_shape - r_pred_shape)**2).mean() + ((c_true_shape - c_pred_shape)**2).mean()
                    diag_true_shape = torch.diagonal(shape_true, dim1=1, dim2=2)
                    diag_pred_shape = torch.diagonal(shape_pred, dim1=1, dim2=2)
                    L_diag = ((diag_true_shape - diag_pred_shape)**2).mean()
                    L_pct = smape_torch(shape_true, shape_pred, eps=args.pct_eps)
                    Yb_pred_t = shape_pred * t_pred.view(-1,1,1)

                    # Enable the remaining regularizers even in total-head mode (computed on reconstructed counts).
                    L_log = log_space_loss(Yb_true_t, Yb_pred_t)
                    shape_for_entropy = shape_pred.clamp_min(1e-12)
                    L_entropy = -(shape_for_entropy * torch.log(shape_for_entropy)).sum(dim=(1,2)).mean()
                    var_true = torch.var(Yb_true_t.view(Yb_true_t.shape[0], -1), dim=1)
                    var_pred = torch.var(Yb_pred_t.view(Yb_pred_t.shape[0], -1), dim=1)
                    L_variance = torch.nn.functional.mse_loss(var_pred, var_true)

                    # Optional: match the validation metric structure by training on per-level aggregated sMAPE.
                    L_level_smape = 0.0
                    if batch_levels is not None and args.lambda_level_smape > 0.0:
                        unique_lvls = torch.unique(batch_levels)
                        acc = 0.0
                        n_used = 0
                        for lv in unique_lvls:
                            mask = (batch_levels == lv)
                            if mask.sum() < 1:
                                continue
                            T_lvl = Yb_true_t[mask].sum(dim=0)
                            P_lvl = Yb_pred_t[mask].sum(dim=0)
                            acc = acc + smape_torch(T_lvl, P_lvl, eps=args.pct_eps)
                            n_used += 1
                        if n_used > 0:
                            L_level_smape = acc / n_used
                    
                    # Level-aggregated variance loss (matches the printed metric) - total-head branch
                    L_level_var = 0.0
                    if batch_levels is not None and args.lambda_level_var > 0.0:
                        # unique levels in this batch
                        unique_lvls = torch.unique(batch_levels)
                        acc = 0.0
                        n_used = 0
                        for lv in unique_lvls:
                            mask = (batch_levels == lv)
                            if mask.sum() < 2:
                                # need at least 2 windows to make a meaningful aggregated variance
                                continue
                            # aggregate all windows for this level inside the batch
                            T_lvl = Yb_true_t[mask].sum(dim=0).view(-1)  # true level-total, flattened
                            P_lvl = Yb_pred_t[mask].sum(dim=0).view(-1)  # pred level-total, flattened
                            v_true = T_lvl.var()
                            v_pred = P_lvl.var()
                            acc = acc + (v_pred - v_true).pow(2)
                            n_used += 1
                        if n_used > 0:
                            L_level_var = acc / n_used
                else:
                    if args.loss_type == "poisson":
                        Yb_pred_t_clamped = torch.clamp(Yb_pred_t, min=1e-6)
                        L_shape = torch.nn.functional.poisson_nll_loss(Yb_pred_t_clamped, Yb_true_t, log_input=False, full=True)
                    elif args.loss_type == "hybrid":
                        if args.use_magnitude_weighting:
                            weights = magnitude_weights(Yb_true_t)
                            L1_loss = torch.nn.functional.l1_loss(Yb_pred_t * weights, Yb_true_t * weights)
                            MSE_loss = (huber_loss if args.use_huber else torch.nn.functional.mse_loss)(Yb_pred_t * weights, Yb_true_t * weights)
                        else:
                            L1_loss = torch.nn.functional.l1_loss(Yb_pred_t, Yb_true_t)
                            MSE_loss = (huber_loss if args.use_huber else torch.nn.functional.mse_loss)(Yb_pred_t, Yb_true_t)
                        L_shape = 0.7 * L1_loss + 0.3 * MSE_loss
                    else:
                        if args.loss_type == "mse":
                            L_shape = nn.MSELoss()(pred_lat, zb)
                        elif args.loss_type == "l1":
                            L_shape = nn.L1Loss()(pred_lat, zb)
                        else:
                            L_shape = primary_loss(pred_lat, zb)
                    r_true = Yb_true_t.sum(dim=2); c_true = Yb_true_t.sum(dim=1)
                    r_pred = Yb_pred_t.sum(dim=2); c_pred = Yb_pred_t.sum(dim=1)
                    L_marg = ((r_true - r_pred)**2).mean() + ((c_true - c_pred)**2).mean()
                    diag_true = torch.diagonal(Yb_true_t, dim1=1, dim2=2)
                    diag_pred = torch.diagonal(Yb_pred_t, dim1=1, dim2=2)
                    L_diag = ((diag_true - diag_pred)**2).mean()
                    L_pct = smape_torch(Yb_true_t, Yb_pred_t, eps=args.pct_eps)
                    L_log = log_space_loss(Yb_true_t, Yb_pred_t)
                    L_total = ((Yb_true_t.sum(dim=(1,2)) - Yb_pred_t.sum(dim=(1,2)))**2).mean()
                    t_pred = Yb_pred_t.sum(dim=(1,2), keepdim=True).clamp_min(1e-9)
                    shape_pred = (Yb_pred_t / t_pred).clamp_min(1e-12)
                    entropy = -(shape_pred * torch.log(shape_pred + 1e-12)).sum(dim=(1,2)).mean()
                    L_entropy = entropy
                    
                    # Variance-preserving loss to help with variance difference
                    var_true = torch.var(Yb_true_t.view(Yb_true_t.shape[0], -1), dim=1)
                    var_pred = torch.var(Yb_pred_t.view(Yb_pred_t.shape[0], -1), dim=1)
                    L_variance = torch.nn.functional.mse_loss(var_pred, var_true)
                    
                    # Optional: per-level aggregated sMAPE loss (non-total-head branch)
                    L_level_smape = 0.0
                    if batch_levels is not None and args.lambda_level_smape > 0.0:
                        unique_lvls = torch.unique(batch_levels)
                        acc = 0.0
                        n_used = 0
                        for lv in unique_lvls:
                            mask = (batch_levels == lv)
                            if mask.sum() < 1:
                                continue
                            T_lvl = Yb_true_t[mask].sum(dim=0)
                            P_lvl = Yb_pred_t[mask].sum(dim=0)
                            acc = acc + smape_torch(T_lvl, P_lvl, eps=args.pct_eps)
                            n_used += 1
                        if n_used > 0:
                            L_level_smape = acc / n_used
                    
                    # Level-aggregated variance loss (matches the printed metric) - non-total-head branch
                    L_level_var = 0.0
                    if batch_levels is not None and args.lambda_level_var > 0.0:
                        # unique levels in this batch
                        unique_lvls = torch.unique(batch_levels)
                        acc = 0.0
                        n_used = 0
                        for lv in unique_lvls:
                            mask = (batch_levels == lv)
                            if mask.sum() < 2:
                                # need at least 2 windows to make a meaningful aggregated variance
                                continue
                            # aggregate all windows for this level inside the batch
                            T_lvl = Yb_true_t[mask].sum(dim=0).view(-1)  # true level-total, flattened
                            P_lvl = Yb_pred_t[mask].sum(dim=0).view(-1)  # pred level-total, flattened
                            v_true = T_lvl.var()
                            v_pred = P_lvl.var()
                            acc = acc + (v_pred - v_true).pow(2)
                            n_used += 1
                        if n_used > 0:
                            L_level_var = acc / n_used
            L = (
                L_shape
                + effective_lambda_marg * L_marg
                + args.lambda_total * L_total
                + args.lambda_diag * L_diag
                + args.lambda_pct * L_pct
                + args.lambda_log * L_log
                + args.lambda_entropy * L_entropy
                + args.lambda_variance * L_variance
                + args.lambda_level_var * L_level_var
                + args.lambda_level_smape * L_level_smape
            )
            L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(float(L.detach().cpu().item()))

        # Validation
        model.eval()
        with torch.no_grad():
            va_losses = []
            val_predictions = []
            val_targets = []
            for batch in dl_va:
                # unpack batch based on predict_fractions & presence of level_ids
                if args.predict_fractions:
                    if len(batch) == 4:
                        xb, zb, batch_totals, batch_levels = batch
                    else:
                        xb, zb, batch_totals = batch
                        batch_levels = None
                else:
                    if len(batch) == 3:
                        xb, zb, batch_levels = batch
                        batch_totals = None
                    else:
                        xb, zb = batch
                        batch_totals = None
                        batch_levels = None
                if args.use_total_head:
                    pred_lat, log_total = model(xb)
                    pred_scale = torch.exp(log_total).squeeze(-1)
                else:
                    pred_lat = model(xb); pred_scale = None
                if svd is None:
                    Yb_true_t = torch.expm1(zb).view(-1, K, K)
                    pred_lat_clamped = torch.clamp(pred_lat, min=-16.0, max=12.0)
                    Yb_pred_t = torch.expm1(pred_lat_clamped).view(-1, K, K)
                else:
                    V_tensor = torch.from_numpy(svd.components_).to(zb.device).float()
                    pred_lat_clamped = torch.clamp(pred_lat, min=-16.0, max=12.0)
                    Yb_pred_cnt = torch.expm1(torch.matmul(pred_lat_clamped, V_tensor)).view(-1, K, K)
                    Yb_true_cnt = torch.expm1(torch.matmul(zb, V_tensor)).view(-1, K, K)
                    eps = 1e-9
                    shp_pred = Yb_pred_cnt / (Yb_pred_cnt.sum(dim=(1,2), keepdim=True) + eps)
                    shp_true = Yb_true_cnt / (Yb_true_cnt.sum(dim=(1,2), keepdim=True) + eps)
                    Yb_true_t = Yb_true_cnt; Yb_pred_t = Yb_pred_cnt
                if args.use_total_head and pred_scale is not None:
                    pattern_mat = torch.clamp(Yb_pred_t, min=1e-12)
                    denom = pattern_mat.sum(dim=(1,2), keepdim=True).clamp_min(1e-9)
                    shape_pred = pattern_mat / denom
                    Yb_pred_t = shape_pred * pred_scale.view(-1, 1, 1)
                val_predictions.append(Yb_pred_t.cpu().numpy())
                val_targets.append(Yb_true_t.cpu().numpy())
                if svd is not None:
                    L1_loss = torch.nn.functional.l1_loss(shp_pred, shp_true)
                    L2_loss = huber_loss(shp_pred, shp_true) if args.use_huber else torch.nn.functional.mse_loss(shp_pred, shp_true)
                    L_pred = 0.7 * L1_loss + 0.3 * L2_loss
                elif args.loss_type == "poisson":
                    Yb_pred_t_clamped = torch.clamp(Yb_pred_t, min=1e-6)
                    L_pred = torch.nn.functional.poisson_nll_loss(Yb_pred_t_clamped, Yb_true_t, log_input=False, full=True)
                elif args.loss_type == "hybrid":
                    if args.use_magnitude_weighting:
                        weights = magnitude_weights(Yb_true_t)
                        L1_loss = torch.nn.functional.l1_loss(Yb_pred_t * weights, Yb_true_t * weights)
                        MSE_loss = (huber_loss if args.use_huber else torch.nn.functional.mse_loss)(Yb_pred_t * weights, Yb_true_t * weights)
                    else:
                        L1_loss = torch.nn.functional.l1_loss(Yb_pred_t, Yb_true_t)
                        MSE_loss = (huber_loss if args.use_huber else torch.nn.functional.mse_loss)(Yb_pred_t, Yb_true_t)
                    L_pred = 0.7 * L1_loss + 0.3 * MSE_loss
                else:
                    if args.loss_type == "mse":
                        L_pred = nn.MSELoss()(pred_lat, zb)
                    elif args.loss_type == "l1":
                        L_pred = nn.L1Loss()(pred_lat, zb)
                    else:
                        L_pred = primary_loss(pred_lat, zb)
                r_true = Yb_true_t.sum(dim=2); c_true = Yb_true_t.sum(dim=1)
                r_pred = Yb_pred_t.sum(dim=2); c_pred = Yb_pred_t.sum(dim=1)
                L_marg = ((r_true - r_pred)**2).mean() + ((c_true - c_pred)**2).mean()
                diag_true = torch.diagonal(Yb_true_t, dim1=1, dim2=2)
                diag_pred = torch.diagonal(Yb_pred_t, dim1=1, dim2=2)
                L_diag = ((diag_true - diag_pred)**2).mean()
                L_pct = smape_torch(Yb_true_t, Yb_pred_t, eps=args.pct_eps)
                if args.use_total_head and batch_totals is not None:
                    L_total = (torch.log(batch_totals.to(pred_scale.dtype) + 1e-6) - log_total.squeeze(-1)).pow(2).mean()
                else:
                    L_total = ((Yb_true_t.sum(dim=(1,2)) - Yb_pred_t.sum(dim=(1,2)))**2).mean()
                L = L_pred + effective_lambda_marg * L_marg + args.lambda_total * L_total + args.lambda_diag * L_diag + args.lambda_pct * L_pct
                va_losses.append(float(L.detach().cpu().item()))
            mtr = float(np.mean(tr_losses)) if tr_losses else np.nan
            mva = float(np.mean(va_losses)) if va_losses else np.nan
            if val_predictions and val_targets:
                Y_val_pred = np.concatenate(val_predictions, axis=0)
                Y_val_true = np.concatenate(val_targets, axis=0)
                if args.use_ipf:
                    Y_ipf = np.empty_like(Y_val_pred)
                    for i in range(len(Y_val_pred)):
                        dep_in, arr_in = va_marginals[i]
                        model_total = Y_val_pred[i].sum()
                        if estimate_total_from_counts is not None:
                            target_total = estimate_total_from_counts(dep_in, arr_in, model_total)
                        else:
                            target_total = 0.5 * (dep_in.sum() + arr_in.sum())
                        dep_s, arr_s = make_consistent_marginals(dep_in, arr_in, target_total)
                        if args.ipf_blend_alpha > 0:
                            r_pred_m, c_pred_m = marginals(Y_val_pred[i].reshape(1, K, K))
                            dep_s = (1 - args.ipf_blend_alpha) * dep_s + args.ipf_blend_alpha * r_pred_m[0]
                            arr_s = (1 - args.ipf_blend_alpha) * arr_s + args.ipf_blend_alpha * c_pred_m[0]
                            dep_s, arr_s = make_consistent_marginals(dep_s, arr_s, target_total)
                        rho = args.ipf_rho; target_total = rho * target_total
                        baseline = np.outer(dep_s, arr_s) / (target_total + 1e-12)
                        beta = args.ipf_beta
                        seed = (1 - beta) * np.maximum(Y_val_pred[i], 1e-12) + beta * baseline
                        Y_ipf[i] = ipf_adjust(seed, dep_s, arr_s, n_iter=max(args.ipf_iter, 40), clip=(args.ipf_clip_min, args.ipf_clip_max), target_total=target_total)
                    Y_val_pred_for_metric = Y_ipf
                else:
                    Y_val_pred_for_metric = Y_val_pred
                va_lv = va_levels
                sum_val = summarize_levelwise(Y_val_true, Y_val_pred_for_metric, va_lv, K, pct_eps=args.pct_eps)
                val_smape = sum_val["smape_all"]
            else:
                val_smape = mva
        sched.step(val_smape)
        improved = (val_smape < (best_val - args.min_delta)) if (args.min_delta and args.min_delta > 0) else (val_smape + 1e-9 < best_val)
        if improved:
            best_val = val_smape
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            if epoch >= args.min_epochs:
                patience -= 1
                if patience <= 0:
                    print(f"Early stop @ {epoch} | best val sMAPE {best_val:.4f}%")
                    break
        if args.tune_report_every and (epoch % args.tune_report_every == 0):
            # Machine-readable progress line for external tuners/pruners
            print(f"TUNE_REPORT {epoch} {val_smape:.6f}", flush=True)
        if epoch % 25 == 0:
            print(f"Epoch {epoch:4d} | train {mtr:.5f} | val {mva:.5f} | val_sMAPE {val_smape:.2f}% | lr {opt.param_groups[0]['lr']:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    if args.print_best_val:
        # Easy to parse from an external tuner script
        print(f"BEST_VAL_SMAPE {best_val:.6f}")

    if args.skip_save:
        return

    # ---- Evaluation on test ----
    def eval_split(Xs, Zs, split_name="TEST", levels=None, windows=None, input_marginals=None, totals=None, groups=None, mapping=None):
        model.eval()
        with torch.no_grad():
            if args.use_total_head:
                Z_pred, log_total = model(torch.from_numpy(Xs.astype(np.float32)))
                Z_pred = Z_pred.cpu().numpy()
                scale_pred = torch.exp(log_total).cpu().numpy()
            else:
                Z_pred = model(torch.from_numpy(Xs.astype(np.float32))).cpu().numpy()
                scale_pred = None
        totals_for_decode = None
        if args.predict_fractions and not args.use_total_head:
            totals_for_decode = totals
        Y_pred = decode_to_OD(Z_pred, totals_for_decode)
        if args.use_total_head:
            pattern = np.maximum(Y_pred, 1e-12)
            shape = pattern / (pattern.sum(axis=1, keepdims=True) + 1e-9)
            Y_pred = shape * scale_pred.reshape(-1, 1)
        if args.global_rescale and not args.use_total_head and not args.use_ipf and input_marginals is not None:
            Y_pred = apply_global_rescale(Y_pred, input_marginals, estimate_total_from_counts)
        # Apply gamma sharpening only to VAL and TEST, not TRAIN (to reduce TRAIN var diff)
        gamma_eff = args.gamma_sharpen
        if split_name == "TRAIN":
            gamma_eff = 1.0  # no sharpening on TRAIN metrics
        if gamma_eff > 1.0:
            Y_pred = sharpen_gamma(Y_pred, gamma=gamma_eff)
        Y_true = decode_to_OD(Zs, totals)
        if args.use_ipf and input_marginals is not None:
            Y_pred_ipf = np.empty_like(Y_pred).reshape(-1, K, K)
            for i in range(len(Y_pred)):
                dep_in, arr_in = input_marginals[i]
                if estimate_total_from_counts is not None:
                    model_total = Y_pred[i].sum()
                    target_total = estimate_total_from_counts(dep_in, arr_in, model_total)
                else:
                    target_total = 0.5 * (dep_in.sum() + arr_in.sum())
                dep_s, arr_s = make_consistent_marginals(dep_in, arr_in, target_total)
                alpha = args.ipf_blend_alpha
                if alpha > 0:
                    r_pred, c_pred = marginals(Y_pred[i].reshape(1, K, K))
                    r_pred, c_pred = r_pred[0], c_pred[0]
                    dep_s = (1 - alpha) * dep_s + alpha * r_pred
                    arr_s = (1 - alpha) * arr_s + alpha * c_pred
                    dep_s, arr_s = make_consistent_marginals(dep_s, arr_s, target_total)
                rho = args.ipf_rho; target_total = rho * target_total
                baseline = np.outer(dep_s, arr_s) / (target_total + 1e-12)
                beta = args.ipf_beta
                Y_pred_2d = Y_pred[i].reshape(K, K)
                seed = (1 - beta) * np.maximum(Y_pred_2d, 1e-12) + beta * baseline
                Y_pred_ipf[i] = ipf_adjust(seed, dep_s, arr_s, n_iter=max(args.ipf_iter, 40), clip=(args.ipf_clip_min, args.ipf_clip_max), target_total=target_total)
        else:
            Y_pred_ipf = None
        # Keep non-IPF predictions separate for aggregation
        Y_pred_non_ipf = Y_pred.copy()
        
        # For fair comparison, aggregate predictions to match the chunked structure
        if groups is not None and len(groups) > 0:
            # Aggregate non-IPF predictions
            Y_pred_agg, levels_agg, windows_agg, marginals_agg = aggregate_predictions_for_comparison(
                Y_pred_non_ipf, groups, levels, windows, input_marginals, K, mapping
            )
            # Aggregate true values to match the same structure
            Y_true_agg, _, _, _ = aggregate_predictions_for_comparison(
                Y_true, groups, levels, windows, input_marginals, K, mapping
            )
            
            # If IPF is enabled, aggregate IPF predictions separately
            if args.use_ipf and Y_pred_ipf is not None:
                Y_pred_ipf_agg, _, _, _ = aggregate_predictions_for_comparison(
                    Y_pred_ipf.reshape(Y_pred_ipf.shape[0], -1), groups, levels, windows, input_marginals, K, mapping
                )
            else:
                Y_pred_ipf_agg = None
            
            Y_true_3d = Y_true_agg.reshape(-1, K, K) if Y_true_agg.ndim == 2 else Y_true_agg
            Y_pred_3d = Y_pred_agg.reshape(-1, K, K) if Y_pred_agg.ndim == 2 else Y_pred_agg
            levels_final = levels_agg
            windows_final = windows_agg
        else:
            # No aggregation needed (legacy mode)
            Y_true_3d = Y_true.reshape(-1, K, K) if Y_true.ndim == 2 else Y_true
            Y_pred_3d = Y_pred_non_ipf.reshape(-1, K, K) if Y_pred_non_ipf.ndim == 2 else Y_pred_non_ipf
            levels_final = levels
            windows_final = windows
            Y_pred_ipf_agg = Y_pred_ipf.reshape(-1, K, K) if Y_pred_ipf is not None and Y_pred_ipf.ndim == 2 else Y_pred_ipf
        
        # Use IPF-corrected predictions for summary if IPF is enabled
        if args.use_ipf and Y_pred_ipf_agg is not None:
            Y_pred_3d_for_summary = Y_pred_ipf_agg.reshape(-1, K, K) if Y_pred_ipf_agg.ndim == 2 else Y_pred_ipf_agg
        else:
            Y_pred_3d_for_summary = Y_pred_3d
        
        # Calculate summary using the FINAL data (after IPF if enabled)
        summary = summarize_levelwise(Y_true_3d, Y_pred_3d_for_summary, levels_final, K, pct_eps=args.pct_eps)
        
        if args.excel_output and levels_final is not None and windows_final is not None:
            print(f"\nCreating Excel output for {split_name}...")
            # Pass aggregated IPF predictions to Excel
            create_excel_output(Y_true_3d, Y_pred_3d, Y_pred_ipf_agg, levels_final, windows_final, K, args.save_dir, split_name, pct_eps=args.pct_eps)
        print(f"\n== {split_name} ==")
        print(f"sMAPE(all): {summary['smape_all']:.2f}% (PRIMARY) | Stabilized: {summary['smape_all_stab']:.2f}%")
        print(f"MAE(all): {summary['mae_all']:.6f} | RMSE(all): {summary['rmse_all']:.6f}")
        print(f"MAE(nz) : {summary['mae_nz']:.6f} | RMSE(nz) : {summary['rmse_nz']:.6f} | sMAPE(nz): {summary['smape_nz']:.2f}% | Stabilized: {summary['smape_nz_stab']:.2f}%")
        print(f"Marginal MAE (rows/cols avg): {summary['marg_mae']:.6f}")
        print(f"Total Traffic - True: {summary['total_true']:.2f} | Pred: {summary['total_pred']:.2f} | Error: {summary['total_error']:.2f} ({summary['total_error_pct']:.2f}%)")
        print(f"Mean - True: {summary['mean_true']:.6f} | Pred: {summary['mean_pred']:.6f}")
        print(f"Variance - True: {summary['var_true']:.6f} | Pred: {summary['var_pred']:.6f} | Var Diff: {summary['var_diff_pct']:.2f}% (avg per-level)")
        print(f"Total flow Diff (avg per-level): {summary['total_error_pct_avg']:.2f}%")
        if args.use_ipf and Y_pred_ipf_agg is not None:
            Yipf_3d = Y_pred_ipf_agg if Y_pred_ipf_agg.ndim == 3 else Y_pred_ipf_agg.reshape(-1, K, K)
            sum_ipf = summarize_levelwise(Y_true_3d, Yipf_3d, levels_final, K, pct_eps=args.pct_eps)
            print(f"\n== {split_name} (IPF-corrected, REALISTIC) ==")
            print(f"sMAPE(all, IPF): {sum_ipf['smape_all']:.2f}% (PRIMARY)")
            print(f"MAE(all): {sum_ipf['mae_all']:.6f} | RMSE(all): {sum_ipf['rmse_all']:.6f} | sMAPE(nz, IPF): {sum_ipf['smape_nz']:.2f}%")
            print(f"Marginal MAE (rows/cols avg): {sum_ipf['marg_mae']:.6f}")
            print(f"Total Traffic - True: {sum_ipf['total_true']:.2f} | Pred: {sum_ipf['total_pred']:.2f} | Error: {sum_ipf['total_error']:.2f} ({sum_ipf['total_error_pct']:.2f}%)")
            print(f"Mean - Pred: {sum_ipf['mean_pred']:.6f}")
            print(f"Variance - Pred (IPF): {sum_ipf['var_pred']:.6f} | Var Diff: {sum_ipf['var_diff_pct']:.2f}% (avg per-level)")
            print(f"Total flow Diff (avg per-level): {sum_ipf['total_error_pct_avg']:.2f}%")
            return dict(mae_all=summary['mae_all'], rmse_all=summary['rmse_all'], mae_nz=summary['mae_nz'], rmse_nz=summary['rmse_nz'], 
                        marg_mae=summary['marg_mae'], mean_true=summary['mean_true'], mean_pred=summary['mean_pred'], 
                        var_true=summary['var_true'], var_pred=summary['var_pred'],
                        mae_ipf=sum_ipf['mae_all'], rmse_ipf=sum_ipf['rmse_all'], marg_mae_ipf=sum_ipf['marg_mae'],
                        mean_pred_ipf=sum_ipf['mean_pred'], var_pred_ipf=sum_ipf['var_pred'])
        else:
            return dict(mae_all=summary['mae_all'], rmse_all=summary['rmse_all'], mae_nz=summary['mae_nz'], rmse_nz=summary['rmse_nz'], 
                        marg_mae=summary['marg_mae'], mean_true=summary['mean_true'], mean_pred=summary['mean_pred'], 
                        var_true=summary['var_true'], var_pred=summary['var_pred'])

    print("\nEvaluating…")
    if args.optionA and not args.aggregate_per_level:
        # Pass groups for proper aggregation - use original metadata for aggregation
        _ = eval_split(Xtr_s, Ztr, "TRAIN", levels, windows, marginals_from_input, tr_totals, groups_tr, mapping_tr)
        _ = eval_split(Xva_s, Zva, "VAL",   levels, windows, marginals_from_input, va_totals, groups_va, mapping_va)
        _ = eval_split(Xte_s, Zte, "TEST",  levels, windows, marginals_from_input, te_totals, groups_te, mapping_te)
    else:
        # Legacy mode - no groups
        _ = eval_split(Xtr_s, Ztr, "TRAIN", tr_levels, tr_windows, tr_marginals, tr_totals)
        _ = eval_split(Xva_s, Zva, "VAL",   va_levels, va_windows, va_marginals, va_totals)
        _ = eval_split(Xte_s, Zte, "TEST",  te_levels, te_windows, te_marginals, te_totals)

    # ---- Save artifacts ----
    import joblib
    joblib.dump(x_scaler, os.path.join(args.save_dir, "x_scaler_tunned.pkl"))
    if svd is not None:
        joblib.dump(svd, os.path.join(args.save_dir, "y_svd_tunned.pkl"))
    torch.save(model.state_dict(), os.path.join(args.save_dir, "model_tunned.pt"))
    meta_levels = sorted(list(set(tr_levels + va_levels + te_levels)))
    meta = dict(K=int(K), use_svd=args.use_svd, ry=args.ry, hidden=tuple(int(x) for x in (args.hidden.split(",") if args.hidden else [])),
                lambda_marg=args.lambda_marg, levels=meta_levels,
                optionA=dict(keep_first_slots=args.keep_first_slots, split_slots=args.split_slots, chunk_size=args.chunk_size) if args.optionA else None,
                use_total_head=args.use_total_head, pct_eps=args.pct_eps, log_x=args.log_x,
                use_ipf=args.use_ipf, global_rescale=args.global_rescale, gamma_sharpen=args.gamma_sharpen,
                ipf_rho=args.ipf_rho, ipf_beta=args.ipf_beta, ipf_iter=args.ipf_iter,
                ipf_clip_min=args.ipf_clip_min, ipf_clip_max=args.ipf_clip_max, ipf_blend_alpha=args.ipf_blend_alpha)
    with open(os.path.join(args.save_dir, "meta_tunned.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[Saved] artifacts in {args.save_dir}")

if __name__ == "__main__":
    main()
