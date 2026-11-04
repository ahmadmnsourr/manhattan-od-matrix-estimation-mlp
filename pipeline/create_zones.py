#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intersections: Take all non-virtual intersections and their coordinates (x,y). Run k-means clustering with K clusters on these points. 
Each intersection gets the label of its nearest cluster center. The zone centroid is the k-means center.

Roads: For each road/edge, if its from or to intersection has a zone, use that zone. Otherwise, take the mean of its polyline points as a fallback location 
and assign the zone of the nearest intersection zone center.
"""
"""
Zoning for a CityFlow-style road network.

Usage:
  python make_zones.py --roadnet /path/to/roadnet_manhattan.json --zones 20 --out /path/to/prefix
"""

import os
import re
import math
import json
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# Default roadnet path: current script directory + Manhattan Network/MPLight/data/manhattan/roadnet_manhattan.json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROADNET = os.path.join(
    SCRIPT_DIR,
    "Manhattan Network",
    "MPLight",
    "data",
    "manhattan",
    "roadnet_manhattan.json",
)

# Default output prefix: current script directory + Refined Data/Zonage/Zonage (so files are inside the folder)
DEFAULT_OUT_PREFIX = os.path.join(
    SCRIPT_DIR,
    "Refined Data",
    "Zonage",
    "Zonage",
)

# ---------------------------
# KMeans (numpy) with k-means++ init
# ---------------------------
def _kmeans_pp_init(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centroids = np.empty((K, X.shape[1]), dtype=float)
    # first centroid
    i0 = rng.integers(0, n)
    centroids[0] = X[i0]
    # remaining centroids by D^2 sampling
    d2 = np.sum((X - centroids[0])**2, axis=1)
    for k in range(1, K):
        probs = d2 / (d2.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        centroids[k] = X[idx]
        d2 = np.minimum(d2, np.sum((X - centroids[k])**2, axis=1))
    return centroids

def kmeans_numpy(X: np.ndarray, K: int, max_iter: int = 200, tol: float = 1e-4, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    C = _kmeans_pp_init(X, K, rng)
    prev_inertia = None
    for _ in range(max_iter):
        # assign
        dists = np.sqrt(((X[:, None, :] - C[None, :, :])**2).sum(axis=2))
        labels = dists.argmin(axis=1)
        # update
        C_new = np.zeros_like(C)
        for k in range(K):
            pts = X[labels == k]
            if len(pts) > 0:
                C_new[k] = pts.mean(axis=0)
            else:
                C_new[k] = X[rng.integers(0, X.shape[0])]
        inertia = ((X - C_new[labels])**2).sum()
        if prev_inertia is not None and abs(prev_inertia - inertia) < tol * max(1.0, prev_inertia):
            C = C_new
            break
        C = C_new
        prev_inertia = inertia
    return labels.astype(int), C

# ---------------------------
# Roadnet parsing (supports two common schemas)
#   A) {"static": {"nodes": [...], "edges": [...]}}
#   B) {"intersections": [...], "roads": [...]}
# ---------------------------
def load_roadnet(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_nodes_edges(data: dict) -> Tuple[List[dict], List[dict], str]:
    """
    Returns (nodes, edges, schema) where:
      nodes: list of dict with keys -> id, point(x,y), virtual
      edges: list of dict with keys -> id, from, to, points (list of [x,y])
    """
    # Schema A: {"static": {"nodes": [...], "edges": [...]} }
    if isinstance(data, dict) and "static" in data and isinstance(data["static"], dict):
        nodes_raw = data["static"].get("nodes", [])
        edges_raw = data["static"].get("edges", [])
        nodes = []
        for n in nodes_raw:
            nid = str(n.get("id"))
            pt = n.get("point", [None, None])
            x, y = float(pt[0]), float(pt[1])
            virtual = bool(n.get("virtual", False))
            nodes.append({"id": nid, "x": x, "y": y, "virtual": virtual})
        edges = []
        for e in edges_raw:
            rid = str(e.get("id"))
            frm = str(e.get("from"))
            to = str(e.get("to"))
            pts = e.get("points", []) or []
            # ensure [[x,y], ...]
            pts = [[float(p[0]), float(p[1])] for p in pts if p is not None and len(p) >= 2]
            edges.append({"id": rid, "from": frm, "to": to, "points": pts})
        return nodes, edges, "static"

    # Schema B: {"intersections": [...], "roads": [...]}
    if "intersections" in data and "roads" in data:
        nodes = []
        for it in data["intersections"]:
            nid = str(it.get("id"))
            pt = it.get("point", {}) or {}
            x, y = float(pt.get("x", 0.0)), float(pt.get("y", 0.0))
            virtual = bool(it.get("virtual", False))
            nodes.append({"id": nid, "x": x, "y": y, "virtual": virtual})
        edges = []
        for rd in data["roads"]:
            rid = str(rd.get("id"))
            frm = str(rd.get("startIntersection"))
            to = str(rd.get("endIntersection"))
            pts = rd.get("points", []) or []
            pts = [[float(p.get("x", 0.0)), float(p.get("y", 0.0))] for p in pts]
            edges.append({"id": rid, "from": frm, "to": to, "points": pts})
        return nodes, edges, "intersections/roads"

    raise ValueError("Unrecognized roadnet schema: expected 'static' or 'intersections/roads'.")

# ---------------------------
# Zoning
# ---------------------------
def build_zones(nodes: List[dict], K: int, drop_virtual: bool = True, seed: int = 42):
    coords, ids = [], []
    for n in nodes:
        if drop_virtual and n.get("virtual", False):
            continue
        x, y = n.get("x"), n.get("y")
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        ids.append(n["id"])
        coords.append([x, y])
    X = np.asarray(coords, dtype=float)
    if len(X) < K:
        raise ValueError(f"Not enough intersections ({len(X)}) to form {K} zones.")
    labels, centers = kmeans_numpy(X, K, seed=seed)
    int_zone = {ids[i]: int(labels[i]) for i in range(len(ids))}
    zones_meta = []
    for z in range(K):
        members = X[labels == z]
        zones_meta.append({
            "zone": int(z),
            "n_intersections": int(len(members)),
            "center_x": float(centers[z, 0]),
            "center_y": float(centers[z, 1]),
        })
    return int_zone, zones_meta

def assign_roads_to_zones(edges: List[dict], int_zone: Dict[str, int], nodes_index: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    if len(int_zone) == 0:
        raise ValueError("No intersections were assigned to zones.")
    # Pre-build array for nearest fallback
    zone_keys = list(int_zone.keys())
    zone_xy = np.array([[nodes_index[i]["x"], nodes_index[i]["y"]] for i in zone_keys], dtype=float)

    for e in edges:
        rid = e["id"]
        frm = e.get("from")
        to = e.get("to")
        z = None
        if frm in int_zone:
            z = int_zone[frm]
        elif to in int_zone:
            z = int_zone[to]
        else:
            pts = e.get("points") or []
            if pts:
                cx = float(np.mean([p[0] for p in pts]))
                cy = float(np.mean([p[1] for p in pts]))
                d2 = ((zone_xy - np.array([cx, cy]))**2).sum(axis=1)
                nearest = zone_keys[int(np.argmin(d2))]
                z = int_zone[nearest]
        rows.append({"road_id": rid, "zone": (None if z is None else int(z)), "from": frm, "to": to})
    return pd.DataFrame(rows)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Cluster road network into K zones and assign roads.")
    ap.add_argument(
        "--roadnet",
        required=False,
        default=DEFAULT_ROADNET,
        help="Path to roadnet JSON (CityFlow-style). Defaults to bundled Manhattan path.",
    )
    ap.add_argument("--zones", type=int, default=None, help="Number of zones (K). If omitted, you will be prompted.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for k-means.")
    ap.add_argument(
        "--out",
        required=False,
        default=DEFAULT_OUT_PREFIX,
        help="Output file prefix (without extension). Defaults to ./Refined Data/Zonage/Zonage",
    )
    plot_group = ap.add_mutually_exclusive_group()
    plot_group.add_argument("--plot", dest="plot", action="store_true", default=True,
                            help="Save a PNG scatter of intersections colored by zone (default: on).")
    plot_group.add_argument("--no-plot", dest="plot", action="store_false",
                            help="Disable saving the PNG scatter plot.")
    args = ap.parse_args()

    # If zones not provided, prompt the user for a positive integer
    if args.zones is None:
        while True:
            try:
                user_in = input("Enter number of zones (positive integer): ").strip()
                k = int(user_in)
                if k > 0:
                    args.zones = k
                    break
            except Exception:
                pass
            print("Invalid input. Please enter a positive integer.")

    data = load_roadnet(args.roadnet)
    nodes, edges, schema = parse_nodes_edges(data)

    # Build a quick index for nodes
    nodes_index = {n["id"]: n for n in nodes}

    # Zoning
    int_zone, zones_meta = build_zones(nodes, K=args.zones, drop_virtual=True, seed=args.seed)

    # DataFrames
    df_intersections = pd.DataFrame([
        {"intersection_id": nid, "x": nodes_index[nid]["x"], "y": nodes_index[nid]["y"],
         "virtual": nodes_index[nid]["virtual"], "zone": int_zone.get(nid)}
        for nid in nodes_index.keys()
    ])
    df_zones = pd.DataFrame(zones_meta).sort_values("zone").reset_index(drop=True)
    df_roads = assign_roads_to_zones(edges, int_zone, nodes_index)

    # Ensure output directory exists and Save
    inter_path = f"{args.out}_K{args.zones}_intersections.csv"
    zones_path = f"{args.out}_K{args.zones}_zones.csv"
    roads_path = f"{args.out}_K{args.zones}_roads.csv"
    out_dir = os.path.dirname(inter_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    df_intersections.to_csv(inter_path, index=False)
    df_zones.to_csv(zones_path, index=False)
    df_roads.to_csv(roads_path, index=False)

    print(f"[OK] Wrote:\n  {inter_path}\n  {zones_path}\n  {roads_path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt  # no seaborn, default colors
            plt.figure(figsize=(6, 8))
            for z in sorted(df_intersections["zone"].dropna().unique()):
                mask = df_intersections["zone"] == z
                plt.scatter(df_intersections.loc[mask, "x"],
                            df_intersections.loc[mask, "y"],
                            s=3, label=f"Z{int(z)}")
            plt.title(f"Intersections by Zone (K={args.zones})")
            plt.xlabel("x"); plt.ylabel("y")
            plt.legend(markerscale=4, fontsize=7, ncol=2)
            png_path = f"{args.out}_K{args.zones}_scatter.png"
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[OK] Plot saved: {png_path}")
        except Exception as e:
            print(f"[WARN] Plotting failed: {e}")

if __name__ == "__main__":
    main()