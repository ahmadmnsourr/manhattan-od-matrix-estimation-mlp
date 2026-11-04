
#!/usr/bin/env python3
"""
Process transportation network + traffic counts to produce aggregated inter-zone flows.

Inputs (expected in the same folder by default)
- roadnet_manhattan.json (road network for NetworkX)
- manhattan_7846_traffic-counts-*.csv (traffic counts; one or more chunk files like 0_299, 300_599, ...)
- Zonage_K20_intersections.csv (nodes to zones)
- Zonage_K20_roads.csv (optional; edges metadata, useful for IDs)
- Zonage_K20_zones.csv (zone polygons/metadata; not required for flow aggregation)

Outputs
- For each traffic-counts CSV, a companion file named:
  <input-stem>-zone-<chunk>.csv (e.g., manhattan_7846_traffic-counts-zone-0_299.csv)
  Columns: [time_bin, zone_o, zone_d, traffic_count]
  Where time_bin is the 5-minute slot identifier from the input data.
  If you set --split-per-slot, individual files per time_bin will also be written.

Notes
- The script is schema-tolerant: it tries to infer key column names if they differ slightly.
- Missing or unmapped nodes/edges are skipped with warnings (graceful handling).
- Boundary links are identified where the upstream node's zone differs from the downstream node's zone.
- Internal flows are links whose endpoints map to the same zone (zone_o == zone_d).
- Traffic directionality is respected if present in the counts (e.g., u->v direction).
- Use --exclude-proportion to randomly exclude a proportion of links (e.g., 0.2 for 20%%) to simulate missing data.

Usage
------
python process_interzone_flows.py \\
    --data-dir /path/to/folder \\
    --network roadnet_manhattan.json \\
    --counts manhattan_7846_traffic-counts-0_299.csv \\
    --intersections Zonage_K20_intersections.csv \\
    --roads Zonage_K20_roads.csv \\
    --zones Zonage_K20_zones.csv \\
    [--split-per-slot] [--output-dir /path/to/out] \\
    [--exclude-proportion 0.2] [--random-seed 42]
"""

import argparse
import json
import os
import re
import sys
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# -----------------------------
# Utility: tolerant column fetch
# -----------------------------
def pick_col(df: pd.DataFrame, *candidates: str, required: bool = True) -> Optional[str]:
    """Pick the first existing column in df among candidates (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise KeyError(f"None of the columns {candidates} found in: {list(df.columns)}")
    return None


def infer_time_col(df: pd.DataFrame) -> str:
    """Try several common names for the 5-minute time bin column."""
    # common options: 'time', 't', 'time_bin', 'interval', 'step', 'slot', 'minute', 'bin'
    candidates = ["time_bin", "time", "interval", "step", "slot", "t", "minute", "bin"]
    return pick_col(df, *candidates)


def infer_edge_id_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return a tuple (edge_id, u, v) of detected column names (some may be None)."""
    id_col = None
    for c in ("edge_id", "link_id", "id", "eid", "edge", "edgeID", "edgeId"):
        if c in df.columns:
            id_col = c
            break
    u_col = None
    for c in ("u", "from", "from_node", "tail", "src", "source"):
        if c in df.columns:
            u_col = c
            break
    v_col = None
    for c in ("v", "to", "to_node", "head", "dst", "target"):
        if c in df.columns:
            v_col = c
            break
    return id_col, u_col, v_col


def log(msg: str) -> None:
    print(f"[process_interzone_flows] {msg}")


# -----------------------------
# Loaders
# -----------------------------
def load_network(network_path: Path) -> nx.DiGraph:
    with open(network_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle MPLight format with "static" wrapper
    if isinstance(data, dict) and "static" in data:
        data = data["static"]
    
    # Heuristic: allow either {nodes:[], links:[]} or networkx node-link format
    if isinstance(data, dict) and "nodes" in data and ("links" in data or "edges" in data):
        links_key = "links" if "links" in data else "edges"
        G = nx.DiGraph()
        # add nodes
        for n in data["nodes"]:
            node_id = n.get("id") if isinstance(n, dict) else n
            G.add_node(node_id, **(n if isinstance(n, dict) else {}))
        # add edges
        for e in data[links_key]:
            u = e.get("source") if "source" in e else e.get("u") or e.get("from") or e.get("tail")
            v = e.get("target") if "target" in e else e.get("v") or e.get("to") or e.get("head")
            if u is None or v is None:
                continue
            # DiGraph doesn't support parallel edges with keys; ignore key here.
            G.add_edge(u, v, **e)
    else:
        # Try networkx node-link format directly
        G = nx.node_link_graph(data, directed=True, multigraph=False)
    return G


def load_zoning_files(intersections_csv: Path, roads_csv: Optional[Path] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    inter = pd.read_csv(intersections_csv)
    if roads_csv and roads_csv.exists():
        roads = pd.read_csv(roads_csv)
    else:
        roads = None
    return inter, roads


def build_node_zone_mapping(inter_df: pd.DataFrame) -> Dict:
    zone_col = pick_col(inter_df, "zone_id", "zone", "ZoneID", "zoneId", "ZONE", "k20", required=True)
    node_col = pick_col(inter_df, "node_id", "id", "node", "intersection_id", "junction", "osm_id", required=True)
    mapping = inter_df[[node_col, zone_col]].dropna().drop_duplicates()
    node_to_zone = dict(zip(mapping[node_col], mapping[zone_col]))
    return node_to_zone


# -----------------------------
# Boundary detection
# -----------------------------
def find_boundary_links(G: nx.DiGraph, node_to_zone: Dict) -> pd.DataFrame:
    """Return a DataFrame with columns [u, v, zone_o, zone_d, is_boundary]."""
    recs = []
    for u, v, data in G.edges(data=True):
        zu = node_to_zone.get(u, None)
        zv = node_to_zone.get(v, None)
        if zu is None or zv is None:
            continue
        recs.append({
            "u": u,
            "v": v,
            "zone_o": zu,
            "zone_d": zv,
            "is_boundary": (zu != zv),
        })
    df = pd.DataFrame.from_records(recs)
    return df


# -----------------------------
# Traffic counts aggregation
# -----------------------------
def load_counts(counts_path: Path) -> pd.DataFrame:
    df = pd.read_csv(counts_path)
    # Try to infer edge id columns
    id_col, u_col, v_col = infer_edge_id_cols(df)
    count_col = None
    for c in ("count", "traffic_count", "flow", "vehicles", "vol", "volume"):
        if c in df.columns:
            count_col = c
            break
    if count_col is None:
        # if no obvious count col, try to detect the only numeric column aside from ids
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {col for col in [id_col, u_col, v_col] if col}
        candidates = [c for c in numeric_cols if c not in exclude]
        if len(candidates) == 1:
            count_col = candidates[0]
        elif len(candidates) > 1 and "value" in candidates:
            count_col = "value"
        else:
            raise KeyError(f"Could not infer traffic count column from {df.columns}")
    
    # Basic cleanliness - no time column needed
    cols = [c for c in [id_col, u_col, v_col, count_col] if c is not None]
    df = df[cols].copy()
    rename_map = {}
    if id_col: rename_map[id_col] = "edge_id"
    if u_col: rename_map[u_col] = "u"
    if v_col: rename_map[v_col] = "v"
    rename_map[count_col] = "traffic_count"
    df.rename(columns=rename_map, inplace=True)
    return df


def attach_zones_to_counts(counts: pd.DataFrame, edge_zones: pd.DataFrame) -> pd.DataFrame:
    """Join counts with zone_o/zone_d using either (u,v) or edge ids if available."""
    merged = None
    if {"u", "v"}.issubset(counts.columns) and {"u", "v"}.issubset(edge_zones.columns):
        merged = counts.merge(edge_zones[["u", "v", "zone_o", "zone_d"]].drop_duplicates(),
                              on=["u", "v"], how="left")
    else:
        if "edge_id" in counts.columns and "edge_id" in edge_zones.columns:
            merged = counts.merge(edge_zones[["edge_id", "zone_o", "zone_d"]].drop_duplicates(),
                                  on="edge_id", how="left")
    if merged is None:
        raise RuntimeError("Unable to merge counts to zone pairs; ensure counts have (u,v) or edge_id matching edge_zones.")
    missing = merged["zone_o"].isna().sum()
    if missing:
        merged = merged.dropna(subset=["zone_o", "zone_d"]).copy()
    return merged


def exclude_random_links(annotated_counts: pd.DataFrame, exclude_proportion: float, random_seed: Optional[int] = None) -> Tuple[pd.DataFrame, int]:
    """
    Randomly exclude a proportion of unique links from the counts.
    
    Args:
        annotated_counts: DataFrame with zone annotations and traffic counts
        exclude_proportion: Proportion of links to exclude (0.0 to 1.0)
        random_seed: Optional random seed for reproducibility
    
    Returns:
        Tuple of (filtered DataFrame, number of excluded links)
    """
    if exclude_proportion <= 0.0:
        return annotated_counts, 0
    
    if exclude_proportion >= 1.0:
        log(f"WARNING: exclude_proportion >= 1.0, excluding all links!")
        return annotated_counts.iloc[0:0].copy(), len(annotated_counts)
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Identify unique links - use (u, v) if available, otherwise edge_id
    if {"u", "v"}.issubset(annotated_counts.columns):
        # Create unique link identifiers from (u, v) pairs
        annotated_counts = annotated_counts.copy()
        annotated_counts['_link_id'] = annotated_counts.apply(
            lambda row: (row['u'], row['v']), axis=1
        )
        link_id_col = '_link_id'
    elif "edge_id" in annotated_counts.columns:
        link_id_col = "edge_id"
    else:
        raise RuntimeError("Cannot identify unique links: need either (u, v) or edge_id columns")
    
    # Get unique links
    unique_links = annotated_counts[link_id_col].unique()
    n_unique_links = len(unique_links)
    n_to_exclude = int(n_unique_links * exclude_proportion)
    
    if n_to_exclude == 0:
        return annotated_counts, 0
    
    # Randomly select links to exclude
    links_to_exclude = set(random.sample(list(unique_links), n_to_exclude))
    
    # Filter out rows corresponding to excluded links
    mask = ~annotated_counts[link_id_col].isin(links_to_exclude)
    filtered = annotated_counts[mask].copy()
    
    # Remove temporary column if we created it
    if link_id_col == '_link_id':
        filtered = filtered.drop(columns=['_link_id'])
    
    return filtered, n_to_exclude


def aggregate_flows(annotated_counts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by zone_o, zone_d."""
    for c in ("zone_o", "zone_d"):
        annotated_counts[c] = annotated_counts[c].astype(int)
    out = (annotated_counts.groupby(["zone_o", "zone_d"], dropna=False)["traffic_count"]
           .sum()
           .reset_index())
    return out


# -----------------------------
# Exporters
# -----------------------------
def export_results(out_df: pd.DataFrame, counts_file: Path, output_dir: Path, split_per_slot: bool = False) -> Path:
    """Write one aggregated CSV per counts file."""
    # Derive chunk tag from filename, e.g., "-0_299" in '...traffic-counts-0_299.csv'
    m = re.search(r"traffic-counts-(.+?)\.csv$", counts_file.name)
    chunk = m.group(1) if m else "all"
    out_name = counts_file.name.replace("traffic-counts-", "traffic-counts-zone-")
    out_path = output_dir / out_name
    out_df.to_csv(out_path, index=False)
    return out_path


# -----------------------------
# Orchestration
# -----------------------------
def process_one(counts_file: Path, data_dir: Path, network_file: Path, intersections_file: Path,
                roads_file: Optional[Path], zones_file: Optional[Path], output_dir: Path,
                exclude_proportion: float = 0.0, random_seed: Optional[int] = None) -> Tuple[Path, int, int, int]:
    # Load network and zoning
    G = load_network(data_dir / network_file)
    inter_df, roads_df = load_zoning_files(data_dir / intersections_file, data_dir / roads_file if roads_file else None)
    node_to_zone = build_node_zone_mapping(inter_df)

    # Find boundary/internal edges from the graph
    edge_zones = find_boundary_links(G, node_to_zone)

    # If roads file has explicit edge IDs and (u,v), add those IDs into edge_zones to improve joins
    if roads_df is not None:
        # Try to detect useful columns
        eid = None
        for c in ("edge_id", "id", "eid", "link_id", "edge", "road_id"):
            if c in roads_df.columns:
                eid = c; break
        ucol = None
        for c in ("u", "from", "from_node", "src", "source"):
            if c in roads_df.columns:
                ucol = c; break
        vcol = None
        for c in ("v", "to", "to_node", "dst", "target"):
            if c in roads_df.columns:
                vcol = c; break
        if eid and ucol and vcol:
            roads_min = roads_df[[eid, ucol, vcol]].drop_duplicates().rename(columns={eid: "edge_id", ucol: "u", vcol: "v"})
            edge_zones = edge_zones.merge(roads_min, on=["u", "v"], how="left")

    # Load counts and attach zones
    counts_df = load_counts(data_dir / counts_file)
    
    # Calculate original total before processing
    original_total = counts_df['traffic_count'].sum()
    
    counts_with_zones = attach_zones_to_counts(counts_df, edge_zones)
    
    # Exclude random links if requested
    if exclude_proportion > 0.0:
        counts_with_zones, n_excluded = exclude_random_links(counts_with_zones, exclude_proportion, random_seed)
        log(f"  Excluded {n_excluded} unique links ({exclude_proportion*100:.1f}% of links)")
    else:
        n_excluded = 0

    # Aggregate
    aggregated = aggregate_flows(counts_with_zones)
    
    # Calculate final total
    final_total = aggregated['traffic_count'].sum()

    # Export
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_results(aggregated, data_dir / counts_file, output_dir)
    
    return out_path, original_total, final_total, n_excluded


def find_all_traffic_count_files(output_dir: Path) -> List[Path]:
    """Find all traffic count CSV files matching the pattern manhattan_*_traffic-counts-*.csv"""
    pattern = "manhattan_*_traffic-counts-*.csv"
    files = list(output_dir.glob(pattern))
    return sorted(files)

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    # Get the directory of this script
    script_dir = Path(__file__).parent
    network_path = script_dir / "Manhattan Network" / "MPLight" / "data" / "manhattan" / "roadnet_manhattan.json"
    output_dir = script_dir / "Refined Data"
    zonage_path = script_dir / "Refined Data" / "Zonage"
    output_path = script_dir / "Refined Data" / "Traffic counts"
    
    # Find all traffic count files automatically
    counts_files = find_all_traffic_count_files(output_dir)
    
    p = argparse.ArgumentParser(description="Aggregate inter-zone flows from traffic counts.")
    p.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing inputs.")
    p.add_argument("--network", type=Path, default=network_path)
    p.add_argument("--counts", type=Path, nargs="+", default=counts_files)
    p.add_argument("--intersections", type=Path, default=zonage_path / "Zonage_K20_intersections.csv")
    p.add_argument("--roads", type=Path, default=zonage_path / "Zonage_K20_roads.csv")
    p.add_argument("--zones", type=Path, default=zonage_path / "Zonage_K20_zones.csv")
    p.add_argument("--output-dir", type=Path, default = output_path)
    p.add_argument("--exclude-proportion", type=float, default=0.0, 
                   help="Proportion of links to randomly exclude (0.0 to 1.0, e.g., 0.2 for 20%%)")
    p.add_argument("--random-seed", type=int, default=None,
                   help="Random seed for reproducibility when excluding links")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    log(f"Processing {len(args.counts)} traffic count files...")
    
    if args.exclude_proportion > 0.0:
        log(f"Excluding {args.exclude_proportion*100:.1f}% of links randomly")
        if args.random_seed is not None:
            log(f"Using random seed: {args.random_seed}")
    
    # Initialize totals
    total_original = 0
    total_final = 0
    total_excluded_links = 0
    
    try:
        for i, counts_file in enumerate(args.counts, 1):
            log(f"File {i}/{len(args.counts)}: {counts_file.name}")
            out_path, original, final, n_excluded = process_one(
                counts_file=counts_file,
                data_dir=args.data_dir,
                network_file=args.network,
                intersections_file=args.intersections,
                roads_file=args.roads if args.roads else None,
                zones_file=args.zones if args.zones else None,
                output_dir=args.output_dir,
                exclude_proportion=args.exclude_proportion,
                random_seed=args.random_seed,
            )
            # Accumulate totals
            total_original += original
            total_final += final
            total_excluded_links += n_excluded
        
        # Calculate and display summary statistics
        total_difference = total_original - total_final
        total_percentage = (total_difference / total_original) * 100 if total_original > 0 else 0
        
        log(f"\n=== SUMMARY ===")
        log(f"Total original flow: {total_original}")
        log(f"Total final flow: {total_final}")
        log(f"Total difference (lost): {total_difference}")
        log(f"Total percentage lost: {total_percentage:.2f}%")
        if args.exclude_proportion > 0.0:
            log(f"Total excluded links: {total_excluded_links}")
        log(f"All {len(args.counts)} files processed successfully!")
        return 0
    except Exception as e:
        log(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
