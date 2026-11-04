#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate link-level OD CSVs (long format) into zone-level OD using a road→zone mapping.

Default directory structure:
  - OD matrices (input): script_directory/Refined Data/
  - Zonage mapping: script_directory/Refined Data/Zonage/Zonage_K20_roads.csv
  - Zone-level OD (output): script_directory/Refined Data/OD matrice/

Inputs expected:
  1) A directory with files named like: manhattan_{level}_od_matrices-{start}_{end}.csv
     Each file columns: origin_edge, destination_edge, count
  2) A road→zone CSV from the zoning step:
       columns must include: road_id, zone  (others ignored)

Outputs:
  - Zone-level OD for each input file:
      * long format: manhattan_{level}_zone-od-{start}_{end}_K{K}.csv
      * dense format: manhattan_{level}_zone-od-matrix-{start}_{end}_K{K}.csv
  - Excel summary file: Production_Attraction_Summary_K{K}.xlsx
      * Separate sheets for each traffic level
      * Production and attraction totals for each zone and time interval
      * Percentage calculations for production and attraction
      * Summary sheet with totals across all levels
"""

import os, re, glob, argparse, sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate link OD CSVs to zone OD using a road→zone map.")
    
    # Get script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    ap.add_argument("--od_dir", default=os.path.join(script_dir, "Refined Data"), 
                    help="Directory containing edge-level OD CSVs. Default: script_dir/Refined Data")
    ap.add_argument("--roads_zones", default=os.path.join(script_dir, "Refined Data", "Zonage", "Zonage_K20_roads.csv"), 
                    help="Path to road→zone CSV (columns: road_id, zone). Default: script_dir/Refined Data/Zonage/Zonage_K20_roads.csv")
    ap.add_argument("--out_dir", default=os.path.join(script_dir, "Refined Data", "OD matrice"), 
                    help="Directory to write zone-level OD outputs. Default: script_dir/Refined Data/OD matrice")
    ap.add_argument("--pattern", default="manhattan_*_od_matrices-*_*.csv",
                    help="Glob pattern (relative to --od_dir) for OD CSVs. Default matches your naming.")
    ap.add_argument("--format", choices=["long", "dense", "both"], default="both",
                    help="Output format(s) to save. Default: both.")
    ap.add_argument("--zero_diagonal", action="store_true",
                    help="If set, zero out intra-zone flows (diagonal) in outputs.")
    ap.add_argument("--zones", type=int, default=None,
                    help="K (number of zones). Optional; if omitted, inferred from road→zone map.")
    return ap.parse_args()

# ---------- Helpers ----------
OD_RE = re.compile(r"manhattan_(\d+)_od_matrices-(\d+)_(\d+)\.csv$", re.IGNORECASE)

def parse_od_filename(fname: str) -> Tuple[str,int,int]:
    """Extract (level, start, end) from filename."""
    m = OD_RE.search(os.path.basename(fname))
    if not m:
        raise ValueError(f"Unexpected OD filename: {fname}")
    return m.group(1), int(m.group(2)), int(m.group(3))

def load_road_zone_map(path: str) -> Dict[str, int]:
    """Return road_id -> zone (int)."""
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    if not {"road_id", "zone"} <= set(cols):
        raise ValueError(f"roads_zones file must have columns road_id, zone. Found: {df.columns.tolist()}")
    # coerce types
    df["road_id"] = df["road_id"].astype(str)
    # zone must be int; drop rows with NaN
    df = df.dropna(subset=["zone"])
    df["zone"] = df["zone"].astype(int)
    mp = dict(zip(df["road_id"], df["zone"]))
    return mp

def od_links_to_zones(df_od: pd.DataFrame, mp: Dict[str,int], zero_diag: bool) -> pd.DataFrame:
    """Map link-level OD long df to zone-level OD long df with columns: oz, dz, count."""
    df = df_od.copy()
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"origin_edge", "destination_edge", "count"}
    if not required <= set(df.columns):
        raise ValueError(f"OD csv must have columns {required}. Found: {df.columns.tolist()}")
    # map to zones
    df["oz"] = df["origin_edge"].astype(str).map(mp)
    df["dz"] = df["destination_edge"].astype(str).map(mp)
    before = len(df)
    df = df.dropna(subset=["oz","dz"])
    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"  [warn] dropped {dropped} rows with missing zone mapping.")
    df["oz"] = df["oz"].astype(int)
    df["dz"] = df["dz"].astype(int)
    # aggregate
    agg = df.groupby(["oz","dz"], as_index=False)["count"].sum()
    if zero_diag:
        agg = agg[agg["oz"] != agg["dz"]]
    return agg

def long_to_dense(agg_long: pd.DataFrame, K: int, zero_diag: bool) -> pd.DataFrame:
    """Return a KxK dense matrix as a DataFrame with rows,cols labeled 0..K-1."""
    zones = list(range(K))
    M = pd.DataFrame(0, index=zones, columns=zones, dtype=np.float32)
    if len(agg_long):
        M = M.add(agg_long.pivot(index="oz", columns="dz", values="count").fillna(0), fill_value=0)
    if zero_diag:
        for k in zones:
            M.iat[k, k] = 0.0
    M.index.name = "oz"
    M.columns.name = "dz"
    return M

def calculate_production_attraction(agg_long: pd.DataFrame, level: str, start: int, end: int) -> pd.DataFrame:
    """Calculate production and attraction for each zone from long format OD data."""
    if len(agg_long) == 0:
        return pd.DataFrame(columns=['Level', 'Interval', 'Zone', 'Production', 'Attraction', 'Prod_Percent', 'Attr_Percent'])
    
    # Calculate production (sum of outgoing flows) and attraction (sum of incoming flows)
    production = agg_long.groupby('oz')['count'].sum().reset_index()
    production.columns = ['Zone', 'Production']
    
    attraction = agg_long.groupby('dz')['count'].sum().reset_index()
    attraction.columns = ['Zone', 'Attraction']
    
    # Merge production and attraction
    summary = pd.merge(production, attraction, on='Zone', how='outer').fillna(0)
    
    # Calculate total production and attraction for percentage calculations
    total_production = summary['Production'].sum()
    total_attraction = summary['Attraction'].sum()
    
    # Add percentage columns
    summary['Prod_Percent'] = (summary['Production'] / total_production * 100).round(2)
    summary['Attr_Percent'] = (summary['Attraction'] / total_attraction * 100).round(2)
    
    # Add metadata columns
    summary['Level'] = level
    summary['Interval'] = f"{start}-{end}"
    
    # Reorder columns
    summary = summary[['Level', 'Interval', 'Zone', 'Production', 'Attraction', 'Prod_Percent', 'Attr_Percent']]
    
    return summary

def create_excel_summary(all_summaries: List[pd.DataFrame], output_dir: str, K: int):
    """Create Excel file with production/attraction summaries for each level."""
    excel_path = os.path.join(output_dir, f"Production_Attraction_Summary_K{K}.xlsx")
    
    # Create a new workbook
    wb = openpyxl.Workbook()
    
    # Remove default sheet
    wb.remove(wb.active)
    
    # Group summaries by level
    levels = {}
    for summary in all_summaries:
        level = summary['Level'].iloc[0] if len(summary) > 0 else 'Unknown'
        if level not in levels:
            levels[level] = []
        levels[level].append(summary)
    
    # Create a sheet for each level
    for level, summaries in levels.items():
        # Combine all summaries for this level
        combined_summary = pd.concat(summaries, ignore_index=True)
        
        # Create sheet
        ws = wb.create_sheet(title=f"Level_{level}")
        
        # Add data to sheet
        for r in dataframe_to_rows(combined_summary, index=False, header=True):
            ws.append(r)
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
    
    # Create a summary sheet with totals across all levels
    summary_ws = wb.create_sheet(title="Summary_All_Levels", index=0)
    
    # Calculate totals across all levels
    all_data = pd.concat(all_summaries, ignore_index=True)
    total_summary = all_data.groupby(['Level', 'Interval']).agg({
        'Production': 'sum',
        'Attraction': 'sum'
    }).reset_index()
    
    # Add total row
    total_row = pd.DataFrame({
        'Level': ['TOTAL'],
        'Interval': ['ALL'],
        'Production': [total_summary['Production'].sum()],
        'Attraction': [total_summary['Attraction'].sum()]
    })
    
    # Calculate percentages for total row
    total_prod = total_row['Production'].iloc[0]
    total_attr = total_row['Attraction'].iloc[0]
    total_row['Prod_Percent'] = 100.0
    total_row['Attr_Percent'] = 100.0
    
    # Add total row to summary
    total_summary = pd.concat([total_summary, total_row], ignore_index=True)
    
    # Add data to summary sheet
    for r in dataframe_to_rows(total_summary, index=False, header=True):
        summary_ws.append(r)
    
    # Auto-adjust column widths for summary sheet
    for column in summary_ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        summary_ws.column_dimensions[column_letter].width = adjusted_width
    
    # Save the workbook
    wb.save(excel_path)
    print(f"[info] Created Excel summary: {excel_path}")
    
    return excel_path

def main():
    args = parse_args()
    
    # Show which paths are being used
    print(f"[info] Using OD directory: {args.od_dir}")
    print(f"[info] Using roads-zones file: {args.roads_zones}")
    print(f"[info] Using output directory: {args.out_dir}")
    
    os.makedirs(args.out_dir, exist_ok=True)

    # Road-zone mapping
    road2zone = load_road_zone_map(args.roads_zones)
    if len(road2zone) == 0:
        print("[err] Empty road-zone mapping.")
        sys.exit(2)

    # Infer K if not provided
    if args.zones is None:
        K = max(road2zone.values()) + 1
        print(f"[info] inferred zones K={K} from road-zone map.")
    else:
        K = args.zones
        print(f"[info] using user-provided K={K}.")

    # Glob OD files
    pattern = os.path.join(args.od_dir, args.pattern)
    files = sorted(glob.glob(pattern))
    files = [f for f in files if OD_RE.search(os.path.basename(f))]
    if not files:
        print(f"[err] no OD files matched pattern: {pattern}")
        sys.exit(1)

    print(f"[info] found {len(files)} OD files.")

    # List to collect all summaries for Excel export
    all_summaries = []

    for fp in files:
        level, start, end = parse_od_filename(fp)
        print(f"[file] {os.path.basename(fp)}  (level={level}, window={start}-{end})")

        df = pd.read_csv(fp)
        agg_long = od_links_to_zones(df, road2zone, zero_diag=args.zero_diagonal)

        # Calculate production and attraction summary
        summary = calculate_production_attraction(agg_long, level, start, end)
        all_summaries.append(summary)

        # Save long format
        if args.format in ("long", "both"):
            out_long = os.path.join(
                args.out_dir,
                f"manhattan_{level}_zone-od-{start}_{end}_K{K}.csv"
            )
            agg_long.to_csv(out_long, index=False)
            print(f"   -> wrote LONG:  {out_long}")

        # Save dense format
        if args.format in ("dense", "both"):
            M = long_to_dense(agg_long, K=K, zero_diag=args.zero_diagonal)
            out_dense = os.path.join(
                args.out_dir,
                f"manhattan_{level}_zone-od-matrix-{start}_{end}_K{K}.csv"
            )
            M.to_csv(out_dense)
            print(f"   -> wrote DENSE: {out_dense}")

    # Create Excel summary file
    if all_summaries:
        excel_path = create_excel_summary(all_summaries, args.out_dir, K)
        print(f"[info] Excel summary created with {len(all_summaries)} time intervals")

    print("[done] zone aggregation complete.")

if __name__ == "__main__":
    main()
