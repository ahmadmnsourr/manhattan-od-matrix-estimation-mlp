import json, math, csv, re
from collections import Counter
from pathlib import Path
import os

def load_json_any(path: Path):
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text2 = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
        text2 = re.sub(r"/\*.*?\*/", "", text2, flags=re.DOTALL)
        return json.loads(text2)
    
def emissions_in_window(start, end, interval, q_start, q_end):
    """
    Manhattan demand semantics:
      - If end == start      -> single departure at 'start'
      - If end > start       -> repeated departures every 'interval' seconds
      - All times in seconds
    Returns the count of departures in [q_start, q_end] (inclusive)
    """
    # Single departure
    if end == start or interval is None or interval <= 0:
        return 1 if (q_start <= start <= q_end) else 0

    # Repeated departures: start, start+interval, ..., <= end
    left = max(q_start, start)
    right = min(q_end, end)
    if right < left:
        return 0

    k_first = math.ceil((left - start) / interval)
    k_last  = math.floor((right - start) / interval)
    if k_first < 0:
        k_first = 0

    return max(0, k_last - k_first + 1)

def process_manhattan_file(file_path, output_dir):
    """Process a single Manhattan JSON file and generate OD matrices for each 5-minute interval"""
    print(f"\n{'='*60}")
    print(f"Processing file: {file_path.name}")
    print(f"{'='*60}")
    
    data = load_json_any(file_path)

    # Find the time range in the data
    earliest_start = float("inf")
    latest_end = float("-inf")
    
    for item in data:
        start = float(item.get("startTime", 0))
        end = float(item.get("endTime", start))
        if start < earliest_start:
            earliest_start = start
        if end > latest_end:
            latest_end = end

    print(f"Data time range: {earliest_start:.0f} to {latest_end:.0f} seconds")
    
    # Calculate number of 5-minute intervals (300 seconds each)
    interval_duration = 300  # 5 minutes in seconds
    total_duration = latest_end - earliest_start
    num_intervals = math.ceil(total_duration / interval_duration)
    
    print(f"Processing {num_intervals} intervals of 5 minutes each")
    
    # Get the base filename without extension for naming
    base_filename = file_path.stem  # e.g., "manhattan_21430"
    
    # Process each 5-minute interval
    for i in range(num_intervals):
        q_start = earliest_start + (i * interval_duration)
        q_end = min(q_start + interval_duration - 1, latest_end)  # -1 to make it inclusive
        
        print(f"\nProcessing interval {i+1}/{num_intervals}: {q_start:.0f}-{q_end:.0f} seconds")
        
        od_counts = Counter()        # (origin_edge, destination_edge) -> count
        unique_edges = set()         # any edge that appears in any route
        total_departures = 0         # total # of vehicle departures within window

        for item in data:
            route = item.get("route", [])
            if not route:
                continue

            # Track edges seen for the summary
            unique_edges.update(route)

            # Times
            start = float(item.get("startTime", 0))
            end   = float(item.get("endTime", start))
            interval = item.get("interval", 0)
            try:
                interval = float(interval)
            except (TypeError, ValueError):
                interval = 0.0

            # Count departures in query window
            n_dep = emissions_in_window(start, end, interval, q_start, q_end)
            if n_dep <= 0:
                continue

            origin = route[0]
            destination = route[-1]
            od_counts[(origin, destination)] += n_dep
            total_departures += n_dep

        # ---- Write sparse OD matrix to CSV for this interval ----
        # Use fixed-naming buckets like 0_299, 300_599, ...
        name_start = i * interval_duration
        name_end = name_start + interval_duration - 1
        out_csv = output_dir / f"{base_filename}_od_matrices-{name_start}_{name_end}.csv"

        # Sorted (largest counts first, then lexicographically)
        rows = sorted(od_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["origin_edge", "destination_edge", "count"])
            for (o, d), c in rows:
                w.writerow([o, d, c])

        # ---- Summary for this interval ----
        print(f"  Saved OD counts to: {out_csv.name}")
        print(f"  Unique edges: {len(unique_edges)}, OD pairs: {len(rows)}, Departures: {total_departures}")
    
    print(f"\nCompleted processing {num_intervals} intervals for {file_path.name}")

def main():
    # ---- Resolve default file next to this script (edit if you prefer) ----
    base = Path(__file__).resolve().parent
    manhattan_dir = base / "Manhattan Network" / "MPLight" / "data" / "manhattan"
    
    # List of Manhattan files to process
    manhattan_files = [
        "manhattan_21430.json",
        "manhattan_7846.json", 
        "manhattan_11756.json",
        "manhattan_31217.json"
    ]
    
    # Create output directory
    output_dir = base / "Refined Data"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Processing {len(manhattan_files)} Manhattan files...")
    print(f"Output directory: {output_dir}")
    
    # Process each Manhattan file
    for filename in manhattan_files:
        file_path = manhattan_dir / filename
        if file_path.exists():
            process_manhattan_file(file_path, output_dir)
        else:
            print(f"Warning: File not found: {file_path}")
    
    print(f"\n{'='*60}")
    print("All Manhattan files processed successfully!")
    print(f"All output files saved to: {output_dir}")
    print(f"{'='*60}")
        
if __name__ == "__main__":
    main()
