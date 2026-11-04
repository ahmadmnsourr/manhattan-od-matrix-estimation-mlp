import json
import math
from collections import Counter
from pathlib import Path
import os
import csv

base = Path(__file__).resolve().parent  # folder where script is
data_dir = base / "Manhattan Network" / "MPLight" / "data" / "manhattan"
json_files = [
    "manhattan_7846.json",
    "manhattan_11756.json", 
    "manhattan_31217.json", 
    "manhattan_21430.json"
]

def load_json_any (path : Path):
    text = path.read_text(encoding="utf-8")
    try: 
        # Try normal JSON first (most Manhattan files are plain JSON)
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: very light JSON5 tolerance for trailing commas/comments
        # Remove // comments and /* */ comments and try again
        import re
        text2 = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
        text2 = re.sub(r"/\*.*?\*/", "", text2, flags=re.DOTALL)
        return json.loads(text2)

def emissions_in_window(start, end, interval, q_start, q_end):
    """
    Count how many vehicles depart between q_start and q_end (inclusive).
    Based on Manhattan-style flow definitions.
    All times are in seconds.
    """
    
    # Case 1: One-off vehicle (start == end)
    if end == start:
        return 1 if q_start <= start <= q_end else 0

    # Case 2: Repeated departures
    if interval is None or interval <= 0:
        # Treat as single departure at start
        return 1 if q_start <= start <= q_end else 0
    
    # Overlap check
    left = max(q_start, start)
    right = min(q_end, end)
    if right < left:
        return 0
    
    k_first = math.ceil((left - start) / interval)
    k_last = math.floor((right - start) / interval)
    
    if k_first < 0:
        k_first = 0

    n = max(0, k_last - k_first + 1)
    return n


import csv
from pathlib import Path
from collections import Counter

# assume: file_path (str or Path), base (Path), load_json_any, emissions_in_window already defined

def process_file(file_path, file_name):
    """Process a single JSON file and generate traffic counts for 5-minute intervals"""
    print(f"\n=== Processing {file_name} ===")
    data = load_json_any(Path(file_path))

    # First pass: determine time range from data
    earliest_start = float("inf")
    latest_end = float("-inf")
    unique_edges = set()

    for item in data:
        start = float(item.get("startTime", 0))
        end = float(item.get("endTime", start))
        route = item.get("route", [])
        
        if start < earliest_start:
            earliest_start = start
        if end > latest_end:
            latest_end = end
        unique_edges.update(route)

    print(f"Data time range: {earliest_start:.0f} to {latest_end:.0f} seconds")
    
    # Generate 5-minute intervals (300 seconds each)
    interval_duration = 300  # 5 minutes in seconds
    start_time = int(earliest_start // interval_duration) * interval_duration
    end_time = int(latest_end // interval_duration + 1) * interval_duration
    
    print(f"Processing intervals from {start_time} to {end_time} seconds")
    
    # Process each 5-minute interval
    for idx, q_start in enumerate(range(start_time, end_time, interval_duration)):
        q_end = q_start + interval_duration - 1  # 0-299, 300-599, etc.
        
        counts = Counter()
        total_entries = 0
        total_emissions = 0
        actual_last_second = q_start  # Initialize with start time

        for item in data:
            route = item.get("route", [])
            start = float(item.get("startTime", 0))
            end = float(item.get("endTime", start))
            interval = item.get("interval", 0)
            try:
                interval = float(interval)
            except (TypeError, ValueError):
                interval = 0.0

            # Count emissions within the current 5-minute window
            n_dep = emissions_in_window(start, end, interval, q_start, q_end)
            if n_dep <= 0 or not route:
                continue

            # Track the actual last second of data in this interval
            if start <= q_end and end >= q_start:
                # This item contributes to this interval
                actual_last_second = max(actual_last_second, min(end, q_end))

            for edge_id in route:
                counts[edge_id] += n_dep
            total_entries += 1
            total_emissions += n_dep

        # Process results for this interval
        rows = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

        # Save CSV for this interval
        output_dir = base / "Refined Data"
        output_dir.mkdir(exist_ok=True)
        # Extract base name without extension for file naming
        base_name = file_name.replace('.json', '')
        # Use fixed-naming buckets like 0_299, 300_599, ... independent of actual data end
        name_start = idx * interval_duration
        name_end = name_start + interval_duration - 1
        interval_name = f"{name_start}_{name_end}"
        out_file = output_dir / f"{base_name}_traffic-counts-{interval_name}.csv"

        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["edge_id", "traffic_counts"])
            for edge_id, cnt in rows:
                writer.writerow([edge_id, cnt])

        print(f"Results for interval {interval_name} saved to {out_file}")

        # Summary for this interval
        print(f"  Interval bucket [{interval_name}] seconds:")
        print(f"    Entries contributing: {total_entries}")
        print(f"    Total departures counted: {total_emissions}")
        print(f"    Unique edges with traffic: {len(rows)}")

    # Overall summary for this file
    print(f"\nOverall Summary for {file_name}:")
    print(f" Total unique edges in file: {len(unique_edges)}")
    print(f" Data time range: {earliest_start:.0f} to {latest_end:.0f} seconds")
    print(f" Processed {(end_time - start_time) // interval_duration} intervals")


def main():
    """Main function to process all JSON files"""
    print("Starting traffic count analysis for all Manhattan files...")
    
    for json_file in json_files:
        file_path = data_dir / json_file
        if file_path.exists():
            process_file(file_path, json_file)
        else:
            print(f"Warning: File {json_file} not found at {file_path}")
    
    print("\n=== All files processed ===")

        
if __name__ == "__main__":
    main()