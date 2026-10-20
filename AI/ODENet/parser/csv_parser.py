import csv
import os

def class_from_path(filepath):
    return os.path.basename(os.path.dirname(os.path.dirname(filepath))).upper()

def extract_features(filepath, label=None):
    label = label or class_from_path(filepath)
    flows, skipped = [], 0
    with open(filepath, "r", newline="") as f:
        for row in csv.reader(f):
            try:
                n = int(row[7])  
            except (IndexError, ValueError):
                skipped += 1
                continue
            times = row[8:8 + n]
            sizes = [x for x in row[9 + n:] if x != ""]  
            if len(times) != n or len(sizes) != n:
                skipped += 1  
                continue
            flows.append({"times": times, "sizes": sizes, "label": label})
    msg = f"[parser] {os.path.basename(filepath)} -> {label}: {len(flows)} flow"
    if skipped:
        msg += f", {skipped} righe scartate (formato inatteso)"
    print(msg)
    return flows