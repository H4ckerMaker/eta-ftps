#!/usr/bin/env python3
"""Classify a PCAP file using a trained model.
Usage:
  python predict.py cattura.pcap --ckpt best_ode_n20.pt --classes BROWSING,CHAT,FILE_TRANSFER,STREAMING
"""
import argparse
import collections
import csv
import sys
import numpy as np
import torch
from models import LSTMBaseline, ODELSTM
from data import build_blocks

def flows_from_pcap(path):
    from pcap_to_csv import build_flows, read_packets
    return build_flows(read_packets(path))


def flows_from_csv(path):
    from parser.csv_parser import extract_features
    out = {}
    for i, f in enumerate(extract_features(path, label="?")):
        t = [float(x) for x in f["times"]]
        s = [float(x) for x in f["sizes"]]
        out[("CSV", f"flow#{i}", "", "", "")] = list(zip(t, s))
    return out


def flows_to_blocks(flows, n, min_packets):
    """
    Extracting blocks from each flow
    """
    items = []
    for (proto, src, sp, dst, dp), pk in flows.items():
        if len(pk) < max(min_packets, n):
            continue
        pk = sorted(pk, key=lambda x: x[0])
        t0 = pk[0][0]
        flow = {"times": [t - t0 for t, _ in pk], "sizes": [length for _, length in pk], "label": 0}
        X, _ = build_blocks([flow], n)
        name = f"{proto} {src}:{sp} -> {dst}:{dp}" if proto != "CSV" else src
        items.append({"flow": name, "packets": len(pk), "duration": pk[-1][0] - t0, "X": X})
    return items


def load_model(ckpt_path, classes_arg, n_arg, time_scale_arg):
    ck = torch.load(ckpt_path, map_location="cpu")
    meta = ck if "state_dict" in ck else {}
    sd = ck["state_dict"] if meta else ck
    n_classes = sd["head.fc.4.weight"].shape[0]
    classes = meta.get("classes") or (classes_arg.split(",") if classes_arg else None)
    if classes is None:
        sys.exit(f"Checkpoint has {n_classes} classies but it doesn't contains the names: pass --classes "
                 "with train.py ordered classes (alfabetic).")
    if len(classes) != n_classes:
        sys.exit(f"Checkpoint has {n_classes} classes, ma you indicated just {len(classes)}: {classes}")
    n = meta.get("n") or n_arg
    ts = meta.get("time_scale", time_scale_arg)
    if any(k.startswith("encoder.") for k in sd):
        model, kind = ODELSTM(n_classes, time_scale=ts), "ode"
    else:
        feats = "size" if sd["head.lstm.weight_ih_l0"].shape[1] == 1 else "both"
        model, kind = LSTMBaseline(n_classes, feats), f"lstm_{feats}"
    model.load_state_dict(sd)
    model.eval()
    return model, classes, kind, n


def predict_probs(model, X):
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X)), dim=1).numpy()


def summarize(items, probs, classes):
    rows = []
    for it, P in zip(items, probs):
        mean = P.mean(axis=0) 
        k = int(mean.argmax())
        rows.append({"flow": it["flow"], "packets": it["packets"], "blocks": len(P),
                     "duration_s": round(it["duration"], 3), "pred": classes[k],
                     "confidence": round(float(mean[k]), 3)})
    return rows

def print_report(rows, classes, top):
    if not rows:
        print("No flows with enough packets: lower --min-packets or check capture.")
        return
    rows = sorted(rows, key=lambda r: r["packets"], reverse=True)
    print(f"\n{'flow':58s} {'pkt':>6s} {'blocks':>7s} {'duration s':>9s}  {'class':14s} conf")
    for r in rows[:top]:
        print(f"{r['flow'][:58]:58s} {r['packets']:6d} {r['blocks']:7d} {r['duration_s']:9.2f}  "
              f"{r['pred']:14s} {r['confidence']:.2f}")
    if len(rows) > top:
        print(f"... others {len(rows) - top} flow (use --top or --csv to see them)")
    tot_blocks = sum(r["blocks"] for r in rows)
    by_blocks, by_flows = collections.Counter(), collections.Counter()
    for r in rows:
        by_blocks[r["pred"]] += r["blocks"]
        by_flows[r["pred"]] += 1
    print("\nSummary (percentage of blocks):")
    for c in classes:
        print(f"  {c:14s} {100 * by_blocks[c] / tot_blocks:5.1f}%   ({by_flows[c]} flow)")
    print(f"\nMost common category: {by_blocks.most_common(1)[0][0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="file .pcap/.pcapng or .csv my_*.raw.csv format")
    ap.add_argument("--ckpt", required=True, help="checkpoint, es. best_ode_n20.pt")
    ap.add_argument("--classes", help="classes names separated by commas")
    ap.add_argument("--n", type=int, default=20, help="packets for each block (default 20)")
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--min-packets", type=int, default=0, help="drop shorter blocks (minimo effettivo: n)")
    ap.add_argument("--top", type=int, default=15, help="how many flows to show (the longer ones)")
    ap.add_argument("--csv", help="Output CSV file")
    a = ap.parse_args()

    model, classes, kind, n = load_model(a.ckpt, a.classes, a.n, a.time_scale)
    print(f"model: {kind}, classes: {classes}, n={n}")

    flows = flows_from_csv(a.input) if a.input.lower().endswith(".csv") else flows_from_pcap(a.input)
    items = flows_to_blocks(flows, n, a.min_packets)
    print(f"{len(flows)} total flows, {len(items)} with at least {max(a.min_packets, n)} packets")
    rows = summarize(items, [predict_probs(model, it["X"]) for it in items], classes)
    print_report(rows, classes, a.top)

    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["packets"], reverse=True))
        print(f"results saved in {a.csv}")


if __name__ == "__main__":
    main()