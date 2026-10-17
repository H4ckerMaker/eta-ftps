#!/usr/bin/env python3
"""
Reads PCAP from a directory (es. dataset/CHAT) and convert it into a CSV that csv_parser_v2.py can read.
Usage:
    python3 pcap_to_csv.py dataset/CHAT --name my_chat
    -> classes_csv/chat/reg/my_chat.raw.csv
"""
import argparse
import csv
import glob
import os
import sys
from scapy.all import IP, IPv6, TCP, UDP, PcapReader

def read_packets(path):
    """
    Reads packet list from the PCAP file and filters for TCP or UDP with IP or IPv6 packets
    """
    packets = []
    with PcapReader(path) as pcap:
        for pkt in pcap:
            if pkt.haslayer(IP):
                ip = pkt[IP]
                length = ip.len or len(pkt) 
            elif pkt.haslayer(IPv6):
                ip = pkt[IPv6]
                length = ip.plen + 40
            else:
                continue
            if pkt.haslayer(TCP):
                l4, proto = pkt[TCP], "TCP"
            elif pkt.haslayer(UDP):  
                l4, proto = pkt[UDP], "UDP"
            else:
                continue
            packets.append((float(pkt.time), proto, ip.src, l4.sport, ip.dst, l4.dport, int(length)))
    return packets

def build_flows(packets):
    """
    Build each flow from a list of packets.
    A flow is a 5-tuple object.
    """
    flows = {}
    for t, proto, src, sp, dst, dp, length in packets:
        flows.setdefault((proto, src, sp, dst, dp), []).append((t, length))
    for pk in flows.values():
        pk.sort(key=lambda x: x[0])
    return flows

def flow_rows(pcap_name, flows, min_packets):
    """
    Reads each flow and build the CSV row for it
    """
    rows = []
    for (proto, src, sp, dst, dp), pk in flows.items():
        if len(pk) < min_packets:
            continue
        t0 = pk[0][0]
        times = [f"{t - t0:.6f}" for t, _ in pk]
        sizes = [str(length) for _, length in pk]
        n = len(pk)
        rows.append([pcap_name, src, sp, dst, dp, proto, f"{t0:.6f}", n, *times, n, *sizes])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="input directory from which you can read the PCAP file for one class")
    ap.add_argument("--name", required=True, help="name of the output file, es. my_chat")
    ap.add_argument("--min-packets", type=int, default=20, help="drop flows with less than given min-packets packets (default 20)")
    ap.add_argument("--out-root", default="classes_csvs")
    ap.add_argument("--force", action="store_true", help="overwrite the output file if already exist")
    a = ap.parse_args()

    cls = os.path.basename(os.path.normpath(a.input_dir)).lower()  
    out = os.path.join(a.out_root, cls, "reg", f"{a.name}.raw.csv")
    if os.path.exists(out) and not a.force:
        sys.exit(f"{out} already exist")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    pcaps = sorted(glob.glob(os.path.join(a.input_dir, "*.pcap")) + glob.glob(os.path.join(a.input_dir, "*.pcapng")))
    if not pcaps:
        sys.exit(f"no pcap files in {a.input_dir}")

    total = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        for p in pcaps:
            packets = read_packets(p)
            flows = build_flows(packets)
            rows = flow_rows(os.path.basename(p), flows, a.min_packets)
            w.writerows(rows)
            total += len(rows)
            print(f"{os.path.basename(p)}: {len(packets)} packets, {len(flows)} flow, {len(rows)} kept")
    print(f"-> {out}: {total} flow (class {cls.upper()})")


if __name__ == "__main__":
    main()