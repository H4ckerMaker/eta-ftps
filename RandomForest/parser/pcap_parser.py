from scapy.all import rdpcap, IP, TCP, UDP
import statistics


# Return all inter_arrival_times for packets in flow
def inter_arrival_times(packets):
    return [
        packets[i]["timestamp"] - packets[i - 1]["timestamp"]
        for i in range(1, len(packets))
    ]

#(src_ip:src_port, dst_ip:dst_port) for a packet
def flow_key(packet):
    endpoint1 = (
        packet["src_ip"],
        packet["src_port"]
    )
    endpoint2 = (
        packet["dst_ip"],
        packet["dst_port"]
    )
    return tuple(sorted([endpoint1, endpoint2]))

# return features for a flow
def extract_features(flow):

    packets = flow["packets"]
    lengths = [p["length"] for p in packets]
    iats = inter_arrival_times(packets)
    forward = [
        p for p in packets if p["direction"] == "forward"
    ]
    backward = [
        p for p in packets if p["direction"] == "backward"
    ]
    features = {
        "packet_count": len(packets),
        "total_bytes": sum(lengths),
        "forward_packet_count" : len(forward),
        "backward_packet_count" : len(backward),
        "forward_bytes" : sum(p["length"] for p in forward),
        "backward_bytes" : sum(p["length"] for p in backward),
        "forward_mean_packet_size" : statistics.mean([p["length"] for p in forward]) if forward else 0,
        "backward_mean_packet_size" : statistics.mean([p["length"] for p in backward]) if backward else 0,
        "mean_packet_length": statistics.mean(lengths),
        "std_packet_length": (
            statistics.stdev(lengths)
            if len(lengths) > 1 else 0
        ),
        "flow_duration": (
            packets[-1]["timestamp"] -
            packets[0]["timestamp"]
        ),
        "mean_iat": statistics.mean(iats) if iats else 0,
        "std_iat": (
            statistics.stdev(iats)
            if len(iats) > 1 else 0
        ),
        "min_iat": min(iats) if iats else 0,
        "max_iat": max(iats) if iats else 0,
    }
    return features

# Return the list of packets that have a TCP/IP or UDP/IP protocol combo
def extract_packets(capture):
    packets = []
    for packet in capture:
        if not packet.haslayer(IP):
            continue
            
        if packet.haslayer(TCP):
            layer = packet[TCP]
            flags = int(layer.flags)
            proto = "TCP"
        elif packet.haslayer(UDP):
            layer = packet[UDP]
            flags = 0  # UDP does not have flags
            proto = "UDP"
        else:
            continue  # Skip ICMP, ARP, etc.

        ip = packet[IP]
        packets.append({
            "timestamp" : float(packet.time),
            "src_ip" : ip.src,
            "dst_ip" : ip.dst,
            "src_port" : layer.sport,
            "dst_port" : layer.dport,
            "length" : len(packet),
            "payload_length" : len(layer.payload),
            "flags": flags,
            "protocol": proto
        })
    return packets

#return single flows for packets
def build_flows(packets):
    flows = {}
    for packet in packets:
        key = flow_key(packet)
        if key not in flows:
            flows[key] = {
                "endpoint_a": key[0],
                "endpoint_b": key[1],
                "packets": []
            }
        flows[key]["packets"].append(packet)
        
    for flow in flows.values():
        initiator = None
        for packet in flow["packets"]:
            if packet["protocol"] == "TCP":
                syn = packet["flags"] & 0x02
                ack = packet["flags"] & 0x10
                if syn and not ack:
                    initiator = (packet["src_ip"], packet["src_port"])
                    break
            elif packet["protocol"] == "UDP":
                initiator = (packet["src_ip"], packet["src_port"])
                break
                
        if initiator is None and len(flow["packets"]) > 0:
            initiator = (flow["packets"][0]["src_ip"], flow["packets"][0]["src_port"])

        for packet in flow["packets"]:
            src = (packet["src_ip"], packet["src_port"])
            if initiator is not None:
                if src == initiator:
                    packet["direction"] = "forward"
                else:
                    packet["direction"] = "backward"
            else:
                packet["direction"] = "unknown"
                
    return flows