from scapy.all import rdpcap, IP, TCP


def extract_packets(capture):
    packets = []
    for packet in capture:
        if not packet.haslayer(IP) or not packet.haslayer(TCP):
            continue
        
        ip = packet[IP]
        tcp = packet[TCP]
        packets.append({
            "timestamp" : float(packet.time),
            "src_ip" : ip.src,
            "dst_ip" : ip.dst,
            "src_port" : tcp.sport,
            "dst_port" : tcp.dport,
            "length" : len(packet),
            "payload_length" : len(tcp.payload),
            "flags": int(tcp.flags)
        })
    return packets

def build_flows(packets):

    def flow_key(packet):
        endpoint1 = (
            packet["src_ip"],
            packet["src_port"]
        )

        endpoint2 = (
            packet["dst_ip"],
            packet["dst_port"]
        )
        #sort key for bidirectional flows
        return tuple(sorted([endpoint1, endpoint2]))

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
        #set flow direction forward as client -> server
        for packet in flow["packets"]:
            syn = packet["flags"] & 0x02
            ack = packet["flags"] & 0x10

            if syn and not ack:
                initiator = (
                    packet["src_ip"],
                    packet["src_port"]
                )
                break

        for packet in flow["packets"]:
            #set direction for each packet
            src = (
                packet["src_ip"],
                packet["src_port"]
            )

            if initiator is not None:
                if src == initiator:
                    packet["direction"] = "forward"
                else:
                    packet["direction"] = "backward"

            else:
                packet["direction"] = "unknown"

    return flows
