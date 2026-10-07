import statistics

def inter_arrival_times(packets):
    return [
        packets[i]["timestamp"] - packets[i - 1]["timestamp"]
        for i in range(1, len(packets))
    ]

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
        
        "forward_mean_packet_size" : statistics.mean([p["length"] for p in forward]),
        "backward_mean_packet_size" : statistics.mean([p["length"] for p in backward]),

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
