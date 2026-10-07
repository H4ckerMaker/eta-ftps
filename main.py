import parser, features
from scapy.all import rdpcap
capture = rdpcap("captures/dump_ftps-1.pcapng")
"""
1. Extract packets 
2. Build flows from pakets
3. Extract features from flow
... do something with the features
"""
packets = parser.extract_packets(capture)

flows = parser.build_flows(packets)

print(*[features.extract_features(flow) for flow in flows.values()], sep="\n")
