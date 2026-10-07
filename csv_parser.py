import csv
import numpy as np

def extract_features(filepath):
    with open(filepath,"r", newline="") as f:
        reader = csv.reader(f)

        extracted_feature = []

        for row in reader:
            start_time = row[6]
            packet_count = row[7]
            timestamps = row[8:8+int(row[7])]
            packet_sizes = row[9+int(row[7]):]
            flow_start_time = float(start_time)
            flow_packet_count = int(packet_count)
            flow_timestamps = [float(x) for x in timestamps]
            flow_packet_sizes = [int(x) for x in packet_sizes]
            flow_mean_packet_size = np.mean(flow_packet_sizes)
            flow_min_packet_size = np.min(flow_packet_sizes)
            flow_max_packet_size = np.max(flow_packet_sizes)
            flow_total_bytes = sum(flow_packet_sizes)
            flow_duration = flow_timestamps[-1]
            flow_interarrival_times = np.diff(flow_timestamps[1:], prepend=flow_timestamps[0])
            flow_min_interarrival_time = np.min(flow_interarrival_times)
            flow_max_interarrival_time = np.max(flow_interarrival_times)
            flow_mean_interarrival_time = np.mean(flow_interarrival_times)
            flow_bytes_per_sec = flow_total_bytes / flow_duration
            flow_package_per_sec = flow_packet_count / flow_duration
            label = ""
            if 'ftps' in row[0]:
                label = "FTPS"
            elif 'chat' in row[0]:
                label = "CHAT"
            elif 'video' in row[0]:
                label = 'VIDEO'
            elif 'browsing' in row[0]:
                label = 'BROWSING'
            elif 'audio' in row[0]:
                label = 'AUDIO'
            else: label = "OTHER"
            
            flow_feature = {
                    "packet_count": flow_packet_count,
                    "total_bytes": flow_total_bytes,
                    "mean_packet_size": flow_mean_packet_size,
                    "duration": flow_duration,
                    "min_size": flow_min_packet_size,
                    "max_size": flow_max_packet_size,
                    "min_interarrival_time": flow_min_interarrival_time,
                    "max_interarrival_time": flow_max_interarrival_time,
                    "mean_interarrival_time": flow_mean_interarrival_time,
                    "bytes_per_sec": flow_bytes_per_sec,
                    "package_per_sec": flow_package_per_sec,
                    "label": label
            }
                
            extracted_feature.append(flow_feature)
        return extracted_feature