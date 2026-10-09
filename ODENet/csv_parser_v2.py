import csv
import numpy as np

def extract_features(filepath):
    with open(filepath,"r", newline="") as f:
        reader = csv.reader(f)

        extracted_feature = []

        for row in reader:
            timestamps = row[8:8+int(row[7])]
            packet_sizes = row[9+int(row[7]):]
            label = ""
            if 'ftps' in row[0].lower():
                label = "FTPS"
            elif 'chat' in row[0].lower():
                label = "CHAT"
            elif 'video' in row[0].lower():
                label = 'VIDEO'
            elif 'browsing' in row[0].lower():   
                label = 'BROWSING'
            elif 'audio' in row[0].lower():
                label = 'AUDIO'
            else: label = "OTHER"
            flow_feature = {
                "times": timestamps,
                "sizes": packet_sizes,
                "label": label
            }
            extracted_feature.append(flow_feature)
        return extracted_feature
