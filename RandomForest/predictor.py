import joblib
import pandas as pd
from scapy.all import rdpcap
import parser.pcap_parser as parser

def predict_pcap(file_path):
    print("Load model and classifiers")
    classifier = joblib.load('rf_model.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
    print(f"parse {file_path}...")
    capture = rdpcap(file_path)
    packets = parser.extract_packets(capture)
    flows = parser.build_flows(packets)
    
    if not flows:
        print("eroor")
        return 1

    #Extract features for each flow
    network_flows = []
    for flow in flows.values():
        features = parser.extract_features(flow)
        network_flows.append(features)

    df = pd.DataFrame(network_flows)
    df = df.reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(df)
    predictions = classifier.predict(X_scaled)
    
    print("\nPredictions")
    df['Predicted_Label'] = predictions
    
    for i, pred in enumerate(predictions):
        print(f"Flow {i+1}: {pred}")
    
    print("\nThis capture flows")
    print(df['Predicted_Label'].value_counts())

    df['Predicted_Label'] = predictions
    majority_label = df['Predicted_Label'].mode()[0]
    print(f"Predicted activity for this capture: {majority_label}")
    
if __name__ == "__main__":
    test_file = r"C:\Users\Fabian\Documents\Uni\Master\1. Semester\Period 1\Cybersecurity Overview\Cybersecurity Overview Project\eta-ftps\RandomForest\file_transfer_ftps_01.pcapng"
    predict_pcap(test_file)