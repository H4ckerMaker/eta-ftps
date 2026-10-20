import pandas as pd
import sklearn as sci
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import parser.pcap_parser as parser
from scapy.all import rdpcap
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import joblib

network_flows = []

dataset_dir = Path(r"C:\Users\Fabian\Documents\Uni\Master\1. Semester\Period 1\Cybersecurity Overview\Cybersecurity Overview Project\eta-ftps\ODENet (AI)\dataset")
files = list(dataset_dir.rglob("*.pcap")) + list(dataset_dir.rglob("*.pcapng"))
#print(files)
print(len(files))
network_flows = []

for file_path in files:
    filename = file_path.name.lower()
    label = ""
    if "browse" in filename:
        label = "BROWSING"
    if "chat" in filename:
        label = "CHAT"
    if "file" in filename:
        label = "FILE TRANSFER"
    if "streaming" in filename:
        label = "STREAMING"
        
    try:
        capture = rdpcap(str(file_path))
    except Exception as e:
        print(f"Skipping {file_path.name} - Not a valid capture file. Error: {e}")
        continue
    packets = parser.extract_packets(capture)
    flows = parser.build_flows(packets)
    for flow in flows.values():
        features = parser.extract_features(flow)
        features["label"] = label 
        features["filename"] = filename
        network_flows.append(features)

df = pd.DataFrame(network_flows)



print(f'Flows: {len(df)}')
print('Distr of each type:\n', df['label'].value_counts())

X = df.drop(columns=['label', 'filename']) 
y = df['label']
groups = df['filename'] 

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(
    n_estimators=120, 
    max_features='sqrt',
    random_state=42,
    max_depth=25,
    n_jobs=-1, 
)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy")
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print("Classification report")
print(classification_report(y_test, y_pred, digits=4))

print('confusion matrix and importances')
labels = sorted(y.unique())
confusion_matr = confusion_matrix(y_test, y_pred, labels=labels)
print(pd.DataFrame(confusion_matr, index=labels, columns=labels))

importances = pd.Series(classifier.feature_importances_, index=X.columns).sort_values(
    ascending=False
)
for feat, val in importances.items():
  print(f'{feat:20s}: {val:.4f} ({val * 100:.1f}%)')
  
print("save model")
joblib.dump(classifier, 'rf_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(X.columns.tolist(), 'feature_names.joblib')