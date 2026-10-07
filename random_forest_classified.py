import pandas as pd
import sklearn as sci
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv_parser



network_flows = []

files = [
    r"C:\Users\Fabian\Downloads\classes_csvs-3\classes_csvs-3\voip\reg\iscx_voip.raw.csv",
    r"C:\Users\Fabian\Downloads\classes_csvs-3\classes_csvs-3\chat\reg\iscx_chat.raw.csv",
    r"C:\Users\Fabian\Downloads\classes_csvs-3\classes_csvs-3\video\reg\iscx_video.raw.csv",
    r"C:\Users\Fabian\Downloads\classes_csvs-3\classes_csvs-3\file_transfer\reg\iscx_file.raw.csv"
]

network_flows = []
for file_path in files:
  network_flows.extend(csv_parser.extract_features(file_path))

df = pd.DataFrame(network_flows)



print(f'Flows: {len(df)}')
print('Distr of each type:\n', df['label'].value_counts())

X = df.drop(columns=['label']) # drops column label, 
y = df['label'] # loads only column label

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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