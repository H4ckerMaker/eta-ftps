import pandas as pd
import sklearn as sci
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

network_flows = []

df = pd.DataFrame(network_flows)

print(f'Geladene Flows: {len(df)}')
print('Verteilung:\n', df['label'].value_counts())

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
conf_matr = confusion_matrix(y_test, y_pred, labels=labels)
print(pd.DataFrame(confusion_matrix, index=labels, columns=labels))

importances = pd.Series(classifier.feature_importances_, index=X.columns).sort_values(
    ascending=False
)
for feat, val in importances.items():
  print(f'{feat:20s}: {val:.4f} ({val * 100:.1f}%)')