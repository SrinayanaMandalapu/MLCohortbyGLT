# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load Dataset
df = pd.read_csv("heart.csv")
print("Dataset shape:", df.shape)
df.head()

# EDA
print(df.info())
print(df.describe())
print(df['HeartDisease'].value_counts())

sns.countplot(x='HeartDisease', data=df)
plt.title("Target Distribution (0=No Disease, 1=Disease)")
plt.show()

# Preprocessing
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-Test Split again (since X changed)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train & Compare Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    results[name] = {
        "Accuracy": acc,
        "Time Taken (s)": round(end - start, 4),
        "Report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    print(f"\n===== {name} =====")
    print("Accuracy:", acc)
    print("Time Taken:", round(end-start, 4), "seconds")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Results Summary
summary = pd.DataFrame(results).T[["Accuracy", "Time Taken (s)"]]
print("\n=== Model Comparison Summary ===")
print(summary)

# Plot accuracy comparison
summary["Accuracy"].plot(kind="bar", color="skyblue", figsize=(8,5))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()

# Plot time comparison
summary["Time Taken (s)"].plot(kind="bar", color="salmon", figsize=(8,5))
plt.title("Model Training Time Comparison")
plt.ylabel("Seconds")
plt.xticks(rotation=45)
plt.show()
