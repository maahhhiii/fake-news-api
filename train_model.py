import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Get project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load datasets
fake_df = pd.read_csv(os.path.join(BASE_DIR, "Fake.csv"))
true_df = pd.read_csv(os.path.join(BASE_DIR, "True.csv"))

# Labels
fake_df["label"] = 0   # Fake
true_df["label"] = 1   # Real

# Combine title and text
fake_df["content"] = fake_df["title"] + " " + fake_df["text"]
true_df["content"] = true_df["title"] + " " + true_df["text"]

# Merge datasets
data = pd.concat([fake_df, true_df], axis=0)

# Shuffle rows
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and target
X = data["content"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save files
joblib.dump(model, os.path.join(BASE_DIR, "model.pkl"))
joblib.dump(vectorizer, os.path.join(BASE_DIR, "vectorizer.pkl"))

print("model.pkl saved successfully")
print("vectorizer.pkl saved successfully")