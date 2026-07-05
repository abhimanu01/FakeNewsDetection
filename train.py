import os
print(os.listdir())

import pandas as pd
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 📂 Load dataset (Indian dataset)
df = pd.read_csv("news_dataset.csv")

# 🔍 Check data (optional)
print(df.head())
# 🧹 Clean text function
def clean_text(text):
    text = str(text).lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return text

# Apply cleaning
df['text'] = df['text'].apply(clean_text)

# 🎯 Features & Labels
X = df['text']
y = df['label']

# 🔢 Convert text to numbers (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

# ✂️ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🤖 Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 📊 Evaluate
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 💾 Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved successfully!")