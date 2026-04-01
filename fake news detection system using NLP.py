# Fake News Detection using NLP
# Author: [Your Name]
# Date: [Insert Date]
# Import Libraries
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,
classification_report
# Download stopwords (run once)
nltk.download('stopwords')
# ------------------------------
# Step 1: Load Dataset
# ------------------------------
# Example: "news.csv" should contain columns - ['title', 'text', 'label']
# label = 'FAKE' or 'REAL'
data = pd.read_csv("news.csv")
print("Dataset loaded successfully!")
print("Shape of dataset:", data.shape)
print(data.head())
# ------------------------------
# Step 2: Data Preprocessing
# ------------------------------
def clean_text(text):
text = text.lower()
text = ''.join([ch for ch in text if ch not in string.punctuation])
cas tokens = text.split()
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]
return ' '.join(tokens)
data['text'] = data['text'].apply(clean_text)
# ------------------------------
# Step 3: Split Data
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
data['text'], data['label'], test_size=0.2, random_state=42
)
# ------------------------------
# Step 4: Feature Extraction (TF-IDF)
# ------------------------------
tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
# ------------------------------
# Step 5: Model Training
# ------------------------------
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)
# ------------------------------
# Step 6: Predictions and Evaluation
# ------------------------------
y_pred = model.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# ------------------------------
# Step 7: Predict Custom News
# ------------------------------
def predict_news(news_text):
news_text = clean_text(news_text)
vectorized = tfidf_vectorizer.transform([news_text])
prediction = model.predict(vectorized)
return prediction[0]
# Example Test
sample_news = "The government has announced a new economic policy to boost
employment."
print("\nSample News Prediction:", predict_news(sample_news))
