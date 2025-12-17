import pandas as pd
import re                          # For cleaning text (regex)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import pickle                      # To save/load model and vectorizer

df = pd.read_csv("Reviews.csv")

# 2. Keep only required columns
df = df[['Text', 'Score']]
df = df.dropna()


# 3. Create sentiment labels from Score
df = df[df['Score'] != 3]   # remove neutral reviews
df['Sentiment'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)

print(df.head())


# 4. Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df['Text'] = df['Text'].apply(clean_text)


# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['Text'], df['Sentiment'], test_size=0.2, random_state=42
)


# 6. TF-IDF Vectorization(Term Frequency-Inverse Document Frequency)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Train Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Model Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 9. Save model and vectorizer
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("✅ Model training completed and files saved successfully")