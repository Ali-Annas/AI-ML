import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Ensure nltk stopwords are downloaded
import nltk
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('../data/email.csv')

# Inspect the dataframe
print(df.head(20))
print(df.columns)
print(df.shape)


# Pre-process the data
def clean_text(text):
    text = text.lower(
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text

# Use the 'Message' column for text data (previously 'message')
df['Message'] = df['Message'].apply(clean_text)
print("\nSample of cleaned text:")
print(df['Message'].head())

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Message'])
print("\nShape of feature matrix X:")
print(X.shape)

# Map the labels in the 'Category' column to binary values (previously 'file')
y = df['Category'].map({'ham': 1, 'spam': 0})
print("\nSample of label mapping:")
print(y.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nShape of training and testing sets:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# Train Naive Bayes Model
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_predictions = nb.predict(X_test)

# Train Decision Tree Model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_test)

# Evaluate Accuracy
nb_accuracy = accuracy_score(y_test, nb_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f'\nNaive Bayes Accuracy: {nb_accuracy}')
print(f'Decision Tree Accuracy: {dt_accuracy}')

# Confusion Matrices
nb_confusion = confusion_matrix(y_test, nb_predictions)
dt_confusion = confusion_matrix(y_test, dt_predictions)
print('\nNaive Bayes Confusion Matrix:')
print(nb_confusion)
print('Decision Tree Confusion Matrix:')
print(dt_confusion)
