import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Expanded sample texts with diverse categories
sample_texts = [
    "data science", "software engineering", "marketing", "finance", 
    "biotechnology", "healthcare", "machine learning", "data analysis",
    "computer vision", "digital marketing", "accounting", "artificial intelligence"
]

sample_labels = [
    "data science", "software engineering", "Business", "Business", 
    "Healthcare", "Healthcare", "data science", "data science", 
    "IT", "Business", "Business", "data science"
]

# Initialize vectorizer and transform the texts into features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sample_texts)

# Train a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X, sample_labels)

# Save the trained model to a file
with open('d:/ML Projects/AI-Powered-Resume-Screening-System/Files/Resume-Screening-System.pickle', 'wb') as file:
    pickle.dump(model, file)

# Save the vectorizer to a file
with open('d:/ML Projects/AI-Powered-Resume-Screening-System/Files/vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

print("Model and vectorizer saved successfully!")
