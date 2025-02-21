import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Improved training data
sample_texts = [
    "I am a dedicated and motivated undergraduate with a strong foundation in software development, problem solving and technical innovation.",
    "Passionate data scientist with expertise in machine learning, deep learning, and data visualization.",
    "Marketing specialist with a focus on digital marketing, SEO, and content creation.",
    "Financial analyst with experience in risk management, investment analysis, and portfolio management.",
    "Biotech researcher specializing in genetics, pharmaceuticals, and biomedical engineering.",
    "Healthcare professional with experience in hospital management, patient care, and medical research.",
    "Machine learning expert working with neural networks, NLP, and AI solutions.",
    "Data analyst skilled in SQL, Excel, Tableau, and statistical modeling.",
    "Proficient in programming languages with hands-on experience in building web and mobile applications, Skilled in front-end and back-end development, utilizing technologies.",
    "Digital marketing strategist skilled in social media campaigns, PPC, and brand management.",
    "Certified accountant with experience in tax planning, auditing, and financial reporting.",
    "AI researcher developing algorithms for computer vision, NLP, and reinforcement learning."
]

sample_labels = [
    "Software Engineering", "Data Science", "Business", "Finance",
    "Healthcare", "Healthcare", "Data Science", "Data Science",
    "Software Engineering", "Business", "Business", "Artificial Intelligence"
]

# Train the model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sample_texts)
model = MultinomialNB()
model.fit(X, sample_labels)

# Save the trained model
with open('Files/Resume-Screening-System.pickle', 'wb') as file:
    pickle.dump(model, file)

# Save the vectorizer
with open('Files/vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

print("Model and vectorizer saved successfully!")
