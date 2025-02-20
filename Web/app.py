from flask import Flask, request, jsonify
import pickle
import re
import nltk
import fitz  # PyMuPDF for PDF reading
import tempfile
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
with open('Files/Resume-Screening-System.pickle', 'rb') as file:
    model = pickle.load(file)

with open('Files/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Initialize NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Resume Screening API is running!"})

def clean_text_nltk(text):
    text = text.lower()
    text = re.sub(r"\W+", " ", text)  # Remove non-word characters
    text = re.sub(r"\d+", "", text)  # Remove digits
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    tokens = text.split()  # Tokenize the text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens)

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")  # Extract text from each page
        return text
    except Exception as e:
        return str(e)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'resume' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['resume']
        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400

        # Ensure the file is a PDF
        if not file.filename.endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        # Save the uploaded file temporarily in a temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.read())
            tmp_pdf_path = tmp_file.name

        # Extract text from the PDF
        resume_text = extract_text_from_pdf(tmp_pdf_path)
        if not resume_text.strip():
            return jsonify({"error": "Failed to extract text from the resume"}), 400

        # Clean and predict
        cleaned_resume = clean_text_nltk(resume_text)
        features = vectorizer.transform([cleaned_resume])
        prediction = model.predict(features)[0]

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
