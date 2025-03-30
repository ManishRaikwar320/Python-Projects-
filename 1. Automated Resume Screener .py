"""
📄 Automated Resume Screener (Python + NLP + Flask)

👉 यह प्रोजेक्ट Natural Language Processing (NLP) का उपयोग करके Resume Screening करेगा।
👉 Flask Web App से Resume Upload करें और Skills Match Score पाएं!
👉 HRs के लिए Useful Tool – यह ATS (Applicant Tracking System) की तरह काम करेगा।


🔹 Step-by-Step Implementation
✅ Step 1: Resume Upload करें और Text Extract करें
✅ Step 2: NLP से Keywords Extract करें
✅ Step 3: Job Description Match करके Score निकालें
✅ Step 4: Flask Web App बनाएं

"""


import os
import docx2txt
import re
import nltk
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK Stopwords Download
nltk.download("stopwords")
from nltk.corpus import stopwords

# Load Job Description
job_description = """
We are looking for a Python Developer with experience in Machine Learning, 
Natural Language Processing, and Flask. The ideal candidate should be proficient in 
data preprocessing, feature engineering, model training, and API development.
"""



def extract_text_from_resume(file_path):
    text = docx2txt.process(file_path)  # Extract text from DOCX
    text = re.sub(r"\n+", " ", text)  # Remove extra newlines
    return text.lower()


def calculate_similarity(resume_text, job_desc):
    # Remove Stopwords
    stop_words = set(stopwords.words("english"))
    resume_text = " ".join([word for word in resume_text.split() if word not in stop_words])
    
    # Convert to Vectors
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    
    # Compute Cosine Similarity
    similarity_score = cosine_similarity(vectors)[0, 1] * 100  # Percentage
    return round(similarity_score, 2)


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "resume" not in request.files:
        return "No file uploaded!", 400

    file = request.files["resume"]
    if file.filename == "":
        return "No selected file", 400

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Extract Resume Text
    resume_text = extract_text_from_resume(file_path)

    # Compute Match Score
    match_score = calculate_similarity(resume_text, job_description)

    return f"Resume Match Score: {match_score}%"

if __name__ == "__main__":
    app.run(debug=True)


<!DOCTYPE html>
<html>
<head>
    <title>Resume Screener</title>
</head>
<body>
    <h2>Upload Your Resume</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="resume">
        <button type="submit">Submit</button>
    </form>
</body>
</html>


