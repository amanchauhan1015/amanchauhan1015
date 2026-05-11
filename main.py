from flask import Flask, render_template, request
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///results.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
class ResumeResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200))
    score = db.Column(db.Float)
    matched_skills = db.Column(db.Text)
    missing_skills = db.Column(db.Text)
    created_at = db.Column(db.DateTime)

# 🔹 Skills List (ATS style)
skills_list = [
    "python", "machine learning", "data science", "pandas",
    "numpy", "sql", "deep learning", "flask", "django"
]

# 🔹 Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# 🔹 Main Route
@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        resume_files = request.files.getlist("resume")
        job_description = request.form["job_description"]

        for file in resume_files:
            resume_text = extract_text_from_pdf(file)
            resume_text_lower = resume_text.lower()

            # Skill Extraction
            matched_skills = []
            for skill in skills_list:
                if skill in resume_text_lower:
                    matched_skills.append(skill)

            # Similarity Score
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([resume_text, job_description])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])
            score = round(similarity[0][0] * 100, 2)

            missing_skills = [skill for skill in skills_list if skill not in matched_skills]

            results.append((file.filename, score, matched_skills, missing_skills))
            results.sort(key=lambda x: x[1], reverse=True)

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)