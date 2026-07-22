from flask import Flask, render_template, request, redirect
import PyPDF2
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load BERT Model
print("Loading BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("BERT model loaded successfully!")

# Database Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///results.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Table
class ResumeResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    candidate_name = db.Column(db.String(100))
    filename = db.Column(db.String(200))

    match_score = db.Column(db.Float)
    ats_score = db.Column(db.Float)

    matched_skills = db.Column(db.Text)
    missing_skills = db.Column(db.Text)

    resume_summary = db.Column(db.Text)
    recommendation = db.Column(db.String(50))

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create Database
class JobDescription(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    job_description = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
with app.app_context():
    db.create_all()
    

# Skills List
skills_list = [
    "python", "java", "c++", "c", "javascript", "html", "css",
    "react", "node.js", "flask", "django", "fastapi",
    "sql", "mysql", "postgresql", "mongodb",
    "machine learning", "deep learning", "data science",
    "artificial intelligence", "nlp", "computer vision",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "git", "github", "docker", "kubernetes",
    "aws", "azure", "gcp",
    "power bi", "excel", "tableau",
    "linux", "rest api"
]
# Extract required skills from Job Description
def extract_jd_skills(job_description):
    jd_text = job_description.lower()

    jd_skills = []

    for skill in skills_list:
        if skill.lower() in jd_text:
            jd_skills.append(skill)

    return jd_skills

# Function to Extract Text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text

# Main Route
@app.route("/", methods=["GET", "POST"])
def index():

    results = []

    if request.method == "POST":

        resume_files = request.files.getlist("resume")
        job_description = request.form["job_description"]
        jd_skills = extract_jd_skills(job_description)

        resume_texts = []
        file_names = []

        # Extract text from all resumes
        for file in resume_files:

            resume_text = extract_text_from_pdf(file)

            if resume_text and resume_text.strip():
                print(f"Processing: {file.filename}")
                resume_texts.append(resume_text[:3000])
                file_names.append(file.filename)

        # Batch BERT Encoding
        resume_embeddings = model.encode(resume_texts)
        jd_embedding = model.encode(job_description)

        # Process each resume
        for i, resume_embedding in enumerate(resume_embeddings):

            similarity = cosine_similarity(
                [resume_embedding],
                [jd_embedding]
            )

            score = round(similarity[0][0] * 100, 2)

            resume_text_lower = resume_texts[i].lower()

            matched_skills = [
                skill for skill in jd_skills
                if skill in resume_text_lower
            ]

            missing_skills = [
                skill for skill in jd_skills
                if skill not in matched_skills
            ]
            ats_score = round((score * 0.7) + (len(matched_skills) * 3), 2)
            if ats_score > 100:
                ats_score = 100

            results.append((
                file_names[i],
                score,
                ats_score,
                matched_skills,
                missing_skills
                
            ))
            resume_result = ResumeResult(
                candidate_name=file_names[i].replace(".pdf", ""),
                filename=file_names[i],

                match_score=score,
                ats_score=ats_score,

                matched_skills=", ".join(matched_skills),
                missing_skills=", ".join(missing_skills),

                resume_summary=resume_texts[i][:500],

                recommendation="Selected" if score >= 75 else "Rejected"
            )

            db.session.add(resume_result)
            print("Saved:", file_names[i])

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        db.session.commit()
        print("Database committed successfully")

    return render_template("index.html", results=results)


import os

@app.route("/history")
def history():

    history = ResumeResult.query.order_by(
        ResumeResult.created_at.desc()
    ).all()

    total_resumes = len(history)

    average_ats = 0
    highest_match = 0
    selected = 0

    if total_resumes > 0:

        average_ats = round(
            sum(item.ats_score for item in history) / total_resumes,
            2
        )

        highest_match = max(
            item.match_score for item in history
        )

        selected = len(
            [item for item in history if item.recommendation == "Selected"]
        )

    return render_template(
        "history.html",
        history=history,
        total_resumes=total_resumes,
        average_ats=average_ats,
        highest_match=highest_match,
        selected=selected
    )
@app.route("/delete/<int:id>")
def delete(id):

    record = ResumeResult.query.get_or_404(id)

    db.session.delete(record)

    db.session.commit()

    return redirect("/history")
@app.route("/view/<int:id>")
def view(id):

    data = ResumeResult.query.get_or_404(id)

    return render_template(
        "view.html",
        data=data
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)