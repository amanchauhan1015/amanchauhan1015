from flask import Flask, render_template, request, redirect
import PyPDF2
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)


try:
    import time
    for model in client.models.list():
        print(model.name)
except Exception as e:
    print("Error listing models:", e)

app = Flask(__name__)

# Load BERT Model
print("Loading BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("BERT model loaded successfully!")

RECRUITER_QUESTIONS = [
    "What is the candidate's highest qualification?",
    "Does the candidate have internship experience?",
    "Which companies has the candidate worked for?",
    "What programming languages does the candidate know?",
    "What technical skills does the candidate have?",
    "What projects has the candidate completed?",
    "What certifications does the candidate have?",
    "Does the candidate have cloud knowledge?",
    "Does the candidate have AI/ML experience?",
    "Is the candidate suitable for this job?"
]
def answer_recruiter_questions(resume_text):

    import json
    import time

    prompt = f"""
You are an expert HR recruiter.

Analyze the following resume and answer ONLY in JSON.

Resume:
{resume_text}

Return JSON exactly like this:

{{
    "Highest Qualification":"",
    "Years of Experience":"",
    "Technical Skills":"",
    "Projects":"",
    "Internship":"",
    "Certifications":"",
    "Strengths":"",
    "Weaknesses":"",
    "Recommended Role":"",
    "Overall Recommendation":""
}}
"""

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

            text = response.text.strip()

            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()

            elif text.startswith("```"):
                text = text.replace("```", "").strip()

            return json.loads(text)

        except Exception as e:

            print("Gemini Error:", e)

            if attempt == 2:
                return {
                    "Highest Qualification":"Unavailable",
                    "Years of Experience":"Unavailable",
                    "Technical Skills":"Unavailable",
                    "Projects":"Unavailable",
                    "Internship":"Unavailable",
                    "Certifications":"Unavailable",
                    "Strengths":"Unavailable",
                    "Weaknesses":"Unavailable",
                    "Recommended Role":"Unavailable",
                    "Overall Recommendation":"Gemini API Busy"
                }

            time.sleep(3)


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
    resume_text = db.Column(db.Text)

    

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
    all_qa_answers = []

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
        all_qa_answers = []

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

            qa_answers = answer_recruiter_questions(resume_texts[i])
            print(qa_answers)
            all_qa_answers.append(qa_answers)
            
            
            resume_result = ResumeResult(
                candidate_name=file_names[i].replace(".pdf", ""),
                filename=file_names[i],

                match_score=score,
                ats_score=ats_score,

                matched_skills=", ".join(matched_skills),
                missing_skills=", ".join(missing_skills),

                resume_summary=resume_texts[i][:500],
                resume_text=resume_texts[i],

                recommendation="Selected" if score >= 75 else "Rejected"
            )

            db.session.add(resume_result)
            print("Saved:", file_names[i])

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        db.session.commit()
        print("Database committed successfully")

    return render_template(
    "index.html",
    results=results,
    all_qa_answers=all_qa_answers
)


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

    print(data.resume_text)  # <-- Add this
    qa_answers = answer_recruiter_questions(data.resume_text)
    print(qa_answers)        # <-- Add this

    return render_template(
        "view.html",
        data=data,
        qa_answers=qa_answers
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)