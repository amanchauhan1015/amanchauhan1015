from flask import Flask, render_template, request
import PyPDF2
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load BERT Model
print("Loading BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='models')
print("BERT model loaded successfully!")

# Database Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///results.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Table
class ResumeResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200))
    score = db.Column(db.Float)
    matched_skills = db.Column(db.Text)
    missing_skills = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create Database
with app.app_context():
    db.create_all()

# Skills List
skills_list = [
    "python",
    "machine learning",
    "data science",
    "pandas",
    "numpy",
    "sql",
    "deep learning",
    "flask",
    "django"
]

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
                skill for skill in skills_list
                if skill in resume_text_lower
            ]

            missing_skills = [
                skill for skill in skills_list
                if skill not in matched_skills
            ]

            results.append((
                file_names[i],
                score,
                matched_skills,
                missing_skills
            ))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

    return render_template("index.html", results=results)
if __name__ == "__main__":
    app.run(debug=True)