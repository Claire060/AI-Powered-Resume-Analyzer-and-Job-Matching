from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify, send_from_directory
import os
import re
import numpy as np
import pandas as pd
import uuid
import json
from werkzeug.security import generate_password_hash, check_password_hash
import random
from datetime import datetime, timedelta
import hashlib
import smtplib
from email.mime.text import MIMEText
from itsdangerous import URLSafeTimedSerializer
from flask import url_for
from resume_analyzer import EnhancedResumeAnalyzer
from job_matcher import JobMatchingService
from werkzeug.utils import secure_filename 
import mysql.connector
from mysql.connector import Error
import zipfile
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from job_matcher import JobMatchingService
from flask_mail import Mail, Message
import csv
from io import StringIO

app = Flask(__name__)
app.secret_key = 'os.urandom(24)' 

# Flask-Mail Configuration for Your Custom SMTP Server
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  
app.config['MAIL_PORT'] = 587  # Port for TLS (if using)
app.config['MAIL_USE_TLS'] = True  # Enable TLS
app.config['MAIL_USE_SSL'] = False  # Disable SSL (or True if using SSL)
app.config['MAIL_USERNAME'] = 'claire@cvision.com'  
app.config['MAIL_PASSWORD'] = 'Hidup@123'  
app.config['MAIL_DEFAULT_SENDER'] = 'no-reply@cvision.com'  # Default sender email
mail = Mail(app)

db_config = {
        'MYSQL_HOST': 'localhost',
        'MYSQL_USER': 'root',
        'MYSQL_PASSWORD': 'Hidup@123',
        'MYSQL_DB': 'cvision',
        'MYSQL_PORT': 3306
    }

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'doc', 'zip'}

def get_db_connection():
    """Function to create and return a database connection."""
    try:
        connection = mysql.connector.connect(
            host=db_config['MYSQL_HOST'],
            user=db_config['MYSQL_USER'],
            password=db_config['MYSQL_PASSWORD'],
            database=db_config['MYSQL_DB'],
            port=db_config['MYSQL_PORT']
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# ------------------- DIRECT MODEL LOADING -------------------
print("Loading AI models directly in main.py...")

# Global variables for job data
jobs_df = None
job_embeddings = None

# Load shared Sentence-BERT model
shared_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Shared Sentence-BERT model loaded!")

# Load your trained Resume Accuracy Model
try:
    resume_accuracy_model = SentenceTransformer(
        r"C:\Users\USER\Documents\GitHub\AI-Powered Resume Analyzer and Job Matching\trained_resume_accuracy_model"
    )
    print("Trained Resume Accuracy Model loaded successfully!")
except Exception as e:
    print(f"Failed to load trained Resume Accuracy Model: {e}")
    print("Using shared model as fallback for resume accuracy...")
    resume_accuracy_model = shared_model

# If your trained model has built-in job matching
try:
    # Just initialize your model - it might have jobs built-in
    job_matching_model = SentenceTransformer(
        r"C:\Users\USER\Documents\GitHub\AI-Powered Resume Analyzer and Job Matching\trained_jobmatching_model"
    )
    print("Trained Job Matching Model loaded with built-in job matching!")
    jobs_df = None
    job_embeddings = None
except Exception as e:
    print(f"Failed to load trained Job Matching Model: {e}")
    job_matching_model = shared_model
    jobs_df = None
    job_embeddings = None 

# ------------------- HELPER FUNCTIONS -------------------
def _embedding_to_accuracy_score(embedding):
    """Your trained model's accuracy scoring logic"""
    try:
        norm = float(np.linalg.norm(embedding))
        # Use your trained scaling factors
        score = norm * 15 + 65  # Your original scaling
        return max(0, min(100, score))
    except:
        return 75.0

def _match_jobs_fast(resume_embedding, top_k=5):
    """Fast job matching using precomputed embeddings"""
    global jobs_df, job_embeddings
    
    if job_embeddings is None or jobs_df is None:
        return []
    
    # Batch similarity computation (much faster than loops)
    similarities = cosine_similarity(
        resume_embedding.reshape(1, -1), 
        job_embeddings
    ).flatten()
    
    # Get top_k indices efficiently
    top_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    
    return [
        {
            "title": jobs_df.iloc[i]["title"],
            "score": round(similarities[i] * 100, 2),
            "description": jobs_df.iloc[i].get("description", "")[:200] + "..."
        }
        for i in top_indices
    ]

def _analyze_resume_nlp(text):
    """Fast NLP analysis without loading separate model"""
    # Skills extraction using simple keyword matching (fast)
    skills_keywords = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 'react', 'node',
                      'html', 'css', 'mongodb', 'mysql', 'postgresql', 'git', 'linux', 'windows',
                      'angular', 'vue', 'django', 'flask', 'fastapi', 'spring', 'hibernate',
                      'machine learning', 'ai', 'data science', 'analytics', 'tableau', 'powerbi']
    
    text_lower = text.lower()
    detected_skills = [skill for skill in skills_keywords if skill in text_lower]
    
    # Experience extraction
    exp_years = _extract_experience(text)
    
    # Structure analysis
    structure_score = _analyze_structure(text)
    
    return {
        'skills': detected_skills[:15],  # Limit to top 15 skills
        'experience_years': exp_years,
        'structure': {
            'score': structure_score,
            'sections_found': _get_sections_found(text)
        }
    }

def _extract_experience(text):
    """Extract years of experience"""
    pattern = r'(\d+)\s*\+?\s*(?:years?|yrs?)'
    matches = re.findall(pattern, text.lower())
    return max([int(m) for m in matches]) if matches else 0

def _analyze_structure(text):
    """Analyze resume structure completeness"""
    sections = {
        "summary": ["summary", "objective", "profile"],
        "experience": ["experience", "employment", "work"],
        "education": ["education", "degree", "university"],
        "skills": ["skills", "technical"],
        "projects": ["projects", "portfolio"]
    }
    
    found_count = 0
    text_lower = text.lower()
    
    for section, keywords in sections.items():
        if any(keyword in text_lower for keyword in keywords):
            found_count += 1
    
    return round((found_count / len(sections)) * 100)

def _get_sections_found(text):
    """Get which sections are found"""
    sections = {
        "summary": ["summary", "objective", "profile"],
        "experience": ["experience", "employment", "work"],
        "education": ["education", "degree", "university"],
        "skills": ["skills", "technical"],
        "projects": ["projects", "portfolio"]
    }
    
    found = {}
    text_lower = text.lower()
    
    for section, keywords in sections.items():
        found[section] = any(keyword in text_lower for keyword in keywords)
    
    return found

def _fallback_scoring(text):
    """Fallback scoring when model fails"""
    words = text.split()
    word_count = len(words)
    
    base_score = 60
    
    # Length factor
    if word_count > 600:
        base_score += 15
    elif word_count > 400:
        base_score += 10
    elif word_count > 200:
        base_score += 5
        
    # Skills factor
    skills_keywords = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 'react', 'node']
    skills_found = sum(1 for skill in skills_keywords if skill in text.lower())
    base_score += min(20, skills_found * 3)
    
    return min(85, base_score)

def extract_name(text):
    """Extract name from resume text (simple heuristic)"""
    lines = text.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if (len(line.split()) in [2, 3] and 
            any(word.istitle() for word in line.split()) and
            not any(keyword in line.lower() for keyword in ['resume', 'cv', 'email', 'phone'])):
            return line
    return "Candidate"

# ------------------- ROUTES -------------------

@app.route("/")
def dashboard():
    user = None

    if "user" in session:
        # Establish a database connection
        connection = get_db_connection()
        
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)

                # Fetch the user details based on the session user_id
                cursor.execute("SELECT * FROM users WHERE user_id=%s", (session["user"],))
                user = cursor.fetchone()

                # Close the cursor
                cursor.close()

            except Exception as e:
                # Handle any exceptions, such as database connection errors
                print(f"Error fetching user data: {e}")
                user = None
            finally:
                # Always close the connection after the query is complete
                connection.close()

    return render_template("dashboard.html", user=user)


@app.route("/profile")
def profile():
    # if "user" not in session:
    #     return redirect(url_for("login"))

    # try:
    #     connection = get_db_connection(db_config)
    #     cursor = connection.cursor(dictionary=True)
    # except mysql.connector.Error as e:
    #     flash(f"Error connecting to the database: {e}", "error")
    #     return redirect(url_for("login"))

    # # Fetch user info
    # cursor.execute("SELECT * FROM users WHERE user_id=%s", (session["user"],))
    # user = cursor.fetchone()

    # # Fetch user's job applications
    # cursor.execute("""
    # SELECT j.title AS job_title, a.status, a.applied_date
    # FROM applications a
    # INNER JOIN jobs j ON a.job_id = j.job_id
    # WHERE a.user_id = %s
    # ORDER BY a.applied_date DESC
    # """, (session["user"],))
    # applications = cursor.fetchall()


    # cursor.close()
    # connection.close()

    # if not user:
    #     flash("User not found", "error")
    #     return redirect(url_for("login"))

    # return render_template("profile.html", user=user, applications=applications)
    return render_template("profile.html")

# @app.route("/edit_profile", methods=["POST"])
# def edit_profile():
#     if "user" not in session:
#         return redirect(url_for("login"))

#     user_id = session["user"]
#     name = request.form["name"]
#     email = request.form["email"]
#     role = request.form["role"]

#     connection = get_db_connection(db_config)
#     cursor = connection.cursor()

#     cursor.execute(
#         "UPDATE users SET name=%s, email=%s, role=%s WHERE user_id=%s",
#         (name, email, role, user_id)
#     )

#     connection.commit()
#     cursor.close()
#     connection.close()

#     flash("Profile updated successfully!", "success")
#     return redirect(url_for("profile"))


# Password strength check
import re
def valid_password(password):
    return (
        len(password) >= 6 and
        re.search(r"[A-Z]", password) and
        re.search(r"[a-z]", password) and
        re.search(r"[0-9]", password) and
        re.search(r"[\W_]", password)
    )

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # New DB connection inside route
        connection = get_db_connection()
        
        if not connection:
            flash("Database connection failed. Please try again later.", "error")
            return redirect(url_for("login"))
        
        cursor = connection.cursor(dictionary=True)

        try:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()
        except Error as e:
            flash("An error occurred while querying the database.", "error")
            return redirect(url_for("login"))
        finally:
            cursor.close()
            connection.close()

        if not user or not check_password_hash(user["password"], password):
            flash("Invalid email or password", "error")
            return redirect(url_for("login"))

        session["user"] = user["user_id"]
        session["role"] = user["role"]
        session["name"] = user["name"]
        session["email"] = user["email"]

        # Check if the user was trying to access a specific page before login
        next_page = request.args.get("next")
        if next_page:
            return redirect(next_page)  # Redirect to the specific page they were trying to visit

        # Default redirection to dashboard if no specific page was requested
        return redirect(url_for("dashboard"))
    
    return render_template("login.html")


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        role = request.form.get('role')
        confirm_password = request.form.get("confirm_password")

        # Empty checks
        if not name:
            flash("Name cannot be empty", "error")
            return redirect(url_for("signup"))

        if not email:
            flash("Email cannot be empty", "error")
            return redirect(url_for("signup"))

        # Email format validation
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Invalid email format", "error")
            return redirect(url_for("signup"))

        # Password strength validation
        if not valid_password(password):
            flash("Password must contain at least 6 characters, uppercase, lowercase, number, and symbol", "error")
            return redirect(url_for("signup"))

        if password != confirm_password:
            flash("Passwords do not match", "error")
            return redirect(url_for("signup"))

        # Establish DB connection
        connection = get_db_connection()
        
        if not connection:
            flash("Database connection failed. Please try again later.", "error")
            return redirect(url_for("signup"))
        
        cursor = connection.cursor(dictionary=True)

        try:
            # Check if email already exists
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            existing = cursor.fetchone()

            if existing:
                flash("Email already registered", "error")
                return redirect(url_for("signup"))

            # Hash password before storing
            hashed = generate_password_hash(password)

            cursor.execute(
                "INSERT INTO users (name, email, password, role) VALUES (%s, %s, %s, %s)",
                (name, email, hashed, role)
            )
            connection.commit()

        except Error as e:
            flash("An error occurred while accessing the database.", "error")
            return redirect(url_for("signup"))

        finally:
            cursor.close()
            connection.close()

        flash("Account created successfully!", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

# Secret key for token generation (can be a random string)
app.config['SECRET_KEY'] = os.urandom(24)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])


@app.route('/forgot_password', methods=['POST', 'GET'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')

        # Fetch user ID based on the email (in a real-world scenario, query the DB)
        user_id = 1  # This would be fetched from the database based on the email

        # Generate OTP (6-digit number)
        otp = str(random.randint(100000, 999999))

        # Set expiration time for OTP (e.g., 10 minutes)
        expiration_time = datetime.now() + timedelta(minutes=10)

        # Store OTP in the database
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute("INSERT INTO password_reset_requests (user_id, otp, expiration_time) VALUES (%s, %s, %s)",
                           (user_id, otp, expiration_time))
            connection.commit()
            cursor.close()
            connection.close()

        # Construct OTP verification URL (temporary token for verification)
        token = serializer.dumps(email, salt='password-reset-salt')

        # Send OTP to the user's email
        subject = 'Password Reset Request'
        body = f'Your OTP for password reset is {otp}. \nIf you did not request this, please ignore this email.\n\n' \
               f'Click this link to verify and reset your password: {url_for("verify_otp", token=token, _external=True)}'

        msg = Message(subject, recipients=[email], body=body)
        
        try:
            # Send the email via SendGrid
            mail.send(msg)
            flash('Password reset email with OTP sent!', 'success')
        except Exception as e:
            flash(f'Failed to send reset email: {str(e)}', 'error')

        return render_template('forgot_password.html')

    return render_template('forgot_password.html')


@app.route('/verify_otp/<token>', methods=['GET', 'POST'])
def verify_otp(token):
    try:
        # Verify the token and extract the email
        email = serializer.loads(token, salt='password-reset-salt', max_age=600)  # OTP valid for 10 minutes
    except Exception as e:
        flash('The reset link is invalid or has expired.', 'error')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        entered_otp = request.form['otp']
        
        # Fetch OTP from the database for this email
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM password_reset_requests WHERE otp = %s AND expiration_time > NOW()', [entered_otp])
        result = cursor.fetchone()

        if result:
            flash('OTP verified. You can now reset your password.', 'success')
            return redirect(url_for('reset_password', token=token))  # Go to password reset page
        else:
            flash('Invalid OTP. Please try again.', 'error')

        cursor.close()
        connection.close()

    return render_template('verify_otp.html', token=token)


@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        # Verify the token and extract the email
        email = serializer.loads(token, salt='password-reset-salt', max_age=600)
    except Exception as e:
        flash('The reset link is invalid or has expired.', 'error')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        new_password = request.form['new_password']
        hashed_password = hashlib.sha256(new_password.encode()).hexdigest()

        # Update the user's password in the database
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute('UPDATE users SET password = %s WHERE email = %s', [hashed_password, email])
        connection.commit()

        cursor.close()
        connection.close()

        flash('Your password has been reset successfully.', 'success')
        return redirect(url_for('login'))  # Redirect to login page

    return render_template('reset_password.html', token=token)


@app.route('/api/job_matches')
def stat_job_matches():
    """Route to get the total number of job matches."""
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500

    try:
        cursor = conn.cursor()
        query = "SELECT COUNT(*) FROM job_matches"
        cursor.execute(query)
        result = cursor.fetchone()
        return jsonify({"job_matches": result[0]})
    except Error as e:
        print(f"Error executing query: {e}")
        return jsonify({"error": "Failed to retrieve job matches"}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/job-matcher')
def job_matcher():
    """Show job matches"""
    
    matches = []
    if job_matching_model:  
        try:
            # Fetch top matches from the job matching model, based on the user data
            matches = job_matching_model.get_top_matches()  
        except Exception as e:
            print(f"Error getting job matches: {e}")
            flash('Error getting job matches. Please try again later.', 'error')
            matches = []  # Empty list if something goes wrong
    else:
        flash('AI Job Matching is currently unavailable. Please try again later.', 'error')
        matches = []  # Empty list when model is not available
    
    return render_template('job_matching.html', 
                         matches=matches, 
                         matches_count=len(matches))

@app.route('/match-resume', methods=['POST'])
def match_resume():
    """Match uploaded resume to jobs"""
    if not job_matching_model:
        return jsonify({
            'success': False, 
            'error': 'Job matching service is not available. Please try again later.'
        }), 500

    if "resume" not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False, 
                'error': f'Invalid file type. Please upload: {", ".join(allowed_extensions)}'
            }), 400

        # Save file
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(save_path)

        # Analyze resume and get text
        analyzer = EnhancedResumeAnalyzer()
        result = analyzer.analyze_resume(save_path)
        
        if 'error' in result:
            return jsonify({
                'success': False, 
                'error': f'Error analyzing resume: {result["error"]}'
            }), 400

        resume_text = result.get('text', '') or result.get('raw_text_preview', '')
        
        if not resume_text.strip():
            return jsonify({
                'success': False, 
                'error': 'Could not extract text from resume. Please try a different file.'
            }), 400

        print(f"Resume text extracted: {len(resume_text)} characters")
        
        
        
        # Match resume to jobs
        job_service = JobMatchingService(model_path=job_matching_model, db_config=db_config)
        matches = job_service.match_single_candidate_to_jobs(resume_text)

        print(f"Found {len(matches)} job matches")
        
        return jsonify({
            'success': True,
            'matches': matches,
            'resume_analysis': result
        })

    except Exception as e:
        print(f"Error in match-resume: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error processing resume: {str(e)}'
        }), 500

@app.route('/resume-analyzer', methods=['GET', 'POST'])
def resume_analyzer():
    if request.method == 'POST':
        print("=== RESUME ANALYZER POST REQUEST ===")
        
        try:
            if "resume" not in request.files:
                return jsonify({'success': False, 'error': 'No file uploaded'}), 400
            
            file = request.files["resume"]
            if file.filename == "":
                return jsonify({'success': False, 'error': 'No selected file'}), 400
            
            print(f"File received: {file.filename}")
            
            # Save file
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            
            # Extract text (using your existing extractor)
            try:
                text_extractor = EnhancedResumeAnalyzer()
                text = text_extractor.extract_text_from_file(save_path)
                
                if not text or isinstance(text, str) and (text.startswith("Error") or text.startswith("Unsupported")):
                    error_msg = text if text else "Text extraction failed"
                    print(f"Text extraction failed: {error_msg}")
                    text = "Resume content extracted with limitations"
                    
            except Exception as e:
                print(f"Text extraction error: {e}")
                text = "Resume content - extraction issues"
            
            print(f"Text for analysis: {len(text)} characters")
            
            # SINGLE ENCODING with trained resume accuracy model
            try:
                print("Encoding resume text with trained Resume Accuracy Model...")
                resume_embedding = resume_accuracy_model.encode(text)
            except Exception as e:
                print(f"Trained model encoding failed, using shared model: {e}")
                resume_embedding = shared_model.encode(text)
            
            # Calculate Resume Accuracy Score using trained model
            try:
                accuracy_score = _embedding_to_accuracy_score(resume_embedding)
                print(f"Accuracy score: {accuracy_score}")
            except Exception as e:
                print(f"Accuracy scoring failed: {e}")
                accuracy_score = _fallback_scoring(text)
            
            # Job Matching using trained job matching model
            job_recommendations = []
            if job_embeddings is not None:
                try:
                    job_recommendations = _match_jobs_fast(resume_embedding, top_k=5)
                    print(f"Found {len(job_recommendations)} job matches")
                except Exception as e:
                    print(f"Job matching failed: {e}")
            
            # NLP Analysis (skills, experience, structure)
            nlp_results = _analyze_resume_nlp(text)
            
            # Combine all results
            result = {
                'score': accuracy_score,
                'accuracy_score_sbert': accuracy_score,
                'job_recommendations_sbert': job_recommendations,
                'skills': nlp_results.get('skills', []),
                'experience_years': nlp_results.get('experience_years', 0),
                'structure': nlp_results.get('structure', {'score': 0, 'sections_found': {}}),
                'summary': 'AI-powered analysis completed',
                'analysis_type': 'trained_models',
                'status': 'ok'
            }
            
            # Clean up temp file
            if os.path.exists(save_path):
                os.remove(save_path)
                print("Temporary file cleaned up")
            
            print("Analysis completed, returning results")
            
            # Return JSON response instead of redirecting
            return jsonify({
                'success': True,
                'result': result
            })
            
        except Exception as e:
            print(f"CRITICAL ERROR in route: {str(e)}")
            import traceback
            traceback.print_exc()

            # Return JSON error response
            return jsonify({
                'success': False,
                'error': f'Error processing resume: {str(e)}'
            }), 500
    
    # GET Request — Show Results (this returns HTML)
    existing_result = session.get('resume_analysis')
    if existing_result:
        return render_template("resume-analyzer.html", 
                             show_results=True, 
                             result=existing_result)
    
    return render_template("resume-analyzer.html", show_results=False)

@app.route('/extract-resume-text', methods=['POST'])
def extract_resume_text():
    """Extract text from resume for preview"""
    if "resume" not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files["resume"]
    if file.filename == "":
        return jsonify({'success': False, 'error': 'No selected file'})
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config["UPLOAD_FOLDER"], f"preview_{filename}")
        file.save(temp_path)
        
        # Extract text using your analyzer
        analyzer = EnhancedResumeAnalyzer()
        result = analyzer.analyze_resume(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']})
        
        # Get the full text
        full_text = result.get('raw_text_preview', '')
        if not full_text:
            # If no preview, try to get text from sections
            sections = result.get('sections_found', {})
            full_text = "\n\n".join(
                "\n".join(section_content) 
                for section_content in sections.values() 
                if section_content
            )
        
        return jsonify({
            'success': True,
            'text': full_text,
            'file_type': filename.split('.')[-1].lower()
        })
        
    except Exception as e:
        print(f"Error extracting text for preview: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

def process_ai_analysis_result(ai_result, resume_text):
    """Process the AI-powered analysis result for template display"""
    
    # Extract data from AI analysis
    score = ai_result.get('score', 75)
    content_score = ai_result.get('content_score', 70)
    structure_score = ai_result.get('structure_score', 65)
    match_score = ai_result.get('match_score', 60)
    
    skills_detected = ai_result.get('skills_detected', [])
    missing_skills = ai_result.get('missing_skills', [])
    suggestions = ai_result.get('suggestions', [])
    experience_level = ai_result.get('experience_level', 'Not specified')
    sections_present = ai_result.get('sections_present', {})
    
    # Get best job match
    best_match = ai_result.get('best_match_job', 'Software Developer')
    
    # Generate improvement points for quick summary
    improvement_points = []
    if missing_skills:
        improvement_points.append(f"Add missing skills: {', '.join(missing_skills[:2])}")
    
    if len(skills_detected) < 5:
        improvement_points.append("Include more technical skills in your resume")
    
    if structure_score < 70:
        improvement_points.append("Improve resume structure and section organization")
    
    # Add AI-generated suggestions
    improvement_points.extend(suggestions[:3])
    
    # If no improvement points from AI, use defaults
    if not improvement_points:
        improvement_points = [
            "Add quantifiable achievements to your work experience",
            "Include more industry-specific keywords",
            "Optimize your resume for ATS systems"
        ]
    
    processed_result = {
        'score': score,
        'content_score': content_score,
        'structure_score': structure_score,
        'match_score': match_score,
        'skills_detected': skills_detected,
        'matched_skills': skills_detected[:8],  # Top 8 skills for display
        'missing_skills': missing_skills[:4],   # Top 4 missing skills
        'suggestions': suggestions[:6],         # Top 6 suggestions
        'improvement_suggestions': improvement_points[:4],  # Top 4 for quick summary
        'experience_level': experience_level,
        'best_match_job': best_match,
        'total_skills': len(skills_detected),
        'sections_present': sections_present,
        'assessment': ai_result.get('analysis_details', {}).get('assessment', 'Good resume with potential'),
        'resume_file': 'uploaded_resume.pdf',
        'uploaded': datetime.now().strftime('%d %b %Y'),
        'name': extract_name(resume_text)  # Use your existing function
    }
    
    return processed_result

def extract_name(text):
    """Extract name from resume text (simple heuristic)"""
    lines = text.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if (len(line.split()) in [2, 3] and 
            any(word.istitle() for word in line.split()) and
            not any(keyword in line.lower() for keyword in ['resume', 'cv', 'email', 'phone'])):
            return line
    return "Candidate"

def extract_experience(analysis_result):
    """Extract experience information"""
    experience_data = analysis_result.get('experience', {})
    indicators = experience_data.get('experience_indicators', 0)
    date_mentions = experience_data.get('date_mentions', 0)
    
    if indicators > 3:
        return "Experienced professional"
    elif indicators > 1:
        return "Mid-level experience"
    else:
        return "Entry-level / Recent graduate"

def extract_education(education_section):
    """Extract education information"""
    if education_section:
        # Return the first education line
        return education_section[0] if education_section else "Education details found"
    return "Education section detected"

def generate_suggestions(analysis_result):
    """Generate improvement suggestions based on analysis"""
    suggestions = []
    
    basic_analysis = analysis_result.get('basic_analysis', {})
    skills_data = analysis_result.get('skills', {})
    sections = analysis_result.get('sections_found', {})
    
    word_count = basic_analysis.get('word_count', 0)
    skill_count = skills_data.get('total_count', 0)
    
    # Word count suggestions
    if word_count < 300:
        suggestions.append("Consider adding more details to your resume. Aim for 300-800 words.")
    elif word_count > 1000:
        suggestions.append("Your resume might be too long. Consider condensing to 1-2 pages.")
    
    # Skills suggestions
    if skill_count < 5:
        suggestions.append("Add more technical skills to make your resume more competitive.")
    
    # Section suggestions
    if not sections.get('experience'):
        suggestions.append("Add a dedicated work experience section with detailed responsibilities.")
    
    if not sections.get('education'):
        suggestions.append("Include an education section with your degrees and institutions.")
    
    if not sections.get('skills'):
        suggestions.append("Create a dedicated skills section to highlight your technical abilities.")
    
    # Always include these general suggestions
    general_suggestions = [
        "Use quantifiable achievements (e.g., 'Improved performance by 25%') instead of just responsibilities",
        "Include industry-specific keywords for better ATS compatibility",
        "Add relevant certifications and professional development courses"
    ]
    
    suggestions.extend(general_suggestions)
    return suggestions[:5]  # Return top 5 suggestions

# Sample data for analysis results
sample_results = [
    {
        "id": 1,
        "name": "John Doe",
        "title": "Software Engineer",
        "status": "processed",
        "experience": "5 years",
        "match": 85,
        "education": "Master's Degree",
        "email": "john.doe@example.com",
        "phone": "+1 (555) 123-4567",
        "location": "San Francisco, CA",
        "summary": "Experienced software engineer with 5+ years in full-stack development...",
        "skills": ["JavaScript", "React", "Node.js", "Python", "AWS", "MongoDB"]
    },
    {
        "id": 2,
        "name": "Sarah Johnson",
        "title": "Product Manager",
        "status": "processed",
        "experience": "7 years",
        "match": 92,
        "education": "MBA",
        "email": "sarah.johnson@example.com",
        "phone": "+1 (555) 987-6543",
        "location": "New York, NY",
        "summary": "Strategic product manager with 7+ years of experience in the tech industry...",
        "skills": ["Product Strategy", "Agile", "User Research", "Data Analysis", "Roadmapping"]
    }
] 

@app.route('/recruiter_analyzer')
def recruiter_analyzer():
    """Handle file upload and analysis for recruiter resume analyzer"""
    
    # If it's a GET request, render the upload page
    if request.method == 'GET':
        return render_template('recruiter_analyzer.html')

    # If it's a POST request, handle the file upload and analysis
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})
        
        # Get the file from the request
        file = request.files['file']
        
        # If the file exists and is a ZIP file, we handle the ZIP upload
        if file and file.filename.endswith('.zip'):
            # Generate a unique name for the uploaded ZIP file
            zip_filename = f"{uuid.uuid4()}.zip"
            zip_path = os.path.join('uploads', zip_filename)
            
            # Save the uploaded ZIP file
            file.save(zip_path)
            
            # Initialize the Resume Analyzer
            analyzer = EnhancedResumeAnalyzer()

            # Process ZIP file and get analysis results
            results = analyzer.analyze_zip(zip_path, 'uploads/')
            
            return jsonify({"status": "success", "results": results})
        
        # Otherwise, handle individual resume files (PDF, DOCX, etc.)
        elif file and file.filename:
            # Generate a unique name for the uploaded resume file
            file_filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
            file_path = os.path.join('uploads', file_filename)
            
            # Save the uploaded resume file
            file.save(file_path)
            
            # Initialize the Resume Analyzer
            analyzer = EnhancedResumeAnalyzer()

            # Analyze the single resume file
            results = analyzer.analyze_resume(file_path)
            
            return jsonify({"status": "success", "results": results})
        
        # If no file was selected or the file is not valid
        return jsonify({"error": "Invalid file format. Please upload a PDF, DOCX, or ZIP file."})

# In-memory list to keep track of shortlisted candidates
shortlisted_candidates = []

@app.route('/shortlist', methods=['POST'])
def shortlist_candidate():
    """Add candidate to shortlist"""
    data = request.get_json()
    candidate_id = data.get('candidate_id')
    
    if not candidate_id:
        return jsonify({"error": "Candidate ID is required"}), 400
    
    # Find candidate by ID
    candidate = next((c for c in sample_results if c['id'] == candidate_id), None)
    
    if not candidate:
        return jsonify({"error": "Candidate not found"}), 404
    
    # Check if candidate is already shortlisted
    if candidate in shortlisted_candidates:
        return jsonify({"message": f"{candidate['name']} is already shortlisted"}), 200
    
    # Add to shortlist
    shortlisted_candidates.append(candidate)
    
    return jsonify({"message": f"{candidate['name']} has been added to the shortlist"}), 200

@app.route('/get_shortlisted', methods=['GET'])
def get_shortlisted():
    """Get all shortlisted candidates"""
    return jsonify({"shortlisted_candidates": shortlisted_candidates}), 200

@app.route('/shortlisted', methods=['GET'])
def view_shortlisted():
    """Render the shortlisted candidates page"""
    return render_template('shortlisted.html', shortlisted_candidates=shortlisted_candidates)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download the uploaded file"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Simple storage (in real app, use database)
applications = [
    {'id': '1', 'name': 'Jacob Taylor', 'title': 'Senior UX Designer', 'match': 85, 'status': 'pending', 'applied': 'Jun 12, 2023'},
    {'id': '2', 'name': 'Amanda Lee', 'title': 'Product Manager', 'match': 78, 'status': 'pending', 'applied': 'Jun 14, 2023'},
    {'id': '3', 'name': 'Michael Rodriguez', 'title': 'Frontend Developer', 'match': 92, 'status': 'accepted', 'applied': 'Jun 15, 2023'},
    {'id': '4', 'name': 'Sarah Patterson', 'title': 'Data Analyst', 'match': 65, 'status': 'rejected', 'applied': 'Jun 11, 2023'}
]

# Get all applications
@app.route('/api/applications')
def get_applications():
    return jsonify(applications)

# Update application status
@app.route('/api/applications/<app_id>/status', methods=['POST'])
def update_status(app_id):
    new_status = request.json.get('status')
    
    for app in applications:
        if app['id'] == app_id:
            app['status'] = new_status
            return jsonify({'success': True, 'message': f"Updated {app['name']} to {new_status}"})
    
    return jsonify({'error': 'Application not found'}), 404

# Get stats
@app.route('/api/stats')
def get_stats():
    total = len(applications)
    accepted = len([a for a in applications if a['status'] == 'accepted'])
    pending = len([a for a in applications if a['status'] == 'pending'])
    rejected = len([a for a in applications if a['status'] == 'rejected'])
    
    return jsonify({
        'total': total,
        'accepted': accepted,
        'pending': pending,
        'rejected': rejected
    })


@app.route('/jobs')
def jobs():
    return render_template('jobs.html')

@app.route('/post_job')
def post_job():
    return render_template('post_job.html')


def send_status_email(to_email, job_title, status):
    msg = MIMEText(f"Your application for '{job_title}' has been updated to: {status}")
    msg["Subject"] = "Job Application Status Update"
    msg["From"] = "yourapp@gmail.com"
    msg["To"] = to_email

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login("yourapp@gmail.com", "your_app_password")
    server.send_message(msg)
    server.quit()

@app.route("/apply/<int:job_id>")
def apply(job_id):
    if "user" not in session:
        flash("Please log in first", "error")
        return redirect(url_for("login"))

    connection = get_db_connection(app.config)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO applications (user_id, job_id)
        VALUES (%s, %s)
    """, (session["user"], job_id))

    connection.commit()
    cursor.close()
    connection.close()

    flash("Application submitted!", "success")
    return redirect(url_for("profile"))


# def get_resume_scores_trend():
#     """Fetch resume score trends."""
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500

#     try:
#         cursor = conn.cursor()

#         # SQL query to fetch average resume scores per month
#         query = """
#             SELECT DATE_FORMAT(upload_date, '%Y-%m') AS month, AVG(score) 
#             FROM resumes 
#             GROUP BY month
#             ORDER BY month;
#         """
#         cursor.execute(query)
#         scores = cursor.fetchall()

#         # Prepare data for frontend rendering
#         months = [score['month'] for score in scores]
#         avg_scores = [score['AVG(score)'] for score in scores]

#         return jsonify({
#             "months": months,
#             "average_scores": avg_scores
#         })
#     except mysql.MySQLError as e:
#         print(f"Error executing query: {e}")
#         return jsonify({"error": "Failed to retrieve resume scores trend"}), 500
#     finally:
#         # Close the cursor and the connection
#         cursor.close()
#         conn.close()

# @app.route('/api/resume_scores_trend')
# def resume_scores_trend():
#     data = get_resume_scores_trend()
#     return jsonify(data)


# def get_score_distribution():
#     """Function to get the distribution of resume scores."""
#     conn = get_db_connection()  # Get database connection
#     if conn is None:
#         return {"error": "Failed to connect to the database"}  # Return error if connection fails
    
#     try:
#         cursor = conn.cursor()
        
#         # Query to count the number of resumes in each score range
#         query = """
#             SELECT 
#                 CASE 
#                     WHEN score >= 81 THEN 'Excellent'
#                     WHEN score >= 71 THEN 'Good'
#                     WHEN score >= 61 THEN 'Average'
#                     ELSE 'Needs Work'
#                 END AS score_category,
#                 COUNT(*) 
#             FROM resumes 
#             GROUP BY score_category;
#         """
#         cursor.execute(query)
#         distribution = cursor.fetchall()

#         # Convert the data into a dictionary for easy rendering
#         categories = ['Excellent', 'Good', 'Average', 'Needs Work']
#         counts = {category: 0 for category in categories}  # Default counts to 0
#         for row in distribution:
#             counts[row['score_category']] = row['COUNT(*)']  # Map score category to its count

#         return counts
#     except mysql.MySQLError as e:
#         print(f"Error executing query: {e}")
#         return {"error": "Failed to fetch score distribution"}
#     finally:
#         # Always close the connection and cursor
#         cursor.close()
#         conn.close()

# @app.route('/api/score_distribution')
# def score_distribution():
#     data = get_score_distribution()
#     return jsonify(data)

# def get_top_skills_analysis():
#     """Function to get the top skills and their matching percentages."""
#     conn = get_db_connection()
#     if conn is None:
#         return {}  # Return empty dict if DB connection fails

#     try:
#         cursor = conn.cursor()
        
#         # Query to get the top skills with their matching percentages
#         query = """
#             SELECT skill, (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM resumes)) AS match_percentage
#             FROM resume_skills
#             GROUP BY skill
#             ORDER BY match_percentage DESC;
#         """
#         cursor.execute(query)
#         skills = cursor.fetchall()

#         # Convert results into a dictionary for easy processing
#         skills_dict = {skill['skill']: skill['match_percentage'] for skill in skills}
#         return skills_dict

#     except mysql.MySQLError as e:
#         print(f"Error executing query for top skills analysis: {e}")
#         return {}  # Return empty dict on error
#     finally:
#         conn.close()

# @app.route('/api/top_skills_analysis')
# def top_skills_analysis():
#     """Route to get top skills analysis."""
#     data = get_top_skills_analysis()
#     return jsonify(data)

# def get_job_match_rates():
#     """Function to get job match rates based on resume data."""
#     conn = get_db_connection()
#     if conn is None:
#         return {}  # Return empty dict if DB connection fails

#     try:
#         cursor = conn.cursor()
        
#         # Query to calculate the match rate for different job roles
#         query = """
#             SELECT 
#                 j.title AS job_role,
#                 (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM resumes)) AS match_rate
#             FROM job_matcher jm
#             JOIN jobs j ON jm.job_id = j.job_id
#             GROUP BY j.title;
#         """
#         cursor.execute(query)
#         job_matches = cursor.fetchall()

#         # Convert results into a dictionary for easy processing
#         job_dict = {job['job_role']: job['match_rate'] for job in job_matches}
#         return job_dict

#     except mysql.MySQLError as e:
#         print(f"Error executing query for job match rates: {e}")
#         return {}  # Return empty dict on error
#     finally:
#         conn.close()

# @app.route('/api/job_match_rates')
# def job_match_rates():
#     """Route to get job match rates."""
#     data = get_job_match_rates()
#     return jsonify(data)

# @app.route('/api/resumes_analyzed')
# def stat_resumes_analyzed():
#     """Route to get the total number of resumes analyzed."""
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500

#     try:
#         cursor = conn.cursor()
#         query = "SELECT COUNT(*) FROM resumes"
#         cursor.execute(query)
#         result = cursor.fetchone()
#         return jsonify({"resumes_analyzed": result[0]})
#     except Error as e:
#         print(f"Error executing query: {e}")
#         return jsonify({"error": "Failed to retrieve resumes analyzed"}), 500
#     finally:
#         cursor.close()
#         conn.close()



# @app.route('/api/applications_submitted')
# def applications_submitted():
#     """Route to get the total number of applications submitted."""
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500

#     try:
#         cursor = conn.cursor()
#         query = "SELECT COUNT(*) FROM applications"
#         cursor.execute(query)
#         result = cursor.fetchone()
#         return jsonify({"applications_submitted": result[0]})
#     except Error as e:
#         print(f"Error executing query: {e}")
#         return jsonify({"error": "Failed to retrieve applications submitted"}), 500
#     finally:
#         cursor.close()
#         conn.close()

@app.route('/api/edit_profile', methods=['POST'])
def edit_profile():
    # Extract user data from the request
    user_data = request.get_json()
    user_id = user_data.get('user_id')
    name = user_data.get('name')
    email = user_data.get('email')

    # Get a database connection
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500
    
    cursor = conn.cursor()

    # SQL query to update user profile
    update_query = """
        UPDATE users
        SET name = %s, email = %s
        WHERE id = %s
    """
    cursor.execute(update_query, (name, email, user_id))
    conn.commit()
    
    # Close the database connection
    cursor.close()
    conn.close()

    return jsonify({"message": "Profile updated successfully!"})

# @app.route('/api/update_application_status', methods=['POST'])
# def update_application_status():
#     # Get data from the request
#     data = request.get_json()
#     application_id = data.get('application_id')
#     status = data.get('status')  # 'Pending', 'Accepted', 'Rejected'

#     # Validate status
#     if status not in ['Pending', 'Accepted', 'Rejected']:
#         return jsonify({"error": "Invalid status"}), 400

#     # Connect to the database
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500

#     cursor = conn.cursor()

#     # SQL query to update application status
#     update_query = """
#         UPDATE applications
#         SET status = %s
#         WHERE id = %s
#     """
#     cursor.execute(update_query, (status, application_id))
#     conn.commit()

#     # Close the connection
#     cursor.close()
#     conn.close()

#     return jsonify({"message": "Application status updated successfully!"})

# ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/api/apply_job', methods=['POST'])
# def apply_job():
#     """Handles job application submission."""
    
#     # Extract the job application data from the request
#     data = request.form
#     job_id = data.get('job_id')
#     candidate_id = data.get('candidate_id')  # Assuming candidate is logged in and their ID is available
#     cover_letter = data.get('cover_letter')
#     years_of_experience = data.get('years_of_experience')
#     notice_period = data.get('notice_period')
#     expected_salary = data.get('expected_salary')
#     current_location = data.get('current_location')
#     portfolio_url = data.get('portfolio_url')
#     preferred_location = data.get('preferred_location')

#     # Handle resume file upload
#     file = request.files.get('resume')
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
#     else:
#         return jsonify({"error": "Invalid file format. Please upload a PDF or DOC file."}), 400

#     # Get a database connection
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500
    
#     cursor = conn.cursor()

#     # SQL query to insert the job application into the database
#     insert_query = """
#         INSERT INTO applications (candidate_id, job_id, cover_letter, resume_url, status,
#             years_of_experience, notice_period, expected_salary, current_location, portfolio_url, preferred_location)
#         VALUES (%s, %s, %s, %s, 'Pending', %s, %s, %s, %s, %s, %s)
#     """
#     cursor.execute(insert_query, (candidate_id, job_id, cover_letter, file_path,
#                                   years_of_experience, notice_period, expected_salary,
#                                   current_location, portfolio_url, preferred_location))
#     conn.commit()

#     # Close the connection
#     cursor.close()
#     conn.close()

#     return jsonify({"message": "Your application has been successfully submitted!"}), 200

# @app.route('/api/filter_jobs', methods=['GET'])
# def filter_jobs():
#     # Extract filter parameters from the query string
#     location = request.args.get('location', '')
#     category = request.args.get('category', '')
#     job_type = request.args.get('job_type', '')
#     keyword = request.args.get('keyword', '')

#     # Create the base query
#     query = """
#         SELECT id, job_title, company_name, location, category, job_type, salary_range, job_description
#         FROM jobs
#         WHERE 1=1
#     """
    
#     # Add filters based on the user's input
#     if location:
#         query += " AND location LIKE %s"
#     if category:
#         query += " AND category LIKE %s"
#     if job_type:
#         query += " AND job_type = %s"
#     if keyword:
#         query += " AND (job_title LIKE %s OR company_name LIKE %s OR job_description LIKE %s)"

#     # Connect to the database
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500

#     cursor = conn.cursor()

#     # Prepare the query parameters based on filters
#     params = []
#     if location:
#         params.append(f"%{location}%")
#     if category:
#         params.append(f"%{category}%")
#     if job_type:
#         params.append(job_type)
#     if keyword:
#         params.extend([f"%{keyword}%" for _ in range(3)])  # Search in title, company, and description

#     # Execute the query with the dynamic parameters
#     cursor.execute(query, tuple(params))
#     jobs = cursor.fetchall()

#     # Close the connection
#     cursor.close()
#     conn.close()

#     # Return the filtered job listings
#     job_list = []
#     for job in jobs:
#         job_list.append({
#             'id': job[0],
#             'job_title': job[1],
#             'company_name': job[2],
#             'location': job[3],
#             'category': job[4],
#             'job_type': job[5],
#             'salary_range': job[6],
#             'job_description': job[7]
#         })

#     return jsonify({"jobs": job_list})

# @app.route('/api/job_listings', methods=['GET'])
# def get_job_listings():
#     """Fetch job listings from the database, including featured jobs."""
    
#     # Get a database connection
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500

#     cursor = conn.cursor()

#     # SQL query to fetch all job listings including featured jobs
#     query = """
#         SELECT id, job_title, company_name, location, salary_range, job_description, is_featured
#         FROM jobs
#         ORDER BY is_featured DESC, posted_on DESC
#     """
#     cursor.execute(query)
#     jobs = cursor.fetchall()

#     # Close the connection
#     cursor.close()
#     conn.close()

#     # Prepare the response data
#     job_list = []
#     for job in jobs:
#         job_list.append({
#             'id': job[0],
#             'job_title': job[1],
#             'company_name': job[2],
#             'location': job[3],
#             'salary_range': job[4],
#             'job_description': job[5],
#             'is_featured': job[6]
#         })

#     return jsonify({"jobs": job_list})

# @app.route('/api/post_job', methods=['POST'])
# def post_job():
#     """Handles job posting."""
    
#     # Extract job details from the form data
#     job_data = request.get_json()
#     job_title = job_data.get('job_title')
#     company_name = job_data.get('company_name')
#     location = job_data.get('location')
#     job_type = job_data.get('job_type')  # Full Time, Part Time, Contract
#     salary_range = job_data.get('salary_range')
#     job_description = job_data.get('job_description')
#     required_skills = job_data.get('required_skills')
    
#     # Validate the data (simple checks)
#     if not job_title or not company_name or not job_type or not job_description:
#         return jsonify({"error": "Missing required fields"}), 400

#     # Get a database connection
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500
    
#     cursor = conn.cursor()

#     # SQL query to insert the job posting into the database
#     insert_query = """
#         INSERT INTO jobs (job_title, company_name, location, job_type, salary_range, job_description, required_skills)
#         VALUES (%s, %s, %s, %s, %s, %s, %s)
#     """
#     cursor.execute(insert_query, (job_title, company_name, location, job_type, salary_range, job_description, required_skills))
#     conn.commit()

#     # Close the connection
#     cursor.close()
#     conn.close()

#     return jsonify({"message": "Job posted successfully!"}), 201

# @app.route('/api/candidate_list', methods=['GET'])
# def get_candidate_list():
#     """Fetch the list of candidates with relevant details like match percentage and status."""
    
#     # Get a database connection
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500
    
#     cursor = conn.cursor()
    
#     # Query to fetch candidates' details
#     query = """
#         SELECT id, name, job_title, match_percentage, status, applied_on, resume_url, profile_url
#         FROM candidates
#         ORDER BY applied_on DESC
#     """
    
#     cursor.execute(query)
#     candidates = cursor.fetchall()

#     # Close the connection
#     cursor.close()
#     conn.close()

#     # Prepare the response data
#     candidate_list = []
#     for candidate in candidates:
#         candidate_list.append({
#             'id': candidate[0],
#             'name': candidate[1],
#             'job_title': candidate[2],
#             'match_percentage': candidate[3],
#             'status': candidate[4],
#             'applied_on': candidate[5].strftime('%Y-%m-%d %H:%M:%S'),
#             'resume_url': candidate[6],
#             'profile_url': candidate[7]
#         })

#     return jsonify({"candidates": candidate_list})

# @app.route('/api/update_candidate_status', methods=['POST'])
# def update_candidate_status():
#     """Update the status of a candidate (Accepted, Pending, Rejected)."""
#     data = request.json
#     candidate_id = data.get('candidate_id')
#     new_status = data.get('status')

#     # Validate the data
#     if not candidate_id or not new_status:
#         return jsonify({"error": "Missing required fields"}), 400

#     # Get a database connection
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500
    
#     cursor = conn.cursor()

#     # Update candidate status
#     update_query = "UPDATE candidates SET status = %s WHERE id = %s"
#     cursor.execute(update_query, (new_status, candidate_id))
#     conn.commit()

#     # Close the connection
#     cursor.close()
#     conn.close()

#     return jsonify({"message": f"Candidate status updated to {new_status}"}), 200

# @app.route('/api/shortlisted_candidates', methods=['GET'])
# def get_shortlisted_candidates():
#     """Fetch shortlisted candidates with filtering options."""
    
#     # Get filter options (status, default to 'All')
#     status_filter = request.args.get('status', 'All')
    
#     # Get a database connection
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500
    
#     cursor = conn.cursor()

#     # Query based on the status filter
#     if status_filter == 'All':
#         query = "SELECT id, name, job_title, match_percentage, status, applied_on FROM shortlisted_candidates"
#     else:
#         query = "SELECT id, name, job_title, match_percentage, status, applied_on FROM shortlisted_candidates WHERE status = %s"
#         cursor.execute(query, (status_filter,))
    
#     cursor.execute(query)
#     candidates = cursor.fetchall()

#     # Close the connection
#     cursor.close()
#     conn.close()

#     # Prepare the response data
#     candidate_list = []
#     for candidate in candidates:
#         candidate_list.append({
#             'id': candidate[0],
#             'name': candidate[1],
#             'job_title': candidate[2],
#             'match_percentage': candidate[3],
#             'status': candidate[4],
#             'applied_on': candidate[5].strftime('%Y-%m-%d %H:%M:%S')
#         })

#     return jsonify({"candidates": candidate_list})

# @app.route('/api/shortlisted_statistics', methods=['GET'])
# def get_shortlisted_statistics():
#     """Get statistics about shortlisted candidates."""
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500
    
#     cursor = conn.cursor()

#     # Query to get the total candidates and the breakdown by status
#     query = """
#         SELECT 
#             COUNT(*) AS total_candidates,
#             SUM(CASE WHEN status = 'Accepted' THEN 1 ELSE 0 END) AS accepted,
#             SUM(CASE WHEN status = 'Pending' THEN 1 ELSE 0 END) AS pending,
#             SUM(CASE WHEN status = 'Rejected' THEN 1 ELSE 0 END) AS rejected
#         FROM shortlisted_candidates
#     """
    
#     cursor.execute(query)
#     stats = cursor.fetchone()

#     # Close the connection
#     cursor.close()
#     conn.close()

#     return jsonify({
#         "total_candidates": stats[0],
#         "accepted": stats[1],
#         "pending": stats[2],
#         "rejected": stats[3]
#     })

# @app.route('/api/export_candidates', methods=['GET'])
# def export_candidates():
#     """Export shortlisted candidates to a CSV file."""
    
#     # Get a database connection
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500
    
#     cursor = conn.cursor()
#     query = "SELECT name, job_title, match_percentage, status, applied_on FROM shortlisted_candidates"
#     cursor.execute(query)
#     candidates = cursor.fetchall()

#     # Create a CSV in memory
#     output = StringIO()
#     writer = csv.writer(output)
#     writer.writerow(['Name', 'Job Title', 'Match Percentage', 'Status', 'Applied On'])
    
#     for candidate in candidates:
#         writer.writerow([candidate[0], candidate[1], candidate[2], candidate[3], candidate[4]])
    
#     output.seek(0)

#     # Close the connection
#     cursor.close()
#     conn.close()

#     # Send CSV file as a response
#     return send_file(output, mimetype='text/csv', as_attachment=True, download_name="shortlisted_candidates.csv")

# @app.route('/api/add_candidate', methods=['POST'])
# def add_candidate():
#     """Add a new candidate to the shortlist."""
    
#     data = request.json
#     name = data.get('name')
#     job_title = data.get('job_title')
#     match_percentage = data.get('match_percentage')
#     status = data.get('status', 'Pending')  # Default to 'Pending' if no status is provided

#     # Validate the data
#     if not name or not job_title or not match_percentage:
#         return jsonify({"error": "Missing required fields"}), 400
    
#     # Get a database connection
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500
    
#     cursor = conn.cursor()

#     # Insert the candidate into the database
#     insert_query = """
#         INSERT INTO shortlisted_candidates (name, job_title, match_percentage, status)
#         VALUES (%s, %s, %s, %s)
#     """
#     cursor.execute(insert_query, (name, job_title, match_percentage, status))
#     conn.commit()

#     # Close the connection
#     cursor.close()
#     conn.close()

#     return jsonify({"message": "Candidate added successfully!"}), 200

# # Route for fetching analysis results
# @app.route('/api/analysis_results', methods=['GET'])
# def get_analysis_results():
#     """Fetch the analysis results for candidates."""
    
#     # Get filter options (status filter: default to 'All')
#     status_filter = request.args.get('status', 'All')
    
#     # Get a database connection
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500
    
#     cursor = conn.cursor()

#     # Query based on the status filter
#     if status_filter == 'All':
#         query = """
#             SELECT id, name, job_title, skills_match, experience, education_level, status 
#             FROM candidates
#         """
#     else:
#         query = """
#             SELECT id, name, job_title, skills_match, experience, education_level, status 
#             FROM candidates 
#             WHERE status = %s
#         """
#         cursor.execute(query, (status_filter,))
    
#     cursor.execute(query)
#     candidates = cursor.fetchall()

#     # Close the connection
#     cursor.close()
#     conn.close()

#     # Prepare the response data
#     candidate_list = []
#     for candidate in candidates:
#         candidate_list.append({
#             'id': candidate[0],
#             'name': candidate[1],
#             'job_title': candidate[2],
#             'skills_match': candidate[3],
#             'experience': candidate[4],
#             'education_level': candidate[5],
#             'status': candidate[6]
#         })

#     return jsonify({"candidates": candidate_list})

# # Route for fetching detailed candidate profile
# @app.route('/api/candidate_details/<int:candidate_id>', methods=['GET'])
# def get_candidate_details(candidate_id):
#     """Fetch detailed profile of a candidate."""
    
#     # Get a database connection
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500
    
#     cursor = conn.cursor()

#     # Query to get basic candidate details
#     query = "SELECT name, job_title, skills_match, experience, education_level FROM candidates WHERE id = %s"
#     cursor.execute(query, (candidate_id,))
#     candidate = cursor.fetchone()

#     if candidate is None:
#         return jsonify({"error": "Candidate not found"}), 404

#     # Query to get skills for the candidate
#     cursor.execute("SELECT skill FROM candidate_skills WHERE candidate_id = %s", (candidate_id,))
#     skills = cursor.fetchall()

#     # Query to get education details for the candidate
#     cursor.execute("SELECT degree, university, graduation_year FROM education WHERE candidate_id = %s", (candidate_id,))
#     education = cursor.fetchall()

#     # Close the connection
#     cursor.close()
#     conn.close()

#     # Prepare detailed response data
#     detailed_profile = {
#         'name': candidate[0],
#         'job_title': candidate[1],
#         'skills_match': candidate[2],
#         'experience': candidate[3],
#         'education_level': candidate[4],
#         'skills': [skill[0] for skill in skills],  # List of skills
#         'education': [{'degree': edu[0], 'university': edu[1], 'graduation_year': edu[2]} for edu in education]  # Education details
#     }

#     return jsonify(detailed_profile)

# # Route for fetching resume analysis results
# @app.route('/api/resume_analysis_results/<int:resume_id>', methods=['GET'])
# def get_resume_analysis_results(resume_id):
#     """Fetch detailed resume analysis results for the given resume ID."""
    
#     # Get a database connection
#     conn = get_db_connection()
#     if conn is None:
#         return jsonify({"error": "Failed to connect to the database"}), 500
    
#     cursor = conn.cursor()

#     # Fetch resume score
#     cursor.execute("SELECT resume_score, status FROM resumes WHERE id = %s", (resume_id,))
#     resume = cursor.fetchone()
#     if not resume:
#         return jsonify({"error": "Resume not found"}), 404

#     resume_score = resume[0]
#     status = resume[1]

#     # Fetch matched skills
#     cursor.execute("SELECT skill FROM matched_skills WHERE resume_id = %s", (resume_id,))
#     matched_skills = [skill[0] for skill in cursor.fetchall()]

#     # Fetch missing skills
#     cursor.execute("SELECT skill FROM missing_skills WHERE resume_id = %s", (resume_id,))
#     missing_skills = [skill[0] for skill in cursor.fetchall()]

#     # Fetch strengths and areas to improve (these could be stored as text, or fetched dynamically from analysis)
#     strengths = [
#         "Strong technical background in full-stack development",
#         "Experience with modern JavaScript frameworks like React and Angular",
#         "Cloud deployment using AWS"
#     ]
    
#     areas_to_improve = [
#         "Limited experience with containerization technologies like Docker",
#         "Need more leadership experience",
#         "Lack of quantifiable achievements in project descriptions"
#     ]

#     # Fetch resume breakdown (Education, Experience, Certifications, Projects)
#     cursor.execute("SELECT section_name, completion_percentage FROM resume_breakdown WHERE resume_id = %s", (resume_id,))
#     breakdown = cursor.fetchall()
    
#     breakdown_data = {section[0]: section[1] for section in breakdown}

#     # Fetch suggestions for improvement
#     cursor.execute("SELECT suggestion FROM suggestions WHERE resume_id = %s", (resume_id,))
#     suggestions = [suggestion[0] for suggestion in cursor.fetchall()]

#     # Close the connection
#     cursor.close()
#     conn.close()

#     # Prepare the response data
#     analysis_results = {
#         'resume_score': resume_score,
#         'status': status,
#         'matched_skills': matched_skills,
#         'missing_skills': missing_skills,
#         'strengths': strengths,
#         'areas_to_improve': areas_to_improve,
#         'resume_breakdown': breakdown_data,
#         'suggestions': suggestions
#     }

#     return jsonify(analysis_results)


# ------------------- MAIN -------------------
if __name__ == "__main__":
    app.run(debug=True)