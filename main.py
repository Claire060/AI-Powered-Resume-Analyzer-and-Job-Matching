from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify, send_file
import os
import re
import uuid
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
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        connection = get_db_connection(db_config)
        cursor = connection.cursor(dictionary=True)
    except mysql.connector.Error as e:
        flash(f"Error connecting to the database: {e}", "error")
        return redirect(url_for("login"))

    # Fetch user info
    cursor.execute("SELECT * FROM users WHERE user_id=%s", (session["user"],))
    user = cursor.fetchone()

    # Fetch user's job applications
    cursor.execute("""
    SELECT j.title AS job_title, a.status, a.applied_date
    FROM applications a
    INNER JOIN jobs j ON a.job_id = j.job_id
    WHERE a.user_id = %s
    ORDER BY a.applied_date DESC
    """, (session["user"],))
    applications = cursor.fetchall()


    cursor.close()
    connection.close()

    if not user:
        flash("User not found", "error")
        return redirect(url_for("login"))

    return render_template("profile.html", user=user, applications=applications)


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

@app.route('/resume-analyzer', methods=['POST'])
def resume_analyzer():
    """
    Handle resume file upload, analyze the resume, and return results.
    """
    if 'resume' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400

    file = request.files['resume']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file format. Please upload a PDF, DOCX, or TXT file.'}), 400

    # Secure the filename
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        # Save the file to the server
        file.save(file_path)
        
        # Initialize the resume analyzer
        analyzer = EnhancedResumeAnalyzer()

        # Analyze the resume
        job_description = request.form.get('job_description', None)  # Optionally pass job description
        analysis_results = analyzer.analyze_resume(file_path, job_description)
        
        # Remove the uploaded file after analysis (optional)
        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({
            'success': True,
            'analysis': analysis_results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"An error occurred during the resume analysis: {str(e)}"
        }), 500

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


def get_resume_scores_trend():
    """Fetch resume score trends."""
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500

    try:
        cursor = conn.cursor()

        # SQL query to fetch average resume scores per month
        query = """
            SELECT DATE_FORMAT(upload_date, '%Y-%m') AS month, AVG(score) 
            FROM resumes 
            GROUP BY month
            ORDER BY month;
        """
        cursor.execute(query)
        scores = cursor.fetchall()

        # Prepare data for frontend rendering
        months = [score['month'] for score in scores]
        avg_scores = [score['AVG(score)'] for score in scores]

        return jsonify({
            "months": months,
            "average_scores": avg_scores
        })
    except mysql.MySQLError as e:
        print(f"Error executing query: {e}")
        return jsonify({"error": "Failed to retrieve resume scores trend"}), 500
    finally:
        # Close the cursor and the connection
        cursor.close()
        conn.close()

@app.route('/api/resume_scores_trend')
def resume_scores_trend():
    data = get_resume_scores_trend()
    return jsonify(data)


def get_score_distribution():
    """Function to get the distribution of resume scores."""
    conn = get_db_connection()  # Get database connection
    if conn is None:
        return {"error": "Failed to connect to the database"}  # Return error if connection fails
    
    try:
        cursor = conn.cursor()
        
        # Query to count the number of resumes in each score range
        query = """
            SELECT 
                CASE 
                    WHEN score >= 81 THEN 'Excellent'
                    WHEN score >= 71 THEN 'Good'
                    WHEN score >= 61 THEN 'Average'
                    ELSE 'Needs Work'
                END AS score_category,
                COUNT(*) 
            FROM resumes 
            GROUP BY score_category;
        """
        cursor.execute(query)
        distribution = cursor.fetchall()

        # Convert the data into a dictionary for easy rendering
        categories = ['Excellent', 'Good', 'Average', 'Needs Work']
        counts = {category: 0 for category in categories}  # Default counts to 0
        for row in distribution:
            counts[row['score_category']] = row['COUNT(*)']  # Map score category to its count

        return counts
    except mysql.MySQLError as e:
        print(f"Error executing query: {e}")
        return {"error": "Failed to fetch score distribution"}
    finally:
        # Always close the connection and cursor
        cursor.close()
        conn.close()

@app.route('/api/score_distribution')
def score_distribution():
    data = get_score_distribution()
    return jsonify(data)

def get_top_skills_analysis():
    """Function to get the top skills and their matching percentages."""
    conn = get_db_connection()
    if conn is None:
        return {}  # Return empty dict if DB connection fails

    try:
        cursor = conn.cursor()
        
        # Query to get the top skills with their matching percentages
        query = """
            SELECT skill, (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM resumes)) AS match_percentage
            FROM resume_skills
            GROUP BY skill
            ORDER BY match_percentage DESC;
        """
        cursor.execute(query)
        skills = cursor.fetchall()

        # Convert results into a dictionary for easy processing
        skills_dict = {skill['skill']: skill['match_percentage'] for skill in skills}
        return skills_dict

    except mysql.MySQLError as e:
        print(f"Error executing query for top skills analysis: {e}")
        return {}  # Return empty dict on error
    finally:
        conn.close()

@app.route('/api/top_skills_analysis')
def top_skills_analysis():
    """Route to get top skills analysis."""
    data = get_top_skills_analysis()
    return jsonify(data)

def get_job_match_rates():
    """Function to get job match rates based on resume data."""
    conn = get_db_connection()
    if conn is None:
        return {}  # Return empty dict if DB connection fails

    try:
        cursor = conn.cursor()
        
        # Query to calculate the match rate for different job roles
        query = """
            SELECT 
                j.title AS job_role,
                (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM resumes)) AS match_rate
            FROM job_matcher jm
            JOIN jobs j ON jm.job_id = j.job_id
            GROUP BY j.title;
        """
        cursor.execute(query)
        job_matches = cursor.fetchall()

        # Convert results into a dictionary for easy processing
        job_dict = {job['job_role']: job['match_rate'] for job in job_matches}
        return job_dict

    except mysql.MySQLError as e:
        print(f"Error executing query for job match rates: {e}")
        return {}  # Return empty dict on error
    finally:
        conn.close()

@app.route('/api/job_match_rates')
def job_match_rates():
    """Route to get job match rates."""
    data = get_job_match_rates()
    return jsonify(data)

@app.route('/api/resumes_analyzed')
def stat_resumes_analyzed():
    """Route to get the total number of resumes analyzed."""
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500

    try:
        cursor = conn.cursor()
        query = "SELECT COUNT(*) FROM resumes"
        cursor.execute(query)
        result = cursor.fetchone()
        return jsonify({"resumes_analyzed": result[0]})
    except Error as e:
        print(f"Error executing query: {e}")
        return jsonify({"error": "Failed to retrieve resumes analyzed"}), 500
    finally:
        cursor.close()
        conn.close()



@app.route('/api/applications_submitted')
def applications_submitted():
    """Route to get the total number of applications submitted."""
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500

    try:
        cursor = conn.cursor()
        query = "SELECT COUNT(*) FROM applications"
        cursor.execute(query)
        result = cursor.fetchone()
        return jsonify({"applications_submitted": result[0]})
    except Error as e:
        print(f"Error executing query: {e}")
        return jsonify({"error": "Failed to retrieve applications submitted"}), 500
    finally:
        cursor.close()
        conn.close()

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

@app.route('/api/update_application_status', methods=['POST'])
def update_application_status():
    # Get data from the request
    data = request.get_json()
    application_id = data.get('application_id')
    status = data.get('status')  # 'Pending', 'Accepted', 'Rejected'

    # Validate status
    if status not in ['Pending', 'Accepted', 'Rejected']:
        return jsonify({"error": "Invalid status"}), 400

    # Connect to the database
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500

    cursor = conn.cursor()

    # SQL query to update application status
    update_query = """
        UPDATE applications
        SET status = %s
        WHERE id = %s
    """
    cursor.execute(update_query, (status, application_id))
    conn.commit()

    # Close the connection
    cursor.close()
    conn.close()

    return jsonify({"message": "Application status updated successfully!"})

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/apply_job', methods=['POST'])
def apply_job():
    """Handles job application submission."""
    
    # Extract job application data from the request
    data = request.form
    job_id = data.get('jobId')
    applicant_name = data.get('applicantName')
    applicant_email = data.get('applicantEmail')
    applicant_phone = data.get('applicantPhone')
    applicant_experience = data.get('applicantExperience')
    applicant_notice_period = data.get('applicantNoticePeriod')
    applicant_salary = data.get('applicantSalary')
    applicant_cover_letter = data.get('applicantCoverLetter')

    # Handle resume file upload
    file = request.files.get('resume')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
    else:
        return jsonify({"error": "Invalid file format. Please upload a PDF, DOC, or DOCX file."}), 400

    # Get a database connection
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500
    
    cursor = conn.cursor()

    # SQL query to insert the job application into the database
    insert_query = """
        INSERT INTO job_applications (candidate_name, candidate_email, candidate_phone, job_id, cover_letter, 
            resume_url, years_of_experience, notice_period, expected_salary, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'Pending')
    """
    cursor.execute(insert_query, (applicant_name, applicant_email, applicant_phone, job_id, applicant_cover_letter, 
                                  file_path, applicant_experience, applicant_notice_period, applicant_salary))
    conn.commit()

    # Close the connection
    cursor.close()
    conn.close()

    return jsonify({"message": "Your application has been successfully submitted!"}), 200


@app.route('/api/filter_jobs', methods=['GET'])
def filter_jobs():
    """Filter jobs based on parameters."""
    
    # Extract filter parameters from the query string
    location = request.args.get('location', '')
    category = request.args.get('category', '')
    job_type = request.args.get('job_type', '')
    keyword = request.args.get('keyword', '')
    salary_range = request.args.get('salary', '')

    # Create the base query
    query = """
        SELECT job_id, title, description, required_skills, location, recruiter_id, salary
        FROM jobs
        WHERE 1=1
    """
    
    # Add filters based on the user's input
    params = []
    if location:
        query += " AND location LIKE %s"
        params.append(f"%{location}%")
    if category:
        query += " AND category LIKE %s"
        params.append(f"%{category}%")
    if job_type:
        query += " AND job_type = %s"
        params.append(job_type)
    if keyword:
        query += " AND (title LIKE %s OR description LIKE %s OR required_skills LIKE %s)"
        params.extend([f"%{keyword}%" for _ in range(3)])  # Search in title, description, and skills
    if salary_range:
        query += " AND salary LIKE %s"
        params.append(f"%{salary_range}%")

    # Get a database connection
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500

    cursor = conn.cursor()

    # Execute the query with the dynamic parameters
    cursor.execute(query, tuple(params))
    jobs = cursor.fetchall()

    # Close the connection
    cursor.close()
    conn.close()

    # Return the filtered job listings
    job_list = []
    for job in jobs:
        job_list.append({
            'job_id': job[0],
            'title': job[1],
            'description': job[2],
            'required_skills': job[3],
            'location': job[4],
            'recruiter_id': job[5],
            'salary': job[6]
        })

    return jsonify({"jobs": job_list})

@app.route('/api/job_listings', methods=['GET'])
def get_job_listings():
    """Fetch job listings from the database, including featured jobs."""
    
    # Get a database connection
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500

    cursor = conn.cursor()

    # SQL query to fetch all job listings
    query = """
        SELECT job_id, title, description, required_skills, location, recruiter_id, salary
        FROM jobs
        ORDER BY posted_on DESC
    """
    cursor.execute(query)
    jobs = cursor.fetchall()

    # Close the connection
    cursor.close()
    conn.close()

    # Prepare the response data
    job_list = []
    for job in jobs:
        job_list.append({
            'job_id': job[0],
            'title': job[1],
            'description': job[2],
            'required_skills': job[3],
            'location': job[4],
            'recruiter_id': job[5],
            'salary': job[6]
        })

    return jsonify({"jobs": job_list})


@app.route('/api/post_job', methods=['POST'])
def post_job():
    """Handles job posting."""
    
    # Extract job details from the request data
    job_data = request.get_json()
    job_title = job_data.get('title')
    job_description = job_data.get('description')
    required_skills = job_data.get('required_skills')
    location = job_data.get('location')
    recruiter_id = job_data.get('recruiter_id')  
    salary = job_data.get('salary')
    job_type = job_data.get('job_type')  
    category = job_data.get('category')  
    featured = job_data.get('featured', False)  # Default to False
    expiry_date = job_data.get('expiry_date')  
    posted_by = job_data.get('posted_by') 

    # Validate the data (simple checks)
    if not job_title or not job_description or not required_skills or not location or not recruiter_id or not posted_by:
        return jsonify({"error": "Missing required fields"}), 400

    # Get a database connection
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500
    
    cursor = conn.cursor()

    # SQL query to insert the job posting into the database
    insert_query = """
        INSERT INTO jobs (
            title, description, required_skills, location, recruiter_id, salary, 
            job_type, category, featured, expiry_date, posted_by
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    cursor.execute(insert_query, (job_title, job_description, required_skills, location, 
                                  recruiter_id, salary, job_type, 
                                  category, featured, expiry_date, posted_by))
    conn.commit()

    # Close the connection
    cursor.close()
    conn.close()

    return jsonify({"message": "Job posted successfully!"}), 201


@app.route('/api/candidate_list', methods=['GET'])
def get_candidate_list():
    """Fetch the list of candidates with relevant details like match percentage and status."""
    
    # Get a database connection
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500
    
    cursor = conn.cursor()
    
    # Query to fetch candidates' details
    query = """
        SELECT u.name, j.title AS job_title, a.match_score, a.status, a.applied_date, r.resume_url, r.profile_url
        FROM applications a
        JOIN users u ON a.user_id = u.user_id
        JOIN jobs j ON a.job_id = j.job_id
        LEFT JOIN resumes r ON a.user_id = r.user_id
        ORDER BY a.applied_date DESC
    """
    
    cursor.execute(query)
    candidates = cursor.fetchall()

    # Close the connection
    cursor.close()
    conn.close()

    # Prepare the response data
    candidate_list = []
    for candidate in candidates:
        candidate_list.append({
            'name': candidate[0],
            'job_title': candidate[1],
            'match_percentage': candidate[2],
            'status': candidate[3],
            'applied_on': candidate[4].strftime('%Y-%m-%d %H:%M:%S'),
            'resume_url': candidate[5],
            'profile_url': candidate[6]
        })

    return jsonify({"candidates": candidate_list})


@app.route('/api/update_candidate_status', methods=['POST'])
def update_candidate_status():
    """Update the status of a candidate (Accepted, Pending, Rejected)."""
    data = request.json
    candidate_id = data.get('candidate_id')
    new_status = data.get('status')

    # Validate the data
    if not candidate_id or not new_status:
        return jsonify({"error": "Missing required fields"}), 400

    # Get a database connection
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500
    
    cursor = conn.cursor()

    # Update candidate status
    update_query = "UPDATE applications SET status = %s WHERE application_id = %s"
    cursor.execute(update_query, (new_status, candidate_id))
    conn.commit()

    # Close the connection
    cursor.close()
    conn.close()

    return jsonify({"message": f"Candidate status updated to {new_status}"}), 200


@app.route('/api/shortlisted_candidates', methods=['GET'])
def get_shortlisted_candidates():
    """Fetch shortlisted candidates with filtering options."""
    
    # Get filter options (status, default to 'All')
    status_filter = request.args.get('status', 'All')
    
    # Get a database connection
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500
    
    cursor = conn.cursor()

    # Query based on the status filter
    if status_filter == 'All':
        query = """
            SELECT u.name, j.title AS job_title, sc.shortlisted_date, sc.notes
            FROM shortlisted_candidates sc
            JOIN resumes r ON sc.resume_id = r.resume_id
            JOIN users u ON r.user_id = u.user_id
            JOIN jobs j ON sc.job_id = j.job_id
        """
    else:
        query = """
            SELECT u.name, j.title AS job_title, sc.shortlisted_date, sc.notes
            FROM shortlisted_candidates sc
            JOIN resumes r ON sc.resume_id = r.resume_id
            JOIN users u ON r.user_id = u.user_id
            JOIN jobs j ON sc.job_id = j.job_id
            WHERE sc.status = %s
        """
        cursor.execute(query, (status_filter,))
    
    cursor.execute(query)
    shortlisted_candidates = cursor.fetchall()

    # Close the connection
    cursor.close()
    conn.close()

    # Prepare the response data
    shortlisted_list = []
    for candidate in shortlisted_candidates:
        shortlisted_list.append({
            'name': candidate[0],
            'job_title': candidate[1],
            'shortlisted_date': candidate[2].strftime('%Y-%m-%d %H:%M:%S'),
            'notes': candidate[3]
        })

    return jsonify({"shortlisted_candidates": shortlisted_list})


@app.route('/api/shortlisted_statistics', methods=['GET'])
def get_shortlisted_statistics():
    """Get statistics about shortlisted candidates."""
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500
    
    cursor = conn.cursor()

    # Query to get the total candidates and the breakdown by status
    query = """
        SELECT 
            COUNT(*) AS total_candidates,
            SUM(CASE WHEN status = 'Accepted' THEN 1 ELSE 0 END) AS accepted,
            SUM(CASE WHEN status = 'Pending' THEN 1 ELSE 0 END) AS pending,
            SUM(CASE WHEN status = 'Rejected' THEN 1 ELSE 0 END) AS rejected
        FROM shortlisted_candidates
    """
    
    cursor.execute(query)
    stats = cursor.fetchone()

    # Close the connection
    cursor.close()
    conn.close()

    return jsonify({
        "total_candidates": stats[0],
        "accepted": stats[1],
        "pending": stats[2],
        "rejected": stats[3]
    })


@app.route('/api/export_candidates', methods=['GET'])
def export_candidates():
    """Export shortlisted candidates to a CSV file."""
    
    # Get a database connection
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500
    
    cursor = conn.cursor()
    query = "SELECT u.name, j.title AS job_title, sc.shortlisted_date, sc.notes FROM shortlisted_candidates sc JOIN resumes r ON sc.resume_id = r.resume_id JOIN users u ON r.user_id = u.user_id JOIN jobs j ON sc.job_id = j.job_id"
    cursor.execute(query)
    candidates = cursor.fetchall()

    # Create a CSV in memory
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'Job Title', 'Shortlisted Date', 'Notes'])
    
    for candidate in candidates:
        writer.writerow([candidate[0], candidate[1], candidate[2], candidate[3]])
    
    output.seek(0)

    # Close the connection
    cursor.close()
    conn.close()

    # Send CSV file as a response
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name="shortlisted_candidates.csv")

@app.route('/api/add_candidate', methods=['POST'])
def add_candidate():
    """Add a new candidate to the shortlist."""
    
    data = request.json
    name = data.get('name')
    job_title = data.get('job_title')
    match_percentage = data.get('match_percentage')
    status = data.get('status', 'Pending')  # Default to 'Pending' if no status is provided

    # Validate the data
    if not name or not job_title or not match_percentage:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Get a database connection
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500
    
    cursor = conn.cursor()

    # Insert the candidate into the database
    insert_query = """
        INSERT INTO shortlisted_candidates (name, job_title, match_percentage, status)
        VALUES (%s, %s, %s, %s)
    """
    cursor.execute(insert_query, (name, job_title, match_percentage, status))
    conn.commit()

    # Close the connection
    cursor.close()
    conn.close()

    return jsonify({"message": "Candidate added successfully!"}), 200


@app.route('/api/analysis_results', methods=['GET'])
def get_analysis_results():
    """Fetch the analysis results for candidates."""
    
    # Get filter options (status filter: default to 'All')
    status_filter = request.args.get('status', 'All')
    
    # Get a database connection
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500
    
    cursor = conn.cursor()

    # Query based on the status filter
    if status_filter == 'All':
        query = """
            SELECT u.user_id, u.name, r.job_title, ra.match_score, r.experience, r.education
            FROM users u
            JOIN resumes r ON u.user_id = r.user_id
            LEFT JOIN resume_analysis ra ON r.resume_id = ra.resume_id
        """
    else:
        query = """
            SELECT u.user_id, u.name, r.job_title, ra.match_score, r.experience, r.education
            FROM users u
            JOIN resumes r ON u.user_id = r.user_id
            LEFT JOIN resume_analysis ra ON r.resume_id = ra.resume_id
            WHERE ra.status = %s
        """
        cursor.execute(query, (status_filter,))
    
    cursor.execute(query)
    candidates = cursor.fetchall()

    # Close the connection
    cursor.close()
    conn.close()

    # Prepare the response data
    candidate_list = []
    for candidate in candidates:
        candidate_list.append({
            'id': candidate[0],               # user_id
            'name': candidate[1],             # candidate name
            'job_title': candidate[2],        # job title
            'match_score': candidate[3],      # match score from analysis
            'experience': candidate[4],       # experience
            'education_level': candidate[5],  # education level
            'status': status_filter           # status filter
        })

    return jsonify({"candidates": candidate_list})


@app.route('/api/candidate_details/<int:candidate_id>', methods=['GET'])
def get_candidate_details(candidate_id):
    """Fetch detailed profile of a candidate."""
    
    # Get a database connection
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500
    
    cursor = conn.cursor()

    # Query to get basic candidate details (now updated for your schema)
    query = """
        SELECT u.name, r.job_title, ra.match_score, r.experience, r.education
        FROM users u
        JOIN resumes r ON u.user_id = r.user_id
        LEFT JOIN resume_analysis ra ON r.resume_id = ra.resume_id
        WHERE u.user_id = %s
    """
    cursor.execute(query, (candidate_id,))
    candidate = cursor.fetchone()

    if candidate is None:
        return jsonify({"error": "Candidate not found"}), 404

    # Query to get skills for the candidate
    cursor.execute("SELECT skill FROM resume_skills WHERE resume_id = (SELECT resume_id FROM resumes WHERE user_id = %s)", (candidate_id,))
    skills = cursor.fetchall()

    # Prepare detailed response data
    detailed_profile = {
        'name': candidate[0],
        'job_title': candidate[1],
        'match_score': candidate[2],
        'experience': candidate[3],
        'education': candidate[4],
        'skills': [skill[0] for skill in skills]  # List of skills
    }

    # Close the connection
    cursor.close()
    conn.close()

    return jsonify(detailed_profile)


@app.route('/api/resume_analysis_results/<int:resume_id>', methods=['GET'])
def get_resume_analysis_results(resume_id):
    """Fetch detailed resume analysis results for the given resume ID."""
    
    # Get a database connection
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Failed to connect to the database"}), 500
    
    cursor = conn.cursor()

    # Fetch resume score and status
    cursor.execute("SELECT resume_score, status FROM resumes WHERE resume_id = %s", (resume_id,))
    resume = cursor.fetchone()
    if not resume:
        return jsonify({"error": "Resume not found"}), 404

    resume_score = resume[0]
    status = resume[1]

    # Fetch matched skills
    cursor.execute("SELECT skill FROM matched_skills WHERE resume_id = %s", (resume_id,))
    matched_skills = [skill[0] for skill in cursor.fetchall()]

    # Fetch missing skills
    cursor.execute("SELECT skill FROM missing_skills WHERE resume_id = %s", (resume_id,))
    missing_skills = [skill[0] for skill in cursor.fetchall()]

    # Fetch strengths dynamically
    cursor.execute("SELECT strength FROM strengths WHERE resume_id = %s", (resume_id,))
    strengths = [strength[0] for strength in cursor.fetchall()]

    # Fetch areas to improve dynamically
    cursor.execute("SELECT area FROM areas_to_improve WHERE resume_id = %s", (resume_id,))
    areas_to_improve = [area[0] for area in cursor.fetchall()]

    # Fetch resume breakdown (Education, Experience, Certifications, Projects)
    cursor.execute("SELECT section_name, completion_percentage FROM resume_breakdown WHERE resume_id = %s", (resume_id,))
    breakdown = cursor.fetchall()
    
    breakdown_data = {section[0]: section[1] for section in breakdown}

    # Fetch suggestions for improvement
    cursor.execute("SELECT suggestion FROM suggestions WHERE resume_id = %s", (resume_id,))
    suggestions = [suggestion[0] for suggestion in cursor.fetchall()]

    # Close the connection
    cursor.close()
    conn.close()

    # Prepare the response data
    analysis_results = {
        'resume_score': resume_score,
        'status': status,
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'strengths': strengths,
        'areas_to_improve': areas_to_improve,
        'resume_breakdown': breakdown_data,
        'suggestions': suggestions
    }

    return jsonify(analysis_results)



# ------------------- MAIN -------------------
if __name__ == "__main__":
    app.run(debug=True)