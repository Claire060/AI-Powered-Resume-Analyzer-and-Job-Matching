USE cvision;

CREATE TABLE users (
         user_id INT AUTO_INCREMENT PRIMARY KEY,
         name VARCHAR(100) NOT NULL,
		 email VARCHAR(100) UNIQUE NOT NULL,
	     password VARCHAR(255) NOT NULL,
		 role ENUM('job_seeker','recruiter') NOT NULL
	   );

-- Password Reset Requests Table (for OTPs)
CREATE TABLE password_reset_requests (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    otp VARCHAR(6) NOT NULL,  -- 6-digit OTP
    expiration_time DATETIME NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE jobs (
        job_id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(150) NOT NULL,
		description TEXT,
		required_skills TEXT,
		location VARCHAR(100),
		recruiter_id INT,
        FOREIGN KEY (recruiter_id) REFERENCES users(user_id)
		);

ALTER TABLE users ADD COLUMN company_name VARCHAR(150);

ALTER TABLE jobs ADD COLUMN salary VARCHAR(100);
 
CREATE TABLE resumes (
        resume_id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        personal_info TEXT,
        skills TEXT,
        experience TEXT,
        education TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
		);

CREATE TABLE resume_analysis_results (
    analysis_id INT AUTO_INCREMENT PRIMARY KEY,
    resume_id INT NOT NULL,
    analyzer_type ENUM('ResumeAnalyzer', 'RecruiterAnalyzer'),
    score INT,
    skills TEXT,
    strengths TEXT,
    areas_to_improve TEXT,
    suggestions TEXT,
    FOREIGN KEY (resume_id) REFERENCES resumes(resume_id)
);

-- Table for storing resume analysis results by recruiters
CREATE TABLE resume_analysis (
    analysis_id INT AUTO_INCREMENT PRIMARY KEY,
    resume_id INT,
    recruiter_id INT,
    match_score INT,
    analysis_date DATETIME,
    status ENUM('processed', 'error') NOT NULL DEFAULT 'processed',
    notes TEXT,
    FOREIGN KEY (resume_id) REFERENCES resumes(resume_id),
    FOREIGN KEY (recruiter_id) REFERENCES users(user_id)
);

-- Table for shortlisted candidates
CREATE TABLE shortlisted_candidates (
    shortlist_id INT AUTO_INCREMENT PRIMARY KEY,
    resume_id INT,
    recruiter_id INT,
    shortlisted_date DATETIME,
    notes TEXT,
    FOREIGN KEY (resume_id) REFERENCES resumes(resume_id),
    FOREIGN KEY (recruiter_id) REFERENCES users(user_id),
    UNIQUE KEY unique_shortlist (resume_id, recruiter_id)
);


 CREATE TABLE reports (
        report_id INT AUTO_INCREMENT PRIMARY KEY,
        resume_id INT,
        score INT,
        graph_data TEXT,
        FOREIGN KEY (resume_id) REFERENCES resumes(resume_id)
     );

 CREATE TABLE resume_analyzer (
         analyzer_id INT AUTO_INCREMENT PRIMARY KEY,
         resume_id INT,
         model_name VARCHAR(100),
         algorithm_type VARCHAR(100),
         FOREIGN KEY (resume_id) REFERENCES resumes(resume_id)
     );

 CREATE TABLE job_matcher (
         matcher_id INT AUTO_INCREMENT PRIMARY KEY,
         algorithm_type VARCHAR(100),
         resume_id INT,
         job_id INT,
		 match_score INT,
         FOREIGN KEY (resume_id) REFERENCES resumes(resume_id),
         FOREIGN KEY (job_id) REFERENCES jobs(job_id)
     );


 CREATE TABLE applications (
     application_id INT AUTO_INCREMENT PRIMARY KEY,
     user_id INT NOT NULL,
     job_id INT NOT NULL,
     applied_date DATETIME DEFAULT CURRENT_TIMESTAMP,
     status ENUM('Pending', 'Accepted', 'Rejected') DEFAULT 'Pending',
     FOREIGN KEY (user_id) REFERENCES users(user_id),
     FOREIGN KEY (job_id) REFERENCES jobs(job_id)
     );
     
-- Add match_score column to applications table 
ALTER TABLE applications ADD COLUMN match_score INT DEFAULT 0; 
-- Add notes column to applications table 
ALTER TABLE applications ADD COLUMN notes TEXT;

ALTER TABLE resumes
    ADD COLUMN upload_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ADD COLUMN score INT;
    
CREATE TABLE resume_skills (
    id INT AUTO_INCREMENT PRIMARY KEY,
    resume_id INT NOT NULL,
    skill VARCHAR(100) NOT NULL,
    FOREIGN KEY (resume_id) REFERENCES resumes(resume_id)
);

INSERT INTO jobs (title, description, required_skills, location, recruiter_id)
VALUES
    ('Software Developer', 'Responsible for developing and maintaining software applications.', 'Python, Java, SQL', 'Remote', 1),
    ('Data Scientist', 'Analyzes and interprets complex data to help organizations make data-driven decisions.', 'Python, R, SQL, Machine Learning', 'New York, USA', 2),
    ('Frontend Developer', 'Responsible for building and maintaining the user-facing part of websites and applications.', 'HTML, CSS, JavaScript, React', 'San Francisco, USA', 3),
    ('DevOps Engineer', 'Works to streamline operations and development, focusing on automation, CI/CD, and cloud infrastructure.', 'AWS, Docker, Kubernetes, Jenkins', 'Remote', 4),
    ('UX/UI Designer', 'Designs user interfaces and ensures user experiences are smooth and intuitive.', 'Figma, Sketch, Adobe XD', 'London, UK', 5),
    ('Product Manager', 'Oversees product development and ensures the product meets user needs and business goals.', 'Project Management, Agile, Communication', 'Berlin, Germany', 6);

INSERT INTO jobs (title, description, required_skills, location, recruiter_id)
VALUES
    ('Senior Full Stack Developer', 'We are looking for an experienced Full Stack Developer to join our dynamic team. You will be responsible for developing and maintaining web applications using modern technologies.', 'JavaScript, React, Node.js, Python, AWS, MongoDB', 'San Francisco, CA', 7),
    ('Frontend Engineer', 'Join our frontend team to build beautiful and responsive user interfaces. Experience with modern JavaScript frameworks required.', 'React, TypeScript, HTML, CSS, Redux', 'New York, NY', 8),
    ('Backend Developer', 'Looking for a backend developer to work on our scalable cloud infrastructure. Strong experience with Node.js and databases required.', 'Node.js, Python, PostgreSQL, AWS, Docker', 'Austin, TX', 9),
    ('Full Stack Developer', 'Early-stage startup looking for a versatile full-stack developer to help build our product from the ground up.', 'JavaScript, React, Node.js, MongoDB, Express', 'Remote', 10),
    ('Software Engineer', 'Join our enterprise software team to develop mission-critical applications for Fortune 500 companies.', 'Java, Spring Boot, SQL, AWS, Microservices', 'Chicago, IL', 11),
    ('React Developer', 'Frontend developer position focusing on React applications. Experience with state management and modern build tools required.', 'React, JavaScript, Redux, Webpack, HTML5', 'Boston, MA', 12),
    ('DevOps Engineer', 'DevOps engineer needed to manage our cloud infrastructure and CI/CD pipelines.', 'AWS, Docker, Kubernetes, CI/CD, Linux', 'Seattle, WA', 13),
    ('Mobile App Developer', 'Develop cross-platform mobile applications using React Native and modern mobile technologies.', 'React Native, JavaScript, iOS, Android, Firebase', 'Los Angeles, CA', 14),
    ('Data Engineer', 'Build and maintain data pipelines for our analytics platform. Experience with big data technologies required.', 'Python, SQL, Spark, AWS, Data Pipelines', 'Denver, CO', 15);


INSERT INTO resumes (user_id, personal_info, skills, experience, education, upload_date, score)
VALUES
    (1, 'John Doe, experienced software developer with 5 years of experience in Python and Java.', 'Python, Java, SQL', 'Worked at ABC Corp. for 3 years', 'B.Sc. in Computer Science', '2025-01-01 08:00:00', 85),
    (2, 'Jane Smith, data scientist with expertise in machine learning and data analysis.', 'Python, R, SQL, Machine Learning', 'Worked at XYZ Inc. for 4 years', 'M.Sc. in Data Science', '2025-02-01 08:00:00', 90),
    (3, 'Alex Johnson, frontend developer skilled in React and modern web technologies.', 'HTML, CSS, JavaScript, React', 'Worked at WebWorks Ltd. for 2 years', 'B.A. in Web Development', '2025-03-01 08:00:00', 78),
    (4, 'Emily Davis, experienced DevOps engineer with expertise in AWS and Kubernetes.', 'AWS, Docker, Kubernetes, Jenkins', 'Worked at CloudTech for 3 years', 'M.Sc. in Computer Engineering', '2025-04-01 08:00:00', 88),
    (5, 'Chris Lee, UX/UI designer with strong skills in Figma and Adobe XD.', 'Figma, Sketch, Adobe XD', 'Worked at Design Studio for 2 years', 'B.A. in Graphic Design', '2025-05-01 08:00:00', 76),
    (6, 'Sarah Brown, product manager with expertise in agile methodologies.', 'Project Management, Agile, Communication', 'Worked at Innovate Ltd. for 3 years', 'MBA in Product Management', '2025-06-01 08:00:00', 82);

INSERT INTO resume_skills (resume_id, skill)
VALUES
    (1, 'Python'),
    (1, 'Java'),
    (1, 'SQL'),
    (2, 'Python'),
    (2, 'R'),
    (2, 'SQL'),
    (2, 'Machine Learning'),
    (3, 'HTML'),
    (3, 'CSS'),
    (3, 'JavaScript'),
    (3, 'React'),
    (4, 'AWS'),
    (4, 'Docker'),
    (4, 'Kubernetes'),
    (4, 'Jenkins'),
    (5, 'Figma'),
    (5, 'Sketch'),
    (5, 'Adobe XD'),
    (6, 'Project Management'),
    (6, 'Agile'),
    (6, 'Communication');

INSERT INTO job_matcher (algorithm_type, resume_id, job_id, match_score)
VALUES
    ('Algorithm 1', 1, 1, 85),  
    ('Algorithm 1', 2, 2, 90),   
    ('Algorithm 1', 3, 3, 78),   
    ('Algorithm 1', 4, 4, 88),   
    ('Algorithm 1', 5, 5, 76),   
    ('Algorithm 1', 6, 6, 82);   

INSERT INTO users (user_id, name, email, password, role)
VALUES
    (2, 'John Doe', 'john.doe@example.com', '$pbkdf2-sha256$29000$XjJsyWpvCl0LO9nX35f2Ow$kdd8.sOsqgHvqRHXzLejJbyzwjsJf1gtHUMkprCvEkQ=', 'job_seeker'),
    (3, 'Jane Smith', 'jane.smith@example.com', '$pbkdf2-sha256$29000$XjJsyWpvCl0LO9nX35f2Ow$kdd8.sOsqgHvqRHXzLejJbyzwjsJf1gtHUMkprCvEkQ=', 'job_seeker'),
    (4, 'Alex Johnson', 'alex.johnson@example.com', '$pbkdf2-sha256$29000$XjJsyWpvCl0LO9nX35f2Ow$kdd8.sOsqgHvqRHXzLejJbyzwjsJf1gtHUMkprCvEkQ=', 'job_seeker'),
    (5, 'Emily Davis', 'emily.davis@example.com', '$pbkdf2-sha256$29000$XjJsyWpvCl0LO9nX35f2Ow$kdd8.sOsqgHvqRHXzLejJbyzwjsJf1gtHUMkprCvEkQ=', 'job_seeker'),
    (6, 'Chris Lee', 'chris.lee@example.com', '$pbkdf2-sha256$29000$XjJsyWpvCl0LO9nX35f2Ow$kdd8.sOsqgHvqRHXzLejJbyzwjsJf1gtHUMkprCvEkQ=', 'job_seeker'),
    (7, 'Sarah Brown', 'sarah.brown@example.com', '$pbkdf2-sha256$29000$XjJsyWpvCl0LO9nX35f2Ow$kdd8.sOsqgHvqRHXzLejJbyzwjsJf1gtHUMkprCvEkQ=', 'job_seeker');


INSERT INTO users (user_id, name, email, password, role)
VALUES
    (8, 'Jacob Taylor', 'jacob.taylor@example.com', '$pbkdf2-sha256$29000$XjJsyWpvCl0LO9nX35f2Ow$kdd8.sOsqgHvqRHXzLejJbyzwjsJf1gtHUMkprCvEkQ=', 'job_seeker'),
    (9, 'Lisa Thompson', 'lisa.thompson@example.com', '$pbkdf2-sha256$29000$XjJsyWpvCl0LO9nX35f2Ow$kdd8.sOsqgHvqRHXzLejJbyzwjsJf1gtHUMkprCvEkQ=', 'job_seeker'),
    (10, 'Emily Wilson', 'emily.wilson@example.com', '$pbkdf2-sha256$29000$XjJsyWpvCl0LO9nX35f2Ow$kdd8.sOsqgHvqRHXzLejJbyzwjsJf1gtHUMkprCvEkQ=', 'job_seeker'),
    (11, 'Amanda Lee', 'amanda.lee@example.com', '$pbkdf2-sha256$29000$XjJsyWpvCl0LO9nX35f2Ow$kdd8.sOsqgHvqRHXzLejJbyzwjsJf1gtHUMkprCvEkQ=', 'job_seeker'),
    (12, 'Olivia Martinez', 'olivia.martinez@example.com', '$pbkdf2-sha256$29000$XjJsyWpvCl0LO9nX35f2Ow$kdd8.sOsqgHvqRHXzLejJbyzwjsJf1gtHUMkprCvEkQ=', 'job_seeker'),
    (13, 'Daniel Kim', 'daniel.kim@example.com', '$pbkdf2-sha256$29000$XjJsyWpvCl0LO9nX35f2Ow$kdd8.sOsqgHvqRHXzLejJbyzwjsJf1gtHUMkprCvEkQ=', 'job_seeker'),
    (14, 'Emily Rodriguez', 'emily.rodriguez@example.com', '$pbkdf2-sha256$29000$XjJsyWpvCl0LO9nX35f2Ow$kdd8.sOsqgHvqRHXzLejJbyzwjsJf1gtHUMkprCvEkQ=', 'job_seeker'),
    (15, 'Michael Chen', 'michael.chen@example.com', '$pbkdf2-sha256$29000$XjJsyWpvCl0LO9nX35f2Ow$kdd8.sOsqgHvqRHXzLejJbyzwjsJf1gtHUMkprCvEkQ=', 'job_seeker');

-- Sample data for applications table
INSERT INTO applications (user_id, job_id, applied_date, status, match_score, notes) VALUES
(1, 1, '2023-11-01 09:30:00', 'Pending', 80, 'Strong programming skills in Python and Java.'),
(2, 2, '2023-11-02 11:00:00', 'Accepted', 90, 'Great knowledge of data analytics and machine learning.'),
(3, 1, '2023-11-05 14:00:00', 'Rejected', 50, 'Lacks relevant experience in software development.'),
(1, 3, '2023-11-06 16:30:00', 'Pending', 70, 'Good management skills, but lacks some technical expertise.'),
(2, 3, '2023-11-10 10:15:00', 'Rejected', 60, 'Does not have project management experience.');


