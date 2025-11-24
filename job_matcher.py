import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Any
import mysql.connector
from mysql.connector import Error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobMatchingService:
    def __init__(self, model_path: str, db_config):
        self.model_path = model_path
        self.db_config = db_config
        self.model = self._load_model()
        logger.info("Job Matching Service initialized")
    
    def _load_model(self):
        """Load the sentence transformer model with fallbacks"""
        try:
            if os.path.exists(self.model_path) and os.path.isdir(self.model_path):
                logger.info(f"Loading custom model from: {self.model_path}")
                model = SentenceTransformer(self.model_path)
            else:
                logger.info(f"Using pretrained model: {self.model_path}")
                model = SentenceTransformer(self.model_path)
            
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_path}: {e}")
            logger.info("Loading default model: all-MiniLM-L6-v2")
            return SentenceTransformer('all-MiniLM-L6-v2')
    
    def get_db_connection(self):
        """Get database connection"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            return connection
        except Error as e:
            logger.error(f"Database connection error: {e}")
            return None
    
    def encode_texts(self, texts: List[str]):
        """Encode texts to embeddings"""
        return self.model.encode(texts, convert_to_tensor=False)
    
    def calculate_cosine_similarity(self, text1: str, texts2: List[str]):
        """Calculate cosine similarity between one text and multiple texts"""
        # Combine all texts for encoding
        all_texts = [text1] + texts2
        embeddings = self.encode_texts(all_texts)
        
        # Calculate similarities
        similarities = cosine_similarity([embeddings[0]], embeddings[1:])
        return similarities[0]
    
    def _extract_skills(self, text: str):
        """Extract skills from text (basic implementation)"""
        common_skills = ['python', 'javascript', 'java', 'html', 'css', 'react', 'node', 'sql', 
                        'mongodb', 'aws', 'docker', 'git', 'machine learning', 'ai', 'data analysis',
                        'fastapi', 'flask', 'django', 'vue', 'angular', 'typescript', 'rest api',
                        'linux', 'windows', 'agile', 'scrum', 'devops', 'ci/cd']
        
        if not text:
            return []
        
        text_lower = text.lower()
        found_skills = [skill for skill in common_skills if skill in text_lower]
        return found_skills
    
    def _calculate_skill_overlap(self, candidate_skills: List[str], job_skills: List[str]):
        """Calculate skill overlap percentage"""
        if not job_skills:
            return 0.0
        
        common_skills = set(candidate_skills) & set(job_skills)
        return len(common_skills) / len(job_skills)
    
    def _match_experience_level(self, candidate_profile: str, job_requirements: str):
        """Basic experience level matching"""
        profile_lower = candidate_profile.lower()
        requirements_lower = job_requirements.lower()
        
        # Check for experience keywords
        exp_keywords = ['senior', 'lead', 'manager', 'director', 'head']
        junior_keywords = ['junior', 'entry', 'trainee', 'intern', 'fresh grad']
        
        has_senior_keywords = any(keyword in requirements_lower for keyword in exp_keywords)
        has_junior_keywords = any(keyword in requirements_lower for keyword in junior_keywords)
        
        # Simple matching logic
        if has_senior_keywords and 'year' in profile_lower:
            return 0.7  # Moderate match for senior roles
        elif has_junior_keywords:
            return 0.9  # Good match for junior roles
        else:
            return 0.8  # Default match
    
    def calculate_match_score(self, candidate_profile: str, job_data: dict):
        """Calculate match score between candidate profile and job"""
        try:
            # Extract key components
            candidate_skills = self._extract_skills(candidate_profile)
            
            # Combine job skills and requirements
            job_skills_text = job_data.get('required_skills', '') 
            if job_data.get('skills'):
                job_skills_text += " " + job_data['skills']
                
            job_skills = self._extract_skills(job_skills_text)
            
            # Skill overlap score (40% weight)
            skill_overlap = self._calculate_skill_overlap(candidate_skills, job_skills)
            
            # Text similarity score (40% weight)
            job_text = f"{job_data['title']} {job_data['description']} {job_data.get('required_skills', '')}"
            text_similarities = self.calculate_cosine_similarity(candidate_profile, [job_text])
            text_similarity = text_similarities[0] if len(text_similarities) > 0 else 0.0
            
            # Experience level matching (20% weight)
            experience_match = self._match_experience_level(candidate_profile, job_data.get('required_skills', ''))
            
            # Combined score
            final_score = (skill_overlap * 0.4) + (text_similarity * 0.4) + (experience_match * 0.2)
            
            # Scale and cap the score
            scaled_score = min(1.0, final_score * 1.3)
            
            logger.debug(f"Match scores for {job_data['title']}: "
                        f"skills={skill_overlap:.2f}, text={text_similarity:.2f}, "
                        f"exp={experience_match:.2f}, final={scaled_score:.2f}")
            
            return scaled_score
            
        except Exception as e:
            logger.error(f"Error calculating match score: {e}")
            return 0.0
    
    def get_jobs_from_database(self):
        """Fetch all jobs from cvision database"""
        try:
            # Ensure that the database connection is valid
            connection = self.get_db_connection()
            if not connection:
                logger.error("Database connection not available.")
                return []  # No jobs available if no connection

            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM jobs")  # Query to fetch jobs from your actual job table
            jobs = cursor.fetchall()
            cursor.close()
            
            return jobs  # Return the jobs directly without demo jobs
            
        except Exception as e:
            logger.error(f"Failed to fetch jobs: {e}")
            return []  # Return an empty list if there's an error
    
    def get_top_matches(self, user_id: int = None, candidate_profile: str = None, top_k: int = 10):
        """Get top matching jobs for a candidate"""
        try:
            # Use user profile or candidate profile to fetch job matches
            if candidate_profile:
                profile = candidate_profile
            else:
                # Fetch the profile of the user from the database
                profile = self.get_user_profile(user_id)  # Assume you have this function

            # Fetch job data from the database
            jobs = self.get_jobs_from_database()  # This will return real jobs

            matches = []
            for job in jobs:
                # Calculate the match score using your matching logic
                match_score = self.calculate_match_score(profile, job)
                match_data = job.copy()
                match_data["match_score"] = match_score
                matches.append(match_data)

            # Sort the matches by score
            matches.sort(key=lambda x: x["match_score"], reverse=True)
            return matches[:top_k]  # Return the top_k matches
        
        except Exception as e:
            logger.error(f"Error fetching job matches: {e}")
            return []  # Return an empty list if an error occurs

    
    def match_single_candidate_to_jobs(self, candidate_profile: str, top_k: int = 10):
        """Match a single candidate profile to jobs"""
        return self.get_top_matches(candidate_profile=candidate_profile, top_k=top_k)