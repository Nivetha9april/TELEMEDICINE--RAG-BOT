# ==============================================================================
# MENTAL HEALTH SCREENING MICROSERVICE (mental_health_service.py)
# Port: 5003
# ==============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthScreening:
    """
    Mental health screening tool based on validated assessment scales
    Incorporates elements from PHQ-9, GAD-7, and stress assessment
    """
    
    def __init__(self):
        # Scoring thresholds for different aspects
        self.mood_thresholds = {
            (1, 3): {'level': 'severe_depression', 'score': 4},
            (3, 5): {'level': 'moderate_depression', 'score': 3},
            (5, 7): {'level': 'mild_depression', 'score': 2},
            (7, 8): {'level': 'normal_low', 'score': 1},
            (8, 11): {'level': 'normal', 'score': 0}
        }
        self.sleep_thresholds = {
            (1, 3): {'level': 'severe_sleep_issues', 'score': 3},
            (3, 5): {'level': 'moderate_sleep_issues', 'score': 2},
            (5, 7): {'level': 'mild_sleep_issues', 'score': 1},
            (7, 11): {'level': 'good_sleep', 'score': 0}
        }
        self.stress_thresholds = {
            (8, 11): {'level': 'severe_stress', 'score': 4},
            (6, 8): {'level': 'high_stress', 'score': 3},
            (4, 6): {'level': 'moderate_stress', 'score': 2},
            (2, 4): {'level': 'mild_stress', 'score': 1},
            (1, 2): {'level': 'low_stress', 'score': 0}
        }
        self.social_thresholds = {
            (7, 11): {'level': 'severe_isolation', 'score': 3},
            (5, 7): {'level': 'moderate_isolation', 'score': 2},
            (3, 5): {'level': 'mild_isolation', 'score': 1},
            (1, 3): {'level': 'well_connected', 'score': 0}
        }
    
    def get_score_for_range(self, value, thresholds):
        """Get score based on value and thresholds"""
        for (min_val, max_val), data in thresholds.items():
            if min_val <= value < max_val:
                return data
        return list(thresholds.values())[-1]
    
    def assess_mental_health(self, mood_score, sleep_score, stress_level, social_withdrawal):
        try:
            mood_data = self.get_score_for_range(mood_score, self.mood_thresholds)
            sleep_data = self.get_score_for_range(sleep_score, self.sleep_thresholds)
            stress_data = self.get_score_for_range(stress_level, self.stress_thresholds)
            social_data = self.get_score_for_range(social_withdrawal, self.social_thresholds)
            
            total_score = (
                mood_data['score'] * 0.35 +
                stress_data['score'] * 0.25 +
                sleep_data['score'] * 0.25 +
                social_data['score'] * 0.15
            )
            
            if total_score < 1.0:
                risk_level, overall_status, color = "LOW", "Good Mental Health", "green"
            elif total_score < 2.5:
                risk_level, overall_status, color = "MODERATE", "Some Concerns", "yellow"
            else:
                risk_level, overall_status, color = "HIGH", "Significant Concerns", "red"
            
            factor_analysis = {
                'mood': {
                    'input_score': mood_score,
                    'assessment': mood_data['level'],
                    'risk_contribution': mood_data['score'],
                    'status': 'Concerning' if mood_data['score'] >= 2 else 'Normal'
                },
                'sleep': {
                    'input_score': sleep_score,
                    'assessment': sleep_data['level'],
                    'risk_contribution': sleep_data['score'],
                    'status': 'Concerning' if sleep_data['score'] >= 2 else 'Normal'
                },
                'stress': {
                    'input_score': stress_level,
                    'assessment': stress_data['level'],
                    'risk_contribution': stress_data['score'],
                    'status': 'Concerning' if stress_data['score'] >= 3 else 'Normal'
                },
                'social_connection': {
                    'input_score': social_withdrawal,
                    'assessment': social_data['level'],
                    'risk_contribution': social_data['score'],
                    'status': 'Concerning' if social_data['score'] >= 2 else 'Normal'
                }
            }
            
            recommendations = self._generate_recommendations(
                mood_data, sleep_data, stress_data, social_data, risk_level
            )
            
            return {
                'overall_score': round(total_score, 2),
                'risk_level': risk_level,
                'overall_status': overall_status,
                'color_indicator': color,
                'factor_analysis': factor_analysis,
                'recommendations': recommendations,
                'screening_date': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in mental health assessment: {str(e)}")
            raise
    
    def _generate_recommendations(self, mood_data, sleep_data, stress_data, social_data, risk_level):
        recs = ["Practice mindfulness daily", "Maintain regular routines"]
        
        if mood_data['score'] >= 2:
            recs += ["Try journaling", "Engage in enjoyable activities"]
        if sleep_data['score'] >= 1:
            recs += ["Set consistent sleep schedule", "Limit screens before bed"]
        if stress_data['score'] >= 2:
            recs += ["Deep breathing", "Regular exercise for stress relief"]
        if social_data['score'] >= 1:
            recs += ["Connect with friends/family", "Join social activities"]
        
        if risk_level == "HIGH":
            recs += [
                "URGENT: Seek professional mental health support",
                "Contact healthcare provider immediately",
                "Consider therapy or counseling"
            ]
        elif risk_level == "MODERATE":
            recs += [
                "Consider talking with a counselor",
                "Monitor symptoms closely"
            ]
        return recs

# API endpoint
@app.route("/mental_health", methods=["POST"])
def mental_health_api():
    data = request.json
    required = ["mood_score", "sleep_score", "stress_level", "social_withdrawal"]
    if not all(k in data for k in required):
        return jsonify({"error": "Missing required fields"}), 400
    
    mh = MentalHealthScreening()
    result = mh.assess_mental_health(
        data["mood_score"],
        data["sleep_score"],
        data["stress_level"],
        data["social_withdrawal"]
    )
    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5003, debug=True)
