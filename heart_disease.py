# ==============================================================================
# HEART DISEASE RISK ASSESSMENT MICROSERVICE (heart_service.py)
# Port: 5002
# ==============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeartDiseaseRiskAssessment:
    """
    Heart disease risk assessment based on Framingham Risk Score and clinical parameters
    """
    
    def __init__(self):
        # Risk scoring based on clinical guidelines
        self.age_points = {
            'male': {
                (20, 35): -9, (35, 40): -4, (40, 45): 0, (45, 50): 3,
                (50, 55): 6, (55, 60): 8, (60, 65): 10, (65, 70): 11,
                (70, 75): 12, (75, 120): 13
            },
            'female': {
                (20, 35): -7, (35, 40): -3, (40, 45): 0, (45, 50): 3,
                (50, 55): 6, (55, 60): 8, (60, 65): 10, (65, 70): 12,
                (70, 75): 14, (75, 120): 16
            }
        }
        
        self.cholesterol_points = {
            (0, 160): 0, (160, 200): 4, (200, 240): 7,
            (240, 280): 9, (280, 1000): 11
        }
        
        self.bp_points = {
            (0, 120): 0, (120, 130): 1, (130, 140): 2,
            (140, 160): 3, (160, 1000): 4
        }
    
    def calculate_age_points(self, age, gender='male'):
        """Calculate age-based risk points"""
        age_scores = self.age_points.get(gender, self.age_points['male'])
        for (min_age, max_age), points in age_scores.items():
            if min_age <= age < max_age:
                return points
        return 13  # Maximum points for very old age
    
    def calculate_cholesterol_points(self, cholesterol):
        """Calculate cholesterol-based risk points"""
        for (min_chol, max_chol), points in self.cholesterol_points.items():
            if min_chol <= cholesterol < max_chol:
                return points
        return 11  # Maximum points
    
    def calculate_bp_points(self, systolic_bp):
        """Calculate blood pressure risk points"""
        for (min_bp, max_bp), points in self.bp_points.items():
            if min_bp <= systolic_bp < max_bp:
                return points
        return 4  # Maximum points
    
    def assess_heart_risk(self, age, cholesterol, bp_systolic, smoking=False, 
                         family_history=False, gender='male'):
        """
        Comprehensive heart disease risk assessment
        """
        try:
            # Calculate base risk points
            age_points = self.calculate_age_points(age, gender)
            chol_points = self.calculate_cholesterol_points(cholesterol)
            bp_points = self.calculate_bp_points(bp_systolic)
            
            # Additional risk factors
            smoking_points = 8 if smoking else 0
            family_points = 5 if family_history else 0
            
            # Total risk score
            total_points = age_points + chol_points + bp_points + smoking_points + family_points
            
            # Convert to risk percentage (simplified Framingham equation)
            if total_points < 0:
                risk_percentage = 1
            elif total_points <= 4:
                risk_percentage = 1
            elif total_points <= 6:
                risk_percentage = 2
            elif total_points <= 8:
                risk_percentage = 3
            elif total_points <= 10:
                risk_percentage = 4
            elif total_points <= 12:
                risk_percentage = 6
            elif total_points <= 14:
                risk_percentage = 8
            elif total_points <= 16:
                risk_percentage = 11
            elif total_points <= 18:
                risk_percentage = 14
            elif total_points <= 20:
                risk_percentage = 17
            else:
                risk_percentage = min(30, 17 + (total_points - 20) * 2)
            
            # Risk categorization
            if risk_percentage < 7.5:
                risk_level = "LOW"
                color = "green"
            elif risk_percentage < 20:
                risk_level = "MODERATE"
                color = "yellow"
            else:
                risk_level = "HIGH"
                color = "red"
            
            # Factor analysis
            factor_analysis = {
                'age': {
                    'value': age,
                    'points': age_points,
                    'status': 'Low Risk' if age < 45 else 'Increased Risk'
                },
                'cholesterol': {
                    'value': cholesterol,
                    'points': chol_points,
                    'category': self._get_cholesterol_category(cholesterol),
                    'status': 'Normal' if cholesterol < 200 else 'Elevated'
                },
                'blood_pressure': {
                    'value': bp_systolic,
                    'points': bp_points,
                    'category': self._get_bp_category(bp_systolic),
                    'status': 'Normal' if bp_systolic < 120 else 'Elevated'
                },
                'smoking': {
                    'status': smoking,
                    'points': smoking_points,
                    'impact': 'High' if smoking else 'None'
                },
                'family_history': {
                    'status': family_history,
                    'points': family_points,
                    'impact': 'Moderate' if family_history else 'None'
                }
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                age, cholesterol, bp_systolic, smoking, family_history, risk_level
            )
            
            return {
                'total_risk_points': total_points,
                'risk_percentage': risk_percentage,
                'risk_level': risk_level,
                'color_indicator': color,
                'factor_analysis': factor_analysis,
                'recommendations': recommendations,
                'assessment_date': datetime.now().isoformat(),
                'next_assessment': self._get_next_assessment_date(risk_level),
                'ten_year_risk': f"{risk_percentage}% chance of heart disease in next 10 years"
            }
            
        except Exception as e:
            logger.error(f"Error in heart risk assessment: {str(e)}")
            raise
    
    def _get_cholesterol_category(self, cholesterol):
        """Get cholesterol category"""
        if cholesterol < 200:
            return "Optimal"
        elif cholesterol < 240:
            return "Borderline High"
        else:
            return "High"
    
    def _get_bp_category(self, bp):
        """Get blood pressure category"""
        if bp < 120:
            return "Normal"
        elif bp < 130:
            return "Elevated"
        elif bp < 140:
            return "Stage 1 Hypertension"
        else:
            return "Stage 2 Hypertension"
    
    def _generate_recommendations(self, age, cholesterol, bp_systolic, smoking, 
                                family_history, risk_level):
        """Generate personalized heart health recommendations"""
        recommendations = []
        
        # General heart health recommendations
        recommendations.append("Follow a heart-healthy diet rich in fruits, vegetables, and whole grains")
        recommendations.append("Engage in regular physical activity (150 min/week moderate or 75 min/week vigorous)")
        
        # Cholesterol-specific recommendations
        if cholesterol >= 200:
            recommendations.append("Reduce saturated fats and trans fats in diet")
            recommendations.append("Consider cholesterol-lowering medications if prescribed")
            recommendations.append("Include omega-3 fatty acids in diet")
        
        # Blood pressure recommendations
        if bp_systolic >= 130:
            recommendations.append("Reduce sodium intake to less than 2300mg daily")
            recommendations.append("Monitor blood pressure regularly at home")
            recommendations.append("Practice stress reduction techniques")
        
        # Smoking cessation
        if smoking:
            recommendations.append("URGENT: Quit smoking immediately - single most important step")
            recommendations.append("Seek smoking cessation support programs")
            recommendations.append("Consider nicotine replacement therapy")
        
        # Family history considerations
        if family_history:
            recommendations.append("Discuss family history details with cardiologist")
            recommendations.append("Consider earlier and more frequent screenings")
            recommendations.append("Be extra vigilant about modifiable risk factors")
        
        # Age-related recommendations
        if age >= 50:
            recommendations.append("Consider aspirin therapy (consult physician)")
            recommendations.append("Regular cardiac check-ups annually")
        
        # Risk-level specific recommendations
        if risk_level == "HIGH":
            recommendations.append("URGENT: Schedule immediate consultation with cardiologist")
            recommendations.append("Consider stress testing and advanced cardiac imaging")
            recommendations.append("Discuss preventive medications (statins, ACE inhibitors)")
            recommendations.append("Implement intensive lifestyle modifications")
        elif risk_level == "MODERATE":
            recommendations.append("Schedule appointment with primary care physician")
            recommendations.append("Consider lipid panel and cardiac risk assessment")
            recommendations.append("Implement lifestyle changes promptly")
        
        return recommendations
    
    def _get_next_assessment_date(self, risk_level):
        """Recommend next assessment interval"""
        if risk_level == "HIGH":
            return "3-6 months"
        elif risk_level == "MODERATE":
            return "6-12 months"
        else:
            return "1-2 years"

# Initialize assessor
assessor = HeartDiseaseRiskAssessment()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'heart_disease_risk_assessment',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/assess_risk', methods=['POST'])
def assess_risk():
    """Assess heart disease risk based on input parameters"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['age', 'cholesterol', 'bp_systolic']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract parameters
        age = int(data['age'])
        cholesterol = int(data['cholesterol'])
        bp_systolic = int(data['bp_systolic'])
        smoking = data.get('smoking', False)
        family_history = data.get('family_history', False)
        gender = data.get('gender', 'male').lower()
        
        # Validate ranges
        if not (18 <= age <= 120):
            return jsonify({'error': 'Age must be between 18 and 120'}), 400
        if not (100 <= cholesterol <= 400):
            return jsonify({'error': 'Cholesterol must be between 100 and 400 mg/dL'}), 400
        if not (80 <= bp_systolic <= 200):
            return jsonify({'error': 'Systolic BP must be between 80 and 200 mmHg'}), 400
        if gender not in ['male', 'female']:
            gender = 'male'  # Default
        
        # Assess risk
        result = assessor.assess_heart_risk(
            age, cholesterol, bp_systolic, smoking, family_history, gender
        )
        
        logger.info(f"Heart risk assessment completed for age:{age}, cholesterol:{cholesterol}, "
                   f"bp:{bp_systolic}, smoking:{smoking}, family_history:{family_history}")
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input values: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in assess_risk: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/risk_factors', methods=['GET'])
def get_risk_factors():
    """Get information about heart disease risk factors"""
    return jsonify({
        'risk_factors': {
            'age': {
                'description': 'Age increases cardiovascular risk, especially after 45 (men) or 55 (women)',
                'modifiable': False
            },
            'cholesterol': {
                'description': 'Total cholesterol levels',
                'categories': {
                    'optimal': '<200 mg/dL',
                    'borderline_high': '200-239 mg/dL',
                    'high': '240+ mg/dL'
                },
                'modifiable': True
            },
            'blood_pressure': {
                'description': 'Systolic blood pressure',
                'categories': {
                    'normal': '<120 mmHg',
                    'elevated': '120-129 mmHg',
                    'stage_1': '130-139 mmHg',
                    'stage_2': '140+ mmHg'
                },
                'modifiable': True
            },
            'smoking': {
                'description': 'Tobacco use significantly increases heart disease risk',
                'impact': 'High',
                'modifiable': True
            },
            'family_history': {
                'description': 'Family history of premature heart disease',
                'impact': 'Moderate',
                'modifiable': False
            }
        },
        'prevention_tips': [
            "Maintain healthy diet low in saturated fats",
            "Exercise regularly (150 minutes moderate activity/week)",
            "Don't smoke or quit if you do",
            "Manage stress effectively",
            "Maintain healthy weight",
            "Regular health screenings"
        ]
    })

if __name__ == '__main__':
    print("Starting Heart Disease Risk Assessment Service on port 5002...")
    app.run(debug=True, port=5002, host='0.0.0.0')