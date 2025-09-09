# ==============================================================================
# DIABETES RISK CALCULATOR MICROSERVICE (diabetes_service.py)
# Run: python diabetes_service.py
# Endpoint: http://localhost:5001/assess
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

class DiabetesRiskCalculator:
    """
    Diabetes risk assessment based on clinical parameters
    """

    def __init__(self):
        # Risk factors and their weights
        self.age_weights = {
            (18, 30): 0.1,
            (30, 40): 0.2,
            (40, 50): 0.4,
            (50, 60): 0.6,
            (60, 70): 0.8,
            (70, 120): 1.0
        }

    def calculate_age_risk(self, age):
        """Calculate age-based risk factor"""
        for (min_age, max_age), weight in self.age_weights.items():
            if min_age <= age < max_age:
                return weight
        return 1.0

    def calculate_bmi_risk(self, bmi):
        """Calculate BMI-based risk factor"""
        if bmi < 18.5:
            return 0.1
        elif 18.5 <= bmi < 25:
            return 0.2
        elif 25 <= bmi < 30:
            return 0.5
        elif 30 <= bmi < 35:
            return 0.7
        elif 35 <= bmi < 40:
            return 0.9
        else:
            return 1.0

    def calculate_glucose_risk(self, glucose):
        """Calculate glucose-based risk factor"""
        if glucose < 100:
            return 0.1
        elif 100 <= glucose < 126:
            return 0.6  # Prediabetes
        else:
            return 1.0  # Diabetes range

    def calculate_bp_risk(self, systolic_bp):
        """Calculate blood pressure risk factor"""
        if systolic_bp < 120:
            return 0.1
        elif 120 <= systolic_bp < 140:
            return 0.3
        elif 140 <= systolic_bp < 160:
            return 0.6
        else:
            return 0.8

    def assess_risk(self, age, bmi, glucose, blood_pressure):
        """
        Comprehensive diabetes risk assessment
        Returns risk score and recommendations
        """
        try:
            # Calculate individual risk factors
            age_risk = self.calculate_age_risk(age)
            bmi_risk = self.calculate_bmi_risk(bmi)
            glucose_risk = self.calculate_glucose_risk(glucose)
            bp_risk = self.calculate_bp_risk(blood_pressure)

            # Weighted risk calculation
            weights = {
                'glucose': 0.4,  # Most important factor
                'bmi': 0.25,
                'age': 0.2,
                'bp': 0.15
            }

            overall_risk = (
                glucose_risk * weights['glucose'] +
                bmi_risk * weights['bmi'] +
                age_risk * weights['age'] +
                bp_risk * weights['bp']
            )

            # Risk categorization
            if overall_risk < 0.3:
                risk_level = "LOW"
                color = "green"
            elif overall_risk < 0.6:
                risk_level = "MODERATE"
                color = "yellow"
            else:
                risk_level = "HIGH"
                color = "red"

            # Generate recommendations
            recommendations = self._generate_recommendations(age, bmi, glucose, blood_pressure)

            # Factor analysis
            factor_analysis = {
                'age': {
                    'value': age,
                    'risk_score': age_risk,
                    'status': 'Normal' if age < 45 else 'Increased Risk'
                },
                'bmi': {
                    'value': bmi,
                    'risk_score': bmi_risk,
                    'category': self._get_bmi_category(bmi),
                    'status': 'Normal' if bmi < 25 else 'Overweight/Obese'
                },
                'glucose': {
                    'value': glucose,
                    'risk_score': glucose_risk,
                    'category': self._get_glucose_category(glucose),
                    'status': 'Normal' if glucose < 100 else 'Elevated'
                },
                'blood_pressure': {
                    'value': blood_pressure,
                    'risk_score': bp_risk,
                    'category': self._get_bp_category(blood_pressure),
                    'status': 'Normal' if blood_pressure < 120 else 'Elevated'
                }
            }

            return {
                'overall_risk_score': round(overall_risk, 3),
                'risk_level': risk_level,
                'risk_percentage': round(overall_risk * 100, 1),
                'color_indicator': color,
                'factor_analysis': factor_analysis,
                'recommendations': recommendations,
                'assessment_date': datetime.now().isoformat(),
                'next_screening': self._get_next_screening_date(risk_level)
            }

        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            raise

    def _get_bmi_category(self, bmi):
        """Get BMI category"""
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 25:
            return "Normal"
        elif 25 <= bmi < 30:
            return "Overweight"
        elif 30 <= bmi < 35:
            return "Obese Class I"
        elif 35 <= bmi < 40:
            return "Obese Class II"
        else:
            return "Obese Class III"

    def _get_glucose_category(self, glucose):
        """Get glucose category"""
        if glucose < 100:
            return "Normal"
        elif 100 <= glucose < 126:
            return "Prediabetes"
        else:
            return "Diabetes Range"

    def _get_bp_category(self, bp):
        """Get blood pressure category"""
        if bp < 120:
            return "Normal"
        elif 120 <= bp < 140:
            return "Elevated/Stage 1"
        else:
            return "Stage 2 Hypertension"

    def _generate_recommendations(self, age, bmi, glucose, bp):
        """Generate personalized recommendations"""
        recs = []
        recs.append("Maintain a balanced diet rich in vegetables, whole grains, and lean proteins.")
        recs.append("Engage in at least 150 minutes of moderate aerobic activity weekly.")

        if bmi >= 25:
            recs.append("Consider weight reduction through diet and exercise.")
            recs.append("Aim for 5-10% weight loss to significantly reduce diabetes risk.")

        if glucose >= 100:
            recs.append("Monitor carbohydrate intake and consider low glycemic index foods.")
            recs.append("Check blood glucose regularly.")

        if bp >= 130:
            recs.append("Monitor blood pressure; consider reducing salt intake.")
            recs.append("Consult a healthcare provider if blood pressure remains elevated.")

        return recs

    def _get_next_screening_date(self, risk_level):
        """Recommend next screening date"""
        if risk_level == "LOW":
            return (datetime.now().replace(microsecond=0)).isoformat()
        elif risk_level == "MODERATE":
            return (datetime.now().replace(microsecond=0)).isoformat()
        else:
            return (datetime.now().replace(microsecond=0)).isoformat()


@app.route("/assess", methods=["POST"])
def assess_diabetes():
    try:
        data = request.json
        age = data.get("age")
        bmi = data.get("bmi")
        glucose = data.get("glucose")
        bp = data.get("blood_pressure")

        calculator = DiabetesRiskCalculator()
        result = calculator.assess_risk(age, bmi, glucose, bp)

        return jsonify({
            "input": data,
            "result": result
        }), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
