
import streamlit as st
import google.genai as genai
import requests
import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pymongo
from typing import List, Dict
import os
import tempfile
from gtts import gTTS
from googletrans import Translator
import speech_recognition as sr


class Config:
    GEMINI_API_KEY = "AIzaSyBT7MgEapM5V_UFo8KR3N9DsET3CGg1iYI"
    MONGODB_URI = "mongodb+srv://Nivetha:Welcome%40123@cluster0.0d1ja3a.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    DIABETES_SERVICE_URL = "http://localhost:5001"
    HEART_SERVICE_URL = "http://localhost:5002"
    MENTAL_HEALTH_SERVICE_URL = "http://localhost:5003"

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH = "medical_knowledge_index.faiss"
    KNOWLEDGE_DOCS_PATH = "medical_docs.json"

# -----------------------------
# CACHING RESOURCES
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_mongo_client(uri: str):
    try:
        return pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def get_genai_client(api_key: str):
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None

# -----------------------------
# UTILS: translation & TTS
# -----------------------------
translator = Translator()

LANG_OPTIONS = {
    "English": ("en", "English", "en"),
    "Hindi": ("hi", "Hindi", "hi"),
    "Tamil": ("ta", "Tamil", "ta"),
    "Gujarati": ("gu", "Gujarati", "gu"),
    "Marathi": ("mr", "Marathi", "mr"),
    "Bengali": ("bn", "Bengali", "bn"),
    "Telugu": ("te", "Telugu", "te"),
    "Kannada": ("kn", "Kannada", "kn"),
    "Malayalam": ("ml", "Malayalam", "ml"),
    "Punjabi": ("pa", "Punjabi", "pa"),
    "Urdu": ("ur", "Urdu", "ur"),
    "Hinglish (Roman Hindi)": ("hi", "Hindi", "hi"),
    "Thanglish (Roman Tamil)": ("ta", "Tamil", "ta")
}

def get_language_prompt(lang_name: str) -> str:
    """Get appropriate language instruction for Gemini"""
    language_prompts = {
        "English": "Respond in clear English.",
        "Hindi": "हिंदी में उत्तर दें। Hindi mein jawab dein.",
        "Tamil": "தமிழில் பதில் அளிக்கவும். Tamil-il pathil alikkavum.",
        "Gujarati": "ગુજરાતીમાં જવાબ આપો। Gujarati-ma javab apo.",
        "Marathi": "मराठीत उत्तर द्या। Marathi-t uttar dya.",
        "Bengali": "বাংলায় উত্তর দিন। Bengali-te uttor din.",
        "Telugu": "తెలుగులో సమాధానం ఇవ్వండి। Telugu-lo samadhanam ivvandi.",
        "Kannada": "ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ। Kannada-dalli uttarisi.",
        "Malayalam": "മലയാളത്തിൽ ഉത്തരം നൽകുക। Malayalam-il utharam nalkuka.",
        "Punjabi": "ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦਿਓ। Punjabi vich jawab dio.",
        "Urdu": "اردو میں جواب دیں۔ Urdu mein jawab dein."
    }
    return language_prompts.get(lang_name, "Respond in English.")

def translate_text(text: str, target_lang_code: str) -> str:
    try:
        if target_lang_code == "en":
            return text
        translated = translator.translate(text, dest=target_lang_code)
        return translated.text
    except Exception:
        return text

def speak_text(text: str, tts_lang_code: str):
    try:
        tts = gTTS(text=text, lang=tts_lang_code)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_name = tmp.name
        tmp.close()
        tts.save(tmp_name)
        os.system(f"start {tmp_name}" if os.name == 'nt' else f"afplay {tmp_name}")
    except Exception as e:
        st.warning(f"TTS failed: {e}")

def record_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... speak now!")
        audio = r.listen(source, phrase_time_limit=5)
    try:
        return r.recognize_google(audio, language='en-IN')
    except Exception as e:
        st.warning(f"Could not recognize speech: {e}")
        return ""

# -----------------------------
# RAG
# -----------------------------
class MedicalRAG:
    def __init__(self):
        try:
            self.embedding_model = get_embedding_model(Config.EMBEDDING_MODEL)
        except Exception:
            self.embedding_model = None
        self.index = None
        self.documents = []
        self.mongo_client = get_mongo_client(Config.MONGODB_URI)
        self.collection = self.mongo_client.medical_chatbot.medical_knowledge if self.mongo_client else None

    def load_medical_knowledge(self):
        if self.collection is None or self.embedding_model is None:
            return
        docs = list(self.collection.find({}))
        if not docs:
            self._initialize_medical_data()
            docs = list(self.collection.find({}))
        self.documents = [d["content"] for d in docs]
        embeddings = self.embedding_model.encode(self.documents)
        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def _initialize_medical_data(self):
        if not self.collection:
            return
        sample_docs = [
            {"category":"diabetes","content":"Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels. Type 1 diabetes results from autoimmune destruction of pancreatic beta cells. Type 2 diabetes involves insulin resistance and relative insulin deficiency. Common symptoms include polyuria (frequent urination), polydipsia (excessive thirst), polyphagia (increased hunger), unexplained weight loss, fatigue, blurred vision, and slow wound healing. Risk factors include family history, obesity, sedentary lifestyle, age >45 years, hypertension, and gestational diabetes history."},
            
            {"category":"heart_disease","content":"Cardiovascular disease encompasses conditions affecting the heart and blood vessels. Coronary artery disease occurs when plaque buildup narrows coronary arteries, reducing blood flow to heart muscle. Major risk factors include high cholesterol (LDL >130 mg/dL), hypertension (>140/90 mmHg), smoking, diabetes, obesity, physical inactivity, family history, and age. Symptoms may include chest pain (angina), shortness of breath, palpitations, dizziness, and fatigue. Prevention involves lifestyle modifications: regular exercise, heart-healthy diet, smoking cessation, stress management."},
            
            {"category":"mental_health","content":"Mental health conditions include depression, anxiety disorders, bipolar disorder, and schizophrenia. Depression symptoms include persistent low mood, loss of interest, sleep disturbances, appetite changes, fatigue, concentration difficulties, feelings of worthlessness, and suicidal thoughts. Anxiety symptoms include excessive worry, restlessness, muscle tension, and panic attacks. Risk factors include family history, trauma, chronic stress, substance abuse, and medical conditions. Treatment involves psychotherapy, medications, lifestyle changes, and social support."},
            
            {"category":"general_health","content":"Preventive healthcare focuses on maintaining wellness and preventing disease. Key components include regular health screenings (blood pressure, cholesterol, cancer screenings), vaccinations, healthy diet rich in fruits/vegetables/whole grains, regular physical activity (150 minutes moderate exercise weekly), adequate sleep (7-9 hours nightly), stress management, avoiding tobacco/excessive alcohol, and maintaining healthy weight. Annual check-ups help detect conditions early when most treatable."},
            
            {"category":"hypertension","content":"Hypertension (high blood pressure) is defined as systolic BP ≥140 mmHg or diastolic BP ≥90 mmHg. Often called 'silent killer' as it typically has no symptoms. Risk factors include age, family history, obesity, high sodium intake, excessive alcohol, physical inactivity, stress, and chronic conditions like diabetes. Complications include heart attack, stroke, kidney disease, and heart failure. Management involves lifestyle modifications and medications if needed."},
            
            {"category":"obesity","content":"Obesity is defined as BMI ≥30 kg/m². It increases risk for type 2 diabetes, heart disease, stroke, sleep apnea, certain cancers, and mental health issues. Causes include genetic factors, poor diet, physical inactivity, medications, and medical conditions. Treatment involves caloric restriction, increased physical activity, behavior modification, and sometimes medical intervention. Goal is gradual, sustained weight loss of 1-2 pounds per week."}
        ]
        try:
            self.collection.insert_many(sample_docs)
        except Exception:
            pass

    def retrieve_relevant_info(self, query: str, top_k: int = 3) -> List[str]:
        if self.index is None or self.embedding_model is None:
            return []
        try:
            q_emb = self.embedding_model.encode([query]).astype("float32")
            scores, indices = self.index.search(q_emb, top_k)
            relevant_docs = []
            threshold = 0.30
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                score = float(scores[0][i])
                if score > threshold:
                    relevant_docs.append(self.documents[idx])
            if not relevant_docs and len(indices[0]) > 0:
                first_idx = int(indices[0][0])
                if 0 <= first_idx < len(self.documents):
                    relevant_docs.append(self.documents[first_idx])
            return relevant_docs
        except Exception:
            return []

# -----------------------------
# Disease Prediction Logic
# -----------------------------
class DiseasePrediction:
    @staticmethod
    def calculate_diabetes_risk(age, bmi, glucose, bp, family_history, physical_activity, smoking):
        """Calculate diabetes risk score based on multiple factors"""
        risk_score = 0
        
        # Age factor
        if age >= 45:
            risk_score += 2
        elif age >= 35:
            risk_score += 1
            
        # BMI factor
        if bmi >= 30:
            risk_score += 3
        elif bmi >= 25:
            risk_score += 2
        elif bmi >= 23:
            risk_score += 1
            
        # Glucose factor
        if glucose >= 126:
            risk_score += 4
        elif glucose >= 100:
            risk_score += 2
            
        # Blood pressure factor
        if bp >= 140:
            risk_score += 2
        elif bp >= 130:
            risk_score += 1
            
        # Other factors
        if family_history:
            risk_score += 2
        if not physical_activity:
            risk_score += 1
        if smoking:
            risk_score += 1
            
        # Risk assessment
        if risk_score >= 8:
            return {"risk": "High", "score": risk_score, "recommendation": "Immediate medical consultation recommended"}
        elif risk_score >= 5:
            return {"risk": "Moderate", "score": risk_score, "recommendation": "Regular monitoring and lifestyle changes advised"}
        else:
            return {"risk": "Low", "score": risk_score, "recommendation": "Maintain healthy lifestyle"}

    @staticmethod
    def calculate_heart_risk(age, gender, cholesterol, bp_systolic, smoking, family_history, diabetes, physical_activity):
        """Calculate cardiovascular risk score"""
        risk_score = 0
        
        # Age and gender factors
        if gender == "Male":
            if age >= 55:
                risk_score += 3
            elif age >= 45:
                risk_score += 2
        else:  # Female
            if age >= 65:
                risk_score += 3
            elif age >= 55:
                risk_score += 2
                
        # Cholesterol factor
        if cholesterol >= 240:
            risk_score += 3
        elif cholesterol >= 200:
            risk_score += 2
            
        # Blood pressure factor
        if bp_systolic >= 160:
            risk_score += 3
        elif bp_systolic >= 140:
            risk_score += 2
        elif bp_systolic >= 130:
            risk_score += 1
            
        # Other factors
        if smoking:
            risk_score += 3
        if family_history:
            risk_score += 2
        if diabetes:
            risk_score += 2
        if not physical_activity:
            risk_score += 1
            
        # Risk assessment
        if risk_score >= 10:
            return {"risk": "High", "score": risk_score, "recommendation": "Urgent cardiology consultation required"}
        elif risk_score >= 6:
            return {"risk": "Moderate", "score": risk_score, "recommendation": "Regular cardiac monitoring and lifestyle modifications"}
        else:
            return {"risk": "Low", "score": risk_score, "recommendation": "Continue preventive measures"}

    @staticmethod
    def assess_mental_health(mood_score, sleep_score, stress_level, social_withdrawal, concentration, appetite_change, energy_level):
        """Assess mental health status"""
        total_score = mood_score + sleep_score + stress_level + social_withdrawal + concentration + appetite_change + energy_level
        
        if total_score >= 20:
            return {"risk": "High", "score": total_score, "recommendation": "Professional mental health support strongly recommended"}
        elif total_score >= 12:
            return {"risk": "Moderate", "score": total_score, "recommendation": "Consider counseling and stress management techniques"}
        else:
            return {"risk": "Low", "score": total_score, "recommendation": "Maintain good mental health practices"}

# -----------------------------
# Microservice integrator
# -----------------------------
class MicroserviceIntegrator:
    @staticmethod
    def check_diabetes_risk(age: int, bmi: float, glucose: float, bp: int) -> Dict:
        try:
            r = requests.post(f"{Config.DIABETES_SERVICE_URL}/calculate_risk",
                              json={"age": age, "bmi": bmi, "glucose": glucose, "blood_pressure": bp}, timeout=6)
            return r.json()
        except Exception as e:
            # Fallback to local prediction
            return DiseasePrediction.calculate_diabetes_risk(age, bmi, glucose, bp, False, True, False)

    @staticmethod
    def assess_heart_risk(age: int, cholesterol: int, bp_systolic: int, smoking: bool, family_history: bool) -> Dict:
        try:
            r = requests.post(f"{Config.HEART_SERVICE_URL}/assess_risk",
                              json={"age": age, "cholesterol": cholesterol, "bp_systolic": bp_systolic, "smoking": smoking, "family_history": family_history}, timeout=6)
            return r.json()
        except Exception as e:
            # Fallback to local prediction
            return DiseasePrediction.calculate_heart_risk(age, "Male", cholesterol, bp_systolic, smoking, family_history, False, True)

    @staticmethod
    def screen_mental_health(mood_score: int, sleep_score: int, stress_level: int, social_withdrawal: int) -> Dict:
        try:
            r = requests.post(f"{Config.MENTAL_HEALTH_SERVICE_URL}/mental_health",
                              json={"mood_score": mood_score, "sleep_score": sleep_score, "stress_level": stress_level, "social_withdrawal": social_withdrawal}, timeout=6)
            return r.json()
        except Exception as e:
            # Fallback to local assessment
            return DiseasePrediction.assess_mental_health(mood_score, sleep_score, stress_level, social_withdrawal, 2, 2, 2)

# -----------------------------
# Enhanced Chatbot
# -----------------------------
class MedicalChatbot:
    def __init__(self):
        self.client = get_genai_client(Config.GEMINI_API_KEY)
        self.rag = MedicalRAG()
        self.rag.load_medical_knowledge()
        self.integrator = MicroserviceIntegrator()

    def detect_intent(self, user_input: str) -> str:
        user_lower = user_input.lower()
        if any(word in user_lower for word in ['diabetes', 'blood sugar', 'glucose', 'thirst', 'urination', 'diabetic']):
            return 'diabetes_risk'
        if any(word in user_lower for word in ['heart', 'chest pain', 'cardiovascular', 'cholesterol', 'palpitations', 'cardiac']):
            return 'heart_risk'
        if any(word in user_lower for word in ['depression', 'anxiety', 'stress', 'mental health', 'mood', 'hopeless', 'sad', 'worried']):
            return 'mental_health'
        return 'general_medical'

    def generate_response(self, user_input: str, conversation_history: List[Dict]) -> str:
        intent = self.detect_intent(user_input)
        relevant_docs = self.rag.retrieve_relevant_info(user_input)
        context = "\n\n".join(relevant_docs) if relevant_docs else "No matching medical docs found."
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])

        # Get language settings
        lang_code, lang_name, tts_code = LANG_OPTIONS.get(st.session_state.ui_lang, ("en", "English", "en"))
        language_instruction = get_language_prompt(lang_name)

        prompt = f"""
You are a compassionate and knowledgeable medical assistant. {language_instruction}

IMPORTANT LANGUAGE INSTRUCTION: 
- Always respond in {lang_name} language
- Use native script and vocabulary appropriate for {lang_name}
- Be culturally sensitive in your language use
- If responding in regional Indian languages, you may use some English medical terms if commonly understood

MEDICAL CONTEXT FROM KNOWLEDGE BASE:
{context}

RECENT CONVERSATION HISTORY:
{history_text}

USER QUERY: {user_input}
DETECTED INTENT: {intent}

RESPONSE GUIDELINES:
1. Provide accurate, empathetic medical information
2. Use the medical context to support your answers
3. Be conservative with medical advice - always suggest consulting healthcare professionals for serious concerns
4. Ask relevant follow-up questions to better understand symptoms
5. Provide practical health tips when appropriate
6. If severe symptoms are mentioned, strongly advise immediate medical attention
7. Educate the user about their condition using the provided medical context
8. Keep responses concise but comprehensive (max 150 words)

Respond naturally in {lang_name} with appropriate medical guidance."""

        if self.client:
            try:
                
                response = self.client.models.generate_content(
                model="models/gemini-2.0-flash-exp", 
                contents=prompt
)

                bot_response = getattr(response, "text", None) or getattr(response, "output_text", None) or str(response)
            except Exception as e:
                bot_response = f"Sorry, I'm temporarily unavailable. However, I can still run health assessments. Error: {str(e)}"
        else:
            bot_response = "LLM service unavailable. I can still perform health risk assessments and provide basic guidance."

        if intent != 'general_medical':
            bot_response += "\n\n" + self._add_assessment_suggestion(intent, lang_name)
        
        return bot_response

    def _add_assessment_suggestion(self, intent: str, lang_name: str) -> str:
        suggestions = {
            "English": {
                'diabetes_risk': "💡 Would you like me to run a comprehensive diabetes risk assessment?",
                'heart_risk': "💡 I can perform a detailed cardiovascular risk evaluation if you'd like.",
                'mental_health': "💡 Would you benefit from a mental health screening questionnaire?"
            },
            "Hindi": {
                'diabetes_risk': "💡 क्या आप एक विस्तृत मधुमेह जोखिम मूल्यांकन चाहेंगे?",
                'heart_risk': "💡 मैं एक विस्तृत हृदय जोखिम मूल्यांकन कर सकता हूं।",
                'mental_health': "💡 क्या आपको मानसिक स्वास्थ्य स्क्रीनिंग की आवश्यकता है?"
            },
            "Tamil": {
                'diabetes_risk': "💡 நீரிழிவு நோய் அபாய மதிப்பீடு செய்ய விரும்புகிறீர்களா?",
                'heart_risk': "💡 இதய நோய் அபாய மதிப்பீடு செய்யலாமா?",
                'mental_health': "💡 மன ஆரோக்கிய பரிசோதனை தேவையா?"
            }
        }
        
        default_suggestions = {
            'diabetes_risk': "💡 Would you like a diabetes risk assessment?",
            'heart_risk': "💡 Would you like a heart risk evaluation?",
            'mental_health': "💡 Would you like a mental health screening?"
        }
        
        return suggestions.get(lang_name, default_suggestions).get(intent, "")

# -----------------------------
# Assessment UI Functions
# -----------------------------
def show_diabetes_assessment():
    st.markdown("### 🍬 Diabetes Risk Assessment")
    
    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
            height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0)
            glucose = st.number_input("Fasting Blood Glucose (mg/dL)", min_value=70, max_value=300, value=90)
            
        with col2:
            bp_systolic = st.number_input("Blood Pressure - Systolic", min_value=80, max_value=200, value=120)
            family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])
            physical_activity = st.selectbox("Regular Physical Activity", ["Yes", "No"])
            smoking = st.selectbox("Smoking Status", ["No", "Yes"])
        
        submit = st.form_submit_button("Calculate Diabetes Risk")
        
        if submit:
            bmi = weight / ((height/100) ** 2)
            
            result = DiseasePrediction.calculate_diabetes_risk(
                age=age,
                bmi=bmi,
                glucose=glucose,
                bp=bp_systolic,
                family_history=(family_history == "Yes"),
                physical_activity=(physical_activity == "Yes"),
                smoking=(smoking == "Yes")
            )
            
            # Display results with color coding
            risk_colors = {"Low": "green", "Moderate": "orange", "High": "red"}
            st.markdown(f"### Risk Level: <span style='color:{risk_colors[result['risk']]}'>{result['risk']}</span>", unsafe_allow_html=True)
            st.write(f"**Risk Score:** {result['score']}/12")
            st.write(f"**BMI:** {bmi:.1f}")
            st.info(f"**Recommendation:** {result['recommendation']}")
            
            # Add to chat history
            assessment_summary = f"Diabetes Risk Assessment: {result['risk']} risk (Score: {result['score']}/12). BMI: {bmi:.1f}. {result['recommendation']}"
            st.session_state.messages.append({"role": "assistant", "content": assessment_summary})

def show_heart_assessment():
    st.markdown("### ❤️ Cardiovascular Risk Assessment")
    
    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
            bp_systolic = st.number_input("Systolic Blood Pressure", min_value=80, max_value=220, value=120)
            
        with col2:
            smoking = st.selectbox("Smoking Status", ["No", "Yes"])
            family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            physical_activity = st.selectbox("Regular Exercise", ["Yes", "No"])
        
        submit = st.form_submit_button("Calculate Heart Risk")
        
        if submit:
            result = DiseasePrediction.calculate_heart_risk(
                age=age,
                gender=gender,
                cholesterol=cholesterol,
                bp_systolic=bp_systolic,
                smoking=(smoking == "Yes"),
                family_history=(family_history == "Yes"),
                diabetes=(diabetes == "Yes"),
                physical_activity=(physical_activity == "Yes")
            )
            
            # Display results
            risk_colors = {"Low": "green", "Moderate": "orange", "High": "red"}
            st.markdown(f"### Risk Level: <span style='color:{risk_colors[result['risk']]}'>{result['risk']}</span>", unsafe_allow_html=True)
            st.write(f"**Risk Score:** {result['score']}/15")
            st.info(f"**Recommendation:** {result['recommendation']}")
            
            # Add to chat history
            assessment_summary = f"Heart Risk Assessment: {result['risk']} risk (Score: {result['score']}/15). {result['recommendation']}"
            st.session_state.messages.append({"role": "assistant", "content": assessment_summary})

def show_mental_health_assessment():
    st.markdown("### 🧠 Mental Health Screening")
    st.write("Rate each item from 1 (not at all) to 5 (extremely)")
    
    with st.form("mental_health_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            mood_score = st.slider("How often do you feel sad or depressed?", 1, 5, 2)
            sleep_score = st.slider("How often do you have trouble sleeping?", 1, 5, 2)
            stress_level = st.slider("How stressed do you feel?", 1, 5, 2)
            social_withdrawal = st.slider("How often do you avoid social activities?", 1, 5, 2)
            
        with col2:
            concentration = st.slider("Difficulty concentrating?", 1, 5, 2)
            appetite_change = st.slider("Changes in appetite?", 1, 5, 2)
            energy_level = st.slider("Low energy or fatigue?", 1, 5, 2)
        
        submit = st.form_submit_button("Complete Mental Health Screening")
        
        if submit:
            result = DiseasePrediction.assess_mental_health(
                mood_score, sleep_score, stress_level, social_withdrawal,
                concentration, appetite_change, energy_level
            )
            
            # Display results
            risk_colors = {"Low": "green", "Moderate": "orange", "High": "red"}
            st.markdown(f"### Mental Health Status: <span style='color:{risk_colors[result['risk']]}'>{result['risk']}</span>", unsafe_allow_html=True)
            st.write(f"**Total Score:** {result['score']}/35")
            st.info(f"**Recommendation:** {result['recommendation']}")
            
            if result['risk'] == "High":
                st.error("⚠️ If you're having thoughts of self-harm, please contact emergency services or a mental health crisis hotline immediately.")
            
            # Add to chat history
            assessment_summary = f"Mental Health Screening: {result['risk']} concern level (Score: {result['score']}/35). {result['recommendation']}"
            st.session_state.messages.append({"role": "assistant", "content": assessment_summary})

# -----------------------------
# Enhanced Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="MediChat — AI Medical Assistant", page_icon="🩺", layout="wide")

    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = MedicalChatbot()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "assessment_mode" not in st.session_state:
        st.session_state.assessment_mode = None
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "English"
    if "tts" not in st.session_state:
        st.session_state.tts = False

    # Main layout
    left_col, right_col = st.columns([3, 1])
    
    with right_col:
        st.markdown("### 🗣️ Language & Voice")
        st.session_state.ui_lang = st.selectbox(
            "UI Language", 
            list(LANG_OPTIONS.keys()), 
            index=list(LANG_OPTIONS.keys()).index(st.session_state.ui_lang)
        )
        st.session_state.tts = st.checkbox("Enable Voice (TTS)", value=st.session_state.tts)
        
        st.markdown("---")
        st.markdown("### 🩺 Health Assessments")
        
        if st.button("🍬 Diabetes Risk Assessment", use_container_width=True):
            st.session_state.assessment_mode = "diabetes"
        if st.button("❤️ Heart Risk Assessment", use_container_width=True):
            st.session_state.assessment_mode = "heart"
        if st.button("🧠 Mental Health Screening", use_container_width=True):
            st.session_state.assessment_mode = "mental_health"
        if st.button("💬 Return to Chat", use_container_width=True):
            st.session_state.assessment_mode = None
            
        st.markdown("---")
        st.markdown("### ℹ️ Quick Info")
        st.write("🔬 **AI-Powered Medical Assistant**")
        st.write("🌍 **Multi-language Support**")
        st.write("🎯 **Disease Risk Predictions**")
        st.write("🗣️ **Voice Interaction**")
        
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


    with left_col:
        st.markdown(
            f"<h1 style='color:#0b63d6; text-align: center;'>🩺 MediChat - AI Medical Assistant</h1>", 
            unsafe_allow_html=True
        )
        
        # Show assessment mode or chat
        if st.session_state.assessment_mode == "diabetes":
            show_diabetes_assessment()
        elif st.session_state.assessment_mode == "heart":
            show_heart_assessment()
        elif st.session_state.assessment_mode == "mental_health":
            show_mental_health_assessment()
        else:
            # Chat interface
            st.markdown("---")
            
            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(
                            f"<div style='background:#e3f2fd;color:#1565c0;padding:10px;border-radius:10px;margin:8px 0;border-left:4px solid #2196f3'>"
                            f"<strong>👤 You:</strong> {message['content']}</div>", 
                            unsafe_allow_html=True
                        )
                    else:
                        # Get language settings for translation
                        lang_code, lang_name, tts_code = LANG_OPTIONS.get(st.session_state.ui_lang, ("en", "English", "en"))
                        
                        # Only translate if not English and not already in target language
                        displayed_content = message['content']
                        if lang_code != "en" and not any(char in displayed_content for char in ['हिंदी', 'தமிழ்', 'ગુજરાતી', 'मराठी', 'বাংলা', 'తెలుగు', 'ಕನ್ನಡ', 'മലയാളം', 'ਪੰਜਾਬੀ', 'اردو']):
                            displayed_content = translate_text(message['content'], lang_code)
                        
                        st.markdown(
                            f"<div style='background:#f3e5f5;color:#4a148c;padding:10px;border-radius:10px;margin:8px 0;border-left:4px solid #9c27b0'>"
                            f"<strong>🩺 MediChat:</strong> {displayed_content}</div>", 
                            unsafe_allow_html=True
                        )
                        
                        # Text-to-speech
                        if st.session_state.tts:
                            try:
                                speak_text(displayed_content, tts_code)
                            except Exception as e:
                                st.warning(f"TTS not available: {e}")

            # Chat input
            st.markdown("---")
            with st.form(key="chat_form", clear_on_submit=True):
                user_input = st.text_input(
                    "", 
                    placeholder="Ask about symptoms, conditions, treatments, or health advice...",
                    help="Type your medical question or health concern"
                )
                
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    submit = st.form_submit_button("💬 Send Message", use_container_width=True)
                with col2:
                    voice_btn = st.form_submit_button("🎤 Voice", use_container_width=True)
                with col3:
                    if st.form_submit_button("🔄 Reset", use_container_width=True):
                        st.session_state.messages = []
                        st.rerun()


                # Handle form submission
                if submit and user_input.strip():
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
                    
                    # Generate bot response
                    with st.spinner("🤔 Thinking..."):
                        try:
                            bot_response = st.session_state.chatbot.generate_response(
                                user_input.strip(), 
                                st.session_state.messages
                            )
                            st.session_state.messages.append({"role": "assistant", "content": bot_response})
                        except Exception as e:
                            error_msg = f"I apologize, but I encountered an issue: {str(e)}. Please try rephrasing your question."
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
                    st.rerun()

                    
                elif voice_btn:
                    with st.spinner("🎤 Listening..."):
                        spoken_text = record_voice()
                        if spoken_text:
                            st.session_state.messages.append({"role": "user", "content": spoken_text})
                            
                            with st.spinner("🤔 Processing..."):
                                try:
                                    bot_response = st.session_state.chatbot.generate_response(
                                        spoken_text, 
                                        st.session_state.messages
                                    )
                                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                                except Exception as e:
                                    error_msg = f"I apologize, but I encountered an issue: {str(e)}. Please try again."
                                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            
                            st.rerun()


            # Sample questions for user guidance
            if not st.session_state.messages:
                st.markdown("### 💡 Sample Questions")
                sample_questions = [
                    "What are the symptoms of diabetes?",
                    "How can I prevent heart disease?",
                    "I have been feeling anxious lately, what should I do?",
                    "What is a healthy BMI range?",
                    "How much exercise should I do weekly?"
                ]
                
                cols = st.columns(len(sample_questions))
                for i, question in enumerate(sample_questions):
                    with cols[i]:
                        if st.button(f"❓ {question[:30]}...", key=f"sample_{i}", use_container_width=True):
                            st.session_state.messages.append({"role": "user", "content": question})
                            with st.spinner("🤔 Thinking..."):
                                bot_response = st.session_state.chatbot.generate_response(question, st.session_state.messages)
                                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                            st.rerun()


    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 12px;'>"
        "⚠️ <strong>Disclaimer:</strong> This AI assistant provides general health information only. "
        "Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment. "
        "In case of medical emergencies, contact emergency services immediately."
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
