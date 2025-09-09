# main.py
# ====================================================================
# MediChat — Multilingual + Voice + Performance improvements
# ====================================================================
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

# -----------------------------
# CONFIG
# -----------------------------
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
    "English": ("en", "English"),
    "Hindi": ("hi", "Hindi"),
    "Tamil": ("ta", "Tamil"),
    "Gujarati": ("gu", "Gujarati"),
    "Marathi": ("mr", "Marathi"),
    "Hinglish (Roman Hindi)": ("hi", "Hindi"),
    "Thanglish (Roman Tamil)": ("ta", "Tamil")
}

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
            {"category":"diabetes","content":"Diabetes is a metabolic disorder characterized by high blood glucose levels. Symptoms include thirst, frequent urination, fatigue."},
            {"category":"heart_disease","content":"Heart disease: risk factors include high cholesterol, hypertension, smoking, obesity. Symptoms: chest pain, shortness of breath."},
            {"category":"mental_health","content":"Mental health conditions include depression and anxiety. Symptoms include low mood, sleep problems, concentration difficulties."},
            {"category":"general_health","content":"Maintain regular exercise, balanced diet, sleep, stress management and preventive healthcare."}
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
            return {"error": f"Service unavailable: {e}"}

    @staticmethod
    def assess_heart_risk(age: int, cholesterol: int, bp_systolic: int, smoking: bool, family_history: bool) -> Dict:
        try:
            r = requests.post(f"{Config.HEART_SERVICE_URL}/assess_risk",
                              json={"age": age, "cholesterol": cholesterol, "bp_systolic": bp_systolic, "smoking": smoking, "family_history": family_history}, timeout=6)
            return r.json()
        except Exception as e:
            return {"error": f"Service unavailable: {e}"}

    @staticmethod
    def screen_mental_health(mood_score: int, sleep_score: int, stress_level: int, social_withdrawal: int) -> Dict:
        try:
            r = requests.post(f"{Config.MENTAL_HEALTH_SERVICE_URL}/mental_health",
                              json={"mood_score": mood_score, "sleep_score": sleep_score, "stress_level": stress_level, "social_withdrawal": social_withdrawal}, timeout=6)
            return r.json()
        except Exception as e:
            return {"error": f"Service unavailable: {e}"}

# -----------------------------
# Chatbot
# -----------------------------
class MedicalChatbot:
    def __init__(self):
        self.client = get_genai_client(Config.GEMINI_API_KEY)
        self.rag = MedicalRAG()
        self.rag.load_medical_knowledge()
        self.integrator = MicroserviceIntegrator()

    def detect_intent(self, user_input: str) -> str:
        user_lower = user_input.lower()
        if any(word in user_lower for word in ['diabetes', 'blood sugar', 'glucose', 'thirst', 'urination']):
            return 'diabetes_risk'
        if any(word in user_lower for word in ['heart', 'chest pain', 'cardiovascular', 'cholesterol', 'palpitations']):
            return 'heart_risk'
        if any(word in user_lower for word in ['depression', 'anxiety', 'stress', 'mental health', 'mood', 'hopeless']):
            return 'mental_health'
        return 'general_medical'

    def generate_response(self, user_input: str, conversation_history: List[Dict]) -> str:
        intent = self.detect_intent(user_input)
        relevant_docs = self.rag.retrieve_relevant_info(user_input)
        context = "\n\n".join(relevant_docs) if relevant_docs else "No matching medical docs found."
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])

        # Language selection
        lang_code, lang_name = LANG_OPTIONS.get(st.session_state.ui_lang, ("en", "English"))

        prompt = f"""
You are a concise empathetic medical assistant. Use the context below for factual claims and be conservative with medical advice.
Respond in {lang_name}.

MEDICAL CONTEXT:
{context}

HISTORY:
{history_text}

USER: {user_input}

INTENT: {intent}

Provide a concise, empathetic answer. If severe symptoms are mentioned, advise urgent clinical evaluation; otherwise ask other factors or further details (not more than 7) and suggest the prescription.
Also try to educate the user with relevant info from the context."""
        if self.client:
            try:
                response = self.client.models.generate_content(model="models/gemini-2.0-flash-exp", contents=prompt)
                bot_response = getattr(response, "text", None) or getattr(response, "output_text", None) or str(response)
            except Exception:
                bot_response = "Sorry — LLM is temporarily unavailable; I can still run screenings or give general guidance."
        else:
            bot_response = "LLM unavailable — I can still run screenings or provide general info."

        if intent != 'general_medical':
            bot_response += "\n\n" + self._add_assessment_suggestion(intent)
        return bot_response

    def _add_assessment_suggestion(self, intent: str) -> str:
        return {
            'diabetes_risk': "💡 I can do a quick diabetes risk check — would you like that?",
            'heart_risk': "💡 I can run a short heart-risk check if you share basic info.",
            'mental_health': "💡 I can do a quick mental health screening."
        }.get(intent, "")

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="MediChat — AI Medical Assistant", page_icon="🩺", layout="wide")

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

    left_col, right_col = st.columns([3,1])
    with right_col:
        st.markdown("### 🗣️ Language & Voice")
        st.session_state.ui_lang = st.selectbox("UI Language", list(LANG_OPTIONS.keys()), index=list(LANG_OPTIONS.keys()).index(st.session_state.ui_lang))
        st.session_state.tts = st.checkbox("Enable Voice (TTS)", value=st.session_state.tts)
        st.markdown("---")
        st.markdown("### 🩺 Quick Tools")
        if st.button("🍬 Diabetes Risk"):
            st.session_state.assessment_mode = "diabetes"
        if st.button("❤️ Heart Risk"):
            st.session_state.assessment_mode = "heart"
        if st.button("🧠 Mental Health"):
            st.session_state.assessment_mode = "mental_health"
        if st.button("💬 Return to Chat"):
            st.session_state.assessment_mode = None
        st.markdown("---")

    with left_col:
        st.markdown(f"<h1 style='color:#0b63d6'>MediChat — AI Medical Assistant 🩺</h1>", unsafe_allow_html=True)

        st.markdown('<div style="background:#fff;padding:12px;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.04)">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"<div style='background:black;color:white;padding:8px;border-radius:8px;margin:6px 0'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
            else:
                lang_code, _ = LANG_OPTIONS.get(st.session_state.ui_lang, ("en", "English"))
                translated = translate_text(message['content'], lang_code)
                st.markdown(f"<div style='background:#0b63d6;color:black;padding:8px;border-radius:8px;margin:6px 0'><strong>MediChat:</strong> {translated}</div>", unsafe_allow_html=True)
                if st.session_state.tts:
                    speak_text(translated, lang_code)
        st.markdown('</div>', unsafe_allow_html=True)

        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("", placeholder="Ask about symptoms, risk, or medical info...")
            col1, col2 = st.columns([4,1])
            with col1:
                submit = st.form_submit_button("Send")
            with col2:
                voice_btn = st.form_submit_button("🎤 Speak")

            if submit and user_input.strip():
                st.session_state.messages.append({"role":"user", "content": user_input.strip()})
                bot_resp = st.session_state.chatbot.generate_response(user_input.strip(), st.session_state.messages)
                st.session_state.messages.append({"role":"assistant", "content": bot_resp})
                st.experimental_rerun()
            elif voice_btn:
                spoken_text = record_voice()
                if spoken_text:
                    st.session_state.messages.append({"role":"user", "content": spoken_text})
                    bot_resp = st.session_state.chatbot.generate_response(spoken_text, st.session_state.messages)
                    st.session_state.messages.append({"role":"assistant", "content": bot_resp})
                    st.experimental_rerun()

if __name__ == "__main__":
    main()
