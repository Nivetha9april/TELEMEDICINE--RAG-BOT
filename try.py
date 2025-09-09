import streamlit as st
import google.genai as genai

# --------------------------
# Configure Gemini API
# --------------------------
client = genai.Client(api_key="AIzaSyBT7MgEapM5V_UFo8KR3N9DsET3CGg1iYI")  # Replace with your key

# --------------------------
# Initialize session state
# --------------------------
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'appointments' not in st.session_state:
    st.session_state['appointments'] = []
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --------------------------
# Streamlit UI
# --------------------------
st.title("Interactive Rural Health Assistant 🤖🏥")
st.markdown("""
Ask about symptoms, medications, prescriptions, general health info (e.g., "What is COVID?"), vaccines, or rural healthcare advice.  
The assistant will ask follow-ups, suggest next steps, and store a mock patient history.  
**Disclaimer:** AI-generated information. Always consult a healthcare professional.
""")

# Language selection
language = st.selectbox(
    "Preferred language / பாசிட்டீ / ભાષા / भाषा",
    ["English", "Tamil", "Thanglish", "Hindi", "Gujarati"]
)

# --------------------------
# Function to query Gemini
# --------------------------
def ask_gemini(user_input):
    # Build conversation context
    history_text = "\n".join([f"- {item}" for item in st.session_state['history']])
    conversation_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state['messages']])
    
    prompt_text = f"""
You are a rural health assistant AI. Answer in {language}.
Current patient history:
{history_text if history_text else 'No previous history.'}

Conversation context:
{conversation_context}

User input: {user_input}

Tasks:
1. Ask follow-up questions for symptoms if needed.
2. Suggest medications/prescriptions (mock).
3. Provide structured info (tables/lists) if asked.
4. Suggest next steps or tests.
5. Give rural health advice if relevant.
6. Maintain context of the conversation.
"""

    try:
        # Fixed: Pass contents as a simple string, not a list with dictionaries
        response = client.models.generate_content(
            model="models/gemini-2.0-flash-exp",
            contents=prompt_text
        )

        # Access the response text correctly
        bot_reply = response.text

        # Update conversation
        st.session_state['messages'].append({"role": "user", "content": user_input})
        st.session_state['messages'].append({"role": "assistant", "content": bot_reply})

        # Store medications/prescriptions in history
        if "medication" in bot_reply.lower() or "prescription" in bot_reply.lower():
            st.session_state['history'].append(bot_reply)

        return bot_reply
    
    except Exception as e:
        error_msg = f"Error communicating with Gemini API: {str(e)}"
        st.error(error_msg)
        return "Sorry, I'm having trouble processing your request. Please try again."

# --------------------------
# Chat input
# --------------------------
user_input = st.text_input("You:", key="input")

if st.button("Send") and user_input.strip() != "":
    reply = ask_gemini(user_input)
    st.text_area("Bot:", value=reply, height=200, key="bot_reply")

# --------------------------
# Display conversation history
# --------------------------
st.subheader("Conversation History")
for msg in st.session_state['messages']:
    role = "You" if msg['role'] == "user" else "Bot"
    st.markdown(f"**{role}:** {msg['content']}")

# --------------------------
# Mock appointment booking
# --------------------------
st.subheader("Book an Appointment (Optional)")

clinic_slots = [
    "10:00 AM - 11:00 AM",
    "11:30 AM - 12:30 PM",
    "2:00 PM - 3:00 PM",
    "3:30 PM - 4:30 PM"
]

selected_slot = st.selectbox("Choose an available slot", clinic_slots)
if st.button("Confirm Appointment"):
    st.session_state['appointments'].append(selected_slot)
    st.success(f"Appointment confirmed for {selected_slot} at Nearby Clinic.")

if st.session_state['appointments']:
    st.subheader("Your Booked Appointments")
    for idx, slot in enumerate(st.session_state['appointments'], start=1):
        st.write(f"{idx}. {slot}")

# --------------------------
# Display mock patient history
# --------------------------
if st.session_state['history']:
    st.subheader("Mock Patient History / Previous Medications")
    for idx, item in enumerate(st.session_state['history'], start=1):
        st.write(f"{idx}. {item}")