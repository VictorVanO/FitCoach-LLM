import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv(override=True)

# ChatMistralAI configuration
template = """You are an expert workout coach with 10 years of experience.
Please create a workout program with the following informations about the user:

Gender: {gender}
Age: {age}
Height: {height} cm
Weight: {weight} kg
Goals: {goals}
Target Zones: {target_zones}
Daily available time for training: {daily_time} minutes
Days per week available for training: {days_per_week} days

Program Design Principles:
1. Include proper warm-up and cool-down routines
2. Balance strength, endurance, and flexibility based on goals
3. Provide specific exercises with sets, reps, and rest periods
4. Consider user's age and potential limitations
5. Ensure progressive overload for muscle growth if that's the goal
6. Include safety considerations

Workout Plan Structure:
For each training day, provide:
1. Day focus (e.g., Upper Body, Lower Body, Cardio, etc.)
2. Warm-up (5-10 minutes)
3. Main exercises (3-5 exercises per session)
4. Cool-down/stretching (5-10 minutes)

Output Format:
- Workout Plan (detailed breakdown for each training day):
  Day 1: [Focus]
    Warm-up: [description]
    Exercise 1: [name] - Sets: [#] Reps: [#] Rest: [#]s
    Exercise 2: [name] - Sets: [#] Reps: [#] Rest: [#]s
    ...
    Cool-down: [description]
  Day 2: [Focus]
    ...
- Trained Muscles: [list of muscle groups targeted each day]
- Workout Duration: [total minutes per session]
- Frequency per week: [days per week]
- Additional Notes: [any special considerations, modifications, or tips]
"""
# Create prompt and chain
prompt = PromptTemplate.from_template(template)
model = ChatMistralAI(model="magistral-small-latest", temperature=0.3)
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Fit Coach LLM", layout="centered", initial_sidebar_state="collapsed")

# --- CSS ---
st.markdown("""
<style>
    h1 { text-align: center; color: #1E90FF; margin-top: 0px; padding-top: 0px; }
    h2 { margin-top: 25px; margin-bottom: 10px; }
    header, footer { display: none !important; }

    div[data-testid="stCheckbox"] label {
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.title("Fit Coach LLM")
st.markdown("<br>", unsafe_allow_html=True) 

# --- SECTION: INPUTS ---
# --- PERSONAL INFOS ---
st.subheader("Personal Informations")
col1, col2 = st.columns(2)
with col1:
    user_gender = st.selectbox("Gender:", ["Male ♂️", "Female ♀️", "Other"])
with col2:
    user_age = st.number_input("Age (years) 🎂", min_value=1, max_value=120, value=30, step=1, format="%d")

# --- PHYSICAL MEASUREMENTS ---
st.subheader("Physical Measurements")
col3, col4 = st.columns(2)
with col3:
    user_weight = st.number_input("Weight (kg) ⚖️", min_value=1.0, max_value=300.0, value=70.0, step=0.1, format="%.1f")
with col4:
    user_height = st.number_input("Height (cm) 📏", min_value=50, max_value=250, value=175, step=1, format="%d")

# --- GOALS ---
st.subheader("Fitness Goals")
user_goals = st.selectbox(
    "What is your main fitness goal? 🎯",
    ["Muscle gain", "Weight loss", "Endurance improvement", "Flexibility", "General fitness"],
    index=0
)

# --- AVAILABILITY ---
st.subheader("Availability")

user_days_per_week = st.number_input(
    "Number of available days every week 🗓️", 
    min_value=1, 
    max_value=7, 
    value=3, 
    step=1, 
    format="%d"
)

user_daily_time = st.number_input(
    "Time available per session (minutes) ⏳", 
    min_value=0, 
    max_value=180, 
    value=30, 
    step=5, 
    format="%d"
)



# --- BODY PARTS SELECTION ---
st.subheader("Target Zones")
st.write("Select the body parts you want to train:")

BODY_PARTS = [
    "Legs 🦵",
    "Back 💪",
    "Shoulders 🤷‍♂️",
    "Chest 🏋️",
    "Arms 💪",
    "Abs 🍫",
]

if "full_body_prev" not in st.session_state:
    st.session_state.full_body_prev = False

# Checkbox All
full_body = st.checkbox("Check All — FULL BODY 🔥", key="full_body")

st.markdown(
    """<hr style="margin-top: -8px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.2);">""",
    unsafe_allow_html=True
)

if st.session_state.full_body != st.session_state.full_body_prev:
    if st.session_state.full_body:   # vient d'être coché
        for part in BODY_PARTS:
            st.session_state[f"check_{part}"] = True
    else:                            # vient d'être décoché
        for part in BODY_PARTS:
            st.session_state[f"check_{part}"] = False

    st.session_state.full_body_prev = st.session_state.full_body

# Normal Checkboxes
selected_parts = {}
for part in BODY_PARTS:
    selected_parts[part] = st.checkbox(part, key=f"check_{part}")

st.markdown("---")



# SECTION: BUTTON SEND
if "program_response" not in st.session_state:
    st.session_state.program_response = None

generate_clicked = st.button("Generate My Program", type="primary")

if generate_clicked:
    final_selection = [part for part, is_checked in selected_parts.items() if is_checked]
    
    if not final_selection:
        st.warning("Please select at least one zone to work on.")
    else:
        with st.spinner("⌛ Generating your personalized workout program..."):
            response = chain.invoke({
                "gender": user_gender,
                "age": str(user_age),
                "height": str(user_height),
                "weight": str(user_weight),
                "goals": user_goals,
                "target_zones": ", ".join(final_selection),
                "daily_time": str(user_daily_time),
                "days_per_week": str(user_days_per_week)
            })
        # On sauvegarde la réponse dans la session
        st.session_state.program_response = response

# Affichage persistant du dernier programme généré
if st.session_state.program_response:
    st.write("### ✅ Your personalized workout plan:")
    st.write(st.session_state.program_response)