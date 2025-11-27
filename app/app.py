import streamlit as st
import requests
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import re

load_dotenv(override=True)

BASE_URL = "https://v2.exercisedb.dev/api/v1"

# ---------------------------------------------------------
# 🔥 ExerciseDB search & video function
# ---------------------------------------------------------
def fetch_exercise_video(ex_name):
    """
    Search ExerciseDB for the exercise name and return the closest match videoUrl.
    """
    try:
        # 1️⃣ Search exercises
        search_url = f"{BASE_URL}/exercises/search?search={ex_name}"
        resp = requests.get(search_url)
        if resp.status_code != 200:
            return None
        results = resp.json().get("data", [])
        if not results:
            return None

        # 2️⃣ Exact match first (case-insensitive)
        best = None
        for ex in results:
            if ex_name.lower() == ex.get("name", "").lower():
                best = ex
                break

        # If no exact match, pick first result as closest
        if not best:
            best = results[0]

        ex_id = best.get("exerciseId")
        if not ex_id:
            return None

        # 3️⃣ Fetch full exercise details to get videoUrl
        details_url = f"{BASE_URL}/exercises/{ex_id}"
        details_resp = requests.get(details_url)
        if details_resp.status_code != 200:
            return None
        details = details_resp.json().get("data", None)
        if not details:
            return None

        return {
            "id": ex_id,
            "name": details.get("name"),
            "video": details.get("videoUrl"),
            "image": details.get("imageUrl")
        }

    except:
        return None

# ---------------------------------------------------------
# 🔥 LLM Prompt
# ---------------------------------------------------------
template = """
You are an expert workout coach with 10 years of experience.
Please create a workout program with the following information about the user:

Gender: {gender}
Age: {age}
Height: {height} cm
Weight: {weight} kg
Goals: {goals}
Target Zones: {target_zones}
Daily available time for training: {daily_time} minutes
Days per week available for training: {days_per_week} days

Program Design Principles:
1. Warm-up + cool-down included
2. Balance strength, endurance, flexibility
3. Provide exercises with sets, reps, rest
4. Age considerations
5. Progressive overload when needed
6. Injury prevention

Workout Plan Structure (for each day):
• Day focus
• Warm-up (5–10 min)
• 3–5 main exercises
• Cool-down (5–10 min)

📌 IMPORTANT:  
After each exercise you provide, ALWAYS output a separate line:

Canonical Exercise Name: <simple common name>

Rules:
- MUST be a real, common exercise name found in public databases
- Avoid niche variations
- Keep it short (e.g., Squat, Bench Press, Pull-up, Plank)
"""

prompt = PromptTemplate.from_template(template)
model = ChatMistralAI(model="magistral-small-latest", temperature=0.3)
output_parser = StrOutputParser()
chain = prompt | model | output_parser

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Fit Coach LLM", layout="centered", initial_sidebar_state="collapsed")
st.title("Fit Coach LLM")
st.markdown("<br>", unsafe_allow_html=True)

# --- PERSONAL INFO ---
st.subheader("Personal Informations")
col1, col2 = st.columns(2)
with col1:
    user_gender = st.selectbox("Gender:", ["Male ♂️", "Female ♀️", "Other"])
with col2:
    user_age = st.number_input("Age (years) 🎂", min_value=1, max_value=120, value=30)

# --- PHYSICAL MEASUREMENTS ---
st.subheader("Physical Measurements")
col3, col4 = st.columns(2)
with col3:
    user_weight = st.number_input("Weight (kg) ⚖️", min_value=1.0, max_value=300.0, value=70.0)
with col4:
    user_height = st.number_input("Height (cm) 📏", min_value=50, max_value=250, value=175)

# --- GOALS ---
st.subheader("Fitness Goals")
user_goals = st.selectbox(
    "Main goal 🎯",
    ["Muscle gain", "Weight loss", "Endurance improvement", "Flexibility", "General fitness"]
)

# --- AVAILABILITY ---
st.subheader("Availability")
user_days_per_week = st.number_input("Days per week 🗓️", 1, 7, 3)
user_daily_time = st.number_input("Time per session (minutes) ⏳", 0, 180, 30)

# --- TARGET ZONES ---
st.subheader("Target Zones")
BODY_PARTS = ["Legs 🦵", "Back 💪", "Shoulders 🤷‍♂️", "Chest 🏋️", "Arms 💪", "Abs 🍫"]

if "full_body_prev" not in st.session_state:
    st.session_state.full_body_prev = False

full_body = st.checkbox("Check All — FULL BODY 🔥", key="full_body")

if st.session_state.full_body != st.session_state.full_body_prev:
    for part in BODY_PARTS:
        st.session_state[f"check_{part}"] = full_body
    st.session_state.full_body_prev = full_body

selected_parts = {p: st.checkbox(p, key=f"check_{p}") for p in BODY_PARTS}

st.markdown("---")

# ---------------------------------------------------------
# GENERATE BUTTON
# ---------------------------------------------------------
if st.button("Generate My Program", type="primary"):

    final_zones = [p for p, v in selected_parts.items() if v]
    if not final_zones:
        st.warning("Please select at least one zone.")
        st.stop()

    with st.spinner("⌛ Generating your personalized workout program..."):
        response = chain.invoke({
            "gender": user_gender,
            "age": str(user_age),
            "height": str(user_height),
            "weight": str(user_weight),
            "goals": user_goals,
            "target_zones": ", ".join(final_zones),
            "daily_time": str(user_daily_time),
            "days_per_week": str(user_days_per_week)
        })

    st.write("### ✅ Your personalized workout plan:")
    st.write(response)

    # ---------------------------------------------------------
    # 🔍 PARSE CANONICAL EXERCISE NAMES
    # ---------------------------------------------------------
    canonical_names = []
    for line in response.split("\n"):
        match = re.search(r"Canonical Exercise Name:\s*(.*)", line)
        if match:
            canonical_names.append(match.group(1).strip())

    if not canonical_names:
        st.error("❌ No canonical exercise names found.")
        st.stop()

    st.markdown("---")
    st.subheader("🏋️ Exercise Videos")

    # ---------------------------------------------------------
    # FETCH + DISPLAY VIDEOS
    # ---------------------------------------------------------
    seen = set()
    for ex_name in canonical_names:
        if ex_name.lower() in seen:
            continue
        seen.add(ex_name.lower())

        info = fetch_exercise_video(ex_name)

        if info and info.get("video"):
            st.markdown(f"**{info['name']}**")
            st.video(info["video"])
        elif info and info.get("image"):
            st.markdown(f"**{info['name']}** (video not available, showing image)")
            st.image(info["image"], caption=info["name"])
        else:
            st.warning(f"No media found for '{ex_name}'")
