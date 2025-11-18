import streamlit as st
import numpy as np

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
st.write("Check the body parts you want to work on:")

BODY_PARTS = [
    "Legs 🦵",
    "Back 💪",
    "Shoulders 🤷‍♂️",
    "Chest 🏋️",
    "Arms 💪",
    "Abs 🍫",
]

selected_parts = {}

for part_name_with_emoji in BODY_PARTS:
    selected_parts[part_name_with_emoji] = st.checkbox(
        part_name_with_emoji, 
        key=f"check_{part_name_with_emoji}"
    )


st.markdown("---")



# SECTION: BUTTON SEND
if st.button("Generate My Program", type="primary"):
    final_selection = [part for part, is_checked in selected_parts.items() if is_checked]
    
    if not final_selection:
        st.warning("Please select at least one zone to work on.")
    else:
        st.success(f"Generation complete for {user_days_per_week} days/week: {', '.join(final_selection)}")
