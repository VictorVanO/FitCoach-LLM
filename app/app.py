import streamlit as st
import requests
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import re
from prompt import workout_prompt
from ecologits_tracker import EcoMistralTracker
from pint import UnitRegistry

eco_tracker = EcoMistralTracker()

load_dotenv(override=True)

BASE_URL = "https://v2.exercisedb.dev/api/v1"

# Unit Registry for formatting
u = UnitRegistry()
u.define("mWh = milliwatt_hour")
u.define("kWh = kilowatt_hour")
u.define("mgCO2eq = milligram")
u.define("kgCO2eq = kilogram")
u.define("µgSbeq = microgram")
u.define("kgSbeq = kilogram")
u.define("kJ = kilojoule")
u.define("MJ = megajoule")
u.define("mL = milliliter")
u.define("L = liter")

def format_energy(energy_value: float, energy_unit: str = "kWh") -> tuple[float, str]:
    """Format energy to mWh"""
    val = u.Quantity(energy_value, energy_unit)
    val = val.to("mWh")
    return round(val.magnitude, 3), "mWh"

def format_gwp(gwp_value: float, gwp_unit: str = "kgCO2eq") -> tuple[float, str]:
    """Format GHG emissions to mgCO2eq"""
    val = u.Quantity(gwp_value, gwp_unit)
    val = val.to("mgCO2eq")
    return round(val.magnitude, 3), "mgCO2eq"

def format_adpe(adpe_value: float, adpe_unit: str = "kgSbeq") -> tuple[float, str]:
    """Format abiotic resources to µgSbeq"""
    val = u.Quantity(adpe_value, adpe_unit)
    val = val.to("µgSbeq")
    return round(val.magnitude, 4), "µgSbeq"

def format_pe(pe_value: float, pe_unit: str = "MJ") -> tuple[float, str]:
    """Format primary energy to kJ"""
    val = u.Quantity(pe_value, pe_unit)
    val = val.to("kJ")
    return round(val.magnitude, 4), "kJ"

def format_wcf(wcf_value: float, wcf_unit: str = "L") -> tuple[float, str]:
    """Format water consumption to mL"""
    val = u.Quantity(wcf_value, wcf_unit)
    val = val.to("mL")
    return round(val.magnitude, 4), "mL"

# ExerciseDB API search & video function
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
    
model = ChatMistralAI(model="magistral-small-latest", temperature=0.3)
output_parser = StrOutputParser()
chain = workout_prompt | model | output_parser

# STREAMLIT UI
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
    ["Muscle gain (Hypertrophy)", "Weight loss", "Endurance improvement", "Flexibility", "General fitness", "Cardio"]
)

# --- AVAILABILITY ---
st.subheader("Availability")
user_days_per_week = st.number_input("Days per week 🗓️", 1, 7, 3)
user_daily_time = st.number_input("Time per session (minutes) ⏳", 0, 180, 60)

# --- TARGET ZONES ---
st.subheader("Target Zones")
BODY_PARTS = ["Legs 🦵", "Back 💪", "Shoulders 🤷‍♂️", "Chest 🏋️", "Arms 💪", "Abs 🍫"]

if "full_body_prev" not in st.session_state:
    st.session_state.full_body_prev = False

full_body = st.checkbox("Check All — FULL BODY 🔥", key="full_body", value=True)

st.markdown(
    """<hr style="margin-top: -8px; margin-bottom: 10px; border: 0.1px solid rgba(255,255,255,0.2);">""",
    unsafe_allow_html=True
)

if st.session_state.full_body != st.session_state.full_body_prev:
    for part in BODY_PARTS:
        st.session_state[f"check_{part}"] = full_body
    st.session_state.full_body_prev = full_body

selected_parts = {p: st.checkbox(p, key=f"check_{p}") for p in BODY_PARTS}

# --- EQUIPMENT SECTION ---
st.subheader("Equipment")

equipment_mode = st.radio(
    "Do you have training equipment available?",
    ["No equipment", "With equipment"],
    index=0,
    horizontal=True
)

EQUIPMENT_OPTIONS = [
    "Dumbbells",
    "Barbell",
    "Kettlebell",
    "Resistance bands",
    "Pull-up bar",
    "Bench",
    "Gym machines",
    "TRX / suspension trainer",
]

# With Equipement
if "equip_all_prev" not in st.session_state:
    st.session_state.equip_all_prev = False

selected_equipment = []
if equipment_mode == "With equipment":
    # --- Check all ---
    equip_all = st.checkbox("Check All — Equipment 💼", key="equip_all", value=True)

    st.markdown(
        """<hr style="margin-top: -8px; margin-bottom: 10px; border: 0.5px solid rgba(255,255,255,0.2);">""",
        unsafe_allow_html=True
    )

    if st.session_state.equip_all != st.session_state.equip_all_prev:
        if st.session_state.equip_all:
            for eq in EQUIPMENT_OPTIONS:
                st.session_state[f"eq_{eq}"] = True
        else:
            for eq in EQUIPMENT_OPTIONS:
                st.session_state[f"eq_{eq}"] = False

        st.session_state.equip_all_prev = st.session_state.equip_all

    # Normal checkboxes
    for eq in EQUIPMENT_OPTIONS:
        if st.checkbox(eq, key=f"eq_{eq}", value=True):
            selected_equipment.append(eq)

else:
    # No equipement
    selected_equipment = []


# LINE FOR SEPARATION
st.markdown(
    """<hr style="margin-top: 20px; margin-bottom: 10px; border: 2px solid rgba(255,255,255,0.2);">""",
    unsafe_allow_html=True
)


# SECTION: BUTTON SEND
if "program_response" not in st.session_state:
    st.session_state.program_response = None

generate_clicked = st.button("Generate My Program", type="primary")

if generate_clicked:
    final_selection = [part for part, is_checked in selected_parts.items() if is_checked]
    
    if not final_selection:
        st.warning("Please select at least one zone to work on.")
    else:
        # Texte propre pour le matériel
        equipment_text = (
            ", ".join(selected_equipment)
            if selected_equipment
            else "No equipment (bodyweight / home workouts only)"
        )

        with st.spinner("⌛ Generating your personalized workout program... This can take up to a minute."):
            response = chain.invoke({
                "gender": user_gender,
                "age": str(user_age),
                "height": str(user_height),
                "weight": str(user_weight),
                "goals": user_goals,
                "target_zones": ", ".join(final_selection),
                "daily_time": str(user_daily_time),
                "days_per_week": str(user_days_per_week),
                "equipment": equipment_text,
            })
        st.session_state.program_response = response
        
        # ✅ Make ONE tracked inference just for impact measurement
        eco_text, eco_impacts = eco_tracker.tracked_inference(
            "Generate a short summary of this workout program for environmental tracking."
        )

        st.session_state.eco_impact = eco_impacts

if st.session_state.program_response:
    st.write("### ✅ Your personalized workout plan:")
    st.write(st.session_state.program_response)

canonical_names = []

if st.session_state.program_response:
    for line in st.session_state.program_response.split("\n"):
        match = re.search(r"Canonical Exercise Name:\s*(.*)", line)
        if match:
            canonical_names.append(match.group(1).strip())

# Deduplicate canonical names early
canonical_names = list(dict.fromkeys([ex.strip() for ex in canonical_names]))

# EXERCISE VIDEO SECTION
if canonical_names:
    st.markdown("---")
    st.subheader("🏋️ Exercise Videos")

    if "selected_exercise" not in st.session_state:
        st.session_state.selected_exercise = None

    st.write("Click an exercise to view its video:")

    # Remove duplicates & normalize names
    unique_exercises = list(dict.fromkeys([ex.strip().lower() for ex in canonical_names]))

    # Map back to original formatting while keeping stable unique keys
    for ex_raw in canonical_names:
        ex_key = "btn_" + re.sub(r'[^a-zA-Z0-9]', '_', ex_raw.lower()).strip("_")

        if st.button(ex_raw, key=ex_key):
            st.session_state.selected_exercise = ex_raw

    # Modal
    if st.session_state.selected_exercise:
        info = fetch_exercise_video(st.session_state.selected_exercise)

        # Use expander to show/hide content
        with st.expander(f"{st.session_state.selected_exercise} — Video", expanded=True):
            if info and info.get("video"):
                st.video(info["video"], start_time=0)
            elif info and info.get("image"):
                st.image(info["image"], caption=info["name"])
            else:
                st.warning("No media found.")

            if st.button("Close", key="close_modal"):
                st.session_state.selected_exercise = None

if "eco_impact" in st.session_state and st.session_state.eco_impact:
    impact = st.session_state.eco_impact

    st.markdown("---")
    st.subheader("🌱 Environmental Impact (EcoLogits)")

    energy_val, energy_unit = format_energy(impact.energy.value, impact.energy.unit)
    gwp_val, gwp_unit = format_gwp(impact.gwp.value, impact.gwp.unit)
    adpe_val, adpe_unit = format_adpe(impact.adpe.value, impact.adpe.unit)
    pe_val, pe_unit = format_pe(impact.pe.value, impact.pe.unit)
    wcf_val, wcf_unit = format_wcf(impact.wcf.value, impact.wcf.unit)

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, _ = st.columns(2)

    with col1:
        st.metric(
            "Energy ⚡",
            f"{energy_val} {energy_unit}"
        )

    with col2:
        st.metric(
            "Climate Impact 🌍",
            f"{gwp_val} {gwp_unit}"
        )

    with col3:
        st.metric(
            "Abiotic Resources 🪨",
            f"{adpe_val} {adpe_unit}"
        )

    with col4:
        st.metric(
            "Primary Energy 🔋",
            f"{pe_val} {pe_unit}"
        )

    with col5:
        st.metric(
            "Water Consumption 🚰",
            f"{wcf_val} {wcf_unit}"
        )