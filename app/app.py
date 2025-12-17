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
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import MistralAIEmbeddings
from ml_injury_risk_synthetic import load_model, predict_injury_risk



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
u.define("h = hour")
u.define("min = minute")
u.define("s = second")
u.define("km = kilometer")
u.define("m = meter")

# Equivalence constants
RUNNING_ENERGY_EQ = u.Quantity(294, "kJ / km")  # running 1 km at 10 km/h with a weight of 70 kg
WALKING_ENERGY_EQ = u.Quantity(196, "kJ / km")  # walking 1 km at 3 km/h with a weight of 70 kg
EV_ENERGY_EQ = u.Quantity(0.17, "kWh / km")
STREAMING_GWP_EQ = u.Quantity(15.6, "h / kgCO2eq")  # hours of video streaming per kgCO2eq

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

def get_physical_activity_equivalent(energy: u.Quantity) -> tuple[str, float, str]:
    """Compare energy consumption with walking or running"""
    energy_kj = energy.to("kJ")
    running_distance = energy_kj / RUNNING_ENERGY_EQ
    
    if running_distance > u.Quantity(1, "km"):
        return "🏃 Running", round(running_distance.magnitude, 3), "km"
    
    walking_distance = energy_kj / WALKING_ENERGY_EQ
    if walking_distance < u.Quantity(1, "km"):
        walking_distance = walking_distance.to("m")
        return "🚶 Walking", round(walking_distance.magnitude, 3), "m"
    
    return "🚶 Walking", round(walking_distance.magnitude, 3), "km"

def get_ev_equivalent(energy: u.Quantity) -> tuple[float, str]:
    """Compare energy consumption with electric vehicle driving"""
    energy_kwh = energy.to("kWh")
    ev_distance = energy_kwh / EV_ENERGY_EQ
    
    if ev_distance < u.Quantity(1, "km"):
        ev_distance = ev_distance.to("m")
        return round(ev_distance.magnitude, 3), "m"
    
    return round(ev_distance.magnitude, 3), "km"

def get_streaming_equivalent(gwp: u.Quantity) -> tuple[float, str]:
    """Compare GHG emissions with video streaming hours"""
    gwp_kg = gwp.to("kgCO2eq")
    streaming_time = gwp_kg * STREAMING_GWP_EQ
    
    if streaming_time < u.Quantity(1, "h"):
        streaming_time = streaming_time.to("min")
        return round(streaming_time.magnitude, 3), "min"
    
    if streaming_time < u.Quantity(1, "min"):
        streaming_time = streaming_time.to("s")
        return round(streaming_time.magnitude, 3), "s"
    
    return round(streaming_time.magnitude, 3), "h"

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

@st.cache_resource
def get_rag_chain():
    """
    Builds a RAG chain based on the physio/sport PDFs present in the ./pdfs_rag folder.
    Used to explain and justify a workout plan with evidence-based guidelines.
    """
    pdf_files = [
        "pdfs_rag/back_exercises.pdf",
        "pdfs_rag/exercise_starter_guide_mayo_clinic.pdf",
        "pdfs_rag/full_body_stretching_guide.pdf",
        "pdfs_rag/pep_program_training_plan.pdf",
        "pdfs_rag/program_exercices_epicondylite_aaos.pdf",
        "pdfs_rag/resistance-training-ACSM.pdf",
        "pdfs_rag/rotator_cuff_shoulder_rehab_program_aaos.pdf",
        "pdfs_rag/spine_conditioning_rehabilitation_program_aaos.pdf",
        "pdfs_rag/strength_training_guidelines_acsm.pdf",
        "pdfs_rag/who_physical_activity_2020.pdf",
    ]

    docs = []
    for path in pdf_files:
        loader = PyMuPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    split_docs = splitter.split_documents(docs)

    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    rag_prompt = PromptTemplate.from_template(
        """
        You are an assistant specialized in sports science, physiotherapy
        and strength training.

        You receive:
        - CONTEXT: excerpts from evidence-based guidelines (PDFs).
        - REQUEST: a workout plan and a short description of the user.

        Your task:
        Using ONLY the CONTEXT:
        - explain in 4–6 bullet points why the workout plan is coherent (or how to slightly adjust it) regarding:
        • training volume and intensity,
        • choice of exercise types (strength, cardio, mobility),
        • rest times between sets and between training days,
        • warm-up and cool-down,
        • injury prevention and joint protection.

        Rules:
        - DO NOT rewrite the workout plan.
        - DO NOT invent data that is not in the CONTEXT.
        - If something is unclear in the CONTEXT, say so briefly.
        - Be concise, practical, and educational.

        CONTEXT:
        {context}

        REQUEST:
        {question}

        EVIDENCE-BASED EXPLANATION:
        """
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | model
        | StrOutputParser()
    )

    return rag_chain








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

# --- EXTRA USER NOTES / PROFILE ---
st.subheader("Additional Information about Your Profile")
user_notes = st.text_area(
    "Tell the coach anything important about your profile (experience, sports you already do, lifestyle, preferences, possible limits):",
    placeholder=(
        "Example: I play football 2x/week, desk job, weak core, I dislike running on treadmill, "
        "sometimes light knee discomfort, want to improve mobility."
    )
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
        equipment_text = (
            ", ".join(selected_equipment)
            if selected_equipment
            else "No equipment (bodyweight / home workouts only)"
        )

        user_profile_text = user_notes or "No additional specific information has been indicated."

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
                "user_notes": user_profile_text,
            })
        st.session_state.program_response = response

        #RAG
        rag_explanation = None
        try:
            rag_chain = get_rag_chain()
            rag_question = f"""
                    User profile:
                    {user_profile_text}

                    Workout plan:
                    {response}

                    Explain and justify this plan using the guidelines.
                """
            rag_explanation = rag_chain.invoke(rag_question)
        except Exception as e:
            rag_explanation = f"RAG explanation unavailable (error: {e})"

        st.session_state.rag_explanation = rag_explanation

        #EcoLogits
        eco_text, eco_impacts = eco_tracker.tracked_inference(
            "Generate a short summary of this workout program for environmental tracking."
        )
        st.session_state.eco_impact = eco_impacts



if st.session_state.program_response:
    st.write("### ✅ Your personalized workout plan:")
    st.write(st.session_state.program_response)

if "rag_explanation" in st.session_state and st.session_state.rag_explanation:
    st.markdown("---")
    st.write("### 📚 Evidence-based explanation:")
    st.write(st.session_state.rag_explanation)



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

    # Format impacts using EcoLogits metrics
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

    # EQUIVALENCES SECTION
    st.markdown("---")
    st.markdown('<h3 align="center">Equivalences</h3>', unsafe_allow_html=True)
    st.markdown('<p align="center">Making this request to the LLM is equivalent to the following actions:</p>', unsafe_allow_html=True)

    # Convert impact objects to Quantity for equivalence calculations
    energy_quantity = u.Quantity(impact.energy.value, impact.energy.unit)
    gwp_quantity = u.Quantity(impact.gwp.value, impact.gwp.unit)

    # Get equivalences
    activity, activity_distance, activity_unit = get_physical_activity_equivalent(energy_quantity)
    ev_distance, ev_unit = get_ev_equivalent(energy_quantity)
    streaming_time, streaming_unit = get_streaming_equivalent(gwp_quantity)

    eq_col1, eq_col2, eq_col3 = st.columns(3)

    with eq_col1:
        st.markdown(f'<h4 align="center">{activity}</h4>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 24px; text-align: center;">{activity_distance} {activity_unit}</p>', unsafe_allow_html=True)

    with eq_col2:
        st.markdown(f'<h4 align="center">🔋 Electric Vehicle</h4>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 24px; text-align: center;">{ev_distance} {ev_unit}</p>', unsafe_allow_html=True)

    with eq_col3:
        st.markdown(f'<h4 align="center">⏯️ Video Streaming</h4>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 24px; text-align: center;">{streaming_time} {streaming_unit}</p>', unsafe_allow_html=True)