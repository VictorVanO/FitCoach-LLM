import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load Mistral API KEY information
load_dotenv(override=True)

# ChatMistralAI configuration
template = """You are an expert workout coach with 10 years of experience.
Please create a workout program with the following informations about the user:

Gender: {gender}
Age: {age}
Height: {height} cm
Weight: {weight} kg
Goals: {goals}
Daily available time for training: {daily_time} minutes
Days per week available for training: {days_per_week} days

Format your response as:
- Workout Plan:
- Trained muscles:
- Workout Duration:
- Frequency per week:
"""

# Create prompt and chain
prompt = PromptTemplate.from_template(template)
model = ChatMistralAI(model="magistral-small-latest", temperature=0.3)
output_parser = StrOutputParser()

chain = prompt | model | output_parser

st.title("Fit Coach LLM")

# Improved input fields with better UX
user_gender = st.selectbox(
    "Select your gender:",
    options=["Male", "Female", "Other"],
    index=0
)

user_age = st.number_input(
    "Enter your age:",
    min_value=1,
    max_value=120,
    value=25,
    step=1
)

user_height = st.number_input(
    "Enter your height (in cm):",
    min_value=100,
    max_value=250,
    value=180,
    step=1
)

user_weight = st.number_input(
    "Enter your weight (in kg):",
    min_value=30,
    max_value=200,
    value=75,
    step=1
)

user_goals = st.text_input(
    "Enter your fitness goals (e.g., build muscle, lose weight, improve endurance):",
    value="build muscle"
)

user_daily_time = st.number_input(
    "Enter your daily training time (in minutes):",
    min_value=10,
    max_value=180,
    value=60,
    step=5
)

user_days_per_week = st.number_input(
    "Enter your available training days per week:",
    min_value=1,
    max_value=7,
    value=5,
    step=1
)

if st.button("Generate Workout Plan"):
    response = chain.invoke({
        "gender": user_gender,
        "age": str(user_age),
        "height": str(user_height),
        "weight": str(user_weight),
        "goals": user_goals,
        "daily_time": str(user_daily_time),
        "days_per_week": str(user_days_per_week)
    })
    st.write("Your personalized workout plan:")
    st.write(response)