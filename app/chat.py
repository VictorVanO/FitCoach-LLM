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
model = ChatMistralAI(model="mistral-small-latest", temperature=0.3)
output_parser = StrOutputParser()

chain = prompt | model | output_parser


st.title("Fit Coach LLM")

user_age = st.text_input("Enter your age:")
user_gender = st.text_input("Enter your gender:")
user_height = st.text_input("Enter your height (in cm):")
user_weight = st.text_input("Enter your weight (in kg):")
user_goals = st.text_input("Enter your goals")
user_daily_time = st.text_input("Enter your time per day (in minutes):")
user_days_per_week = st.text_input("Enter your available days per week:")

if st.button("Send"):
    response = chain.invoke({
        "gender": user_gender,
        "age": user_age,
        "height": user_height,
        "weight": user_weight,
        "goals": user_goals,
        "daily_time": user_daily_time,
        "days_per_week": user_days_per_week
    })
    st.write("Mistral's answer:", response)