import streamlit as st
st.title("Fit Coach LLM")

user_age = st.text_input("Enter your age:")
user_gender = st.text_input("Enter your gender:")
user_height = st.text_input("Enter your height:")
user_weight = st.text_input("Enter your weight:")
user_daily_time = st.text_input("Enter your time per day:")

st.button("Send")