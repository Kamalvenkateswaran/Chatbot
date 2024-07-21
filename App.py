import streamlit as st
import pickle
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import openai
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
import os
from google import generativeai

def st_sidebar():
     with st.sidebar:
        st.title("AI Chatbot")
        st.text("""This is an AI chatbot which answers 
quetions about Deep Learning.
This app lets you maintain
chat history.
""")
        Rate = st.slider("How much do you rate our App?",0,5,0)


def session(session,user_prompt):
    for message in st.session_state.chat_session.history:
        st.markdown(message.parts[0].text)
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)

    response = st.session_state.chat_session.send_message(user_prompt)
    # Display Gemini's response
    with st.chat_message("assistant"):
        st.markdown(response.text)
        
def main():
    load_dotenv() # To load api key
    
    # MAIN LAYOUT

    st.set_page_config(
        page_title="First App",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🤖  AI - ChatBot")

    # Setting up apis and models
    Google_API_KEY = os.getenv("GOOGLE_API_KEY")
    generativeai.configure(api_key=Google_API_KEY)

    llm = generativeai.GenerativeModel("gemini-pro")

    st_sidebar() # Initializing sidebar

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = llm.start_chat(history=[])
    
    user_prompt = st.chat_input("Ask Quetions about deep learning")
    if user_prompt is not None:
        session("chat_session",user_prompt=user_prompt)



main()


