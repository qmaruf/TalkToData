import openai
import streamlit as st
from loguru import logger
from pathlib import Path
from pathlib import Path
from langchain.document_loaders import TextLoader, PyPDFLoader
from utils import get_basic_qa_chain, create_vectordb
import random
import time
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


file_uploaded = False

Path('uploads').mkdir(parents=True, exist_ok=True)

st.title("Talk to Your Data")

openai.api_key = st.secrets["OPENAI_API_KEY"]


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

uploaded_file = st.sidebar.file_uploader("Upload a file", type=["pdf", "txt"])


def process_uploaded_file(uploaded_file):
    if 'context_added' not in st.session_state:
        logger.info(f'file uploaded {uploaded_file}')
        upath = f'uploads/{uploaded_file.name}' 
        logger.info(f'file saved to {upath}')
        
        with open(upath,"wb") as hndl:
            hndl.write(uploaded_file.getbuffer())

        create_vectordb(upath)
        st.session_state['context_added'] = True
    

if uploaded_file is not None:
    qr_chain = get_basic_qa_chain()
    
    if prompt := st.chat_input("Send a message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = qr_chain({'question': prompt})['answer']
            logger.info(f'assistant response {assistant_response}')

            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.01)

                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})