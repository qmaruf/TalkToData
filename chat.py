from __future__ import annotations

import time
from pathlib import Path

import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from loguru import logger

from config import Config
from utils import create_vectordb
from utils import get_qa_chain

openai.api_key = st.secrets['OPENAI_API_KEY']
Path('docs').mkdir(parents=True, exist_ok=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

uploaded_file = st.sidebar.file_uploader('Upload a file', type=['pdf', 'txt'])


def set_status():
    if uploaded_file is None:
        Path(Config.vectorstore_path).unlink(missing_ok=True)
        st.sidebar.info('Upoad a file to start a conversation')
    else:
        st.sidebar.info(f'Let"s talk to {Path(uploaded_file.name)}')


def process_uploaded_file(uploaded_file):
    if 'context' not in st.session_state:
        logger.info(f'file uploaded {uploaded_file}')
        upath = f'uploads/{uploaded_file.name}'
        logger.info(f'file saved to {upath}')

        with open(upath, 'wb') as hndl:
            hndl.write(uploaded_file.getbuffer())

        create_vectordb(upath)
        st.session_state['context'] = True


set_status()


if uploaded_file is not None:
    process_uploaded_file(uploaded_file)
    qr_chain = get_qa_chain()

    if prompt := st.chat_input('Send a message'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            full_response = ''
            assistant_response = qr_chain({'question': prompt})['answer']
            logger.info(f'assistant response {assistant_response}')

            for chunk in assistant_response.split():
                full_response += chunk + ' '
                time.sleep(0.01)

                message_placeholder.markdown(full_response + '▌')
            message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {'role': 'assistant', 'content': full_response},
        )
