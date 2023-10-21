from __future__ import annotations

import os
import time
from pathlib import Path

import openai
import streamlit as st
from dotenv import load_dotenv
from loguru import logger

from config import Config
from utils import create_vectordb
from utils import get_qa_chain
from utils import load_file
from utils import load_url
from utils import save_file_locally
load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

uploaded_file = st.sidebar.file_uploader('Upload a file', type=['pdf', 'txt'])
doc_url = st.sidebar.text_input('Or enter a URL to a document')

if uploaded_file is not None and doc_url != '':
    st.sidebar.error('Please choose one or the other')
    st.stop()


def set_status():
    if uploaded_file is None:
        # Path(Config.vectorstore_path).unlink(missing_ok=True)
        st.sidebar.info('Upoad a file to start a conversation')
    else:
        st.sidebar.info(f'Let"s talk to {Path(uploaded_file.name)}')


def process_data(data, data_type):
    if 'context' not in st.session_state:
        if data_type == 'file':
            upath = f'docs/{uploaded_file.name}'
            save_file_locally(data, upath)
            load_file(upath)
        else:
            load_url(data)
        st.session_state['context'] = True


def process_uploaded_doc():
    if 'context' not in st.session_state:
        loader = Uns
        st.session_state['context'] = True


set_status()


if uploaded_file is not None or doc_url != '':
    if uploaded_file is not None:
        process_data(uploaded_file, data_type='file')
    else:
        process_data(doc_url, data_type='url')

    qr_chain = get_qa_chain()

    if prompt := st.chat_input('Send a message'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            full_response = ''
            assistant_response = qr_chain({'question': prompt})['answer']
            logger.info(f'question {prompt}')
            logger.info(f'assistant response {assistant_response}')

            for chunk in assistant_response.split():
                full_response += chunk + ' '
                time.sleep(0.01)

                message_placeholder.markdown(full_response + '▌')
            message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {'role': 'assistant', 'content': full_response},
        )
