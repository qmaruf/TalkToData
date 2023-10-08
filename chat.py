from langchain.document_loaders import TextLoader, PyPDFLoader
import os
import gradio as gr
import random
import time
from config import Config
from loguru import logger
import openai
from utils import get_text_splitter, get_vectordb, get_embedding
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = os.environ['OPENAI_API_KEY']


def upload_file(files):
    file_paths = [file.name for file in files]
    print (file_paths)
    if file_paths.endswith(".pdf"):
        loader = PDFLoader(file_paths)
    else:
        loader = TextLoader(file_paths)
    document = loader.load()

    text_splitter = get_text_splitter()
    embedding = get_embedding()
    splits = text_splitter.split_documents(document)
    vectordb = get_vectordb(splits, embedding)
    return vectordb


def llm_interaction(message):
    pass


with gr.Blocks() as demo:
    file_output = gr.File()
    upload_button = gr.UploadButton("Click to Upload a File", file_count="single")
    upload_button.upload(upload_file, upload_button, file_output)
    logger.info("File uploaded")
    # logger.info(vectordb)
    chatbot = gr.Chatbot()
    msg = gr.Textbox()

    def respond(message, chat_history):
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()