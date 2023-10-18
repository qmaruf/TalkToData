from __future__ import annotations

import pickle

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores.faiss import FAISS

from config import Config


def get_prompt():
    """
    This function creates a prompt template that will be used to generate the prompt for the model.
    """
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Ask follow up question if the user question is not clear. You must explain the reasoning behind your answer.
    {context}
    Question: {question}
    Answer:"""
    qa_prompt = PromptTemplate(
        template=template, input_variables=[
            'question', 'context',
        ],
    )
    return qa_prompt


def create_vectordb(file_path):
    """
    This function creates a vectorstore from a file.
    """
    loader = UnstructuredFileLoader(file_path)
    raw_documents = loader.load()

    print('Splitting text...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    documents = text_splitter.split_documents(raw_documents)

    print('Creating vectorstore...')

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    with open(Config.vectorstore_path, 'wb') as f:
        pickle.dump(vectorstore, f)


def load_retriever():
    """
    This function loads the vectorstore from a file.
    """
    with open(Config.vectorstore_path, 'rb') as f:
        vectorstore = pickle.load(f)
    retriever = VectorStoreRetriever(
        vectorstore=vectorstore, search_type='similarity',
    )
    return retriever


def get_qa_chain():
    """
    This function creates a question answering chain.
    """
    llm = ChatOpenAI(model_name=Config.chatgpt_model_name, temperature=0)
    retriever = load_retriever()
    prompt = get_prompt()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True,
    )
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt},
    )
    return model
