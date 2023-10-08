from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=Config.chunk_size,
        chunk_overlap=Config.chunk_overlap,
        length_function=len
    )

def get_vectordb(splits, embedding):
    vectordb = Chroma(persist_directory=Config.vectordb_persist_directory)
    vectordb.delete_collection()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=Config.vectordb_persist_directory,
        collection_name='HP1'
    )
    return vectordb

def get_embedding():
    return OpenAIEmbeddings()
