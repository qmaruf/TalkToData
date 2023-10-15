from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import pickle
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.prompts.prompt import PromptTemplate



# def get_memory():
#     memory = ConversationBufferMemory(
#         memory_key='chat_history',
#         return_messages=True
#     )
#     return memory

# def get_text_splitter():
#     return RecursiveCharacterTextSplitter(
#         chunk_size=Config.chunk_size,
#         chunk_overlap=Config.chunk_overlap,
#         length_function=len
#     )

# def get_vectordb(splits, embedding):
#     vectordb = Chroma(persist_directory=Config.vectordb_persist_directory)
#     vectordb.delete_collection()
#     vectordb = Chroma.from_documents(
#         documents=splits,
#         embedding=embedding,
#         persist_directory=Config.vectordb_persist_directory,
#         collection_name='HP1'
#     )
#     return vectordb

# def get_embedding():
#     return OpenAIEmbeddings()

# def get_prompt():
#     template = """You are an AI assistant for answering question about a givent context. You are given
#     the following extracted parts of a long document and a question. You must answer the question
#     based on the context. Also explain your answer based on the context. If you don't find an answer, just say 'I don't know'. Don't make up an answer'.
#     Context: {context}
#     Question: {question}
#     Answer:"""
#     qa_chain_prompt = PromptTemplate.from_template(template)
#     return qa_chain_prompt

# def get_qa_chain(llm, retriever, prompt):
#     qa_chain = RetrievalQA.from_chain_type(
#         llm,
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt}
#     )
#     return qa_chain

# def get_retriever(vectordb):
#     retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 10, 'fetch_k': 50})
#     return retriever

# def get_llm():
#     llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
#     return llm

# def get_cr_chain(vectordb):
#     llm = get_llm()
#     retriever = get_retriever(vectordb)
#     memory = get_memory()
#     prompt = get_prompt()
#     cr_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         combine_docs_chain_kwargs={"prompt": prompt},
#         verbose=True,
#         memory=memory
#     )
#     return cr_chain

from langchain.document_loaders import UnstructuredFileLoader


def get_prompt():
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Ask follow up question if the user question is not clear. With each answer, explain your answer based on the context. 
    {context}
    Question: {question}
    Answer:"""
    qa_prompt = PromptTemplate(template=template, input_variables=[
                            "question", "context"])
    return qa_prompt                    

def create_vectordb(file_path):

    loader = UnstructuredFileLoader(file_path)
    raw_documents = loader.load()

    print ("Splitting text...")
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
    )
    documents = text_splitter.split_documents(raw_documents)
    
    print("Creating vectorstore...")
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    with open(Config.vectorstore_path, "wb") as f:
        pickle.dump(vectorstore, f)


def load_retriever():
    with open(Config.vectorstore_path, "rb") as f:
        vectorstore = pickle.load(f)
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever




# def get_basic_qa_chain():
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#     retriever = load_retriever()
#     memory = ConversationBufferMemory(
#         memory_key="chat_history", return_messages=True)
#     model = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory)
#     return model

def get_basic_qa_chain():
    llm = ChatOpenAI(model_name=Config.chatgpt_model_name, temperature=0)
    retriever = load_retriever()
    prompt = get_prompt()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt})
    return model
