from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
import os

# Initialize global variables
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH = "chroma_db"

def preprocess_text(file_path):
    """Loads and splits the medical dialogue text file into chunks."""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return text_splitter.split_text(text)

def store_in_vector_db(chunks):
    """Stores text chunks into a Chroma vector database."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = Chroma.from_texts(chunks, embeddings, persist_directory=VECTOR_DB_PATH)
    vector_db.persist()
    print("Vector database created successfully.")

def retrieve_relevant_context(patient_input):
    """Retrieves relevant past doctor-patient discussions based on symptoms."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    retriever = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings).as_retriever()
    docs = retriever.get_relevant_documents(patient_input)
    return "\n".join([doc.page_content for doc in docs])

def generate_prompt(patient_input, context):
    """Generates the next best question to ask the patient using Llama model."""
    prompt = f"""
    You are a virtual medical assistant analyzing symptoms.
    Based on the patient’s symptoms: "{patient_input}"
    And past doctor-patient discussions:
    {context}
    What is the most relevant next question to ask the patient?
    """
    return prompt