#imports
import os
from groq import Groq
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response

#load in the environment
load_dotenv()

#load in your environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_PATH = "./data/"
CHROMA_PATH = "./chroma/"

#create the flask app
app = Flask(__name__)

#create the flask app route
@app.route("/rag-response", methods=["POST"])
def rag_response():
    #try to get the user input
    try:
        #get the prompt from the user
        data = request.get_json()
        #get the user input
        prompt = data["prompt"]
    except Exception as e:
        #if improper data then return
        return make_response(jsonify({
            "response": "data input format error",
            "message": e
        }), 400)
    #first we must load in the documents using a PDF loader
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    #load in the content
    documents = document_loader.load()

    #split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex = False
    )
    #split the content into chunks
    chunks = text_splitter.split_documents(documents)

    #get the ollama embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = embeddings
    )
    
    return make_response(jsonify({
        "response": "this is a test"
    }), 200)