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

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_PATH = "./data/"
CHROMA_PATH = "./chroma/"

app = Flask(__name__)

@app.route("/rag-response", methods=["POST"])
def rag_response():
    
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
    #first we must load in the documents
    document_loader = PyPDFDirectoryLoader(DATA_PATH)

    documents = document_loader.load()
    
    return make_response(jsonify({
        "response": "this is a test"
    }), 200)