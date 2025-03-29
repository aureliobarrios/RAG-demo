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

app = Flask(__name__)

@app.route("/rag-response", methods=["POST"])
def rag_response():
    
    try:
        #get the prompt from the user
        data = request.get_json()
        #get the user input
        user_input = data["prompt"]
    except Exception as e:
        #if improper data then return
        return make_response(jsonify({
            "response": "data input format error",
            "message": e
        }), 400)
    
    
    return make_response(jsonify({
        "response": "this is a test"
    }), 200)