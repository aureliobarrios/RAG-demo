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

    #create the chroma database
    db = Chroma(
        persist_directory = CHROMA_PATH,
        embedding_function = embeddings
    )
    
    #set the ids
    last_page_id = None
    current_chunk_index = 0
    #loop through chunks and add ids
    for chunk in chunks:
        #get the source of the chunk
        source = chunk.metadata.get("source")
        #get the page of the chunk
        page = chunk.metadata.get("page")
        #create an id for that chunk
        current_page_id = f"{source}:{page}"
        #ensures that we have proper indexing
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        #create a chunk id
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        #create a last page id
        last_page_id = current_page_id
        #add the current id to the current chunks metadata
        chunk.metadata["id"] = chunk_id
    
    #take note of all the existing items in db
    existing_items = db.get(include=[])
    #take note of all existing ids in the db
    existing_ids = set(existing_items["ids"])

    
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        db_message = f"Added new documents: {len(new_chunks)}"
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        db_message = "No new documents to add"
    
    return make_response(jsonify({
        "response": "this is a test",
        "db-message": db_message
    }), 200)