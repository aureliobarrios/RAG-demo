import os
from groq import Groq
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#function to invoke groq response
def groq_invoke(prompt):
    
    client = Groq(
        api_key=GROQ_API_KEY,
    )
    
    chat_completion = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ],
        model = "llama3-8b-8192"
    )
    
    return chat_completion.choices[0].message.content