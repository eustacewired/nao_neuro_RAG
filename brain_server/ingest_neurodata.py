# brain_server/ingest_data.py
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables (your GOOGLE_API_KEY)
load_dotenv()

# Configuration
DATA_DIR = "./data"
DB_DIR = "./neuro_db"

def build_cognitive_memory():
    print(f"Scanning {DATA_DIR} for neurodiversity documents...")
    
    # 1. Load all PDFs from the data directory
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    
    if not documents:
        print("Error: No PDFs found in the 'data' folder. Please add some before running.")
        return
        
    print(f"Loaded {len(documents)} pages. Chunking text...")

    # 2. Split the documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks. Initializing Google embeddings engine...")

    # 3. Initialize embeddings and ChromaDB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Initialize an empty Chroma database
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # 4. Process chunks in batches to avoid 429 Rate Limits
    BATCH_SIZE = 90  # Stay safely under the 100 requests-per-minute limit
    
    print(f"Processing text chunks in batches of {BATCH_SIZE} with cooling periods...")
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"Sending batch {i//BATCH_SIZE + 1} ({len(batch)} chunks) to Gemini...")
        
        # Add the current batch of text chunks to the database
        vector_db.add_documents(documents=batch)
        
        # If there are more chunks left, pause to reset Google's 1-minute quota window
        if i + BATCH_SIZE < len(chunks):
            print("Approaching API limit. Pausing for 65 seconds to reset quota...")
            time.sleep(65)
            
    print(f"Success! Vector database fully initialized and saved to {DB_DIR}.")

if __name__ == "__main__":
    build_cognitive_memory()
