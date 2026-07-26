# rag_server.py (Run on Host PC)
import os
from fastapi import FastAPI, UploadFile, File
import whisper
from dotenv import load_dotenv

# Modern LangChain imports (Replacing deprecated syntax)
# UPDATE THESE LINES TO THE CLASSIC NAMESPACE:
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma


# Load environment variables (API Keys)
load_dotenv()

app = FastAPI()

print("Loading OpenAI Whisper Speech-to-Text model...")
stt_model = whisper.load_model("base")

print("Connecting to your local Chroma Neurodiversity Vector Database...")
# FIX 1: Match your ingestion engine! Use Gemini Embeddings, not OpenAI
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# Force Chroma to read the correct, legacy default collection name
vector_db = Chroma(
    persist_directory="brain_server/neuro_db", 
    embedding_function=embeddings,
    collection_name="langchain"
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 relevant chunks

# FIX 2: Set up Gemini LLM instead of OpenAI Chat
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)

# FIX 3: Define an explicit system prompt to guide the AI's behavior
system_prompt = (
    "You are an expert, empathetic assistant specializing in neurodiversity (ADHD, ASD).\n"
    "Answer the user's question using the provided context below. If the context does not contain "
    "the information, use your own broad knowledge base to answer the question accurately and concisely, "
    "but briefly mention that the detail was not found in the primary documents.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# FIX 4: Build a modern RAG chain using the current LangChain standard
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.post("/ask")
async def ask_nao(audio: UploadFile = File(...)):
    # 1. Save incoming audio temporarily
    temp_filename = "temp.wav"
    with open(temp_filename, "wb") as f:
        f.write(await audio.read())
    
    print("Transcribing audio question from NAO...")
    # 2. Convert Speech to Text
    result = stt_model.transcribe(temp_filename)
    question = result.get("text", "").strip()
    
    # Clean up the temporary file immediately
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    if not question:
        return {"question": "", "answer": "I couldn't hear or understand the audio."}
        
    print(f"NAO Asked: '{question}' -> Querying RAG Database...")
    
    try:
        # 3. Query the Modern RAG Pipeline using clean key fallbacks
        # create_retrieval_chain expects 'input' key for user queries
        response = rag_chain.invoke({"input": question})
        
        # Safely extract the answer depending on library version formats
        answer = response.get("answer", response.get("result", "No answer field returned."))
        
    except Exception as e:
        print(f"❌ RAG Chain Execution Error: {e}")
        answer = "I encountered an structural error parsing my vector database."
        
    return {"question": question, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    # Start the local server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
