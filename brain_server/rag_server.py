# rag_server.py (Run on Host PC)
import os
import time
from fastapi import FastAPI, UploadFile, File
import whisper
from dotenv import load_dotenv

# Modern LangChain imports (Replacing deprecated syntax)
# UPDATE THESE LINES TO THE CLASSIC NAMESPACE:
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma


def invoke_with_retry(chain, payload, max_attempts=3, base_delay=3):
    """
    Gemini's free tier occasionally returns a transient 503 'high demand'
    error. This retries with a short increasing delay before giving up,
    so a single busy moment doesn't kill the whole conversation.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return chain.invoke(payload)
        except Exception as e:
            last_error = e
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"Gemini busy (attempt {attempt}/{max_attempts}), retrying...")
                time.sleep(base_delay * attempt)
            else:
                raise  # not a transient issue, don't retry
    raise last_error


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

# FIX 2: Generation LLM -- using Claude here (Gemini embeddings above stay
# unchanged, since the existing vector DB was built with them; only the
# final answer-generation step is swapped).
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.3)

# FIX 3: Define an explicit system prompt to guide the AI's behavior
system_prompt = (
    "You are NAO, a friendly robot talking OUT LOUD to curious people about "
    "neurodiversity (autism, ADHD, dyslexia, and related topics). This is a "
    "spoken conversation, not a written document, so:\n"
    "- Keep answers SHORT: 2-3 sentences maximum, ever.\n"
    "- Use warm, plain, everyday language -- no citations, no bullet points, "
    "no clinical jargon, no academic tone, even if the source material is "
    "written that way. Translate it into how a friendly person would "
    "actually say it out loud.\n"
    "- Base answers on the context below; if it's missing something, briefly "
    "say so rather than improvising heavily.\n"
    "- ALWAYS end your answer with one short, natural follow-up question "
    "inviting the person to keep the conversation going (e.g. asking if "
    "they'd like to know more about something specific, or how it relates "
    "to them or someone they know).\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# FIX 4: Build a modern RAG chain using the current LangChain standard
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

INTRO_TEXT = (
    "Hi there! I'm NAO, and I love talking about how different brains work. "
    "I can tell you about things like autism, ADHD, and dyslexia, and what "
    "makes each brain unique. What would you like to know?"
)

@app.get("/intro")
async def get_intro():
    """
    Returns a fixed, canned introduction -- no RAG/LLM call needed,
    so it's instant and doesn't depend on the vector DB or API key
    being warmed up yet. Call this once at the start of a session.
    """
    return {"answer": INTRO_TEXT}

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
        response = invoke_with_retry(rag_chain, {"input": question})
        
        # Safely extract the answer depending on library version formats
        answer = response.get("answer", response.get("result", "No answer field returned."))
        
    except Exception as e:
        print(f"❌ RAG Chain Execution Error: {e}")
        answer = "Sorry, my brain is a bit overloaded right now. Could you ask that again?"
        
    return {"question": question, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    # Start the local server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)