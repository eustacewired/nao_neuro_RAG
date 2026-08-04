# brain_server/test_rag_text.py
#
# Quick isolation test: exercises the SAME RAG chain rag_server.py uses,
# but skips FastAPI, Whisper, and audio entirely. Just type a question,
# get an answer back. Run this from the REPO ROOT (nao_neuro_RAG/), not
# from inside brain_server/, so the "brain_server/neuro_db" path resolves.

import os
from dotenv import load_dotenv

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

load_dotenv()

print("Connecting to Chroma vector database...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_db = Chroma(
    persist_directory="brain_server/neuro_db",
    embedding_function=embeddings,
    collection_name="langchain",
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.3)

system_prompt = (
    "You are an expert, empathetic assistant specializing in neurodiversity (ADHD, ASD).\n"
    "Answer the user's question using the provided context below. If the context does not "
    "contain the information, use your own broad knowledge base to answer the question "
    "accurately and concisely, but briefly mention that the detail was not found in the "
    "primary documents.\n\nContext:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("\nReady. Type a question (or 'quit').\n")

while True:
    q = input("Question: ").strip()
    if q.lower() in ("quit", "exit"):
        break
    if not q:
        continue
    response = rag_chain.invoke({"input": q})
    answer = response.get("answer", response.get("result", "No answer field returned."))
    print(f"\nAnswer: {answer}\n")

    # Show which chunks were actually retrieved, so you can sanity-check
    # whether retrieval is pulling relevant material from your PDFs.
    print("--- Retrieved context (top 3) ---")
    for i, doc in enumerate(response.get("context", [])):
        print(f"[{i+1}] {doc.page_content[:120]}...")
    print()