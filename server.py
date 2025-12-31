
"""
Enhanced Medical RAG Chatbot using OpenAI API
- Fixed indentation issues
- Improved retrieval from PDFs
- Better context usage
"""

import os
import time
import json
import requests
import re
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import uuid
import redis

# ---------------- Load environment variables ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# ---------------- Configuration ----------------
INDEX_DIR = os.getenv("INDEX_DIR", "vector_index")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "metadata.json")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-mpnet-base-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K = int(os.getenv("TOP_K", 8))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", 4))
MAX_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))

# ---------------- FastAPI App ----------------
app = FastAPI(title="Enhanced Medical RAG Chatbot (OpenAI GPT)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- Models ----------------
class Query(BaseModel):
    question: str
    location: Optional[str] = None
    session_id: Optional[str] = None

class ChatSession(BaseModel):
    session_id: str
    history: List[Dict[str, str]]
    created_at: float
    last_accessed: float

# ---------------- Initialize Components ----------------
print("Loading FAISS index...")
if not os.path.exists(FAISS_PATH) or not os.path.exists(META_PATH):
    raise RuntimeError("FAISS index or metadata not found. Run ingest_docs.py first.")

index = faiss.read_index(FAISS_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Loading embedding model...")
embedder = SentenceTransformer(EMB_MODEL)

print("Loading re-ranker model...")
reranker = CrossEncoder(RERANKER_MODEL)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Redis for session storage
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
    print("Connected to Redis successfully")
except redis.ConnectionError:
    print("Redis not available, using in-memory storage")
    redis_client = None

# In-memory fallback storage
session_storage = {}
BM25_INDEX = None

# ---------------- Helper Functions ----------------
def simple_tokenize(text: str) -> List[str]:
    """Simple tokenizer that doesn't require NLTK"""
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    return [token for token in tokens if token.strip()]

def initialize_bm25():
    """Initialize BM25 index for hybrid search with fallback tokenization"""
    global BM25_INDEX
    if BM25_INDEX is None:
        corpus = [doc['text'] for doc in metadata]
        tokenized_corpus = [simple_tokenize(doc) for doc in corpus]
        
        try:
            from rank_bm25 import BM25Okapi
            BM25_INDEX = BM25Okapi(tokenized_corpus)
            print("BM25 index initialized successfully")
        except ImportError:
            print("rank_bm25 not available, using vector search only")
            BM25_INDEX = None

def classify_query_intent(query: str) -> str:
    """Use LLM to classify query intent for better safety"""
    query_lower = query.lower()
    
    emergency_keywords = ["chest pain", "difficulty breathing", "severe bleeding", "suicidal", "heart attack", "stroke"]
    urgent_keywords = ["high fever", "severe pain", "can't breathe", "unconscious", "broken bone"]
    greeting_keywords = ["hello", "hi", "hey", "good morning", "good afternoon"]
    
    if any(keyword in query_lower for keyword in emergency_keywords):
        return "EMERGENCY"
    elif any(keyword in query_lower for keyword in urgent_keywords):
        return "URGENT"
    elif any(keyword in query_lower for keyword in greeting_keywords):
        return "GREETING"
    
    try:
        prompt = f"""
        Classify the user's medical query into exactly one of these categories:
        - EMERGENCY: Life-threatening symptoms
        - URGENT: Serious symptoms needing prompt attention
        - ROUTINE: General medical questions
        - GREETING: Hello, hi, etc.
        - OTHER: Non-medical questions

        Query: "{query}"

        Return only the category name.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical triage classifier. Return only the category name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        category = response.choices[0].message.content.strip().upper()
        return category if category in ["EMERGENCY", "URGENT", "ROUTINE", "GREETING", "OTHER"] else "ROUTINE"
    except Exception as e:
        print(f"OpenAI classification failed: {e}, defaulting to ROUTINE")
        return "ROUTINE"

def get_session(session_id: str) -> ChatSession:
    """Get or create chat session"""
    if redis_client:
        try:
            session_data = redis_client.get(f"session:{session_id}")
            if session_data:
                return ChatSession(**json.loads(session_data))
        except redis.RedisError:
            print("Redis error, using in-memory storage")
    
    if session_id in session_storage:
        return session_storage[session_id]
    
    new_session = ChatSession(
        session_id=session_id,
        history=[],
        created_at=time.time(),
        last_accessed=time.time()
    )
    
    session_storage[session_id] = new_session
    return new_session

def update_session(session: ChatSession):
    """Update session storage"""
    session.last_accessed = time.time()
    session_storage[session.session_id] = session

def retrieve_context_enhanced(query: str, top_k: int = TOP_K) -> List[dict]:
    """Enhanced retrieval with multiple strategies"""
    strategies = []
    
    # Strategy 1: Original vector search
    query_embedding = embedder.encode([query], convert_to_numpy=True)[0].astype("float32")
    D, I = index.search(np.array([query_embedding]), top_k * 3)
    vector_results = [metadata[idx] for idx in I[0] if 0 <= idx < len(metadata)]
    strategies.extend(vector_results)
    
    # Strategy 2: Keyword-based fallback
    query_keywords = simple_tokenize(query)
    keyword_matches = []
    for doc in metadata:
        doc_text = doc['text'].lower()
        if any(keyword in doc_text for keyword in query_keywords):
            keyword_matches.append(doc)
    
    strategies.extend(keyword_matches[:top_k])
    
    # Strategy 3: Source-based (if you know your PDF topics)
    cancer_terms = ['cancer', 'oncology', 'chemotherapy', 'tumor', 'malignant']
    mental_terms = ['depression', 'mental', 'therapy', 'psychology', 'anxiety', 'mood']
    
    if any(term in query.lower() for term in cancer_terms):
        cancer_docs = [doc for doc in metadata if any(term in doc['text'].lower() for term in cancer_terms)]
        strategies.extend(cancer_docs[:2])
    
    if any(term in query.lower() for term in mental_terms):
        mental_docs = [doc for doc in metadata if any(term in doc['text'].lower() for term in mental_terms)]
        strategies.extend(mental_docs[:2])
    
    # Remove duplicates and return top results
    unique_results = []
    seen_texts = set()
    for result in strategies:
        if result['text'] not in seen_texts:
            unique_results.append(result)
            seen_texts.add(result['text'])
    
    return unique_results[:FINAL_TOP_K]

def build_enhanced_prompt(question: str, contexts: List[dict], conversation_history: List[dict] = None) -> str:
    """Build prompt that forces using context"""
    context_text = "\n\n".join([
        f"📄 Source: {c['source']}\nContent: {c['text']}"
        for c in contexts
    ]) if contexts else "No specific medical context available in knowledge base."

    return f"""IMPORTANT: You are a medical assistant with access to specific medical documents. 
You MUST use the context below from uploaded medical PDFs to answer the question.

If the context doesn't contain the answer, say: "My medical sources don't cover this specific question, but generally..." 
and provide brief general advice while recommending professional consultation.

CONTEXT FROM UPLOADED MEDICAL PDFS:
{context_text}

USER QUESTION: {question}

YOUR RESPONSE (based strictly on context when possible):"""

def generate_emergency_response(query: str, location: str = None) -> dict:
    """Generate emergency response"""
    emergency_advice = """
    🚨 **EMERGENCY ALERT** 
    
    Based on your symptoms, this may be a medical emergency. Please:
    
    1. Call your local emergency number immediately (911, 112, etc.)
    2. Go to the nearest emergency department
    3. Do not attempt to self-treat serious symptoms
    
    If you're with someone, have them call for help or drive you to the hospital.
    """
    
    emergency_suggestions = [
        "Call emergency services immediately",
        "Go to the nearest emergency department",
        "Do not delay seeking medical attention"
    ]
    
    return {
        "answer": emergency_advice,
        "suggestions": emergency_suggestions,
        "sources": [],
        "meta": {"gen_time_s": 0, "intent": "EMERGENCY"},
        "disclaimer": "THIS IS AN EMERGENCY. SEEK IMMEDIATE MEDICAL ATTENTION."
    }

def generate_answer_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """Generate answer with enhanced error handling"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a safe medical assistant. Be conservative and cite sources when possible."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please consult a healthcare professional for accurate medical advice."

# ---------------- Debug Endpoints ----------------
@app.get("/debug/search")
async def debug_search(query: str):
    """Debug endpoint to see what's being retrieved"""
    contexts = retrieve_context_enhanced(query)
    
    return {
        "query": query,
        "found_contexts": len(contexts),
        "contexts": contexts,
        "sources": list({c["source"] for c in contexts})
    }

@app.get("/debug/metadata")
async def debug_metadata():
    """See what documents are in your index"""
    sources = list({m["source"] for m in metadata})
    return {
        "total_chunks": len(metadata),
        "sources": sources,
        "sample_chunks": metadata[:3]
    }

# ---------------- API Endpoints ----------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse("static/index.html")

@app.post("/chat")
async def chat(q: Query, request: Request):
    try:
        question = (q.question or "").strip()
        location = q.location
        client_ip = request.client.host

        session_id = q.session_id or f"session_{client_ip}_{uuid.uuid4().hex[:8]}"
        session = get_session(session_id)

        if not question:
            raise HTTPException(status_code=400, detail="Question required")

        start_time = time.time()
        
        intent = classify_query_intent(question)
        
        if intent == "EMERGENCY":
            response = generate_emergency_response(question, location)
            gen_time = time.time() - start_time
            response["meta"]["gen_time_s"] = gen_time
            return response

        elif intent == "GREETING":
            greeting_response = {
                "answer": "Hello! I'm your medical AI assistant. I can help with general health information, but remember I'm not a doctor. How can I help you today?",
                "suggestions": ["Ask about symptoms", "Request general health information"],
                "sources": [],
                "meta": {"gen_time_s": 0, "intent": "GREETING"},
                "disclaimer": "I provide informational guidance only. Always consult healthcare professionals."
            }
            return greeting_response

        contexts = retrieve_context_enhanced(question)
        
        conversation_history = session.history[-3:] if session.history else None
        prompt = build_enhanced_prompt(question, contexts, conversation_history)
        
        answer = generate_answer_openai(prompt)
        gen_time = time.time() - start_time

        session.history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.time(),
            "sources": [c['source'] for c in contexts]
        })
        update_session(session)

        safe_suggestions = [
            "Monitor your symptoms carefully",
            "Contact a healthcare professional for personalized advice",
            "Seek emergency care if symptoms worsen"
        ]

        sources = list({c["source"] for c in contexts})

        return {
            "answer": answer,
            "suggestions": safe_suggestions,
            "sources": sources,
            "meta": {
                "gen_time_s": gen_time,
                "intent": intent,
                "session_id": session_id,
                "chunks_retrieved": len(contexts)
            },
            "disclaimer": "🚨 IMPORTANT: I am an AI assistant, not a medical professional. Always consult qualified healthcare providers."
        }
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/session/{session_id}")
async def get_session_status(session_id: str):
    """Get session information"""
    try:
        session = get_session(session_id)
        return {
            "session_id": session.session_id,
            "message_count": len(session.history),
            "last_accessed": session.last_accessed
        }
    except:
        return {"error": "Session not found"}

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session history"""
    if session_id in session_storage:
        del session_storage[session_id]
    return {"status": "session_cleared"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# ---------------- Main ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)