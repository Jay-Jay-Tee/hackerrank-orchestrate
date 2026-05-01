import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llm import generate_json, generate_text

# Paths for FAISS
INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index.bin")
META_PATH = os.path.join(os.path.dirname(__file__), "faiss_meta.json")

# Global instances (lazy loaded)
_embed_model = None
_faiss_index = None
_faiss_meta = None

def load_retriever():
    global _embed_model, _faiss_index, _faiss_meta
    if _embed_model is None:
        print("Loading local embedding model...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    if _faiss_index is None:
        print("Loading FAISS index...")
        _faiss_index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            _faiss_meta = json.load(f)

def retrieve_context(query: str, company: str, top_k=3) -> str:
    load_retriever()
    query_vector = _embed_model.encode([query]).astype("float32")
    distances, indices = _faiss_index.search(query_vector, top_k * 2) # Fetch extra in case of domain filtering
    
    results = []
    # Optional domain routing: if company is provided, prioritize it
    target_domain = str(company).lower().strip()
    
    for idx in indices[0]:
        if idx == -1:
            continue
        meta = _faiss_meta[idx]
        
        # Simple domain filter boost (you can make this stricter)
        if target_domain and target_domain != "none" and target_domain not in meta["product_area"]:
            continue # Skip non-matching domain if company is explicitly given
            
        results.append(meta["text"])
        if len(results) >= top_k:
            break
            
    # Fallback if domain filter was too strict
    if not results:
        for idx in indices[0][:top_k]:
            if idx != -1:
                results.append(_faiss_meta[idx]["text"])
                
    return "\n\n---\n\n".join(results)

def map_request_type(internal_type: str) -> str:
    """Maps the internal detailed request type to the allowed Hackathon output schema."""
    mapping = {
        "faq": "product_issue",
        "bug": "bug",
        "billing": "product_issue",
        "account_access": "product_issue",
        "fraud": "invalid",
        "assessment_issue": "product_issue",
        "permissions": "product_issue",
        "other": "invalid"
    }
    return mapping.get(internal_type, "product_issue")

def classify_ticket(issue: str, subject: str) -> dict:
    """Step 1: Classify the ticket according to strict categories."""
    prompt = f"""You are a terminal-based AI support triage agent.

TICKET INFO:
Subject: {subject}
Issue: {issue}

STEP 1: CLASSIFICATION
Classify the ticket into exactly one Request Type and one Product Area.
Request Type options: faq, bug, billing, account_access, fraud, assessment_issue, permissions, other
Product Area options: hackerrank, claude, visa

Respond ONLY with a JSON object. No markdown, no explanations.
{{
    "internal_request_type": "...",
    "product_area": "...",
    "decision": "answer or escalate",
    "confidence": 0.9,
    "justification": "One brief sentence explaining why you chose this request type and decision."
}}"""
    
    # generate_json forces Ollama into JSON mode
    result = generate_json(prompt)
    
    # Fallback structure if LLM hallucinates keys
    return {
        "internal_request_type": result.get("internal_request_type", "other"),
        "product_area": result.get("product_area", "unknown"),
        "decision": result.get("decision", "escalate"),
        "confidence": float(result.get("confidence", 0.0)),
        "justification": result.get("justification", "Default classification fallback.")
    }

def apply_safety_rules(classification: dict) -> dict:
    """Step 2: Deterministic Safety & Escalation Rules."""
    esc_types = ["fraud", "billing", "account_access"]
    req_type = classification.get("internal_request_type", "")
    
    if req_type in esc_types:
        classification["decision"] = "escalate"
        classification["justification"] = "Escalated for safety: " + classification.get("justification", "")
    
    if classification.get("confidence", 1.0) < 0.6:
        classification["decision"] = "escalate"
        classification["justification"] = "Escalated due to low confidence: " + classification.get("justification", "")
        
    return classification

def generate_final_response(issue: str, context: str, decision: str, justification: str) -> tuple:
    """Step 3 & 4: Response Generation."""
    
    action_instruction = "Answer the user's issue using ONLY the retrieved documents provided below."
    if decision == "escalate":
        action_instruction = "This issue has been flagged for human escalation because it is sensitive or unsupported. You MUST politely refuse to answer the user's specific request. Instead, inform them empathetically that you are transferring their ticket to a human support agent. Do NOT attempt to solve their problem."

    prompt = f"""You are a warm, friendly, and empathetic customer support agent. 

CRITICAL RULES:
1. {action_instruction}
2. Be extremely concise and direct. Keep your answer under 4 sentences. Do NOT write long essays.
3. Start with a friendly greeting (e.g., "Hi there! I'm sorry you're dealing with this.").
4. If the documents do not explicitly contain the answer, say exactly: "I cannot answer this" so we can escalate it.

DOCUMENTS:
{context}

USER ISSUE:
{issue}

Helpful Response:"""
    
    response = generate_text(prompt)
    
    if "cannot answer" in response.lower():
        decision = "escalate"
        response = "This issue requires human support. Please contact official support channels."
        justification = "Escalated because retrieved corpus lacked sufficient information to answer safely."
        
    return response, justification, decision

def process_ticket(issue: str, subject: str, company: str) -> dict:
    """Main pipeline wrapper for a single ticket."""
    # 1. Classify
    classification = classify_ticket(issue, subject)
    
    # 2. Safety Gate
    classification = apply_safety_rules(classification)
    
    # 3. Retrieve
    # We retrieve even if escalated, just in case we need it for justification context
    context = retrieve_context(f"{subject} {issue}", company)
    
    # 4. Generate
    response, justification, final_decision = generate_final_response(issue, context, classification["decision"], classification["justification"])
    
    # 5. Output Formatting
    # Ensure decision conforms to allowed 'status' values
    status = "escalated" if final_decision == "escalate" else "replied"
    
    return {
        "status": status,
        "product_area": classification["product_area"],
        "response": response,
        "justification": justification,
        "request_type": map_request_type(classification["internal_request_type"])
    }
