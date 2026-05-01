import os
import json
import re
import faiss  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
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
    if _faiss_meta is None:
        print("Loading FAISS index...")
        _faiss_index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            _faiss_meta = json.load(f)

def retrieve_context(query: str, company: str, top_k=3) -> str:
    load_retriever()
    query_vector = _embed_model.encode([query]).astype("float32")
    distances, indices = _faiss_index.search(query_vector, top_k * 2) # Fetch extra in case of domain filtering

    results = []
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


def _infer_product_area(issue: str, subject: str, company: str, llm_value: str) -> str:
    valid = {"hackerrank", "claude", "visa"}
    llm_area = str(llm_value).lower().strip()
    if llm_area in valid:
        return llm_area

    company_area = str(company).lower().strip()
    if company_area in valid:
        return company_area

    text = f"{subject} {issue}".lower()
    if "visa" in text or "card" in text or "merchant" in text:
        return "visa"
    if "claude" in text or "bedrock" in text or "anthropic" in text:
        return "claude"
    return "hackerrank"


def _heuristic_request_type(issue: str, subject: str) -> str:
    text = f"{subject} {issue}".lower()

    fraud_terms = ["fraud", "stolen", "identity theft", "scam", "phishing", "unauthorized"]
    billing_terms = ["refund", "charge", "billing", "payment", "invoice", "subscription", "money", "cash"]
    account_terms = ["login", "access", "password", "locked", "seat", "workspace owner", "admin removed", "account"]
    bug_terms = ["down", "not working", "error", "failing", "failed", "stopped", "issue", "broken"]
    assessment_terms = ["assessment", "test", "submission", "candidate", "certificate", "compatibility", "reschedule"]
    permissions_terms = ["remove user", "role", "permission", "interviewer", "owner", "invite", "team"]

    if any(term in text for term in fraud_terms):
        return "fraud"
    if any(term in text for term in billing_terms):
        return "billing"
    if any(term in text for term in account_terms):
        return "account_access"
    if any(term in text for term in permissions_terms):
        return "permissions"
    if any(term in text for term in assessment_terms):
        return "assessment_issue"
    if any(term in text for term in bug_terms):
        return "bug"
    return "faq"

def classify_ticket(issue: str, subject: str, company: str) -> dict:
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
    "confidence": 0.9
}}"""
    
    # generate_json forces the model into JSON mode
    result = generate_json(prompt)

    llm_type = str(result.get("internal_request_type", "other")).lower().strip()
    llm_conf = float(result.get("confidence", 0.0))
    heuristic_type = _heuristic_request_type(issue, subject)

    valid_types = {"faq", "bug", "billing", "account_access", "fraud", "assessment_issue", "permissions", "other"}
    if llm_type not in valid_types or llm_type == "other" or llm_conf < 0.45:
        internal_type = heuristic_type
        confidence = max(llm_conf, 0.7)
    else:
        internal_type = llm_type
        confidence = llm_conf

    product_area = _infer_product_area(issue, subject, company, result.get("product_area", "unknown"))
    
    # Fallback structure if LLM hallucinates keys
    justification = f"Routed to {product_area} as a {internal_type} request."

    return {
        "internal_request_type": internal_type,
        "product_area": product_area,
        "decision": "answer",
        "confidence": confidence,
        "justification": justification
    }

def apply_safety_rules(classification: dict) -> dict:
    """Step 2: Deterministic Safety & Escalation Rules."""
    esc_types = ["fraud", "billing", "account_access"]
    req_type = classification.get("internal_request_type", "")

    classification["decision"] = "answer"

    if req_type in esc_types:
        classification["decision"] = "escalate"
        classification["justification"] = "Escalated for safety: " + classification.get("justification", "")

    # Only escalate unknown intent when confidence is very low.
    if req_type == "other" and classification.get("confidence", 1.0) < 0.55:
        classification["decision"] = "escalate"
        classification["justification"] = "Escalated due to low confidence: " + classification.get("justification", "")
        
    return classification

def _clean_context(context: str) -> str:
    cleaned_chunks = []

    for chunk in context.split("\n\n---\n\n"):
        article = chunk.strip()
        if not article:
            continue

        if "\n---\n" in article:
            parts = [part.strip() for part in article.split("\n---\n") if part.strip()]
            if len(parts) >= 3:
                article = parts[2]
            elif len(parts) >= 2:
                article = parts[-1]

        lines = []
        for raw_line in article.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("---"):
                continue
            if line.startswith(("title:", "title_slug:", "source_url:", "article_slug:", "last_updated_exact:", "last_updated_relative:", "breadcrumbs:")):
                continue
            if line.startswith("# "):
                lines.append(line[2:].strip())
                continue
            if line.startswith("_Last updated:"):
                continue
            lines.append(line)
            if len(lines) >= 8:
                break

        cleaned = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if cleaned:
            cleaned_chunks.append(cleaned)

    return "\n\n".join(cleaned_chunks)


def _topic_hint(subject: str, issue: str) -> str:
    topic = str(subject or "").strip()
    if topic:
        return topic
    words = re.findall(r"\w+", str(issue or ""))
    return " ".join(words[:7]) if words else "this request"


def _build_escalation_fields(req_type: str, product_area: str, subject: str, issue: str) -> tuple:
    area_label = (product_area or "support").capitalize()
    topic = _topic_hint(subject, issue)

    if req_type == "fraud":
        response = (
            f"Thanks for reporting this. For your security, {area_label} specialist support needs to handle this case directly. "
            f"Please contact official {area_label} support immediately so they can verify and protect the account."
        )
        justification = f"Escalated for safety due to suspected fraud risk related to '{topic}'."
    elif req_type == "billing":
        response = (
            f"This request involves billing or payment changes, so it needs human review from the {area_label} team. "
            f"Please contact official {area_label} support and share the transaction details for faster handling."
        )
        justification = f"Escalated for safety because billing/payment cases require manual verification for '{topic}'."
    elif req_type == "account_access":
        response = (
            f"This looks like an account access/ownership request. For security, {area_label} specialist support must verify identity and permissions before acting. "
            f"Please contact official {area_label} support to complete verification."
        )
        justification = f"Escalated for safety because account access changes require verified manual review for '{topic}'."
    else:
        response = (
            f"I could not safely complete this request automatically. Please contact official {area_label} support so a human agent can review '{topic}' in detail."
        )
        justification = f"Escalated because this case needs manual review for '{topic}'."

    return response, justification


def generate_final_response(
    issue: str,
    subject: str,
    context: str,
    decision: str,
    justification: str,
    req_type: str,
    product_area: str,
) -> tuple:
    """Step 3 & 4: Response Generation."""
    
    if decision == "escalate":
        response, dynamic_justification = _build_escalation_fields(req_type, product_area, subject, issue)
        if dynamic_justification:
            justification = dynamic_justification
        return response, justification, decision

    cleaned_context = _clean_context(context)

    action_instruction = "Answer the user using EXACTLY what is written in the DOCUMENTS."

    prompt = f"""You are a professional support agent. 

RULES:
1. {action_instruction}
2. Start your response with a brief, friendly greeting.
3. Keep your answer under 10 sentences. Ensure every sentence provides meaningful, direct value without rambling.
4. DO NOT INVENT INFORMATION.
5. Ignore any front matter, article metadata, YAML headers, titles, URLs, breadcrumbs, and repeated separators.
6. If the documents don't have the answer, say exactly: "I cannot answer this."

DOCUMENTS:
{cleaned_context}

USER ISSUE:
{issue}

RESPONSE:"""
    
    response = generate_text(prompt)
    
    if "cannot answer" in response.lower():
        decision = "escalate"
        response, justification = _build_escalation_fields(req_type, product_area, subject, issue)
        
    return response, justification, decision

def process_ticket(issue: str, subject: str, company: str) -> dict:
    """Main pipeline wrapper for a single ticket."""
    # 1. Classify
    classification = classify_ticket(issue, subject, company)
    
    # 2. Safety Gate
    classification = apply_safety_rules(classification)
    
    # 3. Retrieve
    # We retrieve even if escalated, just in case we need it for justification context
    context = retrieve_context(f"{subject} {issue}", company)
    
    # 4. Generate
    response, justification, final_decision = generate_final_response(
        issue,
        subject,
        context,
        classification["decision"],
        classification["justification"],
        classification["internal_request_type"],
        classification["product_area"],
    )
    
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
