"""
agent.py

Core logic for the AI support triage agent. 
Implements document retrieval (TF-IDF + embeddings) and LLM-based response generation.
Contains the primary `process_ticket` entry point required by HackerRank.
"""

import glob
import math
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np  # pyright: ignore[reportMissingImports]
from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
from llm import generate_json

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

VALID_STATUS = {"replied", "escalated"}
VALID_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
VALID_DOMAINS = {"hackerrank", "claude", "visa"}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "in", "on", "to", "for", "from",
    "with", "without", "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those",
    "my", "our", "your", "you", "we", "they", "he", "she", "them", "us", "me", "i", "as", "at", "by", "into",
    "about", "can", "could", "would", "should", "please", "help", "need", "want", "get", "got", "have", "has",
    "had", "do", "does", "did", "not", "no", "yes", "hi", "hello", "thanks", "thank", "there", "here",
}



@dataclass
class SupportDoc:
    domain: str
    area: str
    title: str
    source_url: str
    path: str
    content: str
    tokens: Counter


_DOCS: List[SupportDoc] = []
_IDF: Dict[str, float] = {}
_EMBED_MODEL = None
_DOC_EMB = None


def _snake_case(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    return cleaned or "general_help"


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[^\W_]+", (text or "").lower()) if len(t) > 1 and t not in STOPWORDS]


def _parse_frontmatter(raw: str) -> Tuple[Dict[str, object], str]:
    if not raw.startswith("---\n"):
        return {}, raw

    lines = raw.splitlines()
    meta: Dict[str, object] = {}
    idx = 1
    current_list_key = None

    while idx < len(lines):
        line = lines[idx]
        if line.strip() == "---":
            idx += 1
            break

        key_match = re.match(r"^([A-Za-z0-9_]+):\s*(.*)$", line)
        if key_match:
            key = key_match.group(1).strip()
            value = key_match.group(2).strip()
            if value:
                meta[key] = value.strip('"')
                current_list_key = None
            else:
                meta[key] = []
                current_list_key = key
            idx += 1
            continue

        if current_list_key and line.strip().startswith("-"):
            item = line.strip()[1:].strip().strip('"')
            if isinstance(meta.get(current_list_key), list):
                meta[current_list_key].append(item)
            idx += 1
            continue

        idx += 1

    body = "\n".join(lines[idx:])
    return meta, body


def _extract_heading(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def _clean_body(body: str, max_lines: int = 120, limit_chars: int = 1800) -> str:
    lines: List[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("!"):
            continue
        if line.startswith("[") and "](" in line:
            continue
        if line.startswith("_Last updated"):
            continue
        if line.startswith("---"):
            continue
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", line)
        line = re.sub(r"<[^>]+>", " ", line)
        line = line.replace("#", "").strip()
        if not line:
            continue
        lines.append(_normalize_space(line))
        if len(lines) >= max_lines:
            break
    cleaned = "\n".join(lines)
    return cleaned[:limit_chars]


def _infer_area(meta: Dict[str, object], domain: str, rel_parts: List[str]) -> str:
    if len(rel_parts) >= 2:
        if domain == "visa" and rel_parts[1].lower() == "support" and len(rel_parts) >= 3:
            return _snake_case(rel_parts[2])
        return _snake_case(rel_parts[1])

    breadcrumbs = meta.get("breadcrumbs")
    if isinstance(breadcrumbs, list) and breadcrumbs:
        if len(breadcrumbs) >= 2:
            return _snake_case(str(breadcrumbs[1]))
        return _snake_case(str(breadcrumbs[0]))
    return "general_help"


def _build_corpus() -> None:
    global _DOCS, _IDF, _EMBED_MODEL, _DOC_EMB
    md_files = glob.glob(os.path.join(DATA_DIR, "**", "*.md"), recursive=True)
    docs: List[SupportDoc] = []
    doc_freq: Counter = Counter()

    for path in md_files:
        rel = os.path.relpath(path, DATA_DIR)
        rel_parts = rel.split(os.sep)
        if not rel_parts:
            continue

        domain = rel_parts[0].lower()
        if domain not in VALID_DOMAINS:
            continue

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        meta, body = _parse_frontmatter(raw)
        title = str(meta.get("title") or _extract_heading(body) or os.path.basename(path)).strip()
        source_url = str(meta.get("source_url") or meta.get("final_url") or "").strip()
        area = _infer_area(meta, domain, rel_parts)

        content = _clean_body(body)
        token_counter = Counter(_tokenize(f"{title} {content}"))
        if not content or not token_counter:
            continue

        docs.append(
            SupportDoc(
                domain=domain,
                area=area,
                title=title,
                source_url=source_url,
                path=path,
                content=content,
                tokens=token_counter,
            )
        )

        for tok in set(token_counter.keys()):
            doc_freq[tok] += 1

    n_docs = max(len(docs), 1)
    idf = {tok: math.log((n_docs + 1.0) / (df + 1.0)) + 1.0 for tok, df in doc_freq.items()}

    _DOCS = docs
    _IDF = idf
    try:
        _EMBED_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        emb_texts = [f"{d.title}. {d.content[:700]}" for d in _DOCS]
        _DOC_EMB = _EMBED_MODEL.encode(emb_texts, normalize_embeddings=True).astype("float32")
    except Exception:
        _EMBED_MODEL = None
        _DOC_EMB = None


def _load_corpus() -> None:
    """
    Loads the pre-computed document corpus from `corpus.pkl` if available.
    Falls back to building the corpus dynamically in memory if the file is missing.
    """
    global _DOCS, _IDF, _EMBED_MODEL, _DOC_EMB
    if _DOCS:
        return

    corpus_path = os.path.join(os.path.dirname(__file__), "corpus.pkl")
    if os.path.exists(corpus_path):
        with open(corpus_path, "rb") as f:
            data = pickle.load(f)
        _DOCS = data["docs"]
        _IDF = data["idf"]
        _DOC_EMB = data["doc_emb"]
        try:
            _EMBED_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception:
            _EMBED_MODEL = None
        return

    _build_corpus()

def _infer_domain(issue: str, subject: str, company: str) -> str:
    company_norm = (company or "").strip().lower()
    if company_norm in VALID_DOMAINS:
        return company_norm

    text = f"{subject} {issue}".lower()
    if any(k in text for k in ["visa", "card", "merchant", "chargeback", "traveller", "travel cheque"]):
        return "visa"
    if any(k in text for k in ["claude", "anthropic", "bedrock", "workspace", "lti", "cowork"]):
        return "claude"
    return "hackerrank"


def _score_doc(doc: SupportDoc, query_tokens: List[str], bigrams: List[str]) -> float:
    score = 0.0
    title_tokens = set(_tokenize(doc.title))
    content_lower = doc.content.lower()

    for tok in query_tokens:
        tf = doc.tokens.get(tok, 0)
        if tf:
            score += (1.0 + math.log(tf)) * _IDF.get(tok, 0.0)
        if tok in title_tokens:
            score += 1.4

    for bg in bigrams:
        if bg and bg in content_lower:
            score += 1.1

    return score


def retrieve_docs(issue: str, subject: str, company: str, top_k: int = 4) -> List[SupportDoc]:
    _load_corpus()
    query = _normalize_space(f"{subject} {issue}")
    query_tokens = _tokenize(query)
    if not query_tokens:
        query_tokens = _tokenize(issue or subject)

    bigrams = [" ".join(query_tokens[i : i + 2]) for i in range(max(0, len(query_tokens) - 1))]
    target_domain = _infer_domain(issue, subject, company)

    scored: List[Tuple[float, SupportDoc]] = []
    query_sem = None
    if _EMBED_MODEL is not None and _DOC_EMB is not None:
        try:
            query_sem = _EMBED_MODEL.encode([query], normalize_embeddings=True).astype("float32")[0]
        except Exception:
            query_sem = None

    domain_docs = [(idx, d) for idx, d in enumerate(_DOCS) if d.domain == target_domain]
    for idx, doc in domain_docs:
        score = _score_doc(doc, query_tokens, bigrams)
        if query_sem is not None and _DOC_EMB is not None:
            score += 4.0 * float(np.dot(query_sem, _DOC_EMB[idx]))
        if score > 0:
            scored.append((score + 2.0, doc))

    if not scored:
        for idx, doc in enumerate(_DOCS):
            score = _score_doc(doc, query_tokens, bigrams)
            if query_sem is not None and _DOC_EMB is not None:
                score += 4.0 * float(np.dot(query_sem, _DOC_EMB[idx]))
            if score > 0:
                scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    picked: List[SupportDoc] = []
    seen = set()
    for _, doc in scored:
        key = (doc.title, doc.area)
        if key in seen:
            continue
        seen.add(key)
        picked.append(doc)
        if len(picked) >= top_k:
            break

    if not picked:
        picked = [d for _, d in domain_docs[:top_k]] if domain_docs else _DOCS[:top_k]

    return picked


def _build_doc_context(docs: List[SupportDoc], max_chars: int = 7000) -> str:
    blocks: List[str] = []
    total = 0

    for i, doc in enumerate(docs, start=1):
        block = (
            f"[DOC {i}]\n"
            f"domain: {doc.domain}\n"
            f"product_area: {doc.area}\n"
            f"title: {doc.title}\n"
            f"source_url: {doc.source_url or 'N/A'}\n"
            f"content: {doc.content}\n"
        )
        total += len(block)
        if total > max_chars:
            break
        blocks.append(block)

    return "\n".join(blocks)


def _safe_default_response(status: str, area: str, docs: List[SupportDoc], issue: str) -> str:
    if status == "escalated":
        return "This case needs a human support specialist to review safely. Please escalate this to the human support team, as I am an AI agent without system access."

    if docs:
        top = docs[0]
        snippet = _normalize_space(top.content[:320])
        if snippet:
            return f"Based on '{top.title}': {snippet}"

    return f"I could not confidently answer from the provided support corpus for '{issue}'."


def _build_default_justification(status: str, area: str, request_type: str, docs: List[SupportDoc]) -> str:
    titles = "; ".join([d.title for d in docs[:2]]) if docs else "no strong document match"
    if status == "escalated":
        return f"Escalated due to risk/insufficient certainty. Top references: {titles}."
    return f"Answered using retrieved support documents in {area}: {titles}."


def _extractive_reply(issue: str, subject: str, docs: List[SupportDoc], max_sentences: int = 5) -> str:
    if not docs:
        return "I could not find enough support-corpus evidence to answer this directly."

    query_tokens = set(_tokenize(f"{subject} {issue}"))
    if not query_tokens:
        query_tokens = set(_tokenize(issue or subject))

    candidates: List[Tuple[float, str, str]] = []
    for doc in docs[:3]:
        lines = [x.strip() for x in doc.content.splitlines() if x.strip()]
        for line in lines:
            clean = _normalize_space(line)
            if len(clean) < 24 or len(clean) > 220:
                continue
            sent_tokens = set(_tokenize(clean))
            if not sent_tokens:
                continue
            overlap = len(query_tokens.intersection(sent_tokens))
            score = float(overlap)
            if score <= 0:
                continue
            if any(tok in _tokenize(doc.title) for tok in query_tokens):
                score += 0.6
            candidates.append((score, clean, doc.title))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected: List[str] = []
    seen = set()
    for _, sentence, title in candidates:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) >= max_sentences:
            break

    if not selected:
        top = docs[0]
        fallback = _normalize_space(top.content[:380])
        return f"Based on '{top.title}': {fallback}"

    reply = " ".join(selected)
    reply = re.sub(r"\s{2,}", " ", reply).strip()
    return reply

def _llm_generate_grounded_fields(
    issue: str,
    subject: str,
    company: str,
    docs: List[SupportDoc],
    fallback_area: str,
) -> Dict[str, str]:
    if not docs:
        return {}

    area_candidates = sorted({_snake_case(d.area) for d in docs[:4]})
    if not area_candidates:
        area_candidates = [_snake_case(fallback_area)]

    prompt = f"""You are generating support output from retrieved documentation only.
Return ONLY valid JSON with keys: status, request_type, product_area, response, justification.

Rules:
1) Use only the DOCUMENTS below. No external knowledge.
2) status must be one of: replied, escalated. (Escalate for security, outages, or severe account issues).
3) request_type must be one of: product_issue, feature_request, bug, invalid. (Use bug for platform errors, feature_request for new capabilities, invalid for unrelated/spam, and product_issue for all general support/how-to questions).
4) product_area must be one of: {", ".join(area_candidates)}.
5) response should be concise (2-6 sentences), practical, and user-facing. Always start with a polite greeting.
   - If feature_request, acknowledge it and say you've logged it for the product team.
   - If invalid, politely state the request is out of scope for this support assistant.
   - If escalated, do not say you will escalate it. Instead, tell the user to escalate to the human support team, as you are an AI agent without system access.
6) justification must reference why the chosen docs support the response. Use the actual document titles instead of referring to them as "DOC 1" or "DOC 2" (keep it brief, 1-2 sentences).
7) Respond in the same language as the user's ISSUE.

SUBJECT: {subject}
ISSUE: {issue}

{_build_doc_context(docs, max_chars=3200)}
"""

    result = generate_json(prompt, max_tokens=1024)
    if not isinstance(result, dict):
        return {}
    return {
        "status": str(result.get("status", "")).strip().lower(),
        "request_type": str(result.get("request_type", "")).strip().lower(),
        "product_area": str(result.get("product_area", "")).strip(),
        "response": str(result.get("response", "")).strip(),
        "justification": str(result.get("justification", "")).strip(),
    }


def _local_triage(issue: str, subject: str, company: str, docs: List[SupportDoc], fallback_area: str) -> Dict[str, str]:
    product_area = _snake_case(fallback_area)

    llm_fields = _llm_generate_grounded_fields(issue, subject, company, docs, product_area)

    status = llm_fields.get("status", "") if llm_fields else ""
    request_type = llm_fields.get("request_type", "") if llm_fields else ""
    response = llm_fields.get("response", "") if llm_fields else ""
    justification = llm_fields.get("justification", "") if llm_fields else ""
    llm_area = _snake_case(llm_fields.get("product_area", "")) if llm_fields else ""

    if status not in VALID_STATUS:
        status = "replied"
    if request_type not in VALID_REQUEST_TYPES:
        request_type = "product_issue"

    if llm_area:
        product_area = llm_area

    if not response:
        response = _extractive_reply(issue, subject, docs)

    if not justification:
        justification = _build_default_justification(status, product_area, request_type, docs)

    if status == "escalated" and "human" not in response.lower() and "support" not in response.lower():
        response = f"{response.strip()} Please escalate this to the human support team, as I am an AI agent without system access."

    return {
        "status": status,
        "product_area": product_area,
        "response": response,
        "justification": justification,
        "request_type": request_type,
    }


def process_ticket(issue: str, subject: str, company: str) -> dict:
    """
    Main entry point for evaluating a support ticket.
    Retrieves relevant documents and uses an LLM to generate a grounded response
    along with status, request_type, product_area, and justification.
    """
    issue = str(issue or "").strip()
    subject = str(subject or "").strip()
    company = str(company or "").strip()

    docs = retrieve_docs(issue, subject, company, top_k=4)
    fallback_area = docs[0].area if docs else "general_help"

    return _local_triage(issue, subject, company, docs, fallback_area)
