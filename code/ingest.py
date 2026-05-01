"""
ingest.py

Offline document ingestion script for HackerRank Orchestrate.
Pre-computes TF-IDF counts and dense text embeddings using SentenceTransformer,
then saves them to a local `corpus.pkl` file. This prevents the agent from
having to process the entire Markdown corpus at runtime, significantly improving speed.
"""
import os
import pickle
from agent import _build_corpus
import agent

def run_ingestion():
    print("Starting document ingestion and embedding generation...")
    print("This may take a moment while SentenceTransformer encodes the documents...")
    
    # Force build the corpus, parsing all Markdown files
    _build_corpus()
    
    corpus_data = {
        "docs": agent._DOCS,
        "idf": agent._IDF,
        "doc_emb": agent._DOC_EMB
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "corpus.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(corpus_data, f)
        
    print(f"Success! Ingested {len(agent._DOCS)} documents and saved offline cache to {output_path}")

if __name__ == "__main__":
    run_ingestion()