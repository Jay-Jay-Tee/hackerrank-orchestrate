import os
import glob
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index.bin")
META_PATH = os.path.join(os.path.dirname(__file__), "faiss_meta.json")

# A small, fast embedding model suitable for local CPU usage
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
OVERLAP = 100

def chunk_text(text, max_length, overlap):
    """Splits text into chunks of max_length with given overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + max_length])
        chunks.append(chunk)
        i += max_length - overlap
    return chunks

def run_ingestion():
    print(f"Loading embedding model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Scanning data directory: {DATA_DIR}")
    md_files = glob.glob(os.path.join(DATA_DIR, "**", "*.md"), recursive=True)
    
    all_chunks = []
    metadata = []
    
    for filepath in md_files:
        # Determine product area from the directory structure
        # path is like data/claude/something.md
        parts = filepath.split(os.sep)
        product_area = "unknown"
        if "claude" in parts:
            product_area = "claude"
        elif "hackerrank" in parts:
            product_area = "hackerrank"
        elif "visa" in parts:
            product_area = "visa"
            
        filename = os.path.basename(filepath)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        chunks = chunk_text(content, CHUNK_SIZE, OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({
                "product_area": product_area,
                "filename": filename,
                "chunk_id": i,
                "text": chunk
            })
            
    if not all_chunks:
        print("No markdown files found!")
        return
        
    print(f"Total chunks created: {len(all_chunks)}. Computing embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    
    print("Building FAISS index...")
    # L2 distance index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"Saving FAISS index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)
    
    print(f"Saving metadata to {META_PATH}...")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print("Ingestion complete!")

if __name__ == "__main__":
    run_ingestion()
