# Support Triage Agent

This is a terminal-based AI agent designed to triage support tickets for HackerRank, Claude, and Visa ecosystems. It operates **100% locally** using Ollama and FAISS.

## Architecture

1. **Local RAG**: Uses `sentence-transformers` (all-MiniLM-L6-v2) to chunk and embed the support markdown corpus into a `faiss-cpu` vector store.
2. **Deterministic Triage**: A two-step classification engine that extracts specific intents (e.g. fraud, billing) to trigger strict hard-coded safety escalations, before mapping them to the official Hackathon categories.
3. **Local LLM**: Uses `qwen2.5:0.5b` via `ollama` for fast, private, and rate-limit-free inference.

---

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.9+ installed. Run the following command from this directory:
```bash
pip install -r requirements.txt
```

### 2. Setup Ollama
You must have [Ollama](https://ollama.com/) installed and running locally.
Open a separate terminal and run:
```bash
ollama serve
```

Then, download the required lightweight model:
```bash
ollama pull qwen2.5:0.5b
```

*(Optional: If you want to use a different model or endpoint, you can create a `.env` file from `.env.example` and set `OLLAMA_MODEL` or `OLLAMA_URL`)*

---

## Running the Agent

### Step 1: Ingest Data
Before processing tickets, you must build the FAISS index from the provided `data/` corpus.
```bash
python ingest.py
```
This will create `faiss_index.bin` and `faiss_meta.json` in the `code/` directory.

### Step 2: Process Tickets
Run the main triage pipeline. By default, it looks for `support_tickets/support_tickets.csv`.

```bash
python main.py
```

To test against the sample dataset instead:
```bash
python main.py --input support_tickets/sample_support_tickets.csv --output support_tickets/sample_output.csv
```

The script will display a progress bar and output the final predictions to the designated output CSV.
