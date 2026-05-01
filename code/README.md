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
By default, the script runs in headless batch mode for automated grading. It will read `support_tickets/support_tickets.csv` and show a beautifully styled progress bar.

```bash
python main.py
```

### Step 3: Interactive TUI (Bonus!)
To access the interactive Terminal User Interface (TUI), run the script with the `--ui` flag. 

```bash
python main.py --ui
```

This launches a HackerRank-themed menu where you can:
1. **Chatbot Mode**: Type issues interactively and see how the agent routes, escalates, and justifies its decisions in real-time.
2. **Batch Process**: Visually run the dataset evaluation.
