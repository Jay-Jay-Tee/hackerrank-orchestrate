# AI Triage Agent Codebase

This directory contains the Python implementation of the AI support triage agent built for the HackerRank Orchestrate hackathon.

## Architecture Overview

1. **`ingest.py`**: An offline pre-processing script. It reads all Markdown documentation from the `../data/` directory, extracts metadata, tokenizes text, generates TF-IDF mappings, and uses SentenceTransformer to compute dense vector embeddings. It saves everything to an offline cache (`corpus.pkl`).
2. **`agent.py`**: The core RAG engine. It matches incoming tickets against the cached corpus using hybrid search (semantic vector similarity + keyword matching), and passes the relevant context to the LLM to generate grounded support responses. Also contains the required `process_ticket` entry point.
3. **`llm.py`**: Manages all interactions with the LLM providers (e.g., OpenAI or local fallbacks). Configured strictly via environment variables.
4. **`main.py`**: A Terminal User Interface (TUI) orchestrator that provides a chatbot for interactive testing and a batch processor to run predictions across the HackerRank dataset.

## Getting Started

Before running the agent, make sure you have installed the required dependencies from `requirements.txt` and have your `.env` file configured.

### 0. Setup Environment
```bash
pip install -r requirements.txt
cp ../.env.example .env # Or create a .env file with OPENAI_API_KEY
```

### 1. Build the Document Corpus
You must generate the embeddings cache first:
```bash
python ingest.py
```

### 2. Run the App
To run the interactive UI mode:
```bash
python main.py --ui
```

To run the batch processor for evaluation (which processes `support_tickets.csv`):
```bash
python main.py
```