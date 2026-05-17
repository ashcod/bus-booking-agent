# Setup Guide

## Prerequisites
- Python 3.10+
- uv package manager: `pip install uv`
- Docker Desktop
- Ollama: https://ollama.com/download

## Quick Start

### 1. Clone and install
git clone https://github.com/ashcod/bus-booking-agent.git
cd bus-booking-agent
uv sync

### 2. Environment variables
cp .env.example .env
# Edit .env with your API keys

### 3. Pull AI models
ollama pull nomic-embed-text
ollama pull llama3.2

### 4. Start infrastructure
docker compose up -d

### 5. Generate data and index
uv run python scripts/generate_data.py
uv run python scripts/build_documents.py
uv run python -m app.rag.indexer

### 6. Run the system
# Terminal 1 - Tool server
uv run python -m app.tools.tool_server

# Terminal 2 - Main app
uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

### 7. Open browser
http://localhost:8000

## Run evaluation
uv run python evals/rag_eval.py