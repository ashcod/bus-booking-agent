# 🚌 BusBot — Multi-Agent AI Bus Booking System

A production-grade multi-agent AI system that adds a conversational booking interface on top of India's bus travel ecosystem. Search routes, select seats, and confirm bookings in natural language — no forms, no 6-screen flows.

**🚀 Live Demo:** https://busbot-main.happyforest-6f22e5e0.southindia.azurecontainerapps.io

> Deployed on Azure Container Apps · South India region

---

## Screenshots

### Home screen
![Home Screen](screenshots/UI_Screenshots_Home.png)

### Bus search results
![Bus Search Results](screenshots/UI_Screenshots_1.png)

### Seat map selection
![Seat Map](screenshots/UI_Screenshots_2.png)

### Booking confirmation
![Booking Confirmation](screenshots/UI_Screenshots_3.png)

---

## What it does

Type naturally — BusBot handles the rest:

```
User: Show me evening AC buses from Hyderabad to Bangalore

Bot:  Found 2 buses:
      1. MSRTC AC Sleeper  18:30 -> 04:00  Rs 1,140  21 seats
      2. MSRTC Seater      21:30 -> 07:00  Rs 456    15 seats

User: [clicks Select Seat on option 1]
      [picks window seat from interactive map]
      [clicks Confirm and Book]

Bot:  Booking confirmed! BK4F2A1C3D
      Download your PDF ticket below.
```

---

## Evaluation Results

| Metric | Score | What it means |
|--------|-------|---------------|
| Faithfulness | **1.000** | Zero hallucination — LLM only uses retrieved bus data |
| Context Recall | **1.000** | Retriever finds right documents every time |
| Answer Relevancy | **0.755** | Factually correct, improvement in progress |
| **Overall** | **0.918** | Custom RAGAS-equivalent harness |

---

## Architecture

```
User (Natural Language)
        |
FastAPI + SSE Streaming (port 8000)
        |
Orchestrator Agent
  ├── Semantic Guardrails (embedding-based, not keyword lists)
  ├── Intent Classification
  └── Param Extraction
        |
   ┌────┴─────────────────────┐
   |                          |
Clarification Agent     Search Agent
(asks one question       (Hybrid RAG:
 when info missing)       BM25 + Dense
                          + RRF fusion)
                               |
                         Reflection Agent
                         (self-critique,
                          retries if < 7/10)
   |                          |
Booking Agent           Support Agent
(MCP tool calls)        (cancel + refund
                         calculation)
        |
MCP Tool Server (port 8001)
  ├── search_buses
  ├── book_ticket
  ├── get_seat_map
  └── cancel_booking
        |
   ┌────┴──────────┐
Qdrant          SQLite
(vector DB)     (bookings)
```

**Key concepts demonstrated:**
- Multi-agent orchestration with LangGraph StateGraph
- Advanced RAG: hybrid BM25 + dense retrieval + Reciprocal Rank Fusion
- MCP (Model Context Protocol) tool server
- Short-term memory (LangGraph MemorySaver) + long-term memory (user profiles)
- Embedding-based semantic guardrails (no keyword lists)
- Self-critique reflection loop
- LangSmith observability
- Custom RAGAS-equivalent evaluation harness

---

## Tech Stack

| Layer | Technology | Why chosen |
|-------|-----------|------------|
| Agent framework | LangGraph + LangChain | Explicit graph flow, built-in state management |
| LLM | llama-3.3-70b via Groq | Free tier, fast inference, strong instruction following |
| Embeddings (local) | nomic-embed-text via Ollama | Free, offline, 768-dim |
| Embeddings (cloud) | sentence-transformers all-MiniLM-L6-v2 | CPU-only, no API key, Azure-ready |
| Vector DB | Qdrant | Fast HNSW, metadata pre-filtering, free |
| Sparse retrieval | BM25 (rank-bm25) | In-memory, sub-millisecond, complements dense |
| API | FastAPI + SSE | Async streaming, automatic validation |
| Database | SQLite | Zero config, same SQL as PostgreSQL |
| Tools | MCP protocol | Loose coupling, independently deployable |
| Tracing | LangSmith | Every agent hop and tool call traced |
| PDF | ReportLab | Pure Python ticket generation |
| Deployment | Azure Container Apps | Auto-scaling, managed HTTPS |

---

## Project Structure

```
bus-booking-agent/
|
├── app/                        # Application source code
|   ├── agents/                 # All AI agents
|   |   ├── state.py            # BookingState — shared memory schema
|   |   ├── orchestrator.py     # Entry point: guardrails + intent + routing
|   |   ├── clarification_agent.py  # Progressive disclosure of missing params
|   |   ├── search_agent.py     # Hybrid RAG retrieval + result formatting
|   |   ├── booking_agent.py    # Seat reservation via MCP tools
|   |   ├── support_agent.py    # Cancellation + exact refund calculation
|   |   ├── reflection_agent.py # Self-critique: scores and retries responses
|   |   └── graph.py            # LangGraph StateGraph wiring all agents
|   |
|   ├── api/                    # FastAPI interface layer
|   |   ├── main.py             # /chat/stream, /health, session management
|   |   ├── pdf_generator.py    # ReportLab PDF ticket generation
|   |   └── static/
|   |       └── index.html      # Full chat UI: bus cards, seat map, booking
|   |
|   ├── core/                   # Shared configuration
|   |   └── config.py           # Single source of truth: models, URLs, paths
|   |
|   ├── memory/                 # User preference store
|   |   └── user_memory.py      # Long-term: infers preferred city from history
|   |
|   ├── rag/                    # Retrieval pipeline
|   |   ├── embedder.py         # Adapter: Ollama (local) / sentence-transformers (cloud)
|   |   ├── indexer.py          # One-time: embed documents into Qdrant
|   |   └── retriever.py        # Hybrid BM25 + dense + RRF + time filtering
|   |
|   └── tools/                  # MCP tool server
|       ├── tool_server.py      # FastAPI port 8001: search, book, seat map, cancel
|       └── mcp_client.py       # HTTP client agents use to call tools
|
├── data/                       # Data layer
|   ├── db/
|   |   ├── bus_booking.db      # SQLite: routes, schedules, bookings, seats
|   |   └── user_memory.json    # Long-term user profiles
|   └── raw/
|       └── documents.json      # Natural language documents for embedding
|
├── docs/                       # Documentation
|   └── SETUP.md                # Step-by-step setup guide
|
├── evals/                      # Quality assurance
|   ├── rag_eval.py             # Custom RAGAS-equivalent evaluation harness
|   ├── test_cases.md           # 30 documented test cases
|   └── results.json            # Timestamped eval scores (regression history)
|
├── scripts/                    # One-time data scripts
|   ├── generate_data.py        # Creates 105 routes, 309 schedules, 12K seats
|   └── build_documents.py      # Converts SQLite rows to natural language docs
|
├── screenshots/                # UI screenshots for README
|
├── health_check.py             # Verifies all services are running
├── docker-compose.yml          # One command: start Qdrant + Redis locally
├── Dockerfile                  # Main app container
├── Dockerfile.tools            # Tool server container
├── .env.example                # Environment variable template (no real keys)
└── pyproject.toml              # Python dependencies
```

---

## Data

- **105 routes** across 15 Indian cities
- **309 bus schedules** with realistic pricing and operating days
- **~12,000 seat records** with gender, deck (upper/lower), and window status
- **Cities:** Hyderabad, Bangalore, Chennai, Mumbai, Pune, Delhi, Kolkata, Ahmedabad, Jaipur, Lucknow, Nagpur, Visakhapatnam, Kochi, Coimbatore, Madurai

---

## Local Setup (Step by Step)

### Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| Python 3.10+ | Runtime | python.org |
| uv | Package manager | `pip install uv` |
| Docker Desktop | Runs Qdrant locally | docker.com |
| Ollama | Local embedding model | ollama.com |
| Groq API key | Free LLM | console.groq.com |
| LangSmith API key | Free tracing | smith.langchain.com |

### Step 1 — Clone and install

```bash
git clone https://github.com/ashcod/bus-booking-agent.git
cd bus-booking-agent
uv sync
```

### Step 2 — Set environment variables

```bash
cp .env.example .env
```

Edit `.env` with your actual API keys:

```
GROQ_API_KEY=your_groq_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=bus-booking-agent
EMBEDDING_PROVIDER=ollama
QDRANT_URL=http://localhost:6333
TOOL_SERVER_URL=http://localhost:8001
```

### Step 3 — Start infrastructure

```bash
# Start Qdrant and Redis
docker compose up -d

# Pull embedding model
ollama pull nomic-embed-text
```

### Step 4 — Generate data and index

```bash
# Generate synthetic bus data (105 routes, 309 schedules, ~12K seats)
uv run python scripts/generate_data.py

# Convert to natural language documents for embedding
uv run python scripts/build_documents.py

# Embed documents and store in Qdrant
uv run python -m app.rag.indexer
```

### Step 5 — Run the system

Open two terminals:

**Terminal 1 — MCP tool server:**
```bash
uv run python -m app.tools.tool_server
```

**Terminal 2 — Main app:**
```bash
uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 6 — Open browser

```
http://localhost:8000
```

### Step 7 — Verify everything is working

```bash
uv run python health_check.py
```

Expected output:
```
BUSBOT HEALTH CHECK
Main app (8000):    OK - {'status': 'ok', 'service': 'bus-booking-agent'}
Tool server (8001): OK - 4 tools
Chat endpoint:      OK - agent=clarification
```

---

## Run Evaluation

```bash
uv run python evals/rag_eval.py
```

Scores are saved to `evals/results.json` with a timestamp after every run. Use this as a regression test after code changes.

---

## Azure Deployment (Step by Step)

This section documents exactly how the live deployment was created. Follow these steps to deploy your own instance.

### Prerequisites

- Azure account (free tier works)
- Azure CLI installed: https://aka.ms/installazurecliwindows
- Docker Desktop running
- Images already built locally

### Step 1 — Login to Azure

```powershell
az login
az account show
```

### Step 2 — Set variables

```powershell
$RESOURCE_GROUP = "busbot-rg"
$LOCATION = "centralindia"
$ACR_NAME = "busbotregistry"
$STORAGE_ACCOUNT = "busbotstorage"
$SHARE_NAME = "busbotdata"
```

### Step 3 — Create resource group and registry

```powershell
az group create --name $RESOURCE_GROUP --location $LOCATION

az acr create `
  --resource-group $RESOURCE_GROUP `
  --name $ACR_NAME `
  --sku Basic `
  --admin-enabled true
```

### Step 4 — Create storage for SQLite persistence

```powershell
az storage account create `
  --name $STORAGE_ACCOUNT `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION `
  --sku Standard_LRS

az storage share create `
  --name $SHARE_NAME `
  --account-name $STORAGE_ACCOUNT

$STORAGE_KEY = az storage account keys list `
  --resource-group $RESOURCE_GROUP `
  --account-name $STORAGE_ACCOUNT `
  --query "[0].value" -o tsv
```

### Step 5 — Register required Azure providers

```powershell
az provider register --namespace Microsoft.ContainerInstance
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

Wait until all show Registered:

```powershell
az provider show --namespace Microsoft.ContainerInstance --query "registrationState" -o tsv
az provider show --namespace Microsoft.App --query "registrationState" -o tsv
az provider show --namespace Microsoft.OperationalInsights --query "registrationState" -o tsv
```

### Step 6 — Deploy Qdrant to Azure Container Instance

```powershell
az container create `
  --resource-group $RESOURCE_GROUP `
  --name busbot-qdrant `
  --image qdrant/qdrant:v1.11.3 `
  --os-type Linux `
  --ports 6333 `
  --protocol TCP `
  --cpu 1 `
  --memory 2 `
  --location $LOCATION `
  --azure-file-volume-account-name $STORAGE_ACCOUNT `
  --azure-file-volume-account-key $STORAGE_KEY `
  --azure-file-volume-share-name $SHARE_NAME `
  --azure-file-volume-mount-path /qdrant/storage `
  --ip-address Public

$QDRANT_IP = az container show `
  --resource-group $RESOURCE_GROUP `
  --name busbot-qdrant `
  --query "ipAddress.ip" -o tsv

echo "Qdrant IP: $QDRANT_IP"
```

### Step 7 — Build and push Docker images

```powershell
az acr login --name $ACR_NAME

docker build -t "$ACR_NAME.azurecr.io/busbot-main:latest" -f Dockerfile .
docker push "$ACR_NAME.azurecr.io/busbot-main:latest"

docker build -t "$ACR_NAME.azurecr.io/busbot-tools:latest" -f Dockerfile.tools .
docker push "$ACR_NAME.azurecr.io/busbot-tools:latest"
```

### Step 8 — Get registry credentials

```powershell
$ACR_USERNAME = az acr credential show --name $ACR_NAME --query "username" -o tsv
$ACR_PASSWORD = az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv
```

### Step 9 — Deploy tool server (internal)

```powershell
az containerapp create `
  --name busbot-tools `
  --resource-group $RESOURCE_GROUP `
  --environment "YOUR_CONTAINER_APP_ENVIRONMENT_ID" `
  --image $ACR_NAME.azurecr.io/busbot-tools:latest `
  --registry-server $ACR_NAME.azurecr.io `
  --registry-username $ACR_USERNAME `
  --registry-password $ACR_PASSWORD `
  --target-port 8001 `
  --ingress internal `
  --cpu 0.5 `
  --memory 1.0Gi `
  --min-replicas 1 `
  --max-replicas 3 `
  --env-vars DEPLOYMENT_MODE=cloud QDRANT_URL=http://$QDRANT_IP:6333

$TOOL_SERVER_URL = az containerapp show `
  --name busbot-tools `
  --resource-group $RESOURCE_GROUP `
  --query "properties.configuration.ingress.fqdn" -o tsv
```

> **Note:** Replace `YOUR_CONTAINER_APP_ENVIRONMENT_ID` with your environment ID.
> Free Azure subscriptions allow only 1 Container Apps environment.
> If you already have one, use it: `az containerapp env list -o table`

### Step 10 — Deploy main app (public)

```powershell
az containerapp create `
  --name busbot-main `
  --resource-group $RESOURCE_GROUP `
  --environment "YOUR_CONTAINER_APP_ENVIRONMENT_ID" `
  --image $ACR_NAME.azurecr.io/busbot-main:latest `
  --registry-server $ACR_NAME.azurecr.io `
  --registry-username $ACR_USERNAME `
  --registry-password $ACR_PASSWORD `
  --target-port 8000 `
  --ingress external `
  --cpu 0.5 `
  --memory 1.0Gi `
  --min-replicas 1 `
  --max-replicas 5 `
  --env-vars DEPLOYMENT_MODE=cloud EMBEDDING_PROVIDER=sentence-transformers VECTOR_SIZE=384 QDRANT_URL=http://$QDRANT_IP:6333 TOOL_SERVER_URL=https://$TOOL_SERVER_URL GROQ_API_KEY=YOUR_KEY LANGCHAIN_TRACING_V2=true LANGCHAIN_API_KEY=YOUR_KEY LANGCHAIN_PROJECT=bus-booking-agent-prod
```

### Step 11 — Get your live URL

```powershell
az containerapp show `
  --name busbot-main `
  --resource-group $RESOURCE_GROUP `
  --query "properties.configuration.ingress.fqdn" -o tsv
```

Open the URL in your browser — BusBot is live.

### Step 12 — Upload database to Azure File Share

```powershell
az storage file upload `
  --account-name $STORAGE_ACCOUNT `
  --account-key $STORAGE_KEY `
  --share-name $SHARE_NAME `
  --source data/db/bus_booking.db `
  --path bus_booking.db
```

---

## Azure Architecture

```
Internet
    |
Azure Container Apps (external ingress, HTTPS auto)
    |
busbot-main (0.5 vCPU, 1GB, auto-scale 1-5)
    |
    ├── busbot-tools (internal ingress, port 8001, 1-3 replicas)
    |       |
    |       └── SQLite on Azure File Share (persistent)
    |
    └── busbot-qdrant (Azure Container Instance, public IP)
            |
            └── Azure File Share (vector storage, persistent)

Azure Container Registry
    └── busbotregistry.azurecr.io
        ├── busbot-main:latest
        └── busbot-tools:latest
```

---

## Environment Variables Reference

| Variable | Local value | Cloud value | Purpose |
|----------|-------------|-------------|---------|
| `GROQ_API_KEY` | From console.groq.com | Same | LLM access |
| `LANGCHAIN_API_KEY` | From smith.langchain.com | Same | Tracing |
| `LANGCHAIN_TRACING_V2` | true | true | Enable tracing |
| `EMBEDDING_PROVIDER` | ollama | sentence-transformers | Swap embedder |
| `VECTOR_SIZE` | 768 | 384 | Match embedding dims |
| `QDRANT_URL` | http://localhost:6333 | http://QDRANT_IP:6333 | Vector DB |
| `TOOL_SERVER_URL` | http://localhost:8001 | https://internal-url | MCP tools |
| `DEPLOYMENT_MODE` | local | cloud | Feature flags |

---

## Author

**Sai Ashish** — AI Engineer  
8+ years experience · 4 years at Amazon · Now building independently

[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/ashcod)

Open to senior AI Engineer / LLM Engineer roles in Hyderabad and remote.

---

## License

MIT
