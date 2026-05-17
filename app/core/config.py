# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL    = "llama-3.3-70b-versatile"
LLM_PROVIDER = "groq"

# Embeddings
EMBEDDING_MODEL = "nomic-embed-text"

# Infrastructure
QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "bus_schedules"
DB_PATH         = "data/db/bus_booking.db"
TOP_K           = 5

# Tool server
TOOL_SERVER_URL = "http://localhost:8001"

# LangSmith — loaded from .env automatically by LangChain
# Just needs LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY set
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "bus-booking-agent")
