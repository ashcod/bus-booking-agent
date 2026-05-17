# Dockerfile — main app (FastAPI + agents)
FROM python:3.11-slim

# install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install uv
RUN pip install uv

WORKDIR /app

# copy dependency files first (Docker layer caching)
COPY pyproject.toml .
COPY uv.lock* .

# install dependencies
RUN uv sync --frozen --no-dev

# copy application code
COPY app/ ./app/
COPY data/ ./data/

# environment defaults (overridden by Azure secrets)
ENV EMBEDDING_PROVIDER=sentence-transformers
ENV DEPLOYMENT_MODE=cloud

# expose port
EXPOSE 8000

# start the main app
CMD ["uv", "run", "uvicorn", "app.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]