FROM python:3.11-slim

# Install git (needed by GitPython for cloning repos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface \
    ANONYMIZED_TELEMETRY=False \
    ALLOW_LOCAL_PATHS=false \
    LLM_PROVIDER=cloud

WORKDIR $HOME/app

# CPU-only PyTorch keeps the image small (the default wheel bundles ~2GB of CUDA libraries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements first and install dependencies for Docker layer caching
COPY --chown=user backend/requirements.txt $HOME/app/backend/requirements.txt
RUN pip install --no-cache-dir -r $HOME/app/backend/requirements.txt

# Download the embedding model at build time so the app starts fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application files
COPY --chown=user backend/ $HOME/app/backend/
COPY --chown=user frontend/ $HOME/app/frontend/

WORKDIR $HOME/app/backend

EXPOSE 7860

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
