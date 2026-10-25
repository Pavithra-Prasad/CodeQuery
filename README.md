---
title: CodeQuery AI
emoji: ⚡
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: RAG-powered Q&A over any GitHub codebase
---

# ⚡ CodeQuery AI — Codebase Q&A Assistant

A RAG-powered application that lets you **ask questions about any codebase in natural language** and get accurate, source-cited answers.

![Built with](https://img.shields.io/badge/Built%20with-RAG-7c5cfc)
![Python](https://img.shields.io/badge/Python-3.10+-blue)

## 🏗️ Architecture

```
User Question → Embed Query → Search Vector DB → Retrieve Code Chunks → LLM Generates Answer
```

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini API (free tier) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Vector DB | ChromaDB (local, persistent) |
| Backend | FastAPI (Python) |
| Frontend | Vanilla HTML/CSS/JS |

## 🚀 Quick Start

### 1. Get a Gemini API Key (Free)

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in → Click "Get API Key" → Create a key
3. Set it as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 2. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **Note:** First run will download the embedding model (~80MB). It's cached after that.

### 3. Run the Server

```bash
python main.py
```

### 4. Open the App

Open [http://localhost:8080](http://localhost:8080) in your browser.

## ☁️ Deploy to Hugging Face Spaces (free, no card)

1. Create a free account at [huggingface.co](https://huggingface.co/join)
2. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space): name it `codequery-ai`, pick **Docker** → **Blank**, hardware **CPU basic (free)**
3. In the Space → **Settings** → **Variables and secrets**, add `GROQ_API_KEY` and/or `GEMINI_API_KEY` as **secrets**
4. Create a **Write** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
5. Run the deploy script (use your HF username and the token as the password when git asks):

```bash
./deploy.sh YOUR_HF_USERNAME codequery-ai
```

The app will be live at `https://YOUR_HF_USERNAME-codequery-ai.hf.space` after the build finishes.

> **Note:** On the deployed app only GitHub URLs can be indexed (local paths are disabled). Free Spaces sleep after 48 hours without visitors and lose indexed repos on restart, so re-index after a wake-up.

## 📖 Usage

1. **Index a Repository** — Enter a local path (e.g., `/Users/you/project`) or GitHub URL in the sidebar
2. **Select the Repo** — Click on it in the indexed repositories list
3. **Ask Questions** — Type natural language questions about the code

### Example Questions

- "How does authentication work?"
- "Explain the database models and their relationships"
- "What API endpoints are available?"
- "How is error handling implemented?"
- "What design patterns are used in this codebase?"

## 🔧 How It Works (RAG Pipeline)

1. **Indexing**: Code files are split into logical chunks (respecting function/class boundaries), converted to vector embeddings, and stored in ChromaDB
2. **Retrieval**: When you ask a question, it's embedded and the most similar code chunks are retrieved via semantic search
3. **Generation**: The retrieved chunks + your question are sent to Gemini, which generates a grounded answer with source citations

## 📁 Project Structure

```
codebase-qa/
├── backend/
│   ├── main.py              # FastAPI server + API endpoints
│   ├── chunker.py           # Code-aware text splitting
│   ├── embeddings.py        # Local sentence-transformer embeddings
│   ├── vector_store.py      # ChromaDB wrapper
│   ├── rag_engine.py        # RAG pipeline (retrieve + generate)
│   ├── indexer.py           # Repo cloning + file indexing
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # Chat UI
│   ├── style.css            # Dark theme styling
│   └── script.js            # Frontend logic
└── README.md
```

## 🛠️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/index` | Index a codebase (local path or GitHub URL) |
| POST | `/api/query` | Ask a question about an indexed repo |
| GET | `/api/repos` | List all indexed repositories |
| GET | `/api/index-status/{name}` | Check indexing progress |
| DELETE | `/api/repos/{name}` | Delete an indexed repository |

