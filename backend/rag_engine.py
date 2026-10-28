"""
RAG Engine — the core of the Q&A system.
Orchestrates: embed query → retrieve chunks → build prompt → call LLM → return answer.
Supports Groq (free, fast) and Gemini (free tier) as LLM providers.
"""

import os
from embeddings import embed_query
from vector_store import query_documents


# System prompt for the LLM
SYSTEM_PROMPT = """You are a helpful code assistant that answers questions about a codebase.
You will be given relevant code snippets retrieved from the codebase as context.

RULES:
1. Answer ONLY based on the provided code context. If the context doesn't contain enough information, say so honestly.
2. Always cite the source files when referencing code (e.g., "In `src/auth.py` (lines 15-30)...").
3. When showing code, use proper markdown code blocks with the correct language identifier.
4. Be concise but thorough. Explain the "why" behind code patterns, not just the "what".
5. If asked about something not in the context, say "I don't have enough context about that part of the codebase."
6. Structure your answers clearly with headers or bullet points when appropriate.
"""


def build_context(results: dict) -> str:
    """Build a context string from retrieved chunks for the LLM prompt."""
    if not results["documents"]:
        return "No relevant code found in the indexed codebase."
    
    context_parts = []
    for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        file_path = meta.get("file_path", "unknown")
        language = meta.get("language", "")
        start_line = meta.get("start_line", "?")
        end_line = meta.get("end_line", "?")
        
        context_parts.append(
            f"### Source {i+1}: `{file_path}` (lines {start_line}-{end_line})\n"
            f"```{language}\n{doc}\n```"
        )
    
    return "\n\n".join(context_parts)


def format_sources(results: dict) -> list[dict]:
    """Format source information for the frontend."""
    sources = []
    seen = set()
    
    for doc, meta, distance in zip(
        results["documents"], results["metadatas"], results["distances"]
    ):
        file_path = meta.get("file_path", "unknown")
        key = f"{file_path}:{meta.get('start_line', 0)}"
        
        if key not in seen:
            seen.add(key)
            sources.append({
                "file_path": file_path,
                "language": meta.get("language", ""),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "snippet": doc[:200] + "..." if len(doc) > 200 else doc,
                # ChromaDB's default distance is squared L2 (0-4). For the unit-length MiniLM
                # embeddings that equals 2 - 2*cosine, so cosine similarity = 1 - distance / 2.
                "relevance": round(max(0.0, 1 - distance / 2), 3),
            })
    
    return sources


def _call_groq(user_prompt: str) -> str:
    """Call Groq API (free, fast)."""
    from groq import Groq
    
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content


def _call_gemini(user_prompt: str) -> str:
    """Call Gemini API (free tier)."""
    import google.generativeai as genai
    
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-lite",
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(user_prompt)
    return response.text


def _call_llm(user_prompt: str) -> str:
    """Try available LLM providers in order: Groq → Gemini."""
    errors = []
    
    # Try Groq first (free, fast, generous limits)
    if os.environ.get("GROQ_API_KEY"):
        try:
            print("🤖 Using Groq (Llama 3.3 70B)...")
            return _call_groq(user_prompt)
        except Exception as e:
            print(f"⚠️ Groq failed: {e}")
            errors.append(f"Groq: {e}")
    
    # Try Gemini as fallback
    if os.environ.get("GEMINI_API_KEY"):
        try:
            print("🤖 Using Gemini...")
            return _call_gemini(user_prompt)
        except Exception as e:
            print(f"⚠️ Gemini failed: {e}")
            errors.append(f"Gemini: {e}")
    
    # No providers worked
    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        raise ValueError(
            "No API key set! Set at least one:\n"
            "  export GROQ_API_KEY='your-key'  (get free at https://console.groq.com)\n"
            "  export GEMINI_API_KEY='your-key' (get free at https://aistudio.google.com)"
        )
    
    raise Exception(
        "All LLM providers failed:\n" + "\n".join(errors) +
        "\n\nTry getting a free Groq API key at https://console.groq.com"
    )


async def ask_question(question: str, repo_name: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline: embed query → retrieve → generate answer.
    """
    # Step 1: Embed the query
    query_embedding = embed_query(question)
    
    # Step 2: Retrieve relevant chunks from ChromaDB
    results = query_documents(repo_name, query_embedding, top_k=top_k)
    
    if not results["documents"]:
        return {
            "answer": "I couldn't find any relevant code in the indexed codebase. "
                      "Please make sure the repository has been indexed first.",
            "sources": [],
        }
    
    # Step 3: Build context and prompt
    context = build_context(results)
    
    user_prompt = f"""## Retrieved Code Context
{context}

## Question
{question}

Please answer the question based on the code context above. Cite specific files and line numbers."""
    
    # Step 4: Call LLM (tries Groq first, then Gemini)
    answer = _call_llm(user_prompt)
    
    # Step 5: Format and return
    sources = format_sources(results)
    
    return {
        "answer": answer,
        "sources": sources,
    }
