"""
Settings for the LLM stack, read from environment variables (or backend/.env).

Defaults run everything locally and for free with Ollama:
    LLM_PROVIDER          "ollama" (default) or "cloud" (the old Groq -> Gemini chain, needs API keys)
    OLLAMA_HOST           where the Ollama server listens (default http://localhost:11434)
    LLM_MODEL             chat model used for answers and the agent (default qwen2.5:3b; qwen2.5:7b is
                          stronger but needs ~5 GB of RAM)
    JUDGE_MODEL           model used by the eval harness to grade answers (default: same as LLM_MODEL)
    LLM_TEMPERATURE       sampling temperature (default 0.2)
    OLLAMA_NUM_CTX        context window in tokens (default 8192, room for several tool results)
    AGENT_MAX_ITERATIONS  most tool-calling rounds before the agent must answer (default 6)
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    ollama_host: str
    llm_model: str
    judge_model: str
    temperature: float
    num_ctx: int
    agent_max_iterations: int


def get_settings() -> Settings:
    """Read the current settings (re-read on every call, so env changes apply immediately)."""
    llm_model = os.environ.get("LLM_MODEL", "qwen2.5:3b")
    return Settings(
        llm_provider=os.environ.get("LLM_PROVIDER", "ollama").strip().lower(),
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        llm_model=llm_model,
        judge_model=os.environ.get("JUDGE_MODEL", llm_model),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        num_ctx=int(os.environ.get("OLLAMA_NUM_CTX", "8192")),
        agent_max_iterations=int(os.environ.get("AGENT_MAX_ITERATIONS", "6")),
    )
