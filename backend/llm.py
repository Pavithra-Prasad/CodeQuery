"""
Local LLM access through Ollama, plus a startup check that the server is up and the models are pulled.
"""

from typing import Any, Optional, Sequence

import ollama

from config import get_settings


class OllamaNotReady(RuntimeError):
    """Ollama isn't running, or a required model hasn't been pulled. The message says how to fix it."""


def get_client() -> ollama.Client:
    """An Ollama client pointed at the configured host."""
    return ollama.Client(host=get_settings().ollama_host)


def _with_tag(model: str) -> str:
    """Ollama lists models with a tag; a bare name means ':latest'."""
    return model if ":" in model else f"{model}:latest"


def check_ollama(models: Optional[Sequence[str]] = None) -> None:
    """
    Verify that Ollama is reachable and that every required model is pulled.

    Args:
        models: Models to check. Defaults to the configured LLM_MODEL.

    Raises:
        OllamaNotReady: with the exact command to run when something is missing.
    """
    settings = get_settings()
    required = list(dict.fromkeys(models or [settings.llm_model]))

    try:
        available = {m.model for m in get_client().list().models}
    except Exception as e:
        raise OllamaNotReady(
            f"Can't reach Ollama at {settings.ollama_host} ({e}).\n"
            "Install it from https://ollama.com (or: brew install ollama), then start it:\n"
            "  ollama serve"
        ) from e

    missing = [m for m in required if _with_tag(m) not in available]
    if missing:
        commands = "\n".join(f"  ollama pull {m}" for m in missing)
        raise OllamaNotReady(f"Ollama is running but these models aren't pulled yet: {', '.join(missing)}\nRun:\n{commands}")


def chat(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    model: Optional[str] = None,
    format: Optional[Any] = None,
    temperature: Optional[float] = None,
) -> ollama.ChatResponse:
    """
    Send a chat request to the configured Ollama model.

    Args:
        messages: Chat history in Ollama's format ({"role": ..., "content": ...}).
        tools: Tool schemas the model may call (JSON-schema dicts).
        model: Override the configured LLM_MODEL.
        format: "json" or a JSON schema to force structured output.
        temperature: Override the configured LLM_TEMPERATURE.

    Raises:
        OllamaNotReady: if the server can't be reached or the model is missing.
    """
    settings = get_settings()
    model = model or settings.llm_model
    options = {
        "temperature": settings.temperature if temperature is None else temperature,
        "num_ctx": settings.num_ctx,
    }
    try:
        return get_client().chat(model=model, messages=messages, tools=tools, format=format, options=options)
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise OllamaNotReady(f"Model '{model}' isn't pulled yet. Run:\n  ollama pull {model}") from e
        raise
    except ConnectionError as e:
        raise OllamaNotReady(
            f"Can't reach Ollama at {settings.ollama_host}. Start it with:\n  ollama serve"
        ) from e
