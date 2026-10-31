"""
The agent's tools: typed functions over one indexed repository.

A "document" is one file of the indexed repo. Every tool returns a JSON-serializable dict.
Bad input raises ToolError with a message the model can act on; the agent loop feeds that
message back to the model instead of crashing.
"""

import os
from typing import Callable, Optional

import llm
from embeddings import embed_query
from vector_store import get_chunks, query_documents

MIN_TOP_K = 5   # small models ask for too few results and then miss the relevant chunk
MAX_TOP_K = 10
MAX_PASSAGE_CHARS = 1500
MAX_SECTION_LINES = 150
MAX_LISTED_DOCUMENTS = 200
MAX_SUMMARY_INPUT_CHARS = 12000

SUMMARY_PROMPT = """Summarize this source file for a developer who hasn't seen it.
Cover its purpose, its main functions/classes and what each does, and anything notable
(constants, limits, error handling). Refer to functions by name. Keep it under 150 words.

File: {file}
```
{code}
```"""


class ToolError(ValueError):
    """A tool got input it can't handle. The message tells the model what to do instead."""


def _llm_summarize(prompt: str) -> str:
    """Summarize with the configured local model."""
    return llm.chat([{"role": "user", "content": prompt}]).message.content


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


class RepoTools:
    """The agent's tools, bound to one indexed repository."""

    def __init__(self, repo_name: str, summarize_fn: Optional[Callable[[str], str]] = None):
        """
        Args:
            repo_name: Name of the indexed repository the tools work on.
            summarize_fn: Function that turns a prompt into a summary (defaults to the local LLM).
        """
        self.repo_name = repo_name
        self._summarize = summarize_fn or _llm_summarize

    def functions(self) -> list[Callable]:
        """The tool functions, in the order they're offered to the model."""
        return [self.search_docs, self.list_documents, self.read_document_section, self.summarize_document]

    # ── Tools ─────────────────────────────────────────────────

    def search_docs(self, query: str, top_k: int = 5) -> dict:
        """Semantic search over the indexed code. Start here to find where something is implemented.

        Args:
            query: What to look for, in plain words or code terms (e.g. "how are passwords hashed").
            top_k: How many passages to return, 5 to 10.
        """
        if not query.strip():
            raise ToolError("query is empty. Describe what to search for.")
        top_k = max(MIN_TOP_K, min(top_k, MAX_TOP_K))

        results = query_documents(self.repo_name, embed_query(query), top_k=top_k)
        passages = []
        for chunk_id, doc, meta, distance in zip(
            results["ids"], results["documents"], results["metadatas"], results["distances"]
        ):
            passages.append({
                "chunk_id": chunk_id,
                "file": meta.get("file_path", "unknown"),
                "language": meta.get("language", ""),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                # Squared L2 distance on unit vectors -> cosine similarity (same as rag_engine)
                "relevance": round(max(0.0, 1 - distance / 2), 3),
                "text": _truncate(doc, MAX_PASSAGE_CHARS),
            })
        return {"query": query, "results": passages}

    def list_documents(self) -> dict:
        """List the files in the indexed codebase with their language, number of chunks and length in lines."""
        files: dict[str, dict] = {}
        for chunk in get_chunks(self.repo_name):
            meta = chunk["metadata"]
            path = meta.get("file_path", "unknown")
            info = files.setdefault(path, {"file": path, "language": meta.get("language", ""), "chunks": 0, "lines": 0})
            info["chunks"] += 1
            info["lines"] = max(info["lines"], meta.get("end_line", 0) or 0)

        documents = sorted(files.values(), key=lambda d: d["file"])
        result = {"repo": self.repo_name, "total_documents": len(documents), "documents": documents[:MAX_LISTED_DOCUMENTS]}
        if len(documents) > MAX_LISTED_DOCUMENTS:
            result["note"] = f"Only the first {MAX_LISTED_DOCUMENTS} files are listed. Use search_docs to find others."
        return result

    def read_document_section(self, doc_name: str, start: int, end: int) -> dict:
        """Read specific lines of one file, e.g. to see a whole function after search_docs found part of it.

        Args:
            doc_name: File path as shown by search_docs or list_documents (e.g. "app/auth.py").
            start: First line to read (1-based).
            end: Last line to read, inclusive. At most 150 lines are returned per call.
        """
        path = self._resolve(doc_name)
        lines = self._file_lines(path)
        last = max(lines)

        start = max(1, start)
        if end < start:
            raise ToolError(f"end ({end}) is before start ({start}).")
        if start > last:
            raise ToolError(f"{path} only has {last} lines.")
        stop = min(end, last, start + MAX_SECTION_LINES - 1)

        text = "\n".join(f"{n}: {lines[n]}" for n in range(start, stop + 1) if n in lines)
        result = {"file": path, "start_line": start, "end_line": stop, "total_lines": last, "text": text}
        if stop < min(end, last):
            result["note"] = f"Only {MAX_SECTION_LINES} lines are returned per call. Ask for {stop + 1}-{end} next."
        return result

    def summarize_document(self, doc_name: str) -> dict:
        """Summarize one whole file: its purpose and its main functions and classes. Good for overview questions.

        Args:
            doc_name: File path as shown by search_docs or list_documents (e.g. "app/auth.py").
        """
        path = self._resolve(doc_name)
        lines = self._file_lines(path)
        code = "\n".join(lines[n] for n in sorted(lines))
        truncated = len(code) > MAX_SUMMARY_INPUT_CHARS

        summary = self._summarize(SUMMARY_PROMPT.format(file=path, code=code[:MAX_SUMMARY_INPUT_CHARS]))
        return {"file": path, "total_lines": max(lines), "summary": summary.strip(), "truncated_input": truncated}

    # ── Helpers ───────────────────────────────────────────────

    def _resolve(self, doc_name: str) -> str:
        """Map a model-supplied name to an indexed file path (exact path, or a unique file name / path suffix)."""
        name = doc_name.strip().removeprefix("./")
        if not name:
            raise ToolError("doc_name is empty. Call list_documents to see the available files.")
        if get_chunks(self.repo_name, file_path=name):
            return name

        paths = sorted({c["metadata"].get("file_path", "") for c in get_chunks(self.repo_name)})
        matches = [p for p in paths if p.endswith("/" + name) or os.path.basename(p) == name]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ToolError(f"'{doc_name}' matches several files: {', '.join(matches)}. Use the full path.")
        raise ToolError(f"No document named '{doc_name}'. Call list_documents to see the available files.")

    def _file_lines(self, path: str) -> dict[int, str]:
        """
        Rebuild a file's lines from its stored chunks (the original files aren't kept after indexing).
        Line numbers come from each chunk's start line.
        """
        lines: dict[int, str] = {}
        for chunk in get_chunks(self.repo_name, file_path=path):
            start = chunk["metadata"].get("start_line", 1) or 1
            for offset, text in enumerate(chunk["document"].split("\n")):
                lines.setdefault(start + offset, text)
        if not lines:
            raise ToolError(f"No content stored for '{path}'.")
        return lines
