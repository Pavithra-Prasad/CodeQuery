"""Unit tests for code-aware chunking and codebase walking."""

import os

import pytest

import chunker
from chunker import CodeChunk, chunk_code_file, get_language, should_skip_path, walk_codebase


def python_module(num_functions: int = 3, body_lines: int = 20) -> str:
    """An import header followed by functions and a class, long enough to need splitting."""
    lines = ["import os", "import sys", ""]
    for n in range(num_functions):
        lines.append(f"def function_{n}(value):")
        lines += [f"    value = value + {i}" for i in range(body_lines)]
        lines += ["    return value", ""]
    lines.append("class Settings:")
    lines += [f"    option_{i} = {i}" for i in range(body_lines)]
    return "\n".join(lines)


# ── Language detection and skipping ─────────────────────────────

@pytest.mark.parametrize("path, language", [
    ("app.py", "python"),
    ("APP.PY", "python"),
    ("src/index.tsx", "typescript"),
    ("Main.java", "java"),
    ("query.sql", "sql"),
    ("logo.png", None),
    ("Makefile", None),
])
def test_get_language(path, language):
    assert get_language(path) == language


@pytest.mark.parametrize("path, skipped", [
    (os.path.join("repo", "node_modules", "lib", "index.js"), True),
    (os.path.join("repo", ".git", "config"), True),
    (os.path.join("repo", "venv", "lib", "site.py"), True),
    (os.path.join("repo", "package-lock.json"), True),
    (os.path.join("repo", ".DS_Store"), True),
    (os.path.join("repo", "src", "app.py"), False),
])
def test_should_skip_path(path, skipped):
    assert should_skip_path(path) is skipped


# ── chunk_code_file ─────────────────────────────────────────────

@pytest.mark.parametrize("content", ["", "   \n\n\t\n"])
def test_empty_files_give_no_chunks(tmp_path, content):
    f = tmp_path / "empty.py"
    f.write_text(content)
    assert chunk_code_file(str(f)) == []


def test_unsupported_files_give_no_chunks(tmp_path):
    f = tmp_path / "notes.unknownext"
    f.write_text("some text\n")
    assert chunk_code_file(str(f)) == []


def test_missing_file_gives_no_chunks(tmp_path):
    assert chunk_code_file(str(tmp_path / "does-not-exist.py")) == []


def test_files_over_the_size_limit_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(chunker, "MAX_FILE_SIZE", 100)
    f = tmp_path / "big.py"
    f.write_text("x = 1\n" * 50)
    assert chunk_code_file(str(f)) == []


def test_small_file_is_one_module_chunk_with_a_relative_path(tmp_path):
    f = tmp_path / "src" / "util.py"
    f.parent.mkdir()
    f.write_text("def add(a, b):\n    return a + b\n")
    chunks = chunk_code_file(str(f), base_path=str(tmp_path))
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.file_path, c.language, c.chunk_type) == (os.path.join("src", "util.py"), "python", "module")
    assert (c.start_line, c.end_line) == (1, 3)
    assert "return a + b" in c.content


def test_python_file_splits_on_function_and_class_boundaries(tmp_path):
    f = tmp_path / "service.py"
    f.write_text(python_module())
    chunks = chunk_code_file(str(f))

    assert [c.chunk_type for c in chunks] == ["module_header", "function", "function", "function", "class"]
    assert chunks[0].content == "import os\nimport sys"
    assert chunks[1].content.startswith("def function_0")
    assert chunks[-1].content.startswith("class Settings")
    # Chunks cover the file in order with no gaps
    assert chunks[0].start_line == 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start_line == prev.end_line + 1
    assert chunks[-1].end_line == len(python_module().split("\n"))


def test_long_function_is_sub_chunked_with_overlap(tmp_path):
    body = "\n".join(f"    total = total + {i}" for i in range(150))
    f = tmp_path / "long.py"
    f.write_text(f"import os\n\ndef long_function():\n    total = 0\n{body}\n    return total\n")
    chunks = chunk_code_file(str(f), max_chunk_lines=60, overlap_lines=5)

    assert chunks[0].chunk_type == "module_header"
    pieces = chunks[1:]
    assert len(pieces) == 3 and all(c.chunk_type == "code" for c in pieces)
    assert pieces[0].content.startswith("def long_function")
    for prev, nxt in zip(pieces, pieces[1:]):
        assert nxt.start_line == prev.end_line - 5 + 1   # 5 lines shared between neighbours
        assert nxt.end_line - nxt.start_line + 1 <= 60


def test_file_without_boundaries_uses_a_sliding_window(tmp_path):
    f = tmp_path / "guide.md"
    f.write_text("\n".join(f"Line {i} of the guide." for i in range(1, 151)))
    chunks = chunk_code_file(str(f), max_chunk_lines=60, overlap_lines=5)

    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (56, 115), (111, 150)]
    assert all(c.chunk_type == "code" and c.language == "markdown" for c in chunks)


def test_javascript_boundaries(tmp_path):
    body = "\n".join(f"  step{i}();" for i in range(25))
    f = tmp_path / "app.js"
    f.write_text(
        f"const express = require('express');\n\n"
        f"function start() {{\n{body}\n}}\n\n"
        f"const handler = (req, res) => {{\n{body}\n}};\n\n"
        f"export default class App {{\n{body}\n}}\n"
    )
    chunks = chunk_code_file(str(f))
    starts = [c.content.split("\n")[0] for c in chunks]
    assert starts == [
        "const express = require('express');",
        "function start() {",
        "const handler = (req, res) => {",
        "export default class App {",
    ]


def test_java_class_is_a_boundary(tmp_path):
    body = "\n".join(f"        total += {i};" for i in range(70))
    f = tmp_path / "Main.java"
    f.write_text(f"package demo;\n\npublic class Main {{\n    void run() {{\n{body}\n    }}\n}}\n")
    chunks = chunk_code_file(str(f))
    assert any(c.content.startswith("public class Main") for c in chunks)


def test_java_static_methods_are_boundaries():
    lines = ["public class Main {", "    public static void main(String[] args) {", "    }", "}"]
    assert 1 in chunker._find_boundaries(lines, "java")


def test_chunk_id_combines_path_and_lines():
    c = CodeChunk(content="x", file_path="src/a.py", language="python", start_line=3, end_line=9)
    assert c.id == "src/a.py:3-9"


# ── walk_codebase ───────────────────────────────────────────────

def test_walk_codebase_returns_only_indexable_source_files(sample_repo):
    found = sorted(os.path.relpath(p, sample_repo) for p in walk_codebase(str(sample_repo)))
    assert found == ["README.md", "auth.py", "orders.js"]


def test_walk_codebase_skips_files_over_the_size_limit(sample_repo, monkeypatch):
    monkeypatch.setattr(chunker, "MAX_FILE_SIZE", 120)
    (sample_repo / "huge.py").write_text("x = 1\n" * 100)
    found = [os.path.basename(p) for p in walk_codebase(str(sample_repo))]
    assert "huge.py" not in found


def test_walk_codebase_ignores_skip_names_above_the_repo_root(tmp_path):
    repo = tmp_path / "build" / "my-project"
    repo.mkdir(parents=True)
    (repo / "app.py").write_text("print('hi')\n")
    assert [os.path.basename(p) for p in walk_codebase(str(repo))] == ["app.py"]
