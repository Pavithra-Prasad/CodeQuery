"""
Code-aware chunking module.
Splits source code files into logical chunks (functions, classes, blocks)
while preserving metadata like file path and language.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional


# File extensions we support and their language names
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".txt": "text",
    ".toml": "toml",
    ".cfg": "config",
    ".ini": "config",
    ".env": "config",
}

# Directories to always skip
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "env", ".env", "dist", "build", ".next", ".nuxt",
    "target", "bin", "obj", ".idea", ".vscode",
    ".gradle", "vendor", "coverage", ".tox",
    "egg-info", ".eggs", ".mypy_cache", ".pytest_cache",
}

# Files to always skip
SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Pipfile.lock", "poetry.lock", ".DS_Store",
}

# Max file size to process (500KB)
MAX_FILE_SIZE = 500 * 1024


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    content: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    chunk_type: str = "code"  # "code", "function", "class", "module_header"
    metadata: dict = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        """Generate a unique ID for this chunk."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


def get_language(file_path: str) -> Optional[str]:
    """Get the language of a file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    return SUPPORTED_EXTENSIONS.get(ext)


def should_skip_path(path: str) -> bool:
    """Check if a path should be skipped during indexing."""
    parts = path.split(os.sep)
    for part in parts:
        if part in SKIP_DIRS:
            return True
    basename = os.path.basename(path)
    if basename in SKIP_FILES:
        return True
    return False


def chunk_code_file(file_path: str, base_path: str = "", max_chunk_lines: int = 60, overlap_lines: int = 5) -> list[CodeChunk]:
    """
    Split a code file into logical chunks.
    
    Strategy:
    1. Try to split on function/class boundaries
    2. If a block is too large, use sliding window with overlap
    3. Always preserve the file header (imports) as a separate chunk
    
    Args:
        file_path: Absolute path to the file
        base_path: Base path to make relative paths from
        max_chunk_lines: Maximum lines per chunk
        overlap_lines: Number of overlapping lines between chunks
    
    Returns:
        List of CodeChunk objects
    """
    # Read the file
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except (OSError, IOError):
        return []

    if not content.strip():
        return []

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        return []

    language = get_language(file_path)
    if language is None:
        return []

    # Make relative path for display
    rel_path = os.path.relpath(file_path, base_path) if base_path else file_path
    
    lines = content.split("\n")
    
    # If the file is small enough, return as a single chunk
    if len(lines) <= max_chunk_lines:
        return [CodeChunk(
            content=content,
            file_path=rel_path,
            language=language,
            start_line=1,
            end_line=len(lines),
            chunk_type="module",
        )]

    # Try to find logical boundaries (functions, classes)
    boundaries = _find_boundaries(lines, language)
    
    if boundaries:
        return _chunk_by_boundaries(lines, boundaries, rel_path, language, max_chunk_lines, overlap_lines)
    else:
        return _chunk_by_sliding_window(lines, rel_path, language, max_chunk_lines, overlap_lines)


def _find_boundaries(lines: list[str], language: str) -> list[int]:
    """Find line indices where new logical blocks start (functions, classes)."""
    boundaries = []
    
    # Language-specific patterns for block starts
    patterns = {
        "python": r"^(class |def |async def )",
        "javascript": r"^(function |class |const \w+ = (\(|async)|export (default )?(function|class|const))",
        "typescript": r"^(function |class |const \w+ = (\(|async)|export (default )?(function|class|const)|interface |type )",
        "java": r"^(((public|private|protected|static|final|abstract|synchronized)\s+)*(class |interface |enum |void |int |String |boolean |double |float |long ))",
        "go": r"^(func |type )",
        "rust": r"^(fn |pub fn |struct |impl |enum |trait )",
        "cpp": r"^(\w+\s+\w+\s*\(|class |struct |namespace )",
        "c": r"^(\w+\s+\w+\s*\(|struct |enum )",
        "ruby": r"^(def |class |module )",
        "php": r"^(\s*(public|private|protected)?\s*(function |class ))",
    }
    
    pattern = patterns.get(language)
    if not pattern:
        return []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and re.match(pattern, stripped):
            boundaries.append(i)
    
    return boundaries


def _chunk_by_boundaries(
    lines: list[str],
    boundaries: list[int],
    file_path: str,
    language: str,
    max_chunk_lines: int,
    overlap_lines: int,
) -> list[CodeChunk]:
    """Create chunks based on logical boundaries."""
    chunks = []
    
    # If the first boundary isn't at line 0, create a header chunk
    if boundaries[0] > 0:
        header_lines = lines[:boundaries[0]]
        header_content = "\n".join(header_lines).strip()
        if header_content:
            chunks.append(CodeChunk(
                content=header_content,
                file_path=file_path,
                language=language,
                start_line=1,
                end_line=boundaries[0],
                chunk_type="module_header",
            ))
    
    # Create chunks between boundaries
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
        block_lines = lines[start:end]
        block_content = "\n".join(block_lines).strip()
        
        if not block_content:
            continue
        
        # If this block is too large, sub-chunk it
        if len(block_lines) > max_chunk_lines:
            sub_chunks = _chunk_by_sliding_window(
                block_lines, file_path, language,
                max_chunk_lines, overlap_lines,
                start_offset=start
            )
            chunks.extend(sub_chunks)
        else:
            chunks.append(CodeChunk(
                content=block_content,
                file_path=file_path,
                language=language,
                start_line=start + 1,
                end_line=end,
                chunk_type="function" if any(
                    kw in block_lines[0] for kw in ["def ", "func ", "function ", "fn "]
                ) else "class",
            ))
    
    return chunks


def _chunk_by_sliding_window(
    lines: list[str],
    file_path: str,
    language: str,
    max_chunk_lines: int,
    overlap_lines: int,
    start_offset: int = 0,
) -> list[CodeChunk]:
    """Create chunks using a sliding window approach."""
    chunks = []
    i = 0
    
    while i < len(lines):
        end = min(i + max_chunk_lines, len(lines))
        chunk_lines = lines[i:end]
        chunk_content = "\n".join(chunk_lines).strip()
        
        if chunk_content:
            chunks.append(CodeChunk(
                content=chunk_content,
                file_path=file_path,
                language=language,
                start_line=i + start_offset + 1,
                end_line=end + start_offset,
                chunk_type="code",
            ))
        
        if end >= len(lines):
            break
        
        i = end - overlap_lines
    
    return chunks


def walk_codebase(root_path: str) -> list[str]:
    """
    Walk a codebase directory and return all indexable file paths.
    Skips binary files, node_modules, .git, etc.
    """
    files = []
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Filter out directories we should skip (modifies in-place)
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            
            # Check the path inside the repo only, so a parent folder like build/ doesn't skip everything
            if should_skip_path(os.path.relpath(file_path, root_path)):
                continue
            
            if get_language(file_path) is None:
                continue
            
            # Skip large files
            try:
                if os.path.getsize(file_path) > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            
            files.append(file_path)
    
    return files
