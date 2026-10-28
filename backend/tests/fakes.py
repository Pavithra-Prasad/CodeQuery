"""Test doubles shared by the test modules."""

import re
import zlib

EMBED_DIM = 256


def fake_embed(text: str) -> list[float]:
    """Bag-of-words vector: texts that share words land close together."""
    vec = [0.0] * EMBED_DIM
    for word in re.findall(r"[a-z]+", text.lower()):
        vec[zlib.crc32(word.encode()) % EMBED_DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]
