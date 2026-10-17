"""Text features for the local students: hashed words, word pairs and character n-grams.

No vocabulary file and no model download: every feature is a 32-bit hash of a
token, so the same text always gives the same features on every machine.
Character n-grams inside words make the features tolerant of typos and
inflections, and work for languages without spaces between words.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from functools import lru_cache

__all__ = ["BIAS", "featurize"]

_WORD = re.compile(r"\w+", re.UNICODE)
BIAS = 0
"""Feature id present in every example, so no example is empty."""
MAX_CHARS = 4000


@lru_cache(maxsize=1 << 18)
def _hash(token: str) -> int:
    # Tokens repeat across calls, so the cache saves most of the hashing.
    return zlib.crc32(token.encode("utf-8")) or 1


_LOG = [0.0] + [1.0 + math.log(c) for c in range(1, 64)]


def featurize(text: str) -> dict[int, float]:
    """Sparse, L2-normalized features of ``text`` as ``{feature id: weight}``."""
    if len(text) > MAX_CHARS:
        half = MAX_CHARS // 2
        text = text[:half] + "\n" + text[-half:]
    words = _WORD.findall(text.lower())
    counts: Counter[str] = Counter()
    for i, word in enumerate(words):
        counts["w" + word] += 1
        if i:
            counts["b" + words[i - 1] + " " + word] += 1
        padded = f"<{word}>"
        for n in (3, 4):
            for j in range(len(padded) - n + 1):
                counts["c" + padded[j : j + n]] += 1
    features: dict[int, float] = {}
    for token, count in counts.items():
        key = _hash(token)
        features[key] = features.get(key, 0.0) + (
            _LOG[count] if count < len(_LOG) else 1.0 + math.log(count)
        )
    norm = math.sqrt(sum(v * v for v in features.values())) or 1.0
    scaled = {key: value / norm for key, value in features.items()}
    scaled[BIAS] = 1.0
    return scaled
