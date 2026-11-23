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
from collections.abc import ItemsView, Iterator, KeysView, Mapping
from functools import lru_cache

import numpy as np

__all__ = ["BIAS", "PackedFeatures", "featurize", "pack"]

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


class PackedFeatures(Mapping[int, float]):
    """The features of one text in two arrays, for holding many at once.

    A ``dict`` of a few thousand features costs about 100 bytes each; these
    arrays cost 8. Learning holds the features of up to ``max_calls`` texts
    per site, so this keeps a site of long inputs in tens of megabytes rather
    than gigabytes. Reads like the ``dict`` :func:`featurize` returns.
    """

    __slots__ = ("_keys", "_values")

    def __init__(self, features: Mapping[int, float]) -> None:
        # Ids are 32-bit hashes (and 0 for the bias), so they fit in uint32.
        self._keys = np.fromiter(features.keys(), dtype=np.uint32, count=len(features))
        self._values = np.fromiter(features.values(), dtype=np.float32, count=len(features))

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """The feature ids (int64) and weights (float32)."""
        return self._keys.astype(np.int64), self._values

    def __getitem__(self, key: int) -> float:
        hits = np.flatnonzero(self._keys == key)
        if not len(hits):
            raise KeyError(key)
        return float(self._values[hits[0]])

    def __iter__(self) -> Iterator[int]:
        return iter(self._keys.tolist())

    def __len__(self) -> int:
        return len(self._keys)

    def keys(self) -> KeysView[int]:
        return dict.fromkeys(self._keys.tolist()).keys()

    def items(self) -> ItemsView[int, float]:
        return dict(zip(self._keys.tolist(), self._values.tolist(), strict=True)).items()


def pack(features: Mapping[int, float]) -> PackedFeatures:
    """The compact form of :func:`featurize`'s output."""
    return PackedFeatures(features)
