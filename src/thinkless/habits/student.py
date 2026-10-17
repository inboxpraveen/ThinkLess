"""Local students: small models that learn a site's answers from the LLM.

Each answer field gets its own student:

* :class:`LinearModel`, a multinomial logistic regression over hashed text
  features, for closed fields (``choice``, ``yes_no``, ``score``). It trains
  in seconds on a CPU and answers in well under a millisecond.
* :class:`ShapeExtractor` for ``extract`` fields: it learns the shape of the
  values the LLM copied out of the input (``\\d{4,5}`` for order numbers, for
  example) and finds that shape in new inputs.
* :class:`ConstantModel` for fields that always had the same answer.

:class:`SiteModel` holds the students of one site and answers a whole call:
every field's label, and a joint confidence (the product of the fields'
confidences), which is what the gate thresholds.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .features import BIAS, featurize
from .wire import MISSING, label_of, value_of

__all__ = ["ConstantModel", "LinearModel", "ShapeExtractor", "SiteModel"]

Features = Mapping[int, float]


class LinearModel:
    """Multinomial logistic regression on sparse features, trained with AdaGrad.

    Args:
        labels: Class labels, in weight-column order.
        vocab: Feature id to weight row.
        weights: ``(features, classes)`` weights.
    """

    kind = "linear"

    def __init__(
        self, labels: Sequence[str], vocab: Mapping[int, int], weights: np.ndarray
    ) -> None:
        self.labels = list(labels)
        self.vocab = dict(vocab)
        self.weights = weights

    @classmethod
    def fit(
        cls,
        examples: Sequence[Features],
        labels: Sequence[str],
        *,
        epochs: int = 6,
        rate: float = 0.6,
        l2: float = 1e-4,
        batch: int = 64,
        max_features: int = 1 << 16,
        min_count: int = 2,
        seed: int = 0,
    ) -> LinearModel:
        classes = sorted(set(labels))
        column = {label: i for i, label in enumerate(classes)}
        frequency: Counter[int] = Counter()
        for example in examples:
            frequency.update(example.keys())
        # Features seen in a single call cannot generalize; dropping them keeps
        # the weights small enough to stay in cache while training.
        kept = [f for f, count in frequency.most_common(max_features) if count >= min_count]
        if BIAS not in kept:
            kept.append(BIAS)
        vocab = {feature: row for row, feature in enumerate(kept)}
        rows: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for example in examples:
            pairs = [(vocab[f], v) for f, v in example.items() if f in vocab]
            rows.append(np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs)))
            values.append(np.fromiter((p[1] for p in pairs), dtype=np.float32, count=len(pairs)))
        target = np.array([column[label] for label in labels], dtype=np.int64)
        weights = np.zeros((len(kept), len(classes)), dtype=np.float32)
        history = np.full_like(weights, 1e-6)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(examples))
            for start in range(0, len(order), batch):
                members = order[start : start + batch]
                lengths = np.array([len(rows[m]) for m in members])
                index = np.concatenate([rows[m] for m in members])
                value = np.concatenate([values[m] for m in members])
                offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
                logits = np.add.reduceat(weights[index] * value[:, None], offsets, axis=0)
                logits -= logits.max(axis=1, keepdims=True)
                probs = np.exp(logits)
                probs /= probs.sum(axis=1, keepdims=True)
                probs[np.arange(len(members)), target[members]] -= 1.0
                owner = np.repeat(np.arange(len(members)), lengths)
                # Sum the per-feature gradients with one sort and a segmented sum.
                order_nz = np.argsort(index, kind="stable")
                ordered = index[order_nz]
                starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
                unique = ordered[starts]
                contrib = probs[owner[order_nz]] * value[order_nz, None]
                grad = np.add.reduceat(contrib, starts, axis=0)
                grad += l2 * weights[unique]
                seen = history[unique] + grad * grad
                history[unique] = seen
                weights[unique] -= rate * grad / np.sqrt(seen)
        return cls(classes, vocab, weights)

    def proba(self, example: Features) -> np.ndarray:
        pairs = [(self.vocab[f], v) for f, v in example.items() if f in self.vocab]
        if not pairs:
            return np.full(len(self.labels), 1.0 / len(self.labels))
        index = np.array([p[0] for p in pairs])
        value = np.array([p[1] for p in pairs], dtype=np.float32)
        logits = (self.weights[index] * value[:, None]).sum(axis=0).astype(np.float64)
        logits -= logits.max()
        probs = np.exp(logits)
        return np.asarray(probs / probs.sum())

    def predict(self, example: Features, text: str) -> tuple[str, float]:
        probs = self.proba(example)
        best = int(probs.argmax())
        return self.labels[best], float(probs[best])

    def arrays(self, prefix: str) -> dict[str, np.ndarray]:
        keys = np.array(list(self.vocab), dtype=np.int64)
        order = np.array([self.vocab[k] for k in self.vocab], dtype=np.int64)
        return {f"{prefix}.keys": keys, f"{prefix}.weights": self.weights[order]}

    def meta(self) -> dict[str, Any]:
        return {"kind": self.kind, "labels": self.labels}

    @classmethod
    def restore(
        cls, meta: Mapping[str, Any], arrays: Mapping[str, np.ndarray], prefix: str
    ) -> LinearModel:
        keys = arrays[f"{prefix}.keys"]
        vocab = {int(k): i for i, k in enumerate(keys.tolist())}
        return cls(meta["labels"], vocab, np.asarray(arrays[f"{prefix}.weights"], dtype=np.float32))


class ConstantModel:
    """A field that always had the same answer."""

    kind = "constant"

    def __init__(self, label: str, seen: int) -> None:
        self.label = label
        self.seen = seen

    def predict(self, example: Features, text: str) -> tuple[str, float]:
        return self.label, (self.seen + 1) / (self.seen + 2)

    def arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {}

    def meta(self) -> dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "seen": self.seen}

    @classmethod
    def restore(
        cls, meta: Mapping[str, Any], arrays: Mapping[str, np.ndarray], prefix: str
    ) -> ConstantModel:
        return cls(meta["label"], int(meta["seen"]))


_RUN = re.compile(r"\d+|[A-Z]+|[a-z]+|\s+|.", re.DOTALL)
_CLASS = {"d": r"\d", "U": "[A-Z]", "L": "[a-z]", "s": r"\s"}
MAX_SHAPES = 6


def _shape(value: str) -> list[tuple[str, int]]:
    parts: list[tuple[str, int]] = []
    for match in _RUN.finditer(value):
        run = match.group()
        if run.isdigit() and run.isascii():
            parts.append(("d", len(run)))
        elif run.isascii() and run.isalpha() and run.isupper():
            parts.append(("U", len(run)))
        elif run.isascii() and run.isalpha() and run.islower():
            parts.append(("L", len(run)))
        elif run.isspace():
            parts.append(("s", 1))
        else:
            parts.append(("=" + run, 1))
    return parts


def _pattern(values: Sequence[str]) -> str | None:
    """A regular expression matching the shapes of ``values``, or ``None`` if they vary too much."""
    groups: dict[tuple[str, ...], list[list[int]]] = {}
    for value in values:
        shape = _shape(value)
        signature = tuple(kind for kind, _ in shape)
        groups.setdefault(signature, []).append([length for _, length in shape])
    common = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    covered = 0
    alternatives = []
    for signature, lengths in common[:MAX_SHAPES]:
        covered += len(lengths)
        pieces = []
        for position, kind in enumerate(signature):
            if kind.startswith("="):
                pieces.append(re.escape(kind[1:]))
                continue
            low = min(item[position] for item in lengths)
            high = max(item[position] for item in lengths)
            quantifier = f"{{{low}}}" if low == high else f"{{{low},{high}}}"
            pieces.append(_CLASS[kind] + quantifier)
        alternatives.append("".join(pieces))
    if not alternatives or covered < 0.9 * len(values):
        return None
    return r"(?<![0-9A-Za-z])(?:" + "|".join(alternatives) + r")(?![0-9A-Za-z])"


class ShapeExtractor:
    """Finds values shaped like the ones the LLM extracted before.

    Confidence depends on how many candidates the input holds (none, one,
    several) and is the rate at which the LLM agreed with that rule on the
    training calls, smoothed toward one half.
    """

    kind = "extract"

    def __init__(self, pattern: str, empty: str, rates: Mapping[str, float]) -> None:
        self.pattern = pattern
        self.empty = empty
        self.rates = dict(rates)
        self._regex = re.compile(pattern)

    @classmethod
    def fit(cls, texts: Sequence[str], labels: Sequence[str]) -> ShapeExtractor | None:
        values = [
            value_of(label)
            for label in labels
            if label not in (MISSING, "null") and isinstance(value_of(label), str)
        ]
        if not values:
            return None
        pattern = _pattern(values)
        if pattern is None:
            return None
        empties = Counter(label for label in labels if label in (MISSING, "null"))
        empty = empties.most_common(1)[0][0] if empties else "null"
        model = cls(pattern, empty, {})
        tally: dict[str, list[int]] = {"none": [0, 0], "one": [0, 0], "many": [0, 0]}
        for text, label in zip(texts, labels, strict=True):
            predicted, case = model._find(text)
            tally[case][0] += predicted == label
            tally[case][1] += 1
        model.rates = {case: (hit + 1) / (total + 2) for case, (hit, total) in tally.items()}
        return model

    def _find(self, text: str) -> tuple[str, str]:
        found: list[str] = []
        for match in self._regex.finditer(text):
            if match.group() not in found:
                found.append(match.group())
        if not found:
            return self.empty, "none"
        return label_of(found[0]), "one" if len(found) == 1 else "many"

    def predict(self, example: Features, text: str) -> tuple[str, float]:
        label, case = self._find(text)
        return label, self.rates.get(case, 0.5)

    def arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {}

    def meta(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "pattern": self.pattern,
            "empty": self.empty,
            "rates": self.rates,
        }

    @classmethod
    def restore(
        cls, meta: Mapping[str, Any], arrays: Mapping[str, np.ndarray], prefix: str
    ) -> ShapeExtractor:
        return cls(meta["pattern"], meta["empty"], meta["rates"])


FieldModel = LinearModel | ConstantModel | ShapeExtractor
_KINDS: dict[str, Any] = {
    "linear": LinearModel,
    "constant": ConstantModel,
    "extract": ShapeExtractor,
}


class SiteModel:
    """The students of one site.

    Args:
        fields: One student per answer field.
    """

    def __init__(self, fields: Mapping[str, FieldModel]) -> None:
        self.fields = dict(fields)

    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        labels: Mapping[str, Sequence[str]],
        kinds: Mapping[str, str],
        *,
        features: Sequence[Features] | None = None,
        seed: int = 0,
    ) -> SiteModel | None:
        """Train one student per field. ``None`` if some field cannot be learned.

        Args:
            texts: The input part of each call.
            labels: Per field, the LLM's answer label for each call.
            kinds: Per field, its kind from the site profile.
            features: Precomputed features of ``texts``.
        """
        examples = list(features) if features is not None else [featurize(t) for t in texts]
        models: dict[str, FieldModel] = {}
        for name, field_labels in labels.items():
            kind = kinds[name]
            distinct = Counter(field_labels)
            if kind == "extract":
                extractor = ShapeExtractor.fit(texts, field_labels)
                if extractor is None:
                    return None
                models[name] = extractor
            elif kind in ("choice", "yes_no", "score", "constant"):
                if len(distinct) == 1:
                    label, seen = distinct.most_common(1)[0]
                    models[name] = ConstantModel(label, seen)
                else:
                    models[name] = LinearModel.fit(examples, field_labels, seed=seed)
            else:
                return None
        return cls(models)

    def predict(self, text: str, features: Features | None = None) -> tuple[dict[str, str], float]:
        """Each field's label, and the joint confidence."""
        example = features if features is not None else featurize(text)
        answer: dict[str, str] = {}
        confidence = 1.0
        for name, model in self.fields.items():
            label, probability = model.predict(example, text)
            answer[name] = label
            confidence *= probability
        return answer, confidence

    def save(self, path: str | Path) -> None:
        """Write ``<path>.json`` and ``<path>.npz``, atomically."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {}
        meta = {}
        for i, (name, model) in enumerate(self.fields.items()):
            prefix = f"f{i}"
            arrays.update(model.arrays(prefix))
            meta[name] = {"prefix": prefix, **model.meta()}
        npz = target.with_suffix(".npz")
        tmp_npz = npz.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_npz, **arrays)
        tmp_npz.replace(npz)
        meta_path = target.with_suffix(".json")
        tmp_meta = meta_path.with_suffix(".tmp")
        tmp_meta.write_text(json.dumps({"fields": meta}, ensure_ascii=False), encoding="utf-8")
        tmp_meta.replace(meta_path)

    @classmethod
    def load(cls, path: str | Path) -> SiteModel:
        target = Path(path)
        meta = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(target.with_suffix(".npz"), allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        fields: dict[str, FieldModel] = {}
        for name, spec in meta["fields"].items():
            fields[name] = _KINDS[spec["kind"]].restore(spec, arrays, spec["prefix"])
        return cls(fields)
