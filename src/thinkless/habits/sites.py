"""Finding the decisions hidden in an agent's LLM calls.

A *site* is one place in the code that calls the LLM: the intent router, the
document grader, the reply writer. Calls from the same site share a model, an
output contract (the same JSON schema or forced tool) and a prompt template;
only the variable part of the prompt changes.

Sites are found in two steps. Calls are grouped by model, contract and the
shape of the conversation. Inside each group, lines that recur across many
calls are the template (instructions, few-shot examples, fixed labels) and
the other lines are the input. Calls with the same template lines form one
site. A caller can also name the site with the ``x-thinkless-site`` header,
which overrides the grouping.

Each field of a site's answers is then profiled. A field whose answers come
from a small closed set is a decision (``choice``, ``yes_no`` or ``score``);
a string field copied from the input is an ``extract``; anything else is
``open`` text that only a generative model can write.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..questions import Choice, Extract, Question, Score, YesNo
from .calls import Observation
from .wire import MISSING, ROOT_FIELD, TEXT_FIELD, TOOLS_FIELD, Call, fingerprint, value_of

__all__ = [
    "FieldProfile",
    "SiteProfile",
    "base_key",
    "call_lines",
    "discover",
    "profile_field",
    "variable_text",
]

MAX_OPTIONS = 200
"""A field with more distinct answers than this is treated as open text."""
MAX_UNSEEN = 0.05
"""Largest estimated chance that the next answer is one never seen before, for a closed field."""
MIN_EXTRACTIVE = 0.9
"""Share of a string field's answers that must appear in the input to count as an extract."""
TEMPLATE_SHARE = 0.3
"""A system or developer line is part of the template when this share of a group's calls has it."""
USER_TEMPLATE_SHARE = 0.8
"""The share for other lines. Higher, because inputs repeat too: a retrieved passage or a
dataset's boilerplate can recur in many calls without being part of the prompt template."""
INSTRUCTION_ROLES = ("system", "developer")
MIN_TEMPLATE_CALLS = 3
EXAMPLES = 3


def base_key(call: Call) -> str:
    """The group a call belongs to before template lines are considered."""
    if call.site_name:
        return fingerprint(["named", call.site_name, call.model])
    return fingerprint([call.model, call.contract_id, call.roles])


def call_lines(call: Call) -> list[tuple[str, str, str]]:
    """``(line id, role, text)`` for every non-empty line of the conversation."""
    lines = []
    for message in call.messages:
        for raw in message.text.splitlines():
            text = raw.strip()
            if len(text) < 2:
                continue
            lines.append((fingerprint([message.role, text], 16), message.role, text))
    return lines


def variable_text(call: Call, template: frozenset[str]) -> str:
    """The input part of a call: every line that is not part of its site's template."""
    parts = []
    for line_id, role, text in call_lines(call):
        if line_id in template:
            continue
        parts.append(text if role == "user" else f"{role}: {text}")
    return "\n".join(parts)


@dataclass
class FieldProfile:
    """What one field of a site's answers looks like.

    Attributes:
        name: The JSON key, ``$`` for a bare JSON value, ``$text`` for a text answer.
        kind: ``choice``, ``yes_no``, ``score``, ``extract``, ``constant`` or ``open``.
        counts: How often each answer label was seen.
        unseen: Estimated chance that the next answer is a label not seen
            yet (the Good-Turing estimate: labels seen once over answers).
        extractive: Share of string answers found verbatim in the input.
        description: The field's description from the JSON schema, if any.
        enum: Allowed labels from the JSON schema, if it lists them.
    """

    name: str
    kind: str
    counts: Counter[str]
    unseen: float
    extractive: float | None = None
    description: str | None = None
    enum: list[str] | None = None

    @property
    def closed(self) -> bool:
        return self.kind in ("choice", "yes_no", "score", "constant")

    @property
    def local(self) -> bool:
        """Whether a small local model can answer this field."""
        return self.closed or self.kind == "extract"

    @property
    def options(self) -> list[str]:
        """Every label the field can take, most common first."""
        labels = [label for label, _ in self.counts.most_common()]
        for label in self.enum or []:
            if label not in labels:
                labels.append(label)
        return labels

    def display(self, label: str) -> str:
        if label == MISSING:
            return "(missing)"
        value = value_of(label)
        return value if isinstance(value, str) else label

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "counts": dict(self.counts.most_common()),
            "unseen": round(self.unseen, 4),
            "extractive": None if self.extractive is None else round(self.extractive, 4),
            "description": self.description,
            "enum": self.enum,
        }


def _schema_property(schema: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    if not schema:
        return {}
    if name == ROOT_FIELD:
        return schema
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        found = properties.get(name)
        if isinstance(found, Mapping):
            return _resolve(found, schema)
    return {}


def _resolve(prop: Mapping[str, Any], schema: Mapping[str, Any]) -> Mapping[str, Any]:
    """Follow one local ``$ref`` and flatten ``anyOf`` with a null option."""
    ref = prop.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        node: Any = schema
        for part in ref[2:].split("/"):
            node = node.get(part) if isinstance(node, Mapping) else None
        if isinstance(node, Mapping):
            merged = dict(node)
            merged.update({k: v for k, v in prop.items() if k != "$ref"})
            return merged
    variants = prop.get("anyOf")
    if isinstance(variants, list):
        non_null = [v for v in variants if isinstance(v, Mapping) and v.get("type") != "null"]
        if len(non_null) == 1:
            merged = dict(_resolve(non_null[0], schema))
            merged.setdefault("description", prop.get("description"))
            if any(isinstance(v, Mapping) and v.get("type") == "null" for v in variants):
                merged["nullable"] = True
            return merged
    return prop


def _enum_labels(prop: Mapping[str, Any]) -> list[str] | None:
    if "const" in prop:
        return [json.dumps(prop["const"], ensure_ascii=False)]
    values = prop.get("enum")
    if isinstance(values, list) and values:
        labels = [json.dumps(v, ensure_ascii=False) for v in values]
        if prop.get("nullable") and "null" not in labels:
            labels.append("null")
        return labels
    if prop.get("type") == "boolean":
        return ["true", "false"]
    return None


def _is_int(label: str) -> bool:
    try:
        value = value_of(label)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(value, int) and not isinstance(value, bool)


def profile_field(
    name: str,
    labels: Sequence[str],
    inputs: Sequence[str],
    schema: Mapping[str, Any] | None = None,
) -> FieldProfile:
    """Classify one answer field from the labels a site's calls produced."""
    counts = Counter(labels)
    n = max(1, len(labels))
    singletons = sum(1 for c in counts.values() if c == 1)
    unseen = singletons / n
    prop = _schema_property(schema, name)
    description = prop.get("description") if isinstance(prop.get("description"), str) else None
    enum = _enum_labels(prop)
    present = {label for label in counts if label != MISSING}

    def kind_of_closed(options: Iterable[str]) -> str:
        options = [o for o in options if o != MISSING]
        if set(options) <= {"true", "false"}:
            return "yes_no"
        if options and all(_is_int(o) for o in options) and len(options) <= 11:
            return "score"
        return "choice"

    if name == TOOLS_FIELD:
        return FieldProfile(name, "open", counts, unseen, description=description)
    if enum is not None:
        kind = "constant" if len(set(enum) | present) == 1 else kind_of_closed(set(enum) | present)
        return FieldProfile(name, kind, counts, unseen, description=description, enum=enum)
    if len(present) <= 1 and len(labels) >= 2:
        kind = "constant" if len(counts) == 1 else kind_of_closed(counts)
        return FieldProfile(name, kind, counts, unseen, description=description)
    if len(counts) <= MAX_OPTIONS and unseen <= MAX_UNSEEN:
        return FieldProfile(name, kind_of_closed(counts), counts, unseen, description=description)
    strings = [
        (value_of(label), text)
        for label, text in zip(labels, inputs, strict=True)
        if label not in (MISSING, "null") and isinstance(value_of(label), str)
    ]
    if strings and len(strings) >= 0.5 * sum(1 for lab in labels if lab not in (MISSING, "null")):
        found = sum(1 for value, text in strings if value and value in text)
        rate = found / len(strings)
        if rate >= MIN_EXTRACTIVE:
            return FieldProfile(
                name, "extract", counts, unseen, extractive=rate, description=description
            )
        return FieldProfile(name, "open", counts, unseen, extractive=rate, description=description)
    return FieldProfile(name, "open", counts, unseen, description=description)


@dataclass
class SiteProfile:
    """One call site: its template, its traffic and what its answers look like.

    Attributes:
        id: Stable id, used in the plan, the logs and response headers.
        base: The group key (model, contract, conversation shape).
        template: Ids of the lines every call of the site shares.
        template_lines: The text of those lines, in order, for display.
        observations: The calls, oldest first.
        fields: One profile per answer field.
    """

    id: str
    base: str
    template: frozenset[str]
    template_lines: list[str]
    observations: list[Observation]
    fields: dict[str, FieldProfile] = field(default_factory=dict)

    @property
    def call(self) -> Call:
        return self.observations[-1].call

    @property
    def name(self) -> str:
        call = self.call
        if call.site_name:
            return call.site_name
        if call.schema_name:
            return call.schema_name
        for line in self.template_lines:
            return line if len(line) <= 60 else line[:57] + "..."
        return f"{call.contract} site"

    @property
    def calls(self) -> int:
        return len(self.observations)

    @property
    def answered(self) -> list[Observation]:
        """Calls the model answered, the ones ThinkLess can learn from."""
        return [o for o in self.observations if o.teacher]

    @property
    def verdict(self) -> str:
        """``decision``, ``mixed`` (a decision packed with text) or ``generation``."""
        if not self.fields:
            return "generation"
        local = [f.local for f in self.fields.values()]
        if all(local):
            return "decision"
        if any(f.closed and f.kind != "constant" for f in self.fields.values()):
            return "mixed"
        return "generation"

    def inputs(self) -> list[str]:
        return [variable_text(o.call, self.template) for o in self.answered]

    def labels(self, name: str) -> list[str]:
        return [o.output.fields.get(name, MISSING) for o in self.answered if o.output is not None]

    def response_template(self) -> dict[str, Any]:
        """How the model formats its answers on this site, for local answers."""
        orders: Counter[tuple[str, ...]] = Counter()
        finishes: Counter[str] = Counter()
        texts: dict[str, Counter[str]] = defaultdict(Counter)
        for observation in self.answered:
            output = observation.output
            assert output is not None
            if output.key_order:
                orders[output.key_order] += 1
            if output.finish_reason:
                finishes[output.finish_reason] += 1
            if TEXT_FIELD in output.fields and output.text is not None:
                texts[output.fields[TEXT_FIELD]][output.text.strip()] += 1
        return {
            "key_order": list(orders.most_common(1)[0][0]) if orders else [],
            "finish_reason": finishes.most_common(1)[0][0] if finishes else None,
            "texts": {label: c.most_common(1)[0][0] for label, c in texts.items()},
        }

    def questions(self) -> list[Question]:
        """The typed ThinkLess questions this site's decisions correspond to."""
        found: list[Question] = []
        extract: dict[str, str] = {}
        for profile in self.fields.values():
            if profile.kind == "extract":
                extract[profile.name] = profile.description or profile.name.replace("_", " ")
                continue
            if not profile.closed or profile.kind == "constant":
                continue
            instructions = profile.description or self._instructions(profile.name)
            key = profile.name if profile.name not in (TEXT_FIELD, ROOT_FIELD) else "answer"
            options = [o for o in profile.options if o != MISSING]
            if profile.kind == "yes_no":
                found.append(YesNo(instructions, name=key))
            elif profile.kind == "score":
                levels = sorted({str(value_of(o)) for o in options}, key=float)
                if len(levels) >= 2:
                    found.append(Score(instructions, levels, name=key))
            elif len(options) >= 2:
                found.append(Choice(instructions, [profile.display(o) for o in options], name=key))
        if extract:
            found.append(Extract(fields=extract, name="extract"))
        return found

    def _instructions(self, name: str) -> str:
        if name in (TEXT_FIELD, ROOT_FIELD):
            return self.template_lines[0] if self.template_lines else "What is the answer?"
        return name.replace("_", " ").capitalize() + "?"

    def cost(self) -> tuple[float, int]:
        """Reported or estimated dollars, and how many calls had no known cost."""
        total = 0.0
        unknown = 0
        for observation in self.observations:
            if observation.answered_by == "habit":
                continue
            if observation.cost_usd is None:
                unknown += 1
            else:
                total += observation.cost_usd
        return total, unknown

    def examples(self, count: int = EXAMPLES) -> list[dict[str, Any]]:
        picked = []
        for observation in self.answered[-count:]:
            assert observation.output is not None
            picked.append(
                {
                    "input": variable_text(observation.call, self.template)[:240],
                    "answer": {
                        name: self.fields[name].display(label) if name in self.fields else label
                        for name, label in observation.output.fields.items()
                    },
                }
            )
        return picked


def _template_ids(calls: Sequence[Call]) -> set[str]:
    seen: Counter[str] = Counter()
    roles: dict[str, str] = {}
    for call in calls:
        for line_id, role, _ in call_lines(call):
            roles.setdefault(line_id, role)
        seen.update({line_id for line_id, _, _ in call_lines(call)})
    instructions = max(MIN_TEMPLATE_CALLS, math.ceil(TEMPLATE_SHARE * len(calls)))
    others = max(MIN_TEMPLATE_CALLS, math.ceil(USER_TEMPLATE_SHARE * len(calls)))
    return {
        line_id
        for line_id, count in seen.items()
        if count >= (instructions if roles[line_id] in INSTRUCTION_ROLES else others)
    }


def discover(observations: Iterable[Observation]) -> list[SiteProfile]:
    """Group calls into sites and profile each site's answers. Largest site first."""
    groups: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        groups[base_key(observation.call)].append(observation)
    sites: list[SiteProfile] = []
    for base, members in groups.items():
        template_ids = _template_ids([o.call for o in members])
        clusters: dict[frozenset[str], list[Observation]] = defaultdict(list)
        for observation in members:
            own = frozenset(i for i, _, _ in call_lines(observation.call) if i in template_ids)
            clusters[own].append(observation)
        for template, cluster in clusters.items():
            first = cluster[-1].call
            lines = [text for line_id, _, text in call_lines(first) if line_id in template]
            site = SiteProfile(
                id=fingerprint([base, sorted(template)]),
                base=base,
                template=template,
                template_lines=lines,
                observations=cluster,
            )
            _profile(site)
            sites.append(site)
    sites.sort(key=lambda s: s.calls, reverse=True)
    return sites


def _profile(site: SiteProfile) -> None:
    answered = site.answered
    if not answered:
        return
    inputs = site.inputs()
    names: list[str] = []
    for observation in answered:
        assert observation.output is not None
        for name in observation.output.fields:
            if name not in names:
                names.append(name)
    schema = site.call.schema
    for name in names:
        site.fields[name] = profile_field(name, site.labels(name), inputs, schema)
