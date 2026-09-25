"""Typed questions: the bounded answer spaces a decision can take.

A question describes what the application wants to know and which answers are
allowed. Providers never see free-form prompts, only these declarations, so the
same question can be answered by a rule, a small decision model or an LLM
without changing application code.

The four kinds mirror the System One wire format used by Jev, Kev, OpenJev and
Laya (``choice``, ``score``, ``noul``), plus ``extract`` for pulling typed
fields out of text, which GLiNER-style models handle well.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = ["Choice", "Extract", "Kind", "Question", "Score", "YesNo", "question_from_spec"]


class Kind(str, Enum):
    """The answer space of a question."""

    CHOICE = "choice"
    SCORE = "score"
    YES_NO = "yes_no"
    EXTRACT = "extract"


_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(text: str, limit: int = 48) -> str:
    slug = _SLUG.sub("_", text.lower()).strip("_")
    return slug[:limit].rstrip("_") or "question"


class Question(BaseModel):
    """Base class for all question kinds.

    Attributes:
        instructions: What is being asked, in plain language.
        name: Stable identifier used in traces, thresholds and rules. Derived
            from the instructions when omitted.
        threshold: Minimum normalized confidence needed to accept an answer.
            Overrides the engine default for this question.
        providers: Optional allowlist of provider names that may answer. Use it
            to keep consequential questions away from weaker backends.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ClassVar[Kind]

    instructions: str = Field(min_length=1)
    name: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    providers: tuple[str, ...] | None = None

    def __init__(self, instructions: str | None = None, /, **data: Any) -> None:
        if instructions is not None:
            data["instructions"] = instructions
        super().__init__(**data)

    @property
    def key(self) -> str:
        """The name used to identify this question."""
        return self.name or _slug(self.instructions)

    def allows(self, provider: str) -> bool:
        return self.providers is None or provider in self.providers

    def spec(self) -> dict[str, Any]:
        """A JSON-serializable description, used in traces."""
        data = self.model_dump(exclude_none=True)
        data["kind"] = self.kind.value
        return data


class Choice(Question):
    """Pick exactly one option.

    Options are given as a mapping of label to description, or as a plain list
    of labels. Descriptions matter: every provider uses them to tell options
    apart, so write them the way you would brief a new teammate.

    Example:
        >>> Choice(
        ...     "What does the customer want?",
        ...     options={"refund": "wants money back", "status": "asks where an order is"},
        ... )
    """

    kind: ClassVar[Kind] = Kind.CHOICE

    options: dict[str, str | None]

    def __init__(
        self,
        instructions: str,
        options: Mapping[str, str | None] | Sequence[str],
        *,
        name: str | None = None,
        threshold: float | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            instructions, options=options, name=name, threshold=threshold, providers=providers
        )

    @field_validator("options", mode="before")
    @classmethod
    def _coerce_options(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return {str(label): None for label in value}
        return value

    @field_validator("options")
    @classmethod
    def _check_options(cls, value: dict[str, str | None]) -> dict[str, str | None]:
        if len(value) < 2:
            raise ValueError("a Choice needs at least two options")
        return value

    @property
    def labels(self) -> list[str]:
        return list(self.options)


class Score(Question):
    """Place the input on an ordered scale.

    Levels are ordered from lowest to highest. The decision value is the
    expected level index (a float between 0 and ``len(levels) - 1``) and the
    most likely level is available as ``Decision.level``.
    """

    kind: ClassVar[Kind] = Kind.SCORE

    levels: tuple[str, ...]

    def __init__(
        self,
        instructions: str,
        levels: Sequence[str],
        *,
        name: str | None = None,
        threshold: float | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            instructions, levels=levels, name=name, threshold=threshold, providers=providers
        )

    @field_validator("levels")
    @classmethod
    def _check_levels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) < 2:
            raise ValueError("a Score needs at least two levels")
        return value


class YesNo(Question):
    """A binary question. The decision value is a bool.

    ``yes_means`` and ``no_means`` optionally spell out what each answer covers,
    which helps with questions whose boundary is subtle.
    """

    kind: ClassVar[Kind] = Kind.YES_NO

    yes_means: str | None = None
    no_means: str | None = None

    def __init__(
        self,
        instructions: str,
        *,
        yes_means: str | None = None,
        no_means: str | None = None,
        name: str | None = None,
        threshold: float | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            instructions,
            yes_means=yes_means,
            no_means=no_means,
            name=name,
            threshold=threshold,
            providers=providers,
        )


class Extract(Question):
    """Pull typed fields out of the input.

    ``fields`` maps a field name to a description. Fields listed in
    ``required`` must be found for the answer to count as confident; a missing
    optional field is simply ``None``.

    Example:
        >>> Extract(fields={"order_id": "the order number, digits only"})
    """

    kind: ClassVar[Kind] = Kind.EXTRACT

    instructions: str = "Extract the listed fields from the input."
    fields: dict[str, str]
    required: tuple[str, ...] = ()

    def __init__(
        self,
        instructions: str | None = None,
        *,
        fields: Mapping[str, str],
        required: Sequence[str] = (),
        name: str | None = None,
        threshold: float | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            instructions,
            fields=fields,
            required=required,
            name=name,
            threshold=threshold,
            providers=providers,
        )

    @model_validator(mode="after")
    def _check_required(self) -> Extract:
        if not self.fields:
            raise ValueError("an Extract needs at least one field")
        unknown = set(self.required) - set(self.fields)
        if unknown:
            raise ValueError(f"required fields not declared in fields: {sorted(unknown)}")
        return self

    @property
    def key(self) -> str:
        return self.name or "extract_" + "_".join(self.fields)[:40]


_KINDS: dict[str, type[Question]] = {
    Kind.CHOICE.value: Choice,
    Kind.SCORE.value: Score,
    Kind.YES_NO.value: YesNo,
    Kind.EXTRACT.value: Extract,
}


def question_from_spec(spec: Mapping[str, Any], *, name: str | None = None) -> Question:
    """Rebuild a question from :meth:`Question.spec` output, or from JSON sent by a client.

    Args:
        spec: A mapping with ``kind`` (``choice``, ``score``, ``yes_no`` or
            ``extract``) and that kind's fields.
        name: Overrides ``spec["name"]``.

    Raises:
        ValueError: If the kind is unknown or the fields do not validate.
    """
    data = dict(spec)
    kind = data.pop("kind", None)
    cls = _KINDS.get(str(kind))
    if cls is None:
        raise ValueError(f"unknown question kind {kind!r}; use one of {', '.join(_KINDS)}")
    if name is not None:
        data["name"] = name
    return cls.model_validate(data)
