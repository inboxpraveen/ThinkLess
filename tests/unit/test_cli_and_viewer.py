from __future__ import annotations

import json

from typer.testing import CliRunner

from tests.conftest import FakeProvider, choice_answer
from thinkless import Choice, Engine, JSONLSink, MemorySink, Tracer, YesNo
from thinkless.cli.main import app
from thinkless.tracing.viewer import build_viewer, collect_traces

runner = CliRunner()


def _write_trace(directory) -> None:
    provider = FakeProvider(
        "fast", {"intent": choice_answer("refund", {"refund": 0.99, "other": 0.01})}
    )
    engine = Engine([provider], tracer=Tracer([JSONLSink(directory)]))
    with engine.run("ticket", mode="hybrid", ticket_id="T-1"), engine.step("triage"):
        engine.decide(
            "I want my money back </script>",
            Choice("Intent?", options=["refund", "other"], name="intent"),
        )


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "thinkless" in result.stdout


def test_doctor_runs() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "TYPESAFE_API_KEY" in result.stdout


def test_trace_commands(tmp_path) -> None:
    _write_trace(tmp_path)
    listed = runner.invoke(app, ["trace", "ls", str(tmp_path)])
    assert listed.exit_code == 0
    assert "ticket" in listed.stdout

    file = next(tmp_path.glob("*.jsonl"))
    shown = runner.invoke(app, ["trace", "show", str(file)])
    assert shown.exit_code == 0
    assert "intent" in shown.stdout

    out = tmp_path / "viewer.html"
    viewed = runner.invoke(app, ["trace", "view", str(tmp_path), "--out", str(out), "--no-open"])
    assert viewed.exit_code == 0
    assert out.exists()


def test_viewer_is_self_contained_and_escapes_script_tags(tmp_path) -> None:
    _write_trace(tmp_path)
    html = build_viewer(collect_traces([tmp_path]), title="Test")
    assert html.count("</script>") == 2
    assert "<\\/script>" in html
    assert "<script src" not in html
    assert "<link" not in html
    start = html.index('type="application/json">') + len('type="application/json">')
    payload = json.loads(html[start : html.index("</script>", start)].replace("<\\/", "</"))
    assert payload["traces"][0]["summary"]["decisions"] == 1


def test_calibrate_command(tmp_path, monkeypatch) -> None:
    data = tmp_path / "labeled.jsonl"
    rows = [{"text": f"row {i}", "label": i % 4 != 0} for i in range(20)]
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    def answer(state, question):
        from thinkless import Answer

        index = int(state.split()[1])
        truth = index % 4 != 0
        p_yes = 0.97 if truth else 0.6  # confident when right, unsure when wrong
        return Answer(value=p_yes >= 0.5, probabilities={"yes": p_yes, "no": 1 - p_yes})

    monkeypatch.setattr(
        "thinkless.bench.intents.build_provider",
        lambda name, **kwargs: FakeProvider("fake", {"label": answer}),
    )
    result = runner.invoke(
        app,
        [
            "calibrate",
            str(data),
            "--kind",
            "yes_no",
            "--question",
            "Is it good?",
            "--target",
            "0.99",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Recommended threshold" in result.stdout


def test_otel_sink_exports_spans() -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from thinkless.tracing.otel import OTelSink

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    engine = Engine(
        [FakeProvider("fast", {"urgent": None})],
        tracer=Tracer([OTelSink(provider), MemorySink()]),
    )
    with engine.run("ticket", input={"message": "private"}):
        engine.decide("private text", YesNo("Urgent?", name="urgent"))
    spans = exporter.get_finished_spans()
    names = sorted(s.name for s in spans)
    assert names == ["attempt fast", "decide urgent", "run ticket"]
    run = next(s for s in spans if s.name == "run ticket")
    attempt = next(s for s in spans if s.name == "attempt fast")
    assert (
        attempt.parent.span_id
        == next(s for s in spans if s.name == "decide urgent").context.span_id
    )
    assert "thinkless.input" not in run.attributes
    assert attempt.attributes["gen_ai.provider.name"] == "fast"
