import pytest
from pydantic import ValidationError

from rootcause_agents import schemas as s


def _walk(node, path="$"):
    """Yield (path, node) for every sub-schema of a JSON schema."""
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


@pytest.mark.parametrize("model", s.LLM_OUTPUT_MODELS)
def test_llm_output_schemas_have_no_free_form_objects(model):
    """OpenAI structured outputs reject free-form dicts; every object must declare properties."""
    schema = model.model_json_schema()
    for path, node in _walk(schema):
        if node.get("type") == "object":
            assert "properties" in node, f"{model.__name__}: free-form object at {path}"
            assert node.get("additionalProperties") in (None, False), f"{model.__name__}: {path}"


def test_hypothesis_confidence_is_clamped():
    h = s.Hypothesis(
        id="H1", title="x", root_cause_component="a", root_cause_category="unknown",
        mechanism=[], supporting_evidence_ids=[], contradicting_evidence_ids=[],
        unexplained_symptoms=[], confidence=1.5,
    )
    assert h.confidence == 1.0


@pytest.mark.parametrize("model", s.LLM_OUTPUT_MODELS)
def test_openai_strict_schema_has_no_unsupported_keywords(model):
    """OpenAI strict structured outputs: avoid 'default' and numeric range keywords."""
    from openai.lib._pydantic import to_strict_json_schema

    for path, node in _walk(to_strict_json_schema(model)):
        for kw in ("default", "minimum", "maximum"):
            assert kw not in node, f"{model.__name__}: '{kw}' at {path}"


def test_root_cause_category_is_constrained():
    with pytest.raises(ValidationError):
        s.Hypothesis(
            id="H1", title="x", root_cause_component="a", root_cause_category="aliens",
            mechanism=[], supporting_evidence_ids=[], contradicting_evidence_ids=[],
            unexplained_symptoms=[], confidence=0.5,
        )


def test_tool_result_error_roundtrip():
    r = s.ToolResult(
        tool="query_logs", status="error", data=None, completeness=0.0,
        error=s.ToolError(code="TIMEOUT", message="timed out", retryable=True),
    )
    assert r.is_error
    assert s.ToolResult.model_validate_json(r.model_dump_json()).error.code == "TIMEOUT"


def test_critic_decision_constrained():
    with pytest.raises(ValidationError):
        s.CriticVerdict(decision="MAYBE", issues=[], evidence_requests=[])
