"""Live smoke tests against the real OpenAI API. Run with: uv run pytest -m live"""

import os

import pytest

from rootcause_agents.config import Settings

pytestmark = pytest.mark.live


@pytest.fixture
def live_settings():
    s = Settings.from_env()
    if not s.openai_api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return s


@pytest.mark.parametrize("system", ["mas", "single"])
async def test_live_investigation_runs(live_settings, tmp_path, system):
    from rootcause_agents.runner import investigate

    report = await investigate("S01", system, settings=live_settings, out_dir=tmp_path)
    assert report.ranked_hypotheses
    assert report.stats.input_tokens > 0
    assert os.path.exists(tmp_path / "report.md")
