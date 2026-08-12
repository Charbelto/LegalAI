"""Tests for the judge provider layer, cost accounting and budget guard.

No network calls and no spend: the HTTP layer is stubbed.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def _openai_payload(content='{"accuracy": 4, "completeness": 3, "groundedness": 5, "rationale": "ok"}',
                    prompt_tokens=1000, completion_tokens=100):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


@pytest.fixture()
def judge_module(monkeypatch, tmp_path):
    """Load llm_judge with an isolated cache/spend file and openai provider."""
    monkeypatch.setenv("JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("JUDGE_MODEL", "test-model-1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("JUDGE_BUDGET_USD", "1.00")
    monkeypatch.setenv("JUDGE_PRICE_IN_PER_M", "1.00")
    monkeypatch.setenv("JUDGE_PRICE_OUT_PER_M", "10.00")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    sys.modules.pop("llm_judge", None)
    module = importlib.import_module("llm_judge")

    # Redirect cache and spend to the temp dir so real files are untouched.
    module.CACHE_FILE = tmp_path / "judge_cache.json"
    module.SPEND_FILE = tmp_path / "judge_spend.json"
    module._cache = {}
    yield module
    sys.modules.pop("llm_judge", None)


def test_openai_scores_are_parsed_and_spend_recorded(judge_module, monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "payload": json})
        return FakeResponse(200, _openai_payload())

    monkeypatch.setattr("requests.post", fake_post)

    result = judge_module.judge("q", "gold text", "answer text")

    assert result["ok"] is True
    assert (result["accuracy"], result["completeness"], result["groundedness"]) == (4, 3, 5)
    assert result["judge_provider"] == "openai"
    assert calls[0]["payload"]["model"] == "test-model-1"

    # $1/M in, $10/M out => 1000 in + 100 out = 0.001 + 0.001 = $0.002
    spend = judge_module.load_spend()
    assert spend["calls"] == 1
    assert spend["total_usd"] == pytest.approx(0.002, abs=1e-9)


def test_cached_scores_do_not_spend_again(judge_module, monkeypatch):
    post_count = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        post_count["n"] += 1
        return FakeResponse(200, _openai_payload())

    monkeypatch.setattr("requests.post", fake_post)

    judge_module.judge("q", "gold", "answer")
    judge_module.judge("q", "gold", "answer")   # identical content
    judge_module.judge("q", "gold", "answer")

    assert post_count["n"] == 1, "repeat analysis must not pay for the same triple twice"
    assert judge_module.load_spend()["calls"] == 1


def test_budget_cap_stops_further_spending(judge_module, monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda url, json=None, headers=None, timeout=None: FakeResponse(200, _openai_payload()),
    )

    # Pre-load the spend log above the $1.00 budget.
    judge_module.SPEND_FILE.write_text(json.dumps({"total_usd": 1.5, "calls": 10}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="budget exhausted"):
        judge_module.judge("q", "gold", "a different answer")


def test_missing_api_key_is_a_clear_error(judge_module, monkeypatch):
    monkeypatch.setattr(judge_module, "OPENAI_API_KEY", "")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        judge_module.judge("q", "gold", "answer")


def test_unknown_model_reports_how_to_list_models(judge_module, monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda url, json=None, headers=None, timeout=None: FakeResponse(
            404, {}, text='{"error": {"message": "The model does not exist"}}'
        ),
    )

    with pytest.raises(RuntimeError, match="v1/models"):
        judge_module.judge("q", "gold", "answer")


def test_temperature_rejection_is_retried_without_it(judge_module, monkeypatch):
    attempts = []

    def fake_post(url, json=None, headers=None, timeout=None):
        attempts.append(dict(json))
        if "temperature" in json:
            return FakeResponse(
                400, {}, text='{"error": {"message": "Unsupported value: temperature"}}'
            )
        return FakeResponse(200, _openai_payload())

    monkeypatch.setattr("requests.post", fake_post)

    result = judge_module.judge("q", "gold", "answer")

    assert result["ok"] is True
    assert len(attempts) == 2
    assert "temperature" in attempts[0] and "temperature" not in attempts[1]


def test_rate_limit_is_retried_then_succeeds(judge_module, monkeypatch):
    state = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        state["n"] += 1
        if state["n"] == 1:
            return FakeResponse(429, {}, text="rate limited")
        return FakeResponse(200, _openai_payload())

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    result = judge_module.judge("q", "gold", "answer")

    assert result["ok"] is True
    assert state["n"] == 2


def test_unparseable_judge_output_is_flagged_not_scored_as_one(judge_module, monkeypatch):
    monkeypatch.setattr(
        "requests.post",
        lambda url, json=None, headers=None, timeout=None: FakeResponse(
            200, _openai_payload(content="I think this answer is pretty good, honestly.")
        ),
    )

    result = judge_module.judge("q", "gold", "answer")

    assert result["ok"] is False
    assert result["accuracy"] is None, "a failed judge call must not become a real score of 1"
    # And it must not be cached, so a later run can retry it.
    assert judge_module._cache == {}


def test_json_wrapped_in_prose_is_recovered(judge_module, monkeypatch):
    wrapped = 'Sure! Here you go:\n{"accuracy": 5, "completeness": 4, "groundedness": 4, "rationale": "good"}'
    monkeypatch.setattr(
        "requests.post",
        lambda url, json=None, headers=None, timeout=None: FakeResponse(
            200, _openai_payload(content=wrapped)
        ),
    )

    result = judge_module.judge("q", "gold", "answer")

    assert result["ok"] is True
    assert result["accuracy"] == 5


def test_deepseek_provider_scores_and_hits_deepseek_base_url(monkeypatch, tmp_path):
    """The new default provider: same call path as openai, different key/url."""
    monkeypatch.setenv("JUDGE_PROVIDER", "deepseek")
    monkeypatch.setenv("JUDGE_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test-not-real")
    monkeypatch.setenv("JUDGE_BUDGET_USD", "1.00")
    monkeypatch.setenv("JUDGE_PRICE_IN_PER_M", "0.14")
    monkeypatch.setenv("JUDGE_PRICE_OUT_PER_M", "0.28")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    sys.modules.pop("llm_judge", None)
    module = importlib.import_module("llm_judge")
    module.CACHE_FILE = tmp_path / "judge_cache.json"
    module.SPEND_FILE = tmp_path / "judge_spend.json"
    module._cache = {}

    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "auth": headers.get("Authorization")})
        return FakeResponse(200, _openai_payload())

    monkeypatch.setattr("requests.post", fake_post)

    try:
        result = module.judge("q", "gold text", "answer text")

        assert result["ok"] is True
        assert result["judge_provider"] == "deepseek"
        assert calls[0]["url"] == "https://api.deepseek.com/v1/chat/completions"
        assert calls[0]["auth"] == "Bearer sk-deepseek-test-not-real"
    finally:
        sys.modules.pop("llm_judge", None)


def test_cache_key_separates_providers_and_models(judge_module):
    key_a = judge_module.get_cache_key("q", "g", "a")
    judge_module.JUDGE_MODEL = "test-model-2"
    key_b = judge_module.get_cache_key("q", "g", "a")

    assert key_a != key_b, "scores from different judges must not collide in the cache"


def test_env_loader_does_not_override_real_environment(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT))
    import env_loader

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# a comment",
                # A key that cannot already be sitting in the real process
                # environment (unlike JUDGE_MODEL, which legalai/.env now sets
                # for real - importing env_loader/llm_judge earlier in the test
                # session would otherwise have already loaded it once,
                # making this assertion order-dependent).
                "TEST_ENV_LOADER_NEW_KEY=from-file",
                'QUOTED="quoted value"',
                "WITH_COMMENT=value # trailing",
                "ALREADY_SET=from-file",
                "malformed line",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("ALREADY_SET", "from-environment")
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.delenv("WITH_COMMENT", raising=False)
    monkeypatch.delenv("TEST_ENV_LOADER_NEW_KEY", raising=False)

    applied = env_loader.load_env(env_file)

    import os

    assert applied["TEST_ENV_LOADER_NEW_KEY"] == "from-file"
    assert os.environ["QUOTED"] == "quoted value"
    assert os.environ["WITH_COMMENT"] == "value"
    assert os.environ["ALREADY_SET"] == "from-environment", "shell must win over .env"
