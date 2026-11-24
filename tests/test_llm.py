from types import SimpleNamespace

import pytest

from tools import llm


def test_requires_credentials_only_when_called(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="缺少 DASHSCOPE_API_KEY"):
        llm.generate("test")


def test_generate_uses_configured_provider_and_timeout(monkeypatch):
    observed = {}

    def create(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="reply"))])

    def client(**kwargs):
        assert kwargs["api_key"] == "test-only"
        assert kwargs["base_url"] == "https://example.invalid/v1"
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_S", "3")
    monkeypatch.setattr(llm, "OpenAI", client)
    assert llm.generate("question") == "reply"
    assert observed["model"] == "test-model"
    assert observed["timeout"] == 3
    assert observed["messages"][-1] == {"role": "user", "content": "question"}
