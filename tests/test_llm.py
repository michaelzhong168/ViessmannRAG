# 用 SDK 替身覆盖模型客户端的配置和错误处理。
from types import SimpleNamespace

import pytest

from tools import llm


@pytest.fixture
def provider(monkeypatch):
    request, client_config = {}, {}

    def create(**kwargs):
        request.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="reply"))])

    def client(**kwargs):
        client_config.update(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    # 禁止从本机 .env 读取密钥，只在 SDK 边界替换网络调用。
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(llm, "OpenAI", client)
    return request, client_config


def test_model_probe_without_credentials(monkeypatch, capsys):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    # 退出码区分凭证缺失和命令用法错误。
    assert llm.main(["question"]) == 1
    assert "请检查 DASHSCOPE_API_KEY" in capsys.readouterr().err
    assert llm.main([]) == 2


def test_generate_uses_configured_provider_and_timeout(monkeypatch, provider):
    request, client_config = provider
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_TIMEOUT_S", "3")
    assert llm.generate("question") == "reply"
    assert client_config == {"api_key": "test-only", "base_url": "https://example.invalid/v1"}
    assert request["model"] == "test-model"
    assert request["timeout"] == 3
    assert request["messages"] == [
        {"role": "system", "content": llm.SYSTEM_PROMPT}, {"role": "user", "content": "question"},
    ]


def test_defaults_and_explicit_model_priority(monkeypatch, provider):
    request, client_config = provider
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT_S", raising=False)
    # 保留夹具的环境模型，验证调用方传入的型号优先。
    assert llm.generate("question", model="explicit-model") == "reply"
    assert client_config["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert request["model"] == "explicit-model"
    assert request["timeout"] == 10
    assert request["temperature"] == 0.1 and request["max_tokens"] == 400


def test_no_implicit_model(monkeypatch, provider):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        llm.generate("question")
