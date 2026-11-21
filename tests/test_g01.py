"""G01 验收测试：环境约定 + 唯一模型入口（A2 九条断言，逐条对应）。

全部断言离线可跑：真实模型调用一律用 fake client 替身（只替换 ``_get_client``
这个外部边界，不 mock ``generate`` 本身）。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import llm  # noqa: E402  （需要先把仓库根塞进 sys.path）

#: 拼接构造，避免测试文件自身命中 A2#9 的字面量扫描
SECRET_PREFIX = "sk" + "-"

#: B2 表的 14 个键：7 模型 + 3 Langfuse + 4 部署
CONTRACT_ENV_KEYS = {
    "DASHSCOPE_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_ROUTER_MODEL",
    "EMBED_MODEL",
    "RERANK_MODEL",
    "LLM_TIMEOUT_S",
    "LANGFUSE_HOST",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "APP_ENV",
    "COS_BUCKET",
    "COS_SECRET_ID",
    "COS_SECRET_KEY",
}

#: 占位符形状：密钥前缀（可带 lf-）之后只有 x / . / _ / - / <...>，故不拼字面量
PLACEHOLDER_VALUE = re.compile(
    r"^" + re.escape(SECRET_PREFIX) + r"(?:lf-)?(?:x+|\.+|_+|-+|<[^>]*>)$"
)


# ────────────────────────── 夹具 ──────────────────────────


class _FakeCompletions:
    """记录 create 入参，返回固定文本（模拟 OpenAI 响应结构）。"""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content=self.content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)
        self.chat = types.SimpleNamespace(completions=self.completions)


def _mock_payload() -> dict:
    path = ROOT / "tests" / "fixtures" / "mock_llm.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _fake_client(monkeypatch) -> _FakeClient:
    client = _FakeClient(_mock_payload()["content"])
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    return client


def _parse_env_file(path: Path) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        pairs[key.strip()] = value.strip()
    return pairs


# ────────────────────────── A2#1 ──────────────────────────


def test_gitignore_covers_env_but_not_env_example():
    entries = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert ".env" in entries, ".gitignore 缺少 .env"

    # .env.example 必须可入库：git check-ignore 返回 1 表示「未被忽略」
    proc = subprocess.run(
        ["git", "check-ignore", "-q", ".env.example"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, f".env.example 被 .gitignore 命中（returncode={proc.returncode}）"


# ────────────────────────── A2#2 ──────────────────────────


def test_env_example_declares_all_contract_keys():
    values = _parse_env_file(ROOT / ".env.example")
    missing = CONTRACT_ENV_KEYS - set(values)
    assert not missing, f".env.example 缺少契约键：{sorted(missing)}"
    assert len(values) == len(CONTRACT_ENV_KEYS), f".env.example 键数量应为 14，实际 {len(values)}"


# ────────────────────────── A2#3 ──────────────────────────


def test_env_example_values_are_placeholders():
    values = _parse_env_file(ROOT / ".env.example")
    for key, value in values.items():
        if SECRET_PREFIX not in value:
            continue
        assert PLACEHOLDER_VALUE.match(value), f"{key} 的值不是占位符：{value!r}"


# ────────────────────────── A2#4 ──────────────────────────


def test_generate_returns_mocked_content(monkeypatch):
    payload = _mock_payload()
    client = _fake_client(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "qwen3.7-plus")

    assert llm.generate(payload["prompt"]) == payload["content"]

    messages = client.completions.calls[-1]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == llm.SYSTEM_PROMPT
    assert messages[-1] == {"role": "user", "content": payload["prompt"]}


# ────────────────────────── A2#5 ──────────────────────────


def test_generate_passes_contract_params(monkeypatch):
    client = _fake_client(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "qwen3.7-plus")
    monkeypatch.delenv("LLM_TIMEOUT_S", raising=False)

    llm.generate("你好")

    kwargs = client.completions.calls[-1]
    assert kwargs["model"] == "qwen3.7-plus"
    assert kwargs["temperature"] == 0.1
    assert kwargs["max_tokens"] == 400
    assert kwargs["timeout"] == 10.0


# ────────────────────────── A2#6 ──────────────────────────


def test_model_follows_env_variable(monkeypatch):
    client = _fake_client(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "qwen3.7-plus")
    llm.generate("你好")
    assert client.completions.calls[-1]["model"] == "qwen3.7-plus"

    # 只改一行环境变量
    monkeypatch.setenv("LLM_MODEL", "qwen3.7-flash")
    llm.generate("你好")
    assert client.completions.calls[-1]["model"] == "qwen3.7-flash"


# ────────────────────────── A2#7 ──────────────────────────


def test_explicit_model_overrides_env(monkeypatch):
    client = _fake_client(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "qwen3.7-plus")

    llm.generate("你好", model="qwen3.7-max")

    assert client.completions.calls[-1]["model"] == "qwen3.7-max"


# ────────────────────────── A2#8 ──────────────────────────


def test_cli_without_api_key_exits_nonzero_with_hint(tmp_path):
    env = dict(os.environ)
    # 空串即可阻止 python-dotenv(override=False) 用仓库 .env 里的真实 KEY 覆盖
    env["DASHSCOPE_API_KEY"] = ""
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    proc = subprocess.run(
        [sys.executable, "-m", "tools.llm", "你好"],
        cwd=tmp_path,  # 避开 CWD 下的 .env 发现
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, "缺密钥必须以非零码退出"
    assert "请检查 DASHSCOPE_API_KEY" in output, f"缺提示文案，实际输出：{output!r}"
    assert "Traceback" not in output, f"不得抛未捕获 traceback，实际输出：{output!r}"


# ────────────────────────── A2#9 ──────────────────────────


def test_no_secret_literal_in_repo():
    skip_dirs = {".git", ".venv", "docs", "node_modules", ".pytest_cache", "__pycache__"}
    skip_suffixes = {".md", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff", ".woff2"}

    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in skip_dirs:
            continue
        if path.name.startswith(".env"):  # .env / .env.example / .env.local
            continue
        if path.suffix.lower() in skip_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if SECRET_PREFIX in text:
            offenders.append(str(relative))

    assert not offenders, f"仓库内出现密钥字面量：{offenders}"
