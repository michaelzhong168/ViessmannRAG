import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local_inputs_and_credentials_are_ignored():
    protected = subprocess.check_output(
        ["git", "check-ignore", ".env", "data/raw/manual.pdf", "data/semantic/index.json"],
        cwd=ROOT, text=True,
    )
    assert set(protected.splitlines()) == {".env", "data/raw/manual.pdf", "data/semantic/index.json"}
    template = dict(line.split("=", 1) for line in (ROOT / ".env.example").read_text().splitlines()
                    if line.strip() and not line.startswith("#"))
    assert set(template) == {"DASHSCOPE_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_TIMEOUT_S"}
    assert template["DASHSCOPE_API_KEY"] == ""


def test_tracked_text_has_no_key_literal():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, text=True)
    pattern = re.compile("s" + r"k-(?:lf-)?[a-zA-Z0-9]{16,}")
    offenders = []
    for name in filter(None, tracked.split("\0")):
        try:
            contents = (ROOT / name).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if pattern.search(contents):
            offenders.append(name)
    assert offenders == []
