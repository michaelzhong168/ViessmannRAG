"""跨平台页图及 trace 存储契约。"""

from collections.abc import Iterator
from typing import Protocol


class Storage(Protocol):
    def get_page_url(self, page_no: int) -> str: ...

    def append_trace(self, record: dict) -> None: ...

    def iter_trace(self) -> Iterator[dict]: ...
