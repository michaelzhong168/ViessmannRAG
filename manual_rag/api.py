"""H5 开发联调使用的本地问答 HTTP 服务。"""

import argparse
import json
import logging
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import isfinite
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from manual_rag.answer import AnswerResult, ask
from manual_rag.index import DEFAULT_INDEX, load_index
from manual_rag.source import CONTROL_TYPES

_ASK_PATH = "/api/ask"
_MAX_BODY = 16 * 1024
_UPSTREAM_STATUS = {
    "model_timeout": 504,
    "model_service_error": 502,
    "invalid_model_response": 502,
}


class AskServer(ThreadingHTTPServer):
    """启动时加载一次索引，并限制单个开发实例同时进行的问答。"""

    def __init__(
        self, address: tuple[str, int], index_path: Path, *,
        origins: tuple[str, ...] = (), timeout: float = 30,
    ) -> None:
        self.index = load_index(index_path, timeout=timeout)
        self.origins = frozenset(origins)
        self.upstream_timeout = timeout
        self.ask_lock = Lock()
        super().__init__(address, AskHandler)


class AskHandler(BaseHTTPRequestHandler):
    """接收一个 JSON 问题，使用同一套结果字段返回问答或错误。"""

    server: AskServer
    timeout = 10  # 防止未发送完整请求的客户端长期占用连接。

    def _reply(self, status: int, result: AnswerResult | None = None) -> None:
        body = json.dumps(asdict(result), ensure_ascii=False).encode() if result else b""
        self.send_response(status)
        self.send_header("Vary", "Origin")
        self.send_header("Cache-Control", "no-store")
        origin = self.headers.get("Origin")
        if origin in self.server.origins:
            self.send_header("Access-Control-Allow-Origin", origin)
        if status == 204:
            self.send_header("Access-Control-Allow-Methods", "POST")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
        elif result is not None:
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
        if status == 405:
            self.send_header("Allow", "POST, OPTIONS")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _check_request(self) -> bool:
        if self.path != _ASK_PATH:
            self._reply(404, AnswerResult("接口不存在。", error_code="not_found"))
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.server.origins:
            self._reply(403, AnswerResult("此 H5 来源未获允许。", error_code="origin_not_allowed"))
            return False
        return True

    def do_OPTIONS(self) -> None:
        if self._check_request():
            self._reply(204)

    def do_GET(self) -> None:
        if self._check_request():
            self._reply(405, AnswerResult("请使用 POST 提问。", error_code="method_not_allowed"))

    def do_POST(self) -> None:
        if not self._check_request():
            return
        try:
            question, control_type = self._read_question()
        except TimeoutError:
            self._reply(408, AnswerResult("接收请求超时，请重试。", error_code="request_timeout"))
            return
        except ValueError as exc:
            self._reply(400, AnswerResult(str(exc), error_code="invalid_request"))
            return

        if not self.server.ask_lock.acquire(blocking=False):
            self._reply(503, AnswerResult("问答服务正在处理另一条问题，请稍后重试。", error_code="service_busy"))
            return
        try:
            result = ask(self.server.index, question, control_type, timeout=self.server.upstream_timeout)
        except Exception as exc:
            logging.error("处理问答时发生内部错误（%s）", type(exc).__name__)
            status = 500
            result = AnswerResult("问答服务暂时不可用。", error_code="internal_error")
        else:
            status = _UPSTREAM_STATUS.get(result.error_code, 200)
        finally:
            self.server.ask_lock.release()
        self._reply(status, result)

    def _read_question(self) -> tuple[str, str | None]:
        if self.headers.get_content_type() != "application/json":
            raise ValueError("请求需要使用 application/json。")
        lengths = self.headers.get_all("Content-Length") or []
        if (
            self.headers.get("Transfer-Encoding") is not None
            or len(lengths) != 1
            or not lengths[0].isascii()
            or not lengths[0].isdecimal()
            or len(lengths[0]) > len(str(_MAX_BODY))
        ):
            raise ValueError("请求需要有效的 Content-Length。")
        length = int(lengths[0])
        if not 0 < length <= _MAX_BODY:
            raise ValueError("请求体需要介于 1 字节与 16 KiB 之间。")
        raw = self.rfile.read(length)
        try:
            if len(raw) != length:
                raise ValueError
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, RecursionError):
            raise ValueError("请求需要有效的 UTF-8 JSON。") from None
        if not isinstance(body, dict):
            raise ValueError("请求体需要是 JSON 对象。")
        question = body.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question 需要是非空字符串。")
        try:
            question.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("question 需要包含有效的 Unicode 字符。") from None
        control_type = body.get("control_type")
        if control_type is not None and control_type not in CONTROL_TYPES:
            raise ValueError("control_type 需要是旋钮式、按键式或触控式。")
        return question, control_type


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="已发布的索引文件")
    parser.add_argument("--allow-origin", action="append", default=[], help="H5 完整来源，可重复指定")
    parser.add_argument("--upstream-timeout", type=float, default=30, help="上游网络操作超时秒数")
    args = parser.parse_args()
    if not isfinite(args.upstream_timeout) or args.upstream_timeout <= 0:
        parser.error("--upstream-timeout 需要是大于 0 的有限秒数")
    for origin in args.allow_origin:
        url = urlsplit(origin)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username is not None
            or origin != f"{url.scheme}://{url.netloc}"
        ):
            parser.error("--allow-origin 需要是 H5 的完整来源，如 http://localhost:5173")
    try:
        with AskServer(
            (args.host, args.port), args.index, origins=args.allow_origin, timeout=args.upstream_timeout,
        ) as server:
            print(f"H5 问答 API 已监听 {args.host}:{server.server_port}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"启动问答 API 失败。请检查监听地址、索引和环境配置（{type(exc).__name__}）。\n")


if __name__ == "__main__":
    main()
