"""ViessmannRAG 离线构建流水线包。

模块用合法标识符命名；下游直接 ``from pipeline.parse import Document``，
包本身不做 re-export（避免 ``python -m pipeline.parse`` 的重复导入警告）。
"""
