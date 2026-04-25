"""
Shared service instances for the web tier (one process = one hot pool).
"""

from __future__ import annotations

from typing import Optional

from kb.generation import Generator
from kb.retrieval import Retriever
from kb.settings import Settings, get_settings

_gen: Optional[Generator] = None
_ret: Optional[Retriever] = None


def get_settings_cached() -> Settings:
    return get_settings()


def get_retriever() -> Retriever:
    global _ret
    if _ret is None:
        _ret = Retriever(settings=get_settings_cached())
    return _ret


def get_generator() -> Generator:
    global _gen
    if _gen is None:
        _gen = Generator(
            settings=get_settings_cached(),
            retriever=get_retriever(),
        )
    return _gen
