"""Answer generation — context assembly, prompting, streaming, citations."""

from kb.generation.citations import extract_citations
from kb.generation.context import AssembledContext, ContextAssembler
from kb.generation.generator import Generator
from kb.generation.prompt import PromptBuilder
from kb.generation.types import (
    Citation,
    CitationExtraction,
    GenerationConfig,
    GenerationResult,
    StreamEvent,
)

__all__ = [
    "Generator",
    "PromptBuilder",
    "ContextAssembler",
    "AssembledContext",
    "GenerationConfig",
    "GenerationResult",
    "StreamEvent",
    "Citation",
    "CitationExtraction",
    "extract_citations",
]
