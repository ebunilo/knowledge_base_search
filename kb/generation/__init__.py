"""Answer generation — context assembly, prompting, streaming, citations,
post-stream NLI faithfulness verification, and aggregate confidence."""

from kb.generation.citations import extract_citations
from kb.generation.confidence import compute_confidence
from kb.generation.context import AssembledContext, ContextAssembler
from kb.generation.faithfulness import FaithfulnessChecker
from kb.generation.generator import Generator
from kb.generation.nli_client import NLIClient, NLIClientError
from kb.generation.prompt import PromptBuilder
from kb.generation.segmentation import Sentence, split_sentences
from kb.generation.types import (
    Citation,
    CitationExtraction,
    FaithfulnessReport,
    GenerationConfig,
    GenerationResult,
    SentenceCheck,
    SentenceStatus,
    StreamEvent,
)

__all__ = [
    "Generator",
    "PromptBuilder",
    "ContextAssembler",
    "AssembledContext",
    "FaithfulnessChecker",
    "NLIClient",
    "NLIClientError",
    "compute_confidence",
    "split_sentences",
    "Sentence",
    "GenerationConfig",
    "GenerationResult",
    "StreamEvent",
    "Citation",
    "CitationExtraction",
    "FaithfulnessReport",
    "SentenceCheck",
    "SentenceStatus",
    "extract_citations",
]
