"""
Structural chunker — groups ParsedBlocks into section-level "parent" chunks.

Strategy:
    * Walk blocks in order, maintaining a section stack keyed by heading level.
    * A new parent chunk starts at each top-of-level heading once the running
      parent has reached `parent_target_tokens`. If a single section is larger
      than 2× the target, we split it at paragraph boundaries so parents stay
      roughly uniform in size (important for the LLM context budget).
    * For structured formats (YAML/JSON), we use heading level on the key path
      the same way — deep keys group into the same parent until the budget is
      hit.

Returns: list[dict] with {text, section_path, token_count, block_refs}.
The caller (parent_child.py) converts these to ParentChunk objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kb.chunking.tokens import count_tokens
from kb.types import ParsedBlock


@dataclass
class _ParentDraft:
    section_path: list[str] = field(default_factory=list)
    blocks: list[ParsedBlock] = field(default_factory=list)
    token_count: int = 0

    def add(self, block: ParsedBlock, tokens: int) -> None:
        self.blocks.append(block)
        self.token_count += tokens

    def render(self) -> str:
        parts: list[str] = []
        for b in self.blocks:
            if b.kind == "heading":
                prefix = "#" * (b.level or 1)
                parts.append(f"{prefix} {b.text}")
            elif b.kind == "code":
                lang = (b.meta or {}).get("language", "")
                parts.append(f"```{lang}\n{b.text}\n```")
            else:
                parts.append(b.text)
        return "\n\n".join(parts).strip()


def build_parents(
    blocks: list[ParsedBlock],
    *,
    parent_target_tokens: int,
    hard_cap_tokens: int | None = None,
) -> list[dict]:
    """
    Group consecutive ParsedBlocks into parent-sized sections.

    Args:
        blocks: ordered list of parsed blocks.
        parent_target_tokens: soft target; we cut on heading boundaries once
            this is reached.
        hard_cap_tokens: hard ceiling; we must cut regardless of structure.
            Defaults to 2× the target.

    Returns:
        list of dicts with keys: text, section_path, token_count.
    """
    if hard_cap_tokens is None:
        hard_cap_tokens = parent_target_tokens * 2

    parents: list[_ParentDraft] = []
    current = _ParentDraft()

    def commit():
        nonlocal current
        if current.blocks and current.render().strip():
            parents.append(current)
        current = _ParentDraft()

    for block in blocks:
        block_tokens = count_tokens(block.text)

        # A new top-level heading after we've accumulated enough is the natural
        # cut point. Top-level = level 1 or 2 for text formats; any heading
        # for structured formats (walker always emits ascending levels).
        is_high_heading = (
            block.kind == "heading"
            and block.level is not None
            and block.level <= 2
        )

        if (
            current.token_count >= parent_target_tokens
            and is_high_heading
        ):
            commit()

        # Hard cap: we must cut even mid-section.
        if current.token_count + block_tokens > hard_cap_tokens and current.blocks:
            commit()

        if not current.section_path and block.section_path:
            current.section_path = list(block.section_path)
        elif block.kind == "heading":
            # Update the parent's section_path to the deepest heading seen so
            # far, so citations point to the right sub-section.
            current.section_path = list(block.section_path)

        current.add(block, block_tokens)

    commit()

    # Materialize
    out: list[dict] = []
    for p in parents:
        text = p.render()
        if not text:
            continue
        out.append({
            "text": text,
            "section_path": p.section_path,
            "token_count": count_tokens(text),
        })
    return out
