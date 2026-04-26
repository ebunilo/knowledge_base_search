# Advanced RAG: What Is Implemented and Why It Improves Quality

This document describes the **knowledge base search** application: a **standalone, production-oriented** RAG (Retrieval-Augmented Generation) system. It explains **what was built** in each stage—**chunking**, **embedding**, **retrieval**, and **generation**—and **how** those design choices improve relevance, groundedness, safety, and operability compared to a minimal “embed chunks + single vector search + one-shot LLM” baseline.

For deployment topology and component boundaries, see `architecture.md`. For NLI-based faithfulness in more detail, see `nli_faithfulness.md`.

---

## 1. Chunking

**Goal:** Produce indexable units that stay **linguistically coherent**, preserve **document structure** for citations, and support **retrieval at the right granularity** (find the right place, then surface enough **parent** context to answer).

### 1.1 Structure-aware parsing

- **PDF:** Primary path uses **pymupdf4llm** to produce **Markdown** with **headings** and structure; **pypdf** fallback for resilience (`kb/parsers/pdf.py`). Clean structure improves downstream section paths and chunk boundaries.
- **Structured data (YAML, JSON, etc.):** Walkers preserve **key paths** so section labels remain meaningful in citations (`kb/parsers/`).

**Why it helps RAG:** Answers reference **where** a fact came from; retrieval and ranking can use **hierarchical** hints, not only flat text.

### 1.2 Hierarchical parent–child chunking

**Implementation** (`kb/chunking/parent_child.py`, `structural.py`):

- **Parents** — **section-scale** segments bounded by **token targets** (`parent_chunk_target_tokens` in `kb/settings.py`). These are the units the **generator** can consume as high-level **context** after a child hits.
- **Children** — **searchable** segments inside each parent: **paragraph-first** split, then **sentence-level** packing, with **small overlap** between adjacent children to reduce “lost” facts at chunk edges. **Code** blocks are split carefully to avoid breaking syntax mid-block.

**Why it helps RAG:** Children match **query precision**; parents give **sufficient** surrounding text for **grounded** answers, matching a proven **“small to big”** pattern.

### 1.3 Token-budgeted sizing

- Chunk limits are expressed in **tokens** (`kb/chunking/tokens.py`) so budgets align with **embedder and LLM** limits rather than arbitrary character cuts.

**Why it helps RAG:** Fewer oversized chunks (weak embeddings, truncation noise) and fewer pathological micro-chunks (fragmented retrieval).

### 1.4 Sensitivity and policy at chunk time

- Each `ChunkedDocument` is classified to a **sensitivity lane** (`kb/classifier/`, `parent_child` flow) so the **most restrictive** policy is visible end-to-end.

**Why it helps RAG + production:** Drives **consistent** **lane-aware** **generation** later; avoids mixing **policy** at answer time in ad hoc ways.

### 1.5 Ingestion-time enrichment (optional)

- The pipeline can add **synthetic questions** and **summaries** to chunks (`kb/enrichment/`, `kb/orchestration/pipeline.py`).

**Why it helps RAG:** Extra **retrieval query–like** text and short **semantic** descriptors increase **recall** when the user’s wording does not match raw document diction, without changing the **authoritative** source text stored for answering.

**Relation to query-time rewriting:** Ingestion enrichment **widens the index**; the **query rewriter** (see §3) **adapts the user question** at request time. Both improve recall; they operate on different parts of the pipeline.

---

## 2. Embedding and indexing

**Goal:** **Stable**, **batched** vectors for children (and optional question fields) with **robust I/O** to the vector store.

### 2.1 Implementation

- **BGE**-class models (e.g. **bge-m3** via `kb/embeddings/client.py`), with **serverless** or **dedicated endpoint** configuration from `kb/settings.py`.
- **Batched** requests with **max token** and item ceilings, **retries** with backoff, and **dimension** discovery for Qdrant payloads.
- Vectors are written with **enrichment**-aware payloads when enrichment is enabled, so the **same** retrieval path can match on **content + questions** (where configured).

**Why it helps RAG:** **Throughput** and **reliability** in production; consistent **vector geometry** for query vs document side when the same model family is used; optional **enrichment** **surface** for better **dense** **match** rates.

### 2.2 Multi-store indexing (production durability)

- **Idempotent** writes: **Postgres** (source of truth for text), **Qdrant** (dense), **BM25** (sparse), with **content-hash** decisions, ordered writes, and **rollback** behavior on failure (`kb/indexing/multi_writer.py`).

**Why it helps RAG + operations:** The **knowledge** layer stays **consistent** across modalities; re-ingestion does not silently duplicate or leave **orphan** vectors that would **harm** **retrieval** quality and **trust**.

---

## 3. Retrieval

**Goal:** **High recall** (find candidates) and **high precision** (right ordering after fusion), with **governance** and **degradation** when subsystems fail.

### 3.1 Query rewriting (`kb/retrieval/rewrite.py`)

The **`QueryRewriter`** can combine several techniques in a **single** **JSON**-structured **LLM** call (toggles via `RetrievalConfig` and settings):

| Technique | Role | RAG effect |
|----------|------|------------|
| **Multi-query** | Extra paraphrases of the same information need | Diverse dense/sparse query variants; better recall for vague or differently phrased questions |
| **HyDE**-style | Hypothetical passage for embedding (not the final answer) | Query embedding closer to document-style phrasing; helps short or mismatched queries |
| **Coreference / “canonical”** resolution | Uses prior turns in the session | Follow-ups like “its limits” become self-contained; essential for chat |
| **Step-back** (Zheng et al., 2023) | Broader variant of the same intent | Improves recall when the user query is more specific than indexed wording |

On **failure**, the rewriter falls back to the **original** query so retrieval always has a path forward.

**Why it helps RAG:** The retriever is no longer limited to a single raw string; variants align the query with how the corpus is written, while the fallback path preserves liveness if the rewriter model errors.

### 3.2 Hybrid search and fusion

- **Dense** (Qdrant) and **sparse** (BM25) run in **parallel**; **RRF** (Reciprocal Rank Fusion) merges ranked lists (`kb/retrieval/fusion.py`, `kb/retrieval/retriever.py`).
- Tunables include per-modality top-k, RRF constant, and per-channel weights (`RetrievalConfig` in `kb/retrieval/types.py`).

**Why it helps RAG:** Vectors help with paraphrase and meaning; sparse retrieval helps with rare terms, codes, and identifiers. Fusion produces a more stable top set across query phrasings than a single retriever alone.

### 3.3 Cross-encoder reranking (optional)

- A second stage reorders candidates after ACL using a cross-encoder (`kb/retrieval/rerank.py`), with fallback to the fused list on error.

**Why it helps RAG:** Improves precision in the final top-k and reduces distractor chunks before the generator runs.

### 3.4 Access control and safety

- ACL filter pushdown and Python-side checks so out-of-policy hits are not used for ranking or later stages (`kb/retrieval/acl.py`).

**Why it helps production RAG:** Wrong chunks are not only quality bugs—they can be compliance violations. The reranker and fusion only see post-ACL candidates.

### 3.5 Parent expansion

- Retrieval scores child segments; **parent** text is loaded for context assembly and display (`ParentStore`, retriever flow).

**Why it helps RAG:** Answers cite coherent passages with stable source identity and section paths, not token-split fragments in isolation.

---

## 4. Generation

**Goal:** **Grounded** answers with **explicit** **citation** **contracts**, **refusals** when **evidence** is **insufficient**, **optional** **verification**, and **lane-consistent** **LLM** **use**.

### 4.1 Prompts (`kb/generation/prompt.py`)

- **System:** Numbered `CONTEXT` blocks, `[N]` citation contract, no facts outside `CONTEXT`, explicit copy when evidence is missing.
- **User:** `CONTEXT` + `QUESTION` + `ANSWER` scaffold for instruction-tuned models.

**Why it helps RAG:** Reduces hallucination and keeps citation parsing deterministic for the UI and optional faithfulness checks.

### 4.2 Context assembly and refusal gates

- Token-budgeted assembly of retrieved passages (`kb/generation/context.py`, `ContextAssembler`).
- Refusal when there are no hits or scores fall below policy (configurable in `Generator` / `GenerationConfig`), so the model is not called without evidence.

**Why it helps RAG:** Cuts cost and prevents confident answers when nothing relevant was retrieved.

### 4.3 Sensitivity lanes and one lane per answer

- If any hit requires the self-hosted lane, the full answer uses that lane only—no mixing hosted and self-hosted models in one response (`kb/generation/generator.py`).

**Why it helps production RAG:** Respects data-sovereignty rules and prevents cross-boundary leakage through paraphrase.

### 4.4 Citations, confidence, faithfulness, streaming

- Citations parsed and linked to retrieval metadata (`kb/generation/citations.py`).
- Confidence for UX and optional gating (`kb/generation/confidence.py`).
- Optional NLI faithfulness over cited sentences (`kb/generation/faithfulness.py`).
- Streaming for the same pipeline (`StreamEvent` in generation types).

**Why it helps RAG:** Verifiable answers, user-visible uncertainty, better perceived latency, and an extra groundedness check where policy requires it.

### 4.5 Sessions (Redis)

- Turn history with TTL and caps; feeds the query rewriter for coreference-aware retrieval (`kb/sessions/`).

**Why it helps RAG:** Multi-turn conversations behave like a real product without restating the whole context every turn.

---

## 5. Cross-cutting production concerns

| Concern | Implementation (indicative) |
|--------|----------------------------------|
| Ingestion stages | `chunk` → `enrich` → `embed` → `index` in `kb/orchestration/pipeline.py` |
| Configuration | Pydantic `Settings`, profiles, lane priorities in `kb/settings.py` |
| Entitlements | `UserContext`, staff directory, ACL in `kb/retrieval/acl.py` and `data/staff_directory.json` |
| Eval | Golden sets and metrics harness in `kb/eval/` |
| Runtimes | Web API + static UI in `kb/web/`; Docker in `docker-compose.yml` and `Dockerfile` |
| Observability | Logging; optional LangSmith via settings |

These are **operational and governance** requirements for a production RAG system—distinct from the ranking and prompting choices above, but required for a defensible deployable product.

---

## 6. References in this repository

- `architecture.md` — system architecture and data flow  
- `nli_faithfulness.md` — NLI / faithfulness checking  
- `kb/retrieval/rewrite.py` — query rewriter (multi-query, HyDE-style, step-back, coref)  
- `kb/retrieval/retriever.py` — hybrid retrieve + fusion + rerank + parent expansion  
- `kb/chunking/parent_child.py` — parent/child chunking  
- `kb/generation/prompt.py` and `kb/generation/generator.py` — prompts and generation orchestration  

**External (techniques named above):** Gao et al. — *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE, 2022); Zheng et al. — *Take a Step Back* (step-back prompting, 2023).
