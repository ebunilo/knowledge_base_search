# RAG features for grounded answers (reducing hallucination)

This document summarizes **implemented** mechanisms in the knowledge base search solution that support **factual grounding** and **detection of ungrounded claims**, so that LLM outputs stay tied to the corpus and unsafe or low-trust cases are visible or blocked.

It is *not* a guarantee of correctness: models can still err, and heuristics have limits. The design combines **retrieval quality**, **prompting constraints**, **post-hoc verification**, and **gating** so hallucinations are *harder* and *more observable*.

---

## 1. Hybrid retrieval and re-ranking

**Why it helps:** Wrong or weak retrieval is a primary source of “confident nonsense.” Combining complementary signals and re-ordering with a cross-encoder improves the odds that the **CONTEXT** shown to the model actually contains the right evidence.

- **Dense (embedding) + sparse (BM25) search** in parallel, then **Reciprocal Rank Fusion (RRF)** to merge ranked lists. RRF is robust when one retriever misses phrasing the other captures.
- **Optional query rewriting** (`multi_query`, `HyDE`, or both): expands or reframes the query to improve recall; failures fall back to the original query so the path degrades safely.
- **Cross-encoder re-ranking** (e.g. BGE-reranker) scores (query, passage) pairs and re-orders the fused pool for **precision** after recall-oriented retrieval. If the reranker is unavailable, the pipeline keeps the RRF order and logs a warning.
- **Configurable weights and pool sizes** (`RetrievalConfig`) so you can trade latency for quality in different deployments.

---

## 2. Access control and “lane” policy

**Why it helps:** Stops the model from **blending** evidence the user is not allowed to use, which would be a form of *policy* hallucination (wrong provenance) even if the text sounds true.

- **Collection routing and per-hit ACL** (Qdrant pushdown for dense search, and Python checks for the sparse path) ensure only permitted documents enter the candidate set.
- **Reranker** only sees **post-ACL** candidates, so a ranking step cannot surface blocked content.
- **Sensitivity lanes** (e.g. hosted vs. self-hosted): if the hit set straddles incompatible lanes, the generator avoids an answer path that could mix them (defence in depth with faithfulness and settings).

---

## 3. Parent–child (small-to-big) context

**Why it helps:** Chunks that are too small can miss surrounding sentences the model needs to answer; **parent expansion** assembles a richer passage for the same retrieval decision, so answers can be **grounded in fuller source text** within the context budget.

- Retrieval can **deduplicate by parent** and **fill parent content** for assembly; the **context assembler** then packs numbered blocks under a token budget (with per-hit caps and truncations).

---

## 4. Prompting for extractive, cited answers

**Why it helps:** The **system prompt** steers the model toward **“answer only from CONTEXT”**, **mandatory `[N]` citations** for factual claims, and an explicit **refusal** when the sources are insufficient. The **user prompt** uses a clear `CONTEXT` / `QUESTION` / `ANSWER` structure that instruction-tuned models follow reliably.

- **Low generation temperature** (default near `0.1`) reduces creative drift.
- This does **not** stop all fabrication; it **aligns** the model with the intended contract so downstream checks can reason about citations and entailment.

---

## 5. Citation parsing and health signals

**Why it helps:** You can see when the model **cites correctly**, **invents a marker** (`[9]` when only five blocks exist), or **ignores** provided passages.

- After generation, the answer is scanned for `[N]` markers and each is **resolved to a retrieval hit** (with document id, source URI, section path, scores, and visibility metadata).
- **Invalid markers** and **uncited hits** are surfaced for **UX, audits, and evals**, not as a hard model-side fix— they make silent hallucination **measurable**.

---

## 6. Faithfulness (NLI) verification

**Why it helps:** A second model pass checks whether **cited** sentences in the answer are **entailed** by the cited source text, catching **unsupported paraphrase or invention** that still “looks” cited.

- Sentences are split (with rules that **preserve** citation markers like `[1]`), and for each **cited** sentence an **NLI** model scores whether the **premise** (source) **entails** the **hypothesis** (answer sentence), against a configurable threshold.
- **Unverified** sentences (no markers) and **unsupported** (marker present but entailment low) are counted; the result includes **supported ratio**, **mean entailment**, and **call counts** for cost tracing.
- If NLI is down or a lane **skips** verification, the system records a **fallback reason** instead of faking a pass.

This is a strong **post-generation** signal for “did the model stay within what the sources support?” and complements retrieval quality.

---

## 7. Aggregate confidence (retrieval × faithfulness)

**Why it helps:** A single score is easier to **flag** weak answers. The implementation **combines retrieval strength** with the **faithfulness** signal (using a **geometric mean** so a poor score on *either* dimension pulls the overall confidence down; when verification is missing, the score is **capped** to avoid overstating trust).

- Tuning of floors and normalisation is intended to improve over time (e.g. against a golden / eval set), but the **shape** of the score already discourages “good retrieval, bad faithfulness” and vice versa.

---

## 8. Refusal and gating (no-LLM and low-evidence paths)

**Why it helps:** The cheapest way to avoid hallucination is to **not answer** when there is no usable evidence.

- **No retrieval hits:** optional **deterministic refusal** (no LLM call), so the model never invents an answer for out-of-corpus questions.
- **Top-hit score below a floor** (`min_score_threshold`): can refuse **without** calling the LLM when configured (disabled by default until calibrated in your environment).
- **User-query guardrails** (length, prompt-injection heuristics, code-dump patterns) run **before** retrieval/LLM to avoid processing adversarial or oversized inputs; blocked requests return a safe response without invoking the main pipeline.

---

## 9. Conversation-aware rewriting

**Why it helps:** Follow-up questions with pronouns (“it”, “that step”) can **retrieve the wrong** passages if the query is left literal. A **session-aware rewriter** (when enabled) can produce a **canonical** query and variants so **sparse + rerank** stay aligned with what the user meant—reducing **retrieval error** on multi-turn questions (a major indirect cause of grounded answers that look wrong).

---

## 10. Ingestion and corpus hygiene (indirect)

**Why it helps:** Cleaner source text **before** indexing (for example, PDF margin stripping for manuals) **reduces repeated boilerplate** in chunks. That is not a runtime RAG *tool* but it **improves** what hybrid search and the LLM see, so answers are **less** polluted by headers and footers. See `documentation/clean_sample_docs.md` for the PDF path.

---

## 11. Evaluation and calibration hooks

**Why it helps:** **Regression** in retrieval or NLI can **look** like the model “suddenly hallucinating.” The repo includes an **eval** harness and optional **RAGAS**-style batch flows so you can **measure** answer quality and faithfulness over a golden set and **tune** thresholds (scores, NLI cutoffs) with data instead of only by eye.

---

## Summary table

| Area | Mechanism | Primary anti-hallucination role |
|------|------------|---------------------------------|
| Retrieval | RRF, dense+sparse, optional rewrite/HyDE | Better evidence in context |
| Retrieval | Cross-encoder rerank | Precise (query, passage) match |
| Security | ACL + collection routing | No wrong provenance / mixing |
| Context | Parent expansion + token budget | More grounded text per hit |
| Generation | System/user prompts + low temperature | Answer-from-context contract |
| Post-gen | Citation parse + invalid markers | Surface invented `[N]` |
| Post-gen | NLI faithfulness on cited sentences | Detect unsupported claims |
| Scoring | Combined confidence | Flag weak overall answers |
| Gating | No hits / (optional) low score / guardrails | Refuse or block early |
| Multi-turn | Query rewriting with sessions | Less pronoun-driven retrieval error |
| Ops | Eval / golden / calibration | Prevent silent regressions |

Together, these features implement a **defence-in-depth** RAG stack: **retrieve the right text**, **constrain the generator**, **verify claims against cited sources when possible**, and **refuse or signal** when confidence is low.
