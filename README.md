# Knowledge Base Search

**Enterprise RAG (retrieval-augmented generation)** over your own documents: hybrid **semantic + lexical** search, **citation-first** answers, access control, and optional **NLI-based faithfulness** checks—served by a **FastAPI** app and **Click** CLI, with **Docker Compose** for the data plane.

## Problem and solution

**Problem:** Teams need **grounded** answers from internal and customer-facing knowledge—not generic model hallucinations. A minimal “embed and chat” stack often **misses relevant chunks** (wording mismatch), **surfaces the wrong passage**, and cannot show **provenance** or **enforce** who is allowed to see which sources.

**Solution:** This project ingests documents into a **durable, multi-index** store (text + vectors + sparse terms), then answers questions by **retrieving** evidence, **assembling** a token-budgeted **context** with numbered citations, and **generating** with instructions to cite only that context. It combines **dense (embedding) retrieval**, **BM25**, **reciprocal rank fusion (RRF)**, an optional **cross-encoder reranker**, and **per-request ACL** so answers stay tied to **permitted** content. **Optional** post-hoc **NLI** verification scores whether cited sentences are supported by the cited passages, and a **confidence** score reflects retrieval and verification strength.

## Features

- **Ingestion pipeline:** connectors → parse → **parent/child** chunking → optional **LLM enrichment** (synthetic questions + summaries) → **embed** → idempotent **Postgres + Qdrant + BM25** indexing.
- **Hybrid retrieval:** parallel **dense** (Qdrant) and **sparse** (BM25) search, **RRF** fusion, optional **query rewrite** (multi-query, HyDE-style, **step-back**, **conversation** rewriting with Redis sessions), optional **cross-encoder** rerank, **parent** expansion for context.
- **Grounded generation:** system prompt enforces **only-CONTEXT** answers and **`[N]` citations**; refusal paths when there are no hits or low score (configurable); **streaming** and non-streaming **Ask** APIs.
- **Governance:** collection-level **ACL**, **sensitivity lanes** (hosted vs self-hosted model routing without mixing in one answer), **query guardrails** before expensive work.
- **Trust signals:** **citation** parsing (invalid markers, uncited hits), optional **faithfulness** (NLI over cited sentences), **aggregate confidence**.
- **Product surfaces:** **`kb` CLI** (ingest, search, ask, health, eval, sessions), **REST + SSE** under `/api/…`, static **web UI** in `kb/web/static/`.
- **Operations:** `APP_PROFILE` (`demo` / `demo-isolated` / `prod`), optional **LangSmith**, eval harness and optional **RAGAS**-style batches (`eval` extra).

## Technical details

| Area | Stack |
|------|--------|
| **Runtime** | Python **≥3.10**; app served with **Uvicorn** (default port **8765**). |
| **Config** | **Pydantic Settings**; environment variables and `.env` (see **`.env.example`**). |
| **Text / PDF** | **pypdf**, **pymupdf4llm**; optional **geometric** PDF margin prep (`scripts/clean_pdf_margins.py`, `kb/preprocessing/`). |
| **Embeddings** | **Hugging Face** serverless or **dedicated endpoint**; optional local **TEI** via Compose `--profile ingestion` (set **`HF_EMBED_MODE=endpoint`** and point **`HF_EMBED_ENDPOINT_URL`** at the TEI `/embed` URL; see `docker-compose.yml`). |
| **Vector DB** | **Qdrant** (HTTP REST from the app). |
| **Canonical text + metadata** | **PostgreSQL**. |
| **Sessions** | **Redis** (TTL’d conversation for rewriter; optional for one-shot use). |
| **Sparse** | In-process **rank-bm25**; **`BM25_BACKEND`** can target **OpenSearch** in larger deployments. |
| **LLM providers** | **OpenAI**, **Qwen**-compatible, **Hugging Face** (embed, rerank, NLI, self-hosted LLM endpoints) — **lane**-aware routing. |

**Package name:** `kb` (import `kb`, CLI entry: **`kb`** from `pyproject.toml`).

## Quick start (Docker Compose)

1. **Environment**

   ```bash
   cp .env.example .env
   ```

   Fill in API keys, database secrets, and Qdrant API key. See `data/infra_provisioning.md` for profiles and provider notes.

2. **Start the stack** (app + Postgres + Qdrant + Redis):

   ```bash
   docker compose up -d --build --wait
   ```

3. **Open the UI** (default bind in `.env` is often loopback; adjust for remote access and use HTTPS in production):

   - `http://localhost:8765`
   - Health: `GET /api/health`

4. **Ingest** sample or real sources (not automatic on file drop). Example for the public demo tree:

   ```bash
   docker compose exec app kb ingest --source src_sample_public --stage index
   ```

   Sources are defined in **`data/source_inventory.json`**. See **`documentation/clean_sample_docs.md`** for optional PDF margin stripping before ingest.

5. **Stop**

   ```bash
   docker compose down
   ```

**Optional TEI** (local embeddings, bulk ingest): `docker compose --profile ingestion up -d` and configure the embedding client to the TEI base URL (see `docker-compose.yml` service `tei`).

## Configuration highlights

- **`APP_PROFILE`**: `demo` \| `demo-isolated` \| `prod` (production rails; unsafe demo flags rejected in `prod`).
- **Retrieval / generation** tunables via **`RetrievalConfig`** and **`GenerationConfig`** (API/CLI) and defaults in `kb/settings.py` (e.g. faithfulness threshold, context token budget, guardrails).
- **Collections:** `APP_PUBLIC_COLLECTION`, `APP_PRIVATE_COLLECTION` (see `.env.example`).

## CLI and API

- **CLI:** `kb --help` — e.g. `kb ingest`, `kb search`, `kb ask`, `kb health`, `kb eval`, session-related commands.
- **REST (FastAPI):** e.g. `GET /api/health`, `GET /api/config`, `POST /api/search`, `POST /api/ask`, `POST /api/ask/stream` (Server-Sent Events), `GET /api/session/new`. Interactive docs: **`/docs`**, OpenAPI: **`/openapi.json`**.

## Documentation

| Doc | Content |
|-----|--------|
| `documentation/architecture.md` | System design, data flow, deployment notes |
| `documentation/features.md` | RAG mechanisms for grounding and anti-hallucination |
| `documentation/advanced_rag.md` | Deeper RAG design rationale |
| `nli_faithfulness.md` | NLI / faithfulness pipeline |
| `data/infra_provisioning.md` | Profiles, model inventory, provisioning |
| `documentation/clean_sample_docs.md` | PDF margin stripping before ingest |

**Non-goals in-repo:** end-user **SSO**, **billing**, production **secrets management**—integrate with your platform.

## License

MIT (see `pyproject.toml`).
