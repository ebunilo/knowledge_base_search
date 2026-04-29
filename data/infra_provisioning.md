# Infrastructure Provisioning Plan

This document lists exactly what to provision and which environment variables to set, given the available accounts: **Hugging Face** (always), **Qwen API** (optional), **OpenAI** (optional).

It maps each runtime model to one of the two LLM lanes from the design (`hosted_ok` and `self_hosted_only`).

## Deployment Profiles

Three profiles are supported. The application code is identical — only env vars and endpoint configs change.

- `**demo`** (default for this project, data may leave the network): all inference via hosted/serverless APIs, no dedicated GPU endpoints, databases local in Docker. Target: **~$35–$140/month**.
- `**demo-isolated`**: same as `demo` but enforces the self-hosted lane with a scale-to-zero HF endpoint. Use when demo data must not leave the network. Target: ~$55–$180/month.
- `**prod**`: always-on generator, managed databases, OpenSearch for BM25, multi-region. Target: ~$2.2k–$4.1k/month.

Sections §2 and §5 below cover the `**prod**` profile. The `**demo-isolated**` profile is in §7. The `**demo**` profile is in §8, and is what you should provision now.

---

## 1. Model Inventory by Pipeline Role


| Role                                                             | Model                                                                                                                            | Where it runs                                           | Lane                    | Why                                                                                          |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------- |
| **Embeddings (dense + sparse)**                                  | `BAAI/bge-m3`                                                                                                                    | HF Inference Endpoint (dedicated, GPU small)            | both lanes              | Single model produces dense + sparse vectors; multilingual-ready.                            |
| **Cross-encoder reranker**                                       | `BAAI/bge-reranker-v2-m3`                                                                                                        | HF Inference Endpoint (dedicated, GPU small)            | both lanes              | Strong, open, lightweight.                                                                   |
| **NLI citation verifier**                                        | `MoritzLaurer/deberta-v3-large-mnli-fever-anli-ling-wanli`                                                                       | HF Serverless Inference API                             | both lanes              | Sentence-level entailment for post-stream citation check.                                    |
| **PII / sensitivity classifier (lightweight)**                   | `microsoft/Presidio` (CPU) + `dslim/bert-base-NER`                                                                               | Self-hosted CPU service (or HF Inference API for NER)   | both lanes              | Runs at ingestion, never sees user queries.                                                  |
| **Intent classifier**                                            | `Qwen2.5-1.5B-Instruct`                                                                                                          | Qwen API (preferred) or HF Inference API                | both lanes              | Tiny, fast; no sensitive content involved.                                                   |
| **Indexing-time enrichment** (hypothetical questions, summaries) | `Qwen2.5-7B-Instruct`                                                                                                            | Qwen API (cheap, batched)                               | both lanes              | Runs offline; chunks already classified — sensitive ones use a self-hosted endpoint instead. |
| **Indexing enrichment for sensitive chunks**                     | `Qwen2.5-7B-Instruct`                                                                                                            | HF dedicated Inference Endpoint (GPU medium)            | self_hosted_only        | Sensitive content never leaves our HF endpoint.                                              |
| **HyDE / query rewriting**                                       | `Qwen2.5-7B-Instruct`                                                                                                            | Qwen API (hosted lane), HF dedicated (self-hosted lane) | both lanes (lane-aware) | Mid-tier reasoning, on the warm path.                                                        |
| **Generator (hosted lane)**                                      | Primary `gpt-4o-mini` (OpenAI), fallback `claude-3-5-haiku` (if added later), final fallback `Qwen2.5-32B-Instruct` via Qwen API | OpenAI / hosted                                         | hosted_ok               | Best quality/$ for public-cleared content.                                                   |
| **Generator (self-hosted lane)**                                 | Primary `Qwen2.5-32B-Instruct` on HF dedicated endpoint, fallback `Qwen2.5-14B-Instruct` (smaller HF endpoint)                   | HF dedicated                                            | self_hosted_only        | Confidential content never leaves our infra.                                                 |


---

## 2. Hugging Face Provisioning Steps

You will provision **three Inference Endpoints** plus rely on the **Serverless Inference API** for everything else. Costs scale with GPU type and uptime; start small, scale on metrics.

### 2.1 Embeddings endpoint (`bge-m3`)

```
Name:           kb-embed-bge-m3
Model:          BAAI/bge-m3
Task:           Sentence Similarity / Feature Extraction
Hardware:       Nvidia L4 (small) or A10G — start with L4
Region:         eu-west-1 (matches data residency)
Scale-to-zero:  Enabled (5-min idle)
Min replicas:   1
Max replicas:   3
Container:      Default TEI (Text Embeddings Inference)
```

### 2.2 Reranker endpoint (`bge-reranker-v2-m3`)

```
Name:           kb-rerank-bge-v2-m3
Model:          BAAI/bge-reranker-v2-m3
Task:           Sentence Ranking
Hardware:       Nvidia L4
Region:         eu-west-1
Scale-to-zero:  Enabled (5-min idle)
Min replicas:   1
Max replicas:   2
```

### 2.3 Self-hosted-lane generator (`Qwen2.5-32B-Instruct`)

```
Name:           kb-llm-qwen25-32b
Model:          Qwen/Qwen2.5-32B-Instruct
Task:           Text Generation
Hardware:       Nvidia A100 80GB (or 2x A10G with quantization)
Quantization:   AWQ-Int4 (recommended to fit cleanly)
Region:         eu-west-1
Scale-to-zero:  Disabled (keep warm — interactive latency)
Min replicas:   1
Max replicas:   2
Container:      TGI (Text Generation Inference)
Max input tokens:  8192
Max total tokens:  10240
```

### 2.4 Serverless Inference API (no provisioning, pay-per-call)

Use for:

- `MoritzLaurer/deberta-v3-large-mnli-...` (NLI verification)
- `dslim/bert-base-NER` (PII augmentation if Presidio is insufficient)
- Optional fallback for `bge-m3` if dedicated endpoint is down

No setup needed beyond the HF token.

---

## 3. Environment Variables (the contract)

Place these in `.env` (gitignored) and load via `pydantic-settings`. The application code will reference only these names.

```bash
# ---------- Hugging Face ----------
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Dedicated Inference Endpoints (URLs HF gives you after creation)
HF_EMBED_ENDPOINT_URL=https://<your-id>.eu-west-1.aws.endpoints.huggingface.cloud
HF_RERANK_ENDPOINT_URL=https://<your-id>.eu-west-1.aws.endpoints.huggingface.cloud
HF_LLM_SELFHOSTED_ENDPOINT_URL=https://<your-id>.eu-west-1.aws.endpoints.huggingface.cloud

# Serverless Inference API (no URL needed; use HF Hub model IDs)
HF_NLI_MODEL_ID=MoritzLaurer/deberta-v3-large-mnli-fever-anli-ling-wanli

# ---------- Qwen API (optional but recommended for cost) ----------
QWEN_API_KEY=sk-qwen-xxxxxxxxxxxxxx
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1   # OpenAI-compatible
QWEN_MODEL_SMALL=qwen2.5-1.5b-instruct
QWEN_MODEL_MID=qwen2.5-7b-instruct
QWEN_MODEL_LARGE=qwen2.5-32b-instruct

# ---------- OpenAI (optional, hosted-lane primary generator) ----------
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL_GENERATOR=gpt-4o-mini

# ---------- Lane routing policy ----------
HOSTED_LANE_PRIORITY=openai:gpt-4o-mini,qwen:qwen2.5-32b-instruct,hf:kb-llm-qwen25-32b
SELFHOSTED_LANE_PRIORITY=hf:kb-llm-qwen25-32b,hf:kb-llm-qwen25-14b

# ---------- Infra ----------
QDRANT_URL=https://<qdrant-host>:6333
QDRANT_API_KEY=<qdrant-key>
OPENSEARCH_URL=https://<opensearch-host>:9200
OPENSEARCH_USER=acme_kb
OPENSEARCH_PASSWORD=<password>
POSTGRES_URL=postgresql://kb:<pwd>@<host>:5432/kb
REDIS_URL=rediss://default:<pwd>@<host>:6380

# ---------- Observability ----------
LANGSMITH_API_KEY=ls__xxxxxxxxxxxxxxxx
LANGSMITH_PROJECT=acme-kb-search
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# ---------- App ----------
APP_ENV=dev
APP_REGION=eu-west-1
APP_TENANT_INTERNAL_ID=tenant_acme_staff
APP_PUBLIC_COLLECTION=public_v1
APP_PRIVATE_COLLECTION=tenant_acme_staff_private_v1
```

---



---





---

---

## 7. Demo-Isolated Profile (only if data must not leave the network)

Goal: keep the architecture and lane routing intact, cut cost ~95%, accept cold-start on the first query after idle.

### 7.1 Model routing — demo tier


| Role                        | Where it runs (demo)                                                   | Notes                                                                               |
| --------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Embeddings (query-time)     | **HF Serverless Inference API** (`BAAI/bge-m3`)                        | No dedicated endpoint. Small per-call cost.                                         |
| Embeddings (ingestion bulk) | **Self-hosted TEI in Docker** on your machine                          | One-off burst; free. `docker run ghcr.io/huggingface/text-embeddings-inference ...` |
| Reranker                    | **HF Serverless Inference API** (`BAAI/bge-reranker-v2-m3`)            | No dedicated endpoint.                                                              |
| NLI citation verifier       | **HF Serverless Inference API**                                        | Same as prod.                                                                       |
| Self-hosted LLM             | **HF dedicated endpoint, `Qwen2.5-14B-Instruct`, A10G, SCALE-TO-ZERO** | Cold-start ~30–90s on first call after idle.                                        |
| Hosted LLM                  | **OpenAI `gpt-4o-mini`**                                               | Pay-per-call, cheap.                                                                |
| Indexing enrichment         | **Qwen API `qwen2.5-7b-instruct`**, capped to a sample                 | `INGESTION_ENRICH_LIMIT=500` until you demo the full corpus.                        |
| Intent classifier           | **Qwen API `qwen2.5-1.5b-instruct`**                                   | Pay-per-call, negligible cost.                                                      |


### 7.2 Data plane — demo tier

Everything in one `docker-compose.yml`, running on your dev machine or a single small VM:

```
services:
  qdrant:       image: qdrant/qdrant:latest
  opensearch:   (OPTIONAL — or swap for in-process rank_bm25, see below)
  postgres:     image: postgres:16
  redis:        image: redis:7-alpine
  tei-embed:    image: ghcr.io/huggingface/text-embeddings-inference:cpu-latest  # ingestion only
```

**For BM25, prefer `rank_bm25` in-process** for the demo (no OpenSearch needed). The retriever layer exposes a `BM25_BACKEND` env var: `rank_bm25` (demo) or `opensearch` (prod). Same interface.

### 7.3 Demo-tier env vars (overrides)

```bash
# Deployment profile
APP_PROFILE=demo

# Scale-to-zero generator (dedicated endpoint, but cold when idle)
HF_LLM_SELFHOSTED_ENDPOINT_URL=https://<your-14b-id>.eu-west-1.aws.endpoints.huggingface.cloud
HF_LLM_SELFHOSTED_MODEL=Qwen/Qwen2.5-14B-Instruct
HF_LLM_SELFHOSTED_KEEP_WARM=false            # scale to zero after idle
HF_LLM_SELFHOSTED_COLD_START_TIMEOUT_S=120   # client-side wait budget for first call

# No dedicated embed/rerank endpoints — use serverless
HF_EMBED_MODE=serverless
HF_EMBED_MODEL_ID=BAAI/bge-m3
HF_RERANK_MODE=serverless
HF_RERANK_MODEL_ID=BAAI/bge-reranker-v2-m3

# BM25 in-process
BM25_BACKEND=rank_bm25

# Local data plane
QDRANT_URL=http://localhost:6333
OPENSEARCH_URL=                               # unset in demo
POSTGRES_URL=postgresql://kb:kb@localhost:5432/kb
REDIS_URL=redis://localhost:6379

# Cap expensive indexing enrichment
INGESTION_ENRICH_LIMIT=500
INGESTION_ENRICH_STRATEGY=sample_first        # {sample_first, all, off}

# Warm-up controls
WARMUP_ON_APP_START=true                      # fire-and-forget ping at boot
WARMUP_CRON_WINDOW=                           # e.g., "0,10,20,30,40,50 9-11 * * MON-FRI" during demo days
```

### 7.4 Demo-tier cost estimate


| Item                                                             | Cost                  |
| ---------------------------------------------------------------- | --------------------- |
| HF 14B endpoint (A10G, scale-to-zero, ~2h warm/day on demo days) | $30–$80               |
| HF Serverless (embed queries + rerank + NLI)                     | $10–$40               |
| OpenAI `gpt-4o-mini` (light usage)                               | $10–$40               |
| Qwen API (capped enrichment + intent classifier)                 | $5–$20                |
| Databases (Docker on local/VM)                                   | $0                    |
| LangSmith (free tier)                                            | $0                    |
| **Approx total**                                                 | **~$55–$180 / month** |


### 7.5 Warm-up strategy for demo day

Pick one — all three are supported via env vars:

1. **Ping-at-boot** (`WARMUP_ON_APP_START=true`): the app sends a 1-token dummy generation to the HF endpoint when it starts. Covers "demo starts 1 minute after I start the app".
2. **Cron window** (`WARMUP_CRON_WINDOW=...`): pings the endpoint every 10 minutes during your demo window only. No cost outside the window.
3. **Admin endpoint** (`POST /admin/warmup`): click a button in the admin UI ~60s before demoing.

### 7.6 What I need you to provision for the demo tier

1. **One HF Inference Endpoint**: `Qwen/Qwen2.5-14B-Instruct`, Nvidia A10G, region eu-west-1, **scale-to-zero enabled (5-min idle)**. Name it `kb-llm-qwen25-14b-demo`. Send me the URL → goes into `HF_LLM_SELFHOSTED_ENDPOINT_URL`.
2. **HF token** with Inference API access (same one works for serverless + dedicated).
3. **OpenAI API key**.
4. **Qwen API key** (DashScope or equivalent).
5. **A dev machine or one small VM** (4 vCPU, 8 GB RAM is enough) with Docker installed to run Qdrant/Postgres/Redis locally. No GPU needed on this box — the generator is remote.

That's it. No Qdrant Cloud, no OpenSearch cluster, no always-on GPU. Paste the endpoint URL and three API keys into `.env`, I'll run the smoke-test suite, and we're ready for Phase 0.

---

## 8. Demo Profile (use this — data may leave the network)

Goal: **maximum cost cut** by collapsing both lanes onto hosted/serverless APIs. No dedicated HF endpoint. No GPU to provision. No cold-start wait.

The two-lane architecture stays in code — in this profile the self-hosted lane is configured to borrow the hosted providers, gated behind an explicit override flag. Flipping the flag off (production, or any privacy-sensitive environment) immediately re-imposes the isolation boundary.

### 8.1 Model routing — demo profile


| Role                         | Where it runs                                                 |
| ---------------------------- | ------------------------------------------------------------- |
| Embeddings (query-time)      | **HF Serverless Inference API** (`BAAI/bge-m3`)               |
| Embeddings (ingestion bulk)  | **Self-hosted TEI in Docker** on your machine (one-off, free) |
| Reranker                     | **HF Serverless Inference API** (`BAAI/bge-reranker-v2-m3`)   |
| NLI citation verifier        | **HF Serverless Inference API**                               |
| Generator — hosted lane      | **OpenAI `gpt-4o-mini`**                                      |
| Generator — self-hosted lane | **OpenAI `gpt-4o-mini`** (via override; Qwen API as fallback) |
| Indexing enrichment          | **Qwen API `qwen2.5-7b-instruct`**, capped sample             |
| Intent classifier            | **Qwen API `qwen2.5-1.5b-instruct`**                          |


### 8.2 Data plane — demo profile

Identical to §7.2: one `docker-compose.yml` with Qdrant + Postgres + Redis + TEI (for ingestion). BM25 via in-process `rank_bm25`. No OpenSearch.

### 8.3 Demo profile env vars (overrides)

```bash
# Deployment profile
APP_PROFILE=demo

# NO dedicated HF endpoint — both lanes collapse to hosted
HF_LLM_SELFHOSTED_ENDPOINT_URL=                      # intentionally unset
DEMO_ALLOW_HOSTED_FOR_SELFHOSTED_LANE=true           # demo-only override

# Lane routing — both lanes point at hosted providers in demo
HOSTED_LANE_PRIORITY=openai:gpt-4o-mini,qwen:qwen2.5-32b-instruct
SELFHOSTED_LANE_PRIORITY=openai:gpt-4o-mini,qwen:qwen2.5-32b-instruct   # override-enabled

# Serverless embeddings + rerank
HF_EMBED_MODE=serverless
HF_EMBED_MODEL_ID=BAAI/bge-m3
HF_RERANK_MODE=serverless
HF_RERANK_MODEL_ID=BAAI/bge-reranker-v2-m3

# BM25 in-process
BM25_BACKEND=rank_bm25

# Local data plane
QDRANT_URL=http://localhost:6333
OPENSEARCH_URL=                                       # unset in demo
POSTGRES_URL=postgresql://kb:kb@localhost:5432/kb
REDIS_URL=redis://localhost:6379

# Cap expensive indexing enrichment
INGESTION_ENRICH_LIMIT=500
INGESTION_ENRICH_STRATEGY=sample_first                # {sample_first, all, off}

# Warm-up not needed — no cold-start risk with hosted APIs
WARMUP_ON_APP_START=false
WARMUP_CRON_WINDOW=
```

### 8.4 Demo profile cost estimate


| Item                                             | Cost                  |
| ------------------------------------------------ | --------------------- |
| HF Serverless (embed queries + rerank + NLI)     | $10–$40               |
| OpenAI `gpt-4o-mini` (both lanes)                | $20–$80               |
| Qwen API (capped enrichment + intent classifier) | $5–$20                |
| Databases (Docker on local/VM)                   | $0                    |
| LangSmith (free tier)                            | $0                    |
| **Approx total**                                 | **~$35–$140 / month** |


### 8.5 Safety rails for the override flag

Because `DEMO_ALLOW_HOSTED_FOR_SELFHOSTED_LANE=true` weakens the network-isolation guarantee, the application must:

- **Log a WARN-level line at every startup** if the flag is true, listing the loaded provider priorities for both lanes.
- **Refuse to start** if both `APP_PROFILE=prod` and `DEMO_ALLOW_HOSTED_FOR_SELFHOSTED_LANE=true` are set. These are incompatible.
- **Tag every audit-log entry** in demo mode with `lane_isolation: "demo_override"` so traces are distinguishable from prod behavior during later analysis.
- **Emit a LangSmith run tag** (`demo_override=true`) so eval runs don't get mixed across profiles.

These rails are trivial to implement and prevent the demo config from silently ending up in a privacy-sensitive environment.

### 8.6 What I need you to provision for the demo profile

1. **HF token** with serverless Inference API access.
2. **OpenAI API key**.
3. **Qwen API key** (DashScope or equivalent).
4. **A machine with Docker** (4 vCPU, 8 GB RAM). No GPU needed anywhere.

That's it. **No Hugging Face Inference Endpoints to create.** Paste the three API keys into `.env`, start docker-compose, I run the smoke-test suite, and we're unblocked.

---

## 9. Smoke-Test Checklist (after you paste keys/URLs)

Once you've provisioned and shared the keys/URLs, I will run these in order before writing any application code. The set depends on the selected `APP_PROFILE`.

**Common (all profiles):**

1. Round-trip a 1-sentence embedding via HF Serverless → assert vector dim == 1024 (bge-m3).
2. Round-trip a (query, candidate) pair through the serverless reranker → assert score returned.
3. Round-trip a 5-token completion through each configured hosted generator (OpenAI, Qwen API).
4. Round-trip NLI inference against HF Serverless.
5. Start docker-compose data plane (Qdrant + Postgres + Redis) and verify health endpoints.
6. Confirm LangSmith trace appears.

`**demo` profile additional:**

1. Confirm `DEMO_ALLOW_HOSTED_FOR_SELFHOSTED_LANE=true` is logged at startup.
2. Assert the safety rail: app refuses to start when `APP_PROFILE=prod` and the override flag is true.
3. Verify audit-log entries are tagged `lane_isolation: "demo_override"`.

`**demo-isolated` / `prod` additional:**

1. `curl` the HF dedicated endpoint with a minimal payload; measure cold-start time; record as baseline.
2. Round-trip a 5-token completion through the HF self-hosted endpoint.
3. Verify lane-routing logic: a request flagged `self_hosted_only` never hits OpenAI/Qwen API (routing must deny it, not just "by convention").
4. Confirm warm-up mechanism of choice works end-to-end (demo-isolated only).

These checks become the first integration-test suite, gating Phase 1.