# Demo questions (sample corpus, no PDFs)

Use these for a **fast demo** when the ingested text comes from `data/sample_docs/**` **excluding** large PDFs. The snippets below map 1:1 to the current **Acme** sample files: `getting-started.md`, `architecture.md`, `openapi.yaml`, `pricing.json` (public) and `onboarding-policy.md`, `incident-runbook.md`, `deploy-config.yaml` (internal).

**Public vs internal:** run **public** questions as an anonymous or customer user. Run **internal** questions only when impersonating a **staff** user with access to the private collection (see `data/staff_directory.json` and ACL in the app).

---

## Public lane (`data/sample_docs/public/`)

| # | Question |
|---|----------|
| 1 | How do I create an Acme workspace, and which regions are mentioned? |
| 2 | What are the prerequisites to run a sample job (Python version, account, tools)? |
| 3 | What is the example CLI command to submit a first job, and what image does it use? |
| 4 | What are the main platform components (API gateway, control plane, data plane, observability) and what does each do? |
| 5 | Walk through the data flow when a client submits a job — from `POST /jobs` to job completion. |
| 6 | If the control-plane Postgres fails, what is the impact and mitigation in the failure table? |
| 7 | What does the public OpenAPI spec say about listing jobs: query parameters and max `limit`? |
| 8 | For `POST /jobs`, what are the required fields in the `JobSubmission` schema? |
| 9 | How much is the Team plan per month, and what compute hours and support level does it include? |
| 10 | What is the overage rate for compute beyond included hours, according to the pricing FAQ? |

**Quick keyword searches:** `eu-west-1`, `Prometheus`, `POST /jobs`, `Free tier`, `SEV-1` (SEV is internal—should *not* appear in public-only search).

---

## Internal lane (`data/sample_docs/internal/`)

| # | Question |
|---|----------|
| 1 | What must the hiring manager do at least 5 business days before a new employee starts? |
| 2 | What does IT provision before day 1 (hardware, email format, access)? |
| 3 | By the end of the 90-day probation, what is the employee expected to do, and who files the evaluation? |
| 4 | What default access does Engineering get in the access matrix? |
| 5 | Where must PII live, and who do you report breaches to within 24 hours? |
| 6 | Within how many minutes must the on-call acknowledge a PagerDuty page? |
| 7 | What are the four severity levels in the SEV matrix, and when do you notify exec-on-call? |
| 8 | In `DB-CPU-RECOVERY`, what SQL is used to terminate a runaway backend? |
| 9 | What are the steps in the `REGIONAL-FAILOVER` playbook, and when is it allowed? |
| 10 | What is the control-plane service version, primary region, and replica count in `eu-west-1` in the deploy config? |

**Quick keyword searches:** `post-mortem`, `DB-CPU-RECOVERY`, `Workday`, `m7i.2xlarge`, `#sre-alerts`.

---

## Tips for a short demo (under one hour)

- Ingest only the small files: e.g. `kb ingest --source src_sample_public --stage index` and `src_sample_internal` if you need internal content (or one source with both under a combined flow—this repo uses **separate** `localfs` sources in `data/source_inventory.json`).
- **Skip PDFs** in the folder for the demo, or point ingestion at a subset so you are not blocked on embedding timeouts.
- Start with **one** short **Ask** per lane, then one **search** to show BM25 + dense.
- If retrieval is empty, confirm ingest finished and the right **user context** for internal docs.

---

## Optional: Vitis HLS PDF manual

If you also index `vitis_hls_ug*.pdf`, add questions from `documentation/reference.md` or a short follow-up on HLS flow. That path is **not** required for the Acme sample-corpus demo above.
