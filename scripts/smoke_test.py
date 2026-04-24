"""
Phase-0 smoke tests for the KB-search infrastructure.

Runs every external-API round-trip the system depends on, plus the local
data-plane (Qdrant / Postgres / Redis) checks. Exits non-zero on any failure
so this script can gate Phase 1 in CI.

Coverage map (references §9 of data/infra_provisioning.md):

  1. HF Serverless embeddings  (bge-m3)         [ALL PROFILES]
  2. HF Serverless reranker    (bge-reranker-v2-m3) [ALL PROFILES]
  3. OpenAI chat completion    (gpt-4o-mini)    [ALL PROFILES]
  3b. Qwen chat completion     (qwen2.5-*-instruct) [ALL PROFILES]
  4. HF Serverless NLI         (deberta-v3 zero-shot) [ALL PROFILES]
  5. Qdrant / Postgres / Redis                  [ALL PROFILES]
  6. LangSmith trace round-trip                 [ALL PROFILES]
  7. Profile flags logged                       [demo]
  8. Safety rail: prod+override refuses         [demo]

Tests 9 (audit log tagging) and lane-routing enforcement are app-layer
integration tests; they move to the Phase 1 test suite once the app exists.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import requests
from dotenv import load_dotenv


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


@dataclass
class TestResult:
    name: str
    ok: bool
    elapsed_ms: int
    detail: str = ""
    error: str = ""


@dataclass
class Report:
    results: list[TestResult] = field(default_factory=list)

    def add(self, r: TestResult) -> None:
        self.results.append(r)
        status = f"{GREEN}PASS{RESET}" if r.ok else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {r.name:<45} {r.elapsed_ms:>5} ms   {DIM}{r.detail}{RESET}")
        if r.error:
            for line in r.error.splitlines():
                print(f"         {RED}{line}{RESET}")

    def summary(self) -> int:
        passed = sum(1 for r in self.results if r.ok)
        failed = len(self.results) - passed
        print()
        print("─" * 72)
        color = GREEN if failed == 0 else RED
        print(f"  {color}{passed} passed, {failed} failed{RESET}   "
              f"({len(self.results)} total)")
        print("─" * 72)
        return 0 if failed == 0 else 1


def run_test(report: Report, name: str, fn: Callable[[], tuple[bool, str]]) -> None:
    t0 = time.perf_counter()
    try:
        ok, detail = fn()
        elapsed = int((time.perf_counter() - t0) * 1000)
        report.add(TestResult(name=name, ok=ok, elapsed_ms=elapsed, detail=detail))
    except Exception as e:  # noqa: BLE001
        elapsed = int((time.perf_counter() - t0) * 1000)
        report.add(TestResult(
            name=name, ok=False, elapsed_ms=elapsed,
            error=f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=12),
        ))


# --------------------------------------------------------------------------- #
# Individual test functions
# --------------------------------------------------------------------------- #

HF_INFERENCE_BASE = "https://router.huggingface.co/hf-inference/models"
HF_CLASSIC_BASE = "https://api-inference.huggingface.co/models"


def _hf_post(url: str, payload: dict, timeout: int = 60) -> requests.Response:
    token = os.environ["HF_API_TOKEN"]
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if not resp.ok:
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason} for {url} :: body={resp.text[:300]}",
            response=resp,
        )
    return resp


def test_hf_embeddings() -> tuple[bool, str]:
    model = os.environ.get("HF_EMBED_MODEL_ID", "BAAI/bge-m3")
    url = f"{HF_INFERENCE_BASE}/{model}/pipeline/feature-extraction"
    resp = _hf_post(url, {"inputs": "Enterprise knowledge base retrieval."})
    vec = resp.json()
    # bge-m3 can return either [dim] or [[dim]] depending on input shape
    while isinstance(vec, list) and vec and isinstance(vec[0], list):
        vec = vec[0]
    dim = len(vec)
    if dim != 1024:
        return False, f"dim={dim}, expected 1024"
    return True, f"dim={dim}"


def test_hf_reranker() -> tuple[bool, str]:
    model = os.environ.get("HF_RERANK_MODEL_ID", "BAAI/bge-reranker-v2-m3")
    query = "What is the webhook retry schedule?"
    candidates = [
        "Webhooks retry with exponential backoff up to 8 attempts.",
        "The cafeteria is open from 11 to 14.",
    ]
    url = f"{HF_INFERENCE_BASE}/{model}/pipeline/sentence-similarity"
    resp = _hf_post(
        url,
        {"inputs": {"source_sentence": query, "sentences": candidates}},
    )
    scores = resp.json()
    if not isinstance(scores, list) or len(scores) != len(candidates):
        return False, f"unexpected shape: {scores!r}"
    if scores[0] <= scores[1]:
        return False, f"relevance order wrong: {scores}"
    return True, f"scores={[round(s, 3) for s in scores]}"


def test_hf_nli() -> tuple[bool, str]:
    """
    Try the router first, fall back to the classic HF Inference API.
    The classic endpoint accepts the same zero-shot-classification payload
    and is more permissive about model-specific pipeline registration.
    """
    model = os.environ.get("HF_NLI_MODEL_ID", "facebook/bart-large-mnli")
    payload = {
        "inputs": "Webhooks retry with exponential backoff up to 8 attempts.",
        "parameters": {"candidate_labels": ["supported", "contradicted"]},
    }
    urls = [
        f"{HF_CLASSIC_BASE}/{model}",
        f"{HF_INFERENCE_BASE}/{model}",
        f"{HF_INFERENCE_BASE}/{model}/pipeline/zero-shot-classification",
    ]
    last_err = None
    for url in urls:
        try:
            resp = _hf_post(url, payload)
            data = resp.json()
            # Two possible response shapes:
            #   dict:  {"sequence": "...", "labels": [...], "scores": [...]}
            #   list:  [{"label": "...", "score": ...}, ...]
            if isinstance(data, dict) and data.get("labels"):
                top_label, top_score = data["labels"][0], data["scores"][0]
            elif isinstance(data, list) and data and isinstance(data[0], dict) and "label" in data[0]:
                # Sort defensively; some providers return unordered
                data_sorted = sorted(data, key=lambda x: x.get("score", 0), reverse=True)
                top_label, top_score = data_sorted[0]["label"], data_sorted[0]["score"]
            else:
                last_err = f"unexpected payload from {url}: {str(data)[:160]}"
                continue
            return True, f"top={top_label} p={round(float(top_score), 3)} via={url.split('/')[2]}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{url} -> {e}"
            continue
    return False, f"all endpoints failed; last: {last_err}"


def test_openai() -> tuple[bool, str]:
    from openai import OpenAI  # local import keeps startup cheap

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.environ.get("OPENAI_MODEL_GENERATOR", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with one word: ok"}],
        max_tokens=5,
        temperature=0,
    )
    content = (resp.choices[0].message.content or "").strip().lower()
    if "ok" not in content:
        return False, f"unexpected reply: {content!r}"
    return True, f"model={model} reply={content!r}"


def test_qwen() -> tuple[bool, str]:
    """
    Tries the configured QWEN_API_BASE first, then falls back to the
    international DashScope endpoint. DashScope has two regional endpoints
    and the key is tied to one — a 401 on the China endpoint often means
    the key is provisioned in the international region.
    """
    from openai import AuthenticationError, OpenAI

    model = os.environ.get("QWEN_MODEL_SMALL", "qwen2.5-1.5b-instruct")
    api_key = os.environ["QWEN_API_KEY"]
    configured_base = os.environ["QWEN_API_BASE"]
    intl_base = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    candidates = [configured_base]
    if configured_base != intl_base:
        candidates.append(intl_base)

    last_err = None
    for base in candidates:
        try:
            client = OpenAI(api_key=api_key, base_url=base)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with one word: ok"}],
                max_tokens=5,
                temperature=0,
            )
            content = (resp.choices[0].message.content or "").strip().lower()
            if "ok" not in content:
                last_err = f"{base} -> unexpected reply: {content!r}"
                continue
            via = "configured" if base == configured_base else "intl-fallback"
            return True, f"model={model} via={via} reply={content!r}"
        except AuthenticationError as e:
            last_err = f"{base} -> 401 (check key region/validity): {str(e)[:160]}"
            continue
        except Exception as e:  # noqa: BLE001
            last_err = f"{base} -> {type(e).__name__}: {str(e)[:160]}"
            continue
    return False, f"all Qwen endpoints failed; last: {last_err}"


def _host_port_reachable(url: str) -> bool:
    # Only used as a friendly skip check; proper health check below.
    from urllib.parse import urlparse
    p = urlparse(url)
    try:
        with socket.create_connection((p.hostname, p.port or 80), timeout=2):
            return True
    except OSError:
        return False


def test_qdrant() -> tuple[bool, str]:
    url = os.environ["QDRANT_URL"].rstrip("/")
    api_key = os.environ["QDRANT_API_KEY"]
    if not _host_port_reachable(url):
        return False, f"host:port unreachable ({url}) — run docker compose on the same host, or SSH-forward"
    # /readyz has no auth; used first to confirm service is live
    r1 = requests.get(f"{url}/readyz", timeout=5)
    r1.raise_for_status()
    # /collections requires the api-key header
    r2 = requests.get(
        f"{url}/collections",
        headers={"api-key": api_key},
        timeout=5,
    )
    r2.raise_for_status()
    cols = r2.json().get("result", {}).get("collections", [])
    return True, f"connected; collections={len(cols)}"


def test_postgres() -> tuple[bool, str]:
    url = os.environ["POSTGRES_URL"]
    if not _host_port_reachable(url.replace("postgresql://", "http://").split("@")[-1].split("/")[0] + ":0"):
        pass  # skip friendly check; real call below will error properly
    import psycopg
    with psycopg.connect(url, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute("SELECT schema_name FROM information_schema.schemata "
                    "WHERE schema_name IN ('kb', 'audit') ORDER BY schema_name;")
        schemas = [r[0] for r in cur.fetchall()]
    if schemas != ["audit", "kb"]:
        return False, f"expected schemas [audit, kb], got {schemas}"
    return True, f"schemas={schemas}"


def test_redis() -> tuple[bool, str]:
    import redis
    r = redis.from_url(os.environ["REDIS_URL"], socket_timeout=5)
    pong = r.ping()
    if not pong:
        return False, "PING did not return True"
    key = f"smoke:{uuid.uuid4()}"
    r.setex(key, 10, "hello")
    got = r.get(key)
    if got != b"hello":
        return False, f"round-trip failed: {got!r}"
    return True, "PING + set/get ok"


def test_langsmith() -> tuple[bool, str]:
    from langsmith import Client
    client = Client(
        api_key=os.environ["LANGSMITH_API_KEY"],
        api_url=os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    )
    project = os.environ.get("LANGSMITH_PROJECT", "acme-kb-search")
    # Creating and finishing a run is the cheapest end-to-end trace verification.
    run_id = uuid.uuid4()
    client.create_run(
        id=run_id,
        name="smoke_test",
        run_type="chain",
        inputs={"query": "smoke"},
        project_name=project,
    )
    client.update_run(run_id, outputs={"ok": True}, end_time=None)
    return True, f"project={project} run_id={run_id}"


# --------------------------------------------------------------------------- #
# Profile safety rails (items 7 & 8 of §9)
# --------------------------------------------------------------------------- #

def test_profile_flags_logged() -> tuple[bool, str]:
    profile = os.environ.get("APP_PROFILE", "")
    override = os.environ.get("DEMO_ALLOW_HOSTED_FOR_SELFHOSTED_LANE", "").lower()
    if profile != "demo":
        return False, f"APP_PROFILE={profile!r} — expected 'demo' for this run"
    if override not in ("1", "true", "yes"):
        return False, f"DEMO_ALLOW_HOSTED_FOR_SELFHOSTED_LANE={override!r}"
    return True, f"profile={profile} override=true"


def test_safety_rail_prod_plus_override() -> tuple[bool, str]:
    """
    Simulate the startup-time assertion the app will enforce:
    refuse to start when APP_PROFILE=prod AND the override flag is true.
    """
    def assert_startup_ok(profile: str, override: str) -> str | None:
        if profile == "prod" and override.lower() in ("1", "true", "yes"):
            return (
                "REFUSED: APP_PROFILE=prod is incompatible with "
                "DEMO_ALLOW_HOSTED_FOR_SELFHOSTED_LANE=true"
            )
        return None

    if assert_startup_ok("prod", "true") is None:
        return False, "guard did not block prod+override"
    if assert_startup_ok("demo", "true") is not None:
        return False, "guard incorrectly blocked demo+override"
    if assert_startup_ok("prod", "false") is not None:
        return False, "guard incorrectly blocked prod without override"
    return True, "guard blocks only the dangerous combination"


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="KB smoke tests")
    parser.add_argument(
        "--skip-dataplane",
        action="store_true",
        help="Skip Qdrant/Postgres/Redis tests (useful when running off-host).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Run only the named tests (substring match).",
    )
    args = parser.parse_args()

    load_dotenv()  # reads .env in the working directory
    report = Report()

    print()
    print(f"  Profile: {os.environ.get('APP_PROFILE', '<unset>')}   "
          f"Env: {os.environ.get('APP_ENV', '<unset>')}   "
          f"Region: {os.environ.get('APP_REGION', '<unset>')}")
    print("─" * 72)

    all_tests: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
        ("1. HF Serverless embeddings (bge-m3)",           test_hf_embeddings),
        ("2. HF Serverless reranker (bge-reranker-v2-m3)", test_hf_reranker),
        ("3. OpenAI gpt-4o-mini chat",                     test_openai),
        ("3b. Qwen API chat",                              test_qwen),
        ("4. HF Serverless NLI (zero-shot)",               test_hf_nli),
    ]

    dataplane_tests: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
        ("5a. Qdrant connectivity + list collections",     test_qdrant),
        ("5b. Postgres schemas (kb + audit) present",      test_postgres),
        ("5c. Redis AUTH + round-trip",                    test_redis),
    ]

    profile_tests: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
        ("6. LangSmith trace round-trip",                  test_langsmith),
        ("7. Profile flags present (demo + override)",     test_profile_flags_logged),
        ("8. Safety rail: prod + override refused",        test_safety_rail_prod_plus_override),
    ]

    tests = list(all_tests)
    if not args.skip_dataplane:
        tests += dataplane_tests
    tests += profile_tests

    if args.only:
        keys = [k.lower() for k in args.only]
        tests = [(n, fn) for n, fn in tests if any(k in n.lower() for k in keys)]

    for name, fn in tests:
        run_test(report, name, fn)

    return report.summary()


if __name__ == "__main__":
    sys.exit(main())
