"""
FastAPI app — static SPA + JSON API. Bind to 127.0.0.1 in production
demos; add auth if you ever expose the network.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from kb.generation.types import GenerationConfig
from kb.guardrails import QueryGuardError
from kb.web.deps import get_generator, get_retriever, get_settings_cached
from kb.web.models import AskRequest, ConfigResponse, HealthResponse, SearchRequest
from kb.web.users_config import build_user, load_user_options

logger = logging.getLogger(__name__)

_STATIC = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Knowledge Base",
        description="Acme RAG — hybrid search and grounded Q&A",
        version="0.1.0",
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        s: dict[str, str] = {}
        try:
            from kb.indexing import PostgresWriter, QdrantWriter
            from kb.indexing.bm25_writer import BM25Writer

            s["postgres"] = "ok" if PostgresWriter().health() else "down"
            s["qdrant"] = "ok" if QdrantWriter().health() else "down"
            try:
                BM25Writer().stats("__ui_health__")
                s["bm25"] = "ok"
            except Exception:  # noqa: BLE001
                s["bm25"] = "down"
        except Exception as e:  # noqa: BLE001
            s["data_plane"] = f"err:{e!s}"[:50]
        try:
            from kb.sessions import RedisSessionStore
            s["redis"] = "ok" if RedisSessionStore().ping() else "down"
        except Exception:  # noqa: BLE001
            s["redis"] = "down"
        ok = s.get("postgres") == s.get("qdrant") == s.get("bm25") == "ok"
        return HealthResponse(ok=ok, services=s)

    @app.get("/api/session/new")
    def new_session() -> JSONResponse:
        import uuid
        return JSONResponse({"session_id": uuid.uuid4().hex})

    @app.get("/api/config", response_model=ConfigResponse)
    def config() -> ConfigResponse:
        st = get_settings_cached()
        return ConfigResponse(
            app_profile=str(st.app_profile),
            users=load_user_options(),
        )

    @app.post("/api/search")
    def run_search(req: SearchRequest) -> JSONResponse:
        from kb.retrieval.types import RetrievalConfig

        try:
            user = build_user(req.user)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, str(e)) from e
        rcfg = RetrievalConfig(
            top_k_dense=req.top_k_dense,
            top_k_sparse=req.top_k_sparse,
            top_k_final=req.top_k,
            rrf_dense_weight=req.rrf_dense_weight,
            rrf_sparse_weight=req.rrf_sparse_weight,
            rerank=req.rerank,
            rerank_top_n=req.rerank_top_n,
            rewrite_strategy=req.rewrite,  # type: ignore[arg-type]
            multi_query_k=req.multi_query_k,
            include_parent_content=req.include_parent_content,
        )
        try:
            r = get_retriever().retrieve(
                (req.query or "").strip(), user=user, config=rcfg,
            )
        except QueryGuardError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "query_guard",
                    "reason": e.result.reason,
                    "message": e.result.user_message,
                },
            ) from e
        return JSONResponse(content=json.loads(r.model_dump_json()))

    @app.post("/api/ask")
    def run_ask(req: AskRequest) -> JSONResponse:
        from kb.retrieval.types import RetrievalConfig

        try:
            user = build_user(req.user)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, str(e)) from e

        stt = get_settings_cached()
        rcfg = RetrievalConfig(
            top_k_final=req.top_k,
            rerank=req.rerank,
            rewrite_strategy=req.rewrite,  # type: ignore[arg-type]
            multi_query_k=req.multi_query_k,
            stepback=req.stepback,
        )
        gcfg = GenerationConfig(
            stream=False,
            max_answer_tokens=req.max_answer_tokens
            or stt.generation_max_answer_tokens,
            temperature=req.temperature
            if req.temperature is not None
            else stt.generation_temperature,
            check_faithfulness=req.check_faithfulness
            if req.check_faithfulness is not None
            else stt.generation_check_faithfulness,
            faithfulness_threshold=stt.generation_faithfulness_threshold,
        )
        gen = get_generator()
        try:
            res = gen.ask(
                (req.query or "").strip(), user=user,
                retrieval_config=rcfg, generation_config=gcfg,
                session_id=req.session_id,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("ask failed: %s", e)
            raise HTTPException(500, str(e)) from e
        return JSONResponse(content=json.loads(res.model_dump_json()))

    @app.post("/api/ask/stream")
    def run_ask_stream(req: AskRequest) -> StreamingResponse:
        from kb.retrieval.types import RetrievalConfig

        if not (req.query or "").strip():
            raise HTTPException(400, "empty query")

        try:
            user = build_user(req.user)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, str(e)) from e

        stt = get_settings_cached()
        rcfg = RetrievalConfig(
            top_k_final=req.top_k,
            rerank=req.rerank,
            rewrite_strategy=req.rewrite,  # type: ignore[arg-type]
            multi_query_k=req.multi_query_k,
            stepback=req.stepback,
        )
        gcfg = GenerationConfig(
            stream=True,
            max_answer_tokens=req.max_answer_tokens
            or stt.generation_max_answer_tokens,
            temperature=req.temperature
            if req.temperature is not None
            else stt.generation_temperature,
            check_faithfulness=req.check_faithfulness
            if req.check_faithfulness is not None
            else stt.generation_check_faithfulness,
            faithfulness_threshold=stt.generation_faithfulness_threshold,
        )
        gen = get_generator()

        def sse():
            try:
                for ev in gen.ask_stream(
                    (req.query or "").strip(), user=user,
                    retrieval_config=rcfg, generation_config=gcfg,
                    session_id=req.session_id,
                ):
                    yield f"data: {ev.model_dump_json()}\n\n"
            except Exception as e:  # noqa: BLE001
                err = json.dumps(
                    {"kind": "error", "message": str(e)},
                )
                yield f"data: {err}\n\n"

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/")
    def index() -> FileResponse:
        p = _STATIC / "index.html"
        if not p.exists():
            return JSONResponse(
                status_code=500,
                content={"error": f"Missing {p!s} — is static built?"},
            )
        return FileResponse(p, media_type="text/html; charset=utf-8")

    if _STATIC.is_dir() and any(_STATIC.iterdir()):
        app.mount(
            "/static", StaticFiles(directory=str(_STATIC)), name="static",
        )

    return app


# Uvicorn entry: ``uvicorn kb.web.app:app``
app = create_app()
