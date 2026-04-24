"""
Lane-aware LLM client.

Routes a completion request to a provider based on the chunk's sensitivity:
    * hosted_ok          → uses HOSTED_LANE_PRIORITY
    * self_hosted_only   → uses SELFHOSTED_LANE_PRIORITY

Each lane priority is an ordered list of (provider, model) pairs. The client
tries them in order and falls back on the next when one fails, so a hiccup
in one provider doesn't stall ingestion.

Providers:
    * openai   → OpenAI REST (chat.completions)
    * qwen     → DashScope OpenAI-compatible endpoint
    * hf       → Hugging Face text-generation (used for a dedicated endpoint
                 in prod / demo-isolated; not normally used in the demo profile)

Safety rail: when APP_PROFILE=prod AND a self_hosted_only chunk is routed to
a hosted provider, we raise rather than leaking. The settings model already
refuses to start in that combination with the override flag on; this is
defence in depth.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from kb.settings import Profile, Settings, get_settings
from kb.types import SensitivityLane


logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Raised when every provider on a lane failed."""


@dataclass
class CompletionResult:
    text: str
    provider: str
    model: str
    latency_ms: int


class LLMClient:
    """Synchronous, thread-safe LLM client. Safe to share across workers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._openai_client = None
        self._qwen_client = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def complete(
        self,
        *,
        prompt: str,
        lane: SensitivityLane,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> CompletionResult:
        providers = self._providers_for(lane)
        errors: list[str] = []

        for provider, model in providers:
            try:
                t0 = time.monotonic()
                text = self._call(
                    provider=provider,
                    model=model,
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_mode=json_mode,
                )
                latency_ms = int((time.monotonic() - t0) * 1000)
                return CompletionResult(
                    text=text, provider=provider, model=model, latency_ms=latency_ms
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"{provider}:{model} failed: {exc}"
                logger.warning(msg)
                errors.append(msg)
                continue

        raise LLMClientError(
            f"All providers failed for lane={lane.value}. Attempts: {errors}"
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _providers_for(self, lane: SensitivityLane) -> list[tuple[str, str]]:
        if lane == SensitivityLane.HOSTED_OK:
            return self.settings.hosted_lane_providers

        # self_hosted_only lane
        if self.settings.app_profile == Profile.PROD:
            # Belt and braces — the settings model already refuses the dangerous
            # combo at startup, but verify once more at call time.
            providers = self.settings.selfhosted_lane_providers
            for p, _ in providers:
                if p in {"openai", "qwen"}:
                    raise LLMClientError(
                        f"Refusing to route self_hosted_only content to hosted "
                        f"provider {p!r} in prod profile"
                    )
            return providers

        return self.settings.selfhosted_lane_providers

    def _call(
        self,
        *,
        provider: str,
        model: str,
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        if provider == "openai":
            return self._call_openai_compat(
                client=self._get_openai_client(),
                model=model,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
            )
        if provider == "qwen":
            return self._call_openai_compat(
                client=self._get_qwen_client(),
                model=model,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
            )
        if provider == "hf":
            return self._call_hf_endpoint(
                model=model,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        raise LLMClientError(f"Unknown provider {provider!r}")

    # ---------------- Provider: OpenAI & Qwen (OpenAI-compatible) ---------------- #

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            if not self.settings.openai_api_key:
                raise LLMClientError("OPENAI_API_KEY is not configured")
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)
        return self._openai_client

    def _get_qwen_client(self):
        if self._qwen_client is None:
            from openai import OpenAI
            if not self.settings.qwen_api_key:
                raise LLMClientError("QWEN_API_KEY is not configured")
            self._qwen_client = OpenAI(
                api_key=self.settings.qwen_api_key,
                base_url=self.settings.qwen_api_base,
            )
        return self._qwen_client

    @staticmethod
    def _call_openai_compat(
        *,
        client: Any,
        model: str,
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    # ---------------- Provider: HF dedicated endpoint ---------------- #

    def _call_hf_endpoint(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        import requests

        endpoint = self.settings.hf_llm_selfhosted_endpoint_url
        if not endpoint:
            raise LLMClientError(
                "HF_LLM_SELFHOSTED_ENDPOINT_URL is not configured; "
                "cannot route to the self-hosted HF lane"
            )

        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        headers = {
            "Authorization": f"Bearer {self.settings.hf_api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 0.01),  # HF rejects 0.0
                "return_full_text": False,
            },
        }
        r = requests.post(
            endpoint, headers=headers, json=payload,
            timeout=self.settings.hf_llm_selfhosted_cold_start_timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        raise LLMClientError(f"Unexpected HF endpoint response: {data!r:.200}")
