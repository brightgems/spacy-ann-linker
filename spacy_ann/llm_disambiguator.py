# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""LLM-based entity disambiguation via an OpenAI-compatible chat API.

When ``LLMDisambiguator``
is configured, final candidate selection is delegated to a chat completion
model instead of entity-vector / doc-context cosine similarity.

Reasoning / chain-of-thought support
-------------------------------------
Some OpenAI-compatible endpoints (DeepSeek-R1, o1, QwQ, etc.) expose the
model's internal reasoning alongside the final answer. The reasoning text is
read from whichever of these response fields is present (first match wins):

    choices[0].message.reasoning_content   # DeepSeek convention
    choices[0].message.reasoning           # alternate convention

The most recent reasoning is stored on ``self.last_reasoning`` and, when
``verbose=True``, emitted via the ``spacy_ann.llm_disambiguator`` logger at
DEBUG level so it can be captured for debugging without code changes.
"""
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from .types import KnowledgeBaseCandidate

logger = logging.getLogger(__name__)

# HTTP status codes considered transient and worth retrying: rate limiting
# (429) and the common server-side failures (5xx).
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class LLMDisambiguator:
    """Disambiguate an alias mention by asking an LLM to pick a candidate.

    Given a mention, its surrounding context, and a ranked list of KB
    candidates, this builds a prompt that asks the model to reply with the
    1-based index of the most likely entity. It returns the parsed index or
    ``None`` when the call fails or the reply cannot be parsed, so callers can
    fall back to the original ranking.

    The endpoint is expected to be OpenAI-compatible (``POST /chat/completions``
    returning ``choices[0].message.content``).
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
        context_chars: int = 2000,
        temperature: float = 0.0,
        verbose: bool = False,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        """Initialize the LLMDisambiguator.

        base_url (str): Base URL of an OpenAI-compatible API, e.g.
            ``https://api.openai.com/v1``. The ``/chat/completions`` path is
            appended automatically.
        api_key (str): Bearer token for the endpoint. If empty, falls back to
            the ``OPENAI_API_KEY`` environment variable.
        model (str): Model name passed in the request payload.
        timeout (float): HTTP request timeout in seconds.
        context_chars (int): Maximum number of context characters included in
            the prompt to avoid oversized requests.
        temperature (float): Sampling temperature; 0.0 for deterministic output.
        verbose (bool): If True, log the prompt, HTTP status, raw response,
            parsed index, and any reasoning trace at DEBUG level via the
            ``spacy_ann.llm_disambiguator`` logger.
        max_retries (int): Number of retry attempts after the initial request
            on transient failures (timeouts, connection errors, and the status
            codes in ``_RETRYABLE_STATUS``). 0 disables retrying. The total
            number of attempts is ``max_retries + 1``.
        retry_backoff (float): Base seconds for exponential backoff between
            retries. The wait before attempt *n* (0-based) is
            ``retry_backoff * (2 ** n)``.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.timeout = timeout
        self.context_chars = context_chars
        self.temperature = temperature
        self.verbose = verbose
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = float(retry_backoff)
        if verbose:
            logging.getLogger(__name__).setLevel(logging.DEBUG)
        # Most recent reasoning trace (chain-of-thought), or "" if the
        # endpoint did not return one. Updated on every invoke() call.
        self.last_reasoning: str = ""
        # Most recent raw parsed JSON response, for programmatic inspection.
        self.last_response: Optional[Dict[str, Any]] = None

    def build_prompt(
        self,
        mention: str,
        label: str,
        context: str,
        candidates: List[KnowledgeBaseCandidate],
    ) -> str:
        """Build the disambiguation prompt.

        mention (str): The entity mention surface text.
        label (str): NER label of the mention span (may be empty).
        context (str): Surrounding document text.
        candidates (List[KnowledgeBaseCandidate]): Ranked candidates.

        RETURNS (str): Prompt asking the model to reply with one or more
            1-based indices (comma-separated).
        """
        ctx = context.strip()
        if len(ctx) > self.context_chars:
            ctx = ctx[: self.context_chars]
        lines = [
            f"{i}. {c.alias}" for i, c in enumerate(candidates, start=1)
        ]
        cand_block = "\n".join(lines)
        label_part = f"({label})" if label else ""
        return (
            f"上下文: {ctx}\n"
            f'提及词: "{mention}"{label_part}\n'
            f"候选实体列表:\n{cand_block}\n"
            f"请根据上下文判断，该提及词可能指向哪些实体？"
            f"可回复一个或多个编号，用逗号分隔。只回复编号。"
        )

    @staticmethod
    def _extract_reasoning(message: Dict[str, Any]) -> str:
        """Extract a reasoning/chain-of-thought trace from a chat message.

        Tries the common field names used by OpenAI-compatible reasoning
        models (DeepSeek-R1, o1, QwQ, etc.). Returns "" if none are present.

        message (Dict[str, Any]): The ``choices[0].message`` dict.

        RETURNS (str): Reasoning text, or "".
        """
        for key in ("reasoning_content", "reasoning"):
            val = message.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    def invoke(
        self,
        mention: str,
        label: str,
        context: str,
        candidates: List[KnowledgeBaseCandidate],
    ) -> Optional[int]:
        """Ask the LLM to pick the best candidate.

        RETURNS (Optional[int]): 1-based index of the chosen candidate, or
            ``None`` if the call failed or the reply could not be parsed.
            The raw response and reasoning trace are stored on
            ``self.last_response`` and ``self.last_reasoning`` regardless of
            success, for debugging.
        """
        detail = self.invoke_with_detail(mention, label, context, candidates)
        return detail["index"]

    def invoke_multi(
        self,
        mention: str,
        label: str,
        context: str,
        candidates: List[KnowledgeBaseCandidate],
    ) -> List[int]:
        """Ask the LLM to pick one or more candidates.

        Parses all numbers in the reply and returns the 1-based indices of
        the chosen candidates in reply order, deduplicated and
        range-checked. Returns an empty list on HTTP/parse failure.

        RETURNS (List[int]): 1-based indices of chosen candidates, or [].
        """
        detail = self.invoke_with_detail(mention, label, context, candidates)
        return detail.get("indices") or []

    def _post_with_retry(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """POST the chat payload, retrying on transient failures.

        Retries on ``requests.exceptions.Timeout`` /
        ``requests.exceptions.ConnectionError`` and on the HTTP status codes
        in ``_RETRYABLE_STATUS`` (429, 5xx), using exponential backoff
        (``retry_backoff * 2**attempt``). Non-retryable errors (e.g. 404, 400)
        raise immediately via ``raise_for_status``. Raises the last exception
        if all attempts are exhausted.

        url (str): Full request URL.
        headers (Dict[str, str]): Request headers.
        payload (Dict[str, Any]): JSON body.

        RETURNS (Dict[str, Any]): Parsed JSON response body.
        """
        total = self.max_retries + 1
        last_exc: Optional[Exception] = None
        for attempt in range(total):
            try:
                res = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
                logger.debug(
                    "LLM HTTP %s %s (attempt %d/%d)",
                    res.status_code, res.reason, attempt + 1, total,
                )
                if (
                    res.status_code in _RETRYABLE_STATUS
                    and attempt < self.max_retries
                ):
                    wait = self.retry_backoff * (2 ** attempt)
                    logger.debug(
                        "LLM retryable status %s; retrying in %.1fs "
                        "(attempt %d/%d)",
                        res.status_code, wait, attempt + 1, total,
                    )
                    time.sleep(wait)
                    continue
                res.raise_for_status()
                return res.json()
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = self.retry_backoff * (2 ** attempt)
                    logger.debug(
                        "LLM %s; retrying in %.1fs (attempt %d/%d)",
                        type(exc).__name__, wait, attempt + 1, total,
                    )
                    time.sleep(wait)
                    continue
                raise
        # All retryable attempts exhausted via status-code path: surface the
        # last error so the caller can record it.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            "LLM request exhausted retries without a response"
        )

    def invoke_with_detail(
        self,
        mention: str,
        label: str,
        context: str,
        candidates: List[KnowledgeBaseCandidate],
    ) -> Dict[str, Any]:
        """Ask the LLM and return full detail for debugging.

        RETURNS (Dict[str, Any]): Dict with keys:
            index (Optional[int]): 1-based chosen index, or None.
            content (str): Raw message content returned by the model.
            reasoning (str): Reasoning/chain-of-thought trace, or "".
            raw (Optional[Dict]): Full parsed JSON response, or None on
                HTTP/JSON failure.
            prompt (str): The prompt that was sent.
            error (Optional[str]): Exception string on failure, else None.
        """
        result: Dict[str, Any] = {
            "index": None,
            "indices": [],
            "content": "",
            "reasoning": "",
            "raw": None,
            "prompt": "",
            "error": None,
        }
        if not candidates:
            return result
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            prompt = self.build_prompt(mention, label, context, candidates)
            result["prompt"] = prompt
            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            logger.debug("LLM prompt:\n%s", prompt)
            data = self._post_with_retry(url, headers, payload)
        except Exception as exc:
            result["error"] = str(exc)
            self.last_response = None
            self.last_reasoning = ""
            logger.debug("LLM request failed: %s", exc)
            return result

        result["raw"] = data
        self.last_response = data
        try:
            message = data["choices"][0]["message"]
            content = (message.get("content") or "").strip()
            reasoning = self._extract_reasoning(message)
        except (KeyError, IndexError, TypeError) as exc:
            result["error"] = f"unexpected response shape: {exc}"
            logger.debug("LLM unexpected response shape: %s", exc)
            return result

        result["content"] = content
        result["reasoning"] = reasoning
        self.last_reasoning = reasoning
        logger.debug("LLM content: %r", content)
        if reasoning:
            logger.debug("LLM reasoning:\n%s", reasoning)

        numbers = re.findall(r"\d+", content)
        indices: List[int] = []
        for tok in numbers:
            idx = int(tok)
            if 1 <= idx <= len(candidates) and idx not in indices:
                indices.append(idx)
        if indices:
            result["indices"] = indices
            result["index"] = indices[0]
            logger.debug(
                "LLM parsed indices: %s -> %s",
                indices, [candidates[i - 1].alias for i in indices])
        else:
            logger.debug("LLM content had no parseable index")
        return result
