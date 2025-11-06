import json
import logging
from typing import Any, Dict, List, Optional, cast

from app.core.settings import settings

logger = logging.getLogger(__name__)


class LLMProviderError(RuntimeError):
    """Raised when the LLM provider call fails or returns invalid data."""


_OPENAI_CLIENT: Any = None


def _get_openai_client() -> Any:
    if not settings.LLM_API_KEY:
        raise LLMProviderError("Missing LLM_API_KEY configuration")
    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LLMProviderError("The 'openai' package is required for OpenAI provider") from exc

    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        kwargs: Dict[str, Any] = {"api_key": settings.LLM_API_KEY}
        if settings.LLM_BASE_URL:
            kwargs["base_url"] = settings.LLM_BASE_URL
        _OPENAI_CLIENT = OpenAI(**kwargs)
    return _OPENAI_CLIENT


def _call_openai(prompt: str) -> str:
    client = _get_openai_client()
    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a movie specialist that always responds with compact JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise LLMProviderError(f"LLM provider call failed: {exc}") from exc

    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError) as exc:
        raise LLMProviderError("LLM provider returned an unexpected payload") from exc


def _call_gemini(prompt: str) -> str:
    if not settings.LLM_API_KEY:
        raise LLMProviderError("Missing LLM_API_KEY configuration")
    try:
        import google.generativeai as genai  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LLMProviderError("The 'google-generativeai' package is required for Gemini provider") from exc

    genai.configure(api_key=settings.LLM_API_KEY)
    model_name = settings.LLM_MODEL or "gemini-2.5-flash"
    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "candidate_count": 1,
                "response_mime_type": "application/json",
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise LLMProviderError(f"LLM provider call failed: {exc}") from exc

    text = getattr(response, "text", None)
    if text:
        return text

    candidates = getattr(response, "candidates", None)
    if candidates:
        parts: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content and getattr(content, "parts", None):
                for part in content.parts:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        parts.append(part_text)
        if parts:
            return "".join(parts)

    raise LLMProviderError("LLM provider returned an empty payload")


def _call_llm(prompt: str) -> str:
    provider = settings.LLM_PROVIDER.lower()
    if provider in {"openai", "azure_openai"}:
        return _call_openai(prompt)
    if provider in {"google", "gemini", "google_gemini", "google-ai"}:
        return _call_gemini(prompt)
    raise LLMProviderError(f"Unsupported LLM provider '{settings.LLM_PROVIDER}'")


def _strip_json_markers(payload: str) -> str:
    text = payload.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _build_prompt(query: str, filters: Optional[Dict[str, Any]], limit: int) -> str:
    filter_clause = ""
    if filters:
        serialized_filters = json.dumps(filters, ensure_ascii=False)
        filter_clause = f"\nApply these optional filters when relevant: {serialized_filters}."
    return (
        f"You recommend movies based on a catalog."
        f"\nReturn a JSON array with at most {limit} objects."
        " Each object must have the fields 'movie_id', 'title', and 'rationale'."
        " The 'rationale' should briefly justify why the movie fits the request."
        f"\nUser request: {query}."
        f"\nLimit results to {limit} items." + filter_clause
    )


def _sanitize_items(items: Any, limit: int) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        raise LLMProviderError("LLM response must be a JSON array")

    sanitized: List[Dict[str, Any]] = []
    for entry in items:
        if not isinstance(entry, dict):
            logger.debug("Skipping non-dict entry in LLM response: %r", entry)
            continue
        movie_id = entry.get("movie_id")
        title = entry.get("title")
        rationale = entry.get("rationale")
        if not all(isinstance(value, str) and value.strip() for value in (movie_id, title, rationale)):
            logger.debug("Skipping incomplete entry in LLM response: %r", entry)
            continue
        movie_id_str = cast(str, movie_id).strip()
        title_str = cast(str, title).strip()
        rationale_str = cast(str, rationale).strip()
        sanitized.append(
            {
                "movie_id": movie_id_str,
                "title": title_str,
                "rationale": rationale_str,
            }
        )
        if len(sanitized) >= limit:
            break

    if not sanitized:
        raise LLMProviderError("LLM response did not contain any valid movie entries")
    return sanitized


def search_movies(query: str, *, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    max_allowed = max(1, settings.LLM_MAX_RESULTS)
    if limit is None:
        effective_limit = max_allowed
    else:
        effective_limit = max(1, min(limit, max_allowed))

    prompt = _build_prompt(query.strip(), filters, effective_limit)
    content = _call_llm(prompt)

    cleaned = _strip_json_markers(content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMProviderError(f"Failed to parse LLM JSON response: {exc}") from exc

    return _sanitize_items(parsed, effective_limit)


__all__ = ["search_movies", "LLMProviderError"]
