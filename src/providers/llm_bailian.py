from __future__ import annotations

from typing import Any

from util.prompt_logger import log_event


def _extract_content(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("text", "content", "result", "output"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                nested = _extract_content(value)
                if nested:
                    return nested
        if "choices" in payload and isinstance(payload["choices"], list):
            for choice in payload["choices"]:
                if isinstance(choice, dict):
                    message = choice.get("message") or choice.get("delta")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str) and content.strip():
                            return content.strip()
                    text = choice.get("text")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
    if hasattr(payload, "output"):
        return _extract_content(getattr(payload, "output"))
    if hasattr(payload, "choices"):
        return _extract_content({"choices": getattr(payload, "choices")})
    return ""


def _extract_status(payload: Any) -> tuple[int | None, str | None]:
    status = None
    message = None
    if hasattr(payload, "status_code"):
        try:
            status = int(getattr(payload, "status_code"))
        except Exception:
            status = None
    if hasattr(payload, "message"):
        message = str(getattr(payload, "message"))
    if isinstance(payload, dict):
        if status is None and "status_code" in payload:
            try:
                status = int(payload.get("status_code"))
            except Exception:
                status = None
        if message is None and "message" in payload:
            message = str(payload.get("message"))
    return status, message


def generate_answer(
    *,
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
) -> str:
    try:
        import dashscope
    except Exception as exc:
        raise RuntimeError(f"dashscope not available: {exc}") from exc

    dashscope.api_key = api_key

    response = None
    request_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    try:
        response = dashscope.Generation.call(
            model=model,
            messages=messages,
            temperature=temperature,
            result_format="message",
        )
    except TypeError:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        response = dashscope.Generation.call(
            model=model,
            prompt=prompt,
            temperature=temperature,
        )

    status, message = _extract_status(response)
    if status is not None and status >= 400:
        raise RuntimeError(f"LLM call failed: {status} {message or ''}")

    content = _extract_content(response)
    if not content and hasattr(response, "output"):
        content = _extract_content(response.output)
    log_event(
        "llm",
        request=request_payload,
        response={
            "raw": str(response),
            "content": content,
        },
    )
    return content.strip()
