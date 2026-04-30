"""
Model API backends for WildTableBench evaluation.

Supported backends
------------------
- OpenRouter  : 200+ models with a single API key — recommended for most users
                https://openrouter.ai  →  set OPENROUTER_API_KEY
- OpenAI      : GPT-4o, o3, etc. via native OpenAI SDK
                https://platform.openai.com/api-keys  →  set OPENAI_API_KEY
- Gemini      : Gemini models via Google GenAI SDK
                https://aistudio.google.com/apikey  →  set GEMINI_API_KEY
- Local server: any OpenAI-compatible endpoint (vLLM, Ollama, LMStudio, SGLang)
                set LOCAL_MODEL_BASE_URL and LOCAL_MODEL_NAME

Quick-start examples
--------------------

  # OpenAI (GPT-4o)
  export OPENAI_API_KEY=sk-...
  python evaluate.py --data hf_wild/metadata.csv --models gpt4o

  # Gemini
  export GEMINI_API_KEY=AIza...
  python evaluate.py --data hf_wild/metadata.csv --models gemini_2_flash

  # OpenRouter — any model ID, single key
  export OPENROUTER_API_KEY=sk-or-...
  python evaluate.py --data hf_wild/metadata.csv --models openai/gpt-4o google/gemini-2.0-flash-001
  # Full model list: https://openrouter.ai/models

  # Local vLLM server
  vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000
  export LOCAL_MODEL_BASE_URL=http://localhost:8000
  export LOCAL_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
  python evaluate.py --data hf_wild/metadata.csv --models local

  # Local Ollama server
  ollama serve && ollama pull llava:13b
  export LOCAL_MODEL_BASE_URL=http://localhost:11434
  export LOCAL_MODEL_NAME=llava:13b
  python evaluate.py --data hf_wild/metadata.csv --models local
"""

import base64
import os
import time
from typing import List, Optional

import requests


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _mime_from_base64(img_b64: str) -> str:
    """Detect image MIME type from base64 magic bytes. Defaults to image/jpeg."""
    try:
        raw = base64.b64decode(img_b64[:100], validate=True)
        if raw[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if raw[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if raw[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            return "image/webp"
    except Exception:
        pass
    return "image/jpeg"


def _build_image_content(question: str, image_data: List[str] = None) -> list:
    """Build OpenAI-format content list (images first, then text)."""
    content = []
    if image_data:
        for img_b64 in image_data:
            if img_b64:
                mime = _mime_from_base64(img_b64)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                })
    content.append({"type": "text", "text": question})
    return content


def _parse_usage(usage: dict) -> Optional[dict]:
    """Parse token usage from API response (handles OpenAI / Gemini / OpenRouter formats)."""
    if not usage:
        return None
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return None
    result = {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
    }
    details = (usage.get("completion_tokens_details")
               or usage.get("completion_tokens_breakdown")
               or {})
    reasoning = (details.get("reasoning_tokens")
                 or details.get("reasoning")
                 or usage.get("reasoning_tokens")
                 or usage.get("input_tokens_details", {}).get("reasoning_tokens"))
    if reasoning is not None:
        result["thought_token_count"] = int(reasoning)
    return result


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def run_openrouter_model(
    model_id: str,
    question: str,
    image_data: List[str] = None,
    system_prompt: str = None,
    max_tokens: int = 16384,
) -> dict:
    """Run any model via OpenRouter.

    Args:
        model_id:   OpenRouter model ID, e.g. "openai/gpt-4o"
        question:   Prompt text
        image_data: List of base64-encoded images
    Returns:
        dict {"content": str, "usage": dict | None}  or  "Error: ..." string
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY not set"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": _build_image_content(question, image_data)})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model_id, "messages": messages, "max_tokens": max_tokens}

    last_error = None
    for attempt in range(5):
        try:
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if response.status_code == 200:
                result = response.json()
                choice = (result.get("choices") or [{}])[0]
                content_str = choice.get("message", {}).get("content") or ""
                finish = choice.get("finish_reason", "")
                if content_str.strip():
                    return {"content": content_str, "usage": _parse_usage(result.get("usage") or {})}
                last_error = f"Empty response (finish_reason={finish})"
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                if response.status_code in (429, 502, 503) and attempt < 4:
                    time.sleep(3.0 * (attempt + 1))
                    continue
                return f"Error: {last_error}"
        except Exception as e:
            last_error = str(e)
        if attempt < 4:
            time.sleep(2.0 * (attempt + 1))
    return f"Error: {last_error} (after 5 retries)"


# ---------------------------------------------------------------------------
# OpenAI native API
# ---------------------------------------------------------------------------

def run_openai_model(
    model_id: str,
    question: str,
    image_data: List[str] = None,
    max_tokens: int = 16384,
    base_url: str = None,
) -> dict:
    """Call OpenAI API directly (or any OpenAI-compatible endpoint).

    Args:
        model_id: e.g. "gpt-4o", "gpt-4o-mini", "o3", "o4-mini"
        base_url: Override API base URL (for Azure OpenAI, proxies, etc.)
                  Can also be set via OPENAI_BASE_URL env var.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return "Error: pip install openai"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY not set"

    _base_url = base_url or os.getenv("OPENAI_BASE_URL")
    kwargs = {"api_key": api_key}
    if _base_url:
        kwargs["base_url"] = _base_url
    client = OpenAI(**kwargs)

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": _build_image_content(question, image_data)}],
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        return {"content": text, "usage": usage}
    except Exception as e:
        return f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Google Gemini native API
# ---------------------------------------------------------------------------

def run_gemini_model(
    model_id: str,
    question: str,
    image_data: List[str] = None,
    max_output_tokens: int = 16384,
) -> dict:
    """Call Google Gemini API directly.

    Args:
        model_id: e.g. "gemini-2.0-flash", "gemini-2.5-pro-preview-05-06"

    Set GEMINI_VIA_OPENROUTER=1 to route through OpenRouter instead of Google API.
    """
    if os.getenv("GEMINI_VIA_OPENROUTER", "").lower() in ("1", "true", "yes"):
        return run_openrouter_model(f"google/{model_id}", question, image_data)

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "Error: pip install google-genai"

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not set"

    client = genai.Client(api_key=api_key)
    parts = [{"text": question}]
    if image_data:
        for img_b64 in image_data:
            if img_b64:
                parts.append({
                    "inline_data": {
                        "mime_type": _mime_from_base64(img_b64),
                        "data": base64.b64decode(img_b64),
                    }
                })

    last_error = None
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=[{"role": "user", "parts": parts}],
                config=types.GenerateContentConfig(max_output_tokens=max_output_tokens),
            )
            text = ""
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if getattr(part, "text", None):
                        text += part.text
            text = text.strip()
            if not text:
                last_error = "Gemini returned empty"
                continue
            usage = None
            if getattr(response, "usage_metadata", None):
                um = response.usage_metadata
                pt = getattr(um, "prompt_token_count", None)
                ct = getattr(um, "candidates_token_count", None)
                if pt is not None and ct is not None:
                    usage = {"prompt_tokens": int(pt), "completion_tokens": int(ct)}
                    tt = getattr(um, "thoughts_token_count", None)
                    if tt is not None:
                        usage["thought_token_count"] = int(tt)
            return {"content": text, "usage": usage}
        except Exception as e:
            last_error = str(e)
            if any(x in last_error for x in ("429", "503", "resource_exhausted")):
                time.sleep(3.0 * (attempt + 1))
                continue
            return f"Error: {last_error}"
    return f"Error: {last_error or 'Gemini failed after retries'}"


# ---------------------------------------------------------------------------
# Local server — vLLM / Ollama / LMStudio / SGLang
# ---------------------------------------------------------------------------

LOCAL_MODEL_BASE_URL = os.getenv("LOCAL_MODEL_BASE_URL", "http://localhost:8000")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "")


def run_local_model(
    question: str,
    image_data: List[str] = None,
    base_url: str = None,
    model: str = None,
    max_tokens: int = 32768,
) -> dict:
    """Call a locally served model via OpenAI-compatible API.

    Works with vLLM, Ollama, LMStudio, SGLang, and any OpenAI-compatible server.
    Configure via LOCAL_MODEL_BASE_URL and LOCAL_MODEL_NAME env vars.
    """
    _base_url = (base_url or LOCAL_MODEL_BASE_URL).rstrip("/")
    _model = model or LOCAL_MODEL_NAME
    if not _model:
        return "Error: LOCAL_MODEL_NAME not set (or pass model= argument)"

    payload = {
        "model": _model,
        "messages": [{"role": "user", "content": _build_image_content(question, image_data)}],
        "max_tokens": max_tokens,
    }
    try:
        session = requests.Session()
        session.proxies = {"http": None, "https": None}  # bypass proxy for localhost
        response = session.post(
            f"{_base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=600,
        )
        if response.status_code == 200:
            result = response.json()
            choice = (result.get("choices") or [{}])[0]
            content_str = choice.get("message", {}).get("content") or ""
            if content_str:
                return {"content": content_str, "usage": _parse_usage(result.get("usage") or {})}
        return f"Error: HTTP {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Named model shortcuts
# ---------------------------------------------------------------------------
# To add a new model: define a function and add it to SUPPORTED_MODELS below.
# Or pass any OpenRouter model ID directly on the command line,
# e.g. --models meta-llama/llama-3.2-90b-vision-instruct

def run_gpt4o(question: str, image_data: List[str] = None) -> dict:
    if os.getenv("OPENAI_API_KEY"):
        return run_openai_model("gpt-4o", question, image_data)
    return run_openrouter_model("openai/gpt-4o", question, image_data)

def run_gpt4o_mini(question: str, image_data: List[str] = None) -> dict:
    if os.getenv("OPENAI_API_KEY"):
        return run_openai_model("gpt-4o-mini", question, image_data)
    return run_openrouter_model("openai/gpt-4o-mini", question, image_data)

def run_gpt_o3(question: str, image_data: List[str] = None) -> dict:
    if os.getenv("OPENAI_API_KEY"):
        return run_openai_model("o3", question, image_data)
    return run_openrouter_model("openai/o3", question, image_data)

def run_gemini_2_flash(question: str, image_data: List[str] = None) -> dict:
    if os.getenv("GEMINI_API_KEY"):
        return run_gemini_model("gemini-2.0-flash", question, image_data)
    return run_openrouter_model("google/gemini-2.0-flash-001", question, image_data)

def run_gemini_2_5_pro(question: str, image_data: List[str] = None) -> dict:
    if os.getenv("GEMINI_API_KEY"):
        return run_gemini_model("gemini-2.5-pro-preview-05-06", question, image_data)
    return run_openrouter_model("google/gemini-2.5-pro-preview", question, image_data)

def run_gemini_2_5_flash(question: str, image_data: List[str] = None) -> dict:
    if os.getenv("GEMINI_API_KEY"):
        return run_gemini_model("gemini-2.5-flash-preview-04-17", question, image_data)
    return run_openrouter_model("google/gemini-2.5-flash-preview", question, image_data)

def run_claude_sonnet(question: str, image_data: List[str] = None) -> dict:
    return run_openrouter_model("anthropic/claude-sonnet-4-5", question, image_data)

def run_claude_opus(question: str, image_data: List[str] = None) -> dict:
    return run_openrouter_model("anthropic/claude-opus-4-5", question, image_data)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    # OpenAI
    "gpt4o":            run_gpt4o,
    "gpt4o_mini":       run_gpt4o_mini,
    "gpt_o3":           run_gpt_o3,
    # Google Gemini
    "gemini_2_flash":   run_gemini_2_flash,
    "gemini_2_5_pro":   run_gemini_2_5_pro,
    "gemini_2_5_flash": run_gemini_2_5_flash,
    # Anthropic Claude (via OpenRouter)
    "claude_sonnet":    run_claude_sonnet,
    "claude_opus":      run_claude_opus,
    # Local server — configure via LOCAL_MODEL_BASE_URL + LOCAL_MODEL_NAME
    "local":            run_local_model,
}
