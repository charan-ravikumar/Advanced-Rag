"""Verify the model ids in config.yaml against the live Gemini API. Prints model ids and statuses only (no keys, no project data).

    python scripts/check_model_ids.py

* Generator: checks that ``generation.gemini_model`` is among the models that support generateContent and makes ONE tiny call with the
  answer-path settings.
* Judge: the same for ``generation.judge.model``, with the judge's thinking budget and output limit.
The tiny call's text is the fixed string "Reply with OK" (no project data).
Keys come from the environment or .env (rag_core.env.load_env loads an allowlist of names; values are never printed).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TINY_PROMPT = "Reply with OK"


def check_gemini(label: str, settings: dict) -> bool:
    import os

    from google import genai
    from google.genai import types

    model = settings["model"]
    # thinking_config is only accepted on the v1beta API version; the SDK's default version rejects it with HTTP 400
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(api_version="v1beta", timeout=30000,
                                                                                               retry_options=types.HttpRetryOptions(attempts=1)))
    try:
        ids = sorted(m.name.split("/", 1)[-1] for m in client.models.list() if "generateContent" in (m.supported_actions or []))
        print(f"gemini {label}: {len(ids)} generateContent models listed; configured '{model}' exists: {model in ids}")
        print(f"gemini {label}: ids containing 'flash' or 'pro': {[i for i in ids if 'flash' in i or 'pro' in i][:25]}")
    except Exception as exc:  # noqa: BLE001
        print(f"gemini {label}: could not list models ({type(exc).__name__}); trying the tiny call anyway")
        ids = []
    try:
        extra = {}
        if settings.get("thinking_budget") is not None:
            extra["thinking_config"] = types.ThinkingConfig(thinking_budget=int(settings["thinking_budget"]))
        config = types.GenerateContentConfig(temperature=settings.get("temperature", 0.0), max_output_tokens=int(settings["max_output_tokens"]), **extra)
        reply = client.models.generate_content(model=model, contents=TINY_PROMPT, config=config)
        text = (reply.text or "").strip()
        print(f"gemini {label}: tiny call to '{model}' (thinking_budget={settings.get('thinking_budget')}, max_output_tokens={settings['max_output_tokens']}) "
              f"finish={getattr(reply.candidates[0], 'finish_reason', None) if reply.candidates else None}; reply: {text[:20]!r}")
        return bool(text)  # an empty reply is a FAILURE: it is what an exhausted output budget looks like
    except Exception as exc:  # noqa: BLE001
        print(f"gemini {label}: tiny call to '{model}' FAILED ({type(exc).__name__}, status {getattr(exc, 'code', None)})")
        return False


def main() -> int:
    from config import cfg
    from rag_core.config import RagConfig
    from rag_core.env import load_env, present

    load_env()
    have = present("GEMINI_API_KEY")
    print("keys available (booleans):", have)
    rag = RagConfig.from_config(cfg)
    if not have["GEMINI_API_KEY"]:
        print("gemini: skipped (no key)")
        return 1
    generator = {"model": rag.gemini_model, "temperature": rag.temperature, "thinking_budget": rag.gemini_thinking_budget,
                 "max_output_tokens": rag.max_output_tokens}
    ok = check_gemini("generator", generator)
    if rag.judge.get("provider") == "gemini":
        ok &= check_gemini("judge", rag.judge)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
