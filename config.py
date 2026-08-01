"""
╔══════════════════════════════════════════════════════════════╗
║  config.py — Central configuration for the AI Data Suite      ║
║                                                                ║
║  The OpenRouter API key is NEVER hardcoded. It is read from:   ║
║     1. Streamlit secrets   ( .streamlit/secrets.toml )         ║
║     2. Environment / .env  ( OPENROUTER_API_KEY )              ║
║                                                                ║
║  If no key is found the app still runs — AI features simply    ║
║  show a friendly "add your key" message.                       ║
╚══════════════════════════════════════════════════════════════╝
"""
import os

# Load variables from a local .env file if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env in the project root, if present
except Exception:
    pass

# ─── OpenRouter endpoints ──────────────────────────────────────────────────────
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"

# Optional identification headers (used for OpenRouter rankings)
APP_REFERER = "https://ai-data-pipeline.app"
APP_TITLE = "AI Data Engineering Suite"

# ─── Default models ────────────────────────────────────────────────────────────
# Reasoning-capable chat model used for queries by default.
DEFAULT_CHAT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
# Multimodal embedding model used for semantic search.
DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"


def get_api_key() -> str:
    """Resolve the OpenRouter API key from secrets or environment (never hardcoded)."""
    # 1. Streamlit secrets (best place for deployment)
    try:
        import streamlit as st  # local import so non-streamlit callers still work
        if "OPENROUTER_API_KEY" in st.secrets:
            key = str(st.secrets["OPENROUTER_API_KEY"]).strip()
            if key:
                return key
    except Exception:
        pass
    # 2. Environment variable (loaded from .env above, if present)
    return os.environ.get("OPENROUTER_API_KEY", "").strip()
