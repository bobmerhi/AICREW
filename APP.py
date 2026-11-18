import io
import os
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI

# --------------------------------------------------------------------------- #
# 1. Provider definitions
# --------------------------------------------------------------------------- #
PROVIDERS = {
    "NVIDIA NIM": {
        "base_url": "https://nvidia.com",
        "secret": "NVIDIA_API_KEY",
        "default_model": "deepseek-ai/deepseek-v4.1-flash",
    },
    "Groq": {
        "base_url": "https://groq.com",
        "secret": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai",
        "secret": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "Google AI Studio": {
        "base_url": "https://googleapis.com",
        "secret": "GEMINI_API_KEY",
        "default_model": "gemini-3.5-flash-lite",
    },
}

# --------------------------------------------------------------------------- #
# 2. Helper utilities
# --------------------------------------------------------------------------- #
def get_key(provider: str) -> str:
    """Return API key with priority: sidebar > Streamlit secrets > env."""
    typed = st.session_state.api_keys.get(provider, "").strip()
    if typed:
        return typed
    secret_name = PROVIDERS.get(provider, {}).get("secret", "")
    try:
        if secret_name in st.secrets:
            return str(st.secrets[secret_name]).strip()
    except Exception:
        pass
    return os.getenv(secret_name, "").strip()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_model_catalog(api_keys_snapshot: tuple) -> Dict[str, List[Dict[str, Any]]]:
    """Query every configured provider for its live list of models."""
    catalog = {}
    keys_map = dict(api_keys_snapshot)
    for prov, cfg in PROVIDERS.items():
        key = keys_map.get(prov, "")
        if not key:
            catalog[prov] = []
            continue
        try:
            client = OpenAI(api_key=key, base_url=cfg["base_url"], timeout=20)
            resp = client.models.list()
            models = [m.model_dump() for m in resp.data]
            if prov == "OpenRouter":
                models = [m for m in models if ":free" in m["id"]]
            catalog[prov] = sorted(models, key=lambda x: x["id"])
        except Exception:
            catalog[prov] = []
    return catalog


def resolve_model(provider: str, fallback_list: List[str] | None = None) -> str:
    """Pick the selected or best available chat model for a provider."""
    catalog = st.session_state.get("model_catalog", {})
    available = catalog.get(provider, [])

    def is_chat_model(mid: str) -> bool:
        low = mid.lower()
        blocked = ["whisper", "embed", "tts", "guard", "vision-preview", "image", "audio"]
        if any(b in low for b in blocked):
            return False
        tokens = ["gpt", "llama", "mixtral", "gemini", "qwen", "deepseek", "nemotron", "mistral"]
        return any(t in low for t in tokens)

    current = st.session_state.get(f"selected_model_{provider}")
    if current and current != "<none>":
        return current

    for fid in (fallback_list or []):
        if any(m["id"] == fid for m in available):
            return fid

    for m in available:
        if is_chat_model(m["id"]):
            return m["id"]

    return PROVIDERS[provider]["default_model"]


def free_web_search(query: str) -> str:
    """Live DuckDuckGo search skill for the Researcher agent."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No search results found."
        return "\n".join([f"- {r['title']}: {r['body']} ({r['href']})" for r in results])
    except Exception as e:
        return f"Web search skipped ({e}). Relying on model knowledge."


def stream_completion(agent: dict, messages: List[dict], placeholder) -> str:
    """Stream a chat completion into `placeholder` with automatic fallback."""
    provider = agent["provider"]
    key = get_key(provider)
    if not key:
        raise ValueError(f"No API key for {provider}. Add it in the sidebar or Secrets.")

    headers = {"X-Title": "AI Crew Portal"} if provider == "OpenRouter" else None
    client = OpenAI(
        api_key=key,
        base_url=PROVIDERS[provider]["base_url"],
        timeout=120,
        default_headers=headers,
    )

    models_to_try = [agent["model"], PROVIDERS[provider]["default_model"]]
    last_err = None

    for model_id in dict.fromkeys(models_to_try):
        try:
            stream = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=agent.get("temperature", 0.4),
                stream=True,
            )
            text = ""
            for chunk in stream:
                if not chunk.choices:
                    continue
                first_choice = next(iter(chunk.choices))
                delta = first_choice.delta.content or ""
                if delta:
                    text += delta
                    placeholder.markdown(text + "▌")
            placeholder.markdown(text)
            return text
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"{provider} failed: {last_err}")


def transcribe_audio(file_bytes: bytes, filename: str = "voice.wav") -> str:
    """Send audio bytes to Groq Whisper endpoint and return the transcript."""
    key = get_key("Groq")
    if not key:
        raise ValueError("Groq API key required for voice transcription. Paste it in the sidebar!")
    client = OpenAI(
        api_key=key,
        base_url=PROVIDERS["Groq"]["base_url"],
        timeout=60,
    )
    audio_file = io.BytesIO(file_bytes)
    audio_file.name = filename
    resp = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",
        file=audio_file,
    )
    return resp.text


def clip(text: str, limit: int = 8000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


# --------------------------------------------------------------------------- #
# 3. Streamlit UI – Page Config & Session State
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="AI Crew – Enhanced Portal", page_icon="🧑‍🚀", layout="wide")

if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}
if "model_catalog" not in st.session_state:
    st.session_state.model_catalog = {}
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "file_context" not in st.session_state:
    st.session_state.file_context = ""

# --------------------------------------------------------------------------- #
# 4. Sidebar – API Keys & Live Model Picker
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🧑‍🚀 AI Crew Portal")
    st.subheader("🔑 API Keys ($0.00 Tiers)")
    for prov in PROVIDERS:
        typed = st.text_input(
            prov,
            type="password",
            key=f"key_{prov}",
            placeholder="Loaded from Secrets or paste here",
        )
        st.session_state.api_keys[prov] = typed
        st.caption("✅ Ready" if get_key(prov) else "⚠️ Missing key")

    st.divider()
    st.subheader("📦 Live Model Picker")
    keys_snapshot = tuple((p, get_key(p)) for p in PROVIDERS)

    if st.button("🔄 Refresh Model Lists"):
        fetch_model_catalog.clear()
        st.session_state.model_catalog = fetch_model_catalog(keys_snapshot)
        st.rerun()

    if not st.session_state.model_catalog:
        st.session_state.model_catalog = fetch_model_catalog(keys_snapshot)

    for prov, cfg in PROVIDERS.items():
        models = st.session_state.model_catalog.get(prov, [])
        model_options = [m["id"] for m in models]
        if not model_options:
            st.selectbox(
                f"{prov} model",
                options=[cfg["default_model"]],
                key=f"selected_model_{prov}",
                help="Using default model (live list unavailable or key missing).",
            )
            continue
        default_idx = (
            model_options.index(cfg["default_model"])
            if cfg["default_model"] in model_options
            else 0
        )
        st.selectbox(
            f"{prov} model",
            options=model_options,
            index=default_idx,
            key=f"selected_model_{prov}",
        )

    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation = []
        st.session_state.file_context = ""
        st.rerun()

# Pick best available active provider for each role
active_providers = [p for p in PROVIDERS if get_key(p)]


def pick_provider(preferred: str) -> str:
    if preferred in active_providers:
        return preferred
    return next(iter(active_providers), preferred)


mgr_prov = pick_provider("Google AI Studio")
res_prov = pick_provider("Groq")
bld_prov = pick_provider("NVIDIA NIM")
rev_prov = pick_provider("OpenRouter")

AGENTS = {
    "manager": {
        "name": "🧭 Crew Manager",
        "provider": mgr_prov,
        "model": resolve_model(mgr_prov, ["gemini-3.5-flash-lite"]),
        "temperature": 0.2,
    },
    "researcher": {
        "name": "🔍 Web Researcher",
        "provider": res_prov,
        "model": resolve_model(res_prov, ["llama-3.3-70b-versatile"]),
        "temperature": 0.3,
    },
    "builder": {
        "name": "🛠️ Lead Builder & Coder",
        "provider": bld_prov,
        "model": resolve_model(bld_prov, ["deepseek-ai/deepseek-v4.1-flash"]),
        "temperature": 0.3,
    },
    "reviewer": {
        "name": "✅ Quality Reviewer",
        "provider": rev_prov,
