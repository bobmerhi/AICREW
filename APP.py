# app.py
"""
AI Crew – Enhanced UI & Audio Support

Features
--------
* Live‑fetch of available models (“modules”) from Groq, NVIDIA NIM, OpenRouter, Google AI Studio.
* Automatic fallback to a compatible model when the selected one is unavailable.
* File uploader (any type) + voice‑note recorder (via streamlit‑webrtc).
* Whisper‑style speech‑to‑text using Groq’s audio endpoint.
* All existing crew functionality (Manager, Researcher, Coder, Builder, Reviewer) retained.
"""

import json
import os
import uuid
import time
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI

# --------------------------------------------------------------------------- #
# 1️⃣ Provider definitions
# --------------------------------------------------------------------------- #
PROVIDERS = {
    "NVIDIA NIM": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "secret": "NVIDIA_API_KEY",
        "default_model": "meta/llama-3.3-70b-instruct",
        "model_list_endpoint": "/models",  # NVIDIA uses standard OpenAI /v1/models
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "secret": "GROQ_API_KEY",
        "default_model": "qwen/qwen3.8-27b",
        "model_list_endpoint": "/models",
        "audio_endpoint": "/audio/transcriptions",  # Whisper‑style endpoint
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "secret": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "model_list_endpoint": "/models",
    },
    "Google AI Studio": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "secret": "GOOGLE_API_KEY",
        "default_model": "gemini-2.5-flash",
        "model_list_endpoint": "/models",
    },
}

# --------------------------------------------------------------------------- #
# 2️⃣ Helper utilities
# --------------------------------------------------------------------------- #
def get_key(provider: str) -> str:
    """Return API key with priority: sidebar > Streamlit secrets > env."""
    typed = st.session_state.api_keys.get(provider, "").strip()
    if typed:
        return typed
    secret_name = PROVIDERS[provider]["secret"]
    # Google also accepts GEMINI_API_KEY alias
    if provider == "Google AI Studio":
        secret_name = "GEMINI_API_KEY"
    # Streamlit secrets
    if secret_name in st.secrets:
        return str(st.secrets[secret_name]).strip()
    # Environment
    return os.getenv(secret_name, "").strip()


@st.cache_data(ttl=3600)  # cache for 1 hour to respect rate limits
def fetch_model_catalog() -> Dict[str, List[Dict[str, Any]]]:
    """Query every provider for its list of models and store the raw JSON."""
    catalog = {}
    for prov, cfg in PROVIDERS.items():
        key = get_key(prov)
        if not key:
            catalog[prov] = []  # no key → empty list (will be shown as unavailable)
            continue
        try:
            client = OpenAI(
                api_key=key,
                base_url=cfg["base_url"],
                timeout=30,
            )
            resp = client.models.list()  # works for all OpenAI‑compatible endpoints
            catalog[prov] = [m.model_dump() for m in resp.data]  # list of dicts
        except Exception as e:
            st.warning(f"❗ Could not fetch models from **{prov}**: {e}")
            catalog[prov] = []
    return catalog


def resolve_model(
    provider: str,
    capability: str,
    fallback_list: List[str] | None = None,
) -> str:
    """
    Return the best available model ID for *provider* that satisfies *capability*.
    *capability* can be "chat" or "audio".  If the currently selected model is
    unavailable, iterate through *fallback_list* (or a built‑in priority list)
    until a working model is found.
    """
    catalog = st.session_state.get("model_catalog", {})
    available = catalog.get(provider, [])

    # Helper to decide if a model supports the needed capability
    def supports(model: Dict[str, Any]) -> bool:
        # Very simple heuristic: look for keywords in the model ID
        mid = model["id"].lower()
        if capability == "audio":
            return "whisper" in mid or "speech" in mid
        # default: chat/completion capability
        return any(tok in mid for tok in ["gpt", "llama", "mixtral", "gemini", "qwen", "claude"])

    # 1️⃣ Try the currently configured model (if any)
    current = st.session_state.get(f"selected_model_{provider}")
    if current:
        if any(m["id"] == current for m in available):
            if any(supports(m) for m in available if m["id"] == current):
                return current

    # 2️⃣ Scan the provider catalog for the first matching model
    for m in available:
        if supports(m):
            return m["id"]

    # 3️⃣ Use fallback list (ordered by preference)
    fallback = fallback_list or []
    for fid in fallback:
        if any(m["id"] == fid for m in available):
            return fid

    # 4️⃣ As a last resort, return the provider’s default model (may be unavailable)
    return PROVIDERS[provider]["default_model"]


def stream_completion(agent: dict, messages: List[dict], placeholder) -> str:
    """Stream a chat completion into `placeholder` and return the full text."""
    provider = agent["provider"]
    key = get_key(provider)
    if not key:
        raise ValueError(f"No API key for {provider}. Add it in the sidebar or Streamlit secrets.")
    headers = {"X-Title": "AI Crew"} if provider == "OpenRouter" else None
    client = OpenAI(
        api_key=key,
        base_url=PROVIDERS[provider]["base_url"],
        timeout=120,
        default_headers=headers,
    )
    stream = client.chat.completions.create(
        model=agent["model"],
        messages=messages,
        temperature=agent.get("temperature", 0.4),
        stream=True,
    )
    text = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if delta:
            text += delta
            placeholder.markdown(text + "▌")
    placeholder.markdown(text)
    return text


def transcribe_audio(file_bytes: bytes, mime: str) -> str:
    """Send audio bytes to Groq Whisper endpoint and return the transcript."""
    provider = "Groq"
    key = get_key(provider)
    if not key:
        raise ValueError("Groq API key required for audio transcription.")
    client = OpenAI(
        api_key=key,
        base_url=PROVIDERS[provider]["base_url"],
        timeout=120,
    )
    # Groq expects multipart/form-data with fields: file, model, language (optional)
    # We'll use the generic `audio.transcriptions` endpoint.
    resp = client.audio.transcriptions.create(
        model="whisper-large-v3",  # Groq’s Whisper model (adjust if needed)
        file=("audio", file_bytes, mime),
    )
    return resp.text


def clip(text: str, limit: int = 6000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


def enabled_agents():
    return [a for a in st.session_state.agents if a["enabled"]]


def agent_by_id(agent_id):
    return next((a for a in st.session_state.agents if a["id"] == agent_id), None)


# --------------------------------------------------------------------------- #
# 3️⃣ Streamlit UI – Sidebar (API keys + Dynamic Model Picker)
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="AI Crew – Enhanced", page_icon="🧑‍🚀", layout="wide")

with st.sidebar:
    st.title("🧑‍🚀 AI Crew")
    st.subheader("🔑 API keys")
    for prov in PROVIDERS:
        typed = st.text_input(prov, type="password", key=f"key_{prov}", placeholder="paste key (session only)")
        st.session_state.api_keys[prov] = typed
        st.caption("✅ key found" if get_key(prov) else "⚠️ no key")

    st.divider()
    st.subheader("📦 Model picker (live)")
    # Load (or reload) the model catalog
    if st.button("Refresh model list"):
        st.session_state.model_catalog = fetch_model_catalog()
        st.rerun()
    if "model_catalog" not in st.session_state:
        st.session_state.model_catalog = fetch_model_catalog()

    # Show a multiselect per provider with status badges
    for prov, cfg in PROVIDERS.items():
        models = st.session_state.model_catalog.get(prov, [])
        model_options = [m["id"] for m in models]
        # Build a display label with status badge
        display = [
            f"{mid} {'✅' if mid else ''}"
            for mid in model_options
        ]
        # Remember the user‑selected model (fallback will be applied later)
        sel = st.selectbox(
            f"{prov} models",
            options=model_options,
            index=0 if model_options else -1,
            key=f"selected_model_{prov}",
            help="If the selected model becomes unavailable, the app will auto‑fallback.",
        )
        # Store the selection for later use
        st.session_state[f"selected_model_{prov}"] = sel

    st.divider()
    st.subheader("📁 File & Voice upload")
    uploaded = st.file_uploader(
        "Upload any file (PDF, txt, image, audio…) – audio will be transcribed",
        type=None,
        accept_multiple_files=False,
        key="uploaded_file",
    )
    # Voice note recorder (requires streamlit‑webrtc)
    try:
        from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings

        class AudioProcessor(AudioProcessorBase):
            def __init__(self):
                self.audio_frames = []

            def recv(self, frame):
                self.audio_frames.append(frame.to_ndarray())
                return frame

        webrtc_ctx = webrtc_streamer(
            key="voice_note",
            mode="client",
            client_settings=ClientSettings(
                media_stream_constraints={"audio": True, "video": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            ),
            audio_processor_factory=AudioProcessor,
            async_processing=False,
        )
        voice_note_bytes = None
        if webrtc_ctx.audio_receiver:
            # Convert collected frames to a single WAV bytes object
            import numpy as np
            import wave
            import io

            frames =
