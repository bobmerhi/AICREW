# app.py
"""
AI Crew – Enhanced UI & Audio Support

Features
--------
* Live-fetch of available models from Groq, NVIDIA NIM, OpenRouter, Google AI Studio.
* Automatic fallback to a compatible model when the selected one is unavailable.
* File uploader (any type) + built-in voice-note recorder.
* Whisper speech-to-text using Groq's audio endpoint.
* Full multi-agent crew workflow (Manager, Researcher, Coder, Builder, Reviewer).
"""

import os
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI

# --------------------------------------------------------------------------- #
# 1. Page Config & Session State Initialization
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="AI Crew – Enhanced", page_icon="🧑‍🚀", layout="wide")

DEFAULT_AGENTS = [
    {
        "id": "manager",
        "name": "Manager",
        "role": "Project Manager",
        "provider": "Groq",
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.3,
        "enabled": True,
        "system_prompt": (
            "You are the Project Manager of an AI crew. Break down the user's goal "
            "into clear, actionable steps and assign focus areas for the team."
        ),
    },
    {
        "id": "researcher",
        "name": "Researcher",
        "role": "Technical Researcher",
        "provider": "Groq",
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.4,
        "enabled": True,
        "system_prompt": (
            "You are the Researcher. Gather key facts, best practices, edge cases, "
            "and architectural options needed to solve the task accurately."
        ),
    },
    {
        "id": "coder",
        "name": "Coder",
        "role": "Lead Developer",
        "provider": "Groq",
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.2,
        "enabled": True,
        "system_prompt": (
            "You are the Lead Coder. Write clean, well-structured, production-ready "
            "code based on the Manager's plan and Researcher's findings."
        ),
    },
    {
        "id": "builder",
        "name": "Builder",
        "role": "Integration & Setup Engineer",
        "provider": "Groq",
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.3,
        "enabled": True,
        "system_prompt": (
            "You are the Builder. Assemble the components, configuration files, "
            "dependencies, and deployment instructions so everything runs smoothly."
        ),
    },
    {
        "id": "reviewer",
        "name": "Reviewer",
        "role": "QA & Code Reviewer",
        "provider": "Groq",
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.2,
        "enabled": True,
        "system_prompt": (
            "You are the Reviewer. Audit the previous outputs for bugs, security "
            "issues, missing edge cases, and deliver the final polished response."
        ),
    },
]

if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}

if "agents" not in st.session_state:
    st.session_state.agents = [dict(a) for a in DEFAULT_AGENTS]

if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""

if "crew_results" not in st.session_state:
    st.session_state.crew_results = []

# --------------------------------------------------------------------------- #
# 2. Provider Definitions
# --------------------------------------------------------------------------- #
PROVIDERS = {
    "NVIDIA NIM": {
        "base_url": "https://nvidia.com",
        "secrets": ["NVIDIA_API_KEY"],
        "default_model": "meta/llama-3.3-70b-instruct",
    },
    "Groq": {
        "base_url": "https://groq.com",
        "secrets": ["GROQ_API_KEY"],
        "default_model": "llama-3.3-70b-versatile",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai",
        "secrets": ["OPENROUTER_API_KEY"],
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "Google AI Studio": {
        "base_url": "https://googleapis.com",
        "secrets": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "default_model": "gemini-2.5-flash",
    },
}

# --------------------------------------------------------------------------- #
# 3. Helper Utilities
# --------------------------------------------------------------------------- #
def get_key(provider: str) -> str:
    """Return API key with priority: sidebar > Streamlit secrets > env."""
    typed = st.session_state.api_keys.get(provider, "").strip()
    if typed:
        return typed

    for secret_name in PROVIDERS[provider]["secrets"]:
        try:
            if secret_name in st.secrets:
                val = str(st.secrets[secret_name]).strip()
                if val:
                    return val
        except Exception:
            pass

        env_val = os.getenv(secret_name, "").strip()
        if env_val:
            return env_val

    return ""


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_model_catalog(api_keys_snapshot: tuple) -> Dict[str, List[Dict[str, Any]]]:
    """Query every provider for its list of models."""
    catalog = {}
    keys_map = dict(api_keys_snapshot)

    for prov, cfg in PROVIDERS.items():
        key = keys_map.get(prov, "")
        if not key:
            catalog[prov] = []
            continue
        try:
            client = OpenAI(
                api_key=key,
                base_url=cfg["base_url"],
                timeout=20,
            )
            resp = client.models.list()
            catalog[prov] = [m.model_dump() for m in resp.data]
        except Exception as e:
            st.warning(f"Could not fetch models from **{prov}**: {e}")
            catalog[prov] = []
    return catalog


def resolve_model(
    provider: str,
    preferred_model: str = "",
    capability: str = "chat",
    fallback_list: List[str] | None = None,
) -> str:
    """Return the best available model ID for provider with automatic fallback."""
    catalog = st.session_state.get("model_catalog", {})
    available = catalog.get(provider, [])
    available_ids = [m["id"] for m in available if "id" in m]

    def supports(model_id: str) -> bool:
        mid = model_id.lower()
        if capability == "audio":
            return "whisper" in mid or "speech" in mid
        if "whisper" in mid or "tts" in mid or "embed" in mid:
            return False
        return True

    # 1. Try preferred model if present in live catalog
    if preferred_model and preferred_model in available_ids and supports(preferred_model):
        return preferred_model

    # 2. Try sidebar selected model for this provider
    sidebar_selected = st.session_state.get(f"selected_model_{provider}")
    if sidebar_selected and sidebar_selected in available_ids and supports(sidebar_selected):
        return sidebar_selected

    # 3. Try explicit fallback list
    for fid in fallback_list or []:
        if fid in available_ids and supports(fid):
            return fid

    # 4. Try provider default if in catalog
    default_mid = PROVIDERS[provider]["default_model"]
    if default_mid in available_ids and supports(default_mid):
        return default_mid

    # 5. Pick first compatible model from live catalog
    for mid in available_ids:
        if supports(mid):
            return mid

    # 6. Last resort fallback
    return preferred_model or default_mid


def stream_completion(agent: dict, messages: List[dict], placeholder) -> str:
    """Stream a chat completion into placeholder and return the full text."""
    provider = agent["provider"]
    key = get_key(provider)
    if not key:
        raise ValueError(f"No API key for {provider}. Add it in the sidebar or secrets.")

    model_to_use = resolve_model(provider, preferred_model=agent.get("model", ""), capability="chat")
    headers = {"X-Title": "AI Crew"} if provider == "OpenRouter" else None

    client = OpenAI(
        api_key=key,
        base_url=PROVIDERS[provider]["base_url"],
        timeout=120,
        default_headers=headers,
    )

    stream = client.chat.completions.create(
        model=model_to_use,
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


def transcribe_audio(file_bytes: bytes, filename: str = "audio.wav") -> str:
    """Send audio bytes to Groq Whisper endpoint and return the transcript."""
    provider = "Groq"
    key = get_key(provider)
    if not key:
        raise ValueError("Groq API key is required for audio transcription.")

    whisper_model = resolve_model(
        provider,
        preferred_model="whisper-large-v3-turbo",
        capability="audio",
        fallback_list=["whisper-large-v3-turbo", "whisper-large-v3"],
    )

    client = OpenAI(
        api_key=key,
        base_url=PROVIDERS[provider]["base_url"],
        timeout=120,
    )

    resp = client.audio.transcriptions.create(
        model=whisper_model,
        file=(filename, file_bytes),
    )
    return resp.text


def extract_uploaded_file_context(uploaded_file) -> str:
    """Safely extract readable text or metadata from any uploaded file."""
    if uploaded_file is None:
        return ""

    raw_bytes = uploaded_file.getvalue()
    file_size_kb = round(len(raw_bytes) / 1024, 2)
    header = f"[Attached File: {uploaded_file.name} | Type: {uploaded_file.type or 'unknown'} | Size: {file_size_kb} KB]"

    try:
        decoded = raw_bytes.decode("utf-8")
        return f"{header}\nContent:\n{clip(decoded, 8000)}"
    except UnicodeDecodeError:
