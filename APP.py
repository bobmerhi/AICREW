import io
import os
from typing import List, Dict

import streamlit as st
from openai import OpenAI

# --------------------------------------------------------------------------- #
# 1. Provider definitions
# --------------------------------------------------------------------------- #
PROVIDERS = dict(
    nvidia=dict(
        label="NVIDIA NIM",
        base_url="https://nvidia.com",
        secret="NVIDIA_API_KEY",
        default_model="deepseek-ai/deepseek-v4.1-flash",
    ),
    groq=dict(
        label="Groq",
        base_url="https://groq.com",
        secret="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
    ),
    openrouter=dict(
        label="OpenRouter",
        base_url="https://openrouter.ai",
        secret="OPENROUTER_API_KEY",
        default_model="meta-llama/llama-3.3-70b-instruct:free",
    ),
    google=dict(
        label="Google AI Studio",
        base_url="https://googleapis.com",
        secret="GEMINI_API_KEY",
        default_model="gemini-3.5-flash-lite",
    ),
)

# --------------------------------------------------------------------------- #
# 2. Helper utilities
# --------------------------------------------------------------------------- #
def get_key(pkey: str) -> str:
    """Return API key with priority: sidebar > Streamlit secrets > env."""
    typed = st.session_state.api_keys.get(pkey, "").strip()
    if typed:
        return typed
    secret_name = PROVIDERS.get(pkey, {}).get("secret", "")
    try:
        if secret_name in st.secrets:
            return str(st.secrets.get(secret_name, "")).strip()
    except Exception:
        pass
    return os.getenv(secret_name, "").strip()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_model_catalog(api_keys_snapshot: tuple) -> Dict[str, List[str]]:
    """Query every configured provider for its live list of model IDs."""
    catalog = {}
    keys_map = dict(api_keys_snapshot)
    for pkey, cfg in PROVIDERS.items():
        key = keys_map.get(pkey, "")
        if not key:
            catalog[pkey] = []
            continue
        try:
            client = OpenAI(api_key=key, base_url=cfg.get("base_url"), timeout=20)
            resp = client.models.list()
            ids = sorted(m.id for m in resp.data)
            if pkey == "openrouter":
                ids = [mid for mid in ids if ":free" in mid]
            catalog[pkey] = ids
        except Exception:
            catalog[pkey] = []
    return catalog


def resolve_model(pkey: str) -> str:
    """Pick the selected or default chat model for a provider."""
    current = st.session_state.get(f"selected_model_{pkey}")
    if current and current != "<none>":
        return current
    return PROVIDERS.get(pkey, {}).get("default_model", "")


def free_web_search(query: str) -> str:
    """Live DuckDuckGo search skill for the Researcher agent."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No search results found."
        lines = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"- {title}: {body} ({href})")
        return "\n".join(lines)
    except Exception as e:
        return f"Web search skipped ({e}). Relying on model knowledge."


def stream_completion(agent: dict, messages: List[dict], placeholder) -> str:
    """Stream a chat completion into placeholder with automatic fallback."""
    pkey = agent.get("pkey", "nvidia")
    cfg = PROVIDERS.get(pkey, {})
    key = get_key(pkey)
    if not key:
        raise ValueError(f"No API key for {cfg.get('label')}. Add it in the sidebar or Secrets.")

    headers = dict({"X-Title": "AI Crew Portal"}) if pkey == "openrouter" else None
    client = OpenAI(
        api_key=key,
        base_url=cfg.get("base_url"),
        timeout=120,
        default_headers=headers,
    )

    models_to_try = list(dict.fromkeys((agent.get("model"), cfg.get("default_model"))))
    last_err = None

    for model_id in models_to_try:
        if not model_id:
            continue
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

    raise RuntimeError(f"{cfg.get('label')} failed: {last_err}")


def transcribe_audio(file_bytes: bytes, filename: str = "voice.wav") -> str:
    """Send audio bytes to Groq Whisper endpoint and return the transcript."""
    key = get_key("groq")
    if not key:
        raise ValueError("Groq API key required for voice transcription. Paste it in the sidebar!")
    client = OpenAI(
        api_key=key,
        base_url=PROVIDERS.get("groq", {}).get("base_url"),
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
    for pkey, cfg in PROVIDERS.items():
        typed = st.text_input(
            cfg.get("label", pkey),
            type="password",
            key=f"key_{pkey}",
            placeholder="Loaded from Secrets or paste here",
        )
        st.session_state.api_keys[pkey] = typed
        st.caption("✅ Ready" if get_key(pkey) else "⚠️ Missing key")

    st.divider()
    st.subheader("📦 Live Model Picker")
    keys_snapshot = tuple((p, get_key(p)) for p in PROVIDERS)

    if st.button("🔄 Refresh Model Lists"):
        fetch_model_catalog.clear()
        st.session_state.model_catalog = fetch_model_catalog(keys_snapshot)
        st.rerun()

    if not st.session_state.model_catalog:
        st.session_state.model_catalog = fetch_model_catalog(keys_snapshot)

    for pkey, cfg in PROVIDERS.items():
        model_options = st.session_state.model_catalog.get(pkey, [])
        def_mod = cfg.get("default_model", "")
        if not model_options:
            st.selectbox(
                f"{cfg.get('label')} model",
                options=(def_mod,),
                key=f"selected_model_{pkey}",
                help="Using default model (live list unavailable or key missing).",
            )
            continue
        default_idx = model_options.index(def_mod) if def_mod in model_options else 0
        st.selectbox(
            f"{cfg.get('label')} model",
            options=model_options,
            index=default_idx,
            key=f"selected_model_{pkey}",
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


mgr_prov = pick_provider("google")
res_prov = pick_provider("groq")
bld_prov = pick_provider("nvidia")
rev_prov = pick_provider("openrouter")

AGENTS = dict(
    manager=dict(
        name="🧭 Crew Manager",
        pkey=mgr_prov,
        model=resolve_model(mgr_prov),
        temperature=0.2,
    ),
    researcher=dict(
        name="🔍 Web Researcher",
        pkey=res_prov,
        model=resolve_model(res_prov),
        temperature=0.3,
    ),
    builder=dict(
        name="🛠️ Lead Builder & Coder",
        pkey=bld_prov,
        model=resolve_model(bld_prov),
        temperature=0.3,
    ),
    reviewer=dict(
        name="✅ Quality Reviewer",
        pkey=rev_prov,
        model=resolve_model(rev_prov),
        temperature=0.2,
    ),
)

# --------------------------------------------------------------------------- #
# 5. Main Workspace – File Uploader & Voice Input
# --------------------------------------------------------------------------- #
st.title("🧑‍🚀 Multi-Agent AI Command Portal")

if not active_providers:
    st.warning("Add at least one API key in the left sidebar (or Streamlit Secrets) to activate your agents.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "📎 Attach File Context (TXT, CSV, JSON, PY, MD, PDF)",
        type=None,
    )
        if uploaded_file is not None:
        raw_bytes = uploaded_file.read()
        try:
            extracted = raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            extracted = f"[Binary file attached: {uploaded_file.name}, {len(raw_bytes)} bytes]"
        st.session_state.file_context = f"Attached File ({uploaded_file.name}):\n{clip(extracted)}"
        st.success(f"Loaded `{uploaded_file.name}` into agent context.")

with col2:
    voice_note = st.audio_input("🎙️ Record Voice Prompt (Groq Whisper)")
    voice_prompt = ""
    if voice_note is not None:
        with st.spinner("Transcribing voice note with Groq Whisper..."):
            try:
                voice_prompt = transcribe_audio(voice_note.read(), "voice.wav")
                st.info(f"**Transcribed:** {voice_prompt}")
            except Exception as e:
                st.error(f"Audio transcription error: {e}")

# --------------------------------------------------------------------------- #
# 6. Multi-Agent Collaboration Engine
# --------------------------------------------------------------------------- #
def run_multi_agent_pipeline(user_prompt: str) -> str:
    full_context = user_prompt
    if st.session_state.file_context:
        full_context += f"\n\n--- ATTACHED FILE CONTEXT ---\n{st.session_state.file_context}"

    # Step 1: Manager plans & decides whether web search is needed
    with st.expander("🧭 Step 1: Manager Routing & Planning", expanded=True):
        mgr = AGENTS.get("manager", {})
        mgr_label = PROVIDERS.get(mgr.get("pkey", ""), {}).get("label", "")
        st.caption(f"Provider: **{mgr_label}** | Model: `{mgr.get('model')}`")
        mgr_placeholder = st.empty()
        mgr_prompt = (
            dict(
                role="system",
                content=(
                    "You are the Crew Manager. Analyze the user's request and output a brief 3-bullet "
                    "execution plan. On the very last line, write SEARCH_QUERY: <short search query> "
                    "if live internet data is needed, or SEARCH_QUERY: NONE if not needed."
                ),
            ),
            dict(role="user", content=full_context),
        )
        plan_text = stream_completion(mgr, list(mgr_prompt), mgr_placeholder)

    # Step 2: Researcher gathers live web facts if requested by Manager
    research_notes = ""
    search_query = ""
    for line in plan_text.splitlines():
        if "SEARCH_QUERY:" in line:
            _, _, q_part = line.partition("SEARCH_QUERY:")
            q = q_part.strip()
            if q.upper() != "NONE":
                search_query = q

    if search_query:
        with st.expander(f"🔍 Step 2: Web Researcher ({search_query})", expanded=True):
            res_agent = AGENTS.get("researcher", {})
            res_label = PROVIDERS.get(res_agent.get("pkey", ""), {}).get("label", "")
            st.caption(f"Provider: **{res_label}** | Model: `{res_agent.get('model')}`")
            raw_search = free_web_search(search_query)
            res_placeholder = st.empty()
            res_prompt = (
                dict(
                    role="system",
                    content="Summarize these live web search results into accurate technical facts for the Builder agent.",
                ),
                dict(
                    role="user",
                    content=f"User Request: {user_prompt}\n\nSearch Results:\n{raw_search}",
                ),
            )
            research_notes = stream_completion(res_agent, list(res_prompt), res_placeholder)

    # Step 3: Lead Builder produces the solution
    with st.expander("🛠️ Step 3: Lead Builder & Coder", expanded=True):
        bld = AGENTS.get("builder", {})
        bld_label = PROVIDERS.get(bld.get("pkey", ""), {}).get("label", "")
        st.caption(f"Provider: **{bld_label}** | Model: `{bld.get('model')}`")
        bld_placeholder = st.empty()
        bld_prompt = (
            dict(
                role="system",
                content="You are the Lead Engineer & Creator. Deliver a complete, clean, working solution based on the Manager's plan and Researcher's notes.",
            ),
            dict(
                role="user",
                content=(
                    f"Request:\n{full_context}\n\n"
                    f"Manager Plan:\n{plan_text}\n\n"
                    f"Research Notes:\n{research_notes or 'None required.'}"
                ),
            ),
        )
        draft_output = stream_completion(bld, list(bld_prompt), bld_placeholder)

    # Step 4: Quality Reviewer verifies and produces the polished final answer
    with st.expander("✅ Step 4: Quality Reviewer (Final Polish)", expanded=True):
        rev = AGENTS.get("reviewer", {})
        rev_label = PROVIDERS.get(rev.get("pkey", ""), {}).get("label", "")
        st.caption(f"Provider: **{rev_label}** | Model: `{rev.get('model')}`")
        rev_placeholder = st.empty()
        rev_prompt = (
            dict(
                role="system",
                content="You are the Quality & Logic Reviewer. Verify the Builder's draft for bugs, accuracy, and clarity, then output the final polished answer directly to the user.",
            ),
            dict(
                role="user",
                content=f"Original Request:\n{user_prompt}\n\nDraft Solution:\n{draft_output}",
            ),
        )
        final_answer = stream_completion(rev, list(rev_prompt), rev_placeholder)

    return final_answer

# --------------------------------------------------------------------------- #
# 7. Chat History & Prompt Input
# --------------------------------------------------------------------------- #
for msg in st.session_state.conversation:
    with st.chat_message(msg.get("role", "user")):
        st.markdown(msg.get("content", ""))

typed_prompt = st.chat_input("Type your prompt or use the microphone above...", key="main_portal_chat")
active_prompt = typed_prompt or voice_prompt

if active_prompt:
    st.session_state.conversation.append(dict(role="user", content=active_prompt))
    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        try:
            final_reply = run_multi_agent_pipeline(active_prompt)
            st.markdown("### 🎯 Final Verified Response")
            st.markdown(final_reply)
            st.session_state.conversation.append(dict(role="assistant", content=final_reply))
        except Exception as err:
            st.error(f"Pipeline error: {err}")

