import io
import os
from typing import List, Dict, Any

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
        # Read file content based on extension
        try:
            file_name = uploaded_file.name
            file_ext = file_name.split('.')[-1].lower()
            file_content = ""

            if file_ext == 'txt':
                file_content = uploaded_file.read().decode('utf-8', errors='ignore')
            elif file_ext == 'csv':
                file_content = uploaded_file.read().decode('utf-8', errors='ignore')
            elif file_ext == 'json':
                file_content = uploaded_file.read().decode('utf-8', errors='ignore')
            elif file_ext == 'py':
                file_content = uploaded_file.read().decode('utf-8', errors='ignore')
            elif file_ext == 'md':
                file_content = uploaded_file.read().decode('utf-8', errors='ignore')
            elif file_ext == 'pdf':
                # Note: Requires 'pypdf' or 'pdfplumber' installed
                try:
                    import pdfplumber
                    with pdfplumber.open(uploaded_file) as pdf:
                        file_content = "\n".join(page.extract_text() or "" for page in pdf.pages)
                except ImportError:
                    st.error("Please install 'pdfplumber' to support PDF files: pip install pdfplumber")
                    file_content = ""
            else:
                st.warning(f"Unsupported file type: {file_ext}. Please use TXT, CSV, JSON, PY, MD, or PDF.")

            if file_content:
                st.success(f"✅ File '{file_name}' loaded successfully.")
                # Store in session_state for use in prompt
                st.session_state['file_context'] = file_content
                st.session_state['file_name'] = file_name
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Voice Input Section (Optional - requires browser support)
with col2:
    st.subheader("🎤 Voice Input")
    st.caption("Click the mic button to record your voice (requires browser permissions).")

    # Streamlit doesn't have native voice input, so we provide a text box as fallback
    # and a note about browser limitations. For true voice, you'd need a custom component.
    voice_note = st.text_area(
        "Or type your command here:",
        placeholder="Type your command if voice input is not available...",
        height=100,
        key="voice_input_text"
    )

# --------------------------------------------------------------------------- #
# 6. Chat Interface & Agent Orchestration
# --------------------------------------------------------------------------- #

# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for user command
user_input = st.chat_input("Enter your command or question...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare context
    file_context = st.session_state.get('file_context', "")
    file_name = st.session_state.get('file_name', "")

    # Build system prompt with context
    system_prompt = "You are a helpful AI assistant. If file context is provided, use it to answer questions."
    if file_context:
        system_prompt += f"\n\nContext from file '{file_name}':\n{file_context[:5000]}"  # Limit context length

    # Simulate agent orchestration
    # In a real implementation, you'd route to different agents based on intent
    # For now, we'll use a single LLM call with the active providers

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Use the first active provider for simplicity
        if active_providers:
            provider = active_providers[0]

            try:
                # Call the LLM based on provider
                if provider['name'] == 'OpenAI':
                    import openai
                    client = openai.OpenAI(api_key=provider['api_key'])
                    response = client.chat.completions.create(
                        model=provider['model'],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_input}
                        ],
                        stream=True
                    )
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")

                elif provider['name'] == 'Anthropic':
                    import anthropic
                    client = anthropic.Anthropic(api_key=provider['api_key'])
                    message = client.messages.create(
                        model=provider['model'],
                        max_tokens=1024,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_input}]
                    )
                    full_response = message.content[0].text
                    message_placeholder.markdown(full_response)

                elif provider['name'] == 'Google Gemini':
                    import google.generativeai as genai
                    genai.configure(api_key=provider['api_key'])
                    model = genai.GenerativeModel(provider['model'])
                    response = model.generate_content(user_input)
                    full_response = response.text
                    message_placeholder.markdown(full_response)

                else:
                    message_placeholder.warning(f"Provider '{provider['name']}' not implemented yet.")
                    full_response = f"Provider '{provider['name']}' not implemented yet."

            except Exception as e:
                error_msg = f"Error calling {provider['name']}: {str(e)}"
                message_placeholder.error(error_msg)
                full_response = error_msg
        else:
            message_placeholder.warning("No active providers. Add an API key in the sidebar.")
            full_response = "No active providers."

        # Remove cursor
        message_placeholder.markdown(full_response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    # --------------------------------------------------------------------------- #
# 7. Agent Orchestration & Intent Routing
# --------------------------------------------------------------------------- #

def route_to_agent(user_input: str, file_context: str = ""):
    """
    Simple intent-based routing to different 'agents' (prompts/personas).
    In a production system, you might use an LLM to classify intent first.
    """
    input_lower = user_input.lower()

    # Define simple keyword-based routing for demonstration
    if any(kw in input_lower for kw in ["code", "python", "debug", "function", "script"]):
        return "coder_agent"
    elif any(kw in input_lower for kw in ["write", "email", "report", "summary", "text"]):
        return "writer_agent"
    elif any(kw in input_lower for kw in ["analyze", "data", "csv", "json", "insight"]):
        return "analyst_agent"
    else:
        return "general_agent"

def get_agent_prompt(agent_name: str, file_context: str = ""):
    """
    Returns the system prompt for the specific agent.
    """
    base_context = f"\n\nContext from file: {file_context[:2000]}" if file_context else ""

    prompts = {
        "coder_agent": f"""
        You are an expert Python Developer. 
        - Provide clean, efficient, and well-commented code.
        - Explain your logic briefly.
        - If code is provided, suggest improvements or bug fixes.
        {base_context}
        """,
        "writer_agent": f"""
        You are a Professional Copywriter and Editor.
        - Tone: Professional, clear, and engaging.
        - Format output with proper headings and bullet points if appropriate.
        - If asked to summarize, keep it concise.
        {base_context}
        """,
        "analyst_agent": f"""
        You are a Data Analyst.
        - Interpret data from provided context.
        - Provide insights, trends, and actionable recommendations.
        - Use tables or bullet points for clarity.
        {base_context}
        """,
        "general_agent": f"""
        You are a helpful, friendly, and knowledgeable AI assistant.
        - Answer questions accurately and concisely.
        - If you don't know, say so and suggest how to find out.
        {base_context}
        """
    }

    return prompts.get(agent_name, prompts["general_agent"])

# --------------------------------------------------------------------------- #
# 8. Enhanced Chat Loop with Agent Routing
# --------------------------------------------------------------------------- #

# Re-initialize chat history if not present
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for user command
user_input = st.chat_input("Enter your command or question...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Determine which agent to use
    file_context = st.session_state.get('file_context', "")
    agent_name = route_to_agent(user_input, file_context)
    system_prompt = get_agent_prompt(agent_name, file_context)

    # Show which agent is being used
    st.info(f"🤖 Routing to: **{agent_name.replace('_agent', '').title()} Agent**")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Use the first active provider for simplicity
        if active_providers:
            provider = active_providers[0]

            try:
                # Call the LLM based on provider
                if provider['name'] == 'OpenAI':
                    import openai
                    client = openai.OpenAI(api_key=provider['api_key'])
                    response = client.chat.completions.create(
                        model=provider['model'],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_input}
                        ],
                        stream=True
                    )
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")

                elif provider['name'] == 'Anthropic':
                    import anthropic
                    client = anthropic.Anthropic(api_key=provider['api_key'])
                    message = client.messages.create(
                        model=provider['model'],
                        max_tokens=1024,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_input}]
                    )
                    full_response = message.content[0].text
                    message_placeholder.markdown(full_response)

                elif provider['name'] == 'Google Gemini':
                    import google.generativeai as genai
                    genai.configure(api_key=provider['api_key'])
                    model = genai.GenerativeModel(provider['model'])
                    response = model.generate_content(user_input)
                    full_response = response.text
                    message_placeholder.markdown(full_response)

                else:
                    message_placeholder.warning(f"Provider '{provider['name']}' not implemented yet.")
                    full_response = f"Provider '{provider['name']}' not implemented yet."

            except Exception as e:
                error_msg = f"Error calling {provider['name']}: {str(e)}"
                message_placeholder.error(error_msg)
                full_response = error_msg
        else:
            message_placeholder.warning("No active providers. Add an API key in the sidebar.")
            full_response = "No active providers."

        # Remove cursor and finalize
        message_placeholder.markdown(full_response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --------------------------------------------------------------------------- #
# 9. Session Management & UI Footer
# --------------------------------------------------------------------------- #

def clear_session():
    """Clears the chat history and resets the session state."""
    st.session_state.messages = []
    st.session_state.full_response = ""  # Reset any in-progress response state
    st.rerun()

# Display a button to clear the history
if st.button("🧹 Clear Chat History"):
    clear_session()

# Optional: Display a small footer with app info
st.markdown("---")
st.caption("Built with Streamlit & LangChain | [GitHub](https://github.com/your-repo)")
