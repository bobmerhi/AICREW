import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS

# 1. Load API keys from Streamlit Secrets
os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

# 2. Page setup
st.set_page_config(page_title="Multi-Agent AI Portal", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Agent AI Command Portal")

with st.sidebar:
    st.header("Active Cloud Team ($0.00)")
    st.markdown("- **Manager & Researcher:** Gemini 3.5 Flash Lite")
    st.markdown("- **Lead Builder:** DeepSeek V4.1 Flash (NVIDIA NIM)")
    st.markdown("- **Quality Reviewer:** Llama 3.3 70B (OpenRouter)")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# 3. Free Web Search Skill
@tool("Free Web Search")
def web_search(query: str) -> str:
    """Searches the internet for up-to-date facts, docs, and solutions."""
    results = DDGS().text(query, max_results=5)
    return "\n".join([f"- {r['title']}: {r['body']} ({r['href']})" for r in results])

# 4. Build the Crew (cached so it loads fast)
def run_crew(user_prompt: str) -> str:
    manager_llm = LLM(
        model="openai/nvidia/nemotron-3-ultra-550b-a55b",
        base_url="https://nvidia.com",
        api_key=os.environ["NVIDIA_API_KEY"]
    )
    coder_llm = LLM(
        model="openai/deepseek-ai/deepseek-v4.1-flash",
        base_url="https://nvidia.com",
        api_key=os.environ["NVIDIA_API_KEY"]
    )
    reviewer_llm = LLM(
        model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        base_url="https://openrouter.ai",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )

    researcher = Agent(
        role="Web Researcher",
        goal="Find accurate, up-to-date information and documentation online",
        backstory="Expert researcher who verifies facts and APIs using live web search.",
        tools=[web_search],
        llm=manager_llm
    )
    builder = Agent(
        role="Lead Engineer & Creator",
        goal="Build complete, working code, plans, or workflows based on research",
        backstory="Senior developer skilled in turning ideas into clean, ready-to-use solutions.",
        llm=coder_llm
    )
    reviewer = Agent(
        role="Quality & Logic Reviewer",
        goal="Check work for bugs, missing steps, or errors and polish the final output",
        backstory="Detail-oriented reviewer who ensures everything works smoothly.",
        llm=reviewer_llm
    )

    task = Task(
        description=f"Complete this request using the best team members and skills available: {user_prompt}",
        expected_output="A clear, verified final answer with ready-to-use code or steps."
    )

    crew = Crew(
        agents=[researcher, builder, reviewer],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=manager_llm
    )
    return str(crew.kickoff())

# 5. Chat History & Input Box
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your multi-agent team anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Manager is delegating to Researcher, Builder, and Reviewer..."):
            response = run_crew(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
