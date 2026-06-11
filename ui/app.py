import os
from pathlib import Path
from huggingface_hub import snapshot_download

_chroma_path = Path("chroma_db")
if not _chroma_path.exists() or not any(_chroma_path.iterdir()):
    print("Downloading ChromaDB from HuggingFace...")
    snapshot_download(
        repo_id="sadhanasahu/acmecorp-chroma-db",
        repo_type="dataset",
        local_dir="chroma_db",
    )
    print("ChromaDB downloaded successfully.")

"""
Streamlit UI — Institutional Memory Bot (ChromaDB + Neo4j dual-store).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from pathlib import Path

st.set_page_config(page_title="AcmeCorp Memory Bot", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.source-badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;margin:2px 3px}
.badge-confluence{background:#E8F4F8;color:#0052CC}
.badge-slack{background:#F0E6FF;color:#4A154B}
.badge-github{background:#E8F5E9;color:#1B5E20}
.badge-jira{background:#E3F2FD;color:#0D47A1}
.badge-email{background:#FFF3E0;color:#E65100}
.badge-meetings{background:#FCE4EC;color:#880E4F}
.chat-user{background:#F0F2F6;border-radius:12px;padding:12px 16px;margin:8px 0}
.chat-bot{background:#EEF2FF;border-radius:12px;padding:12px 16px;margin:8px 0}
.source-box{background:#FAFAFA;border:1px solid #E0E0E0;border-radius:8px;padding:10px 14px;margin:4px 0;font-size:13px}
.graph-box{background:#F0FFF4;border:1px solid #C6F6D5;border-radius:8px;padding:10px 14px;margin:4px 0;font-size:12px;font-family:monospace}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 AcmeCorp Memory Bot")
    st.caption("Ask anything about decisions, policies, people, and incidents.")
    st.divider()

    st.markdown("### 🗃️ Knowledge stores")
    col1, col2 = st.columns(2)
    col1.metric("Vector (Chroma)", "6 sources")
    col2.metric("Graph (Neo4j)", "8 decisions")

    st.markdown("### 📚 Sources indexed")
    for src, count in [("📄 Confluence","3 pages"),("💬 Slack","4 channels"),
                       ("🐙 GitHub","3 PRs/issues"),("📋 Jira","5 tickets"),
                       ("📧 Email","4 threads"),("🎙️ Meetings","3 transcripts")]:
        c1, c2 = st.columns([2,1])
        c1.caption(src); c2.caption(count)

    st.divider()
    st.markdown("### 💡 Sample questions")
    for q in [
        "How do we handle GDPR deletion requests?",
        "Why did we go multi-provider for payments?",
        "Who do I ask about infrastructure access?",
        "What caused the November 2023 outage?",
        "Who opposed the multi-provider decision and why?",
        "What decisions has James Okafor made?",
        "What is the data retention policy?",
        "How do I deploy to production?",
    ]:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["prefill"] = q

    st.divider()
    show_graph = st.toggle("Show graph context", value=False)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.update({"messages": [], "chat_history": []})
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("messages",[]), ("chat_history",[]), ("agent_ready",False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Load agent ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agent():
    from rag.agent import initialize
    import os
    initialize(
        chroma_dir=Path("chroma_db"),
        neo4j_uri=os.environ.get("NEO4J_URI", "neo4j://localhost:7687"),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_password=os.environ.get("NEO4J_PASSWORD", "password"),
    )
    return True

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🧠 AcmeCorp Institutional Memory")
st.caption("Powered by ChromaDB (semantic search) + Neo4j (knowledge graph) + Mistral-7B")

if not st.session_state["agent_ready"]:
    with st.spinner("Loading AI model and knowledge bases... (1-2 min on GPU)"):
        try:
            load_agent()
            st.session_state["agent_ready"] = True
        except Exception as e:
            st.error(f"Failed to load: {e}")
            st.stop()

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">🧑 **You:** {msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bot">🧠 **Memory Bot:**\n\n{msg["content"]}</div>',
                    unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} document source(s)", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-box">'
                        f'<span class="source-badge badge-{src["source"]}">{src["source"].upper()}</span> '
                        f'<strong>{src["identifier"]}</strong><br>'
                        f'<span style="color:#666;font-size:12px">{src["snippet"]}</span></div>',
                        unsafe_allow_html=True)
        if show_graph and msg.get("graph_context"):
            with st.expander("🕸️ Knowledge graph context used", expanded=False):
                st.markdown(f'<div class="graph-box">{msg["graph_context"]}</div>',
                            unsafe_allow_html=True)

# Input
prefill  = st.session_state.pop("prefill", "")
question = st.chat_input("Ask about a policy, decision, person, or incident...")
if prefill and not question:
    question = prefill

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    st.markdown(f'<div class="chat-user">🧑 **You:** {question}</div>', unsafe_allow_html=True)

    with st.spinner("Searching vector store + knowledge graph..."):
        from rag.agent import ask
        result = ask(question, chat_history=st.session_state["chat_history"])

    answer        = result["answer"]
    sources       = result["sources"]
    graph_context = result.get("graph_context", "")

    st.markdown(f'<div class="chat-bot">🧠 **Memory Bot:**\n\n{answer}</div>',
                unsafe_allow_html=True)

    if sources:
        with st.expander(f"📚 {len(sources)} document source(s)", expanded=False):
            for src in sources:
                st.markdown(
                    f'<div class="source-box">'
                    f'<span class="source-badge badge-{src["source"]}">{src["source"].upper()}</span> '
                    f'<strong>{src["identifier"]}</strong><br>'
                    f'<span style="color:#666;font-size:12px">{src["snippet"]}</span></div>',
                    unsafe_allow_html=True)

    if show_graph and graph_context:
        with st.expander("🕸️ Knowledge graph context used", expanded=False):
            st.markdown(f'<div class="graph-box">{graph_context}</div>', unsafe_allow_html=True)

    st.session_state["messages"].append({
        "role": "assistant", "content": answer,
        "sources": sources, "graph_context": graph_context,
    })
    st.session_state["chat_history"].append({"user": question, "assistant": answer})
    if len(st.session_state["chat_history"]) > 10:
        st.session_state["chat_history"] = st.session_state["chat_history"][-10:]
