import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import gradio as gr

# ── Auto-download ChromaDB from HF Dataset ───────────────────────────────────
_chroma_path = Path("chroma_db")
if not _chroma_path.exists() or not any(_chroma_path.iterdir()):
    print("Downloading ChromaDB...")
    snapshot_download(
        repo_id="Sadhanasahu123/acmecorp-chroma-db",
        repo_type="dataset",
        local_dir="chroma_db",
    )
    print("ChromaDB ready.")

# ── Load agent ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag.agent import initialize, ask

print("Initializing agent...")
initialize(
    chroma_dir=Path("chroma_db"),
    neo4j_uri=os.environ.get("NEO4J_URI", ""),
    neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
    neo4j_password=os.environ.get("NEO4J_PASSWORD", ""),
)
print("Agent ready!")

# ── Chat function ─────────────────────────────────────────────────────────────
def chat(message, history):
    # Convert gradio history format to our format
    chat_history = []
    for human, assistant in history:
        chat_history.append({"user": human, "assistant": assistant})
    
    result = ask(message, chat_history=chat_history)
    answer = result["answer"]
    
    # Append sources to answer
    if result["sources"]:
        sources_text = "\n\n**Sources:** " + " | ".join(
            f"{s['source'].upper()}/{s['identifier']}" 
            for s in result["sources"]
        )
        answer += sources_text

    # Append graph context if available  
    if result.get("graph_context"):
        answer += f"\n\n**Graph context:** {result['graph_context'][:300]}"
    
    return answer

# ── Sample questions ──────────────────────────────────────────────────────────
examples = [
    "Who decided to go multi-provider for payments and who opposed it?",
    "How do we handle GDPR deletion requests?",
    "What caused the November 2023 outage?",
    "Who do I contact about infrastructure access?",
    "What is the data retention policy for logs?",
    "What decisions has James Okafor made?",
]

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="AcmeCorp Memory Bot") as demo:
    gr.Markdown("""
    # 🧠 AcmeCorp Institutional Memory Bot
    Ask anything about company decisions, policies, people, and incidents.
    Powered by **ChromaDB** (vector search) + **Neo4j** (knowledge graph) + **TinyLlama**
    """)

    chatbot = gr.Chatbot(
        height=500,
        placeholder="Ask me about any company decision, policy, or person...",
        show_label=False,
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="e.g. Why did we switch to multi-provider payments?",
            show_label=False,
            scale=4,
        )
        submit_btn = gr.Button("Ask 🧠", variant="primary", scale=1)

    gr.Examples(examples=examples, inputs=msg)

    gr.Markdown("""
    ---
    **Sources indexed:** Confluence · Slack · GitHub · Jira · Email · Meeting transcripts
    """)

    # Wire up
    msg.submit(chat, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)
    submit_btn.click(chat, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)

if __name__ == "__main__":
    demo.launch()
