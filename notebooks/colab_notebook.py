# ============================================================
# INSTITUTIONAL MEMORY BOT — GOOGLE COLAB NOTEBOOK
# Run each cell top to bottom. Use Runtime > Run All.
# Required: Runtime > Change runtime type > T4 GPU
# ============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1: Check GPU
# ─────────────────────────────────────────────────────────────
import subprocess
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else "⚠️  No GPU detected! Go to Runtime > Change runtime type > T4 GPU")

# ─────────────────────────────────────────────────────────────
# CELL 2: Install all dependencies
# ─────────────────────────────────────────────────────────────
# %%capture
subprocess.run([
    "pip", "install", "-q",
    "langchain==0.2.16",
    "langchain-community==0.2.16",
    "langgraph==0.2.28",
    "langchain-core==0.2.38",
    "transformers==4.44.2",
    "sentence-transformers==3.1.1",
    "accelerate==0.34.2",
    "bitsandbytes==0.43.3",
    "chromadb==0.5.5",
    "beautifulsoup4==4.12.3",
    "streamlit==1.38.0",
    "huggingface-hub==0.24.7",
    "pyngrok",
], check=True)
print("✅ All packages installed.")

# ─────────────────────────────────────────────────────────────
# CELL 3: Clone or upload your project
# Option A — if using Google Drive (recommended)
# Option B — create files inline
# ─────────────────────────────────────────────────────────────
import os

# OPTION A: Mount Google Drive (recommended — files persist across sessions)
from google.colab import drive
drive.mount("/content/drive")
PROJECT_DIR = "/content/drive/MyDrive/institutional_memory_bot"

# OPTION B: Use local Colab storage (lost when session ends)
# PROJECT_DIR = "/content/institutional_memory_bot"

os.makedirs(PROJECT_DIR, exist_ok=True)
os.chdir(PROJECT_DIR)
print(f"✅ Working directory: {os.getcwd()}")

# ─────────────────────────────────────────────────────────────
# CELL 4: Create the project folder structure
# ─────────────────────────────────────────────────────────────
folders = [
    "synthetic_data/confluence",
    "synthetic_data/slack",
    "synthetic_data/github",
    "synthetic_data/jira",
    "synthetic_data/email",
    "synthetic_data/meetings",
    "ingestion",
    "rag",
    "ui",
    "chroma_db",
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create __init__.py files
for pkg in ["ingestion", "rag", "ui"]:
    open(f"{pkg}/__init__.py", "w").close()

print("✅ Folder structure created.")

# ─────────────────────────────────────────────────────────────
# CELL 5: Upload your project files
# Upload these files from your computer using the Colab file
# browser (folder icon on the left), or run this cell to
# auto-create them from GitHub / Google Drive.
#
# Files to upload into PROJECT_DIR:
#   synthetic_data/confluence/  → all .html files
#   synthetic_data/slack/       → channels_export.json
#   synthetic_data/github/      → issues_and_prs.json
#   synthetic_data/jira/        → tickets.json
#   synthetic_data/email/       → emails.json
#   synthetic_data/meetings/    → transcripts.json
#   ingestion/ingest.py
#   rag/agent.py
#   ui/app.py
#
# TIP: The easiest way is to zip all files on your computer
# and upload the zip, then run:
# import zipfile
# with zipfile.ZipFile("project.zip") as z:
#     z.extractall(".")
# ─────────────────────────────────────────────────────────────
print("📂 Verify files exist:")
for f in ["ingestion/ingest.py", "rag/agent.py", "ui/app.py",
          "synthetic_data/slack/channels_export.json"]:
    exists = "✅" if os.path.exists(f) else "❌ MISSING"
    print(f"  {exists} {f}")

# ─────────────────────────────────────────────────────────────
# CELL 6: Run ingestion pipeline
# This loads all synthetic data, chunks it, embeds with
# sentence-transformers (free), and stores in ChromaDB.
# Takes ~2-3 minutes on first run.
# ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, PROJECT_DIR)

from ingestion.ingest import run_ingestion
from pathlib import Path

vectorstore = run_ingestion(
    data_dir=Path("synthetic_data"),
    chroma_dir=Path("chroma_db"),
)
print(f"\n✅ Ingestion complete! Vectors in DB: {vectorstore._collection.count()}")

# ─────────────────────────────────────────────────────────────
# CELL 7: Test retrieval (sanity check before loading LLM)
# ─────────────────────────────────────────────────────────────
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
)
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="institutional_memory",
)
test_query = "How do we handle GDPR deletion requests?"
results = db.similarity_search(test_query, k=3)
print(f"Query: '{test_query}'\n")
for i, doc in enumerate(results):
    print(f"Result {i+1} [{doc.metadata.get('source','?').upper()}]:")
    print(doc.page_content[:200])
    print("---")

# ─────────────────────────────────────────────────────────────
# CELL 8: HuggingFace login (needed to download Mistral)
# Get your free token at https://huggingface.co/settings/tokens
# You must also accept Mistral's license at:
# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
# ─────────────────────────────────────────────────────────────
from huggingface_hub import login
# Paste your HuggingFace token below (read-only token is fine)
HF_TOKEN = "hf_YOUR_TOKEN_HERE"   # ← replace this
login(token=HF_TOKEN)
print("✅ Logged in to HuggingFace.")

# ─────────────────────────────────────────────────────────────
# CELL 9: Initialize the full RAG agent
# Downloads Mistral-7B-Instruct (~14GB) on first run.
# Uses 4-bit quantization so it fits in T4's 16GB VRAM.
# Takes 5-10 minutes on first run (cached after that).
# ─────────────────────────────────────────────────────────────
from rag.agent import initialize
initialize(chroma_dir=Path("chroma_db"))
print("✅ Agent ready!")

# ─────────────────────────────────────────────────────────────
# CELL 10: Test the agent directly (no UI)
# ─────────────────────────────────────────────────────────────
from rag.agent import ask

test_questions = [
    "How do we handle GDPR deletion requests?",
    "Why did we decide to go multi-provider for payments?",
    "Who should I talk to about infrastructure access?",
    "What caused the November 2023 outage?",
]

for q in test_questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    result = ask(q)
    print(f"A: {result['answer'][:500]}...")
    print(f"Sources: {[s['source']+'/'+s['identifier'] for s in result['sources']]}")

# ─────────────────────────────────────────────────────────────
# CELL 11: Launch Streamlit with ngrok tunnel
# ngrok gives you a public URL to access the app.
# Get a free ngrok token at https://ngrok.com (free tier)
# ─────────────────────────────────────────────────────────────
from pyngrok import ngrok
import threading

NGROK_TOKEN = "YOUR_NGROK_TOKEN_HERE"   # ← replace this
ngrok.set_auth_token(NGROK_TOKEN)

# Write a small launcher script
launcher = f"""
import sys
sys.path.insert(0, "{PROJECT_DIR}")
exec(open("{PROJECT_DIR}/ui/app.py").read())
"""

def run_streamlit():
    os.system(f"cd {PROJECT_DIR} && streamlit run ui/app.py --server.port 8501 --server.headless true &")

thread = threading.Thread(target=run_streamlit)
thread.start()

import time
time.sleep(5)  # wait for streamlit to start

public_url = ngrok.connect(8501)
print(f"\n{'='*60}")
print(f"🚀 Your app is live at: {public_url}")
print(f"{'='*60}")
print("Open the link above in your browser!")
print("Share it with anyone — it's publicly accessible for this session.")

# ─────────────────────────────────────────────────────────────
# CELL 12 (OPTIONAL): Deploy to Streamlit Cloud permanently
# For permanent free hosting, push your code to GitHub and
# deploy at https://share.streamlit.io
# Steps:
#  1. Create a GitHub repo
#  2. Push all your project files
#  3. Add a secrets.toml for HF token (optional)
#  4. Go to share.streamlit.io, connect repo, set main file to ui/app.py
# NOTE: Streamlit Cloud is CPU-only free tier.
#       For CPU, change LLM_MODEL in rag/agent.py to a smaller model:
#       "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (works on CPU, less capable)
# ─────────────────────────────────────────────────────────────
print("""
For permanent Streamlit Cloud deployment (CPU, free):
  1. Push project to GitHub
  2. In rag/agent.py, change LLM_MODEL to:
     "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  3. Go to https://share.streamlit.io and connect your repo
""")
