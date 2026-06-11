# Complete Deployment Guide
# Institutional Memory Bot — From Zero to Live
# ================================================
# HuggingFace Space URL after deploy:
#   https://huggingface.co/spaces/YOUR_USERNAME/institutional-memory-bot
# ================================================


# ══════════════════════════════════════════════════
# PHASE 1 — ACCOUNTS YOU NEED (all free, 15 min)
# ══════════════════════════════════════════════════

"""
1. HuggingFace account
   → https://huggingface.co/join
   → After signup: Settings → Access Tokens → New Token
     Name: "memory-bot", Role: "write" (write needed to push to Spaces)
   → SAVE THIS TOKEN — you'll use it in Colab and as a Space secret

2. Accept Mistral-7B license (required to download the model)
   → https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
   → Click "Agree and access repository"

3. Neo4j AuraDB (free cloud graph database)
   → https://neo4j.com/cloud/aura-free/
   → Sign up → Create Free Instance
   → Choose region closest to you
   → SAVE the credentials shown ONCE:
       URI:      neo4j+s://XXXXXXXX.databases.neo4j.io
       Username: neo4j
       Password: (auto-generated — copy it NOW, not shown again)
   → Instance takes ~2 min to start (status turns green)

4. GitHub account (to push code and link to HF Spaces)
   → https://github.com/join
   → Create new repo: "institutional-memory-bot" (public, no README)

5. ngrok (for Colab public URL during development)
   → https://ngrok.com → Sign up free
   → Dashboard → Your Authtoken → copy it
"""


# ══════════════════════════════════════════════════
# PHASE 2 — GOOGLE COLAB SETUP (run data pipeline)
# ══════════════════════════════════════════════════

# ─── COLAB CELL 1: Verify GPU ────────────────────
"""
Go to: Runtime > Change runtime type > T4 GPU > Save
Then run:
"""
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
      or "NO GPU — go to Runtime > Change runtime type > T4 GPU")


# ─── COLAB CELL 2: Install packages ──────────────
# %%capture
import subprocess
subprocess.run(["pip", "install", "-q",
    "langchain==0.2.16", "langchain-community==0.2.16",
    "langgraph==0.2.28", "langchain-core==0.2.38",
    "transformers==4.44.2", "sentence-transformers==3.1.1",
    "accelerate==0.34.2", "bitsandbytes==0.43.3",
    "chromadb==0.5.5", "neo4j==5.23.1",
    "beautifulsoup4==4.12.3", "streamlit==1.38.0",
    "huggingface-hub==0.24.7", "pyngrok", "gitpython",
])
print("✅ Packages installed")


# ─── COLAB CELL 3: Mount Drive and set project dir ──
from google.colab import drive
drive.mount("/content/drive")

import os, sys
PROJECT_DIR = "/content/drive/MyDrive/institutional_memory_bot"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
print(f"Working directory: {os.getcwd()}")


# ─── COLAB CELL 4: Upload project files ──────────
"""
OPTION A — Upload zip (easiest):
1. Zip your entire "institutional_memory_bot" folder on your computer
2. Upload it to Google Drive root
3. Run this cell:
"""
import zipfile, shutil
zip_path = "/content/drive/MyDrive/institutional_memory_bot.zip"
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        z.extractall("/content/drive/MyDrive/")
    print("✅ Unzipped")
else:
    print("No zip found — make sure files are already in the project folder")

# Verify
for f in ["ingestion/ingest.py", "ingestion/build_graph.py",
          "rag/agent.py", "ui/app.py",
          "synthetic_data/slack/channels_export.json"]:
    status = "✅" if os.path.exists(f) else "❌ MISSING"
    print(f"{status} {f}")

# Create __init__.py files
for pkg in ["ingestion", "rag", "ui"]:
    os.makedirs(pkg, exist_ok=True)
    open(f"{pkg}/__init__.py", "w").close()


# ─── COLAB CELL 5: Run ChromaDB ingestion ────────
from ingestion.ingest import run_ingestion
from pathlib import Path

vectorstore = run_ingestion(
    data_dir=Path("synthetic_data"),
    chroma_dir=Path("chroma_db"),
)
print(f"✅ ChromaDB ready: {vectorstore._collection.count()} vectors")


# ─── COLAB CELL 6: Build Neo4j knowledge graph ───
"""
IMPORTANT: Replace with YOUR AuraDB credentials from Phase 1 Step 3.
"""
from ingestion.build_graph import build_full_graph

# ← Replace these with your AuraDB values
NEO4J_URI      = "neo4j+s://XXXXXXXX.databases.neo4j.io"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "YOUR_AURA_PASSWORD"

# Patch the config in build_graph.py at runtime
import ingestion.build_graph as bg
bg.NEO4J_URI      = NEO4J_URI
bg.NEO4J_USER     = NEO4J_USER
bg.NEO4J_PASSWORD = NEO4J_PASSWORD

build_full_graph(data_dir=Path("synthetic_data"))
print("✅ Neo4j graph built")


# ─── COLAB CELL 7: Test retrieval only (fast sanity check) ──
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                             model_kwargs={"device":"cuda"})
db  = Chroma(persist_directory="chroma_db", embedding_function=emb,
             collection_name="institutional_memory")
results = db.similarity_search("GDPR deletion policy", k=3)
print("Top 3 results:")
for i, r in enumerate(results):
    print(f"  {i+1}. [{r.metadata['source']}] {r.page_content[:100]}...")


# ─── COLAB CELL 8: HuggingFace login ─────────────
from huggingface_hub import login
HF_TOKEN = "hf_YOUR_TOKEN_HERE"   # ← paste your HF token
login(token=HF_TOKEN)
print("✅ HuggingFace logged in")


# ─── COLAB CELL 9: Load full agent (downloads Mistral ~14GB) ──
"""
First run: ~7-10 min to download Mistral.
Subsequent runs: ~2 min (cached in HF hub cache on Drive).
"""
import rag.agent as agent_module
agent_module.NEO4J_URI      = NEO4J_URI
agent_module.NEO4J_USER     = NEO4J_USER
agent_module.NEO4J_PASSWORD = NEO4J_PASSWORD

from rag.agent import initialize
initialize(chroma_dir=Path("chroma_db"),
           neo4j_uri=NEO4J_URI,
           neo4j_user=NEO4J_USER,
           neo4j_password=NEO4J_PASSWORD)
print("✅ Agent ready")


# ─── COLAB CELL 10: Test agent ────────────────────
from rag.agent import ask

test_questions = [
    "Who decided to go multi-provider for payments and who opposed it?",
    "How do we handle GDPR deletion requests?",
    "What caused the November 2023 outage?",
    "Who should I contact about infrastructure access?",
]
for q in test_questions:
    print(f"\nQ: {q}")
    result = ask(q)
    print(f"A: {result['answer'][:400]}...")
    print(f"Graph ctx: {result['graph_context'][:200] if result['graph_context'] else 'none'}...")


# ─── COLAB CELL 11: Run with ngrok (development preview) ──
"""
For quick preview in browser before pushing to HF Spaces.
"""
from pyngrok import ngrok
import threading, time

NGROK_TOKEN = "YOUR_NGROK_TOKEN"   # ← from ngrok dashboard
ngrok.set_auth_token(NGROK_TOKEN)

# Pass Neo4j creds as environment variables
os.environ["NEO4J_URI"]      = NEO4J_URI
os.environ["NEO4J_USER"]     = NEO4J_USER
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

def run_streamlit():
    os.system(f"cd {PROJECT_DIR} && streamlit run ui/app.py "
              f"--server.port 8501 --server.headless true &")

threading.Thread(target=run_streamlit).start()
time.sleep(6)
url = ngrok.connect(8501)
print(f"\n{'='*50}\n🚀 Live at: {url}\n{'='*50}")


# ══════════════════════════════════════════════════
# PHASE 3 — PUSH CODE TO GITHUB
# ══════════════════════════════════════════════════

# ─── COLAB CELL 12: Push to GitHub ───────────────
"""
First, create a .gitignore so we don't push the chroma_db or model cache.
"""
gitignore = """
chroma_db/
__pycache__/
*.pyc
.env
*.zip
.DS_Store
"""
open(f"{PROJECT_DIR}/.gitignore", "w").write(gitignore)

"""
Then run these in a Colab bash cell (prefix with !):

! cd /content/drive/MyDrive/institutional_memory_bot && \
  git init && \
  git remote add origin https://YOUR_GITHUB_USERNAME:YOUR_GITHUB_PAT@github.com/YOUR_GITHUB_USERNAME/institutional-memory-bot.git && \
  git add . && \
  git commit -m "Initial commit: Institutional Memory Bot" && \
  git branch -M main && \
  git push -u origin main

Replace:
  YOUR_GITHUB_USERNAME  → your GitHub username
  YOUR_GITHUB_PAT       → GitHub Personal Access Token
                          (Settings → Developer Settings → Personal Access Tokens → Classic → New)
                          Give it: repo permissions
"""


# ══════════════════════════════════════════════════
# PHASE 4 — HUGGINGFACE SPACES DEPLOYMENT
# ══════════════════════════════════════════════════

"""
STEP 1 — Create a new Space
  → Go to: https://huggingface.co/new-space
  → Space name: institutional-memory-bot
  → License: MIT
  → SDK: Streamlit          ← important
  → Hardware: CPU Basic (free) or GPU (paid, needed for Mistral)
     ⚠️  NOTE on free tier:
       Free CPU tier can't run Mistral-7B (not enough RAM).
       TWO OPTIONS:
         A) Use a smaller CPU-compatible LLM (see Cell 14 below) — free
         B) Upgrade to T4 Small GPU (~$0.60/hr, pay-as-you-go) — full Mistral
  → Visibility: Public (required for free tier)
  → Click "Create Space"

STEP 2 — Link your GitHub repo to the Space
  → In your new Space → Files → "..." → Settings
  → Scroll to "GitHub repository"
  → Connect to: YOUR_USERNAME/institutional-memory-bot
  → Branch: main
  → This auto-deploys every time you push to main

STEP 3 — Add secrets (your credentials — never hardcode these)
  → In your Space → Settings → Repository secrets → New secret

  Add these 3 secrets:
    Name: NEO4J_URI        Value: neo4j+s://XXXXXXXX.databases.neo4j.io
    Name: NEO4J_USER       Value: neo4j
    Name: NEO4J_PASSWORD   Value: YOUR_AURA_PASSWORD

  The app.py already reads these via os.environ.get("NEO4J_URI", ...)
  so no code changes needed.

STEP 4 — Add the HF Space README (required for Space config)
  Rename README_HF.md to README.md before pushing.
  The YAML block at the top tells HF Spaces which file to run.

STEP 5 — Push the ChromaDB to the Space as a dataset
  The chroma_db/ folder (your vector embeddings) can't be generated
  at Space startup because there's no GPU. You need to upload it.

  Option A (Recommended) — Upload as HF Dataset:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo("YOUR_USERNAME/acmecorp-chroma-db",
                    repo_type="dataset", private=False)
    api.upload_folder(
        folder_path="chroma_db",
        repo_id="YOUR_USERNAME/acmecorp-chroma-db",
        repo_type="dataset",
    )

  Then in your app.py startup, add this download step:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="YOUR_USERNAME/acmecorp-chroma-db",
        repo_type="dataset",
        local_dir="chroma_db",
    )

  Option B — Commit chroma_db directly to the Space repo
    (only if under 1GB — HF has a 10GB LFS limit on free tier)
    git add chroma_db/
    git lfs track "*.bin" "*.sqlite3"
    git commit -m "Add ChromaDB"
    git push

STEP 6 — Handle model loading on HF Spaces
  On a free CPU Space, Mistral-7B won't fit.
  Use this in agent.py instead for free CPU deployment:
    LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  And remove the BitsAndBytesConfig (no 4-bit on CPU):
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="cpu")
  First startup downloads ~2.2GB — HF caches it between restarts.
"""


# ─── COLAB CELL 13: Upload ChromaDB to HF Dataset ──
from huggingface_hub import HfApi
from pathlib import Path

HF_USERNAME = "YOUR_HF_USERNAME"   # ← replace
api = HfApi()

# Create dataset repo
try:
    api.create_repo(f"{HF_USERNAME}/acmecorp-chroma-db",
                    repo_type="dataset", private=False)
    print("Dataset repo created.")
except Exception:
    print("Dataset repo already exists.")

# Upload chroma_db folder
api.upload_folder(
    folder_path=str(Path(PROJECT_DIR) / "chroma_db"),
    repo_id=f"{HF_USERNAME}/acmecorp-chroma-db",
    repo_type="dataset",
)
print(f"✅ ChromaDB uploaded to hf.co/datasets/{HF_USERNAME}/acmecorp-chroma-db")


# ─── COLAB CELL 14: Patch app.py for free CPU Space ──
"""
If using free CPU tier (no GPU), patch these two files:

1. In rag/agent.py, change _build_llm() to:

def _build_llm():
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok   = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="cpu")
    pipe  = pipeline("text-generation", model=model, tokenizer=tok,
                     max_new_tokens=256, temperature=0.1, do_sample=True)
    from langchain_community.llms import HuggingFacePipeline
    return HuggingFacePipeline(pipeline=pipe)

2. Add ChromaDB auto-download at the top of ui/app.py (before imports):

import os
from huggingface_hub import snapshot_download
from pathlib import Path

chroma_path = Path("chroma_db")
if not chroma_path.exists() or not any(chroma_path.iterdir()):
    print("Downloading ChromaDB from HuggingFace...")
    snapshot_download(
        repo_id="YOUR_HF_USERNAME/acmecorp-chroma-db",
        repo_type="dataset",
        local_dir="chroma_db",
    )
    print("ChromaDB downloaded.")
"""


# ══════════════════════════════════════════════════
# PHASE 5 — FINAL PUSH AND VERIFY
# ══════════════════════════════════════════════════

"""
Final checklist before your last git push:

PROJECT ROOT should contain:
  ├── README.md          (renamed from README_HF.md — has the YAML header)
  ├── requirements.txt   (includes neo4j==5.23.1)
  ├── .gitignore         (excludes chroma_db/, __pycache__, .env)
  ├── synthetic_data/
  │   ├── confluence/    (3 HTML files)
  │   ├── slack/         (channels_export.json)
  │   ├── github/        (issues_and_prs.json)
  │   ├── jira/          (tickets.json)
  │   ├── email/         (emails.json)
  │   └── meetings/      (transcripts.json)
  ├── ingestion/
  │   ├── __init__.py
  │   ├── ingest.py
  │   └── build_graph.py
  ├── rag/
  │   ├── __init__.py
  │   └── agent.py       (dual-store: ChromaDB + Neo4j)
  └── ui/
      ├── __init__.py
      └── app.py         (with chroma_db auto-download block)

Then push:
  ! cd PROJECT_DIR && git add -A && git commit -m "Deploy to HF Spaces" && git push

Your Space will auto-rebuild (takes 3-5 min for first build).
Watch the build log at: https://huggingface.co/spaces/YOUR_USERNAME/institutional-memory-bot/logs

Your live URL:
  https://huggingface.co/spaces/YOUR_USERNAME/institutional-memory-bot
"""


# ══════════════════════════════════════════════════
# TROUBLESHOOTING
# ══════════════════════════════════════════════════
"""
❌ "RuntimeError: CUDA out of memory" in Colab
   → Runtime > Restart runtime > re-run from Cell 8
   → Or switch to TinyLlama (Cell 14 instructions)

❌ "Connection refused" for Neo4j
   → Check AuraDB instance is Running (not Paused) at console.neo4j.io
   → Free instances pause after 3 days of inactivity — resume them
   → Make sure URI starts with neo4j+s:// (not neo4j://) for AuraDB

❌ HF Space stuck at "Building"
   → Check Logs tab in your Space
   → Usually a requirements.txt version conflict
   → Try removing version pins: just write "langchain" instead of "langchain==0.2.16"

❌ "Module not found: rag.agent" in Space
   → Make sure __init__.py exists in rag/, ingestion/, ui/
   → Make sure app_file in README.md is "ui/app.py" not "app.py"

❌ ChromaDB empty on Space startup
   → The auto-download snippet (Cell 14 step 2) must run BEFORE initialize()
   → Check hf.co/datasets/YOUR_USERNAME/acmecorp-chroma-db has files

❌ Neo4j graph context shows "None" for all answers
   → AuraDB instance may be paused — resume it at console.neo4j.io
   → Check Space secrets: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD are set
   → The app falls back to vector-only mode gracefully if Neo4j is down

✅ Successful deploy looks like:
   → Space status: "Running"
   → App loads in ~90 seconds (model loading)
   → Sample question "Who decided multi-provider?" returns both
     the ChromaDB text AND Neo4j graph context (names, dates, context)
"""
