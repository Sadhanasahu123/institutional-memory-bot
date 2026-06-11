"""
Ingestion pipeline for Institutional Memory Bot.
Reads from synthetic data files, chunks documents,
generates embeddings with a free model, stores in ChromaDB.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("synthetic_data")
CHROMA_DIR = Path("chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, fast, ~80MB

# ── Text Splitter ─────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ── Source Connectors ─────────────────────────────────────────────────────────

def load_confluence(directory: Path) -> List[Document]:
    """
    Loads Confluence HTML exports.
    In production: swap directory read for Confluence REST API call.
    """
    docs = []
    for html_file in directory.glob("*.html"):
        soup = BeautifulSoup(html_file.read_text(encoding="utf-8"), "html.parser")
        meta = soup.find("div", class_="page-metadata")
        author   = meta.find("span", class_="author").get_text(strip=True) if meta else "Unknown"
        modified = meta.find("span", class_="last-modified").get_text(strip=True) if meta else ""
        page_id  = meta.find("span", class_="page-id").get_text(strip=True) if meta else ""

        # Remove metadata div before extracting body text
        if meta:
            meta.decompose()
        text = soup.get_text(separator="\n", strip=True)

        docs.append(Document(
            page_content=text,
            metadata={
                "source": "confluence",
                "file": html_file.name,
                "author": author,
                "last_modified": modified,
                "page_id": page_id,
                "title": soup.find("h1").get_text(strip=True) if soup.find("h1") else html_file.stem,
            }
        ))
    print(f"[confluence] loaded {len(docs)} pages")
    return docs


def load_slack(directory: Path) -> List[Document]:
    """
    Loads Slack JSON channel export.
    Format: list of channels, each with a 'messages' array.
    In production: swap for Slack API conversations.history calls.
    """
    docs = []
    for json_file in directory.glob("*.json"):
        channels = json.loads(json_file.read_text(encoding="utf-8"))
        for channel in channels:
            channel_name = channel["channel"]
            # Group messages into thread chunks for better context
            threads: Dict[str, List[str]] = {}
            for msg in channel["messages"]:
                root = msg.get("thread_ts", msg["timestamp"])
                threads.setdefault(root, [])
                threads[root].append(
                    f"[{msg['user']} @ {msg['timestamp']}]: {msg['text']}"
                )
            for root_ts, thread_msgs in threads.items():
                text = f"Slack channel: #{channel_name}\n\n" + "\n".join(thread_msgs)
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": "slack",
                        "channel": channel_name,
                        "thread_ts": root_ts,
                        "participants": list({m["user"] for m in channel["messages"]
                                             if m.get("thread_ts", m["timestamp"]) == root_ts}),
                    }
                ))
    print(f"[slack] loaded {len(docs)} thread chunks")
    return docs


def load_github(directory: Path) -> List[Document]:
    """
    Loads GitHub issues/PRs from JSON.
    In production: swap for GitHub REST API /repos/{owner}/{repo}/issues calls.
    """
    docs = []
    for json_file in directory.glob("*.json"):
        items = json.loads(json_file.read_text(encoding="utf-8"))
        for item in items:
            item_type = item.get("type", "issue").upper()
            header = (
                f"GitHub {item_type} #{item['number']}: {item['title']}\n"
                f"Author: {item['author']} | State: {item['state']} | "
                f"Labels: {', '.join(item.get('labels', []))}\n\n"
            )
            body = item.get("body", "")
            comments_text = "\n".join(
                f"[{c['author']} @ {c['timestamp']}]: {c['body']}"
                for c in item.get("comments", [])
            )
            full_text = header + body
            if comments_text:
                full_text += f"\n\n--- Comments ---\n{comments_text}"

            docs.append(Document(
                page_content=full_text,
                metadata={
                    "source": "github",
                    "type": item_type,
                    "number": item["number"],
                    "title": item["title"],
                    "author": item["author"],
                    "state": item["state"],
                    "labels": item.get("labels", []),
                }
            ))
    print(f"[github] loaded {len(docs)} issues/PRs")
    return docs


def load_jira(directory: Path) -> List[Document]:
    """
    Loads Jira tickets from JSON.
    In production: swap for Jira REST API /rest/api/3/search calls.
    """
    docs = []
    for json_file in directory.glob("*.json"):
        tickets = json.loads(json_file.read_text(encoding="utf-8"))
        for ticket in tickets:
            header = (
                f"Jira {ticket['type']}: {ticket['issue_key']} - {ticket['summary']}\n"
                f"Status: {ticket['status']} | Priority: {ticket['priority']}\n"
                f"Assignee: {ticket['assignee']} | Reporter: {ticket['reporter']}\n\n"
            )
            body = ticket.get("description", "")
            comments_text = "\n".join(
                f"[{c['author']} @ {c['timestamp']}]: {c['body']}"
                for c in ticket.get("comments", [])
            )
            full_text = header + body
            if comments_text:
                full_text += f"\n\n--- Comments ---\n{comments_text}"

            docs.append(Document(
                page_content=full_text,
                metadata={
                    "source": "jira",
                    "issue_key": ticket["issue_key"],
                    "type": ticket["type"],
                    "status": ticket["status"],
                    "assignee": ticket["assignee"],
                    "labels": ticket.get("labels", []),
                }
            ))
    print(f"[jira] loaded {len(docs)} tickets")
    return docs


def load_email(directory: Path) -> List[Document]:
    """
    Loads email threads from JSON.
    In production: swap for Gmail/Outlook API calls.
    """
    docs = []
    # Group by thread
    threads: Dict[str, List[dict]] = {}
    for json_file in directory.glob("*.json"):
        emails = json.loads(json_file.read_text(encoding="utf-8"))
        for email in emails:
            tid = email.get("thread_id", email["message_id"])
            threads.setdefault(tid, [])
            threads[tid].append(email)

    for thread_id, messages in threads.items():
        messages.sort(key=lambda x: x["date"])
        subject = messages[0]["subject"]
        thread_text = f"Email thread: {subject}\n\n"
        for msg in messages:
            thread_text += (
                f"From: {msg['from']} | To: {', '.join(msg['to'])} | "
                f"Date: {msg['date']}\n{msg['body']}\n\n{'─'*40}\n\n"
            )
        docs.append(Document(
            page_content=thread_text,
            metadata={
                "source": "email",
                "thread_id": thread_id,
                "subject": subject,
                "participants": list({m["from"] for m in messages} |
                                     {e for m in messages for e in m["to"]}),
            }
        ))
    print(f"[email] loaded {len(docs)} email threads")
    return docs


def load_meetings(directory: Path) -> List[Document]:
    """
    Loads meeting transcripts from JSON.
    In production: swap for Notion/Fireflies/Otter.ai API calls.
    """
    docs = []
    for json_file in directory.glob("*.json"):
        meetings = json.loads(json_file.read_text(encoding="utf-8"))
        for meeting in meetings:
            header = (
                f"Meeting: {meeting['title']}\n"
                f"Date: {meeting['date']} | Duration: {meeting['duration_minutes']} min\n"
                f"Attendees: {', '.join(meeting['attendees'])}\n\n"
            )
            full_text = header + meeting.get("transcript", "")
            docs.append(Document(
                page_content=full_text,
                metadata={
                    "source": "meetings",
                    "meeting_id": meeting["meeting_id"],
                    "title": meeting["title"],
                    "date": meeting["date"],
                    "attendees": meeting["attendees"],
                }
            ))
    print(f"[meetings] loaded {len(docs)} meeting transcripts")
    return docs


# ── Main ingestion ────────────────────────────────────────────────────────────

def run_ingestion(data_dir: Path = DATA_DIR, chroma_dir: Path = CHROMA_DIR):
    print("=" * 60)
    print("Starting ingestion pipeline...")
    print("=" * 60)

    # 1. Load from all sources
    all_docs: List[Document] = []
    loaders = {
        "confluence": load_confluence,
        "slack":      load_slack,
        "github":     load_github,
        "jira":       load_jira,
        "email":      load_email,
        "meetings":   load_meetings,
    }
    for source, loader_fn in loaders.items():
        source_dir = data_dir / source
        if source_dir.exists():
            all_docs.extend(loader_fn(source_dir))
        else:
            print(f"[{source}] directory not found, skipping")

    print(f"\nTotal documents loaded: {len(all_docs)}")

    # 2. Chunk documents
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks after splitting: {len(chunks)}")

    # 3. Load embedding model (downloads on first run, ~80MB)
    print(f"\nLoading embedding model: {EMBED_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda" if _cuda_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Embedding model loaded.")

    # 4. Store in ChromaDB
    print(f"\nStoring {len(chunks)} chunks in ChromaDB at '{chroma_dir}'...")
    CHROMA_DIR.mkdir(exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
        collection_name="institutional_memory",
    )
    print(f"Done! ChromaDB collection ready with {vectorstore._collection.count()} vectors.")
    return vectorstore


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    run_ingestion()
