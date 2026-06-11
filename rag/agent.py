"""
Dual-store RAG + LangGraph agent.
Node 1: retrieve from ChromaDB (semantic similarity)
Node 2: enrich with Neo4j (relationships, decisions, people)
Node 3: generate answer with Mistral-7B (4-bit quantized)
"""

from pathlib import Path
from typing import TypedDict, List, Annotated
import operator

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from neo4j import GraphDatabase

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR   = Path("chroma_db")
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL    = "mistralai/Mistral-7B-Instruct-v0.3"
TOP_K        = 6

NEO4J_URI      = "neo4j://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"

# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:        str
    retrieved_docs:  List[Document]
    graph_context:   str
    answer:          str
    sources:         List[dict]
    chat_history:    Annotated[List[dict], operator.add]

# ── Globals ───────────────────────────────────────────────────────────────────
_retriever    = None
_llm          = None
_neo4j_driver = None
_graph        = None

# ── Neo4j keyword-to-Cypher routing ──────────────────────────────────────────
GRAPH_QUERIES = [
    {
        "keywords": ["decided", "decision", "who chose", "why did", "reason", "chose to"],
        "cypher": """
            MATCH (p:Person)-[r:DECIDED]->(d:Decision)
            RETURN p.name AS person, d.summary AS decision,
                   r.date AS date, r.context AS context
            ORDER BY r.date DESC LIMIT 10
        """,
        "label": "Decisions and who made them",
    },
    {
        "keywords": ["opposed", "pushed back", "disagreed", "against", "argued"],
        "cypher": """
            MATCH (p:Person)-[r:OPPOSED]->(d:Decision)
            RETURN p.name AS person, d.summary AS decision, r.reason AS reason
            LIMIT 10
        """,
        "label": "People who opposed decisions",
    },
    {
        "keywords": ["outage", "incident", "caused", "post-mortem", "downtime", "led to"],
        "cypher": """
            MATCH (i:Incident)-[r:LED_TO]->(d:Decision)
            RETURN i.summary AS incident, i.date AS incident_date,
                   d.summary AS resulting_decision, r.context AS context
            LIMIT 5
        """,
        "label": "Incidents and their resulting decisions",
    },
    {
        "keywords": ["gdpr", "privacy", "deletion", "data retention", "personal data", "compliance"],
        "cypher": """
            MATCH (n)-[:ABOUT]->(t:Topic {name:'GDPR'})
            OPTIONAL MATCH (p:Person)-[:DECIDED]->(d:Decision)
            WHERE d.summary CONTAINS 'GDPR' OR d.summary CONTAINS 'retention'
               OR d.summary CONTAINS 'anonymi' OR d.summary CONTAINS 'deletion'
            RETURN collect(DISTINCT n.title)[..5] AS related_docs,
                   collect(DISTINCT {person: p.name, decision: d.summary, date: d.date})[..5] AS decisions
        """,
        "label": "GDPR-related documents and decisions",
    },
    {
        "keywords": ["novapay", "payment", "multi-provider", "stripe", "braintree"],
        "cypher": """
            MATCH (n)-[:ABOUT|IMPLEMENTS|REFERENCES*1..2]->(t)
            WHERE t.name = 'NovaPay v2.0' OR t.name = 'Payments'
               OR (t:Decision AND t.summary CONTAINS 'payment')
            RETURN DISTINCT n.title AS doc, n.source AS source, n.date AS date
            LIMIT 8
        """,
        "label": "NovaPay-related documents",
    },
    {
        "keywords": ["who", "person", "contact", "ask", "team", "expert", "responsible", "owns"],
        "cypher": """
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:DECIDED]->(d:Decision)
            OPTIONAL MATCH (p)-[:AUTHORED]->(doc:Document)
            RETURN p.name AS name, p.role AS role, p.team AS team,
                   collect(DISTINCT d.summary)[..3] AS decisions_made,
                   collect(DISTINCT doc.title)[..3] AS documents_authored
            LIMIT 8
        """,
        "label": "People, roles and ownership",
    },
    {
        "keywords": ["deploy", "deployment", "production", "approval", "policy", "process", "how to"],
        "cypher": """
            MATCH (p:Person)-[r:DECIDED]->(d:Decision)
            WHERE d.summary CONTAINS 'deploy' OR d.summary CONTAINS 'approval'
               OR d.summary CONTAINS 'policy'
            RETURN p.name, d.summary, r.date, r.context LIMIT 5
        """,
        "label": "Deployment and process decisions",
    },
    {
        "keywords": ["onboard", "new engineer", "access", "setup", "getting started"],
        "cypher": """
            MATCH (p:Person)-[:AUTHORED|OWNS]->(doc:Document)-[:ABOUT]->(t:Topic {name:'Onboarding'})
            RETURN p.name AS contact, doc.title AS document, doc.source AS source
            LIMIT 5
        """,
        "label": "Onboarding documents and contacts",
    },
]

def query_neo4j(question: str, driver) -> str:
    if driver is None:
        return ""
    question_lower = question.lower()
    matched = [q for q in GRAPH_QUERIES if any(kw in question_lower for kw in q["keywords"])]
    people_q = next(q for q in GRAPH_QUERIES if q["label"].startswith("People"))
    if people_q not in matched:
        matched.insert(0, people_q)
    matched = matched[:3]

    parts = []
    try:
        with driver.session() as session:
            for q in matched:
                results = list(session.run(q["cypher"]))
                if results:
                    parts.append(f"[Graph: {q['label']}]")
                    for r in results[:6]:
                        parts.append(f"  {dict(r)}")
    except Exception as e:
        parts.append(f"[Graph error: {e}]")
    return "\n".join(parts)

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT = PromptTemplate(
    input_variables=["vector_context", "graph_context", "question", "history"],
    template="""<s>[INST] You are the Institutional Memory Bot for AcmeCorp.
You have access to TWO knowledge sources — use both to give complete answers.

SOURCE 1 — Document text (vector search):
{vector_context}

SOURCE 2 — Knowledge graph (relationships, decisions, people):
{graph_context}

CONVERSATION HISTORY:
{history}

RULES:
1. Combine both sources — graph tells you WHO and WHY, documents tell you WHAT.
2. Always name the person responsible when answering about decisions.
3. Mention the date of decisions when available.
4. If someone opposed a decision, mention it — it's useful context.
5. Cite sources at the end.
6. If you don't know, say so clearly.

QUESTION: {question}

Answer clearly and concisely. End with a "Sources:" section. [/INST]""",
)

# ── LangGraph nodes ───────────────────────────────────────────────────────────
def retrieve_node(state: AgentState) -> AgentState:
    docs = _retriever.invoke(state["question"])
    return {**state, "retrieved_docs": docs}

def graph_enrich_node(state: AgentState) -> AgentState:
    ctx = query_neo4j(state["question"], _neo4j_driver)
    return {**state, "graph_context": ctx}

def generate_node(state: AgentState) -> AgentState:
    docs = state["retrieved_docs"]
    vector_parts, sources = [], []
    for i, doc in enumerate(docs):
        src   = doc.metadata.get("source", "unknown")
        ident = _get_identifier(doc.metadata)
        vector_parts.append(f"[Doc {i+1} — {src.upper()} / {ident}]\n{doc.page_content}")
        sources.append({"index": i+1, "source": src, "identifier": ident,
                        "snippet": doc.page_content[:120] + "..."})

    history_text = ""
    for turn in state.get("chat_history", [])[-4:]:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    prompt = PROMPT.format(
        vector_context="\n\n".join(vector_parts) or "None.",
        graph_context=state.get("graph_context", "") or "None.",
        question=state["question"],
        history=history_text or "None yet.",
    )
    raw    = _llm.invoke(prompt)
    answer = raw[len(prompt):].strip() if raw.startswith(prompt) else raw.strip()
    return {**state, "answer": answer, "sources": sources}

def _get_identifier(meta: dict) -> str:
    src = meta.get("source", "")
    if src == "confluence": return meta.get("page_id", meta.get("file", "?"))
    if src == "slack":      return f"#{meta.get('channel', '?')}"
    if src == "github":     return f"#{meta.get('number', '?')}"
    if src == "jira":       return meta.get("issue_key", "?")
    if src == "email":      return meta.get("subject", "?")[:40]
    if src == "meetings":   return meta.get("title", "?")[:40]
    return "?"

# ── Build LangGraph ───────────────────────────────────────────────────────────
def _build_langgraph():
    g = StateGraph(AgentState)
    g.add_node("retrieve",     retrieve_node)
    g.add_node("graph_enrich", graph_enrich_node)
    g.add_node("generate",     generate_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve",     "graph_enrich")
    g.add_edge("graph_enrich", "generate")
    g.add_edge("generate",     END)
    return g.compile()

# ── Public API ────────────────────────────────────────────────────────────────
def initialize(chroma_dir=CHROMA_DIR, neo4j_uri=NEO4J_URI,
               neo4j_user=NEO4J_USER, neo4j_password=NEO4J_PASSWORD):
    global _retriever, _llm, _neo4j_driver, _graph

    print("Loading retriever...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": _device()},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = Chroma(persist_directory=str(chroma_dir),
                embedding_function=embeddings,
                collection_name="institutional_memory")
    _retriever = vs.as_retriever(search_type="mmr",
                                 search_kwargs={"k": TOP_K, "fetch_k": 20})

    print("Connecting to Neo4j...")
    try:
        _neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        _neo4j_driver.verify_connectivity()
        print("Neo4j connected.")
    except Exception as e:
        print(f"Neo4j unavailable ({e}). Vector-only mode.")
        _neo4j_driver = None

    print("Loading Mistral-7B (4-bit)...")
    _llm = _build_llm()
    _graph = _build_langgraph()
    print("Agent ready!")

def ask(question: str, chat_history: list = None) -> dict:
    if _graph is None:
        raise RuntimeError("Call initialize() first.")
    state = _graph.invoke({
        "question": question, "retrieved_docs": [],
        "graph_context": "", "answer": "",
        "sources": [], "chat_history": chat_history or [],
    })
    return {"answer": state["answer"], "sources": state["sources"],
            "graph_context": state.get("graph_context", ""),
            "retrieved_docs": state["retrieved_docs"]}

def _build_llm():
    import torch
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                               pipeline, BitsAndBytesConfig)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_use_double_quant=True)
    tok   = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tok,
                    max_new_tokens=512, temperature=0.1,
                    do_sample=True, repetition_penalty=1.1)
    return HuggingFacePipeline(pipeline=pipe)

def _device():
    try:
        import torch; return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
