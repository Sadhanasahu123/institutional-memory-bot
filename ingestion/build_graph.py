"""
Neo4j Knowledge Graph Builder for Institutional Memory Bot.

What this builds:
  (:Person)         — employees (Sarah Chen, Marcus Rivera, etc.)
  (:Document)       — any source document (Confluence page, PR, ticket, etc.)
  (:Topic)          — subject areas (GDPR, NovaPay v2.0, Onboarding, etc.)
  (:Decision)       — explicit decisions made
  (:Incident)       — post-mortems / outages

Relationships:
  (Person)-[:DECIDED {date, context}]->(Decision)
  (Person)-[:AUTHORED]->(Document)
  (Person)-[:MENTIONED_IN]->(Document)
  (Person)-[:OPPOSED {reason}]->(Decision)
  (Document)-[:REFERENCES]->(Document)
  (Document)-[:ABOUT]->(Topic)
  (Decision)-[:CAUSED]->(Decision)
  (Incident)-[:LED_TO]->(Decision)

Run AFTER ingest.py. Requires Neo4j AuraDB (free tier) or local Neo4j.
"""

import json
from pathlib import Path
from bs4 import BeautifulSoup
from neo4j import GraphDatabase

# ── Config ────────────────────────────────────────────────────────────────────
# For local Neo4j (Docker):  neo4j://localhost:7687
# For Neo4j AuraDB (free):   neo4j+s://<your-instance>.databases.neo4j.io
NEO4J_URI      = "neo4j://localhost:7687"   # ← change if using AuraDB
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"                 # ← change to your password

DATA_DIR = Path("synthetic_data")

# ── Driver ────────────────────────────────────────────────────────────────────

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ── Schema / constraints ──────────────────────────────────────────────────────

SCHEMA_QUERIES = [
    "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.decision_id IS UNIQUE",
    "CREATE CONSTRAINT incident_id IF NOT EXISTS FOR (i:Incident) REQUIRE i.incident_id IS UNIQUE",
]

def create_schema(driver):
    with driver.session() as s:
        for q in SCHEMA_QUERIES:
            s.run(q)
    print("✅ Schema/constraints created.")

# ── Helper: merge nodes ────────────────────────────────────────────────────────

def merge_person(tx, name: str, role: str = "", team: str = ""):
    tx.run(
        "MERGE (p:Person {name: $name}) "
        "ON CREATE SET p.role=$role, p.team=$team "
        "ON MATCH SET p.role=CASE WHEN $role <> '' THEN $role ELSE p.role END",
        name=name, role=role, team=team
    )

def merge_document(tx, doc_id: str, title: str, source: str, date: str = ""):
    tx.run(
        "MERGE (d:Document {doc_id: $doc_id}) "
        "ON CREATE SET d.title=$title, d.source=$source, d.date=$date",
        doc_id=doc_id, title=title, source=source, date=date
    )

def merge_topic(tx, name: str):
    tx.run("MERGE (:Topic {name: $name})", name=name)

def merge_decision(tx, decision_id: str, summary: str, date: str = ""):
    tx.run(
        "MERGE (d:Decision {decision_id: $decision_id}) "
        "ON CREATE SET d.summary=$summary, d.date=$date",
        decision_id=decision_id, summary=summary, date=date
    )

def merge_incident(tx, incident_id: str, summary: str, date: str = ""):
    tx.run(
        "MERGE (i:Incident {incident_id: $incident_id}) "
        "ON CREATE SET i.summary=$summary, i.date=$date",
        incident_id=incident_id, summary=summary, date=date
    )

# ── Graph builders per source ──────────────────────────────────────────────────

def build_graph_from_confluence(driver, directory: Path):
    """Extract people, topics, and decisions from Confluence HTML pages."""
    with driver.session() as s:
        # Page 1: Product Launch Spec
        doc_id = "CONF-PROD-1042"
        s.execute_write(merge_document, doc_id,
            "NovaPay v2.0 Product Launch Specification", "confluence", "2024-01-15")
        s.execute_write(merge_person, "Sarah Chen", "Product Lead", "Product")
        s.execute_write(merge_person, "Marcus Rivera", "Backend Engineer", "Engineering")
        s.execute_write(merge_person, "Priya Nair", "DevOps Lead", "Engineering")
        s.execute_write(merge_person, "Anika Sharma", "Frontend Engineer", "Engineering")
        s.execute_write(merge_person, "James Okafor", "Legal Counsel", "Legal")
        s.execute_write(merge_topic, "NovaPay v2.0")
        s.execute_write(merge_topic, "Payments")
        s.execute_write(merge_topic, "Feature Flags")
        s.execute_write(merge_decision, "DEC-001",
            "Go multi-provider for NovaPay v2.0 instead of Stripe-only", "2024-01-28")

        s.run("MATCH (p:Person {name:'Sarah Chen'}), (d:Decision {decision_id:'DEC-001'}) "
              "MERGE (p)-[:DECIDED {date:'2024-01-28', context:'Cost and vendor lock-in concerns after Stripe price increase'}]->(d)")
        s.run("MATCH (p:Person {name:'Marcus Rivera'}), (d:Decision {decision_id:'DEC-001'}) "
              "MERGE (p)-[:OPPOSED {reason:'Integration complexity, tripled test suite, normalized error codes needed'}]->(d)")
        s.run("MATCH (doc:Document {doc_id:'CONF-PROD-1042'}), (t:Topic {name:'NovaPay v2.0'}) "
              "MERGE (doc)-[:ABOUT]->(t)")
        s.run("MATCH (p:Person {name:'Sarah Chen'}), (doc:Document {doc_id:'CONF-PROD-1042'}) "
              "MERGE (p)-[:AUTHORED]->(doc)")
        for person in ["Sarah Chen", "Marcus Rivera", "Priya Nair", "Anika Sharma", "James Okafor"]:
            s.run("MATCH (p:Person {name:$name}), (doc:Document {doc_id:'CONF-PROD-1042'}) "
                  "MERGE (p)-[:MENTIONED_IN]->(doc)", name=person)

        # Page 2: GDPR Policy
        doc_id2 = "CONF-LEGAL-204"
        s.execute_write(merge_document, doc_id2,
            "GDPR Data Handling Policy", "confluence", "2023-09-01")
        s.execute_write(merge_topic, "GDPR")
        s.execute_write(merge_topic, "Data Retention")
        s.execute_write(merge_decision, "DEC-002",
            "90-day activity log retention limit (individual-linked data)", "2023-09-01")
        s.execute_write(merge_decision, "DEC-003",
            "Anonymized aggregate data may be kept indefinitely for ML training", "2023-10-15")

        s.run("MATCH (p:Person {name:'James Okafor'}), (doc:Document {doc_id:'CONF-LEGAL-204'}) "
              "MERGE (p)-[:AUTHORED]->(doc)")
        s.run("MATCH (doc:Document {doc_id:'CONF-LEGAL-204'}), (t:Topic {name:'GDPR'}) "
              "MERGE (doc)-[:ABOUT]->(t)")
        s.run("MATCH (p:Person {name:'James Okafor'}), (d:Decision {decision_id:'DEC-002'}) "
              "MERGE (p)-[:DECIDED {date:'2023-09-01', context:'GDPR Article 5 storage limitation principle'}]->(d)")
        s.run("MATCH (p:Person {name:'Priya Nair'}), (d:Decision {decision_id:'DEC-002'}) "
              "MERGE (p)-[:OPPOSED {reason:'ML model training needs 12 months of behavioral data; 90 days hurts churn prediction accuracy'}]->(d)")
        s.run("MATCH (p:Person {name:'James Okafor'}), (d:Decision {decision_id:'DEC-003'}) "
              "MERGE (p)-[:DECIDED {date:'2023-10-15', context:'Compromise: strip user_id and device fingerprints, use rotating cohort_id'}]->(d)")

        # Page 3: Engineering Onboarding
        doc_id3 = "CONF-ENG-001"
        s.execute_write(merge_document, doc_id3,
            "Engineering Onboarding Guide", "confluence", "2023-06-01")
        s.execute_write(merge_topic, "Onboarding")
        s.execute_write(merge_topic, "Access Control")
        s.run("MATCH (p:Person {name:'Priya Nair'}), (doc:Document {doc_id:'CONF-ENG-001'}) "
              "MERGE (p)-[:AUTHORED]->(doc)")
        s.run("MATCH (doc:Document {doc_id:'CONF-ENG-001'}), (t:Topic {name:'Onboarding'}) "
              "MERGE (doc)-[:ABOUT]->(t)")

    print("✅ Confluence graph built.")


def build_graph_from_slack(driver, directory: Path):
    """Extract decisions and cross-references from Slack channel threads."""
    with driver.session() as s:
        # Channel: product-novapay
        doc_id = "SLACK-product-novapay-thread-20240115"
        s.execute_write(merge_document, doc_id,
            "Slack thread: multi-provider debate (#product-novapay)", "slack", "2024-01-15")
        s.execute_write(merge_topic, "NovaPay v2.0")
        s.run("MATCH (doc:Document {doc_id:$did}), (t:Topic {name:'NovaPay v2.0'}) "
              "MERGE (doc)-[:ABOUT]->(t)", did=doc_id)
        for person in ["Sarah Chen", "Marcus Rivera", "Priya Nair"]:
            s.run("MATCH (p:Person {name:$name}), (doc:Document {doc_id:$did}) "
                  "MERGE (p)-[:PARTICIPATED_IN]->(doc)", name=person, did=doc_id)

        # The Slack thread is the origin of DEC-001 debate
        s.run("MATCH (slack:Document {doc_id:$did}), (conf:Document {doc_id:'CONF-PROD-1042'}) "
              "MERGE (conf)-[:REFERENCES {context:'Decision debated in this thread'}]->(slack)",
              did=doc_id)

        # Channel: legal-gdpr GDPR Redis thread
        doc_id2 = "SLACK-legal-gdpr-thread-20240120"
        s.execute_write(merge_document, doc_id2,
            "Slack thread: GDPR Redis cache compliance (#legal-gdpr)", "slack", "2024-01-20")
        s.execute_write(merge_decision, "DEC-004",
            "Redis cache for payment tokens must be EU region, TTL < 24h", "2024-01-20")
        s.run("MATCH (p:Person {name:'James Okafor'}), (d:Decision {decision_id:'DEC-004'}) "
              "MERGE (p)-[:DECIDED {date:'2024-01-20', context:'Redis cache counts as GDPR storage; must stay EU, TTL < 24h'}]->(d)")
        s.run("MATCH (p:Person {name:'Marcus Rivera'}), (d:Decision {decision_id:'DEC-004'}) "
              "MERGE (p)-[:ASKED_ABOUT]->(d)")

        # Channel: engineering-general - 2-approval policy announcement
        doc_id3 = "SLACK-eng-general-thread-20240110"
        s.execute_write(merge_document, doc_id3,
            "Slack: 2-approval deploy policy announced (#engineering-general)", "slack", "2024-01-10")
        for person in ["Priya Nair", "Marcus Rivera"]:
            s.run("MATCH (p:Person {name:$name}), (doc:Document {doc_id:$did}) "
                  "MERGE (p)-[:PARTICIPATED_IN]->(doc)", name=person, did=doc_id3)

    print("✅ Slack graph built.")


def build_graph_from_github(driver, directory: Path):
    """Extract PR authorship, reviews, and cross-references."""
    with driver.session() as s:
        doc_id = "GH-PR-234"
        s.execute_write(merge_document, doc_id,
            "GitHub PR #234: NovaPay multi-provider abstraction layer", "github", "2024-02-12")
        s.execute_write(merge_topic, "NovaPay v2.0")
        s.run("MATCH (p:Person {name:'Marcus Rivera'}), (doc:Document {doc_id:$did}) "
              "MERGE (p)-[:AUTHORED]->(doc)", did=doc_id)
        for reviewer in ["Priya Nair", "James Okafor", "Anika Sharma"]:
            s.run("MATCH (p:Person {name:$name}), (doc:Document {doc_id:$did}) "
                  "MERGE (p)-[:REVIEWED]->(doc)", name=reviewer, did=doc_id)
        # PR implements the decision
        s.run("MATCH (doc:Document {doc_id:'GH-PR-234'}), (d:Decision {decision_id:'DEC-001'}) "
              "MERGE (doc)-[:IMPLEMENTS]->(d)")
        # PR references the Jira ticket
        s.run("MATCH (pr:Document {doc_id:'GH-PR-234'}), (jira:Document {doc_id:'JIRA-NOVA-112'}) "
              "MERGE (pr)-[:REFERENCES]->(jira)")

        # Bug issue #289
        doc_id2 = "GH-ISSUE-289"
        s.execute_write(merge_document, doc_id2,
            "GitHub Issue #289: Race condition in checkout on fast transactions", "github", "2024-02-05")
        s.run("MATCH (p:Person {name:'Anika Sharma'}), (doc:Document {doc_id:$did}) "
              "MERGE (p)-[:AUTHORED]->(doc)", did=doc_id2)
        s.run("MATCH (p:Person {name:'Marcus Rivera'}), (doc:Document {doc_id:$did}) "
              "MERGE (p)-[:RESOLVED]->(doc)", did=doc_id2)

    print("✅ GitHub graph built.")


def build_graph_from_jira(driver, directory: Path):
    """Extract Jira ticket ownership, blockers, and cross-references."""
    with driver.session() as s:
        # Epic NOVA-100
        doc_id = "JIRA-NOVA-100"
        s.execute_write(merge_document, doc_id,
            "Jira Epic NOVA-100: NovaPay v2.0 multi-provider gateway", "jira", "2024-01-15")
        s.run("MATCH (p:Person {name:'Sarah Chen'}), (doc:Document {doc_id:$did}) "
              "MERGE (p)-[:OWNS]->(doc)", did=doc_id)

        # Story NOVA-112
        doc_id2 = "JIRA-NOVA-112"
        s.execute_write(merge_document, doc_id2,
            "Jira Story NOVA-112: Payment provider abstraction layer", "jira", "2024-01-28")
        s.run("MATCH (p:Person {name:'Marcus Rivera'}), (doc:Document {doc_id:$did}) "
              "MERGE (p)-[:OWNS]->(doc)", did=doc_id2)
        s.run("MATCH (epic:Document {doc_id:'JIRA-NOVA-100'}), (story:Document {doc_id:'JIRA-NOVA-112'}) "
              "MERGE (story)-[:CHILD_OF]->(epic)")
        s.run("MATCH (p:Person {name:'James Okafor'}), (doc:Document {doc_id:$did}) "
              "MERGE (p)-[:COMMENTED_ON]->(doc)", did=doc_id2)

        # GDPR deletion ticket
        doc_id3 = "JIRA-GDPR-089"
        s.execute_write(merge_document, doc_id3,
            "Jira Task GDPR-089: GDPR deletion request user 48821", "jira", "2024-02-10")
        s.execute_write(merge_topic, "GDPR")
        s.run("MATCH (p:Person {name:'James Okafor'}), (doc:Document {doc_id:$did}) "
              "MERGE (p)-[:CREATED]->(doc)", did=doc_id3)
        s.run("MATCH (p:Person {name:'Marcus Rivera'}), (doc:Document {doc_id:$did}) "
              "MERGE (p)-[:OWNS]->(doc)", did=doc_id3)
        s.run("MATCH (doc:Document {doc_id:$did}), (t:Topic {name:'GDPR'}) "
              "MERGE (doc)-[:ABOUT]->(t)", did=doc_id3)

        # Post-mortem ENG-POST-203
        inc_id = "INC-001"
        s.execute_write(merge_incident, inc_id,
            "Payments API outage - DB migration bug, 4h downtime", "2023-11-14")
        doc_id4 = "JIRA-ENG-POST-203"
        s.execute_write(merge_document, doc_id4,
            "Post-mortem: Payments API outage November 2023", "jira", "2023-11-15")
        s.execute_write(merge_decision, "DEC-005",
            "Mandatory 2-approval policy for all production deployments", "2024-01-10")

        s.run("MATCH (i:Incident {incident_id:'INC-001'}), (d:Decision {decision_id:'DEC-005'}) "
              "MERGE (i)-[:LED_TO {context:'Single-approval deploy caused DB migration bug and 4h outage'}]->(d)")
        s.run("MATCH (p:Person {name:'Priya Nair'}), (d:Decision {decision_id:'DEC-005'}) "
              "MERGE (p)-[:DECIDED {date:'2024-01-10', context:'Policy enforcement owner after November outage'}]->(d)")
        s.run("MATCH (doc:Document {doc_id:'JIRA-ENG-POST-203'}), (i:Incident {incident_id:'INC-001'}) "
              "MERGE (doc)-[:DOCUMENTS]->(i)")
        s.run("MATCH (doc:Document {doc_id:'JIRA-ENG-POST-203'}), (slack:Document {doc_id:'SLACK-eng-general-thread-20240110'}) "
              "MERGE (doc)-[:REFERENCES]->(slack)")

    print("✅ Jira graph built.")


def build_graph_from_email(driver, directory: Path):
    """Extract email thread participants and compliance approvals."""
    with driver.session() as s:
        doc_id = "EMAIL-thread-gdpr-nova"
        s.execute_write(merge_document, doc_id,
            "Email: GDPR compliance sign-off for NovaPay v2.0", "email", "2024-01-20")
        for person in ["James Okafor", "Sarah Chen", "Marcus Rivera"]:
            s.run("MATCH (p:Person {name:$name}), (doc:Document {doc_id:$did}) "
                  "MERGE (p)-[:PARTICIPATED_IN]->(doc)", name=person, did=doc_id)
        # Connects to the PR
        s.run("MATCH (email:Document {doc_id:'EMAIL-thread-gdpr-nova'}), (pr:Document {doc_id:'GH-PR-234'}) "
              "MERGE (email)-[:PRECEDED {context:'Legal sign-off email led to compliance items in PR'}]->(pr)")

        doc_id2 = "EMAIL-thread-fastapi-migration"
        s.execute_write(merge_document, doc_id2,
            "Email: Django to FastAPI migration outcome", "email", "2023-08-01")
        s.execute_write(merge_decision, "DEC-006",
            "Migrate backend from Django to FastAPI for async webhook handling", "2023-07-01")
        s.run("MATCH (p:Person {name:'Marcus Rivera'}), (d:Decision {decision_id:'DEC-006'}) "
              "MERGE (p)-[:DECIDED {date:'2023-07-01', context:'3.2x performance improvement for payment endpoints, N+1 query issues in Django ORM'}]->(d)")

    print("✅ Email graph built.")


def build_graph_from_meetings(driver, directory: Path):
    """Extract meeting participants and decisions."""
    with driver.session() as s:
        doc_id = "MTG-2024-02-01"
        s.execute_write(merge_document, doc_id,
            "Meeting: NovaPay v2.0 All-Hands Feature Flag Strategy", "meeting", "2024-02-01")
        s.execute_write(merge_decision, "DEC-007",
            "James Okafor is required approver for all payment-namespace LaunchDarkly flags", "2024-02-01")
        s.execute_write(merge_decision, "DEC-008",
            "NOVAPAY_V2_CHECKOUT UI flag flips all-at-once on Tuesday 6am IST", "2024-02-01")

        for person in ["Sarah Chen", "Marcus Rivera", "Priya Nair", "Anika Sharma", "James Okafor"]:
            s.run("MATCH (p:Person {name:$name}), (doc:Document {doc_id:$did}) "
                  "MERGE (p)-[:ATTENDED]->(doc)", name=person, did=doc_id)
        s.run("MATCH (p:Person {name:'James Okafor'}), (d:Decision {decision_id:'DEC-007'}) "
              "MERGE (p)-[:DECIDED {date:'2024-02-01', context:'Feature flags are not a bypass for compliance review'}]->(d)")

        doc_id2 = "MTG-2023-10-15"
        s.execute_write(merge_document, doc_id2,
            "Meeting: Data Retention Policy Review - Engineering + Legal", "meeting", "2023-10-15")
        for person in ["James Okafor", "Priya Nair", "Sarah Chen"]:
            s.run("MATCH (p:Person {name:$name}), (doc:Document {doc_id:$did}) "
                  "MERGE (p)-[:ATTENDED]->(doc)", name=person, did=doc_id2)
        # This meeting led to DEC-003
        s.run("MATCH (doc:Document {doc_id:'MTG-2023-10-15'}), (d:Decision {decision_id:'DEC-003'}) "
              "MERGE (doc)-[:PRODUCED]->(d)")

    print("✅ Meetings graph built.")


# ── Sample queries (useful for testing) ───────────────────────────────────────

SAMPLE_QUERIES = {
    "Who decided multi-provider payments and who opposed it?": """
        MATCH (p:Person)-[r:DECIDED|OPPOSED]->(d:Decision {decision_id:'DEC-001'})
        RETURN p.name, type(r), r.context, r.reason
    """,
    "What decisions did James Okafor make?": """
        MATCH (p:Person {name:'James Okafor'})-[r:DECIDED]->(d:Decision)
        RETURN d.summary, r.date, r.context
        ORDER BY r.date
    """,
    "What documents reference the NovaPay PR?": """
        MATCH (doc:Document)-[:REFERENCES]->(pr:Document {doc_id:'GH-PR-234'})
        RETURN doc.title, doc.source, doc.date
    """,
    "What incident led to the 2-approval policy?": """
        MATCH (i:Incident)-[:LED_TO]->(d:Decision {decision_id:'DEC-005'})
        RETURN i.summary, i.date, d.summary
    """,
    "Who attended meetings and what decisions came out of them?": """
        MATCH (p:Person)-[:ATTENDED]->(m:Document)-[:PRODUCED]->(d:Decision)
        RETURN m.title, collect(p.name) AS attendees, d.summary
    """,
    "What documents is Priya Nair mentioned in or authored?": """
        MATCH (p:Person {name:'Priya Nair'})-[r]->(doc:Document)
        WHERE type(r) IN ['AUTHORED','MENTIONED_IN','PARTICIPATED_IN','ATTENDED','OWNS']
        RETURN doc.title, doc.source, type(r)
        ORDER BY doc.date
    """,
}

def run_sample_queries(driver):
    print("\n" + "="*60)
    print("Running sample graph queries:")
    print("="*60)
    with driver.session() as s:
        for question, query in SAMPLE_QUERIES.items():
            print(f"\nQ: {question}")
            results = s.run(query)
            for record in results:
                print(f"  → {dict(record)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_full_graph(data_dir: Path = DATA_DIR):
    driver = get_driver()
    print("Connected to Neo4j.")
    print("Creating schema...")
    create_schema(driver)

    build_graph_from_confluence(driver, data_dir / "confluence")
    build_graph_from_slack(driver, data_dir / "slack")
    build_graph_from_github(driver, data_dir / "github")
    build_graph_from_jira(driver, data_dir / "jira")
    build_graph_from_email(driver, data_dir / "email")
    build_graph_from_meetings(driver, data_dir / "meetings")

    run_sample_queries(driver)
    driver.close()
    print("\n✅ Knowledge graph fully built!")


if __name__ == "__main__":
    build_full_graph()
