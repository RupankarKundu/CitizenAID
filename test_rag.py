"""
CitizenAid — Test Suite (Gemini edition)
Run: python3 test_rag.py
Tests: data files, imports, embeddings, RAG ingestion, category detection,
       full query, RTI agent, live API endpoints
"""

import sys, os, time, json

G="\033[92m"; R="\033[91m"; Y="\033[93m"; C="\033[96m"; B="\033[1m"; X="\033[0m"
ok  = lambda s: print(f"  {G}✓{X}  {s}")
err = lambda s: print(f"  {R}✗{X}  {s}")
hdr = lambda s: print(f"\n{B}{C}── {s} ──{X}")


# ── 1. Data files ──────────────────────────────────────────────────────────────
hdr("1 · Data Files")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EXPECTED = [
    "rti_act_2005.txt", "consumer_protection.txt", "constitutional_rights.txt",
    "ipc_sections.txt", "rti_templates.txt", "state_and_misc_laws.txt",
]
for f in EXPECTED:
    p = os.path.join(DATA_DIR, f)
    if os.path.exists(p):
        ok(f"{f}  ({os.path.getsize(p):,} bytes)")
    else:
        err(f"{f}  — NOT FOUND"); sys.exit(1)


# ── 2. Imports ─────────────────────────────────────────────────────────────────
hdr("2 · Import Dependencies")
deps = [
    ("langchain",            "langchain"),
    ("chromadb",             "chromadb"),
    ("sentence_transformers","sentence-transformers"),
    ("google.generativeai",  "google-generativeai"),
    ("langchain_google_genai","langchain-google-genai"),
    ("fastapi",              "fastapi"),
]
for mod, name in deps:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        ok(f"{name}  {ver}")
    except ImportError as e:
        err(f"{name} — NOT INSTALLED: {e}"); sys.exit(1)


# ── 3. Embedding model ─────────────────────────────────────────────────────────
hdr("3 · Multilingual Embedding Model")
try:
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ok(f"Loaded in {time.time()-t0:.1f}s")
    for lang, text in [("English","What is RTI?"), ("Hindi","RTI क्या है?"), ("Bengali","আরটিআই কী?")]:
        v = model.encode([text])
        ok(f"{lang} embedding: dim={v.shape[1]}")
except Exception as e:
    err(f"Embedding model failed: {e}"); sys.exit(1)


# ── 4. RAG pipeline ingestion ──────────────────────────────────────────────────
hdr("4 · RAG Pipeline — Ingestion")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
try:
    from rag_pipeline import CitizenAidRAG, _detect_category
    rag = CitizenAidRAG()
    ok("CitizenAidRAG instantiated")
    t0 = time.time()
    rag.ingest_documents()
    ok(f"Documents ingested / ChromaDB loaded in {time.time()-t0:.1f}s")
    docs = rag.get_retriever().invoke("how to file RTI for PF")
    ok(f"Retriever returned {len(docs)} chunks")
    if docs:
        ok(f"Top chunk: \"{docs[0].page_content[:80].strip()}…\"")
except Exception as e:
    err(f"RAG pipeline error: {e}")
    import traceback; traceback.print_exc(); sys.exit(1)


# ── 5. Category detection ──────────────────────────────────────────────────────
hdr("5 · Category Auto-detection")
CASES = [
    ("how to file RTI for scheme status",    "RTI"),
    ("builder delayed possession by 2 years","Consumer"),
    ("police refused FIR what to do",        "Criminal"),
    ("PF not deposited by employer",         "Labour"),
    ("my fundamental rights violated",       "Constitutional"),
]
for q, expected in CASES:
    got = _detect_category(q)
    sym = "✓" if got == expected else "✗"
    print(f"  {G if got==expected else R}{sym}{X}  \"{q[:48]}\" → {got}")


# ── 6. Full RAG query (needs GOOGLE_API_KEY) ───────────────────────────────────
hdr("6 · Full RAG Query  (needs GOOGLE_API_KEY)")
key = os.getenv("GOOGLE_API_KEY","")
if not key or "paste" in key:
    print(f"  {Y}Skipped — GOOGLE_API_KEY not set.{X}")
else:
    try:
        t0 = time.time()
        res = rag.query("How do I file an RTI about my PF not being deposited?", category="RTI")
        ok(f"Query in {time.time()-t0:.1f}s")
        ok(f"Category={res['category']}  confidence={res['confidence']}  needs_lawyer={res['needs_lawyer']}")
        ok(f"Sources: {[s['file'] for s in res['sources']]}")
        ok(f"Answer[:120]: \"{res['answer'][:120].strip()}…\"")
        ok(f"Quick actions: {res['quick_actions']}")
    except Exception as e:
        err(f"Query failed: {e}"); import traceback; traceback.print_exc()


# ── 7. RTI agent (needs GOOGLE_API_KEY) ───────────────────────────────────────
hdr("7 · RTI Draft Agent  (needs GOOGLE_API_KEY)")
if not key or "paste" in key:
    print(f"  {Y}Skipped — GOOGLE_API_KEY not set.{X}")
else:
    try:
        from rti_agent import RTIAgent, _detect_authority
        auth, portal = _detect_authority("railway ticket refund", None)
        ok(f"Authority detected: {auth}")
        ok(f"Portal: {portal}")
        agent = RTIAgent(rag)
        t0 = time.time()
        res = agent.generate(
            situation="My MGNREGA wages for March 2024 have not been paid after 4 months.",
            state="West Bengal", applicant_name="Test Applicant",
        )
        ok(f"Draft generated in {time.time()-t0:.1f}s  ({len(res['draft'])} chars)")
        ok(f"Authority: {res['authority']}")
        ok(f"Steps: {len(res['steps'])}  Tips: {len(res['tips'])}")
        print(f"\n  {C}── Draft preview ──{X}")
        print("  " + res["draft"][:300].replace("\n", "\n  "))
    except Exception as e:
        err(f"RTI agent failed: {e}"); import traceback; traceback.print_exc()


# ── 8. Live API (needs running server) ────────────────────────────────────────
hdr("8 · Live API Endpoints  (needs server on :8000)")
try:
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=3) as r:
            ok(f"GET /health → {json.loads(r.read())}")
    except Exception:
        print(f"  {Y}Server not running — start with: bash setup.sh{X}")

    try:
        req = urllib.request.Request(
            "http://localhost:8000/api/query",
            data=json.dumps({"question":"What is RTI?","language":"en"}).encode(),
            headers={"Content-Type":"application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
            ok(f"POST /api/query → category={d['category']}  confidence={d['confidence']}")
    except Exception as e:
        print(f"  {Y}POST /api/query failed: {e}{X}")
except Exception as e:
    print(f"  {Y}Live API test skipped: {e}{X}")


# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{B}{'─'*52}{X}")
print(f"{G}{B}Core tests complete.{X}  Run 'bash setup.sh' to launch.\n")
