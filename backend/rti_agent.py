"""
CitizenAid — RTI Draft Agent (LangGraph edition)
Uses a StateGraph with 6 nodes for intelligent, self-correcting RTI generation:

  understand_situation
          ↓
  detect_authority
          ↓
  retrieve_context
          ↓
  generate_draft
          ↓
  validate_draft ──(invalid, retry once)──→ generate_draft
          ↓ (valid)
  finalize_response
"""

import os
from pathlib import Path
from typing import Optional, TypedDict, Literal
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# ── LLM ───────────────────────────────────────────────────────────────────────
def _llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        temperature=0.3,
        max_tokens=1200,
    )

# ── Authority map ──────────────────────────────────────────────────────────────
AUTHORITY_MAP = {
    "passport":    ("Regional Passport Office / Ministry of External Affairs", "https://rtionline.gov.in"),
    "income tax":  ("Income Tax Department / Ministry of Finance",              "https://rtionline.gov.in"),
    "railway":     ("Divisional Railway Manager / Ministry of Railways",        "https://rtionline.gov.in"),
    "post office": ("Department of Posts / Ministry of Communications",         "https://rtionline.gov.in"),
    "epfo":        ("Regional PF Commissioner / Ministry of Labour",            "https://rtionline.gov.in"),
    "pf":          ("Regional PF Commissioner / EPFO / Ministry of Labour",     "https://rtionline.gov.in"),
    "aadhaar":     ("UIDAI Regional Office / Ministry of Electronics & IT",     "https://rtionline.gov.in"),
    "bank":        ("Branch Manager / Head Office (for PSU Banks)",             "https://rtionline.gov.in"),
    "ration":      ("District Supply Officer / State Food & Civil Supplies Dept","State RTI portal"),
    "electricity": ("Executive Engineer / State Electricity Board",             "State RTI portal"),
    "water":       ("Executive Engineer / State Water Supply Board",            "State RTI portal"),
    "road":        ("Executive Engineer, PWD / State Public Works Dept",        "State RTI portal"),
    "school":      ("District Education Officer / State Education Dept",        "State RTI portal"),
    "hospital":    ("Medical Superintendent / State Health Dept",               "State RTI portal"),
    "police":      ("Superintendent of Police / State Home Dept",               "State RTI portal"),
    "land":        ("Tehsildar / Sub-Divisional Magistrate / Revenue Dept",     "State RTI portal"),
    "panchayat":   ("Block Development Officer / Panchayati Raj Dept",         "State RTI portal"),
    "municipality":("Municipal Commissioner / Urban Local Body",                "State RTI portal"),
    "pension":     ("Treasury Officer / Finance Dept / Pension Dept",           "State RTI portal"),
    "recruitment": ("Secretary, State Public Service Commission / UPSC",        "https://rtionline.gov.in"),
    "mgnrega":     ("Block Development Officer / Ministry of Rural Development","State RTI portal"),
    "pm awas":     ("Block Development Officer / Ministry of Housing",          "State RTI portal"),
    "salary":      ("Head of Department / Pay & Accounts Office",               "State RTI portal"),
    "municipal":   ("Municipal Commissioner / Urban Local Body",                "State RTI portal"),
}

TIPS = [
    "📌 Be specific — ask for exact documents, dates, or amounts. Vague questions get vague answers.",
    "📌 File multiple focused RTIs rather than one long one — easier to process and harder to deny.",
    "📌 Keep a copy of your application and note the postal receipt number.",
    "📌 You do NOT need to give any reason for seeking information.",
    "📌 BPL card holders are exempt from the Rs. 10 fee — attach a copy of your BPL card.",
]

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "bn": "Bengali (বাংলা)",
}


# ── Graph State ────────────────────────────────────────────────────────────────
class RTIState(TypedDict):
    # Inputs
    situation:      str
    department:     Optional[str]
    state:          Optional[str]
    applicant_name: str
    language:       str

    # Intermediate
    language_name:  str
    core_issue:     str          # cleaned summary of situation
    authority:      str
    portal_link:    str
    rti_context:    str          # chunks from ChromaDB
    retry_count:    int

    # Outputs
    draft:          str
    is_valid:       bool
    validation_note: str
    steps:          list
    tips:           list


# ── Helper: build filing steps ─────────────────────────────────────────────────
def _build_steps(authority: str, portal: str, state_name: Optional[str]) -> list:
    state_label = state_name or "your state"
    is_central  = "rtionline.gov.in" in portal
    commission  = (
        "Central Information Commission (cic.gov.in)"
        if is_central
        else f"{state_label} Information Commission"
    )
    return [
        f"1️⃣  **Identify the PIO** — Find the Public Information Officer of: *{authority}*.",
        f"2️⃣  **Prepare your application** — Use the draft below. Be specific. No reason needed.",
        f"3️⃣  **Pay the fee** — Rs. 10/- by Indian Postal Order, Demand Draft, or cash. BPL holders exempt.",
        f"4️⃣  **Submit** — Registered Post with AD or in person. Online: {portal}",
        f"5️⃣  **Wait 30 days** — PIO must respond within 30 days (48 hours for life/liberty).",
        f"6️⃣  **First Appeal** — No reply or unsatisfactory? Appeal to FAA within 30 days.",
        f"7️⃣  **Second Appeal** — Still unsatisfied? Approach {commission} within 90 days.",
        f"8️⃣  **Penalty** — Commission can impose Rs. 250/day (max Rs. 25,000) under Section 20.",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Understand the situation
# Extracts the core legal issue from the user's description
# ══════════════════════════════════════════════════════════════════════════════
def node_understand_situation(state: RTIState) -> RTIState:
    print("🔍 [Node 1] Understanding situation…")
    llm = _llm()
    prompt = f"""You are a legal assistant. Read this citizen's situation and extract:
1. The core legal issue in one sentence
2. Key facts: what happened, who is responsible, how long pending

Situation: {state['situation']}

Respond in this exact format:
CORE ISSUE: [one sentence summary]
KEY FACTS: [2-3 bullet points]"""

    response = llm.invoke([HumanMessage(content=prompt)])
    core_issue = response.content.strip()
    print(f"   ✓ Core issue extracted")
    return {**state, "core_issue": core_issue}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Detect correct authority (LLM-powered with hardcoded fast-path)
# Step 1: Check hardcoded map for instant match (fast path)
# Step 2: Always verify/enrich with LLM regardless of map match
# Step 3: LLM also decides the correct portal link
# ══════════════════════════════════════════════════════════════════════════════
def node_detect_authority(state: RTIState) -> RTIState:
    print("🏛️  [Node 2] Detecting authority…")
    text = (state['situation'] + " " + (state['department'] or "")).lower()

    # Step 1: Fast-path hint from hardcoded map
    hint_authority = None
    hint_portal    = None
    for keyword, (auth, port) in AUTHORITY_MAP.items():
        if keyword in text:
            hint_authority = auth
            hint_portal    = port
            break

    # Step 2: Always ask LLM for full authority + portal decision
    llm = _llm()
    hint_text = f"\nHint from keyword match: {hint_authority}" if hint_authority else ""
    
    prompt = f"""You are an Indian legal expert. Based on the citizen's situation below, identify:
1. The EXACT government authority/department to address the RTI application to
2. Whether to use Central RTI portal (rtionline.gov.in) or State RTI portal

Citizen's situation: {state['situation']}
Department mentioned by citizen: {state['department'] or 'Not specified'}
State: {state.get('state') or 'Not specified'}{hint_text}

Rules:
- If it is a Central Government department (Railways, Post Office, EPFO, Income Tax, Passport, UIDAI, Central Ministries): use rtionline.gov.in
- If it is a State Government department (Police, Municipal, School, Hospital, Ration, Roads, State schemes): use State RTI portal
- Be specific — name the exact officer designation and department
- Examples: "Block Development Officer, Panchayati Raj Department", "Regional PF Commissioner, EPFO", "District Supply Officer, Food & Civil Supplies Department"

Respond in EXACTLY this format (two lines only):
AUTHORITY: [exact authority name]
PORTAL: [either "https://rtionline.gov.in" or "State RTI portal"]"""

    response  = llm.invoke([HumanMessage(content=prompt)])
    lines     = response.content.strip().split("\n")
    
    authority = hint_authority or "Concerned State/Central Public Authority"
    portal    = hint_portal    or "State RTI portal / https://rtionline.gov.in"

    for line in lines:
        if line.startswith("AUTHORITY:"):
            authority = line.replace("AUTHORITY:", "").strip()
        elif line.startswith("PORTAL:"):
            portal = line.replace("PORTAL:", "").strip()

    print(f"   ✓ Authority: {authority}")
    print(f"   ✓ Portal:    {portal}")
    return {**state, "authority": authority, "portal_link": portal}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Retrieve RTI law context from ChromaDB
# Gets relevant chunks from the legal knowledge base
# ══════════════════════════════════════════════════════════════════════════════
def node_retrieve_context(state: RTIState, rag=None) -> RTIState:
    print("📚 [Node 3] Retrieving legal context from ChromaDB…")
    if rag is None:
        print("   ⚠️  RAG not available, using empty context")
        return {**state, "rti_context": ""}

    try:
        docs = rag.get_retriever().invoke(
            f"RTI application template filing procedure: {state['situation']}"
        )
        context = "\n\n".join(d.page_content for d in docs[:4])
        print(f"   ✓ Retrieved {len(docs)} chunks from ChromaDB")
    except Exception as e:
        print(f"   ⚠️  Retrieval failed: {e}")
        context = ""

    return {**state, "rti_context": context}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Generate RTI draft
# Calls Groq LLM to write the formal RTI application
# ══════════════════════════════════════════════════════════════════════════════
def node_generate_draft(state: RTIState) -> RTIState:
    retry = state.get("retry_count", 0)
    print(f"✍️  [Node 4] Generating RTI draft (attempt {retry + 1})…")

    language_name = state.get("language_name", "English")
    llm = _llm()

    # On retry, explicitly tell LLM what was wrong
    retry_note = ""
    if retry > 0:
        retry_note = f"\n\nPrevious attempt was rejected because: {state.get('validation_note', 'incomplete draft')}. Fix this issue."

    prompt = f"""You are a legal expert helping an Indian citizen draft a Right to Information (RTI) application.

⚠️ LANGUAGE: Write the entire RTI application in {language_name} only. 
Exception: Standard RTI headers (To, Subject, From) can remain in English as they are official legal terms.

Citizen's situation: {state['situation']}
Core issue identified: {state.get('core_issue', state['situation'])}
Target authority: {state['authority']}
Applicant name: {state['applicant_name']}
State: {state.get('state') or 'Not specified'}

Relevant RTI law context:
{state.get('rti_context', '')}
{retry_note}

Write a COMPLETE formal RTI application that includes ALL of these:
1. Salutation: "To, The Public Information Officer, [Department]"
2. Subject line
3. Introduction paragraph
4. At least 4 specific numbered queries relevant to the situation
5. Request for information within 30 days under Section 7(1) RTI Act
6. Closing: "Yours faithfully, [Name], Date: ___"

Output ONLY the RTI application text. Nothing else."""

    response = llm.invoke([HumanMessage(content=prompt)])
    draft = response.content.strip()
    print(f"   ✓ Draft generated ({len(draft)} chars)")
    return {**state, "draft": draft, "retry_count": retry + 1}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Validate the draft
# Checks if the draft has all required sections
# ══════════════════════════════════════════════════════════════════════════════
def node_validate_draft(state: RTIState) -> RTIState:
    print("✅ [Node 5] Validating draft quality…")
    draft = state.get("draft", "")
    issues = []

    # Check minimum length
    if len(draft) < 300:
        issues.append("Draft is too short — must be at least 300 characters")

    # Check for required sections
    draft_lower = draft.lower()
    if not any(k in draft_lower for k in ["public information officer", "pio", "सूचना अधिकारी", "তথ্য আধিকারিক"]):
        issues.append("Missing salutation to PIO")

    if not any(k in draft_lower for k in ["subject", "विषय", "বিষয়"]):
        issues.append("Missing subject line")

    has_queries = any(k in draft for k in ["1.", "1)", "क्या", "কি", "please provide", "kindly provide", "प्रदान करें", "জানাবেন"])
    if not has_queries:
        issues.append("Missing numbered queries")

    if not any(k in draft_lower for k in ["yours", "faithfully", "sincerely", "भवदीय", "आपका", "আপনার বিশ্বস্ত"]):
        issues.append("Missing closing/signature")

    is_valid = len(issues) == 0
    note = "; ".join(issues) if issues else "Draft looks complete and valid"

    print(f"   {'✓ Valid' if is_valid else '✗ Invalid — ' + note}")
    return {**state, "is_valid": is_valid, "validation_note": note}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6 — Finalize response
# Assembles the complete response with draft + steps + tips
# ══════════════════════════════════════════════════════════════════════════════
def node_finalize(state: RTIState) -> RTIState:
    print("🎯 [Node 6] Finalizing response…")
    steps = _build_steps(
        state["authority"],
        state["portal_link"],
        state.get("state"),
    )
    return {**state, "steps": steps, "tips": TIPS}


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE — After validation: retry or finalize?
# ══════════════════════════════════════════════════════════════════════════════
def decide_after_validation(state: RTIState) -> Literal["generate_draft", "finalize"]:
    if state.get("is_valid"):
        return "finalize"
    if state.get("retry_count", 0) >= 2:
        print("   ⚠️  Max retries reached — proceeding with best draft")
        return "finalize"
    print("   🔄 Retrying draft generation…")
    return "generate_draft"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_rti_graph(rag):
    """Build and compile the LangGraph RTI agent with RAG access."""

    # Wrap node_retrieve_context to inject the rag object
    def retrieve_with_rag(state: RTIState) -> RTIState:
        return node_retrieve_context(state, rag=rag)

    graph = StateGraph(RTIState)

    # Add all nodes
    graph.add_node("understand_situation", node_understand_situation)
    graph.add_node("detect_authority",     node_detect_authority)
    graph.add_node("retrieve_context",     retrieve_with_rag)
    graph.add_node("generate_draft",       node_generate_draft)
    graph.add_node("validate_draft",       node_validate_draft)
    graph.add_node("finalize",             node_finalize)

    # Add edges (flow between nodes)
    graph.set_entry_point("understand_situation")
    graph.add_edge("understand_situation", "detect_authority")
    graph.add_edge("detect_authority",     "retrieve_context")
    graph.add_edge("retrieve_context",     "generate_draft")
    graph.add_edge("generate_draft",       "validate_draft")

    # Conditional edge: retry or finalize
    graph.add_conditional_edges(
        "validate_draft",
        decide_after_validation,
        {
            "generate_draft": "generate_draft",
            "finalize":       "finalize",
        }
    )

    graph.add_edge("finalize", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# RTI AGENT CLASS — drop-in replacement for old RTIAgent
# ══════════════════════════════════════════════════════════════════════════════
class RTIAgent:
    def __init__(self, rag):
        self.rag   = rag
        self.graph = build_rti_graph(rag)
        print("✅ LangGraph RTI agent compiled (6 nodes)")

    def generate(
        self,
        situation:      str,
        department:     Optional[str] = None,
        state:          Optional[str] = None,
        applicant_name: str = "Applicant",
        language:       str = "en",
    ) -> dict:
        language_name = LANGUAGE_NAMES.get(language, "English")
        print(f"\n🚀 RTI LangGraph — starting for: '{situation[:60]}…'")

        initial_state: RTIState = {
            "situation":      situation,
            "department":     department,
            "state":          state,
            "applicant_name": applicant_name,
            "language":       language,
            "language_name":  language_name,
            "core_issue":     "",
            "authority":      "",
            "portal_link":    "",
            "rti_context":    "",
            "retry_count":    0,
            "draft":          "",
            "is_valid":       False,
            "validation_note":"",
            "steps":          [],
            "tips":           [],
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        print(f"✅ LangGraph RTI complete — draft: {len(final_state['draft'])} chars\n")

        return {
            "draft":       final_state["draft"],
            "steps":       final_state["steps"],
            "authority":   final_state["authority"],
            "portal_link": final_state["portal_link"],
            "tips":        final_state["tips"],
        }