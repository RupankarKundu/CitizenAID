"""
CitizenAid — RAG Pipeline
Groq LLM (free, fast, works in India)
"""

import os
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document

DATA_DIR   = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path.home() / ".citizenaid" / "chroma_db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "bn": "Bengali (বাংলা)",
}

SYSTEM_PROMPT = """You are CitizenAid, a knowledgeable legal assistant specialising in Indian law.

⚠️ LANGUAGE INSTRUCTION (MANDATORY): You MUST respond ENTIRELY in {language_name}. 
Every single word of your answer must be in {language_name} only.
Do NOT mix languages. Do NOT use English if the language is Hindi or Bengali.
Do NOT transliterate — use the proper script of {language_name}.

Guidelines:
- Cite the exact law section or act (e.g. "Section 6, RTI Act 2005").
- Give clear numbered step-by-step instructions.
- Use simple language. Explain legal terms immediately.
- If serious or complex, recommend NALSA free legal aid (call 15100).
- Mention government portals and helpline numbers where relevant.
- Be empathetic — users are often frustrated or scared.

Context from legal documents:
{context}

Question: {question}

⚠️ REMINDER: Your ENTIRE answer must be in {language_name} only. Not a single word in any other language.

Answer in {language_name}:"""

CATEGORY_FILES = {
    "RTI":            ["rti_act_2005.txt", "rti_templates.txt"],
    "Consumer":       ["consumer_protection.txt"],
    "Constitutional": ["constitutional_rights.txt"],
    "Criminal":       ["ipc_sections.txt"],
    "Labour":         ["rti_templates.txt", "state_and_misc_laws.txt"],
    "Family":         ["family_law.txt"],
    "Property":       ["property_land.txt", "rent_landlord.txt"],
    "Cyber":          ["cyber_crime_detailed.txt"],
    "Finance":        ["tax_banking_finance.txt"],
    "Schemes":        ["government_schemes.txt"],
    "Company":        ["company_corporate_law.txt"],
}

ALL_FILES = [
    "rent_landlord.txt",
    "motor_vehicles.txt",
    "family_law.txt",
    "property_land.txt",
    "cyber_crime_detailed.txt",
    "education_child_rights.txt",
    "senior_citizen_rights.txt",
    "tax_banking_finance.txt",
    "government_schemes.txt",
    "company_corporate_law.txt",
    "rti_act_2005.txt",
    "consumer_protection.txt",
    "constitutional_rights.txt",
    "ipc_sections.txt",
    "rti_templates.txt",
    "state_and_misc_laws.txt",
]

LOW_CONFIDENCE = [r"not sure", r"unclear", r"cannot determine",
                  r"consult a lawyer", r"complex", r"case-specific", r"varies by"]
LAWYER_KEYWORDS = ["arrest", "custody", "bail", "murder", "rape", "land dispute",
                   "divorce", "criminal case", "high court", "supreme court",
                   "property fraud", "money laundering"]


def _confidence(answer: str, question: str):
    low = sum(bool(re.search(p, answer.lower())) for p in LOW_CONFIDENCE)
    lawyer = any(k in question.lower() or k in answer.lower() for k in LAWYER_KEYWORDS)
    if low >= 2 or lawyer:
        return "low", True
    if low == 1:
        return "medium", False
    return "high", False


def _quick_actions(answer: str, category: Optional[str]) -> list[str]:
    actions = []
    if category == "RTI" or "rtionline" in answer.lower():
        actions.append("🔗 File RTI Online → https://rtionline.gov.in")
    if category == "Consumer" or "edaakhil" in answer.lower():
        actions.append("🔗 Consumer Complaint → https://edaakhil.nic.in")
    if "cybercrime" in answer.lower():
        actions.append("🔗 Report Cybercrime → https://cybercrime.gov.in")
    if "pf" in answer.lower() or "epf" in answer.lower():
        actions.append("🔗 EPFO Grievance → https://epfigms.gov.in")
    if "rbi" in answer.lower():
        actions.append("🔗 RBI Ombudsman → https://cms.rbi.org.in")
    if not actions:
        actions.append("📞 NALSA Free Legal Aid → Call 15100")
    return actions[:4]


def _detect_category(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["rti", "right to information", "pio", "public information"]):
        return "RTI"
    if any(k in q for k in ["consumer", "refund", "product", "builder", "rera", "ecommerce"]):
        return "Consumer"
    if any(k in q for k in ["fir", "police", "arrest", "bail", "criminal", "fraud", "cheating"]):
        return "Criminal"
    if any(k in q for k in ["salary", "pf", "epf", "gratuity", "labour", "maternity", "employer", "wages"]):
        return "Labour"
    return "Constitutional"


class CitizenAidRAG:
    def __init__(self):
        print(f"📂 ChromaDB path: {CHROMA_DIR}")
        key = os.getenv("GROQ_API_KEY", "")
        print(f"🔑 Groq API key loaded: {'Yes ✅' if key else 'NO KEY FOUND ❌'}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = None
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=key,
            temperature=0.2,
            max_tokens=1500,
        )

    def ingest_documents(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        if (CHROMA_DIR / "chroma.sqlite3").exists():
            print("📚 Loading existing ChromaDB …")
            self.vectorstore = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=self.embeddings,
                collection_name="citizenaid",
            )
            print("✅ ChromaDB loaded.")
            return

        print("📚 Ingesting legal documents …")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=120,
            separators=["\n\n", "\n", ".", " "],
        )

        all_docs: list[Document] = []
        for fname in ALL_FILES:
            fpath = DATA_DIR / fname
            if not fpath.exists():
                print(f"  ⚠️  {fname} not found at {fpath}")
                continue
            loader = TextLoader(str(fpath), encoding="utf-8")
            raw    = loader.load()
            meta   = {"source_file": fname, "category": "General"}
            for line in raw[0].page_content.splitlines()[:3]:
                for part in line.split("|"):
                    part = part.strip()
                    if part.startswith("SOURCE:"):
                        meta["source"] = part.replace("SOURCE:", "").strip()
                    if part.startswith("CATEGORY:"):
                        meta["category"] = part.replace("CATEGORY:", "").strip()
            chunks = splitter.split_documents(raw)
            for c in chunks:
                c.metadata.update(meta)
            all_docs.extend(chunks)
            print(f"  ✅  {fname} → {len(chunks)} chunks")

        self.vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            persist_directory=str(CHROMA_DIR),
            collection_name="citizenaid",
        )
        print(f"✅ Ingested {len(all_docs)} chunks into ChromaDB.")

    def query(self, question: str, category: Optional[str] = None, language: str = "en") -> dict:
        language_name = LANGUAGE_NAMES.get(language, "English")

        # Always search in English for better retrieval (embeddings work best in English)
        # But force the LLM answer to be in chosen language
        try:
            if category and category in CATEGORY_FILES:
                docs = self.vectorstore.similarity_search(
                    question, k=6,
                    filter={"source_file": {"$in": CATEGORY_FILES[category]}},
                )
                if not docs:
                    docs = self.vectorstore.similarity_search(question, k=6)
            else:
                docs = self.vectorstore.similarity_search(question, k=6)
        except Exception:
            docs = self.vectorstore.similarity_search(question, k=6)

        context  = "\n\n".join(d.page_content for d in docs)
        prompt   = SYSTEM_PROMPT.replace("{context}", context) \
                                 .replace("{question}", question) \
                                 .replace("{language_name}", language_name)
        response = self.llm.invoke(prompt)
        answer   = response.content

        conf, nl = _confidence(answer, question)
        return {
            "answer":        answer,
            "sources":       self._format_sources(docs),
            "category":      category or _detect_category(question),
            "confidence":    conf,
            "needs_lawyer":  nl,
            "quick_actions": _quick_actions(answer, category),
        }

    def _format_sources(self, docs: list[Document]) -> list[dict]:
        seen, result = set(), []
        for d in docs:
            key = d.metadata.get("source_file", "unknown")
            if key not in seen:
                seen.add(key)
                result.append({
                    "file":     key,
                    "category": d.metadata.get("category", "General"),
                    "source":   d.metadata.get("source", key),
                    "excerpt":  d.page_content[:200].strip(),
                })
        return result[:4]

    def get_retriever(self):
        return self.vectorstore.as_retriever(search_kwargs={"k": 5})

# Patch _detect_category with extended categories (appended)
def _detect_category(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["rti", "right to information", "pio", "public information"]):
        return "RTI"
    if any(k in q for k in ["consumer", "refund", "product", "builder", "rera", "ecommerce", "amazon", "flipkart"]):
        return "Consumer"
    if any(k in q for k in ["fir", "police", "arrest", "bail", "criminal", "fraud", "cheating", "stolen", "theft"]):
        return "Criminal"
    if any(k in q for k in ["salary", "pf", "epf", "gratuity", "labour", "maternity", "employer", "wages", "job", "office"]):
        return "Labour"
    if any(k in q for k in ["rent", "landlord", "tenant", "eviction", "lease", "deposit", "flat", "house", "room"]):
        return "Property"
    if any(k in q for k in ["wife", "husband", "divorce", "marriage", "dowry", "domestic violence", "maintenance", "custody", "child"]):
        return "Family"
    if any(k in q for k in ["cyber", "online fraud", "hacked", "otp", "phishing", "sextortion", "blackmail", "fake website", "upi fraud"]):
        return "Cyber"
    if any(k in q for k in ["tax", "itr", "income tax", "tds", "bank", "loan", "emi", "insurance", "mutual fund", "credit", "cibil"]):
        return "Finance"
    if any(k in q for k in ["scheme", "ration", "pmay", "mgnrega", "ayushman", "pension", "pm kisan", "mudra", "subsidy", "benefit"]):
        return "Schemes"
    if any(k in q for k in ["accident", "vehicle", "car", "bike", "motor", "challan", "licence", "insurance claim"]):
        return "Motor"
    if any(k in q for k in ["property", "land", "plot", "registration", "mutation", "encroachment", "will", "inheritance"]):
        return "Property"
    if any(k in q for k in ["school", "education", "rte", "scholarship", "child", "pocso", "sexual abuse"]):
        return "Education"
    if any(k in q for k in ["senior citizen", "old age", "parent", "mother", "father", "elderly", "pension"]):
        return "Schemes"
    if any(k in q for k in ["company", "startup", "pvt ltd", "private limited", "llp", "director", "shareholder", "incorporation", "mca", "nclt", "msme", "trademark", "patent", "gst registration", "business registration", "ibc", "insolvency"]):
        return "Company"
    return "Constitutional"