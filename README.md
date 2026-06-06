---
title: CitizenAID
emoji: ⚖️
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
# ⚖️ CitizenAid — Legal Rights & RTI Filing Assistant

> Know your rights. File with confidence. No lawyer needed for 80% of common issues.

A production-ready **RAG (Retrieval-Augmented Generation)** application that ingests Indian legal documents into a local vector database and answers citizen queries with step-by-step guidance, RTI draft generation, and exact law citations — in English, Hindi, and Bengali.

**LLM: Google Gemini 2.0 Flash — Free tier, 1,500 requests/day, no credit card needed.**

---

## 🚀 Quick Start (3 steps)

```bash
# 1. Get free Gemini API key at https://aistudio.google.com
# 2. Add key to .env
echo "GOOGLE_API_KEY=AIzaSy..." > .env

# 3. Run
bash setup.sh
```

Open **http://localhost:8000** — done.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CitizenAid Stack                       │
│                                                             │
│  Browser (index.html)                                       │
│       │  fetch()                                            │
│       ▼                                                     │
│  FastAPI  (/api/query  /api/rti/draft  /health)             │
│       │                                                     │
│       ├──► RAG Pipeline (rag_pipeline.py)                   │
│       │         │                                           │
│       │    LangChain RetrievalQA                            │
│       │         ├──► ChromaDB (HNSW)  ◄── HF Embeddings    │
│       │         │       (local, CPU, free)                  │
│       │         └──► Gemini 2.0 Flash (LLM)                 │
│       │                                                     │
│       └──► RTI Agent  (rti_agent.py)                        │
│                 └──► Gemini 2.0 Flash (draft generation)    │
│                                                             │
│  Knowledge Base (data/)                                     │
│    RTI Act 2005 · Consumer Protection Act 2019              │
│    Constitution of India · IPC/BNS 2023                     │
│    RTI Templates · State & Regulatory Laws                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology | Cost |
|---|---|---|
| LLM | Gemini 2.0 Flash | Free (1,500 req/day) |
| Orchestration | LangChain RetrievalQA | Free |
| Vector DB | ChromaDB (local) | Free |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | Free (runs on CPU) |
| API | FastAPI (async) | Free |
| Frontend | Vanilla HTML/CSS/JS | Free |

**Total hosting cost: ₹0**

---

## 📁 Project Structure

```
citizenaid/
├── backend/
│   ├── main.py            FastAPI app — routes, startup
│   ├── rag_pipeline.py    ChromaDB ingestion + LangChain RetrievalQA
│   ├── rti_agent.py       Multi-step RTI draft generator
│   └── requirements.txt   All Python dependencies
│
├── data/                  Legal knowledge base (plain text)
│   ├── rti_act_2005.txt
│   ├── consumer_protection.txt
│   ├── constitutional_rights.txt
│   ├── ipc_sections.txt
│   ├── rti_templates.txt
│   └── state_and_misc_laws.txt
│
├── frontend/
│   └── index.html         Single-page UI — dark/light mode, no build step
│
├── chroma_db/             Auto-created on first run (gitignored)
│
├── test_rag.py            Full test suite (8 sections)
├── setup.sh               One-shot install + launch script
├── Dockerfile             Multi-stage, model pre-baked
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## 🔌 API Reference

### POST `/api/query`
```json
// Request
{ "question": "My PF was not deposited for 6 months. What can I do?",
  "language": "en",
  "category": "Labour" }

// Response
{ "answer": "Under the Employees' Provident Funds Act...",
  "sources": [{ "file": "rti_templates.txt", "category": "Labour Rights", "excerpt": "..." }],
  "category": "Labour",
  "confidence": "high",
  "needs_lawyer": false,
  "quick_actions": ["🔗 EPFO Grievance → https://epfigms.gov.in"] }
```

### POST `/api/rti/draft`
```json
// Request
{ "situation": "My MGNREGA wages for March 2024 not paid after 5 months.",
  "department": "Block Development Office",
  "state": "West Bengal",
  "applicant_name": "Ramesh Mahato" }

// Response
{ "draft": "To,\nThe Public Information Officer...",
  "steps": ["1️⃣ Identify the PIO...", "..."],
  "authority": "Block Development Officer / Panchayati Raj Dept",
  "portal_link": "State RTI portal",
  "tips": ["📌 Be specific...", "..."] }
```

### GET `/health`
```json
{ "status": "ok", "rag_ready": true }
```

---

## 🌐 Language Support

| Code | Language | How to use |
|---|---|---|
| `en` | English | Default |
| `hi` | Hindi | Select HI in UI or pass `"language":"hi"` |
| `bn` | Bengali | Select BN in UI or pass `"language":"bn"` |

The multilingual embedding model handles semantic search across all three languages automatically.

---

## ⚙️ Configuration (`.env`)

```bash
GOOGLE_API_KEY=AIzaSy...   # Required — get free at aistudio.google.com
PORT=8000                   # Optional — default 8000
HOST=0.0.0.0                # Optional
```

---

## 🐳 Docker

```bash
# With docker-compose (recommended)
docker-compose up --build

# Plain Docker
docker build -t citizenaid .
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=AIzaSy... \
  -v citizenaid_chroma:/app/chroma_db \
  citizenaid
```

---

## 🧪 Running Tests

```bash
source venv/bin/activate
export GOOGLE_API_KEY="AIzaSy..."
python3 test_rag.py
```

8 test sections: data files → imports → embeddings → ingestion → category detection → full query → RTI agent → live API.

---

## ➕ Adding More Legal Documents

1. Add a `.txt` file to `data/` starting with:
   ```
   SOURCE: [Act name] | CATEGORY: [RTI/Consumer/Constitutional/Criminal/Labour] | JURISDICTION: India
   ```
2. Delete `chroma_db/` folder
3. Restart — the pipeline rebuilds automatically

Ideas: Motor Vehicles Act, POCSO Act, Forest Rights Act, Domestic Violence Act, state-specific RTI portals.

---

## 📞 Emergency Helplines

| Service | Number |
|---|---|
| Police | 100 |
| Emergency | 112 |
| Women helpline | 1091 / 181 |
| Child helpline | 1098 |
| Cyber fraud (immediate) | 1930 |
| Consumer helpline | 1800-11-4000 |
| Free legal aid (NALSA) | 15100 |
| Senior citizens | 14567 |

---

## ⚠️ Disclaimer

CitizenAid provides general legal information based on publicly available Indian laws. It is not a substitute for professional legal advice. For complex matters — arrest, custody, property disputes, court cases — consult a qualified advocate. Free legal aid: **NALSA 15100**.
