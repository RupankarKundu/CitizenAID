#!/usr/bin/env bash
set -e
BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; X='\033[0m'
info()    { echo -e "${CYAN}[CitizenAid]${X} $*"; }
success() { echo -e "${GREEN}[✓]${X} $*"; }
warn()    { echo -e "${YELLOW}[!]${X} $*"; }
error()   { echo -e "${RED}[✗]${X} $*"; exit 1; }

echo ""
echo -e "${BOLD}⚖  CitizenAid — Legal Rights & RTI Filing Assistant${X}"
echo -e "   RAG · ChromaDB · LangChain · FastAPI · Gemini"
echo ""

# Python check
PY=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo "$PY" | cut -d. -f1); MINOR=$(echo "$PY" | cut -d. -f2)
[ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ] || error "Python 3.10+ required (found $PY)"
success "Python $PY"

# .env
[ -f ".env" ] || { cp .env.example .env; warn ".env created — add your GOOGLE_API_KEY!"; }
set -a; source .env; set +a

if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "AIzaSy_paste_your_key_here" ]; then
  echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${X}"
  echo -e "${YELLOW}  GOOGLE_API_KEY is not set in .env               ${X}"
  echo -e "${YELLOW}  Get your free key: https://aistudio.google.com  ${X}"
  echo -e "${YELLOW}  Edit .env then re-run: bash setup.sh            ${X}"
  echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${X}\n"
  exit 1
fi
success "GOOGLE_API_KEY found"

# Venv
[ -d "venv" ] || python3 -m venv venv
source venv/bin/activate
success "Virtual environment ready"

# Dependencies
info "Installing dependencies…"
pip install --quiet --upgrade pip
pip install --quiet -r backend/requirements.txt
success "Dependencies installed"

# Data files
info "Checking legal data files…"
for f in rti_act_2005.txt consumer_protection.txt constitutional_rights.txt ipc_sections.txt rti_templates.txt state_and_misc_laws.txt; do
  [ -f "data/$f" ] || error "Missing: data/$f"
done
success "All 6 knowledge-base files present"

# Pre-cache embedding model
info "Pre-caching embedding model (one-time ~120 MB download)…"
python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
_ = m.encode(['test'])
print('  Model ready.')
"
success "Embedding model cached"

PORT="${PORT:-8000}"
echo ""
echo -e "${BOLD}🚀 Starting CitizenAid → http://localhost:${PORT}${X}"
echo -e "   API docs: ${CYAN}http://localhost:${PORT}/docs${X}"
echo -e "   Press Ctrl+C to stop."
echo ""

cd backend
GOOGLE_API_KEY="$GOOGLE_API_KEY" python3 -m uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload --log-level info
