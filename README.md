📂 Project Structure
GitHub-Chatbot-LLM/
├── backend/
│   ├── api.py                # FastAPI server
│   ├── chat/
│   │   ├── rag_chat.py       # core RAG logic
│   │   ├── history_utils.py
│   │   └── logging_config.py
│   ├── ingest/
│   │   └── store_embeddings.py  # embedding pipeline
│   ├── clone_dsa_repos.sh    # repository cloning
│   ├── vectorstore/          # FAISS indices
│   └── data/                 # cloned repositories
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # main React app
│   │   ├── main.jsx
│   │   └── App.css
│   ├── index.html
│   └── vite.config.js
└── active.sh                 # environment setup

⚙️ 1) Clone Required Repositories
cd backend
chmod +x clone_dsa_repos.sh
bash clone_dsa_repos.sh

🧠 2) Pull Ollama Model
ollama pull llama3:latest

🐍 3) Create & Activate Python Virtual Environment

From project root:

cd /nashome/bhavesh/GitHub-Chatbot-LLM

# Create venv
python3.13 -m venv backend/.venv

# Activate venv
source backend/.venv/bin/activate
pip install --upgrade pip


Install Torch (GPU-enabled):

pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124


Optional (for PDFs/Office/images):

pip install "unstructured[all-docs]"


Install project dependencies:

pip install -r backend/requirements.txt

🏗️ 4) Build Embeddings

Run once inside the virtual environment:

Base model:

python backend/ingest/store_embeddings.py backend/data/ \
  --out backend/vectorstore \
  --model BAAI/bge-base-en-v1.5 \
  --device auto \
  --batch-size 32


Large model:

python backend/ingest/store_embeddings.py backend/data/ \
  --out backend/vectorstore \
  --model BAAI/bge-large-en-v1.5 \
  --device auto \
  --chunk-size 1200 --overlap 200 \
  --batch-size 64 --chunks-per-batch 8000


Background run:

nohup python backend/ingest/store_embeddings.py backend/data/ \
  --out backend/vectorstore \
  --model BAAI/bge-large-en-v1.5 \
  --device auto \
  --chunk-size 1200 --overlap 200 \
  --batch-size 64 --chunks-per-batch 8000 > embeddings.log 2>&1 &


Re-embedding multiple folders:

nohup python backend/ingest/store_embeddings.py backend/data backend/data/otherfiles \
  --out backend/vectorstore \
  --model BAAI/bge-large-en-v1.5 \
  --device cuda \
  --chunk-size 1200 --overlap 200 \
  --batch-size 128 --chunks-per-batch 16000 \
  --workers 8 --lang en --rembed > embeddings.log 2>&1 &


📌 Output is saved to:

backend/vectorstore/


The model name is stored in:

/nashome/bhavesh/GitHub-Chatbot-LLM/backend/vectorstore/model.txt

🔌 5) Start the Backend API

Interactive run:

uvicorn backend.api:app --host 0.0.0.0 --port 8000


Background run:

cd backend
nohup uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &


Health check:

curl http://localhost:8000/health

💻 6) Launch Frontend
cd frontend
npm install
npm run dev


Background run (from backend):

nohup npm run dev > npm.log 2>&1 &


Open in browser:

http://<host>:5173

🔎 7) Quick API Test
curl -s http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does HistomicsUI talk to HistomicsTK?", "model":"gpt-oss:20b"}' | jq
