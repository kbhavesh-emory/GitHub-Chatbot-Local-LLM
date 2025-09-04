# 🤖 GitHub-Chatbot-LLM

A **local RAG-based chatbot** for exploring GitHub repositories with **Ollama LLMs** (LLaMA 3, Mistral, etc.), **FAISS** embeddings, and a clean **React frontend** + **FastAPI backend**.

---

## ✨ Features
- 🔍 Retrieval-Augmented Generation (RAG) over cloned repositories 
- ⚡ GPU-accelerated embeddings with Hugging Face BGE models 
- 🖥️ FastAPI backend + React (Vite) frontend 
- 💾 Persistent FAISS vectorstore for efficient queries 
- 🛠️ Easy setup with virtual environment & Ollama models 

---

## 📦 Tech Stack
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18-61dafb?logo=react)
![Vite](https://img.shields.io/badge/Vite-5-purple?logo=vite)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLMs-black?logo=ollama)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-orange?logo=databricks)

---

## 📂 Project Structure
```
GitHub-Chatbot-LLM/
├── backend/
│ ├── api.py # FastAPI server
│ ├── chat/
│ │ ├── rag_chat.py # core RAG logic
│ │ ├── history_utils.py
│ │ └── logging_config.py
│ ├── ingest/
│ │ └── store_embeddings.py # embedding pipeline
│ ├── clone_dsa_repos.sh # repository cloning
│ ├── vectorstore/ # FAISS indices
│ └── data/ # cloned repositories
├── frontend/
│ ├── src/
│ │ ├── App.jsx # main React app
│ │ ├── main.jsx
│ │ └── App.css
│ ├── index.html
│ └── vite.config.js
└── active.sh # environment setup
```

---

## ⚙️ Setup Guide

### 1️⃣ Clone Required Repositories
```bash
cd backend
chmod +x clone_dsa_repos.sh
bash clone_dsa_repos.sh
```

### 2️⃣ Pull Ollama Model
```bash
ollama pull llama3:latest
```

### 3️⃣ Create & Activate Virtual Environment
```bash
cd /nashome/bhavesh/GitHub-Chatbot-LLM
python3.13 -m venv backend/.venv
source backend/.venv/bin/activate
pip install --upgrade pip
```

Install Torch (GPU-enabled):
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
```

Optional (for PDFs/Office/images):
```bash
pip install "unstructured[all-docs]"
```

Project dependencies:
```bash
pip install -r backend/requirements.txt
```

---

### 4️⃣ Build Embeddings

**Base model:**
```bash
python backend/ingest/store_embeddings.py backend/data/ --out backend/vectorstore --model BAAI/bge-base-en-v1.5 --device auto --batch-size 32
```

**Large model:**
```bash
python backend/ingest/store_embeddings.py backend/data/ --out backend/vectorstore --model BAAI/bge-large-en-v1.5 --device auto --chunk-size 1200 --overlap 200 --batch-size 64 --chunks-per-batch 8000
```

**Background run:**
```bash
nohup python backend/ingest/store_embeddings.py backend/data/ --out backend/vectorstore --model BAAI/bge-large-en-v1.5 --device auto --chunk-size 1200 --overlap 200 --batch-size 64 --chunks-per-batch 8000 > embeddings.log 2>&1 &
```

**Re-embedding multiple folders:**
```bash
nohup python backend/ingest/store_embeddings.py backend/data backend/data/otherfiles --out backend/vectorstore --model BAAI/bge-large-en-v1.5 --device cuda --chunk-size 1200 --overlap 200 --batch-size 128 --chunks-per-batch 16000 --workers 8 --lang en --rembed > embeddings.log 2>&1 &
```

📌 Output is saved to:
```
backend/vectorstore/
```

The model name is stored in:
```
backend/vectorstore/model.txt
```

---

### 5️⃣ Start Backend API

**Interactive run:**
```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

**Background run:**
```bash
cd backend
nohup uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
```

**Health check:**
```bash
curl http://localhost:8000/health
```

---

### 6️⃣ Launch Frontend
```bash
cd frontend
npm install
npm run dev
```

**Background run:**
```bash
nohup npm run dev > npm.log 2>&1 &
```

Open in browser:
```
http://<host>:5173
```

---

### 7️⃣ Quick API Test
```bash
curl -s http://localhost:8000/chat -H 'Content-Type: application/json' -d '{"question":"How does HistomicsUI talk to HistomicsTK?", "model":"llama3:latest"}' | jq
```

---

## ✅ Done!
Your chatbot should now be running with both **backend** and **frontend** accessible.

---

## 📜 License
Gutman-Lab, Department of Pathology & Laboratory Medicine, Emory University
