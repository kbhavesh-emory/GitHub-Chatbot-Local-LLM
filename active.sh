#!/bin/bash
# active.sh - Enhanced environment setup
echo "Setting up GitHub Chatbot environment..."

# Activate virtual environment
source backend/.venv/bin/activate

# Set project root
cd /opt/bhavesh/GitHub-Chatbot-Local-LLM

# Environment variables for optimized performance
export VECTORSTORE_DIR=/opt/bhavesh/GitHub-Chatbot-Local-LLM/backend/vectorstore
export REPO_DIR=/opt/bhavesh/GitHub-Chatbot-Local-LLM/backend/data
export LLM_MODEL=llama3:latest
export EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
export OLLAMA_BASE_URL=http://localhost:11434
export EMBED_DEVICE=cuda
export TORCH_CUDA_ARCH_LIST="6.0"  # P100 compute capability
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Optimize for P100
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false
export SEARCH_TYPE=similarity_score_threshold
export SCORE_THRESHOLD=0.7

echo " Environment configured for P100 GPU"
echo "   - Model: $LLM_MODEL"
echo "   - Embedding: $EMBEDDING_MODEL"
echo "   - Device: $EMBED_DEVICE"
