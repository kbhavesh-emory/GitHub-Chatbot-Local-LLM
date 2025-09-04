# backend/chat/rag_chat.py
from __future__ import annotations

import os
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator, Iterable, DefaultDict
from collections import defaultdict

import numpy as np
import torch

from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # modern import
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Optional cross-encoder reranker (if installed)
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # type: ignore

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("rag-chat")

# -----------------------------------------------------------------------------
# Config (env-overridable)
# -----------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL   = os.getenv("LLM_MODEL", "llama3:latest")
DEFAULT_INDEX_DIR = Path(os.getenv("VECTORSTORE_DIR", "backend/vectorstore"))

# Embeddings
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
EMBED_DEVICE    = os.getenv("EMBED_DEVICE", "auto")  # "cpu" | "cuda" | "auto"
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))

# Retrieval knobs
DEFAULT_K          = int(os.getenv("RETRIEVE_K", "8"))         # final top-k docs
DEFAULT_FETCH_K    = int(os.getenv("RETRIEVE_FETCH_K", "120")) # how many to pull before fusing/rerank
DEFAULT_LAMBDA     = float(os.getenv("RETRIEVE_LAMBDA", "0.6"))# MMR diversity
SCORE_THRESHOLD    = float(os.getenv("SCORE_THRESHOLD", "0.65"))
RRF_K              = int(os.getenv("RRF_K", "60"))             # RRF smoothing constant
MAX_CONTEXT_DOCS   = int(os.getenv("MAX_CONTEXT_DOCS", "24"))  # after expansion-by-source

# Cross-encoder reranker (optional)
CROSS_ENCODER_MODEL = os.getenv("RERANK_MODEL", "")  # e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_TOPN  = int(os.getenv("RERANK_TOPN", "64"))      # rerank this many fused docs

# LLM generation knobs
LLM_NUM_CTX      = int(os.getenv("LLM_NUM_CTX", "8192"))
LLM_NUM_PREDICT  = int(os.getenv("LLM_NUM_PREDICT", "1536"))
LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Rebuild support (if you want to build the index here)
GLOBS = [
    "**/*.md", "**/*.py", "**/*.txt", "**/*.rst",
    "**/*.yaml", "**/*.yml", "**/*.ini", "**/*.cfg", "**/*.json",
]

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
_LLM: Optional[ChatOllama] = None
_VECTORSTORE: Optional[FAISS] = None
_MMR_RETRIEVER = None
_EMB: Optional[HuggingFaceEmbeddings] = None
_BM25: Optional[BM25Retriever] = None
_CE: Optional[Any] = None  # CrossEncoder instance

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _cuda_ok() -> bool:
    if EMBED_DEVICE == "cpu":
        return False
    if EMBED_DEVICE == "cuda" and not torch.cuda.is_available():
        return False
    if not torch.cuda.is_available():
        return False
    try:
        _ = (torch.ones(1, device="cuda") + 1).item()
        return True
    except Exception:
        return False

def _read_model_txt(index_dir: Path) -> str:
    p = index_dir / "model.txt"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return ""

def _parse_model_name(modeltxt: str) -> str:
    if not modeltxt:
        return DEFAULT_EMBED_MODEL
    return (modeltxt.split("|", 1)[0] or DEFAULT_EMBED_MODEL).strip()

def _mk_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    device = "cuda" if _cuda_ok() else "cpu"
    model_kwargs: Dict[str, Any] = {"device": device}
    encode_kwargs: Dict[str, Any] = {"normalize_embeddings": True, "batch_size": EMBED_BATCH_SIZE}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

def _mk_llm(model_name: str = DEFAULT_MODEL) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_ctx=LLM_NUM_CTX,
        num_predict=LLM_NUM_PREDICT,
        top_p=0.9,
        repeat_penalty=1.1,
        top_k=40,
    )

def _collect_all_docs_from_faiss(vs: FAISS) -> List[Document]:
    docs: List[Document] = []
    for _, doc_id in vs.index_to_docstore_id.items():
        found = vs.docstore.search(doc_id)
        if isinstance(found, Document):
            docs.append(found)
        elif isinstance(found, list):
            for d in found:
                if isinstance(d, Document):
                    docs.append(d)
    return docs

def _build_bm25_from_vs(vs: FAISS) -> BM25Retriever:
    corpus = _collect_all_docs_from_faiss(vs)
    if not corpus:
        return BM25Retriever.from_texts([""], metadatas=[{"source": "empty"}])
    bm25 = BM25Retriever.from_documents(corpus)
    bm25.k = max(DEFAULT_FETCH_K // 2, 20)
    log.info("BM25 ready over %d docs", len(corpus))
    return bm25

def _cosine_similarity(a, b) -> float:
    a = np.asarray(a); b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)

def _embed_rerank(query: str, docs: List[Document], emb: HuggingFaceEmbeddings) -> List[Document]:
    if not docs:
        return docs
    try:
        qv = emb.embed_query(query)
        scored: List[Tuple[float, Document]] = []
        for d in docs:
            dv = emb.embed_query(d.page_content)
            scored.append((_cosine_similarity(qv, dv), d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored]
    except Exception as e:
        log.warning("Embed rerank failed: %s", e)
        return docs

def _get_cross_encoder() -> Optional[Any]:
    global _CE
    if _CE is not None:
        return _CE
    if not CROSS_ENCODER_MODEL or CrossEncoder is None:
        return None
    try:
        device = "cuda" if _cuda_ok() else "cpu"
        _CE = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
        log.info("CrossEncoder loaded: %s (%s)", CROSS_ENCODER_MODEL, device)
        return _CE
    except Exception as e:
        log.warning("CrossEncoder load failed: %s", e)
        return None

def _ce_rerank(query: str, docs: List[Document], topn: int) -> List[Document]:
    ce = _get_cross_encoder()
    if ce is None or not docs:
        return docs
    cand = docs[:topn]
    pairs = [[query, d.page_content[:3000]] for d in cand]
    try:
        scores = ce.predict(pairs)
        order = np.argsort(-np.asarray(scores))
        return [cand[i] for i in order]
    except Exception as e:
        log.warning("Cross-encoder rerank failed: %s", e)
        return docs

def _rrf_fuse(a: List[Document], b: List[Document], k: int = RRF_K, cap: int = DEFAULT_FETCH_K) -> List[Document]:
    ranks: Dict[str, float] = {}
    by_key: Dict[str, Document] = {}

    def key_of(d: Document) -> str:
        return f"{d.metadata.get('source','')}#{d.metadata.get('id', id(d))}"

    def add_list(lst: List[Document], weight: float = 1.0):
        for r, d in enumerate(lst[:cap], start=1):
            key = key_of(d)
            ranks[key] = ranks.get(key, 0.0) + weight * (1.0 / (k + r))
            by_key.setdefault(key, d)

    add_list(a, 1.0)
    add_list(b, 1.0)
    fused = sorted(by_key.items(), key=lambda kv: ranks.get(kv[0], 0.0), reverse=True)
    return [d for _, d in fused]

def _unique_docs(docs: Iterable[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        key = (d.metadata.get("source", ""), d.page_content[:256])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

def _expand_by_source(primary: List[Document], vs: FAISS, limit_per_source: int = 3, total_cap: int = MAX_CONTEXT_DOCS) -> List[Document]:
    """Pull a couple extra chunks from the same files to give more surrounding context."""
    if not primary:
        return primary
    corpus = _collect_all_docs_from_faiss(vs)
    buckets: DefaultDict[str, List[Document]] = defaultdict(list)
    for d in corpus:
        src = d.metadata.get("source") or d.metadata.get("path") or "unknown"
        buckets[src].append(d)

    out: List[Document] = []
    seen = set()

    def add(doc: Document):
        key = (doc.metadata.get("source", ""), doc.page_content[:256])
        if key not in seen:
            seen.add(key)
            out.append(doc)

    # Keep primary ordering
    for d in primary:
        add(d)

    # Expand by source
    for d in primary:
        src = d.metadata.get("source") or d.metadata.get("path") or "unknown"
        avail = buckets.get(src, [])
        added = 0
        for cand in avail:
            if added >= limit_per_source - 1:
                break
            k = (cand.metadata.get("source", ""), cand.page_content[:256])
            if k in seen:
                continue
            add(cand)
            added += 1
        if len(out) >= total_cap:
            break

    return out[:total_cap]

# -----------------------------------------------------------------------------
# Query expansion (deterministic + optional LLM)
# -----------------------------------------------------------------------------
_COMMON_FIXES = {
    r"\bgrider\b": "girder",
    r"\bgirder\s*client\b": "girder-client",
    r"\bjsons\b": "json",
    r"\byamls\b": "yaml",
}

def _normalize_question(q: str) -> str:
    x = " ".join((q or "").strip().split())
    for pat, rep in _COMMON_FIXES.items():
        x = re.sub(pat, rep, x, flags=re.IGNORECASE)
    return x

def _gen_query_variants(llm: ChatOllama, question: str, n: int = 6) -> List[str]:
    base = [_normalize_question(question).lower()]
    if n <= 1:
        return base

    q = base[0]
    expansions = {q}
    expansions.add(q.replace("-", " "))
    expansions.add(q.replace(":", " "))
    expansions.add(q + " install")
    expansions.add(q + " setup")
    expansions.add(q + " error")
    expansions.add(q + " api")
    expansions.add(q + " example")
    words = q.split()
    if len(words) >= 2:
        expansions.add(" ".join(words[:2]))
        expansions.add(" ".join(words[-2:]))

    variants = list(expansions)
    try:
        need = max(0, n - len(variants))
        if need > 0:
            prompt = (
                "Create concise search query variations for technical code/docs. "
                "One per line, no bullets. ORIGINAL: "
                + question
            )
            out = llm.invoke(prompt)
            text = getattr(out, "content", str(out))
            for line in text.splitlines():
                line = line.strip().lower()
                if line and not line[:2].isdigit() and not line.startswith(("-", "*", "•")):
                    variants.append(line)
    except Exception as e:
        log.debug("LLM variant step skipped: %s", e)

    seen = set()
    final: List[str] = []
    for v in [*base, *variants]:
        if v not in seen:
            seen.add(v)
            final.append(v)
        if len(final) >= n:
            break
    log.info("Query variants: %s", final)
    return final

# -----------------------------------------------------------------------------
# Build / load index (optional rebuild path)
# -----------------------------------------------------------------------------
def _gather_docs(repo_dir: str | Path) -> List[Document]:
    repo_dir = Path(repo_dir)
    docs: List[Document] = []
    for pattern in GLOBS:
        loader = DirectoryLoader(
            str(repo_dir),
            glob=pattern,
            loader_cls=TextLoader,
            recursive=True,
            silent_errors=True,
            show_progress=False,
        )
        docs.extend(loader.load())
    if not docs:
        docs = [Document(page_content=f"No parseable files found in {repo_dir}", metadata={"path": str(repo_dir)})]
    return docs

def _split_docs(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 120) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(docs)

def load_or_create_vectorstore(
    repo_dir: Optional[str | Path] = None,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    rebuild: bool = False,
) -> FAISS:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    has_any = index_dir.exists() and any(index_dir.iterdir())
    if has_any and not rebuild:
        model_name = _parse_model_name(_read_model_txt(index_dir))
        emb = _mk_embeddings(model_name)
        vs = FAISS.load_local(str(index_dir), emb, allow_dangerous_deserialization=True)
        return vs

    if repo_dir is None:
        raise ValueError("repo_dir is required to (re)build the vectorstore.")

    log.info("Rebuilding FAISS at %s from repo %s", index_dir, repo_dir)
    docs = _split_docs(_gather_docs(repo_dir))
    emb = _mk_embeddings(DEFAULT_EMBED_MODEL)
    vs = FAISS.from_documents(docs, emb)
    vs.save_local(str(index_dir))
    try:
        (index_dir / "model.txt").write_text(
            f"{DEFAULT_EMBED_MODEL}|backend=langchain_huggingface",
            encoding="utf-8"
        )
    except Exception:
        pass
    return vs

# -----------------------------------------------------------------------------
# Prompts: map → plan → draft → critique → revise (ChatGPT-style)
# -----------------------------------------------------------------------------
MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract only facts from the excerpt that help answer the user's question. "
     "Prefer concrete details: commands, flags, versions, file paths, config keys, input/output, constraints, gotchas. "
     "Avoid speculation. Keep bullets terse; preserve exact strings where possible."),
    ("human",
     "Question:\n{question}\n\nExcerpt:\n{excerpt}\n\n"
     "Return 5–15 concise bullets of relevant facts.")
])

PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Design a concise outline tailored to the question. Choose sections that make sense "
     "(e.g., Overview, Prerequisites, Installation, Configuration, Usage, Validation, Troubleshooting, FAQ, Best Practices, Alternatives). "
     "Only include sections that add value. Keep titles short."),
    ("human",
     "Question:\n{question}\n\nFacts:\n{facts}\n\n"
     "Propose a section outline (1 level deep).")
])

DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Write a well-structured, practical answer in GitHub-flavored Markdown using ONLY the provided facts. "
     "Follow the given outline. Use numbered steps for procedures, bullet points for lists, and short code blocks for commands/config. "
     "Do not invent APIs, flags, or paths not present in facts. If something is missing, say it briefly and move on."),
    ("human",
     "Question:\n{question}\n\nOutline:\n{outline}\n\nFacts:\n{facts}\n\n"
     "Write the answer.")
])

CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Be a critical reviewer. Identify gaps, ambiguities, missing edge-cases, platform/version caveats, safety/pitfalls, and validation/rollback steps. "
     "Suggest concrete edits/additions. Be concise."),
    ("human",
     "Question:\n{question}\n\nDraft:\n{draft}\n\nProvide a bullet list of issues and suggested fixes.")
])

REVISE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Revise the draft by applying the given critique. Keep the structure clear and practical. "
     "Use only the provided facts; do not invent details."),
    ("human",
     "Question:\n{question}\n\nFacts:\n{facts}\n\nDraft:\n{draft}\n\nCritique:\n{critique}\n\n"
     "Produce the improved final answer.")
])

# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------
def init_chatbot(
    model_name: str = DEFAULT_MODEL,
    repo_dir: Optional[str | Path] = None,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    rebuild_index: bool = False,
) -> Tuple[ChatOllama, Any]:
    global _LLM, _VECTORSTORE, _MMR_RETRIEVER, _EMB, _BM25

    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("medium")
            log.info("GPU detected: %s", torch.cuda.get_device_name(0))
        except Exception:
            pass

    _LLM = _mk_llm(model_name)
    _VECTORSTORE = load_or_create_vectorstore(repo_dir=repo_dir, index_dir=index_dir, rebuild=rebuild_index)

    # Reuse vectorstore embedding function for rerank
    _EMB = getattr(_VECTORSTORE, "embedding_function", None)
    if _EMB is None:
        model_name = _parse_model_name(_read_model_txt(Path(index_dir)))
        _EMB = _mk_embeddings(model_name)

    # Dense retriever
    _MMR_RETRIEVER = _VECTORSTORE.as_retriever(
        search_type="mmr",
        search_kwargs={"k": DEFAULT_FETCH_K, "fetch_k": DEFAULT_FETCH_K, "lambda_mult": DEFAULT_LAMBDA},
    )

    # Sparse retriever
    try:
        _BM25 = _build_bm25_from_vs(_VECTORSTORE)
    except Exception as e:
        log.warning("BM25 build failed: %s", e)
        _BM25 = None

    log.info("✅ RAG initialized (index=%s, bm25=%s)", index_dir, "on" if _BM25 else "off")
    return _LLM, _MMR_RETRIEVER

def _ensure_ready(model_name: str = DEFAULT_MODEL, index_dir: str | Path = DEFAULT_INDEX_DIR):
    if _LLM is None or _MMR_RETRIEVER is None or _VECTORSTORE is None:
        init_chatbot(model_name=model_name, index_dir=index_dir)

# -----------------------------------------------------------------------------
# Retrieval pipeline
# -----------------------------------------------------------------------------
def _retrieve_docs(query: str) -> List[Document]:
    dense_docs: List[Document] = []
    sparse_docs: List[Document] = []

    try:
        dense_docs = _MMR_RETRIEVER.invoke(query)
    except Exception as e:
        log.warning("MMR retrieval failed: %s", e)
        try:
            sim = _VECTORSTORE.as_retriever(search_type="similarity", search_kwargs={"k": DEFAULT_FETCH_K})
            dense_docs = sim.invoke(query)
        except Exception as e2:
            log.warning("Similarity fallback failed: %s", e2)

    if _BM25 is not None:
        try:
            sparse_docs = _BM25.invoke(query)  # modern API (no deprecation)
        except Exception as e:
            log.warning("BM25 retrieval failed: %s", e)

    fused = _rrf_fuse(dense_docs, sparse_docs, k=RRF_K, cap=DEFAULT_FETCH_K) if (dense_docs or sparse_docs) else []
    fused = _unique_docs(fused)
    fused = _embed_rerank(query, fused, _EMB) if _EMB else fused
    if CROSS_ENCODER_MODEL:
        fused = _ce_rerank(query, fused, topn=CROSS_ENCODER_TOPN)

    expanded = _expand_by_source(fused[:max(DEFAULT_K, 12)], _VECTORSTORE, limit_per_source=3, total_cap=MAX_CONTEXT_DOCS)
    return expanded[:MAX_CONTEXT_DOCS]

# -----------------------------------------------------------------------------
# Synthesis helpers
# -----------------------------------------------------------------------------
def _map_notes(question: str, docs: List[Document]) -> str:
    if not docs:
        return ""
    blocks: List[str] = []
    for d in docs:
        mp = MAP_PROMPT.format(question=question, excerpt=d.page_content[:4000])
        out = _LLM.invoke(mp)
        txt = getattr(out, "content", str(out)).strip()
        if txt:
            blocks.append(txt)
    return "\n\n".join(blocks)

def _plan_sections(question: str, facts: str) -> str:
    out = _LLM.invoke(PLAN_PROMPT.format(question=question, facts=facts))
    return getattr(out, "content", str(out)).strip()

def _draft_answer(question: str, outline: str, facts: str) -> str:
    out = _LLM.invoke(DRAFT_PROMPT.format(question=question, outline=outline, facts=facts))
    return getattr(out, "content", str(out)).strip()

def _critique(question: str, draft: str) -> str:
    out = _LLM.invoke(CRITIQUE_PROMPT.format(question=question, draft=draft))
    return getattr(out, "content", str(out)).strip()

def _revise(question: str, facts: str, draft: str, critique: str) -> str:
    out = _LLM.invoke(REVISE_PROMPT.format(question=question, facts=facts, draft=draft, critique=critique))
    return getattr(out, "content", str(out)).strip()

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def chat_with_repo(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,  # kept for compatibility; not used in final prompts
    model_name: str = DEFAULT_MODEL,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    top_k: int = DEFAULT_K,  # controls final cap
) -> str:
    try:
        _ensure_ready(model_name=model_name, index_dir=index_dir)

        # 360° retrieval
        variants = _gen_query_variants(_LLM, question, n=6)
        all_docs: List[Document] = []
        for v in variants:
            all_docs.extend(_retrieve_docs(v))
        docs = _unique_docs(all_docs)[:max(top_k, DEFAULT_K, 10)]
        if not docs:
            return (
                "I couldn’t find enough relevant material in the repository index to answer this. "
                "Consider updating/rebuilding the index so READMEs, INSTALL/SETUP guides, and examples are included, "
                "or try a more specific query (exact package name, file, or error message)."
            )

        # plan → draft → critique → revise
        facts = _map_notes(question, docs)
        outline = _plan_sections(question, facts)
        draft = _draft_answer(question, outline, facts)
        critique = _critique(question, draft)
        final_answer = _revise(question, facts, draft, critique)

        return final_answer or draft

    except Exception as e:
        log.exception("chat_with_repo error")
        return (
            f"Sorry — I hit an internal error while answering: {e}\n\n"
            "Tips: verify the FAISS index, rebuild embeddings if stale, or widen retrieval via "
            "`RETRIEVE_FETCH_K` and `MAX_CONTEXT_DOCS`."
        )

async def stream_with_repo(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    model_name: str = DEFAULT_MODEL,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> AsyncGenerator[str, None]:
    """
    Streams only the final revision stage so the user gets a polished, ChatGPT-style answer.
    """
    try:
        _ensure_ready(model_name=model_name, index_dir=index_dir)

        variants = _gen_query_variants(_LLM, question, n=6)
        all_docs: List[Document] = []
        for v in variants:
            all_docs.extend(_retrieve_docs(v))
        docs = _unique_docs(all_docs)[:max(DEFAULT_K, 10)]
        if not docs:
            yield (
                "I couldn’t find enough relevant material in the repository index to answer this. "
                "Consider updating/rebuilding the index or try a more specific query."
            )
            return

        facts = _map_notes(question, docs)
        outline = _plan_sections(question, facts)
        draft = _draft_answer(question, outline, facts)
        critique = _critique(question, draft)

        # Stream the revised final
        rp = REVISE_PROMPT.format(question=question, facts=facts, draft=draft, critique=critique)
        async for chunk in _LLM.astream(rp):
            if isinstance(chunk, str) and chunk:
                yield chunk

    except Exception as e:
        yield f"Internal error: {e}"
