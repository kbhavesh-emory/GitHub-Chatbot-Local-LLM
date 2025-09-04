# backend/ingest/store_embeddings.py
"""
Deep, GPU-optimized embedding builder for FAISS with enhanced quality settings.

Optimized for retrieval quality with:
- Intelligent chunking strategy for code and documentation
- Enhanced file type support with better text extraction
- Quality-focused parameters for optimal RAG performance
- Comprehensive metadata and manifest tracking
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import shutil
import warnings
import mimetypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Sequence

# ---- Runtime tuning (CUDA + tokenizers) -------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Optional: Unstructured for PDFs/Office/images
try:
    from unstructured.partition.auto import partition
    from unstructured.cleaners.core import clean_extra_whitespace
    UNSTRUCTURED_OK = True
except Exception:
    UNSTRUCTURED_OK = False

# ----------------------------- Defaults --------------------------------------
DEFAULT_VECTOR_DIR = "backend/vectorstore"
DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
DEFAULT_CHUNK_SIZE = 1024  # Optimal for code and documentation
DEFAULT_CHUNK_OVERLAP = 200  # Better context preservation
DEFAULT_BATCH_SIZE = 64     # Balanced GPU utilization
DEFAULT_CHUNKS_PER_BATCH = 12000  # Memory-optimized
DEFAULT_LARGE_FILE_MB = 15
DEFAULT_DEVICE = "auto"
DEFAULT_WORKERS = max(4, (os.cpu_count() or 8) - 2)  # Leave cores for other tasks
DEFAULT_LANGS = ["en"]  # Default to English

# Enhanced file type support
ALLOWED_EXTENSIONS = {
    # Code files
    ".py", ".ipynb", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", 
    ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".swift",
    ".php", ".rb", ".pl", ".lua", ".r", ".m", ".matlab", ".scala", ".groovy",
    
    # Config files
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".xml", 
    ".html", ".htm", ".css", ".scss", ".less", ".svg",
    
    # Documentation
    ".md", ".markdown", ".rst", ".txt", ".rtf", ".log", ".tex",
    
    # Data files
    ".csv", ".tsv",
    
    # Build files
    ".dockerfile", ".env", ".gitignore", ".gitattributes", ".make", ".mk",
    ".cmake", ".gradle", ".pom", ".bazel", ".buck",
    
    # Office documents (if unstructured available)
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".ods",
    
    # Images with text (if unstructured available)
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
}

# Language-specific splitters (only supported languages)
LANGUAGE_SPLITTERS = {
    ".py": "python",
    ".js": "js",
    ".jsx": "js",
    ".ts": "js",
    ".tsx": "js",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".go": "go",
    ".php": "php",
    ".rb": "ruby",
    ".rs": "rust",
    ".swift": "swift",
    ".md": "markdown",
    ".html": "html",
}

BUILD_SPEC = {
    "device": DEFAULT_DEVICE,
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "batch_size": DEFAULT_BATCH_SIZE,
    "chunks_per_batch": DEFAULT_CHUNKS_PER_BATCH,
    "model": DEFAULT_MODEL,
    "backend": "huggingface",
    "optimized_for": "retrieval_quality",
    "max_file_size_mb": DEFAULT_LARGE_FILE_MB,
}

@dataclass
class FileResult:
    path: Path
    ok: bool
    reason: str = ""
    size_mb: float = 0.0
    chunks: int = 0

# ------------------------------ Device utils ---------------------------------
def cuda_works() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        # Test CUDA with a simple operation
        _ = (torch.ones(1, device="cuda") + 1).item()
        return True
    except Exception as e:
        print(f"⚠️ CUDA test failed: {e}")
        return False

def decide_device(requested: str) -> str:
    rq = (requested or "").lower()
    if rq == "cpu":
        return "cpu"
    if rq == "cuda":
        return "cuda" if cuda_works() else "cpu"
    # auto
    return "cuda" if cuda_works() else "cpu"

def optimize_cuda():
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("medium")
            print(f"🟢 CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"🔧 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        except Exception as e:
            print(f"⚠️ CUDA tuning skipped: {e}")

# ----------------------------- FS helpers ------------------------------------
def is_supported(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in ALLOWED_EXTENSIONS:
        return True
    
    # Additional MIME type checking
    mime = mimetypes.guess_type(str(path))[0] or ""
    if mime.startswith("text/") or mime.startswith("application/"):
        return True
        
    return False

def file_size_mb(path: Path) -> float:
    try:
        return round(path.stat().st_size / (1024 * 1024), 2)
    except Exception:
        return 0.0

def collect_files(roots: Sequence[str]) -> Tuple[List[Path], List[Path]]:
    roots = [str(Path(r).resolve()) for r in roots]
    print("📂 Scanning directories:\n  " + "\n  ".join(roots))
    
    files: List[Path] = []
    skipped: List[Path] = []
    
    for root in roots:
        rpath = Path(root)
        if not rpath.exists():
            print(f"⚠️  Skipping missing dir: {rpath}")
            continue
            
        for p in tqdm(rpath.rglob("*"), desc=f"Scanning {rpath.name}", leave=False):
            if not p.is_file():
                continue
                
            # Skip hidden files and directories
            if any(part.startswith('.') for part in p.parts):
                continue
                
            if not is_supported(p):
                skipped.append(p)
                continue
                
            files.append(p)
                
    print(f"✅ Candidate files: {len(files)} | Skipped (type): {len(skipped)}")
    return files, skipped

# ----------------------------- Loading & splitting ---------------------------
def load_with_unstructured(path: Path, languages: Optional[List[str]]) -> str:
    """Enhanced text extraction with unstructured"""
    try:
        kwargs = {"languages": languages or ["en"]}
        
        # Special handling for different file types
        if path.suffix.lower() in [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"]:
            kwargs["strategy"] = "hi_res"
            
        elements = partition(filename=str(path), **kwargs)
        text = "\n\n".join([clean_extra_whitespace(str(el).strip()) for el in elements if str(el).strip()])
        return text
        
    except Exception as e:
        raise RuntimeError(f"Unstructured processing failed: {e}")

def robust_read_text(path: Path, error_notes: List[str], languages: Optional[List[str]]) -> str:
    ext = path.suffix.lower()
    size_mb = file_size_mb(path)
    
    # Handle large files with care
    if size_mb > DEFAULT_LARGE_FILE_MB:
        error_notes.append(f"large_file:{size_mb}MB")
    
    # Use unstructured for complex file types
    if ext in {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", 
               ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        if UNSTRUCTURED_OK:
            try:
                return load_with_unstructured(path, languages)
            except Exception as e:
                error_notes.append(f"unstructured_failed:{str(e)[:50]}")
        else:
            error_notes.append("unstructured_not_installed")
            return f"[Binary file: {path.name}]"

    # Use appropriate loader for text files
    try:
        if ext in LANGUAGE_SPLITTERS:
            # Use language-aware loading
            loader = TextLoader(str(path), autodetect_encoding=True)
            docs = loader.load()
            return "\n".join(d.page_content for d in docs)
        else:
            # Fallback to binary read with encoding detection
            return path.read_text(encoding='utf-8', errors='ignore')
            
    except Exception as e:
        try:
            # Final fallback
            return path.read_bytes().decode("utf-8", errors="ignore")
        except Exception as e2:
            raise RuntimeError(f"text loader failed: {e}; bytes decode failed: {e2}")

def get_text_splitter_for_file(path: Path, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Get appropriate text splitter for file type"""
    ext = path.suffix.lower()
    
    if ext in LANGUAGE_SPLITTERS:
        # Use language-specific splitter
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # For supported languages, use appropriate separators
        language = LANGUAGE_SPLITTERS[ext]
        
        if language in ["python", "java", "cpp", "c", "go", "php", "ruby", "rust", "swift"]:
            # Programming languages
            separators = [
                "\n\n\n", "\n\n", "\n", 
                "def ", "class ", "function ", "func ", "public ", "private ", "protected ",
                "//", "/*", "*/", "#", 
                ".", "!", "?", ";", ":",
                " ", "",
            ]
        elif language == "markdown":
            # Markdown files
            separators = [
                "\n\n\n", "\n\n", "\n",
                "# ", "## ", "### ", "#### ", "##### ", "###### ",
                "```", "***", "---", "___",
                ".", "!", "?", ";", ":",
                " ", "",
            ]
        elif language == "html":
            # HTML files
            separators = [
                "\n\n\n", "\n\n", "\n",
                "<div", "<p", "<h1", "<h2", "<h3", "<h4", "<h5", "<h6",
                "<script", "<style", "<header", "<footer", "<nav", "<main",
                ">", 
                ".", "!", "?", ";", ":",
                " ", "",
            ]
        elif language == "js":
            # JavaScript/TypeScript
            separators = [
                "\n\n\n", "\n\n", "\n",
                "function ", "const ", "let ", "var ", "export ", "import ",
                "//", "/*", "*/", 
                "{", "}", "(", ")",
                ".", "!", "?", ";", ":",
                " ", "",
            ]
        else:
            # Generic text splitter
            separators = [
                "\n\n\n", "\n\n", "\n", " ",
                "```", 
                "##", "#", 
                "def ", "class ", "function ",
                ".", "!", "?", ";", ":",
                "</", ">", 
                "--", 
                "//", "/*", "*/", 
                "#", 
            ]
        
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=True,
        )
    else:
        # Use generic splitter with code-aware separators
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n", "\n\n", "\n", " ",
                "```",  # Code blocks
                "##", "#",  # Headers
                "def ", "class ", "function ",  # Code structures
                ".", "!", "?", ";", ":",  # Sentence endings
                "</", ">",  # HTML tags
                "--",  # SQL comments
                "//", "/*", "*/",  # C-style comments
                "#",  # Python/bash comments
            ],
            keep_separator=True,
        )

def process_one_file(
    p: Path,
    chunk_size: int,
    chunk_overlap: int,
    languages: Optional[List[str]],
) -> Tuple[List[Document], FileResult]:
    size_mb = file_size_mb(p)
    err_notes: List[str] = []
    
    try:
        # Read file content
        txt = robust_read_text(p, err_notes, languages)
        if not txt.strip():
            return [], FileResult(p, ok=False, reason="empty_text", size_mb=size_mb)
        
        # Get appropriate splitter
        splitter = get_text_splitter_for_file(p, chunk_size, chunk_overlap)
        
        # Split text
        docs = splitter.create_documents([txt], metadatas=[{"source": str(p)}])
        
        # Enhance metadata with file info
        for doc in docs:
            doc.metadata.update({
                "file_name": p.name,
                "file_size_mb": size_mb,
                "file_extension": p.suffix.lower(),
                "file_path": str(p),
                "chunk_hash": hash(doc.page_content[:100])  # Simple content hash
            })
        
        reason = "ok"
        if size_mb > DEFAULT_LARGE_FILE_MB:
            reason = f"ok (large:{size_mb}MB)"
        if err_notes:
            reason += f" | notes:{' | '.join(err_notes[:2])}"
            
        return docs, FileResult(p, ok=True, reason=reason, size_mb=size_mb, chunks=len(docs))
        
    except Exception as e:
        note = "; ".join(err_notes) if err_notes else ""
        msg = f"{e} | {note}" if note else f"{e}"
        return [], FileResult(p, ok=False, reason=msg, size_mb=size_mb)

# ----------------------------- Embeddings ------------------------------------
def build_embeddings(model_name: str, batch_size: int, device: str) -> HuggingFaceEmbeddings:
    model_kwargs = {"device": ("cuda" if device == "cuda" else "cpu")}
    encode_kwargs = {
        "normalize_embeddings": True, 
        "batch_size": batch_size,
    }
    
    if device == "cuda":
        try:
            # Try mixed precision for GPU
            emb = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={**model_kwargs, "torch_dtype": torch.float16},
                encode_kwargs=encode_kwargs,
            )
            print("🔧 Using mixed precision (FP16) for embeddings")
        except Exception:
            # Fallback to full precision
            emb = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
    else:
        emb = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    # Test the embedding model
    try:
        test_embedding = emb.embed_query("test")
        print(f"✅ Embedding model loaded: {model_name}")
        print(f"🔧 Embedding dimension: {len(test_embedding)}")
    except Exception as e:
        print(f"⚠️ Embedding model test failed: {e}")
        
    return emb

def batched(items: List, batch_size: int) -> Iterable[List]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

# ----------------------------- Main pipeline ---------------------------------
def embed_repo(
    input_dirs: Sequence[str],
    out_dir: str = DEFAULT_VECTOR_DIR,
    model_name: str = DEFAULT_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    chunks_per_batch: int = DEFAULT_CHUNKS_PER_BATCH,
    device_choice: str = DEFAULT_DEVICE,
    workers: int = DEFAULT_WORKERS,
    languages: Optional[List[str]] = DEFAULT_LANGS,
    rembed: bool = False,
):
    t0 = time.time()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    error_log_path = out_path / "embed_errors.log"

    # Clear existing vectorstore if requested
    if rembed and out_path.exists():
        print(f"🧨 --rembed: removing existing vectorstore at {out_path}")
        shutil.rmtree(out_path, ignore_errors=True)
        out_path.mkdir(parents=True, exist_ok=True)

    # Collect files
    files, skipped_by_type = collect_files(input_dirs)
    
    if not files:
        print("❌ No files found to process")
        sys.exit(1)

    results: List[FileResult] = []
    all_chunks: List[Document] = []
    total_chunks = 0

    # Device optimization
    optimize_cuda()
    resolved_device = decide_device(device_choice)
    print(f"🔧 Using device: {resolved_device.upper()}")

    # Parallel reading + splitting
    print(f"🧩 Reading & splitting with {workers} workers (chunk_size={chunk_size}, overlap={chunk_overlap})…")
    
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [
            ex.submit(process_one_file, p, chunk_size, chunk_overlap, languages)
            for p in files
        ]
        
        for f in tqdm(as_completed(futs), total=len(futs), unit="file", desc="Processing files"):
            docs, res = f.result()
            if res.ok:
                all_chunks.extend(docs)
                total_chunks += res.chunks
            results.append(res)

    if not all_chunks:
        print("❌ No content to embed (no chunks produced).")
        error_log_path.write_text("No chunks were produced. Check file types and permissions.\n", encoding="utf-8")
        sys.exit(1)

    # Build embeddings
    emb = build_embeddings(model_name, batch_size, resolved_device)

    # Warmup embedding model
    try:
        print("🔥 Warming up embedding model...")
        _ = emb.embed_query("warmup query for initialization")
        if resolved_device == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠️ Warmup failed: {e}")

    print(f"🧠 Embedding model: {model_name} | device: {resolved_device.upper()} | encode batch={batch_size}")
    print(f"🧱 Total chunks: {len(all_chunks):,} (from {sum(1 for r in results if r.ok)} successful files)")

    # Build FAISS index in batches
    db: Optional[FAISS] = None
    num_batches = math.ceil(len(all_chunks) / float(chunks_per_batch))
    print(f"🚚 Building FAISS in {num_batches} batch(es) of ≤ {chunks_per_batch:,} chunks …")

    for batch_num, chunk_batch in enumerate(tqdm(list(batched(all_chunks, chunks_per_batch)), 
                                                desc="Embedding & indexing", unit="batch")):
        if db is None:
            db = FAISS.from_documents(chunk_batch, emb)
        else:
            db.add_documents(chunk_batch)
        
        # Clear memory
        del chunk_batch
        if resolved_device == "cuda":
            torch.cuda.empty_cache()
            
        print(f"📦 Processed batch {batch_num + 1}/{num_batches}")

    assert db is not None, "FAISS index not created"
    
    # Save the index
    db.save_local(str(out_path))
    print(f"💾 Saved FAISS index to {out_path}")

    # Save model info
    try:
        (out_path / "model.txt").write_text(f"{model_name}|backend=huggingface|device={resolved_device}", encoding="utf-8")
    except Exception as e:
        print(f"⚠️ Could not write model.txt: {e}")

    # Generate comprehensive report
    ok_files = [r for r in results if r.ok]
    bad_files = [r for r in results if not r.ok]
    elapsed = round(time.time() - t0, 2)

    # File type statistics
    file_types = {}
    for r in ok_files:
        ext = r.path.suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1

    manifest = {
        "input_dirs": [str(Path(d).resolve()) for d in input_dirs],
        "vector_dir": str(out_path.resolve()),
        "model": model_name,
        "backend": "huggingface",
        "device": resolved_device,
        "files_total": len(files),
        "files_ok": len(ok_files),
        "files_failed": len(bad_files),
        "chunks_total": len(all_chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "encode_batch_size": batch_size,
        "chunks_per_batch": chunks_per_batch,
        "large_file_mb_threshold": DEFAULT_LARGE_FILE_MB,
        "unstructured_used": UNSTRUCTURED_OK,
        "file_types": file_types,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "chunks_per_second": round(len(all_chunks) / elapsed, 2) if elapsed > 0 else 0,
        "build_spec": BUILD_SPEC,
        "skipped_by_type": len(skipped_by_type),
    }
    
    try:
        (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"⚠️ Could not write manifest.json: {e}")

    # Error log
    lines: List[str] = []
    if not UNSTRUCTURED_OK:
        lines.append("NOTE: 'unstructured' not installed. Binary docs used text fallback where possible.\n")
    
    lines.append("FAILED FILES:\n")
    for r in bad_files:
        lines.append(f"[FAIL] {r.path} ({r.size_mb}MB) :: {r.reason}")
    
    lines.append("\nLARGE OR NOTE-WORTHY FILES:\n")
    for r in ok_files:
        if "large:" in r.reason or "notes:" in r.reason:
            lines.append(f"[INFO] {r.path} :: {r.reason}")

    if lines:
        error_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Summary
    print("\n" + "="*60)
    print("🎉 EMBEDDING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"✅ Vector DB saved to: {out_path}")
    print(f"📝 Manifest:         {out_path / 'manifest.json'}")
    print(f"🧾 Model info:       {out_path / 'model.txt'}")
    
    if lines:
        print(f"⚠️  Error log:        {error_log_path}")
    
    print(f"\n📊 Statistics:")
    print(f"   Files: {len(files):,} total | {len(ok_files):,} ok | {len(bad_files):,} failed")
    print(f"   Chunks: {len(all_chunks):,} (avg: {len(all_chunks)/len(ok_files):.1f} chunks/file)")
    print(f"   Skipped: {len(skipped_by_type):,} unsupported files")
    
    if file_types:
        print(f"   File types: {', '.join(f'{k}:{v}' for k, v in sorted(file_types.items())[:10])}")
        if len(file_types) > 10:
            print(f"              ... and {len(file_types) - 10} more types")
    
    print(f"\n⏱️  Performance:")
    print(f"   Time: {elapsed}s ({elapsed/60:.1f} min)")
    print(f"   Speed: {len(all_chunks)/elapsed:.1f} chunks/s")
    
    print(f"\n💡 Recommendations for RAG usage:")
    print(f"   - top_k: 8-12")
    print(f"   - depth_queries: 3-5") 
    print(f"   - score_threshold: 0.65-0.75")
    print("="*60)

# ----------------------------- CLI -------------------------------------------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(
        description="Quality-optimized embedding builder for FAISS vector stores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python store_embeddings.py backend/data
  
  # Multiple directories with custom parameters
  python store_embeddings.py backend/data backend/docs \\
    --out backend/vectorstore \\
    --model BAAI/bge-large-en-v1.5 \\
    --chunk-size 1024 --overlap 200 \\
    --batch-size 64 --device cuda
  
  # Rebuild existing vectorstore
  python store_embeddings.py backend/data --rembed
        """
    )
    
    ap.add_argument(
        "input_dirs",
        nargs="*",
        default=["backend/data"],
        help="One or more folders to index (e.g., backend/data backend/docs)",
    )
    ap.add_argument("--out", dest="out_dir", default=DEFAULT_VECTOR_DIR, 
                   help="Output vectorstore directory (default: %(default)s)")
    ap.add_argument("--model", dest="model_name", default=DEFAULT_MODEL, 
                   help="HF embedding model name (default: %(default)s)")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, 
                   help="Characters per chunk (default: %(default)d)")
    ap.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, 
                   help="Characters overlapped between chunks (default: %(default)d)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, 
                   help="Embedder micro-batch size (default: %(default)d)")
    ap.add_argument("--chunks-per-batch", type=int, default=DEFAULT_CHUNKS_PER_BATCH, 
                   help="Chunks per FAISS batch (default: %(default)d)")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default=DEFAULT_DEVICE, 
                   help="Device selection (default: %(default)s)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, 
                   help="Reader/splitter workers (default: %(default)d)")
    ap.add_argument("--lang", type=str, default="en", 
                   help="Comma-separated ISO639-1 codes (e.g., 'en' or 'en,fr')")
    ap.add_argument("--rembed", action="store_true", 
                   help="Delete and rebuild the vectorstore")
    ap.add_argument("--list-supported", action="store_true",
                   help="List supported file extensions and exit")
    
    return ap.parse_args()

def list_supported_extensions():
    """Display all supported file extensions"""
    print("📋 Supported File Extensions:")
    extensions = sorted(ALLOWED_EXTENSIONS)
    for i in range(0, len(extensions), 8):
        print("   " + "   ".join(extensions[i:i+8]))
    print(f"\nTotal: {len(extensions)} supported extensions")
    print("\nNote: Unstructured library extends support to Office documents and images")
    if not UNSTRUCTURED_OK:
        print("⚠️  Unstructured not installed - limited binary file support")

if __name__ == "__main__":
    args = parse_args()
    
    if args.list_supported:
        list_supported_extensions()
        sys.exit(0)
        
    langs = [s.strip() for s in args.lang.split(",")] if args.lang else DEFAULT_LANGS
    
    try:
        embed_repo(
            input_dirs=args.input_dirs,
            out_dir=args.out_dir,
            model_name=args.model_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            batch_size=args.batch_size,
            chunks_per_batch=args.chunks_per_batch,
            device_choice=args.device,
            workers=args.workers,
            languages=langs,
            rembed=args.rembed,
        )
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)