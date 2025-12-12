# Minimal FAISS-based “chat over folder” app for LLMBot (no Chroma, no TensorFlow)

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import glob, hashlib, sqlite3, uuid
import numpy as np
import gradio as gr
from pypdf import PdfReader
from markdownify import markdownify as md
import faiss
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from openai import OpenAI
import re
import threading
from pathlib import Path


DB_LOCK = threading.Lock()

# ---------- CONFIG ----------
FOLDER = r"./"  # change if needed
TOP_K = 10
CHUNK_CHARS = 1200
OVERLAP = 200
TEMPERATURE = 0.1
EMB_MODEL = "all-MiniLM-L6-v2"   # 384-dim

# Storage
DB_DIR = "./faiss_store"
os.makedirs(DB_DIR, exist_ok=True)
FAISS_PATH = os.path.join(DB_DIR, "index.faiss")
SQLITE_PATH = os.path.join(DB_DIR, "chunks.sqlite3")
EMB_DIM = 384

# LLMBot config
LLMBOT_BASE_URL = "https://llmbot.gsi.de/api/v1"
LLMBOT_MODEL = "llmbot.qwen3-vl-235b-thinking"
LLMBOT_API_KEY = os.getenv("LLMBOT_API_KEY", "sk-REDACTED")  # set env var

client = OpenAI(base_url=LLMBOT_BASE_URL, api_key=LLMBOT_API_KEY)

FILENAME_RE = re.compile(r"[A-Za-z0-9_\-./]+?\.(h|hpp|hh|hxx|c|cc|cpp|cxx|py|md|txt)$")

# ---------- EMBEDDINGS ----------
st = SentenceTransformer(EMB_MODEL)

def embed_texts(texts):
    vecs = st.encode(
        texts,
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine via inner product
    )
    return vecs.astype("float32")

# ---------- SQLITE HELPERS ----------
def sql_conn():
    con = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY,
            path TEXT,
            chunk INTEGER,
            sha1 TEXT,
            doc TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_path ON chunks(path)")
    return con

def sql_get_ids_by_path(con, path):
    with DB_LOCK:
        cur = con.execute("SELECT id FROM chunks WHERE path=?", (path,))
        return [r[0] for r in cur.fetchall()]

def sql_first_sha_by_path(con, path):
    with DB_LOCK:
        cur = con.execute("SELECT sha1 FROM chunks WHERE path=? LIMIT 1", (path,))
        row = cur.fetchone()
        return row[0] if row else None

def sql_insert_many(con, rows):
    with DB_LOCK:
        con.executemany("INSERT INTO chunks(id,path,chunk,sha1,doc) VALUES(?,?,?,?,?)", rows)
        con.commit()

def sql_delete_ids(con, ids):
    if not ids: return
    with DB_LOCK:
        con.executemany("DELETE FROM chunks WHERE id=?", [(int(i),) for i in ids])
        con.commit()

def sql_fetch_meta(con, ids):
    if not ids: return []
    q = "SELECT id,path,chunk,sha1,doc FROM chunks WHERE id IN (%s)" % ",".join("?"*len(ids))
    with DB_LOCK:
        return con.execute(q, [int(i) for i in ids]).fetchall()

def sql_next_id(con):
    with DB_LOCK:
        r = con.execute("SELECT MAX(id) FROM chunks").fetchone()[0]
        return 1 if r is None else int(r) + 1

con = sql_conn()

def guess_paths_from_query(q):
    hits = []
    for m in FILENAME_RE.finditer(q):
        hits.append(m.group(0))
    return hits

def sql_fetch_by_path_like(con, pattern, limit=TOP_K):
    q = """
    SELECT id, path, chunk, sha1, doc
    FROM chunks
    WHERE path LIKE ?
    ORDER BY chunk ASC
    LIMIT ?
    """
    with DB_LOCK:
        return con.execute(q, (f"%{pattern}%", int(limit))).fetchall()

# ---------- FAISS PERSISTENCE ----------
def faiss_new():
    return faiss.IndexIDMap2(faiss.IndexFlatIP(EMB_DIM))

def faiss_load():
    if os.path.exists(FAISS_PATH):
        return faiss.read_index(FAISS_PATH)
    return faiss_new()

def faiss_save(index):
    with DB_LOCK:
        faiss.write_index(index, FAISS_PATH)

faiss_index = faiss_load()

# ---------- FILE HANDLING ----------
def sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

TEXT_EXTS = {
    ".txt",".py",".md",".json",".yaml",".yml",".csv",
    ".c",".h",".hpp",".hh",".hxx",".cpp",".cc",".cxx",".ino",
    ".cmake",".ini",".cfg",".conf",".make",".mk","CMakeLists.txt"
}

SKIP_DIRS = {
    ".git","build","cmake-build-debug","cmake-build-release","out","dist","node_modules",
    ".cache",".venv","venv","__pycache__"
}

def should_skip(p):
    parts = os.path.normpath(p).split(os.sep)
    return any(s in SKIP_DIRS for s in parts)

def _read(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in TEXT_EXTS or os.path.basename(path) == "CMakeLists.txt":
            return open(path, "r", encoding="utf-8", errors="ignore").read()
        if ext == ".pdf":
            return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        if ext in {".html", ".htm"}:
            return md(open(path, "r", encoding="utf-8", errors="ignore").read())
    except Exception:
        return ""
    return ""

def _chunks(t, n=CHUNK_CHARS, overlap=OVERLAP):
    out = []
    i = 0
    L = len(t)
    step = max(1, n - overlap)
    while i < L:
        j = min(i + n, L)
        s = t[i:j]
        if s.strip():
            out.append(s)
        if j == L:
            break
        i += step
    return out

# ---------- INDEX OPS ----------
def remove_path(path):
    ids = sql_get_ids_by_path(con, path)
    if not ids: return
    with DB_LOCK:
        sel = faiss.IDSelectorBatch(np.asarray(ids, dtype="int64"))
        faiss_index.remove_ids(sel)
    sql_delete_ids(con, ids)

def add_file_to_index(path, file_sha, text):
    chunks = _chunks(text)
    if not chunks: return
    start_id = sql_next_id(con)
    ids = np.arange(start_id, start_id + len(chunks), dtype="int64")

    # prepend path so embeddings “see” the filename
    labeled = [f"[FILE:{path}] {c}" for c in chunks]
    vecs = embed_texts(labeled)

    with DB_LOCK:
        faiss_index.add_with_ids(vecs, ids)
    rows = [(int(ids[i]), path, i, file_sha, chunks[i]) for i in range(len(chunks))]
    sql_insert_many(con, rows)

def index_full():
    patterns = list({ "**/*"+e for e in TEXT_EXTS if e.startswith(".") })
    patterns += ["**/CMakeLists.txt", "**/*.pdf", "**/*.docx", "**/*.htm*"]
    paths = []
    for pat in patterns:
        paths += glob.glob(os.path.join(FOLDER, pat), recursive=True)

    for p in paths:
        if not os.path.isfile(p) or should_skip(p):
            continue
        file_sha = sha1(p)
        old_sha = sql_first_sha_by_path(con, p)
        if old_sha == file_sha:
            continue  # unchanged
        # replace old entries if any
        remove_path(p)
        text = _read(p)
        if text:
            add_file_to_index(p, file_sha, text)

    faiss_save(faiss_index)

# ---------- WATCHER ----------
class _Watcher(FileSystemEventHandler):
    def _reindex(self, p):
        if not os.path.isfile(p) or should_skip(p):
            return
        text = _read(p)
        if text == "":
            remove_path(p)
            faiss_save(faiss_index)
            return
        file_sha = sha1(p)
        old_sha = sql_first_sha_by_path(con, p)
        if old_sha == file_sha:
            return  # unchanged
        remove_path(p)
        add_file_to_index(p, file_sha, text)
        faiss_save(faiss_index)

    def on_created(self, e):
        if not e.is_directory:
            self._reindex(e.src_path)

    def on_modified(self, e):
        if not e.is_directory:
            self._reindex(e.src_path)

    def on_moved(self, e):
        if e.is_directory:
            return
        remove_path(e.src_path)
        self._reindex(e.dest_path)
        faiss_save(faiss_index)

    def on_deleted(self, e):
        if e.is_directory:
            return
        remove_path(e.src_path)
        faiss_save(faiss_index)

def start_watch():
    obs = Observer()
    obs.schedule(_Watcher(), FOLDER, recursive=True)
    obs.daemon = True
    obs.start()

# ---------- LLM CALL ----------
SYSTEM_PROMPT = (
    "You are a precise assistant. Use only the provided context when possible. "
    "Cite file paths inline. If the answer is not in the context, say so."
)

def ask_llm(context, question):
    resp = client.chat.completions.create(
        model="llmbot.qwen3-vl-235b-thinking",
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
    )
    return resp.choices[0].message.content.strip()

# ---------- QUERY ----------
def respond(user_msg, history):
    # 1) filename/path fallback
    candidate_paths = guess_paths_from_query(user_msg)
    rows = []
    for cp in candidate_paths:
        rows.extend(sql_fetch_by_path_like(con, os.path.basename(cp), limit=TOP_K))
        if rows:
            break  # take first hit set

    # 2) if no direct filename match, use vector search
    if not rows:
        if faiss_index.ntotal == 0:
            out = "Index is empty."
            history = history + [(user_msg, out)]
            return "", history, history
        qv = embed_texts([user_msg])
        D, I = faiss_index.search(qv, TOP_K)
        ids = [int(i) for i in I[0] if i != -1]
        rows = sql_fetch_meta(con, ids)  # (id,path,chunk,sha1,doc)
        # keep vector order
        order = {ids[i]: i for i in range(len(ids))}
        rows.sort(key=lambda r: order.get(r[0], 1e9))
    else:
        # rows already sorted by chunk; optionally expand to next chunks
        pass

    context = "\n\n---\n\n".join(
        f"[{j+1}] {r[4]}\n(Source: {r[1]}, chunk {r[2]})" for j, r in enumerate(rows)
    ) or "(no matching context found)"

    try:
        out = ask_llm(context, user_msg)
    except Exception as e:
        out = f"LLM error: {e}"
    history = history + [(user_msg, out)]
    return "", history, history

# ---------- INIT + UI ----------
os.makedirs(FOLDER, exist_ok=True)
index_full()
start_watch()

with gr.Blocks(title="Folder Chat • LLMBot") as demo:
    gr.Markdown("### Chat over local folder")
    chat = gr.Chatbot(height=480, type="tuples")
    msg = gr.Textbox(placeholder="Ask about files in the folder…")
    send = gr.Button("Send")
    clear = gr.Button("Clear")
    state = gr.State([])

    send.click(respond, [msg, state], [msg, chat, state])
    msg.submit(respond, [msg, state], [msg, chat, state])
    clear.click(lambda: ([], [], []), None, [chat, state, msg])

demo.launch(server_name="127.0.0.1", server_port=7860)
