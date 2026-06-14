"""
Campus Handbook Chatbot (Offline RAG) - GUI Edition
=====================================================

A fully offline Retrieval-Augmented Generation (RAG) chatbot with a
desktop GUI built using Tkinter (Python's built-in GUI toolkit - no
extra GUI libraries required).

Tech stack (all local, no external APIs):
- Tkinter                -> graphical user interface (built into Python)
- PyMuPDF (fitz)         -> extract text from the handbook PDF
- Sentence Transformers  -> create embeddings (all-MiniLM-L6-v2)
- FAISS                  -> vector similarity search
- NumPy                  -> store/manage embedding arrays
- Ollama (llama3.2:1b)   -> local LLM for answer generation

SECURITY NOTES (see README for full details):
- Chunk data is stored as JSON-safe pickle (plain dicts of str/int only)
  to avoid arbitrary-code-execution risks from untrusted pickle files.
- Only PDF files selected via the OS file dialog can be loaded.
- All processing happens on the local machine; nothing is sent over
  the network.
"""

import os
import sys
import pickle
import threading
import queue

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import fitz                     # PyMuPDF - for reading PDF files
import numpy as np              # for storing/handling embeddings
import faiss                    # for fast vector similarity search


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DEFAULT_PDF_PATH = "handbook.pdf"
INDEX_PATH = "handbook.index"
CHUNKS_PATH = "chunks.pkl"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "llama3.2:1b"

CHUNK_WORD_SIZE = 300
TOP_K = 3


# ---------------------------------------------------------------------------
# CORE RAG LOGIC (same approach as the CLI version, reused by the GUI)
# ---------------------------------------------------------------------------

def extract_text_by_page(pdf_path):
    """Open a PDF and return a list of (page_number, page_text) tuples."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Could not find PDF file: '{pdf_path}'")

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Selected file is not a .pdf file.")

    doc = fitz.open(pdf_path)
    pages_text = []
    for page_number, page in enumerate(doc, start=1):
        pages_text.append((page_number, page.get_text()))
    doc.close()
    return pages_text


def chunk_text(pages_text, chunk_size=CHUNK_WORD_SIZE):
    """Split each page's text into ~chunk_size word chunks, keeping page numbers."""
    chunks = []
    for page_number, text in pages_text:
        words = text.split()
        if not words:
            continue
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunks.append({"text": " ".join(chunk_words), "page": page_number})
    return chunks


def create_embeddings(chunks, embedding_model):
    """Create float32 embeddings for a list of chunk dicts."""
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings):
    """Build a FAISS L2 flat index from an embeddings array."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index_and_chunks(index, chunks):
    """Persist the FAISS index and chunk metadata to disk."""
    faiss.write_index(index, INDEX_PATH)
    # NOTE: chunks only ever contain plain str/int values (text + page number),
    # so this pickle file cannot be used to smuggle executable objects as long
    # as it is only ever written by this program.
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_index_and_chunks():
    """Load a previously saved FAISS index and chunk metadata."""
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def index_and_chunks_exist():
    return os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)


def retrieve_relevant_chunks(question, embedding_model, index, chunks, top_k=TOP_K):
    """Embed the question and return the top_k most similar chunks."""
    q_emb = embedding_model.encode([question], convert_to_numpy=True)
    q_emb = np.array(q_emb, dtype="float32")
    distances, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0] if i != -1]


def build_prompt(question, retrieved_chunks):
    """Build the RAG prompt containing context blocks with page numbers."""
    context_blocks = [f"[Page {c['page']}]\n{c['text']}" for c in retrieved_chunks]
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant that answers questions using ONLY the
context from a college handbook provided below. Each context block is
labeled with the page number it came from.

Instructions:
- Answer the question using ONLY the information in the context below.
- If the context does not contain the answer, respond exactly with:
  "I could not find this information in the handbook."
- When you do answer, mention the relevant page number(s) in your answer,
  like "(Page X)".

Context:
{context_text}

Question: {question}

Answer:"""
    return prompt


def ask_llm(prompt):
    """Send the prompt to the local Ollama model and return the answer text."""
    import ollama  # imported here so the GUI can still start if ollama isn't installed
    response = ollama.chat(
        model=OLLAMA_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# GUI APPLICATION
# ---------------------------------------------------------------------------
class HandbookChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Campus Handbook Chatbot (Offline RAG)")
        self.root.geometry("760x640")
        self.root.minsize(620, 480)

        # Runtime state
        self.pdf_path = DEFAULT_PDF_PATH
        self.embedding_model = None
        self.index = None
        self.chunks = None
        self.ready = False

        # Thread-safe queue used by background threads to send messages
        # back to the main (GUI) thread.
        self.msg_queue = queue.Queue()

        self._build_ui()
        self.root.after(100, self._poll_queue)

        # Start loading the embedding model + index/chunks in the background
        self._set_status("Loading embedding model...")
        threading.Thread(target=self._setup_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # UI LAYOUT
    # ------------------------------------------------------------------
    def _build_ui(self):
        # --- Top bar: PDF selection ---
        top_frame = ttk.Frame(self.root, padding=8)
        top_frame.pack(fill="x")

        ttk.Label(top_frame, text="Handbook PDF:").pack(side="left")

        self.pdf_label_var = tk.StringVar(value=self.pdf_path)
        ttk.Label(top_frame, textvariable=self.pdf_label_var,
                  foreground="#555").pack(side="left", padx=6)

        ttk.Button(top_frame, text="Choose PDF...",
                   command=self._choose_pdf).pack(side="right")
        ttk.Button(top_frame, text="Rebuild Index",
                   command=self._rebuild_index).pack(side="right", padx=6)

        # --- Chat display area ---
        chat_frame = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        chat_frame.pack(fill="both", expand=True)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap="word", state="disabled",
            font=("Segoe UI", 10)
        )
        self.chat_display.pack(fill="both", expand=True)

        # Tag styles for different message types
        self.chat_display.tag_config("user", foreground="#0b5394", font=("Segoe UI", 10, "bold"))
        self.chat_display.tag_config("bot", foreground="#000000")
        self.chat_display.tag_config("citation", foreground="#888888", font=("Segoe UI", 9, "italic"))
        self.chat_display.tag_config("system", foreground="#b45f06", font=("Segoe UI", 9, "italic"))
        self.chat_display.tag_config("error", foreground="#cc0000")

        # --- Input area ---
        input_frame = ttk.Frame(self.root, padding=8)
        input_frame.pack(fill="x")

        self.question_var = tk.StringVar()
        self.question_entry = ttk.Entry(input_frame, textvariable=self.question_var, font=("Segoe UI", 10))
        self.question_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.question_entry.bind("<Return>", lambda e: self._on_ask())

        self.ask_button = ttk.Button(input_frame, text="Ask", command=self._on_ask, state="disabled")
        self.ask_button.pack(side="right")

        # --- Status bar ---
        self.status_var = tk.StringVar(value="Starting up...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief="sunken", anchor="w", padding=4)
        status_bar.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # HELPERS FOR UPDATING THE UI
    # ------------------------------------------------------------------
    def _append_chat(self, text, tag=None):
        self.chat_display.configure(state="normal")
        if tag:
            self.chat_display.insert("end", text + "\n", tag)
        else:
            self.chat_display.insert("end", text + "\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def _set_status(self, text):
        self.status_var.set(text)

    def _poll_queue(self):
        """Process messages sent from background threads."""
        try:
            while True:
                msg_type, payload = self.msg_queue.get_nowait()
                if msg_type == "status":
                    self._set_status(payload)
                elif msg_type == "chat":
                    text, tag = payload
                    self._append_chat(text, tag)
                elif msg_type == "ready":
                    self.ready = True
                    self.ask_button.configure(state="normal")
                    self._set_status("Chatbot ready! Ask a question below.")
                elif msg_type == "error":
                    messagebox.showerror("Error", payload)
                    self._set_status("Error - see dialog for details.")
                elif msg_type == "not_ready":
                    self.ready = False
                    self.ask_button.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # PDF SELECTION / INDEX BUILDING (background work)
    # ------------------------------------------------------------------
    def _choose_pdf(self):
        path = filedialog.askopenfilename(
            title="Select Handbook PDF",
            filetypes=[("PDF files", "*.pdf")]
        )
        if not path:
            return

        self.pdf_path = path
        self.pdf_label_var.set(os.path.basename(path))

        # A new PDF means we must rebuild the index (delete old cache)
        self._rebuild_index()

    def _rebuild_index(self):
        if not os.path.exists(self.pdf_path):
            messagebox.showerror("Error", f"PDF not found:\n{self.pdf_path}")
            return

        # Remove any cached index/chunks so they get rebuilt
        for path in (INDEX_PATH, CHUNKS_PATH):
            if os.path.exists(path):
                os.remove(path)

        self.msg_queue.put(("not_ready", None))
        self._set_status("Rebuilding index from PDF...")
        threading.Thread(target=self._build_index_worker, daemon=True).start()

    def _setup_worker(self):
        """Background thread: load embedding model, then load/build index."""
        try:
            from sentence_transformers import SentenceTransformer

            self.msg_queue.put(("status", f"Loading embedding model '{EMBEDDING_MODEL_NAME}'..."))
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

            if index_and_chunks_exist():
                self.msg_queue.put(("status", "Loading existing index..."))
                self.index, self.chunks = load_index_and_chunks()
                self.msg_queue.put(("chat", (
                    f"[System] Loaded existing index ({self.index.ntotal} chunks).", "system")))
            else:
                self._build_index_from_pdf()

            self.msg_queue.put(("chat", ("[System] Chatbot ready!", "system")))
            self.msg_queue.put(("ready", None))

        except FileNotFoundError as e:
            self.msg_queue.put(("error", str(e)))
        except Exception as e:
            self.msg_queue.put(("error", f"Setup failed: {e}"))

    def _build_index_worker(self):
        """Background thread used when rebuilding the index for a new PDF."""
        try:
            if self.embedding_model is None:
                from sentence_transformers import SentenceTransformer
                self.msg_queue.put(("status", f"Loading embedding model '{EMBEDDING_MODEL_NAME}'..."))
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

            self._build_index_from_pdf()
            self.msg_queue.put(("chat", ("[System] Index rebuilt. Chatbot ready!", "system")))
            self.msg_queue.put(("ready", None))
        except FileNotFoundError as e:
            self.msg_queue.put(("error", str(e)))
        except Exception as e:
            self.msg_queue.put(("error", f"Failed to rebuild index: {e}"))

    def _build_index_from_pdf(self):
        """Shared logic: extract -> chunk -> embed -> index -> save."""
        self.msg_queue.put(("status", "Loading PDF..."))
        pages_text = extract_text_by_page(self.pdf_path)

        self.msg_queue.put(("status", "Splitting text into chunks..."))
        self.chunks = chunk_text(pages_text)

        if not self.chunks:
            raise ValueError("No extractable text found in this PDF "
                              "(it may be scanned images without OCR text).")

        self.msg_queue.put(("status", "Creating embeddings... (this can take a moment)"))
        embeddings = create_embeddings(self.chunks, self.embedding_model)

        self.msg_queue.put(("status", "Building FAISS index..."))
        self.index = build_faiss_index(embeddings)

        self.msg_queue.put(("status", "Saving index to disk..."))
        save_index_and_chunks(self.index, self.chunks)

    # ------------------------------------------------------------------
    # ASKING QUESTIONS
    # ------------------------------------------------------------------
    def _on_ask(self):
        if not self.ready:
            return

        question = self.question_var.get().strip()
        if not question:
            return

        self.question_var.set("")
        self._append_chat(f"You: {question}", "user")
        self.ask_button.configure(state="disabled")
        self._set_status("Thinking...")

        threading.Thread(target=self._answer_worker, args=(question,), daemon=True).start()

    def _answer_worker(self, question):
        try:
            retrieved = retrieve_relevant_chunks(
                question, self.embedding_model, self.index, self.chunks, top_k=TOP_K
            )
            prompt = build_prompt(question, retrieved)

            try:
                answer = ask_llm(prompt)
            except Exception as e:
                answer = (
                    "I could not find this information in the handbook.\n\n"
                    f"(Note: could not reach the local Ollama model '{OLLAMA_MODEL_NAME}'. "
                    f"Make sure Ollama is running and the model is pulled. Details: {e})"
                )

            pages_used = sorted(set(c["page"] for c in retrieved))
            pages_str = ", ".join(str(p) for p in pages_used)

            self.msg_queue.put(("chat", (f"Bot: {answer}", "bot")))
            self.msg_queue.put(("chat", (f"(Retrieved from page(s): {pages_str})", "citation")))
            self.msg_queue.put(("status", "Chatbot ready! Ask a question below."))
        except Exception as e:
            self.msg_queue.put(("chat", (f"[Error] {e}", "error")))
            self.msg_queue.put(("status", "Error occurred. Chatbot ready! Ask a question below."))
        finally:
            self.msg_queue.put(("ready", None))


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    try:
        # Use a more modern theme if available
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    app = HandbookChatbotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
