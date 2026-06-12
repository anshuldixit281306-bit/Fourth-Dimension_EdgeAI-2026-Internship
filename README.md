# Campus Handbook Chatbot (Offline RAG) — GUI Edition

A **100% offline**, pure-Python desktop chatbot that answers questions from
a college/school handbook PDF and shows **page citations** — built with
**Tkinter** (Python's built-in GUI library, no other languages or
frameworks needed).

## What's Included

- `handbook_bot_gui.py` — the full GUI application
- `handbook.pdf` — a sample "Greenwood High School" student handbook
  (8 pages: attendance, exams, dress code, library, anti-bullying,
  transport, contacts) for testing
- `requirements.txt`
- `.gitignore`

## How It Works

1. **PDF extraction** — `PyMuPDF (fitz)` reads the handbook page by page.
2. **Chunking** — each page is split into ~300-word chunks, tagged with
   its page number.
3. **Embeddings** — `sentence-transformers` (`all-MiniLM-L6-v2`) converts
   each chunk into a vector.
4. **Vector search** — `FAISS` finds the top 3 chunks most relevant to your
   question.
5. **Answer generation** — the retrieved chunks + page numbers are sent to
   a **local** `llama3.2:1b` model via **Ollama**, which writes the final
   answer with citations like "(Page 2)".
6. If the retrieved content doesn't answer the question, the bot replies:
   *"I could not find this information in the handbook."*

Everything runs on your machine — no internet connection or API keys
required after the one-time model downloads.

## Setup

1. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```
   (On Linux, Tkinter may need a separate system package:
   `sudo apt install python3-tk`)

2. **Install Ollama** and pull the model:
   ```bash
   ollama pull llama3.2:1b
   ```
   Make sure `ollama serve` is running (Ollama usually starts this
   automatically after installation).

3. **(One-time, needs internet)** The first run downloads the
   `all-MiniLM-L6-v2` embedding model (~80MB) from Hugging Face and caches
   it locally. After that, everything works fully offline.

## Running the App

```bash
python handbook_bot_gui.py
```

- The app launches with the bundled `handbook.pdf` (Greenwood High School
  sample) by default.
- Use **"Choose PDF..."** to load your own college/school handbook —
  this automatically rebuilds the index for the new file.
- Use **"Rebuild Index"** if you've edited the current PDF and want to
  refresh the search index.
- Type a question in the box at the bottom and press **Enter** or click
  **Ask**.

### Try These Sample Questions (with the bundled handbook)

- "What is the attendance policy?" → answers from **Page 2**
- "Can I use my phone during an exam?" → answers from **Page 3**
- "What is the dress code?" → answers from **Page 4**
- "How many books can I borrow from the library?" → answers from **Page 5**
- "What happens if there's bullying?" → answers from **Page 6**
- "What is the capital of France?" → *"I could not find this information
  in the handbook."* (correctly rejects out-of-scope questions)

## Files Generated Automatically

- `handbook.index` — saved FAISS vector index
- `chunks.pkl` — saved text chunks + page numbers

These are cached so the app starts instantly on future runs. Delete them
(or click **"Rebuild Index"**) to regenerate after changing the PDF.

## Security Notes

- **No external APIs** — all computation (embeddings, search, and LLM
  generation) happens locally via Ollama.
- **No arbitrary file access** — PDFs are only loaded via the OS file
  picker (`Choose PDF...`), and the app verifies the file has a `.pdf`
  extension before processing.
- **Pickle usage** — `chunks.pkl` only ever stores plain text/number data
  (chunk text + page numbers) and is only ever written by this app. Avoid
  replacing it with a `chunks.pkl` file from an untrusted source, since
  `pickle.load()` can execute arbitrary code if the file is crafted
  maliciously.
- **Prompt design** — the LLM is explicitly instructed to answer only from
  retrieved handbook content and to say when information isn't found,
  reducing hallucinated or off-topic answers.
- **Local network only** — Ollama runs on `localhost:11434` by default;
  don't expose this port to the internet.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'tkinter'` | Install `python3-tk` (Linux) |
| "Could not reach the local Ollama model" | Run `ollama serve` and `ollama pull llama3.2:1b` |
| Embedding model download fails | Check internet connection (only needed the first time) |
| "No extractable text found in this PDF" | The PDF is likely scanned images; use an OCR tool first |
