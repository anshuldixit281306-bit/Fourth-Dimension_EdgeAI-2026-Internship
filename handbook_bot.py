"""
Campus Handbook Chatbot (Offline RAG)
=======================================

A fully offline Retrieval-Augmented Generation (RAG) chatbot that answers
questions from a college handbook PDF and provides page citations.

Tech stack (all local, no external APIs):
- PyMuPDF (fitz)        -> extract text from the handbook PDF
- Sentence Transformers -> create embeddings (all-MiniLM-L6-v2)
- FAISS                 -> vector similarity search
- NumPy                 -> store/manage embedding arrays
- Ollama (llama3.2:1b)  -> local LLM for answer generation

Author: Internship Project Demo
"""

import os
import sys
import pickle

import fitz                     # PyMuPDF - for reading PDF files
import numpy as np              # for storing/handling embeddings
import faiss                    # for fast vector similarity search
import ollama                   # for talking to the local Ollama LLM
from sentence_transformers import SentenceTransformer

# CONFIGURATION

PDF_PATH = "handbook.pdf"          # the college handbook PDF
INDEX_PATH = "handbook.index"      # saved FAISS index
CHUNKS_PATH = "chunks.pkl"         # saved text chunks + page numbers

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # sentence-transformers model
OLLAMA_MODEL_NAME = "llama3.2:1b"            # local LLM via Ollama

CHUNK_WORD_SIZE = 300               # approx words per chunk
TOP_K = 3                            # number of chunks to retrieve per query



# STEP 1: EXTRACT TEXT FROM PDF (PAGE BY PAGE)

def extract_text_by_page(pdf_path):
    """
    Open the PDF and extract text from each page separately.

    Returns:
        A list of tuples: (page_number, page_text)
        page_number is 1-indexed (so it matches what a human sees in the PDF).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"Could not find '{pdf_path}'. "
            f"Please place your handbook PDF in the project folder "
            f"and name it '{pdf_path}'."
        )

    print("Loading PDF...")
    doc = fitz.open(pdf_path)

    pages_text = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()
        pages_text.append((page_number, text))

    doc.close()
    print(f"Loaded {len(pages_text)} pages from '{pdf_path}'.")
    return pages_text



# STEP 2: SPLIT TEXT INTO ~300 WORD CHUNKS (KEEPING PAGE NUMBERS)

def chunk_text(pages_text, chunk_size=CHUNK_WORD_SIZE):
    """
    Split each page's text into chunks of approximately `chunk_size` words.
    Each chunk remembers which page it came from, so we can cite it later.

    Returns:
        A list of dicts: {"text": chunk_text, "page": page_number}
    """
    chunks = []

    for page_number, text in pages_text:
        words = text.split()

        # Skip empty pages (e.g., blank pages or images-only pages)
        if not words:
            continue

        # Break the page's words into groups of `chunk_size`
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_str = " ".join(chunk_words)
            chunks.append({"text": chunk_str, "page": page_number})

    print(f"Created {len(chunks)} text chunks (~{chunk_size} words each).")
    return chunks



# STEP 3: GENERATE EMBEDDINGS FOR EACH CHUNK

def create_embeddings(chunks, embedding_model):
    """
    Convert each text chunk into a numeric vector (embedding) using
    the all-MiniLM-L6-v2 sentence-transformers model.

    Returns:
        A NumPy array of shape (num_chunks, embedding_dim), dtype float32.
    """
    print("Creating embeddings...")
    texts = [chunk["text"] for chunk in chunks]

    # encode() returns a list/array of embedding vectors
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # FAISS requires float32 arrays
    embeddings = np.array(embeddings, dtype="float32")
    return embeddings



# STEP 4: BUILD A FAISS INDEX FROM THE EMBEDDINGS

def build_faiss_index(embeddings):
    """
    Build a simple FAISS index using L2 (Euclidean) distance for
    similarity search.

    Returns:
        A FAISS IndexFlatL2 object containing all chunk embeddings.
    """
    print("Building FAISS index...")
    embedding_dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors "
          f"(dimension = {embedding_dim}).")
    return index



# STEP 5: SAVE / LOAD INDEX AND CHUNKS (SO WE DON'T REBUILD EVERY TIME)

def save_index_and_chunks(index, chunks):
    """Save the FAISS index to disk and the chunks list using pickle."""
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index to '{INDEX_PATH}' and chunks to '{CHUNKS_PATH}'.")


def load_index_and_chunks():
    """Load a previously saved FAISS index and chunks list from disk."""
    print("Loading existing index...")
    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded FAISS index ({index.ntotal} vectors) "
          f"and {len(chunks)} chunks.")
    return index, chunks


def index_and_chunks_exist():
    """Check whether saved index/chunks files already exist on disk."""
    return os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)



# STEP 6: RETRIEVE TOP-K MOST RELEVANT CHUNKS FOR A QUESTION

def retrieve_relevant_chunks(question, embedding_model, index, chunks, top_k=TOP_K):
    """
    Convert the user's question into an embedding, search the FAISS index
    for the most similar chunks, and return those chunks (with page numbers).

    Returns:
        A list of chunk dicts: [{"text": ..., "page": ...}, ...]
    """
    # Convert the question into an embedding vector
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )
    question_embedding = np.array(question_embedding, dtype="float32")

    # Search the FAISS index for the top_k nearest chunks
    distances, indices = index.search(question_embedding, top_k)

    # indices[0] contains the row numbers of the closest chunks
    retrieved = [chunks[i] for i in indices[0] if i != -1]
    return retrieved

# STEP 7: BUILD THE PROMPT FOR THE LLM (INCLUDES CHUNKS + PAGE NUMBERS)

def build_prompt(question, retrieved_chunks):
    """
    Combine the retrieved chunks (with page numbers) and the user's question
    into a single prompt for the LLM. Instruct the model to:
      - Only use the provided context.
      - Cite page numbers.
      - Say it could not find the info if the context doesn't answer it.
    """
    context_blocks = []
    for chunk in retrieved_chunks:
        context_blocks.append(f"[Page {chunk['page']}]\n{chunk['text']}")

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

# STEP 8: SEND PROMPT TO OLLAMA (llama3.2:1b) AND GET THE ANSWER

def ask_llm(prompt):
    """
    Send the prompt to the local Ollama model (llama3.2:1b) and return
    the generated answer text.
    """
    response = ollama.chat(
        model=OLLAMA_MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"].strip()


# STEP 9: MAIN SETUP - LOAD/BUILD EMBEDDING MODEL, INDEX, AND CHUNKS

def setup():
    """
    Prepare everything needed for the chatbot:
    - Load the sentence-transformers embedding model.
    - Load existing FAISS index + chunks if available,
      otherwise build them from the PDF.

    Returns:
        (embedding_model, index, chunks)
    """
    # Load the embedding model (used for both indexing and querying)
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    if index_and_chunks_exist():
        # Reuse previously built index/chunks (faster startup)
        index, chunks = load_index_and_chunks()
    else:
        # Build everything from scratch
        pages_text = extract_text_by_page(PDF_PATH)
        chunks = chunk_text(pages_text)
        embeddings = create_embeddings(chunks, embedding_model)
        index = build_faiss_index(embeddings)
        save_index_and_chunks(index, chunks)

    return embedding_model, index, chunks


# STEP 10: CHAT LOOP - CONTINUOUSLY ASK QUESTIONS UNTIL "quit"

def chat_loop(embedding_model, index, chunks):
    """
    Run an interactive terminal loop:
    - Take a question from the user.
    - Retrieve relevant chunks.
    - Build a prompt with context + page numbers.
    - Get an answer from Ollama (llama3.2:1b).
    - Print the answer with citations.
    - Repeat until the user types "quit".
    """
    print("\nChatbot ready! Ask a question about the handbook.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "quit":
            print("Goodbye!")
            break

        if not question:
            continue

        # Step 1: Retrieve the most relevant chunks for this question
        retrieved_chunks = retrieve_relevant_chunks(
            question, embedding_model, index, chunks, top_k=TOP_K
        )

        # Step 2: Build the prompt for the LLM
        prompt = build_prompt(question, retrieved_chunks)

        # Step 3: Get the answer from the local LLM
        try:
            answer = ask_llm(prompt)
        except Exception as e:
            print(f"\nError talking to Ollama: {e}")
            print("Make sure Ollama is running and the model "
                  f"'{OLLAMA_MODEL_NAME}' is pulled "
                  f"(run: ollama pull {OLLAMA_MODEL_NAME}).\n")
            continue

        # Step 4: Display the answer
        print(f"\nBot: {answer}")

        # Step 5: Also list which pages were used for retrieval (for clarity)
        pages_used = sorted(set(chunk["page"] for chunk in retrieved_chunks))
        pages_str = ", ".join(str(p) for p in pages_used)
        print(f"(Retrieved from page(s): {pages_str})\n")


# ENTRY POINT

def main():
    try:
        embedding_model, index, chunks = setup()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during setup: {e}")
        sys.exit(1)

    chat_loop(embedding_model, index, chunks)


if __name__ == "__main__":
    main()
