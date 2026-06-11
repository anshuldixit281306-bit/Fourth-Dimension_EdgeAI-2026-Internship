# Fourth-Dimension_EdgeAI-2026-Internship

# Campus Handbook Chatbot (Offline RAG)

## Project Overview

The Campus Handbook Chatbot is an offline Retrieval-Augmented Generation (RAG) system that allows students to ask questions about their college handbook and receive accurate answers with page citations. Instead of relying on memorized information, the chatbot retrieves relevant content directly from the handbook PDF and uses a Small Language Model (SLM) to generate responses based on the retrieved text.

This project is built entirely offline and does not require any external APIs or internet connectivity after setup.

## Features

- PDF-based question answering
- Retrieval-Augmented Generation (RAG)
- Page citation support
- Offline execution
- Fast semantic search using FAISS
- Lightweight embedding model for efficient retrieval
- Uses Llama 3.2 1B through Ollama

## Technology Stack

- Python
- PyMuPDF
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS
- Ollama
- Llama 3.2 1B

## Workflow

1. Load the college handbook PDF.
2. Extract text from each page.
3. Split the text into smaller chunks.
4. Generate embeddings using MiniLM.
5. Store embeddings in a FAISS vector index.
6. Accept a student's question.
7. Retrieve the most relevant handbook sections.
8. Generate an answer using Llama 3.2 1B.
9. Display the answer with page citations.

## Project Architecture

```text
College Handbook PDF
        ↓
   Text Chunks
        ↓
 MiniLM Embeddings
        ↓
    FAISS Index
        ↓
  User Question
        ↓
 Relevant Chunks
        ↓
  Llama 3.2 1B
        ↓
Answer + Citation
```

## Example Questions

- What is the exam fee?
- How do I apply for hostel accommodation?
- What is the attendance requirement?
- What are the library rules?
- How can I apply for a scholarship?

## Use Cases

- Academic regulations
- Examination information
- Hostel procedures
- Scholarship details
- Attendance policies
- Library guidelines
- Student support services

## Future Enhancements

- Web-based interface
- Multiple PDF support
- Chat history
- Saved FAISS index
- Jetson Orin deployment
- Voice-enabled chatbot

## Objective

The objective of this project is to build a fully offline document question-answering system using Retrieval-Augmented Generation (RAG) and Llama 3.2 1B. The chatbot improves information accessibility for students by providing accurate answers directly from the college handbook with page references.
