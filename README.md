# 📚 HR Policy Assistant

A simple HR Policy Assistant built using Retrieval-Augmented Generation (RAG).

The application allows users to upload HR policy documents and ask questions about them.

## Features

- Upload PDF, DOCX, and TXT HR policy documents
- Extract text from documents
- Split documents into smaller chunks
- Create embeddings using Sentence Transformers
- Search documents using FAISS
- Generate answers using Groq
- Show retrieved source documents
- Answers are based only on uploaded HR policy information

## Technologies

- Python
- Streamlit
- Sentence Transformers
- FAISS
- PyMuPDF
- python-docx
- Groq API

## How It Works

```text
Upload documents
       ↓
Extract text
       ↓
Split into chunks
       ↓
Create embeddings
       ↓
Store embeddings in FAISS
       ↓
User asks question
       ↓
Search relevant chunks
       ↓
Send relevant chunks to Groq
       ↓
Generate answer
       ↓
Show answer and sources
