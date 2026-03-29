# RAG Document Chatbot
Built a Retrieval-Augmented Generation (RAG) chatbot using LangChain and OpenAI to answer questions from PDF documents with semantic search and MMR-based retrieval.

## Features

- Load and process PDF documents
- Text chunking with overlap
- Embeddings generation using OpenAI
- Vector database with Chroma
- Semantic search using MMR
- Question answering with an LLM

## Tech Stack
- Python
- LangChain
- OpenAI
- ChromaDB

## Installation

```bash
pip install -r requirements.txt
```

## Setup

Create a .env file and add your API key:
```
OPEN_API_KEY=your_api_key
```

