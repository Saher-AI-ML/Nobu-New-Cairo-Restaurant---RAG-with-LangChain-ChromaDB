<img width="1894" height="772" alt="Restaurant Interface" src="https://github.com/user-attachments/assets/9523c75c-0bed-486c-ae35-ca4cd4db4d15" />
# Nobu New Cairo Restaurant Q&A — RAG with LangChain & ChromaDB

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about **Nobu New Cairo** restaurant using only content scraped from its own data — no hallucinated menu items or made-up opening hours. Built with LangChain, ChromaDB, and a Gradio chat UI, and designed to run in Google Colab.

## Overview

This project turns a plain text file of restaurant information (menu, location, hours, etc.) into a searchable knowledge base and wraps it in a conversational interface. When a user asks a question, the app:

1. Embeds the question and searches a ChromaDB vector store for the most relevant chunks of restaurant data.
2. Passes those chunks to an LLM as context.
3. Returns an answer that's grounded in the source text, along with the source it came from.

If the answer isn't in the provided data, the assistant says it doesn't know instead of guessing.

## How it works (architecture)

```
nobu_new_cairo_data.txt
 │
 ▼
 Text Splitter (RecursiveCharacterTextSplitter)
 │
 ▼
 OpenAI-compatible Embeddings (via OpenRouter)
 │
 ▼
 ChromaDB Vector Store ──▶ Similarity Search (top-k retrieval)
 │
 ▼
 Prompt Template + Chat LLM (gpt-4o-mini via OpenRouter)
 │
 ▼
 Answer + Cited Sources ──▶ Gradio Web UI
```

The RAG chain is built with LangChain Expression Language (LCEL), chaining together retrieval, context formatting, answer generation, and source collection into a single runnable pipeline.

## Features

- **Document ingestion** — loads and chunks a plain-text knowledge base
- **Semantic search** — ChromaDB vector store with OpenAI-compatible embeddings
- **Grounded answers** — LLM is instructed to answer *only* from retrieved context
- **Source attribution** — every answer includes the source document it drew from
- **Interactive UI** — Gradio chat interface with example questions built in

## Tech stack

| Component | Tool / Library |
|-------------------|------------------------------------------|
| Orchestration | [LangChain](https://python.langchain.com/) |
| Vector store | [ChromaDB](https://www.trychroma.com/) |
| LLM & Embeddings | `gpt-4o-mini` and `text-embedding-3-small` via [OpenRouter](https://openrouter.ai/) |
| UI | [Gradio](https://www.gradio.app/) |
| Runtime | Python 3, Google Colab |

## Prerequisites

- A Google Colab account (or a local Jupyter environment)
- An [OpenRouter](https://openrouter.ai/) API key (used here as an OpenAI-compatible endpoint)
- A text file containing the restaurant's data (`nobu_new_cairo_data.txt`)

## Setup & installation

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies
```bash
pip install langchain langchain-openai openai chromadb gradio tiktoken langchain-community langchain-chroma
```

### 3. Add your API key
This project reads the API key from Colab's secret manager:
```python
from google.colab import userdata
api = userdata.get('openaiapi')
```
In Colab, add a secret named `openaiapi` (via the secrets icon in the left sidebar) containing your OpenRouter API key.

> **Running locally instead of Colab?** Replace the `userdata.get(...)` call with an environment variable, e.g. `api = os.environ["OPENROUTER_API_KEY"]`.

### 4. Add your data file
Place a plain text file named `nobu_new_cairo_data.txt` in the working directory. This should contain the restaurant's menu, hours, location, and any other information you want the assistant to answer from.

## Usage

Open and run `Nobu_New_Cairo_Restaurant_RAG_with_LangChain___ChromaDB.ipynb` top to bottom. The notebook will:

1. Load and chunk `nobu_new_cairo_data.txt`
2. Build embeddings and populate a ChromaDB collection
3. Run a couple of test queries to sanity-check retrieval
4. Assemble the full RAG chain
5. Launch a Gradio app with a shareable link

Once launched, ask things like:
- *"What are the different menu options and prices?"*
- *"Where is the restaurant located?"*
- *"What are the opening hours of Nobu New Cairo?"*
- *"What does Nobu recommend for someone dining there for the first time?"*

## Configuration

Key parameters you can tune in the notebook:

| Parameter | Location | Default | Purpose |
|---|---|---|---|
| `chunk_size` / `chunk_overlap` | Text splitter | 1000 / 50 | Controls how the source text is chunked |
| `k` (retriever) | `as_retriever(search_kwargs={"k": 3})` | 3 | Number of chunks retrieved per query |
| `model` (embeddings) | `OpenAIEmbeddings` | `openai/text-embedding-3-small` | Embedding model |
| `model` (chat) | `ChatOpenAI` | `openai/gpt-4o-mini` | Answer-generation model |
| `temperature` | `ChatOpenAI` | 0 | Lower = more deterministic answers |

## Future improvements

- Generalize this into a RAG chatbot that works for **any restaurant**, not just Nobu New Cairo — swap in a different data file (or let users upload their own) and the same pipeline should work out of the box.

