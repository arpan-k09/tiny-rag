# rag-from-scratch 🔍

> RAG without the magic. No LangChain. No vector DB. Just Python you can actually read.

Retrieval-Augmented Generation explained in **~50 lines per file** — pick your provider and go.
```bash
python rag50_ollama.py --file sample.txt --query "What is the return policy?"
# 📄 Loading & chunking document …
# 🔢 Embedding 12 chunks …
# 🔍 Retrieving relevant context …
# 💬 Generating answer …
#
# Q: What is the return policy?
# A: Customers may return any unused item within 30 days of purchase for a full refund.
```

---

## Why this exists

Most RAG tutorials hide the hard parts behind framework abstractions.  
This repo shows **every step raw** — chunking, embedding, cosine similarity, 
context stuffing — in plain Python you can read top to bottom in 5 minutes.

## Choose your provider

| Feature              | Ollama                | HuggingFace                 | Cohere                       |
| :------------------- | :-------------------- | :-------------------------- | :--------------------------- |
| **Cost**             | 100% Free             | Free Tier                   | Free Tier                    |
| **API Key Needed**   | ❌ No                 | ✅ Yes (HF Token)           | ✅ Yes (Cohere Key)          |
| **Runs Locally**     | ✅ Fully local        | ⚡ Embeddings only           | ❌ Cloud only                |
| **Setup Difficulty** | Medium                | Medium                      | Easiest                      |
| **Best For**         | Privacy & offline     | Local embedding control     | Quick prototyping            |

**Not sure which to pick?**
- No API key, want full privacy → **Ollama**
- Want local embeddings, okay with a free key → **HuggingFace**
- Just want it running in 2 minutes → **Cohere**

---

## How it works (same pipeline, 3 providers)
```
your_doc.txt  →  load_chunks()  →  embed()  →  [vectors in RAM]
                                                      ↑
user query    →  embed()  →  cosine_similarity()  →  retrieve()  →  generate()  →  answer
```

Every file implements the same 5 steps:

1. **Load & Chunk** — split document into overlapping word windows
2. **Embed** — convert chunks to vectors via your chosen provider
3. **Cosine Similarity** — manual numpy (no sklearn magic)
4. **Retrieve** — rank chunks, return top-k
5. **Generate** — stuff context into prompt, get grounded answer

---

## Quick Start

### Ollama (local, no API key)
```bash
ollama pull nomic-embed-text && ollama pull llama3.2
ollama serve   # separate terminal
pip install ollama numpy
python rag50_ollama.py --file sample.txt --query "What is the return policy?"
```

### HuggingFace (free tier)
```bash
pip install sentence-transformers huggingface-hub numpy
export HF_TOKEN="hf_your_token"
python rag50_huggingface.py --file sample.txt --query "What is the return policy?"
```

### Cohere (easiest)
```bash
pip install cohere numpy
export COHERE_API_KEY="your_key"
python rag50_cohere.py --file sample.txt --query "What is the return policy?"
```

---

## Project Structure
```text
rag-from-scratch/
├── rag50_ollama.py        # local, no API key, fully private
├── rag50_huggingface.py   # free cloud, local embeddings
├── rag50_cohere.py        # free tier, easiest setup
└── sample.txt             # demo document (Acme store policies)
```

---

## Contributing

Got a new provider? Keep it to **one file, under 70 lines, 
same 5-step structure** — and open a PR.
