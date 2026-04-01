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

| Feature              | Ollama            | HuggingFace             | Cohere              |
| :------------------- | :---------------- | :---------------------- | :------------------ |
| **Cost**             | 100% Free         | Free Tier               | Free Tier           |
| **API Key Needed**   | ❌ No             | ✅ Yes (HF Token)       | ✅ Yes (Cohere Key) |
| **Runs Locally**     | ✅ Fully local    | ⚡ Embeddings only      | ❌ Cloud only       |
| **Setup Difficulty** | Medium            | Medium                  | Easiest             |
| **Best For**         | Privacy & offline | Local embedding control | Quick prototyping   |

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

## 🧪 Live Example — See RAG in Action

Using the included `sample.txt` (Acme Store policy document):

### Query 1 — Direct fact retrieval
```bash
python rag50_ollama.py --file sample.txt --query "What is the return policy?"
```
```
📄 Loading & chunking document …
🔢 Embedding 12 chunks …
🔍 Retrieving relevant context …
💬 Generating answer …

Q: What is the return policy?
A: You can return any unused item within 30 days of purchase for a full 
   refund. Items must be in original packaging with all tags attached. 
   Digital products and perishables are non-refundable.
```

### Query 2 — Multi-fact reasoning
```bash
python rag50_ollama.py --file sample.txt --query "How do I reach Gold status and what do I get?"
```
```
Q: How do I reach Gold status and what do I get?
A: Gold status is unlocked at 500 loyalty points (earned at $1 per point 
   spent). Benefits include free expedited shipping on all orders.
```

### Query 3 — Hallucination guard (the important one)
```bash
python rag50_ollama.py --file sample.txt --query "Do you offer student discounts?"
```
```
Q: Do you offer student discounts?
A: I don't know.
```

> ☝️ **This is RAG working correctly.** The document doesn't mention 
> student discounts, so the model refuses to guess. 
> This is the core superpower of RAG over plain LLM prompting.

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

## 🦙 Setting Up Ollama (Step by Step)

### macOS
```bash
# Option 1 — Direct download
# Visit https://ollama.com/download and download the macOS app

# Option 2 — Homebrew
brew install ollama
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
```
Visit https://ollama.com/download
Download and run the OllamaSetup.exe installer
```

### After install — pull the models
```bash
# Pull embedding model (~274 MB)
ollama pull nomic-embed-text

# Pull LLM (~2 GB — grab a coffee)
ollama pull llama3.2

# Verify both are ready
ollama list

# Start the server (keep this terminal open)
ollama serve
```

> **Note:** `ollama serve` must be running in the background whenever 
> you run the script. On macOS, the desktop app handles this automatically.

---

## 🤗 Getting Your HuggingFace Token

1. Go to https://huggingface.co/join and create a free account
2. Visit https://huggingface.co/settings/tokens
3. Click **"New token"** → name it anything → Role: **Read** → **Generate**
4. Copy the token (starts with `hf_`)

```bash
# Mac/Linux — add to current session
export HF_TOKEN="hf_your_token_here"

# Mac/Linux — make it permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.zshrc
source ~/.zshrc

# Windows (Command Prompt)
set HF_TOKEN="hf_your_token_here"

# Windows (PowerShell)
$env:HF_TOKEN="hf_your_token_here"
```

---

## 🟣 Getting Your Cohere API Key

1. Go to https://dashboard.cohere.com and create a free account
2. Visit https://dashboard.cohere.com/api-keys
3. Click **"New Trial Key"** → copy it immediately

```bash
# Mac/Linux
export COHERE_API_KEY="your_key_here"

# Mac/Linux — permanent
echo 'export COHERE_API_KEY="your_key_here"' >> ~/.zshrc
source ~/.zshrc

# Windows (Command Prompt)
set COHERE_API_KEY="your_key_here"

# Windows (PowerShell)
$env:COHERE_API_KEY="your_key_here"
```

> **Free tier limits:** 1,000 API calls/month — more than enough 
> for learning and experimentation.

---

## Project Structure

```text
tiny-rag/
├── rag50_ollama.py        # local, no API key, fully private
├── rag50_huggingface.py   # free cloud, local embeddings
├── rag50_cohere.py        # free tier, easiest setup
└── sample.txt             # demo document (Acme store policies)
```

---

## Contributing

Got a new provider? Keep it to **one file, under 70 lines,
same 5-step structure** — and open a PR.
