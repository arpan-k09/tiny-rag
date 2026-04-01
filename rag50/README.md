# rag50: Minimal RAG Implementations

A project demonstrating Retrieval-Augmented Generation (RAG) using three different provider implementations, all sharing one sample document.

## Provider Comparison

| Feature              | Ollama                                   | HuggingFace                           | Cohere                                         |
| :------------------- | :--------------------------------------- | :------------------------------------ | :--------------------------------------------- |
| **Cost**             | 100% Free                                | Free Tier                             | Free Tier limits apply                         |
| **API Key Needed**   | No                                       | Yes (HF Token)                        | Yes (Cohere API Key)                           |
| **Runs Locally**     | Yes (both Models & Embeddings)           | Yes (Embeddings only)                 | No (Cloud APIs)                                |
| **Setup Difficulty** | Medium (Requires local software install) | Medium (Simple pip install + API Key) | Easiest (Drop-in Python API + Key)             |
| **Best For**         | Maximum privacy and offline usage        | Tinkering / local embeddings control  | Quick prototyping without hardware limitations |

## Quick Test Guides

### Quick Test — rag50_ollama.py

```bash
# 1. Start the Ollama server in a separate terminal:
ollama serve

# 2. Pull the required models:
ollama pull nomic-embed-text
ollama pull llama3.2

# 3. Install dependencies:
pip install ollama numpy

# 4. Run the script:
python rag50_ollama.py --file sample.txt --query "What is the return policy?"
```

### Quick Test — rag50_huggingface.py

```bash
# 1. Install dependencies:
pip install sentence-transformers huggingface-hub numpy

# 2. Set the environment variable:
export HF_TOKEN="hf_your_free_api_token"   # (Mac/Linux)
# set HF_TOKEN="hf_your_free_api_token"    # (Windows)

# 3. Run the script:
python rag50_huggingface.py --file sample.txt --query "What is the return policy?"
```

### Quick Test — rag50_cohere.py

```bash
# 1. Install dependencies:
pip install cohere numpy

# 2. Set the environment variable:
export COHERE_API_KEY="your_free_cohere_key"   # (Mac/Linux)
# set COHERE_API_KEY="your_free_cohere_key"    # (Windows)

# 3. Run the script:
python rag50_cohere.py --file sample.txt --query "What is the return policy?"
```
