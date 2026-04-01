# rag50_ollama.py — RAG with Ollama (local, free, no API key)
# Prerequisites:
#   1. Install Ollama:        https://ollama.com/download
#   2. Pull models:           ollama pull nomic-embed-text
#                             ollama pull llama3.2
#   3. Start Ollama server:   ollama serve
#   4. Install deps:          pip install ollama numpy
# Usage: python rag50_ollama.py --file sample.txt --query "What is the return policy?"

import argparse
import numpy as np
import ollama

# ── 1. LOAD & CHUNK ──
def load_chunks(path, chunk_size=300):
    with open(path, "r", encoding="utf-8") as f:
        words = f.read().split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size//2)]

# ── 2. EMBED ──
def embed(texts: list[str]) -> np.ndarray:
    return np.array([ollama.embeddings(model="nomic-embed-text", prompt=t)["embedding"] for t in texts], dtype=np.float32)

# ── 3. SIMILARITY SEARCH ──
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, chunks, vectors, top_k=3):
    q_vec = embed([query])[0]
    scores = [cosine_similarity(q_vec, v) for v in vectors]
    return [chunks[i] for i in np.argsort(scores)[-top_k:][::-1]]

# ── 4. GENERATE ──
def generate(query, context, sys_prompt):
    messages = [
        {"role": "system", "content": f"{sys_prompt}\n\nContext:\n{context}"},
        {"role": "user", "content": query}
    ]
    try:
        response = ollama.chat(model="llama3.2", messages=messages, options={"temperature": 0})
        return response['message']['content']
    except Exception as e:
        return f"Error: {e}"

# ── 5. MAIN ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    print("📄 Loading & chunking document …")
    chunks = load_chunks(args.file)
    print(f"🔢 Embedding {len(chunks)} chunks …")
    vectors = embed(chunks)

    print("🔍 Retrieving relevant context …")
    context_chunks = retrieve(args.query, chunks, vectors, args.top_k)

    print("💬 Generating answer …")
    sys_prompt = "Answer using ONLY the context below.\nIf the answer isn't there, say 'I don't know'."
    ans = generate(args.query, "\n".join(context_chunks), sys_prompt)
    print(f"\nQ: {args.query}\nA: {ans}")

if __name__ == "__main__":
    main()
