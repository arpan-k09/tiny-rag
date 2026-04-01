# rag50_cohere.py — RAG with Cohere (free tier, easiest setup)
# Prerequisites:
#   1. Get a free API key:    https://dashboard.cohere.com/api-keys
#   2. Install deps:          pip install cohere numpy
#   3. Set env variable:      export COHERE_API_KEY=...   (Mac/Linux)
#                             set COHERE_API_KEY=...       (Windows)
# Usage: python rag50_cohere.py --file sample.txt --query "What is the return policy?"

import os
import argparse
import numpy as np
import cohere

client = cohere.ClientV2(os.environ.get("COHERE_API_KEY", ""))

# ── 1. LOAD & CHUNK ──
def load_chunks(path, chunk_size=300):
    with open(path, "r", encoding="utf-8") as f:
        words = f.read().split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size//2)]

# ── 2. EMBED ──
def embed(texts: list[str], input_type: str = "search_document") -> np.ndarray:
    resp = client.embed(texts=texts, model="embed-v4.0", input_type=input_type, embedding_types=["float"])
    return np.array(resp.embeddings.float, dtype=np.float32)

# ── 3. SIMILARITY SEARCH ──
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, chunks, vectors, top_k=3):
    q_vec = embed([query], input_type="search_query")[0]
    scores = [cosine_similarity(q_vec, v) for v in vectors]
    return [chunks[i] for i in np.argsort(scores)[-top_k:][::-1]]

# ── 4. GENERATE ──
def generate(query, context, sys_prompt):
    messages = [
        {"role": "system", "content": f"{sys_prompt}\n\nContext:\n{context}"},
        {"role": "user", "content": query}
    ]
    try:
        response = client.chat(model="command-a-03-2025", messages=messages, temperature=0)
        return response.message.content[0].text
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
