# rag50_huggingface.py — RAG with HuggingFace (free, no GPU needed)
# Prerequisites:
#   1. Get a free API key:    https://huggingface.co/settings/tokens
#   2. Install deps:          pip install sentence-transformers huggingface-hub numpy
#   3. Set env variable:      export HF_TOKEN=hf_...   (Mac/Linux)
#                             set HF_TOKEN=hf_...       (Windows)
# Usage: python rag50_huggingface.py --file sample.txt --query "What is the return policy?"

import os
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── 1. LOAD & CHUNK ──
def load_chunks(path, chunk_size=300):
    with open(path, "r", encoding="utf-8") as f:
        words = f.read().split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size//2)]

# ── 2. EMBED ──
def embed(texts: list[str]) -> np.ndarray:
    return embedding_model.encode(texts).astype(np.float32)

# ── 3. SIMILARITY SEARCH ──
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, chunks, vectors, top_k=3):
    q_vec = embed([query])[0]
    scores = [cosine_similarity(q_vec, v) for v in vectors]
    return [chunks[i] for i in np.argsort(scores)[-top_k:][::-1]]

# ── 4. GENERATE ──
def generate(query, context, sys_prompt):
    client = InferenceClient(token=os.environ["HF_TOKEN"])
    messages = [
        {"role": "system", "content": f"{sys_prompt}\n\nContext:\n{context}"},
        {"role": "user", "content": query}
    ]
    try:
        response = client.chat_completion(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content
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
