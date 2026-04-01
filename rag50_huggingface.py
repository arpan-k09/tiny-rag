"""RAG with HuggingFace: local SentenceTransformer embeddings, free Inference API for chat."""
import os
import sys
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    sys.exit("HF_TOKEN not set. Get yours at https://huggingface.co/settings/tokens")

CHUNK_SIZE = 300
TOP_K = 3
EMBED_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
embedding_model = SentenceTransformer(EMBED_MODEL)
hf_client = InferenceClient(token=HF_TOKEN)

# -- 1. LOAD & CHUNK --
def load_chunks(path: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split file into overlapping word windows of chunk_size with 50% overlap."""
    with open(path, "r", encoding="utf-8") as f:
        words = f.read().split()
    return [c for c in (" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size//2)) if c]

# -- 2. EMBED --
def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts using SentenceTransformer and return as float32 array."""
    return embedding_model.encode(texts).astype(np.float32)

# -- 3. SIMILARITY SEARCH --
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return cosine similarity between two vectors using pure numpy."""
    return np.dot(a, b) / ((np.linalg.norm(a) + 1e-10) * (np.linalg.norm(b) + 1e-10))

def retrieve(query: str, chunks: list[str], chunk_vectors: np.ndarray, top_k: int = TOP_K) -> list[str]:
    """Embed query and return the top_k most similar chunks."""
    q_vec = embed([query])[0]
    scores = [cosine_similarity(q_vec, v) for v in chunk_vectors]
    return [chunks[i] for i in np.argsort(scores)[-top_k:][::-1]]

# -- 4. GENERATE --
def generate(query: str, context: str) -> str:
    """Send query and context to HuggingFace Inference API and return the answer."""
    sys_prompt = "Answer using ONLY the context below. If the answer is not there, say I don't know."
    messages = [
        {"role": "system", "content": f"{sys_prompt}\n\nContext:\n{context}"},
        {"role": "user", "content": query},
    ]
    response = hf_client.chat_completion(model=CHAT_MODEL, messages=messages, temperature=0)
    return response.choices[0].message.content

# -- 5. MAIN --
def main() -> None:
    """Parse CLI args, run the RAG pipeline, and print the answer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    print("Loading and chunking document")
    chunks = load_chunks(args.file)
    print(f"Embedding {len(chunks)} chunks")
    vectors = embed(chunks)
    print("Retrieving relevant context")
    context_chunks = retrieve(args.query, chunks, vectors, args.top_k)
    print("Generating answer")
    answer = generate(args.query, "\n---\n".join(context_chunks))
    print(f"Q: {args.query}")
    print(f"A: {answer}")

if __name__ == "__main__":
    main()
