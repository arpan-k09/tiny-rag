"""RAG with Cohere: cloud embeddings and chat via the free-tier API."""
import os
import sys
import argparse
import numpy as np
import cohere

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
if not COHERE_API_KEY:
    sys.exit("COHERE_API_KEY not set. Get yours at https://dashboard.cohere.com/api-keys")

CHUNK_SIZE = 300
TOP_K = 3
EMBED_MODEL = "embed-v4.0"
CHAT_MODEL = "command-a-03-2025"
co = cohere.ClientV2(COHERE_API_KEY)

# -- 1. LOAD & CHUNK --
def load_chunks(path: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split file into overlapping word windows of chunk_size with 50% overlap."""
    with open(path, "r", encoding="utf-8") as f:
        words = f.read().split()
    return [c for c in (" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size//2)) if c]

# -- 2. EMBED --
def embed(texts: list[str], input_type: str = "search_document") -> np.ndarray:
    """Embed a list of texts via Cohere and return as float32 array."""
    resp = co.embed(texts=texts, model=EMBED_MODEL, input_type=input_type, embedding_types=["float"])
    return np.array(resp.embeddings.float_, dtype=np.float32)

# -- 3. SIMILARITY SEARCH --
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return cosine similarity between two vectors using pure numpy."""
    return np.dot(a, b) / ((np.linalg.norm(a) + 1e-10) * (np.linalg.norm(b) + 1e-10))

def retrieve(query: str, chunks: list[str], chunk_vectors: np.ndarray, top_k: int = TOP_K) -> list[str]:
    """Embed query and return the top_k most similar chunks."""
    q_vec = embed([query], input_type="search_query")[0]
    scores = [cosine_similarity(q_vec, v) for v in chunk_vectors]
    return [chunks[i] for i in np.argsort(scores)[-top_k:][::-1]]

# -- 4. GENERATE --
def generate(query: str, context: str) -> str:
    """Send query and context to Cohere chat and return the answer."""
    sys_prompt = "Answer using ONLY the context below. If the answer is not there, say I don't know."
    messages = [
        {"role": "system", "content": f"{sys_prompt}\n\nContext:\n{context}"},
        {"role": "user", "content": query},
    ]
    response = co.chat(model=CHAT_MODEL, messages=messages, temperature=0)
    return response.message.content[0].text

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
