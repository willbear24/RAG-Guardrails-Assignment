import chromadb
import requests
import json
import os
import time

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"


# Ingest
def load_documents(direcory):
    """Read every .txt and .md file in a folder and split the text into paragraph chunks."""
    docs = []
    for filename in sorted(os.listdir(direcory)):
        if filename.endswith((".txt", ".md")):
            with open(os.path.join(direcory, filename), "r") as f:
                content = f.read()

            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for i, para in enumerate(paragraphs):
                docs.append(
                    {
                        "text": para,
                        "id": f"{filename}_{i}",
                        "metadata": {"source": filename, "chunk_index": str(i)},
                    }
                )
    return docs


def ingest(collection, docs_directory):
    """Load text files, split them into chunks, and store them in ChromaDB."""
    chunks = load_documents(docs_directory)
    if not chunks:
        print("No documents found!")
        return 0

    collection.upsert(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[c["id"] for c in chunks],
    )

    return len(chunks)


# Retrieve
def safe_retrieve(collection, query, n_results=3, max_distance=1.0):
    """Query ChromaDB for the most relevant document chunks for a question."""
    results = collection.query(
        query_texts=[query], n_results=min(n_results, collection.count())
    )
    filtered = []
    for i in range(len(results["documents"][0])):
        dist = results["distances"][0][i]
        if dist <= max_distance:
            filtered.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

    return filtered


def get_confidence(chunks):
    if not chunks:
        return "none", "🔴"
    best = chunks[0]["distance"]
    if best < 0.65:
        return "high", "🟢"
    elif best < 0.9:
        return "medium", "🟡"
    else:
        return "low", "🔴"


# Build Prompt

SYSTEM_PROMPT = """You are a helpful AI assistant who is a tutor for students learning AI and other software engineering skills.
Answer the user's query or prompts based ONLY on the context provided below.

Rules:
- Use only information provided in CONTEXT section below.
- If the context does not contain the answer, respond with 'I'm sorry, I don't have enough information to answer that. Please try a different query that falls within the content of the available documents.
- Cite your sources clearly by mentioning the document name in parenthases. 
- Keep your responses concise, under 200 words.
- Use clear analogies when applicaple to help your student understand the content better.
- NEVER make up information. If you do not have contextual information, respond with "I don't know the answer to that question, or use one of the hardcoded guardrail responses below.
"""


def build_message(question, retrieved_chunks):
    """Create the system and user messages used to ask the LLM a grounded question"""
    context_parts = []
    for chunk in retrieved_chunks:
        source = chunk["metadata"]["source"]
        context_parts.append(
            f"[Source: {source}]\n{chunk['text']}"
        )  # Display which chunks were retrieved before showing the answer

    context = "\n\n---\n\n".join(context_parts)
    system_with_context = f"{SYSTEM_PROMPT}\nCONTEXT:\n{context}"

    return [
        {"role": "system", "content": system_with_context},
        {"role": "user", "content": question},
    ]


# Generate Response


def generate(messages, stream=False):
    """Send the prompt to Ollama and return the model's generated answer."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": MODEL, "messages": messages, "stream": stream},
            stream=stream,
        )

        if stream:
            full_response = ""  # Collect the answer as it streams in
            for (
                line
            ) in (
                response.iter_lines()
            ):  # Read Ollama's streamed JSON response line by line
                if line:
                    data = json.loads(line)
                    token = data.get("message", {}).get(
                        "content", ""
                    )  # Pull out the newest text chunk
                    if token:
                        print(
                            token, end="", flush=True
                        )  # Print each token in real time as it arrives
                        full_response += token
            print()  # Add a newline after the final chunk is printed
            return full_response  # Return the completed answer
        else:
            return response.json()["message"][
                "content"
            ]  # Return the full response in one object

    except requests.exceptions.ConnectionError:
        return "ERROR: Cannot connect to Ollama. Is it running? (ollama serve)"


# Updated rag_query with guardrails built in
def rag_query(collection, question, n_results=3, stream=True):
    """Run the full RAG flow: retrieve relevant chunks, build the prompt, and generate an answer."""
    print(f"\n{'='*60}\nQuestion: {question}\n{'='*60}")

    chunks = safe_retrieve(collection, question, n_results, max_distance=1.2)
    confidence, icon = get_confidence(chunks)
    print(f"\n  Confidence: {icon} {confidence}")
    print(f"  Chunks retrieved: {len(chunks)}")

    if not chunks:
        msg = "I don't have relevant information to answer that."
        print(f"  Answer: {msg}")
        return {
            "answer": msg,
            "sources": [],
            "confidence": confidence,
            "chunks_retrieved": 0,
        }

    for c in chunks:
        preview = " ".join(c["text"].split())[:60]
        print(f"    [{c['distance']:.3f}] {c['metadata']['source']}: {preview}...")

    if confidence == "low":
        msg = (
            "I found some loosely related information, but I'm not confident "
            "it answers your question. Try rephrasing or asking about a topic "
            "covered in the documents."
        )
        print(f"  Answer: {msg}")
        return {
            "answer": msg,
            "sources": sorted({c["metadata"]["source"] for c in chunks}),
            "confidence": confidence,
            "chunks_retrieved": len(chunks),
        }

    messages = build_message(question, chunks)
    print("\nAnswer (streaming):\n")
    start = time.time()
    answer = generate(messages, stream=stream)
    print(f"\n[Generated in {time.time() - start:.1f}s]")
    return {
        "answer": answer,
        "sources": sorted({c["metadata"]["source"] for c in chunks}),
        "confidence": confidence,
        "chunks_retrieved": len(chunks),
    }


test_queries = [
    # In-scope: directly covered by sql-fundamentals.md
    "What is the difference between a primary key and a foreign key in SQL?",
    # Partially in-scope: docs cover FastAPI and SQLAlchemy separately, not deployment
    "How do I deploy a FastAPI app with SQLAlchemy to production on AWS?",
    # Out-of-scope: nothing in the docs relates to this
    "What is the capital of Australia?",
    # Ambiguous: could match Python, SQL, or REST docs
    "How do I handle errors?",
]


def run_tests(collection):
    """Run each test query through the guardrailed pipeline and print a summary."""
    for query in test_queries:
        result = rag_query(collection, query)
        print(
            f"\n  Summary: confidence={result['confidence']}, "
            f"chunks={result['chunks_retrieved']}, sources={result['sources']}"
        )


# --- Main ---
if __name__ == "__main__":
    # Create a persistent ChromaDB database stored in the local ./rag_db folder
    client = chromadb.PersistentClient(path="./rag_db")
    collection = client.get_or_create_collection("course_docs")

    # Ingest documents only if the collection is empty, otherwise reuse the saved data
    if collection.count() == 0:
        count = ingest(collection, "docs")
        print(f"Ingested {count} document chunks.")
    else:
        print(f"Using existing collection: {collection.count()} chunks")

    # Interactive loop for asking questions against the stored documents
    print("\nRAG Pipeline Ready! Ask questions about your documents.")
    print("Type 'test' to run test.\n")
    print("Type 'quit' to exit.\n")

    # An interactive loop (keep asking until the user types "quit")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        elif question.lower() in ("test", "run test"):
            run_tests(collection)
        elif question:
            rag_query(collection, question)
