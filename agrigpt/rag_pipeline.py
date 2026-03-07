import os
import sys

# ── Mac fix: stops tokenizer parallelism warning ──────
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

# ── Load .env from project root regardless of where script is run from ──
script_dir   = os.path.dirname(os.path.abspath(__file__))   # agrigpt/
project_root = os.path.dirname(script_dir)                  # voc-agrigpt/
sys.path.insert(0, project_root)

dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ── Absolute paths (Mac needs these to avoid "file not found" errors) ──
KNOWLEDGE_DIR   = os.path.join(project_root, "data", "knowledge")
VECTORSTORE_DIR = os.path.join(project_root, "models", "agri_vectorstore")


# ══════════════════════════════════════════════════════
# PART 1 — Build vector store (run ONCE to create DB)
# ══════════════════════════════════════════════════════

def build_vectorstore():
    print(f"Loading knowledge files from: {KNOWLEDGE_DIR}")

    # ── 1. Load all .txt knowledge files ─────────────
    loader = DirectoryLoader(
        KNOWLEDGE_DIR,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} knowledge files")

    # ── 2. Split into small searchable chunks ─────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} searchable chunks")

    # ── 3. Create embeddings (free, runs locally) ─────
    print("Downloading embedding model (first time only, ~80MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},       # Mac: use cpu (works on all Macs)
        encode_kwargs={"normalize_embeddings": True}
    )

    # ── 4. Build and save FAISS vector store ──────────
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"✅ Vector store saved to {VECTORSTORE_DIR}")


# ══════════════════════════════════════════════════════
# PART 2 — AgriGPT answer function (used by UI)
# ══════════════════════════════════════════════════════

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_vectorstore():
    return FAISS.load_local(
        VECTORSTORE_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

def agrigpt_answer(question: str, disease_context: str = "") -> str:
    # ── 1. Search knowledge base ──────────────────────
    vectorstore   = get_vectorstore()
    retriever     = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)
    context       = "\n\n".join([doc.page_content for doc in relevant_docs])

    # ── 2. Inject disease from VOC model if provided ──
    disease_note = ""
    if disease_context:
        disease_note = f"The VOC sensor has detected: {disease_context}\n\n"

    # ── 3. Build prompt ────────────────────────────────
    prompt = f"""You are AgriGPT, a knowledgeable and friendly agriculture assistant
for Indian farmers. Answer clearly and practically in simple English.
Always give specific dosages, timings, and actionable steps.

{disease_note}Relevant Knowledge:
{context}

Farmer's Question: {question}

Answer:"""

    # ── 4. Call Groq LLM ──────────────────────────────
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "❌ GROQ_API_KEY not found. Please check your .env file."

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.3
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════════════
# Run directly to build DB + test
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    # Step 1 — build the vector store
    build_vectorstore()

    # Step 2 — test 3 sample questions
    print("\n── Test 1: Disease question ──")
    print(agrigpt_answer("What pesticide should I use for wheat rust?"))

    print("\n── Test 2: Government scheme ──")
    print(agrigpt_answer("How do I apply for PM-KISAN?"))

    print("\n── Test 3: VOC model integration ──")
    print(agrigpt_answer(
        question="What treatment is needed?",
        disease_context="Rust on Wheat with 100.0% confidence"
    ))