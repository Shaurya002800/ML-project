#!/usr/bin/env python3
import os
import sys

# Reduce tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
load_dotenv(os.path.join(project_root, ".env"))

# Guard optional/host-heavy imports so the module can be imported in
# constrained deployment environments. If a package is missing we set the
# symbol to None and surface a clear error only when the feature is actually
# used at runtime.
try:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    _HAS_LC_COMM_DOC = True
except Exception:
    try:
        from langchain.document_loaders import DirectoryLoader, TextLoader
        _HAS_LC_COMM_DOC = True
    except Exception:
        DirectoryLoader = None
        TextLoader = None
        _HAS_LC_COMM_DOC = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except Exception:
        HuggingFaceEmbeddings = None

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    try:
        from langchain.vectorstores import FAISS
    except Exception:
        FAISS = None

try:
    from groq import Groq
except Exception:
    Groq = None

KNOWLEDGE_DIR = os.path.join(project_root, "data", "knowledge")
VECTORSTORE_DIR = os.path.join(project_root, "models", "agri_vectorstore")

_embeddings = None
_vectorstore = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # Prefer LangChain's HuggingFaceEmbeddings when available
        if HuggingFaceEmbeddings is not None:
            _embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
        else:
            # Fallback: try to use sentence-transformers directly if installed
            try:
                from sentence_transformers import SentenceTransformer

                class _STWrapper:
                    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
                        self._m = SentenceTransformer(model_name)

                    def embed_documents(self, texts):
                        return [list(x) if hasattr(x, '__iter__') else x for x in self._m.encode(texts, convert_to_numpy=True)]

                    def embed_query(self, text):
                        return list(self._m.encode(text))

                _embeddings = _STWrapper("all-MiniLM-L6-v2")
            except Exception:
                raise RuntimeError(
                    "No embedding backend available. Install 'langchain_huggingface' or 'sentence-transformers'."
                )
    return _embeddings


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        idx = os.path.join(VECTORSTORE_DIR, "index.faiss")
        if not os.path.exists(idx):
            raise FileNotFoundError("Vectorstore not found; run build_vectorstore()")
        if FAISS is None:
            raise RuntimeError(
                "FAISS vectorstore backend not available. Install 'langchain_community' or 'langchain' with vectorstores support."
            )
        _vectorstore = FAISS.load_local(
            VECTORSTORE_DIR, get_embeddings(), allow_dangerous_deserialization=True
        )
    return _vectorstore


def build_vectorstore():
    if not _HAS_LC_COMM_DOC or DirectoryLoader is None or TextLoader is None:
        raise RuntimeError(
            "Document loader backend not available. Install 'langchain_community' or 'langchain' with document_loaders support."
        )
    if RecursiveCharacterTextSplitter is None:
        raise RuntimeError(
            "Text splitter backend missing. Install 'langchain_text_splitters' or the 'langchain' package."
        )
    if FAISS is None:
        raise RuntimeError(
            "FAISS vectorstore backend not available. Install 'langchain_community' or 'langchain' with vectorstores support."
        )

    loader = DirectoryLoader(
        KNOWLEDGE_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, get_embeddings())
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vs.save_local(VECTORSTORE_DIR)
    return VECTORSTORE_DIR


def agrigpt_answer(question: str, disease_context: str = "") -> str:
    vs = get_vectorstore()

    # retrieval compatibility
    relevant_docs = []
    if hasattr(vs, "similarity_search"):
        try:
            relevant_docs = vs.similarity_search(question, k=4)
        except TypeError:
            relevant_docs = vs.similarity_search(question, 4)
    elif hasattr(vs, "similarity_search_with_score"):
        res = vs.similarity_search_with_score(question, k=4)
        relevant_docs = [r[0] for r in res]
    else:
        retriever = vs.as_retriever(search_kwargs={"k": 4})
        for name in ("get_relevant_documents", "retrieve", "invoke"):
            fn = getattr(retriever, name, None)
            if callable(fn):
                relevant_docs = fn(question)
                break

    context = "\n\n".join(getattr(d, "page_content", str(d)) for d in relevant_docs)
    disease_note = f"VOC sensor detected disease: {disease_context}\n\n" if disease_context else ""

    prompt = f"""
You are AgriGPT, an agriculture assistant helping Indian farmers.

Provide clear practical answers with:
- exact pesticide name
- dosage
- timing
- prevention tips

{disease_note}

Knowledge Base:
{context}

Farmer Question:
{question}

Answer:
"""

    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(model="llama-3.3-70b-versatile",
                                                  messages=[{"role": "user", "content": prompt}],
                                                  max_tokens=600,
                                                  temperature=0.3)
            return getattr(resp.choices[0].message, "content", str(resp))
        except Exception as e:
            print(f"\n[DEBUG] Groq API Error: {e}\n")
            pass

    snippets = [getattr(d, "page_content", str(d)).strip() for d in relevant_docs[:3]]
    if snippets:
        return "[AgriGPT unavailable — returning relevant KB]\n\n" + "\n\n---\n\n".join(snippets)

    return "AgriGPT unavailable and no local knowledge found."


if __name__ == "__main__":
    try:
        idx = os.path.join(VECTORSTORE_DIR, "index.faiss")
        if not os.path.exists(idx):
            build_vectorstore()
    except Exception as e:
        print("Could not build vectorstore:", e)

    print(agrigpt_answer("What pesticide should be used for wheat rust?"))
