import os
import time
import math
import functools
import json
import uuid
import shutil
from pathlib import Path
from typing import TypedDict,List,Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph,START,END

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import AnswerRelevancy,ContextPrecision,Faithfulness

load_dotenv()

faithfulness = Faithfulness()
answer_relevancy = AnswerRelevancy(strictness = 1)
context_precision = ContextPrecision()

model = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0,
    max_tokens = 256,
)

llm2forRAGAS = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0,
    max_tokens = 4096,
)


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

ragas_llm = LangchainLLMWrapper(llm2forRAGAS)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 100,
    "top_k": 3,
    "eval_threshold": 0.6,
    "max_retries": 2,
    "index_path": "faiss_index",
}


class RAGState(TypedDict):
    original_query: str
    query: str
    retrieved_docs: List[str]
    answer: str
    ground_truth: Optional[str]
    eval_score: dict
    retry_count: int
    latency_log: dict


# 
# Latency instrumentation 
def track_latency(node_name:str):
    """Decorator that logs wall - clock time for a node into state['latency_log']. """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state:RAGState) -> RAGState:
            start = time.perf_counter()
            result = fn(state)
            elapsed = time.perf_counter() - start
            log = dict(result.get("latency_log",state.get("latency_log",{})))
            log.setdefault(node_name,[]).append(round(elapsed, 4))
            result["latency_log"] = log
            return result
        return wrapper
    return decorator


# 
# Ingestion (run once, offline - not a graph node) 
def load_documents(file_paths:List[str]) -> List:
    docs = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext in (".txt",".md"):
            loader = TextLoader(path,encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type:{ext}")
        docs.extend(loader.load())
    return docs

def build_index(file_paths:List[str],index_path:str = CONFIG["index_path"]) -> FAISS:
    """ Load documents,chunk,embed and persist a FAISS index to disk. """
    raw_docs = load_documents(file_paths)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
    )

    chunks = splitter.split_documents(raw_docs)
    if not chunks:
        raise ValueError("No extractable text found in the uploaded file(s).")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def load_index(index_path:str = CONFIG["index_path"]) -> FAISS:
    """ Load a FAISS index from disk. """
    return FAISS.load_local(index_path, embeddings,allow_dangerous_deserialization=True)


# 
#Nodes

_vectorstore:Optional[FAISS]

def init_pipeline(index_path:str = CONFIG["index_path"]):
    """ Call once before invoking the graph - loads the FAISS index into memory. """
    global _vectorstore
    _vectorstore = load_index(index_path)

@track_latency("retrieve")
def retrieve_node(state:RAGState) -> RAGState:
    if _vectorstore is None:
        raise RuntimeError("call init_pipeline() before running the graph.")
    docs = _vectorstore.similarity_search(state["query"],k=CONFIG["top_k"])
    return {**state,"retrieved_docs": [d.page_content for d in docs]}

@track_latency("generate")
def generate_node(state:RAGState) -> RAGState:
    context = "\n\n".join(state["retrieved_docs"])
    prompt = (
        "Answer the question using ONLY the context below."
        "If the context does not contain the answer, say so explicitly.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['query']}\n\nAnswer:"
    )
    response = model.invoke(prompt)
    return {**state,"answer":response.content}

@track_latency("evaluate")
def evaluate_node(state:RAGState) -> RAGState:
    """ Score the answer with RAGAS. context_precision needs ground_truth,
    so it's only included when we're running against the golden dataset. """
    row = {
        "question":[state["original_query"]],
        "answer":[state["answer"]],
        "contexts":[state["retrieved_docs"]],
    }
    
    ds = Dataset.from_dict(row)
    result = evaluate(ds,metrics=[faithfulness],llm=ragas_llm,embeddings=ragas_embeddings)
    scores = {k:round(v,3) for k, v in result.scores[0].items()}
    return {**state,"eval_score":scores}

def rewrite_query_node(state:RAGState) -> RAGState:
    """ Ask the LLM to reformulate the query for better retrieval using the 
    weak answer as a signal of what information was missing.  """
    prompt = (
        f"The original question was : '{state['original_query']}'\n"
        f"The retrieved context was insufficient to answer it well.\n"
        f"Rewrite the question to be more specific and likely to retrieve"
        f"the right passage. Return ONLY the written question."
    )
    response = model.invoke(prompt)
    new_query = response.content.strip()

    if not new_query:
        new_query = state["original_query"] 
    return {**state,"query":new_query,"retry_count":state["retry_count"] + 1}

# Conditional edge logic 
def should_retry(state:RAGState) -> str:
    faithfulness_score = state["eval_score"].get("faithfulness",1.0)
    is_invalid = faithfulness_score is None or (isinstance(faithfulness_score,float)and math.isnan(faithfulness_score))
    if (is_invalid or faithfulness_score < CONFIG["eval_threshold"]) and state["retry_count"] < CONFIG["max_retries"]:
        return "retry"
    return "finalize"

# 
# graph assembly

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve",retrieve_node)
    graph.add_node("generate",generate_node)
    graph.add_node("evaluate",evaluate_node)
    graph.add_node("rewrite_query",rewrite_query_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve","generate")
    graph.add_edge("generate","evaluate")

    graph.add_conditional_edges("evaluate",should_retry,{"retry":"rewrite_query","finalize":END})
    graph.add_edge("rewrite_query","retrieve")
    
    return graph.compile()

# 
# single query entry point

def run_query(question:str,ground_truth:Optional[str] = None) -> RAGState:
    app = build_graph()
    initial_state: RAGState = {
        "original_query":question,
        "query":question,
        "retrieved_docs":[],
        "answer":"",
        "ground_truth":ground_truth,
        "eval_score":{},
        "retry_count":0,
        "latency_log":{},
    }
    return app.invoke(initial_state)

# Only use below block when run from python backend.py 

# if __name__ == "__main__":
#     # quick manual smoke test - replace with your own doc path before runnning 
#     init_pipeline()
#     result = run_query("what was the company's revenue last year?")
#     print("Answer: ", result["answer"])
#     print("Eval scores: ", result["eval_score"])
#     print("Retries: ", result["retry_count"])
#     print("Latency: ", result["latency_log"])


# ---------------------------------------------------------------------------
# Session management (append-only addition — do not modify code above this line)
# ---------------------------------------------------------------------------

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)
SESSIONS_INDEX_FILE = SESSIONS_DIR / "sessions.json"
_vectorstore_cache: dict[str, FAISS] = {}


def _load_sessions_meta() -> dict:
    if SESSIONS_INDEX_FILE.exists():
        return json.loads(SESSIONS_INDEX_FILE.read_text())
    return {}


def _save_sessions_meta(meta: dict):
    SESSIONS_INDEX_FILE.write_text(json.dumps(meta, indent=2))


def create_session(title: str = "New chat") -> str:
    session_id = str(uuid.uuid4())[:8]
    meta = _load_sessions_meta()
    meta[session_id] = {"title": title, "has_document": False, "created_at": time.time()}
    _save_sessions_meta(meta)
    _save_messages(session_id, [])
    return session_id


def list_sessions() -> list[dict]:
    meta = _load_sessions_meta()
    return [{"session_id": sid, **info} for sid, info in
            sorted(meta.items(), key=lambda x: x[1]["created_at"], reverse=True)]


def _messages_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}_messages.json"


def _save_messages(session_id: str, messages: list[dict]):
    _messages_path(session_id).write_text(json.dumps(messages, indent=2))


def load_messages(session_id: str) -> list[dict]:
    path = _messages_path(session_id)
    return json.loads(path.read_text()) if path.exists() else []


def append_message(session_id: str, role: str, content: str):
    messages = load_messages(session_id)
    messages.append({"role": role, "content": content})
    _save_messages(session_id, messages)


def rename_session(session_id: str, title: str):
    meta = _load_sessions_meta()
    if session_id in meta:
        meta[session_id]["title"] = title
        _save_sessions_meta(meta)


def delete_session(session_id: str):
    meta = _load_sessions_meta()
    meta.pop(session_id, None)
    _save_sessions_meta(meta)
    _messages_path(session_id).unlink(missing_ok=True)
    idx_path = SESSIONS_DIR / f"{session_id}_faiss"
    if idx_path.exists():
        shutil.rmtree(idx_path)
    _vectorstore_cache.pop(session_id, None)


def attach_document_to_session(session_id: str, file_bytes: bytes, filename: str):
    import tempfile
    suffix = ".pdf" if filename.lower().endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    idx_path = str(SESSIONS_DIR / f"{session_id}_faiss")
    vectorstore = build_index([tmp_path], index_path=idx_path)
    os.remove(tmp_path)
    _vectorstore_cache[session_id] = vectorstore

    meta = _load_sessions_meta()
    if session_id in meta:
        meta[session_id]["has_document"] = True
        _save_sessions_meta(meta)


def _get_session_vectorstore(session_id: str):
    if session_id in _vectorstore_cache:
        return _vectorstore_cache[session_id]
    idx_path = SESSIONS_DIR / f"{session_id}_faiss"
    if idx_path.exists():
        vs = load_index(str(idx_path))
        _vectorstore_cache[session_id] = vs
        return vs
    return None


def send_message(session_id: str, question: str) -> dict:
    """RAG-only — requires a document to be attached to this session first."""
    global _vectorstore
    vs = _get_session_vectorstore(session_id)
    if vs is None:
        raise ValueError("No document attached to this session yet.")

    append_message(session_id, "user", question)
    _vectorstore = vs
    result = run_query(question)
    answer = result["answer"]
    append_message(session_id, "assistant", answer)

    return {
        "answer": answer,
        "faithfulness": result["eval_score"].get("faithfulness"),
        "retries": result["retry_count"],
    }