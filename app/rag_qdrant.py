import hashlib
import os
import re
from typing import Any, Dict, List, Optional

import torch
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from app.utils import Document, list_supported_files, read_file

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_EMBEDDING_MODEL_PATH = "/home/zhaodongmin/RAG/bge-small-en-v1.5"
QDRANT_PATH = os.path.join(BASE_DIR, "storage", "qdrant_data")
COLLECTION_NAME = "mini_rag_docs"
DEFAULT_RERANK_MODEL_CANDIDATES = [
    "/home/zhaodongmin/RAG/qwen3-reranker-0.6b",
    "/home/zhaodongmin/RAG/qwen3-reranker-0.6b",
    "/home/zhaodongmin/llm-app-learning/week5_rag/mini_rag_api/bce-reranker-base_v1",
]

CHAT_MODEL = os.getenv("CHAT_MODEL")
MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH") or DEFAULT_EMBEDDING_MODEL_PATH
DEFAULT_RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
DEFAULT_RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.60"))
RERANK_MAX_DOC_CHARS = int(os.getenv("RERANK_MAX_DOC_CHARS", "700"))
QWEN3_RERANK_MAX_LENGTH = int(os.getenv("QWEN3_RERANK_MAX_LENGTH", "4096"))
DEFAULT_RERANK_INSTRUCTION = os.getenv(
    "RERANK_INSTRUCTION",
    "Given a web search query, retrieve relevant passages that answer the query",
)

client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("MOONSHOT_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
)

_embedding_model = None
_qdrant_client = None
_rerank_tokenizer = None
_rerank_model = None
_rerank_mode = None
_rerank_true_token_id = None
_rerank_false_token_id = None
_rerank_prefix_tokens = None
_rerank_suffix_tokens = None


def _resolve_rerank_model_path() -> str:
    explicit_path = os.getenv("RERANK_MODEL_PATH")
    if explicit_path:
        return explicit_path

    for candidate in DEFAULT_RERANK_MODEL_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate

    return DEFAULT_RERANK_MODEL_CANDIDATES[0]


RERANK_MODEL_PATH = _resolve_rerank_model_path()


def _clip_text(text: str, max_chars: int = RERANK_MAX_DOC_CHARS) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _format_qwen3_rerank_text(query: str, doc: str, instruction: str = DEFAULT_RERANK_INSTRUCTION) -> str:
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction,
        query=query,
        doc=doc,
    )


def _get_effective_score(doc: Dict[str, Any]) -> float:
    if doc.get("rerank_score") is not None:
        return float(doc["rerank_score"])
    if doc.get("embedding_score") is not None:
        return float(doc["embedding_score"])
    return float(doc.get("score", 0.0))


def _sync_doc_score(doc: Dict[str, Any]) -> Dict[str, Any]:
    doc["score"] = _get_effective_score(doc)
    return doc


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("开始加载 embedding 模型...")
        print(f"模型路径: {MODEL_PATH}")
        _embedding_model = SentenceTransformer(MODEL_PATH)
        print("embedding 模型加载完成")
    return _embedding_model


def get_rerank_components():
    global _rerank_tokenizer, _rerank_model, _rerank_mode
    global _rerank_true_token_id, _rerank_false_token_id, _rerank_prefix_tokens, _rerank_suffix_tokens

    if _rerank_tokenizer is None or _rerank_model is None or _rerank_mode is None:
        if not os.path.isdir(RERANK_MODEL_PATH):
            raise FileNotFoundError(f"本地 rerank 模型目录不存在: {RERANK_MODEL_PATH}")

        print("开始加载 rerank 模型...")
        print(f"模型路径: {RERANK_MODEL_PATH}")

        model_name = os.path.basename(RERANK_MODEL_PATH).lower()
        if "qwen3-reranker" in model_name:
            _rerank_mode = "qwen3"
            _rerank_tokenizer = AutoTokenizer.from_pretrained(
                RERANK_MODEL_PATH,
                padding_side="left",
                trust_remote_code=True,
            )
            _rerank_model = AutoModelForCausalLM.from_pretrained(
                RERANK_MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            if _rerank_tokenizer.pad_token_id is None:
                _rerank_tokenizer.pad_token = _rerank_tokenizer.eos_token

            _rerank_true_token_id = _rerank_tokenizer.convert_tokens_to_ids("yes")
            _rerank_false_token_id = _rerank_tokenizer.convert_tokens_to_ids("no")
            if _rerank_true_token_id is None or _rerank_false_token_id is None:
                raise ValueError("Qwen3 reranker tokenizer 缺少 yes/no token id")

            _rerank_prefix_tokens = _rerank_tokenizer(
                '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
                add_special_tokens=False,
            )["input_ids"]
            _rerank_suffix_tokens = _rerank_tokenizer(
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                add_special_tokens=False,
            )["input_ids"]
        else:
            _rerank_mode = "sequence_classification"
            _rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_PATH)
            _rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_PATH)

        _rerank_model.eval()
        print("rerank 模型加载完成")

    return _rerank_tokenizer, _rerank_model, _rerank_mode


def _compute_qwen3_rerank_scores(
    tokenizer,
    rerank_model,
    query: str,
    docs_to_rerank: List[Dict[str, Any]],
) -> List[float]:
    prompt_texts = [
        _format_qwen3_rerank_text(query=query, doc=_clip_text(doc.get("text", "")))
        for doc in docs_to_rerank
    ]
    max_payload_length = max(1, QWEN3_RERANK_MAX_LENGTH - len(_rerank_prefix_tokens) - len(_rerank_suffix_tokens))
    encoded_payloads = tokenizer(
        prompt_texts,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_payload_length,
        add_special_tokens=False,
    )

    batch_inputs = []
    for input_ids in encoded_payloads["input_ids"]:
        batch_inputs.append(_rerank_prefix_tokens + input_ids + _rerank_suffix_tokens)

    batch = tokenizer.pad(
        {"input_ids": batch_inputs},
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
        max_length=QWEN3_RERANK_MAX_LENGTH,
    )
    batch = {key: value.to(rerank_model.device) for key, value in batch.items()}

    with torch.no_grad():
        outputs = rerank_model(**batch)
        logits = outputs.logits[:, -1, :]
        pair_logits = logits[:, [_rerank_false_token_id, _rerank_true_token_id]]
        probs = torch.softmax(pair_logits, dim=-1)[:, 1]

    return probs.detach().cpu().tolist()


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        os.makedirs(QDRANT_PATH, exist_ok=True)
        print(f"正在初始化 Qdrant 本地库: {QDRANT_PATH}")
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
        print("Qdrant 初始化完成")
    return _qdrant_client


def chunk_text_fixed(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    text = text.strip()
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def chunk_text_by_paragraph(text: str, max_chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be > 0")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para

        if len(candidate) <= max_chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(para) <= max_chunk_size:
            current = para
        else:
            sub_chunks = chunk_text_fixed(para, chunk_size=max_chunk_size, overlap=overlap)
            chunks.extend(sub_chunks[:-1])
            current = sub_chunks[-1]

    if current:
        chunks.append(current)

    return chunks


def chunk_text(
    text: str,
    strategy: str = "fixed",
    chunk_size: int = 300,
    overlap: int = 50,
    max_chunk_size: int = 500,
) -> list[str]:
    if strategy == "fixed":
        return chunk_text_fixed(text, chunk_size=chunk_size, overlap=overlap)
    if strategy == "paragraph":
        return chunk_text_by_paragraph(text, max_chunk_size=max_chunk_size, overlap=overlap)
    raise ValueError(f"unsupported chunk strategy: {strategy}")


def make_document_id(source: str) -> str:
    return "doc_" + hashlib.md5(source.encode("utf-8")).hexdigest()[:12]


def make_point_id(source: str, chunk_id: int) -> int:
    raw = f"{source}::{chunk_id}"
    hex_digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return int(hex_digest[:15], 16)


def extract_page_from_chunk(text: str) -> Optional[int]:
    match = re.search(r"\[Page\s+(\d+)\]", text)
    if match:
        return int(match.group(1))
    return None


def infer_metadata_from_document(doc: Document) -> dict:
    return {
        "document_id": make_document_id(doc.source),
        "filename": doc.metadata.get("filename", os.path.basename(doc.source)),
        "title": doc.metadata.get("title", os.path.basename(doc.source)),
        "page_count": doc.metadata.get("page_count"),
    }


def _make_doc_record(
    document: Document,
    chunk: str,
    chunk_id: int,
    chunk_strategy: str,
    document_metadata: dict,
    paragraph_index: Optional[int] = None,
) -> dict:
    return {
        "document_id": document_metadata["document_id"],
        "filename": document_metadata["filename"],
        "title": document_metadata["title"],
        "source": document.source,
        "chunk_id": chunk_id,
        "text": chunk,
        "file_type": document.file_type,
        "chunk_strategy": chunk_strategy,
        "page": extract_page_from_chunk(chunk),
        "paragraph_index": paragraph_index,
        "page_count": document_metadata.get("page_count"),
    }


def load_documents(
    data_dir: str,
    chunk_strategy: str = "fixed",
    chunk_size: int = 300,
    overlap: int = 50,
    max_chunk_size: int = 500,
) -> list[dict]:
    docs = []
    file_paths = list_supported_files(data_dir)

    for file_path in file_paths:
        document = read_file(file_path)
        metadata = infer_metadata_from_document(document)
        chunks = chunk_text(
            document.text,
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            max_chunk_size=max_chunk_size,
        )

        for i, chunk in enumerate(chunks):
            paragraph_index = i if chunk_strategy == "paragraph" else None
            docs.append(
                _make_doc_record(
                    document=document,
                    chunk=chunk,
                    chunk_id=i,
                    chunk_strategy=chunk_strategy,
                    document_metadata=metadata,
                    paragraph_index=paragraph_index,
                )
            )

    return docs


def get_embedding(text: str) -> list[float]:
    model = get_embedding_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def get_embedding_dim() -> int:
    model = get_embedding_model()
    sample = model.encode("test", normalize_embeddings=True)
    return len(sample)


def ensure_collection() -> None:
    qdrant = get_qdrant_client()
    collections = qdrant.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME in collection_names:
        return

    vector_dim = get_embedding_dim()
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    print(f"集合创建完成: {COLLECTION_NAME}")


def collection_count() -> int:
    qdrant = get_qdrant_client()
    ensure_collection()
    result = qdrant.count(collection_name=COLLECTION_NAME, exact=True)
    return result.count


def upsert_docs(docs: list[dict]) -> int:
    if not docs:
        return 0

    ensure_collection()
    qdrant = get_qdrant_client()

    points: List[PointStruct] = []
    for i, doc in enumerate(docs, start=1):
        print(f"正在处理第 {i}/{len(docs)} 个 chunk，来源: {doc['source']}")
        embedding = get_embedding(doc["text"])

        point = PointStruct(
            id=make_point_id(doc["source"], doc["chunk_id"]),
            vector=embedding,
            payload={
                "document_id": doc.get("document_id"),
                "filename": doc.get("filename"),
                "title": doc.get("title"),
                "source": doc["source"],
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "file_type": doc.get("file_type", "unknown"),
                "chunk_strategy": doc.get("chunk_strategy", "fixed"),
                "page": doc.get("page"),
                "paragraph_index": doc.get("paragraph_index"),
                "page_count": doc.get("page_count"),
            },
        )
        points.append(point)

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def index_documents(
    data_dir: str,
    chunk_strategy: str = "paragraph",
    chunk_size: int = 300,
    overlap: int = 50,
    max_chunk_size: int = 500,
) -> int:
    docs = load_documents(
        data_dir,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunk_size=max_chunk_size,
    )
    print(f"扫描到 {len(docs)} 个 chunks，开始入库...")
    return upsert_docs(docs)


def add_text_to_qdrant(
    source: str,
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
    chunk_strategy: str = "fixed",
    max_chunk_size: int = 500,
) -> int:
    document = Document(
        text=text,
        source=source,
        file_type="text",
        metadata={
            "filename": os.path.basename(source),
            "title": os.path.basename(source) or source,
        },
    )
    metadata = {
        "document_id": make_document_id(source),
        "filename": document.metadata["filename"],
        "title": document.metadata["title"],
        "page_count": None,
    }
    chunks = chunk_text(
        text,
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunk_size=max_chunk_size,
    )
    docs = []

    for i, chunk in enumerate(chunks):
        paragraph_index = i if chunk_strategy == "paragraph" else None
        docs.append(
            _make_doc_record(
                document=document,
                chunk=chunk,
                chunk_id=i,
                chunk_strategy=chunk_strategy,
                document_metadata=metadata,
                paragraph_index=paragraph_index,
            )
        )

    return upsert_docs(docs)


def build_qdrant_filter(
    document_id: Optional[str] = None,
) -> Optional[Filter]:
    conditions = []

    if document_id:
        conditions.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))

    if not conditions:
        return None

    return Filter(must=conditions)


def _scroll_all_points(query_filter: Optional[Filter] = None) -> list:
    qdrant = get_qdrant_client()
    ensure_collection()

    all_points = []
    next_offset = None

    while True:
        points, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=query_filter,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(points)
        if next_offset is None:
            break

    return all_points


def list_documents() -> list[dict]:
    points = _scroll_all_points()
    grouped: dict[str, dict] = {}

    for point in points:
        payload = point.payload or {}
        document_id = payload.get("document_id") or make_document_id(payload.get("source", "unknown"))

        if document_id not in grouped:
            grouped[document_id] = {
                "document_id": document_id,
                "filename": payload.get("filename") or os.path.basename(payload.get("source", "")),
                "source": payload.get("source", ""),
                "file_type": payload.get("file_type", "unknown"),
                "title": payload.get("title"),
                "chunk_count": 0,
                "chunk_strategy": payload.get("chunk_strategy"),
            }

        grouped[document_id]["chunk_count"] += 1

    documents = list(grouped.values())
    documents.sort(key=lambda item: (item["filename"], item["document_id"]))
    return documents


def list_chunks(document_id: str) -> list[dict]:
    points = _scroll_all_points(query_filter=build_qdrant_filter(document_id=document_id))
    chunks = []

    for point in points:
        payload = point.payload or {}
        chunks.append(
            {
                "document_id": payload.get("document_id"),
                "filename": payload.get("filename"),
                "source": payload.get("source", ""),
                "chunk_id": payload.get("chunk_id", -1),
                "text": payload.get("text", ""),
                "file_type": payload.get("file_type"),
                "chunk_strategy": payload.get("chunk_strategy"),
                "title": payload.get("title"),
                "page": payload.get("page"),
                "paragraph_index": payload.get("paragraph_index"),
            }
        )

    chunks.sort(key=lambda item: item["chunk_id"])
    return chunks


def delete_document(document_id: str) -> int:
    qdrant = get_qdrant_client()
    ensure_collection()
    points_before = _scroll_all_points(query_filter=build_qdrant_filter(document_id=document_id))
    deleted_chunks = len(points_before)

    if deleted_chunks == 0:
        return 0

    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(filter=build_qdrant_filter(document_id=document_id)),
        wait=True,
    )
    return deleted_chunks


def retrieve(
    query: str,
    top_k: int = 3,
    fetch_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    qdrant = get_qdrant_client()
    ensure_collection()

    query_embedding = get_embedding(query)
    candidate_limit = fetch_k or max(top_k * 3, top_k)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=candidate_limit,
        with_payload=True,
    ).points

    retrieved_docs = []
    for rank, item in enumerate(results, start=1):
        payload = item.payload or {}
        embedding_score = float(item.score)
        retrieved_docs.append(
            {
                "source": payload.get("source", ""),
                "chunk_id": payload.get("chunk_id", -1),
                "text": payload.get("text", ""),
                "score": embedding_score,
                "embedding_score": embedding_score,
                "rerank_score": None,
                "retrieval_rank": rank,
                "rerank_rank": None,
                "file_type": payload.get("file_type"),
                "chunk_strategy": payload.get("chunk_strategy"),
                "document_id": payload.get("document_id"),
                "filename": payload.get("filename"),
                "title": payload.get("title"),
                "page": payload.get("page"),
                "paragraph_index": payload.get("paragraph_index"),
            }
        )

    return retrieved_docs


def rerank_retrieved_docs(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    rerank_top_n: int,
) -> tuple[List[Dict[str, Any]], bool, str]:
    if not retrieved_docs:
        return [], False, "没有候选 chunk 可供 rerank。"

    top_n = max(1, min(rerank_top_n, len(retrieved_docs)))
    docs_to_rerank = retrieved_docs[:top_n]
    passthrough_docs = retrieved_docs[top_n:]

    try:
        tokenizer, rerank_model, rerank_mode = get_rerank_components()
        if rerank_mode == "qwen3":
            logits = _compute_qwen3_rerank_scores(
                tokenizer=tokenizer,
                rerank_model=rerank_model,
                query=query,
                docs_to_rerank=docs_to_rerank,
            )
        else:
            sentence_pairs = [[query, _clip_text(doc.get("text", ""))] for doc in docs_to_rerank]
            inputs = tokenizer(
                sentence_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = rerank_model(**inputs, return_dict=True)
            logits = outputs.logits.view(-1).float().tolist()

        reranked_docs = []
        for idx, doc in enumerate(docs_to_rerank):
            updated = dict(doc)
            updated["rerank_score"] = float(logits[idx])
            reranked_docs.append(_sync_doc_score(updated))

        reranked_docs.sort(
            key=lambda doc: (
                doc.get("rerank_score", 0.0),
                doc.get("embedding_score", 0.0),
            ),
            reverse=True,
        )

        for rank, doc in enumerate(reranked_docs, start=1):
            doc["rerank_rank"] = rank

        final_docs = reranked_docs + [dict(doc) for doc in passthrough_docs]
        for doc in final_docs[top_n:]:
            _sync_doc_score(doc)

        return final_docs, True, f"已使用本地 rerank 模型对前 {top_n} 个候选 chunk 完成重排。"
    except Exception as exc:
        print(f"rerank 失败，回退到向量排序: {exc}")
        fallback_docs = [dict(doc) for doc in retrieved_docs]
        for doc in fallback_docs:
            _sync_doc_score(doc)
        return fallback_docs, False, f"rerank 执行失败，已回退到向量排序：{exc}"


def filter_retrieved_docs(
    retrieved_docs: List[Dict[str, Any]],
    score_threshold: Optional[float],
) -> List[Dict[str, Any]]:
    if score_threshold is None:
        return retrieved_docs
    return [doc for doc in retrieved_docs if _get_effective_score(doc) >= score_threshold]


def limit_context_docs(retrieved_docs: List[Dict[str, Any]], top_k: int, max_context_chars: int) -> List[Dict[str, Any]]:
    used_docs: List[Dict[str, Any]] = []
    total_chars = 0

    for doc in retrieved_docs:
        text = (doc.get("text") or "").strip()
        if not text:
            continue

        estimated_size = len(text) + 240
        if used_docs and total_chars + estimated_size > max_context_chars:
            break

        used_docs.append(doc)
        total_chars += estimated_size

        if len(used_docs) >= top_k:
            break

    return used_docs


def _format_citation_label(doc: dict, index: int) -> str:
    parts = [f"[{index}]"]
    if doc.get("title"):
        parts.append(str(doc["title"]))
    elif doc.get("filename"):
        parts.append(str(doc["filename"]))
    else:
        parts.append(str(doc.get("source", "未知来源")))

    if doc.get("page") is not None:
        parts.append(f"第{doc['page']}页")
    parts.append(f"chunk_id={doc.get('chunk_id')}")
    parts.append(f"score={doc.get('score', 0.0):.4f}")
    return " | ".join(parts)


def build_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):
        part = (
            f"[引用{i}]\n"
            f"来源: {doc['source']} | 文档ID: {doc.get('document_id')} | 文件名: {doc.get('filename')}\n"
            f"chunk_id: {doc['chunk_id']} | score: {doc['score']:.4f} | 页码: {doc.get('page')}\n"
            f"标题: {doc.get('title')} | 文件类型: {doc.get('file_type')}\n"
            f"切块策略: {doc.get('chunk_strategy')} | 段落号: {doc.get('paragraph_index')}\n"
            f"内容: {doc['text']}\n"
        )
        context_parts.append(part)

    return "\n".join(context_parts)


def build_citation_section(retrieved_docs: List[Dict[str, Any]]) -> str:
    if not retrieved_docs:
        return ""

    lines = ["\n参考来源："]
    for i, doc in enumerate(retrieved_docs, start=1):
        lines.append(_format_citation_label(doc, i))
    return "\n".join(lines)


def generate_answer(query: str, context: str, used_docs: List[Dict[str, Any]]) -> str:
    prompt = f"""你是一个基于知识库回答问题的助手。
请严格依据给定的上下文回答问题，不要补充上下文之外的事实。
如果上下文无法支持答案，请明确说“我无法从给定资料中确定答案”。
如果可以回答，请在对应句子末尾用 [引用1]、[引用2] 这样的格式标注依据。
不要编造不存在的引用编号。

以下是检索到的上下文：
{context}

用户问题：
{query}
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个严谨、清晰的中文知识库问答助手。"},
            {"role": "user", "content": prompt},
        ],
    )

    answer = response.choices[0].message.content or ""
    citation_section = build_citation_section(used_docs)
    return answer.strip() + citation_section


def ask(
    query: str,
    retrieve_top_n: int = 30,
    top_k: int = 5,
    score_threshold: Optional[float] = None,
    max_context_chars: int = 1800,
    enable_rerank: bool = DEFAULT_RERANK_ENABLED,
    rerank_top_n: int = 10,
    rerank_threshold: Optional[float] = None,
) -> dict:
    retrieved_docs = retrieve(
        query=query,
        top_k=top_k,
        fetch_k=retrieve_top_n,
    )

    rerank_applied = False
    rerank_message = "未启用 rerank，使用向量排序结果。"

    if enable_rerank and retrieved_docs:
        retrieved_docs, rerank_applied, rerank_message = rerank_retrieved_docs(
            query=query,
            retrieved_docs=retrieved_docs,
            rerank_top_n=rerank_top_n,
        )
    else:
        retrieved_docs = [_sync_doc_score(dict(doc)) for doc in retrieved_docs]

    effective_threshold = rerank_threshold if enable_rerank and rerank_applied else score_threshold

    reliable_docs = filter_retrieved_docs(retrieved_docs, score_threshold=effective_threshold)
    used_docs = limit_context_docs(reliable_docs, top_k=top_k, max_context_chars=max_context_chars)

    if not used_docs:
        return {
            "query": query,
            "answer": "未找到可靠依据，无法基于当前知识库回答这个问题。",
            "retrieved_docs": retrieved_docs,
            "used_docs": [],
            "reliable": False,
            "rerank_applied": rerank_applied,
            "message": (
                f"{rerank_message} 没有检索结果通过阈值过滤。"
                if effective_threshold is None
                else f"{rerank_message} 没有检索结果通过阈值过滤（threshold={effective_threshold:.2f}）。"
            ),
        }

    context = build_context(used_docs)
    answer = generate_answer(query, context, used_docs)

    return {
        "query": query,
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "used_docs": used_docs,
        "reliable": True,
        "rerank_applied": rerank_applied,
        "message": (
            f"{rerank_message} 共检索到 {len(retrieved_docs)} 个候选 chunk，"
            f"实际使用 {len(used_docs)} 个 chunk 生成答案。"
        ),
    }
