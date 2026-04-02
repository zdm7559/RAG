from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.rag_qdrant import (
    COLLECTION_NAME,
    DATA_DIR,
    add_text_to_qdrant,
    ask,
    collection_count,
    delete_document,
    ensure_collection,
    index_documents,
    list_chunks,
    list_documents,
)
from app.schemas import (
    AddTextRequest,
    AddTextResponse,
    ChunkListResponse,
    DeleteDocumentResponse,
    DocumentListResponse,
    IndexRequest,
    IndexResponse,
    RAGRequest,
    RAGResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("服务启动中，准备初始化 Qdrant...")
    ensure_collection()
    print(f"Qdrant 已就绪，当前集合: {COLLECTION_NAME}，已有 {collection_count()} 条数据")
    yield
    print("服务关闭")


app = FastAPI(
    title="Mini RAG API with Qdrant",
    description="使用 Qdrant 的最小版 RAG 接口",
    version="1.5.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/rag/chat", response_model=RAGResponse)
def rag_chat(request: RAGRequest):
    return ask(
        query=request.query,
        retrieve_top_n=request.retrieve_top_n,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        max_context_chars=request.max_context_chars,
        enable_rerank=request.enable_rerank,
        rerank_top_n=request.rerank_top_n,
        rerank_threshold=request.rerank_threshold,
    )


@app.post("/rag/index", response_model=IndexResponse)
def rag_index():
    indexed_chunks = index_documents(DATA_DIR)
    return {
        "message": "文档目录扫描并入库完成（默认 fixed 切块）",
        "collection_name": COLLECTION_NAME,
        "indexed_chunks": indexed_chunks,
    }


@app.post("/rag/index/configurable", response_model=IndexResponse)
def rag_index_configurable(request: IndexRequest):
    indexed_chunks = index_documents(
        DATA_DIR,
        chunk_strategy=request.chunk_strategy,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
        max_chunk_size=request.max_chunk_size,
    )
    return {
        "message": "文档目录扫描并入库完成",
        "collection_name": COLLECTION_NAME,
        "indexed_chunks": indexed_chunks,
    }


@app.post("/rag/add_text", response_model=AddTextResponse)
def rag_add_text(request: AddTextRequest):
    indexed_chunks = add_text_to_qdrant(
        source=request.source,
        text=request.text,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
        chunk_strategy=request.chunk_strategy,
        max_chunk_size=request.max_chunk_size,
    )
    return {
        "message": "文本已加入知识库",
        "source": request.source,
        "indexed_chunks": indexed_chunks,
    }


@app.get("/documents", response_model=DocumentListResponse)
def get_documents():
    documents = list_documents()
    return {"total_documents": len(documents), "documents": documents}


@app.get("/documents/{document_id}/chunks", response_model=ChunkListResponse)
def get_document_chunks(document_id: str):
    chunks = list_chunks(document_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="document_id 不存在或没有对应 chunk")
    return {"document_id": document_id, "total_chunks": len(chunks), "chunks": chunks}


@app.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document_by_id(document_id: str):
    deleted_chunks = delete_document(document_id)
    if deleted_chunks == 0:
        raise HTTPException(status_code=404, detail="document_id 不存在或没有对应 chunk")
    return {
        "message": "文档及其关联 chunk 已从知识库删除",
        "document_id": document_id,
        "deleted_chunks": deleted_chunks,
    }
