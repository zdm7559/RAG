from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class RetrievedDoc(BaseModel):
    source: str
    chunk_id: int
    text: str
    score: float
    embedding_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrieval_rank: Optional[int] = None
    rerank_rank: Optional[int] = None
    file_type: Optional[str] = None
    chunk_strategy: Optional[str] = None
    document_id: Optional[str] = None
    filename: Optional[str] = None
    title: Optional[str] = None
    page: Optional[int] = None
    paragraph_index: Optional[int] = None


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    retrieve_top_n: int = Field(30, ge=1, le=100, description="第一次召回的候选 chunk 数量")
    top_k: int = Field(5, ge=1, le=20, description="最终用于回答的前几个相关片段")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="召回分数阈值，可选")
    max_context_chars: int = Field(1800, ge=200, le=12000, description="传给模型的最大上下文字符数")
    enable_rerank: bool = Field(True, description="是否启用模型式二次重排")
    rerank_top_n: int = Field(10, ge=1, le=50, description="进入 rerank 的候选 chunk 数量")
    rerank_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="rerank 分数阈值，可选",
    )


class RAGResponse(BaseModel):
    query: str
    answer: str
    retrieved_docs: List[RetrievedDoc]
    used_docs: List[RetrievedDoc]
    reliable: bool
    rerank_applied: bool = False
    message: Optional[str] = None


class IndexRequest(BaseModel):
    chunk_strategy: Literal["fixed", "paragraph"] = Field("fixed", description="切块策略")
    chunk_size: int = Field(300, ge=50, le=2000, description="fixed 策略下的 chunk 大小")
    overlap: int = Field(50, ge=0, le=500, description="相邻 chunk 的重叠长度")
    max_chunk_size: int = Field(500, ge=50, le=3000, description="paragraph 策略下的最大 chunk 大小")


class IndexResponse(BaseModel):
    message: str
    collection_name: str
    indexed_chunks: int


class AddTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="要加入知识库的文本")
    source: str = Field(..., min_length=1, description="文本来源标识")
    chunk_size: int = Field(300, ge=50, le=2000)
    overlap: int = Field(50, ge=0, le=500)
    chunk_strategy: Literal["fixed", "paragraph"] = Field("fixed", description="切块策略")
    max_chunk_size: int = Field(500, ge=50, le=3000)


class AddTextResponse(BaseModel):
    message: str
    source: str
    indexed_chunks: int


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    source: str
    file_type: str
    title: Optional[str] = None
    chunk_count: int
    chunk_strategy: Optional[str] = None


class DocumentListResponse(BaseModel):
    total_documents: int
    documents: List[DocumentSummary]


class ChunkInfo(BaseModel):
    document_id: str
    filename: Optional[str] = None
    source: str
    chunk_id: int
    text: str
    file_type: Optional[str] = None
    chunk_strategy: Optional[str] = None
    title: Optional[str] = None
    page: Optional[int] = None
    paragraph_index: Optional[int] = None


class ChunkListResponse(BaseModel):
    document_id: str
    total_chunks: int
    chunks: List[ChunkInfo]


class DeleteDocumentResponse(BaseModel):
    message: str
    document_id: str
    deleted_chunks: int
