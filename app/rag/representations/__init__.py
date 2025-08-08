"""RAG representations module."""

from .request import (
    DocumentCreateRequest,
    DocumentUpdateRequest,
    RAGFeedbackRequest,
    RAGQueryRequest,
)
from .response import (
    DocumentChunk,
    DocumentResponse,
    DocumentSummary,
    Embedding,
    RAGQuery,
    RAGQueryResponse,
    UploadResponse,
)

__all__ = [
    # Request models
    "DocumentCreateRequest",
    "DocumentUpdateRequest",
    "RAGQueryRequest",
    "RAGFeedbackRequest",
    # Response models
    "DocumentChunk",
    "DocumentResponse",
    "DocumentSummary",
    "Embedding",
    "RAGQuery",
    "RAGQueryResponse",
    "UploadResponse",
]
