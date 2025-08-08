"""RAG (Retrieval-Augmented Generation) 서비스."""

import logging
import time
from typing import List, Optional

from app.rag.models import RAGQueryModel
from app.rag.services.embedding_service import EmbeddingService
from app.rag.services.gpt_oss_service import GPTOSSService
from app.rag.services.vector_search_service import VectorSearchService

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 시스템 핵심 서비스."""

    def __init__(self):
        self.gpt_oss_service = GPTOSSService()
        self.embedding_service = EmbeddingService()
        self.vector_search_service = VectorSearchService()
        # TODO: 향후 추가될 서비스들
        # self.document_service = DocumentService()

    async def generate_answer(
        self, question: str, context_documents: List[str], user_id: str, **kwargs
    ) -> str:
        """질문과 컨텍스트 문서를 바탕으로 답변을 생성합니다."""

        if not question or not question.strip():
            raise ValueError("질문이 제공되지 않았습니다")

        if not context_documents:
            raise ValueError("참조 문서가 제공되지 않았습니다")

        try:
            logger.info(f"RAG 답변 생성 시작 - 사용자: {user_id}")
            logger.info(f"질문: {question[:100]}...")
            logger.info(f"참조 문서 수: {len(context_documents)}")

            # GPT-OSS를 통한 답변 생성
            answer = await self.gpt_oss_service.generate_rag_answer(
                question=question, context_documents=context_documents, **kwargs
            )

            logger.info("RAG 답변 생성 완료")
            return answer

        except Exception as e:
            logger.error(f"RAG 답변 생성 중 오류: {str(e)}")
            raise Exception(f"답변 생성에 실패했습니다: {str(e)}")

    async def generate_answer_with_fallback(
        self, question: str, context_documents: List[str], user_id: str, **kwargs
    ) -> str:
        """폴백 메커니즘이 있는 답변 생성 (향후 Gemini 폴백 추가 가능)."""

        try:
            # 주 모델 (GPT-OSS) 시도
            return await self.generate_answer(
                question=question,
                context_documents=context_documents,
                user_id=user_id,
                **kwargs,
            )

        except Exception as e:
            logger.error(f"GPT-OSS 답변 생성 실패: {str(e)}")

            # TODO: Gemini 폴백 구현
            # try:
            #     logger.info("Gemini 폴백 모델로 전환")
            #     return await self.gemini_service.generate_rag_answer(...)
            # except Exception as fallback_e:
            #     logger.error(f"폴백 모델도 실패: {str(fallback_e)}")

            # 현재는 GPT-OSS만 사용하므로 원래 예외 발생
            raise e

    async def process_rag_query(
        self,
        question: str,
        user_id: str,
        max_documents: int = 5,
        similarity_threshold: float = 0.7,
        **kwargs,
    ) -> dict:
        """전체 RAG 프로세스를 수행합니다."""

        if not question or not question.strip():
            raise ValueError("질문이 제공되지 않았습니다")

        if not user_id or not user_id.strip():
            raise ValueError("사용자 ID가 제공되지 않았습니다")

        search_start_time = time.time()
        generation_start_time = None

        try:
            logger.info(f"RAG 쿼리 처리 시작 - 사용자: {user_id}")
            logger.info(f"질문: {question[:100]}...")

            # 1. 질문 임베딩 생성
            logger.info("질문 임베딩 생성 중...")
            query_embedding = await self.embedding_service.encode_text(question)
            logger.info(f"임베딩 생성 완료 - 차원: {len(query_embedding)}")

            # 2. 유사 문서 검색
            logger.info(
                f"벡터 검색 중 - threshold: {similarity_threshold}, max_docs: {max_documents}"
            )
            search_results = await self.vector_search_service.search_similar_documents(
                query_embedding=query_embedding,
                similarity_threshold=similarity_threshold,
                max_docs=max_documents,
                user_id=user_id,
            )

            search_end_time = time.time()
            search_time_ms = int((search_end_time - search_start_time) * 1000)

            if not search_results:
                logger.warning("검색 결과가 없습니다")
                return {
                    "question": question,
                    "answer": "죄송합니다. 질문과 관련된 문서를 찾을 수 없습니다. 다른 질문을 해보시거나 더 구체적으로 질문해주세요.",
                    "sources": [],
                    "confidence_score": 0.0,
                    "search_time_ms": search_time_ms,
                    "generation_time_ms": 0,
                }

            logger.info(f"검색 완료: {len(search_results)}개 문서 발견")

            # 3. 컨텍스트 구성
            context_documents = []
            sources = []

            for result in search_results:
                # 컨텍스트 문서로 추가
                context_documents.append(result.content)

                # 출처 정보 구성
                sources.append(
                    {
                        "document_id": str(result.document_id),
                        "chunk_index": result.chunk_index,
                        "content": (
                            result.content[:200] + "..."
                            if len(result.content) > 200
                            else result.content
                        ),
                        "similarity_score": result.similarity_score,
                    }
                )

            logger.info(f"컨텍스트 구성 완료 - {len(context_documents)}개 문서")

            # 4. 답변 생성
            generation_start_time = time.time()
            logger.info("답변 생성 중...")

            answer = await self.generate_answer(
                question=question,
                context_documents=context_documents,
                user_id=user_id,
                **kwargs,
            )

            generation_end_time = time.time()
            generation_time_ms = int(
                (generation_end_time - generation_start_time) * 1000
            )

            # 5. 신뢰도 계산 (평균 유사도 기반)
            if search_results:
                avg_similarity = sum(r.similarity_score for r in search_results) / len(
                    search_results
                )
                confidence_score = min(avg_similarity, 1.0)  # 최대 1.0으로 제한
            else:
                confidence_score = 0.0

            logger.info("RAG 쿼리 처리 완료")

            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "confidence_score": confidence_score,
                "search_time_ms": search_time_ms,
                "generation_time_ms": generation_time_ms,
            }

        except Exception as e:
            logger.error(f"RAG 쿼리 처리 중 오류: {str(e)}")

            # 검색 시간은 계산 가능한 경우만
            if search_start_time:
                search_time_ms = int((time.time() - search_start_time) * 1000)
            else:
                search_time_ms = 0

            # 생성 시간은 계산 가능한 경우만
            if generation_start_time:
                generation_time_ms = int((time.time() - generation_start_time) * 1000)
            else:
                generation_time_ms = 0

            raise Exception(f"RAG 처리에 실패했습니다: {str(e)}")

    async def check_service_health(self) -> dict:
        """RAG 서비스 상태를 확인합니다."""

        health_status = {"service": "RAG", "status": "healthy", "components": {}}

        try:
            # GPT-OSS 서비스 상태 확인
            gpt_oss_healthy = await self.gpt_oss_service.check_health()
            health_status["components"]["gpt_oss"] = {
                "status": "healthy" if gpt_oss_healthy else "unhealthy",
                "model": self.gpt_oss_service.model,
                "base_url": self.gpt_oss_service.base_url,
            }

            if not gpt_oss_healthy:
                health_status["status"] = "degraded"

        except Exception as e:
            logger.error(f"상태 확인 중 오류: {str(e)}")
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status

    def get_service_info(self) -> dict:
        """RAG 서비스 정보를 반환합니다."""

        return {
            "service": "RAG",
            "version": "1.0.0",
            "primary_model": self.gpt_oss_service.get_model_info(),
            "features": [
                "Korean document Q&A",
                "Context-aware responses",
                "Local model execution",
                "Configurable reasoning levels",
            ],
        }
