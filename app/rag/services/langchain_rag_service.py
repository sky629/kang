"""LangChain 기반 RAG 서비스 - RunnableBranch와 체인 활용."""

import logging
import os
import time
from typing import Dict, List, Any, Optional

from langchain.schema.runnable import RunnableBranch
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

from app.rag.services.embedding_service import EmbeddingService
from app.rag.services.gpt_oss_service import GPTOSSService
from app.rag.services.vector_store_service import VectorStoreService
from config.settings import settings

logger = logging.getLogger(__name__)


class LangChainRAGService:
    """LangChain 체인 기반 RAG 서비스."""

    def __init__(self):
        self.gpt_oss_service = GPTOSSService()
        self.embedding_service = EmbeddingService()
        self.vector_store_service = VectorStoreService()
        
        # LangSmith 추적 설정
        if settings.langchain_tracing_v2 and settings.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2)
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        
        # 체인과 프롬프트 초기화
        self._rag_chain = None
        self._fallback_chain = None

    def _get_rag_prompt(self) -> ChatPromptTemplate:
        """RAG용 프롬프트 템플릿을 생성합니다."""
        return ChatPromptTemplate.from_messages([
            ("system", """당신은 한국어 문서 기반 질의응답 전문가입니다.
주어진 참조 문서들을 바탕으로 정확하고 도움이 되는 답변을 제공해주세요.

지침:
1. 참조 문서의 내용만을 기반으로 답변하세요
2. 참조 문서에 없는 정보는 추측하지 마세요
3. 답변은 자연스러운 한국어로 작성하세요
4. 가능한 한 구체적이고 상세한 답변을 제공하세요
5. 불확실한 경우 "참조 문서에서 명확한 정보를 찾을 수 없습니다"라고 말하세요

참조 문서들:
{context}"""),
            ("human", "질문: {question}")
        ])

    def _get_fallback_prompt(self) -> ChatPromptTemplate:
        """폴백 모드용 프롬프트 템플릿을 생성합니다."""
        return ChatPromptTemplate.from_messages([
            ("system", """당신은 도움이 되는 AI 어시스턴트입니다.
사용자의 질문에 대해 일반적인 지식을 바탕으로 답변해주세요.
답변 앞에 반드시 다음 안내 메시지를 포함하세요:

"※ 이 답변은 업로드된 문서가 없거나 관련 문서를 찾을 수 없어서 일반적인 지식을 바탕으로 생성되었습니다."

그리고 답변 마지막에 다음 메시지를 추가하세요:

"더 정확한 답변을 위해서는 관련 문서를 업로드해 주세요."
"""),
            ("human", "{question}")
        ])

    def _format_documents(self, docs: List[Dict[str, Any]]) -> str:
        """검색된 문서들을 컨텍스트 문자열로 포맷합니다."""
        context_parts = []
        for i, doc in enumerate(docs):
            content = doc.get("content", "")
            context_parts.append(f"[참조 문서 {i+1}]\n{content}")
        return "\n\n".join(context_parts)

    def _calculate_confidence_score(self, docs: List[Dict[str, Any]], fallback_mode: bool = False) -> float:
        """신뢰도 점수를 계산합니다."""
        if fallback_mode:
            return 0.3  # 폴백 모드는 낮은 신뢰도
        
        if not docs:
            return 0.0
        
        # 평균 유사도를 신뢰도로 사용
        avg_similarity = sum(doc.get("similarity_score", 0.0) for doc in docs) / len(docs)
        return min(avg_similarity, 1.0)

    def _should_use_fallback(self, docs: List[Dict[str, Any]], db_status: Dict[str, Any]) -> bool:
        """폴백 모드를 사용해야 하는지 판단합니다."""
        # 문서가 없고 DB가 비어있는 경우
        if not docs and not db_status.get("has_documents", False):
            return True
        # 문서가 없지만 DB에는 문서가 있는 경우 (관련성 없음)
        return not docs

    async def _search_documents_with_retry(
        self,
        question: str,
        max_documents: int = 5,
        similarity_threshold: float = 0.7,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """재시도 로직을 포함한 문서 검색."""
        
        # 첫 번째 시도
        docs = await self.vector_store_service.search_similar_documents(
            query=question,
            similarity_threshold=similarity_threshold,
            max_docs=max_documents,
            user_id=user_id,
        )
        
        if docs:
            logger.info(f"첫 번째 검색 성공: {len(docs)}개 문서 발견")
            return docs
        
        # 임계값을 낮춰서 재시도
        lower_threshold = max(similarity_threshold - 0.2, 0.3)
        logger.info(f"임계값을 {lower_threshold}로 낮춰서 재시도")
        
        retry_docs = await self.vector_store_service.search_similar_documents(
            query=question,
            similarity_threshold=lower_threshold,
            max_docs=max_documents,
            user_id=user_id,
        )
        
        if retry_docs:
            logger.info(f"재시도 검색 성공: {len(retry_docs)}개 문서 발견")
        else:
            logger.info("재시도 검색도 실패")
        
        return retry_docs

    def _build_rag_chain(self):
        """RAG 체인을 구성합니다."""
        if self._rag_chain is None:
            llm = self.gpt_oss_service._get_llm()
            prompt = self._get_rag_prompt()
            
            # RAG 체인: 문서 포맷팅 → 프롬프트 → LLM → 출력 파서
            self._rag_chain = (
                {
                    "context": lambda x: self._format_documents(x["documents"]),
                    "question": lambda x: x["question"]
                }
                | prompt
                | llm
                | StrOutputParser()
            )
        
        return self._rag_chain

    def _build_fallback_chain(self):
        """폴백 체인을 구성합니다."""
        if self._fallback_chain is None:
            llm = self.gpt_oss_service._get_llm()
            prompt = self._get_fallback_prompt()
            
            # 폴백 체인: 프롬프트 → LLM → 출력 파서
            self._fallback_chain = (
                prompt
                | llm
                | StrOutputParser()
            )
        
        return self._fallback_chain

    def _build_branched_chain(self):
        """조건부 분기 체인을 구성합니다."""
        rag_chain = self._build_rag_chain()
        fallback_chain = self._build_fallback_chain()
        
        # RunnableBranch로 조건부 실행
        branched_chain = RunnableBranch(
            # 조건: 문서가 있는 경우
            (lambda x: len(x.get("documents", [])) > 0, rag_chain),
            # 기본: 폴백 체인
            fallback_chain,
        )
        
        return branched_chain

    async def process_rag_query(
        self,
        question: str,
        user_id: str,
        max_documents: int = 5,
        similarity_threshold: float = 0.7,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """LangChain 체인을 사용한 RAG 쿼리 처리."""

        if not question or not question.strip():
            raise ValueError("질문이 제공되지 않았습니다")

        if not user_id or not user_id.strip():
            raise ValueError("사용자 ID가 제공되지 않았습니다")

        search_start_time = time.time()
        
        try:
            logger.info(f"LangChain RAG 쿼리 처리 시작 - 사용자: {user_id}")
            logger.info(f"질문: {question[:100]}...")

            # 1. 문서 검색 (재시도 로직 포함)
            logger.info("문서 검색 중...")
            search_results = await self._search_documents_with_retry(
                question=question,
                max_documents=max_documents,
                similarity_threshold=similarity_threshold,
                user_id=user_id,
            )

            search_end_time = time.time()
            search_time_ms = int((search_end_time - search_start_time) * 1000)

            # 2. DB 상태 확인 (폴백 결정용)
            db_status = await self.vector_store_service.check_database_status()
            
            # 3. 폴백 모드 결정
            fallback_mode = self._should_use_fallback(search_results, db_status)
            
            # 4. 체인 실행 준비
            chain_input = {
                "question": question,
                "documents": search_results,
            }
            
            # 5. LangChain 체인 실행
            generation_start_time = time.time()
            logger.info("LangChain 체인 실행 중...")
            
            branched_chain = self._build_branched_chain()
            
            # 체인 실행 (비동기)
            answer = await branched_chain.ainvoke(chain_input)
            
            generation_end_time = time.time()
            generation_time_ms = int((generation_end_time - generation_start_time) * 1000)

            # 6. 출처 정보 구성
            sources = []
            for result in search_results:
                sources.append({
                    "document_id": str(result.get("document_id", "")),
                    "chunk_index": result.get("chunk_index", 0),
                    "content": (
                        result["content"][:200] + "..."
                        if len(result["content"]) > 200
                        else result["content"]
                    ),
                    "similarity_score": result.get("similarity_score", 0.0),
                })

            # 7. 신뢰도 계산
            confidence_score = self._calculate_confidence_score(search_results, fallback_mode)

            logger.info("LangChain RAG 쿼리 처리 완료")

            result = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "confidence_score": confidence_score,
                "search_time_ms": search_time_ms,
                "generation_time_ms": generation_time_ms,
                "fallback_mode": fallback_mode,
            }

            # 추가 정보 (필요한 경우)
            if fallback_mode:
                result["db_status"] = db_status

            return result

        except Exception as e:
            logger.error(f"LangChain RAG 쿼리 처리 중 오류: {str(e)}")
            
            # 검색 시간 계산
            search_time_ms = int((time.time() - search_start_time) * 1000)
            
            raise Exception(f"RAG 처리에 실패했습니다: {str(e)}")

    async def generate_direct_answer(
        self,
        question: str,
        context_documents: List[str],
        user_id: str,
        **kwargs,
    ) -> str:
        """제공된 컨텍스트로 직접 답변 생성 (기존 API 호환용)."""
        
        if not context_documents:
            raise ValueError("참조 문서가 제공되지 않았습니다")

        try:
            logger.info(f"직접 답변 생성 - 사용자: {user_id}")
            
            # 문서를 딕셔너리 형태로 변환
            docs = [{"content": doc} for doc in context_documents]
            
            # RAG 체인 사용
            rag_chain = self._build_rag_chain()
            
            chain_input = {
                "question": question,
                "documents": docs,
            }
            
            answer = await rag_chain.ainvoke(chain_input)
            
            logger.info("직접 답변 생성 완료")
            return answer

        except Exception as e:
            logger.error(f"직접 답변 생성 중 오류: {str(e)}")
            raise Exception(f"답변 생성에 실패했습니다: {str(e)}")

    async def check_service_health(self) -> Dict[str, Any]:
        """LangChain RAG 서비스 상태를 확인합니다."""
        
        health_status = {
            "service": "LangChain RAG",
            "status": "healthy",
            "components": {},
            "langchain_enabled": True,
        }

        try:
            # LLM 상태 확인
            llm_healthy = await self.gpt_oss_service.check_health()
            health_status["components"]["llm"] = {
                "status": "healthy" if llm_healthy else "unhealthy",
                "model": self.gpt_oss_service.model,
                "base_url": self.gpt_oss_service.base_url,
                "type": "ChatOllama",
            }

            if not llm_healthy:
                health_status["status"] = "degraded"

            # 벡터 스토어 상태 확인
            vector_status = await self.vector_store_service.check_database_status()
            health_status["components"]["vector_store"] = {
                "status": "ready" if vector_status.get("vector_store_ready", False) else "not_ready",
                "type": "PGVector",
                **vector_status,
            }

            # 임베딩 서비스 상태 확인
            embedding_info = self.embedding_service.get_model_info()
            health_status["components"]["embedding_service"] = {
                "status": "ready" if embedding_info["loaded"] else "loading",
                "type": "HuggingFaceEmbeddings",
                **embedding_info,
            }

            # LangSmith 추적 상태
            health_status["components"]["langsmith_tracing"] = {
                "enabled": bool(settings.langchain_tracing_v2 and settings.langchain_api_key),
                "project": settings.langchain_project if settings.langchain_api_key else None,
            }

        except Exception as e:
            logger.error(f"상태 확인 중 오류: {str(e)}")
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status

    def get_service_info(self) -> Dict[str, Any]:
        """LangChain RAG 서비스 정보를 반환합니다."""
        
        return {
            "service": "LangChain RAG",
            "version": "2.0.0",
            "framework": "LangChain",
            "components": {
                "llm": "ChatOllama",
                "embeddings": "HuggingFaceEmbeddings", 
                "vector_store": "PGVector",
                "chains": ["RunnableBranch", "ChatPromptTemplate"],
            },
            "features": [
                "Korean document Q&A",
                "Context-aware responses",
                "Local model execution",
                "Automatic fallback mechanisms",
                "LangSmith tracing support",
                "Configurable similarity thresholds",
                "Retry logic for document search",
            ],
            "primary_model": self.gpt_oss_service.get_model_info(),
        }