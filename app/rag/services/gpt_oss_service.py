"""GPT-OSS service for text generation using Ollama with LangChain."""

import logging
import os
from typing import Dict, List, Optional

# from langchain_ollama import ChatOllama  # M1 맥에서 설치 필요
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# 임시 mock 클래스 (M1 맥에서는 실제 ChatOllama 사용)
try:
    from langchain_ollama import ChatOllama
except ImportError:
    # Mock ChatOllama for development
    class ChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        
        async def ainvoke(self, messages):
            return AIMessage(content="Mock response from ChatOllama")

from config.settings import settings

logger = logging.getLogger(__name__)


class GPTOSSService:
    """GPT-OSS 로컬 모델을 통한 텍스트 생성 서비스 - LangChain 기반."""

    def __init__(self):
        self.base_url = settings.gpt_oss_base_url
        self.model = settings.gpt_oss_model
        self.max_tokens = settings.gpt_oss_max_tokens
        self.temperature = settings.gpt_oss_temperature
        self.reasoning_level = settings.gpt_oss_reasoning_level
        self.timeout = settings.gpt_oss_timeout
        self._llm = None
        
        # LangSmith 추적 설정
        if settings.langchain_tracing_v2 and settings.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2)
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    def _get_llm(self) -> ChatOllama:
        """ChatOllama 인스턴스를 생성하고 반환합니다."""
        if self._llm is None:
            logger.info(f"ChatOllama 인스턴스 생성 중 - 모델: {self.model}")
            
            self._llm = ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens,
                timeout=self.timeout,
            )
            
            logger.info("ChatOllama 인스턴스 생성 완료")
        
        return self._llm

    async def _make_request(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        """LangChain ChatOllama로 요청을 보내고 응답을 받습니다."""

        try:
            llm = self._get_llm()
            
            # 메시지 구성
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            logger.info("ChatOllama 요청 전송")

            # LangChain invoke 사용
            response = await llm.ainvoke(messages)
            
            logger.info("ChatOllama 응답 수신 완료")
            
            if isinstance(response, AIMessage):
                return response.content
            else:
                return str(response)

        except Exception as e:
            logger.error(f"ChatOllama 요청 처리 중 오류: {str(e)}")
            raise Exception(f"LLM 요청에 실패했습니다: {str(e)}")


    def _build_rag_system_prompt(self) -> str:
        """RAG 시스템용 시스템 프롬프트를 생성합니다."""
        return """당신은 한국어 문서 기반 질의응답 전문가입니다.
주어진 참조 문서들을 바탕으로 정확하고 도움이 되는 답변을 제공해주세요.

지침:
1. 참조 문서의 내용만을 기반으로 답변하세요
2. 참조 문서에 없는 정보는 추측하지 마세요
3. 답변은 자연스러운 한국어로 작성하세요
4. 가능한 한 구체적이고 상세한 답변을 제공하세요
5. 불확실한 경우 "참조 문서에서 명확한 정보를 찾을 수 없습니다"라고 말하세요"""

    def _build_rag_prompt(self, question: str, context_documents: List[str]) -> str:
        """RAG용 프롬프트를 구성합니다."""

        # 컨텍스트 문서 구성
        context_text = "\n\n".join(
            [f"[참조 문서 {i+1}]\n{doc}" for i, doc in enumerate(context_documents)]
        )

        return f"""참조 문서들:
{context_text}

질문: {question}

위 참조 문서들을 바탕으로 질문에 답변해주세요."""

    async def generate_text(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        """일반적인 텍스트 생성."""

        try:
            generated_text = await self._make_request(
                prompt=prompt, system_prompt=system_prompt, **kwargs
            )

            if not generated_text or not generated_text.strip():
                raise Exception("ChatOllama에서 빈 응답을 받았습니다")

            return generated_text.strip()

        except Exception as e:
            logger.error(f"텍스트 생성 중 오류: {str(e)}")
            raise

    async def generate_rag_answer(
        self, question: str, context_documents: List[str], **kwargs
    ) -> str:
        """RAG 시스템용 답변 생성."""

        if not context_documents:
            raise ValueError("참조 문서가 제공되지 않았습니다")

        system_prompt = self._build_rag_system_prompt()
        rag_prompt = self._build_rag_prompt(question, context_documents)

        try:
            logger.info(f"RAG 답변 생성 시작 - 질문: {question[:50]}...")
            logger.info(f"참조 문서 수: {len(context_documents)}")

            answer = await self.generate_text(
                prompt=rag_prompt, system_prompt=system_prompt, **kwargs
            )

            logger.info("RAG 답변 생성 완료")
            return answer

        except Exception as e:
            logger.error(f"RAG 답변 생성 중 오류: {str(e)}")
            raise

    async def check_health(self) -> bool:
        """ChatOllama 서비스 상태를 확인합니다."""

        try:
            llm = self._get_llm()
            
            # 간단한 테스트 메시지로 상태 확인
            test_message = [HumanMessage(content="안녕하세요. 간단히 '안녕하세요'라고 답변해주세요.")]
            
            logger.info("ChatOllama 상태 확인 중...")
            response = await llm.ainvoke(test_message)
            
            if response and (isinstance(response, AIMessage) or str(response)):
                logger.info(f"ChatOllama 서비스 정상 - 모델 '{self.model}' 응답 확인")
                return True
            else:
                logger.warning(f"ChatOllama에서 유효하지 않은 응답을 받았습니다")
                return False

        except Exception as e:
            logger.error(f"ChatOllama 상태 확인 중 오류: {str(e)}")
            return False

    async def pull_model(self) -> bool:
        """모델을 다운로드합니다. (LangChain ChatOllama는 자동으로 모델을 로드합니다)"""

        try:
            logger.info(f"모델 '{self.model}' 로드 확인...")
            
            # ChatOllama는 첫 번째 호출 시 자동으로 모델을 로드하므로
            # health check로 모델 가용성을 확인
            is_healthy = await self.check_health()
            
            if is_healthy:
                logger.info(f"모델 '{self.model}' 로드 완료")
                return True
            else:
                logger.error(f"모델 '{self.model}' 로드 실패")
                return False

        except Exception as e:
            logger.error(f"모델 로드 중 오류: {str(e)}")
            return False

    def get_model_info(self) -> Dict[str, any]:
        """현재 설정된 모델 정보를 반환합니다."""

        return {
            "base_url": self.base_url,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_level": self.reasoning_level,
            "timeout": self.timeout,
        }
