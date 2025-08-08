"""Social Account domain models for business logic."""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SocialAccountModel(BaseModel):
    """소셜 계정 도메인 모델."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    provider: str = Field(..., description="OAuth 제공자 (google, apple)")
    provider_user_id: str = Field(..., description="제공자별 사용자 ID")
    provider_data: Optional[Dict] = Field(None, description="제공자별 추가 데이터")
    scope_granted: Optional[List[str]] = Field(
        None, description="부여된 권한 스코프 목록"
    )
    is_primary: bool = Field(True, description="기본 소셜 계정 여부")
    connected_at: datetime = Field(..., description="계정 연결 시간")
    last_used_at: Optional[datetime] = Field(None, description="마지막 사용 시간")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """OAuth 제공자 검증."""
        allowed_providers = {"google", "apple"}
        if v.lower() not in allowed_providers:
            raise ValueError(
                f"지원되지 않는 OAuth 제공자입니다: {v}. 지원 제공자: {allowed_providers}"
            )
        return v.lower()

    @field_validator("provider_user_id")
    @classmethod
    def validate_provider_user_id(cls, v: str) -> str:
        """제공자별 사용자 ID 검증."""
        if not v or not v.strip():
            raise ValueError("제공자 사용자 ID는 비어있을 수 없습니다")

        # 기본적인 형식 검증 (영숫자, 하이픈, 언더스코어만 허용)
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "제공자 사용자 ID는 영숫자, 하이픈, 언더스코어만 포함할 수 있습니다"
            )

        return v.strip()

    @field_validator("scope_granted")
    @classmethod
    def validate_scope_granted(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """부여된 권한 스코프 검증."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("scope_granted는 문자열 리스트여야 합니다")

            # 각 스코프가 유효한 형식인지 확인
            for scope in v:
                if not isinstance(scope, str) or not scope.strip():
                    raise ValueError("모든 스코프는 비어있지 않은 문자열이어야 합니다")

            # 중복 제거
            return list(set(scope.strip() for scope in v))

        return v

    def is_google_account(self) -> bool:
        """Google 계정인지 확인."""
        return self.provider == "google"

    def is_apple_account(self) -> bool:
        """Apple 계정인지 확인."""
        return self.provider == "apple"

    def has_scope(self, scope: str) -> bool:
        """특정 권한 스코프를 보유하고 있는지 확인."""
        if not self.scope_granted:
            return False
        return scope in self.scope_granted

    def get_granted_scopes(self) -> List[str]:
        """부여된 권한 스코프 목록 반환."""
        return self.scope_granted or []

    def is_recently_used(self, days: int = 30) -> bool:
        """최근에 사용된 계정인지 확인."""
        if self.last_used_at is None:
            return False

        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return self.last_used_at >= cutoff_date

    def days_since_connected(self) -> int:
        """계정 연결 후 경과일수."""
        return (datetime.utcnow() - self.connected_at).days

    def is_newly_connected(self, days: int = 7) -> bool:
        """최근에 연결된 계정인지 확인 (기본 7일)."""
        return self.days_since_connected() <= days

    def get_provider_display_name(self) -> str:
        """제공자의 표시용 이름 반환."""
        provider_names = {"google": "Google", "apple": "Apple"}
        return provider_names.get(self.provider, self.provider.title())

    def has_email_scope(self) -> bool:
        """이메일 권한이 있는지 확인."""
        email_scopes = {
            "email",
            "https://www.googleapis.com/auth/userinfo.email",  # Google
            "email",  # Apple
        }
        return any(self.has_scope(scope) for scope in email_scopes)

    def has_profile_scope(self) -> bool:
        """프로필 정보 권한이 있는지 확인."""
        profile_scopes = {
            "profile",
            "https://www.googleapis.com/auth/userinfo.profile",  # Google
            "name",  # Apple
        }
        return any(self.has_scope(scope) for scope in profile_scopes)

    def get_provider_data_value(self, key: str, default=None):
        """제공자별 데이터에서 특정 값 추출."""
        if not self.provider_data:
            return default
        return self.provider_data.get(key, default)

    def update_last_used(self) -> datetime:
        """마지막 사용 시간을 현재 시간으로 업데이트하고 반환."""
        return datetime.utcnow()

    def is_stale_connection(self, days: int = 90) -> bool:
        """오래된 연결인지 확인 (기본 90일)."""
        if self.last_used_at is None:
            # 사용된 적이 없다면 연결된 시간을 기준으로 판단
            return self.days_since_connected() > days
        return not self.is_recently_used(days)
