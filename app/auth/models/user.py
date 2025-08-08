"""User domain models for business logic."""

import re
import uuid
from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserModel(BaseModel):
    """사용자 도메인 모델."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str = Field(..., description="사용자 이메일")
    name: str = Field(..., min_length=1, max_length=100, description="사용자 이름")
    user_level: int = Field(
        ..., description="사용자 권한 레벨 (100: normal, 1000: admin)"
    )
    profile_image_url: Optional[str] = Field(None, description="프로필 이미지 URL")
    is_active: bool = Field(True, description="계정 활성화 상태")
    email_verified: bool = Field(False, description="이메일 인증 상태")
    last_login_at: Optional[datetime] = Field(None, description="마지막 로그인 시간")
    created_at: datetime
    updated_at: datetime

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """이메일 형식 검증."""
        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        if not email_pattern.match(v):
            raise ValueError("올바른 이메일 형식이 아닙니다")
        return v.lower().strip()

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """사용자 이름 검증."""
        if not v.strip():
            raise ValueError("사용자 이름은 비어있을 수 없습니다")

        # 특수문자 제한 (기본적인 문자, 숫자, 공백, 한글만 허용)
        name_pattern = re.compile(r"^[a-zA-Z0-9가-힣\s]+$")
        if not name_pattern.match(v.strip()):
            raise ValueError("사용자 이름에 허용되지 않는 문자가 포함되어 있습니다")

        return v.strip()

    @field_validator("user_level")
    @classmethod
    def validate_user_level(cls, v: int) -> int:
        """사용자 레벨 검증."""
        allowed_levels = {100, 1000}  # 100: normal, 1000: admin
        if v not in allowed_levels:
            raise ValueError(
                f"허용되지 않는 사용자 레벨입니다: {v}. 허용 레벨: {allowed_levels}"
            )
        return v

    @field_validator("profile_image_url")
    @classmethod
    def validate_profile_image_url(cls, v: Optional[str]) -> Optional[str]:
        """프로필 이미지 URL 검증."""
        if v is not None and v.strip():
            url_pattern = re.compile(
                r"^https?://.+\.(jpg|jpeg|png|gif|webp)(\?.*)?$", re.IGNORECASE
            )
            if not url_pattern.match(v.strip()):
                raise ValueError("올바른 이미지 URL 형식이 아닙니다")
            return v.strip()
        return None

    def is_admin(self) -> bool:
        """관리자 권한 여부 확인."""
        return self.user_level >= 1000

    def is_normal_user(self) -> bool:
        """일반 사용자 권한 여부 확인."""
        return self.user_level == 100

    def is_profile_complete(self) -> bool:
        """프로필이 완성되었는지 확인."""
        required_fields = [
            self.email,
            self.name,
            self.email_verified,
        ]
        return all(field for field in required_fields) and self.email_verified

    def get_profile_completion(self) -> float:
        """프로필 완성도를 퍼센트로 반환 (0.0 ~ 1.0)."""
        total_fields = 5
        completed_fields = 0

        # 필수 필드들
        if self.email and self.email.strip():
            completed_fields += 1
        if self.name and self.name.strip():
            completed_fields += 1
        if self.email_verified:
            completed_fields += 1

        # 선택 필드들
        if self.profile_image_url and self.profile_image_url.strip():
            completed_fields += 1
        if self.last_login_at is not None:
            completed_fields += 1

        return completed_fields / total_fields

    def is_recently_active(self, days: int = 30) -> bool:
        """최근 활성 사용자인지 확인."""
        if self.last_login_at is None:
            return False

        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return self.last_login_at >= cutoff_date

    def can_perform_admin_action(self) -> bool:
        """관리자 작업을 수행할 수 있는지 확인."""
        return self.is_admin() and self.is_active and self.email_verified

    def get_display_name(self) -> str:
        """표시용 이름을 반환."""
        if self.name and self.name.strip():
            return self.name
        return self.email.split("@")[0]  # 이메일의 앞부분 사용

    def days_since_creation(self) -> int:
        """계정 생성 후 경과일수."""
        return (datetime.utcnow() - self.created_at).days

    def is_new_user(self, days: int = 7) -> bool:
        """신규 사용자인지 확인 (기본 7일)."""
        return self.days_since_creation() <= days
