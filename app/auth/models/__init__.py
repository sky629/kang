"""Auth domain models module."""

# SQLAlchemy ORM Models (Database Layer)
from .postgres_models import SocialAccount, User

# Pydantic Domain Models (Business Logic Layer)
from .social_account import SocialAccountModel
from .user import UserModel

__all__ = [
    # SQLAlchemy ORM Models
    "User",
    "SocialAccount",
    # Pydantic Domain Models
    "UserModel",
    "SocialAccountModel",
]