"""Service-level authorization; identity lookup and storage stay separate."""

from collections.abc import Sequence
from uuid import UUID

from .models import Permission, Role, User
from .repository import UserRepository


class AuthorizationService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def require_access(
        self, user_id: UUID, required: Permission, *, namespace: str,
        allowed_roles: Sequence[str] | None = None,
    ) -> User:
        """Resolve current user state, check all action bits, then enforce scope."""
        user = self.repository.get_by_id(user_id)
        if user is None or not user.is_active:
            raise PermissionError("Unknown or inactive user")
        if not required:
            raise ValueError("At least one permission must be required")
        granted = self.repository.get_permissions(user.role)
        if (granted & required) != required:
            raise PermissionError(f"{user.username} lacks the required permission")
        if namespace not in user.namespaces:
            raise PermissionError(f"{user.username} cannot access namespace {namespace}")
        if allowed_roles is not None:
            if isinstance(allowed_roles, (str, bytes)) or not allowed_roles:
                raise PermissionError("Document access roles must be a non-empty list")
            try:
                roles = {Role(role) for role in allowed_roles}
            except (ValueError, TypeError) as exc:
                raise PermissionError("Document contains an unknown access role") from exc
            # Administrators manage documents throughout their assigned namespace.
            if user.role != Role.ADMINISTRATOR and user.role not in roles:
                raise PermissionError(f"{user.username} cannot access this document")
        return user
