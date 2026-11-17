"""Storage-independent identity and permission types."""

from dataclasses import dataclass
from enum import IntFlag, StrEnum
from uuid import UUID


class Role(StrEnum):
    EMPLOYEE = "employee"
    CONTRIBUTOR = "contributor"
    ADMINISTRATOR = "administrator"


class Permission(IntFlag):
    READ_KNOWLEDGE = 1
    WRITE_KNOWLEDGE = 2
    MANAGE_KNOWLEDGE = 4
    MANAGE_USERS = 8


@dataclass(frozen=True)
class User:
    id: UUID
    username: str
    display_name: str
    role: Role
    namespaces: tuple[str, ...]
    is_active: bool = True
