"""Data models for books, members and loans, plus the in-memory store that holds them."""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class Book:
    isbn: str
    title: str
    author: str
    total_copies: int = 1
    available_copies: int = 1
    reservations: list[str] = field(default_factory=list)  # member ids, first come first served


@dataclass
class Member:
    member_id: str
    name: str
    email: str
    password_hash: str
    role: str = "member"  # "member" or "librarian"
    outstanding_fees: float = 0.0
    active: bool = True


@dataclass
class Loan:
    loan_id: str
    isbn: str
    member_id: str
    checked_out: date
    due_date: date
    renewals: int = 0
    returned_on: Optional[date] = None

    @property
    def is_returned(self) -> bool:
        return self.returned_on is not None

    def days_overdue(self, today: date) -> int:
        """Whole days past the due date (0 if not overdue). Returned loans count up to the return day."""
        end = self.returned_on or today
        return max(0, (end - self.due_date).days)


@dataclass
class Library:
    """Everything the service knows. Kept in memory: a restart starts from empty."""
    books: dict[str, Book] = field(default_factory=dict)
    members: dict[str, Member] = field(default_factory=dict)
    loans: dict[str, Loan] = field(default_factory=dict)
