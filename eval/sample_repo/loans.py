"""Borrowing rules: checkout, return, renewal and reservations."""

import uuid
from datetime import date, timedelta

from config import (
    FEE_BLOCK_THRESHOLD,
    LOAN_PERIOD_DAYS,
    MAX_ACTIVE_LOANS,
    MAX_RENEWALS,
    RENEWAL_DAYS,
    RESERVATION_HOLD_DAYS,
)
from fees import apply_fee, calculate_late_fee
from models import Loan
from notifications import notify_reservation_ready


class LoanError(Exception):
    """A borrowing rule was broken (loan limit reached, fees owed, no copies left...)."""


def active_loans(loans: dict, member_id: str) -> list:
    """Loans a member has out right now."""
    return [loan for loan in loans.values() if loan.member_id == member_id and not loan.is_returned]


def checkout_book(library, member, isbn: str, today: date | None = None) -> Loan:
    """
    Lend a book to a member. The rules are checked in this order:
    1. the member's account must be active
    2. the member must owe less than FEE_BLOCK_THRESHOLD in fees
    3. the member may have at most MAX_ACTIVE_LOANS books out
    4. a copy must be available, and if the book is reserved only the first member
       in the queue may take it
    The new loan is due LOAN_PERIOD_DAYS from today.
    """
    today = today or date.today()
    if not member.active:
        raise LoanError("Account is suspended")
    if member.outstanding_fees >= FEE_BLOCK_THRESHOLD:
        raise LoanError(f"Outstanding fees of ${member.outstanding_fees:.2f} must be paid first")
    if len(active_loans(library.loans, member.member_id)) >= MAX_ACTIVE_LOANS:
        raise LoanError(f"Limit of {MAX_ACTIVE_LOANS} books reached")

    book = library.books[isbn]
    if book.available_copies < 1:
        raise LoanError("No copies available; you can reserve it instead")
    if book.reservations:
        if book.reservations[0] != member.member_id:
            raise LoanError("This copy is held for another member")
        book.reservations.pop(0)

    book.available_copies -= 1
    loan = Loan(
        loan_id=uuid.uuid4().hex[:8],
        isbn=isbn,
        member_id=member.member_id,
        checked_out=today,
        due_date=today + timedelta(days=LOAN_PERIOD_DAYS),
    )
    library.loans[loan.loan_id] = loan
    return loan


def return_book(library, loan_id: str, today: date | None = None) -> float:
    """
    Check a book back in. Charges any late fee to the member and, if the book has a
    waiting list, emails the first member in line. Returns the fee charged.
    """
    today = today or date.today()
    loan = library.loans[loan_id]
    if loan.is_returned:
        raise LoanError("Already returned")

    loan.returned_on = today
    fee = calculate_late_fee(loan.days_overdue(today))
    if fee:
        apply_fee(library.members[loan.member_id], fee)

    book = library.books[loan.isbn]
    book.available_copies += 1
    if book.reservations:
        next_member = library.members[book.reservations[0]]
        notify_reservation_ready(next_member, book, hold_days=RESERVATION_HOLD_DAYS)
    return fee


def renew_loan(library, loan_id: str, today: date | None = None) -> Loan:
    """
    Push a loan's due date back by RENEWAL_DAYS (counted from the current due date).
    Not allowed when the loan is overdue, when it has been renewed MAX_RENEWALS times
    already, or when another member has reserved the book.
    """
    today = today or date.today()
    loan = library.loans[loan_id]
    if loan.is_returned:
        raise LoanError("Loan already returned")
    if loan.days_overdue(today) > 0:
        raise LoanError("Overdue loans can't be renewed")
    if loan.renewals >= MAX_RENEWALS:
        raise LoanError(f"Renewal limit of {MAX_RENEWALS} reached")
    if library.books[loan.isbn].reservations:
        raise LoanError("Someone has reserved this book")

    loan.due_date += timedelta(days=RENEWAL_DAYS)
    loan.renewals += 1
    return loan


def reserve_book(library, member, isbn: str) -> int:
    """Join the waiting list for a book with no free copies. Returns the member's place in line (1-based)."""
    book = library.books[isbn]
    if book.available_copies > 0:
        raise LoanError("A copy is available; check it out instead")
    if member.member_id in book.reservations:
        raise LoanError("Already reserved")
    book.reservations.append(member.member_id)
    return len(book.reservations)
