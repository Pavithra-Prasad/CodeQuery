"""Email notifications. In development they're only written to the log."""

import logging
from datetime import date, timedelta

from config import GRACE_PERIOD_DAYS

logger = logging.getLogger("bookshelf.notifications")

SENDER = "noreply@bookshelf.example"


def send_email(to: str, subject: str, body: str) -> None:
    """Send an email from SENDER. Development builds only log it; production swaps in an SMTP client."""
    logger.info("email from=%s to=%s subject=%s", SENDER, to, subject)


def notify_reservation_ready(member, book, hold_days: int) -> None:
    """Tell a member their reserved book is back. It's held for them for `hold_days` days."""
    pickup_by = date.today() + timedelta(days=hold_days)
    send_email(
        member.email,
        f"'{book.title}' is ready for pickup",
        f"Hi {member.name}, the book you reserved is back. Please pick it up by {pickup_by:%B %d}.",
    )


def send_overdue_reminders(library, today: date) -> int:
    """Daily job: email every member who has an overdue loan. Returns how many emails were sent."""
    sent = 0
    for loan in library.loans.values():
        if loan.is_returned or loan.days_overdue(today) == 0:
            continue
        member = library.members[loan.member_id]
        book = library.books[loan.isbn]
        send_email(
            member.email,
            "Overdue book",
            f"'{book.title}' was due on {loan.due_date}. Late fees start after a {GRACE_PERIOD_DAYS}-day grace period.",
        )
        sent += 1
    return sent
