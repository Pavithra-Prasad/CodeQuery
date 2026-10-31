"""Late fee rules and fee balances."""

from config import GRACE_PERIOD_DAYS, LATE_FEE_PER_DAY, MAX_LATE_FEE


def calculate_late_fee(days_overdue: int) -> float:
    """
    Fee for returning a book late.

    The first GRACE_PERIOD_DAYS overdue days are free. After that each day costs
    LATE_FEE_PER_DAY, and the fee for one loan never exceeds MAX_LATE_FEE.
    """
    billable_days = max(0, days_overdue - GRACE_PERIOD_DAYS)
    return round(min(billable_days * LATE_FEE_PER_DAY, MAX_LATE_FEE), 2)


def apply_fee(member, amount: float) -> None:
    """Add a fee to a member's outstanding balance."""
    if amount < 0:
        raise ValueError("Fee can't be negative")
    member.outstanding_fees = round(member.outstanding_fees + amount, 2)


def pay_fees(member, amount: float) -> float:
    """Record a payment at the front desk and return the remaining balance. Overpaying isn't allowed."""
    if amount <= 0:
        raise ValueError("Payment must be positive")
    if amount > member.outstanding_fees:
        raise ValueError("Payment is more than the balance owed")
    member.outstanding_fees = round(member.outstanding_fees - amount, 2)
    return member.outstanding_fees


def waive_fees(member, librarian) -> None:
    """Clear a member's whole balance. Only librarians may do this."""
    if librarian.role != "librarian":
        raise PermissionError("Only librarians can waive fees")
    member.outstanding_fees = 0.0
