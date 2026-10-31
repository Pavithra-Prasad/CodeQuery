"""Settings for the Bookshelf lending service."""

import os

# Borrowing
LOAN_PERIOD_DAYS = int(os.environ.get("BOOKSHELF_LOAN_DAYS", "14"))
MAX_ACTIVE_LOANS = int(os.environ.get("BOOKSHELF_MAX_LOANS", "3"))
MAX_RENEWALS = 2
RENEWAL_DAYS = 7
RESERVATION_HOLD_DAYS = 3

# Late fees
GRACE_PERIOD_DAYS = 2
LATE_FEE_PER_DAY = 0.25
MAX_LATE_FEE = 10.00
FEE_BLOCK_THRESHOLD = 5.00  # members owing this much or more can't borrow

# Authentication
TOKEN_TTL_MINUTES = 60
PASSWORD_ITERATIONS = 200_000
