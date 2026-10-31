# Bookshelf

A small lending service for a community library: members borrow, renew and reserve books,
librarians process returns and manage late fees.

## Running it

```bash
pip install flask
flask --app api run
```

All data (books, members, loans and login tokens) lives in memory, so restarting the server clears it.
There is no database yet.

## API

| Method | Path | Who | What |
|--------|------|-----|------|
| POST | `/login` | anyone | Exchange email + password for a bearer token |
| POST | `/logout` | member | Invalidate the current token |
| POST | `/loans` | member | Borrow a book (`{"isbn": "..."}`) |
| POST | `/loans/<loan_id>/renew` | member (own loans) | Extend the due date |
| POST | `/loans/<loan_id>/return` | librarian | Check a book back in and charge late fees |
| POST | `/books/<isbn>/reserve` | member | Join the waiting list for a book with no free copies |
| GET | `/members/<member_id>/fees` | member (self) or librarian | See the outstanding balance |
| POST | `/members/<member_id>/fees/waive` | librarian | Clear a member's balance |

Every route except `/login` needs an `Authorization: Bearer <token>` header.

## Configuration

The loan period and the active-loan limit can be changed with environment variables:

- `BOOKSHELF_LOAN_DAYS` (default 14)
- `BOOKSHELF_MAX_LOANS` (default 3)

Everything else is set in `config.py`.
