"""HTTP API (Flask). Every route except /login needs an 'Authorization: Bearer <token>' header."""

from flask import Flask, jsonify, request

from auth import AuthError, current_member_id, login, logout
from fees import waive_fees
from loans import LoanError, checkout_book, renew_loan, reserve_book, return_book
from models import Library

app = Flask(__name__)
library = Library()  # in memory: restarting the server clears it


def _token() -> str:
    return request.headers.get("Authorization", "").removeprefix("Bearer ").strip()


def _member(required_role: str | None = None):
    """The logged-in member. Raises AuthError (401) or PermissionError (403)."""
    member = library.members[current_member_id(_token())]
    if required_role == "librarian" and member.role != "librarian":
        raise PermissionError("Librarians only")
    return member


@app.errorhandler(AuthError)
def _unauthorized(e):
    return jsonify(error=str(e)), 401


@app.errorhandler(PermissionError)
def _forbidden(e):
    return jsonify(error=str(e)), 403


@app.errorhandler(KeyError)
def _not_found(e):
    return jsonify(error=f"Not found: {e}"), 404


@app.errorhandler(LoanError)
def _conflict(e):
    return jsonify(error=str(e)), 409


@app.post("/login")
def post_login():
    body = request.get_json()
    return jsonify(token=login(library.members, body["email"], body["password"]))


@app.post("/logout")
def post_logout():
    logout(_token())
    return "", 204


@app.post("/loans")
def post_loan():
    member = _member()
    loan = checkout_book(library, member, request.get_json()["isbn"])
    return jsonify(loan_id=loan.loan_id, due_date=loan.due_date.isoformat()), 201


@app.post("/loans/<loan_id>/renew")
def post_renew(loan_id):
    member = _member()
    if library.loans[loan_id].member_id != member.member_id:
        raise PermissionError("You can only renew your own loans")
    loan = renew_loan(library, loan_id)
    return jsonify(due_date=loan.due_date.isoformat(), renewals=loan.renewals)


@app.post("/loans/<loan_id>/return")
def post_return(loan_id):
    _member("librarian")  # returns are processed at the desk
    fee = return_book(library, loan_id)
    return jsonify(late_fee=fee)


@app.post("/books/<isbn>/reserve")
def post_reserve(isbn):
    position = reserve_book(library, _member(), isbn)
    return jsonify(position=position), 201


@app.get("/members/<member_id>/fees")
def get_fees(member_id):
    member = _member()
    if member.member_id != member_id and member.role != "librarian":
        raise PermissionError("You can only see your own fees")
    return jsonify(outstanding=library.members[member_id].outstanding_fees)


@app.post("/members/<member_id>/fees/waive")
def post_waive(member_id):
    librarian = _member("librarian")
    waive_fees(library.members[member_id], librarian)
    return jsonify(outstanding=0.0)
