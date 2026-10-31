"""Password hashing and login tokens."""

import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta

from config import PASSWORD_ITERATIONS, TOKEN_TTL_MINUTES

# token -> (member_id, expires_at). In memory only: tokens don't survive a restart.
_tokens: dict[str, tuple[str, datetime]] = {}


class AuthError(Exception):
    """A login failed, or a token is missing, unknown or expired."""


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a random 16-byte salt. Returns 'salt$hash' in hex."""
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PASSWORD_ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check a password against a stored 'salt$hash' value.

    Uses hmac.compare_digest so the comparison takes the same time whether or not the
    hashes match, which stops timing attacks.
    """
    salt_hex, _, digest_hex = stored.partition("$")
    candidate = hash_password(password, bytes.fromhex(salt_hex)).partition("$")[2]
    return hmac.compare_digest(candidate, digest_hex)


def login(members: dict, email: str, password: str) -> str:
    """Log a member in and return a new bearer token, valid for TOKEN_TTL_MINUTES."""
    member = next((m for m in members.values() if m.email == email), None)
    # Same error for every failure, so the response doesn't reveal which emails exist
    if member is None or not member.active or not verify_password(password, member.password_hash):
        raise AuthError("Invalid email or password")

    token = secrets.token_urlsafe(32)
    _tokens[token] = (member.member_id, datetime.utcnow() + timedelta(minutes=TOKEN_TTL_MINUTES))
    return token


def current_member_id(token: str) -> str:
    """Return the member id a token belongs to. Expired tokens are deleted and rejected."""
    entry = _tokens.get(token)
    if entry is None:
        raise AuthError("Unknown token")
    member_id, expires_at = entry
    if datetime.utcnow() >= expires_at:
        del _tokens[token]
        raise AuthError("Token expired")
    return member_id


def logout(token: str) -> None:
    """Invalidate a token. Logging out twice is harmless."""
    _tokens.pop(token, None)
