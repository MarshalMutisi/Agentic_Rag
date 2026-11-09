"""User accounts and login tokens for the API.

Users live in a small SQLite database. Passwords are stored as salted scrypt
hashes, never in plain text. Logging in returns a signed JWT; the API reads the
user id from that token, so a client can't pretend to be another user.
"""

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

load_dotenv()

USERS_DB = os.getenv("USERS_DB", "users.db")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
TOKEN_LIFETIME = timedelta(days=7)

USERNAME_PATTERN = re.compile(r"^[a-z0-9_.-]{3,32}$")
MIN_PASSWORD_LENGTH = 8

bearer_scheme = HTTPBearer(auto_error=False)


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(USERS_DB)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    return connection


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    key = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"{salt.hex()}:{key.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    salt_hex, _ = stored_hash.split(":")
    return hmac.compare_digest(_hash_password(password, bytes.fromhex(salt_hex)), stored_hash)


def _public_user(row: sqlite3.Row) -> Dict[str, Any]:
    return {"id": row["id"], "username": row["username"]}


def create_user(username: str, password: str) -> Dict[str, Any]:
    """
    Register a new user

    Raises:
        ValueError: If the username is invalid or taken, or the password is too short
    """
    username = username.strip().lower()
    if not USERNAME_PATTERN.match(username):
        raise ValueError("Username must be 3-32 characters: letters, numbers, '_', '.' or '-'")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")

    with _connect() as connection:
        if connection.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            raise ValueError("Username is already taken")
        user_id = str(uuid.uuid4())
        connection.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, username, _hash_password(password), datetime.now(timezone.utc).isoformat()),
        )
        return {"id": user_id, "username": username}


def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Return the user if the username and password match, else None"""
    with _connect() as connection:
        row = connection.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),)).fetchone()
    if row is None or not _verify_password(password, row["password_hash"]):
        return None
    return _public_user(row)


def create_token(user: Dict[str, Any]) -> str:
    """Signed login token that identifies the user"""
    if not JWT_SECRET:
        raise RuntimeError("Set JWT_SECRET in .env")
    payload = {"sub": user["id"], "username": user["username"], "exp": datetime.now(timezone.utc) + TOKEN_LIFETIME}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> Dict[str, Any]:
    """FastAPI dependency: the logged-in user, from the "Authorization: Bearer <token>" header"""
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not logged in")
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired login, please log in again")
    return {"id": payload["sub"], "username": payload["username"]}
