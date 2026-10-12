"""
Authentication utilities for TrafficEye: password hashing, JWT issuance,
and a FastAPI dependency that protects routes requiring a logged-in officer.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import ExpiredSignatureError, JWTError, jwt
from passlib.context import CryptContext

from database import officers_collection

load_dotenv(Path(__file__).parent / ".env")

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY not found in .env")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_officer(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Decode and verify the Bearer token, then look up the officer it belongs
    to. Raises 401 for a missing, malformed, or expired token, or if the
    officer account no longer exists.
    """
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired — please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError:
        raise unauthorized

    username = payload.get("sub")
    if username is None:
        raise unauthorized

    officer = await officers_collection.find_one({"username": username})
    if officer is None:
        raise unauthorized

    return officer
