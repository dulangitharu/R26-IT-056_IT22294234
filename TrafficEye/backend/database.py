"""
MongoDB connection setup (Motor async driver).
Reads MONGODB_URI from .env and exposes the "analysis_records" collection
for other modules to import and use directly.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

# Load .env from this file's own folder so it works regardless of
# the working directory the server was started from.
load_dotenv(Path(__file__).parent / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI not found in .env")

client = AsyncIOMotorClient(MONGODB_URI)
db = client["trafficeye"]
analysis_collection = db["analysis_records"]
officers_collection = db["officers"]


async def ensure_indexes():
    """Enforce unique officer usernames at the database level (defense in
    depth alongside the application-level check in /api/auth/register)."""
    await officers_collection.create_index("username", unique=True)
