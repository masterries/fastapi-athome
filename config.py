import os

DATABASE_URL = os.getenv("DATABASE_PUBLIC_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in environment.")
