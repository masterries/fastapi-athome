from fastapi import FastAPI
import psycopg2
import os

app = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL")  # Set this in Railway Environment Variables

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

@app.get("/")
def read_root():
    return {"message": "FastAPI on Railway is running!"}

@app.get("/listings")
def get_listings(limit: int = 10):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM listings LIMIT %s", (limit,))
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return [dict(zip(columns, row)) for row in rows]
