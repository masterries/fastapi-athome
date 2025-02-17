from fastapi import FastAPI
import psycopg2
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "*",
    "https://super-winner-gx66wvjx9jw3g5w-3000.app.github.dev",
    # Add any other domains you want to allow
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or use ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_PUBLIC_URL = os.getenv("DATABASE_PUBLIC_URL")  # This can be used in main as well.

def get_db_connection():
    return psycopg2.connect(DATABASE_PUBLIC_URL)

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

# Import and include the aggregated router from the router folder.
from router.aggregated_router import router as aggregated_router
app.include_router(aggregated_router)
from router.raw_router import router as raw_router
app.include_router(raw_router)
from router.raster_router import router as raster_router
app.include_router(raster_router, prefix="/raster")