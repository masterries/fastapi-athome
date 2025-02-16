from fastapi import APIRouter, Query
from typing import Optional
import psycopg2
import pandas as pd
import numpy as np
from config import DATABASE_URL  # Import the DB URL from our config

router = APIRouter()

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

@router.get("/aggregated")
def get_aggregated_data(
    country: str = Query("Luxembourg", description="Country to filter on (default: Luxembourg)"),
    transaction: Optional[str] = Query(None, description="Transaction type: 'buy' or 'rent' (default: both)"),
    sold: str = Query("sold", description="Sold status: 'sold', 'unsold', or 'both' (default: sold)")
):
    """
    Returns aggregated €/m² ratio statistics grouped by year, transaction, and country.
    Query parameters:
      - country: Filter by country (default: Luxembourg)
      - transaction: 'buy' or 'rent' (if not provided, both are included)
      - sold: 'sold', 'unsold', or 'both' (default: sold)
    """
    # Fetch raw data from PostgreSQL with minimal filtering.
    conn = get_db_connection()
    query = """
    SELECT 
        id,
        price,
        "characteristic_surface" AS surface,
        createdat,
        soldat,
        transaction,
        "address_country" AS country
    FROM listings
    WHERE transaction IN ('rent', 'buy')
    """
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        conn.close()
        raise e
    conn.close()
    
    # Data cleaning: drop rows with missing essential data.
    df = df.dropna(subset=['price', 'surface', 'createdat'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['surface'] = pd.to_numeric(df['surface'], errors='coerce')
    df = df[(df['price'] > 0) & (df['surface'] > 0)]
    
    # Compute the €/m² ratio.
    df['ratio'] = df['price'] / df['surface']
    
    # Convert createdat (assumed format "YYYYMMDDTHHMMSSZ") to datetime and extract year.
    df['createdat_dt'] = pd.to_datetime(df['createdat'], format='%Y%m%dT%H%M%SZ', errors='coerce')
    df = df.dropna(subset=['createdat_dt'])
    df['year'] = df['createdat_dt'].dt.year

    # Normalize soldat:
    # Strip whitespace and convert to lower case.
    # Replace "nan" with an empty string so we treat it as unsold.
    df['soldat'] = df['soldat'].astype(str).str.strip().str.lower()
    df.loc[df['soldat'] == 'nan', 'soldat'] = ""

    # Filter by country (case-insensitive)
    df = df[df['country'].str.lower() == country.lower()]

    # Filter by transaction if provided.
    if transaction and transaction.lower() in ['buy', 'rent']:
        df = df[df['transaction'].str.lower() == transaction.lower()]
    
    # Filter by sold status:
    # For "sold": keep rows where soldat is not empty.
    # For "unsold": keep rows where soldat is empty.
    # For "both": no filtering.
    if sold.lower() == "sold":
        df = df[df['soldat'] != ""]
    elif sold.lower() == "unsold":
        df = df[df['soldat'] == ""]
    
    # Group by (year, transaction, country) and aggregate ratios into lists.
    grouped = df.groupby(['year', 'transaction', 'country'])['ratio'].agg(list).reset_index()

    # For groups with at least 5 entries, compute aggregated statistics.
    results = []
    for _, row in grouped.iterrows():
        ratios = np.array(row['ratio'], dtype=float)
        count = len(ratios)
        if count < 5:
            continue  # Skip groups with fewer than 5 data points.
        avg = float(np.mean(ratios))
        median = float(np.median(ratios))
        q1 = float(np.percentile(ratios, 25))
        q3 = float(np.percentile(ratios, 75))
        results.append({
            "year": int(row['year']),
            "transaction": row['transaction'],
            "country": row['country'],
            "count": count,
            "average_ratio": avg,
            "median_ratio": median,
            "q1_ratio": q1,
            "q3_ratio": q3
        })

    # Sort results by year, country, and transaction.
    results = sorted(results, key=lambda x: (x["year"], x["country"], x["transaction"]))
    return {"aggregated_data": results}
