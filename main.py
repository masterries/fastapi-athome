from fastapi import FastAPI, HTTPException
import os
import psycopg2
import pandas as pd
import numpy as np

app = FastAPI()

# Get the DATABASE_URL from environment variables (set on Railway)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("Missing DATABASE_URL environment variable!")

def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

def fetch_aggregated_sold_data_postgres():
    """
    Aggregates sold listing data from the PostgreSQL 'listings' table.
    Adjusts the SQL to use PostgreSQL functions.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    query = """
    SELECT 
        id,
        price,
        "characteristic_surface" AS surface,
        price::numeric / "characteristic_surface"::numeric AS ratio,
        CAST(SUBSTRING(createdat FROM 1 FOR 4) AS INTEGER) AS year,
        transaction,
        "address_country" AS country,
        soldat,
        EXTRACT(EPOCH FROM (soldat::timestamp - to_timestamp(createdat, 'YYYYMMDDHH24MISS')))/86400 AS open_days
    FROM listings
    WHERE CAST(SUBSTRING(createdat FROM 1 FOR 4) AS INTEGER) >= 2010
      AND soldat IS NOT NULL
      AND transaction IN ('rent', 'buy')
      AND price::numeric > 0
      AND "characteristic_surface"::numeric IS NOT NULL
      AND "characteristic_surface"::numeric > 0
      AND (EXTRACT(EPOCH FROM (soldat::timestamp - to_timestamp(createdat, 'YYYYMMDDHH24MISS')))/86400) >= 0
    ORDER BY year, transaction, country, id;
    """
    cur.execute(query)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()

    # Create a DataFrame and group by (year, transaction, country)
    df = pd.DataFrame(rows, columns=columns)
    groups = {}
    for _, row in df.iterrows():
        # Row fields: id, price, surface, ratio, year, transaction, country, soldat, open_days
        ratio = row['ratio']
        year = row['year']
        tran = row['transaction']
        country = row['country']
        open_days = row['open_days']
        key = (year, tran, country)
        groups.setdefault(key, {"ratios": [], "open_days": []})
        groups[key]["ratios"].append(ratio)
        groups[key]["open_days"].append(open_days)

    data = []
    for key, stats in groups.items():
        count = len(stats["ratios"])
        if count < 5:
            continue  # Skip groups with fewer than 5 listings.
        year, tran, country = key
        arr_ratio = np.array(stats["ratios"], dtype=float)
        arr_days = np.array(stats["open_days"], dtype=float)
        data.append({
            "year": year,
            "transaction": tran,
            "country": country if country is not None else "Unknown",
            "count": count,
            "median_ratio": float(np.median(arr_ratio)),
            "q1_ratio": float(np.percentile(arr_ratio, 25)),
            "q3_ratio": float(np.percentile(arr_ratio, 75)),
            "median_open_days": float(np.median(arr_days)),
            "q1_open_days": float(np.percentile(arr_days, 25)),
            "q3_open_days": float(np.percentile(arr_days, 75))
        })
    return data

def fetch_aggregated_unsold_data_postgres():
    """
    Aggregates unsold listing data from the PostgreSQL 'listings' table.
    Uses current time for computing open days.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    query = """
    SELECT 
        id,
        price,
        "characteristic_surface" AS surface,
        price::numeric / "characteristic_surface"::numeric AS ratio,
        CAST(SUBSTRING(createdat FROM 1 FOR 4) AS INTEGER) AS year,
        transaction,
        "address_country" AS country,
        createdat,
        EXTRACT(EPOCH FROM (now() - to_timestamp(createdat, 'YYYYMMDDHH24MISS')))/86400 AS open_days
    FROM listings
    WHERE CAST(SUBSTRING(createdat FROM 1 FOR 4) AS INTEGER) >= 2010
      AND soldat IS NULL
      AND transaction IN ('rent', 'buy')
      AND price::numeric > 0
      AND "characteristic_surface"::numeric IS NOT NULL
      AND "characteristic_surface"::numeric > 0
    ORDER BY year, transaction, country, id;
    """
    cur.execute(query)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()

    df = pd.DataFrame(rows, columns=columns)
    groups = {}
    for _, row in df.iterrows():
        ratio = row['ratio']
        year = row['year']
        tran = row['transaction']
        country = row['country']
        open_days = row['open_days']
        key = (year, tran, country)
        groups.setdefault(key, {"ratios": [], "open_days": []})
        groups[key]["ratios"].append(ratio)
        groups[key]["open_days"].append(open_days)

    data = []
    for key, stats in groups.items():
        count = len(stats["ratios"])
        if count < 5:
            continue
        year, tran, country = key
        arr_ratio = np.array(stats["ratios"], dtype=float)
        arr_days = np.array(stats["open_days"], dtype=float)
        data.append({
            "year": year,
            "transaction": tran,
            "country": country if country is not None else "Unknown",
            "count": count,
            "median_ratio": float(np.median(arr_ratio)),
            "q1_ratio": float(np.percentile(arr_ratio, 25)),
            "q3_ratio": float(np.percentile(arr_ratio, 75)),
            "median_open_days": float(np.median(arr_days)),
            "q1_open_days": float(np.percentile(arr_days, 25)),
            "q3_open_days": float(np.percentile(arr_days, 75))
        })
    return data

@app.get("/aggregated/sold")
def aggregated_sold():
    """
    Endpoint returning aggregated sold listings data.
    """
    try:
        data = fetch_aggregated_sold_data_postgres()
        return {"aggregated_sold": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error aggregating sold data: {e}")

@app.get("/aggregated/unsold")
def aggregated_unsold():
    """
    Endpoint returning aggregated unsold listings data.
    """
    try:
        data = fetch_aggregated_unsold_data_postgres()
        return {"aggregated_unsold": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error aggregating unsold data: {e}")
