from fastapi import APIRouter, Query
from typing import Optional
import os
import psycopg2
import pandas as pd

from config import DATABASE_URL  # centralized configuration

router = APIRouter()

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

@router.get("/raw")
def get_raw_data(
    country: str = Query("Luxembourg", description="Country filter (default: Luxembourg)"),
    transaction: Optional[str] = Query(None, description="Transaction type: 'buy' or 'rent'. If omitted, both are returned."),
    sold: str = Query("sold", description="Sold status: 'sold', 'unsold', or 'both' (default: sold)"),
    limit: int = Query(10, description="Number of records per page (default: 10)"),
    page: int = Query(1, description="Page number (default: 1)"),
    all_columns: bool = Query(False, description="If True, returns all columns from the database")
):
    """
    Returns listings filtered by country, transaction, and sold status.
    The results are sorted by created date and paginated.
    When all_columns=True, returns all columns from the database instead of the default subset.
    """
    # Fetch data from the database
    conn = get_db_connection()
    
    # Construct the SELECT part of the query based on all_columns parameter
    select_clause = "*" if all_columns else """
        id,
        price,
        "characteristic_surface" AS surface,
        createdat,
        soldat,
        transaction,
        "address_country" AS country
    """
    
    query = f"""
    SELECT {select_clause}
    FROM listings
    WHERE transaction IN ('rent', 'buy')
    """
    
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        conn.close()
        raise e
    conn.close()

    # If all columns are requested, only do minimal cleaning
    if all_columns:
        # Convert createdat to datetime if it exists
        if 'createdat' in df.columns:
            df['createdat_dt'] = pd.to_datetime(df['createdat'], 
                                              format='%Y%m%dT%H%M%SZ', 
                                              errors='coerce')
            df = df.dropna(subset=['createdat_dt'])
        
        # Normalize soldat if it exists
        if 'soldat' in df.columns:
            df['soldat'] = df['soldat'].astype(str).str.strip().str.lower()
            df.loc[df['soldat'] == 'nan', 'soldat'] = ""
        
        # Filter by country if the column exists
        if 'address_country' in df.columns:
            df = df[df['address_country'].str.lower() == country.lower()]
    else:
        # Original data cleaning for default column set
        df = df.dropna(subset=['price', 'surface', 'createdat'])
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['surface'] = pd.to_numeric(df['surface'], errors='coerce')
        df = df[(df['price'] > 0) & (df['surface'] > 0)]

        df['createdat_dt'] = pd.to_datetime(df['createdat'], 
                                          format='%Y%m%dT%H%M%SZ', 
                                          errors='coerce')
        df = df.dropna(subset=['createdat_dt'])

        df['soldat'] = df['soldat'].astype(str).str.strip().str.lower()
        df.loc[df['soldat'] == 'nan', 'soldat'] = ""

        df = df[df['country'].str.lower() == country.lower()]

    # Common filtering regardless of all_columns
    if transaction and transaction.lower() in ['buy', 'rent']:
        df = df[df['transaction'].str.lower() == transaction.lower()]

    if sold.lower() == "sold":
        df = df[df['soldat'] != ""]
    elif sold.lower() == "unsold":
        df = df[df['soldat'] == ""]

    # Sort by createdat_dt if it exists, otherwise skip sorting
    if 'createdat_dt' in df.columns:
        df = df.sort_values(by='createdat_dt', ascending=True)

    # Pagination
    offset = (page - 1) * limit
    df_page = df.iloc[offset:offset + limit]

    # Convert to records
    result = df_page.to_dict(orient="records")

    return {
        "page": page,
        "limit": limit,
        "total": len(df),
        "data": result
    }