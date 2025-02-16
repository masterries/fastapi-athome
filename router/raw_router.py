from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import time
import logging
from datetime import datetime
from functools import lru_cache
from dataclasses import dataclass
from config import DATABASE_URL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@dataclass(frozen=True)
class QueryParams:
    """Immutable class for cache keys"""
    country: str
    transaction: Optional[str]
    sold: str
    limit: int
    page: int
    all_columns: bool

def get_db_connection():
    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor
    )

# Cache for the actual data fetching function
@lru_cache(maxsize=100)  # Cache last 100 unique queries
def fetch_data(params: QueryParams) -> Dict[str, Any]:
    """Cached function to fetch data from database"""
    query_start = time.time()
    metrics = {}

    # Calculate offset for pagination
    offset = (params.page - 1) * params.limit

    # Build the query with proper indexing
    select_clause = "*" if params.all_columns else """
        id,
        price,
        characteristic_surface AS surface,
        createdat,
        soldat,
        transaction,
        address_country AS country
    """

    # Base query with index-optimized ordering
    query = f"""
    SELECT {select_clause}
    FROM listings
    WHERE transaction IN ('rent', 'buy')
    AND LOWER(address_country) = LOWER(%s)
    """
    query_params = [params.country]

    # Add transaction filter if specified
    if params.transaction:
        query += " AND LOWER(transaction) = LOWER(%s)"
        query_params.append(params.transaction)

    # Add sold status filter
    if params.sold.lower() == "sold":
        query += " AND soldat IS NOT NULL AND soldat != ''"
    elif params.sold.lower() == "unsold":
        query += " AND (soldat IS NULL OR soldat = '')"

    # Get total count with a separate optimized query
    count_query = f"""
    SELECT COUNT(*) as count
    FROM listings
    WHERE transaction IN ('rent', 'buy')
    AND LOWER(address_country) = LOWER(%s)
    """
    count_params = [params.country]
    if params.transaction:
        count_query += " AND LOWER(transaction) = LOWER(%s)"
        count_params.append(params.transaction)

    # Add sorting and pagination
    query += " ORDER BY createdat DESC NULLS LAST"
    query += f" LIMIT {params.limit} OFFSET {offset}"

    try:
        with get_db_connection() as conn:
            # Get total count
            count_start = time.time()
            with conn.cursor() as cursor:
                cursor.execute(count_query, count_params)
                result = cursor.fetchone()
                total_count = list(result.values())[0] if result else 0
            metrics['count_query_time'] = time.time() - count_start
            
            # Get paginated data
            data_start = time.time()
            with conn.cursor() as cursor:
                cursor.execute(query, query_params)
                rows = cursor.fetchall()
            metrics['data_query_time'] = time.time() - data_start

    except Exception as e:
        logger.error(f"Database query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    # Process the results
    processing_start = time.time()
    
    # Convert to list of dicts and clean up data
    result = []
    for row in rows:
        # Convert row from RealDictRow to regular dict
        clean_row = dict(row)
        
        # Convert datetime objects to ISO format strings
        for key, value in clean_row.items():
            if isinstance(value, datetime):
                clean_row[key] = value.isoformat()
        
        result.append(clean_row)

    metrics['processing_time'] = time.time() - processing_start
    total_time = time.time() - query_start

    # Log performance metrics
    logger.info(f"Count query completed in {metrics['count_query_time']:.2f} seconds")
    logger.info(f"Data query returned {len(rows)} rows in {metrics['data_query_time']:.2f} seconds")
    logger.info(f"Data processing completed in {metrics['processing_time']:.2f} seconds")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    
    return {
        "page": params.page,
        "limit": params.limit,
        "total": total_count,
        "data": result,
        "performance_metrics": metrics
    }

@router.get("/raw")
async def get_raw_data(
    country: str = Query("Luxembourg", description="Country filter (default: Luxembourg)"),
    transaction: Optional[str] = Query(None, description="Transaction type: 'buy' or 'rent'. If omitted, both are returned."),
    sold: str = Query("sold", description="Sold status: 'sold', 'unsold', or 'both' (default: sold)"),
    limit: int = Query(10, description="Number of records per page (default: 10)"),
    page: int = Query(1, description="Page number (default: 1)"),
    all_columns: bool = Query(False, description="If True, returns all columns from the database")
) -> Dict[str, Any]:
    """
    Returns listings data with in-memory caching.
    """
    # Create immutable params object for cache key
    params = QueryParams(
        country=country,
        transaction=transaction,
        sold=sold,
        limit=limit,
        page=page,
        all_columns=all_columns
    )
    
    # Log cache info
    cache_info = fetch_data.cache_info()
    logger.info(f"Cache stats: {cache_info}")
    
    # Return cached or fresh data
    return fetch_data(params)