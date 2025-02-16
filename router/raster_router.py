from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Dict, Any, List
import psycopg2
from psycopg2.extras import RealDictCursor
import time
import logging
from datetime import datetime
from functools import lru_cache
from dataclasses import dataclass
from config import DATABASE_URL
import math

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Grid Configuration Defaults ---
DEFAULT_CENTER_LAT = 49.6116
DEFAULT_CENTER_LON = 6.1319
DEFAULT_GRID_SIZE_KM = 1.0
GRID_RADIUS = 100  # maximum number of cells from center to include

# --- Query Parameters Dataclass ---
@dataclass(frozen=True)
class GridQueryParams:
    transaction: Optional[str]
    sold: str
    grid_size: float
    center_lat: float
    center_lon: float
    year: Optional[int] = None

# --- Database Connection ---
def get_db_connection():
    try:
        conn = psycopg2.connect(
            DATABASE_URL,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")

# --- Data Fetching ---
def get_property_data(transaction: Optional[str], sold: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch listings from the database with the fields needed for grid calculations.
    No country filter is applied (all listings are returned).
    The sold filter accepts: "sold", "unsold", or "both".
    If a year is provided, only listings from that year are returned.
    Note: Since createdat is stored as text (e.g. "20240906T171052Z"),
    we use SUBSTRING to extract the year.
    """
    query = """
    SELECT 
        price,
        characteristic_surface AS surface,
        address_pin_lat AS lat,
        address_pin_lon AS lon,
        createdat
    FROM listings
    WHERE 1=1
    """
    query_params = []

    if transaction:
        query += " AND LOWER(transaction) = LOWER(%s)"
        query_params.append(transaction)
    
    sold_lower = sold.lower()
    if sold_lower == "sold":
        query += " AND soldat IS NOT NULL AND soldat != ''"
    elif sold_lower == "unsold":
        query += " AND (soldat IS NULL OR soldat = '')"
    # For "both", no filter is applied.

    if year is not None:
        # Use substring to extract the year from the text-formatted createdat field.
        query += " AND SUBSTRING(createdat, 1, 4)::int = %s"
        query_params.append(year)
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, query_params)
                rows = cursor.fetchall()
        listings = []
        for row in rows:
            listing = dict(row)
            # Convert numeric fields to float
            try:
                listing["lat"] = float(listing["lat"])
            except (TypeError, ValueError):
                listing["lat"] = None
            try:
                listing["lon"] = float(listing["lon"])
            except (TypeError, ValueError):
                listing["lon"] = None
            try:
                listing["price"] = float(listing["price"])
            except (TypeError, ValueError):
                listing["price"] = None
            try:
                listing["surface"] = float(listing["surface"])
            except (TypeError, ValueError):
                listing["surface"] = None

            # Convert datetime to ISO format if needed
            if listing.get("createdat") and isinstance(listing["createdat"], datetime):
                listing["createdat"] = listing["createdat"].isoformat()
            listings.append(listing)
        return listings
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        raise HTTPException(status_code=500, detail="Database query failed")

# --- Grid Cell Calculation ---
def get_grid_cell(lat: float, lon: float, center_lat: float = DEFAULT_CENTER_LAT,
                  center_lon: float = DEFAULT_CENTER_LON, grid_size: float = DEFAULT_GRID_SIZE_KM) -> Optional[str]:
    """
    Calculate a grid cell ID for a given latitude and longitude relative to the provided center.
    Returns a string in the form "x_y".
    """
    lat_diff_km = (lat - center_lat) * 111.32
    lon_diff_km = (lon - center_lon) * (111.32 * math.cos(math.radians(center_lat)))
    x = math.floor(lon_diff_km / grid_size)
    y = math.floor(lat_diff_km / grid_size)
    if abs(x) <= GRID_RADIUS and abs(y) <= GRID_RADIUS:
        return f"{x}_{y}"
    return None

def get_cell_bounds(cell_id: str, center_lat: float = DEFAULT_CENTER_LAT,
                    center_lon: float = DEFAULT_CENTER_LON, grid_size: float = DEFAULT_GRID_SIZE_KM) -> Dict[str, float]:
    """
    Compute the geographic bounding box for a given cell ID ("x_y").
    Returns a dict with lat_min, lon_min, lat_max, lon_max.
    """
    x_str, y_str = cell_id.split("_")
    x = int(x_str)
    y = int(y_str)
    lat_min = center_lat + (y * grid_size) / 111.32
    lat_max = center_lat + ((y + 1) * grid_size) / 111.32
    lon_min = center_lon + (x * grid_size) / (111.32 * math.cos(math.radians(center_lat)))
    lon_max = center_lon + ((x + 1) * grid_size) / (111.32 * math.cos(math.radians(center_lat)))
    return {
        "lat_min": lat_min,
        "lon_min": lon_min,
        "lat_max": lat_max,
        "lon_max": lon_max
    }

# --- Statistics Calculation ---
def calculate_grid_statistics(listings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics for a group of listings.
    Computes count, average, median, minimum, and maximum price per m².
    """
    prices_per_m2 = []
    for listing in listings:
        price = listing.get("price")
        surface = listing.get("surface")
        if price is None or surface in (None, 0):
            continue
        prices_per_m2.append(price / surface)
    count = len(prices_per_m2)
    if count == 0:
        return {}
    avg_price = sum(prices_per_m2) / count
    sorted_prices = sorted(prices_per_m2)
    if count % 2 == 1:
        median_price = sorted_prices[count // 2]
    else:
        median_price = (sorted_prices[count // 2 - 1] + sorted_prices[count // 2]) / 2
    return {
        "count": count,
        "avg_price_per_m2": avg_price,
        "median_price_per_m2": median_price,
        "min_price_per_m2": min(prices_per_m2),
        "max_price_per_m2": max(prices_per_m2)
    }

# --- Data Fetching and Raster Calculation with Caching ---
@lru_cache(maxsize=100)
def fetch_and_calculate_raster(params: GridQueryParams) -> Dict[str, Any]:
    """
    Fetch listings and group them into grid cells.
    For each cell, compute the statistics and geographic boundaries.
    """
    start_time = time.time()
    listings = get_property_data(params.transaction, params.sold, params.year)
    if not listings:
        raise HTTPException(status_code=404, detail="No listings found for the given parameters")
    grid = {}
    for listing in listings:
        lat = listing.get("lat")
        lon = listing.get("lon")
        if lat is None or lon is None:
            continue
        cell_id = get_grid_cell(lat, lon, params.center_lat, params.center_lon, params.grid_size)
        if cell_id is None:
            continue
        grid.setdefault(cell_id, []).append(listing)
    cells = []
    for cell_id, cell_listings in grid.items():
        stats = calculate_grid_statistics(cell_listings)
        bounds = get_cell_bounds(cell_id, params.center_lat, params.center_lon, params.grid_size)
        cells.append({
            "cell_id": cell_id,
            "bounds": bounds,
            "statistics": stats
        })
    total_time = time.time() - start_time
    result = {
        "center": {"lat": params.center_lat, "lon": params.center_lon},
        "grid_size_km": params.grid_size,
        "year": params.year if params.year is not None else "all",
        "total_listings": len(listings),
        "number_of_cells": len(cells),
        "cells": cells,
        "processing_time_seconds": total_time
    }
    return result

# --- API Endpoint ---
@router.get("/raster")
async def get_raster(
    transaction: Optional[str] = Query(None, description="Transaction type: 'buy' or 'rent'. If omitted, both are returned."),
    sold: str = Query("sold", description="Sold status: 'sold', 'unsold', or 'both' (default: sold)"),
    grid_size: float = Query(1.0, description="Grid size in kilometers for the raster (default: 1.0 km)"),
    center_lat: float = Query(DEFAULT_CENTER_LAT, description="Center latitude for the grid (default: 49.6116)"),
    center_lon: float = Query(DEFAULT_CENTER_LON, description="Center longitude for the grid (default: 6.1319)"),
    year: Optional[int] = Query(None, description="Year to filter on. If omitted, averages over all years.")
) -> Dict[str, Any]:
    """
    API endpoint that calculates a grid-based raster from property listings.
    If a year is provided, only listings from that year are used;
    otherwise, listings from all years are used (averaged together).
    Returns meta information about each grid cell, including property count,
    average and median price per m², and the cell's geographic boundaries.
    """
    params = GridQueryParams(
        transaction=transaction,
        sold=sold,
        grid_size=grid_size,
        center_lat=center_lat,
        center_lon=center_lon,
        year=year
    )
    result = fetch_and_calculate_raster(params)
    return result
