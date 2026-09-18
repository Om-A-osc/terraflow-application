"""
Raster utility functions.
Grid operations, binary mask vectorization, normalization helpers.
"""

import cv2
import numpy as np


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 1] range.
    Handles constant arrays (returns zeros).
    """
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    if arr_max - arr_min < 1e-10:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - arr_min) / (arr_max - arr_min)


def mask_to_polygon_coords(
    mask: np.ndarray,
    origin_x: float,
    origin_y: float,
    resolution_m: float,
    to_wgs84,
) -> list:
    """
    Convert a binary mask to polygon coordinates in WGS84.
    Uses OpenCV contour detection for clean polygon extraction.

    Args:
        mask: Binary 2D array (uint8, 0 or 255)
        origin_x: UTM X of the grid's top-left corner
        origin_y: UTM Y of the grid's top-left corner
        resolution_m: Grid cell size in meters
        to_wgs84: pyproj Transformer (UTM → WGS84)

    Returns:
        List of [lon, lat] coordinate rings (GeoJSON-compatible)
    """
    # Ensure proper type for OpenCV
    mask_uint8 = mask.astype(np.uint8)
    if mask_uint8.max() == 1:
        mask_uint8 = mask_uint8 * 255

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Take the largest contour
    largest = max(contours, key=cv2.contourArea)

    # Simplify the contour to reduce point count
    epsilon = 0.005 * cv2.arcLength(largest, True)
    simplified = cv2.approxPolyDP(largest, epsilon, True)

    # Convert pixel coordinates to UTM then to WGS84
    coords = []
    for point in simplified:
        col, row = point[0]
        utm_x = origin_x + col * resolution_m
        utm_y = origin_y - row * resolution_m  # Y decreases downward in raster
        lon, lat = to_wgs84.transform(utm_x, utm_y)
        coords.append([float(lon), float(lat)])

    # Close the ring
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])

    return [coords]


def pixel_to_utm(row: int, col: int, origin_x: float, origin_y: float, resolution_m: float):
    """Convert pixel (row, col) to UTM coordinates."""
    utm_x = origin_x + col * resolution_m
    utm_y = origin_y - row * resolution_m
    return utm_x, utm_y


def utm_to_pixel(utm_x: float, utm_y: float, origin_x: float, origin_y: float, resolution_m: float):
    """Convert UTM coordinates to pixel (row, col)."""
    col = int((utm_x - origin_x) / resolution_m)
    row = int((origin_y - utm_y) / resolution_m)
    return row, col
