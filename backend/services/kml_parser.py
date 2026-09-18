"""
KML/KMZ Parser Service.
Extracts contour lines with elevation values from KML files.
Handles both KML (plain XML) and KMZ (zipped KML) formats.

The parser is generic — it extracts elevation from the <name> tag of each
Placemark containing a LineString, which is the standard output format
of contour map generators.
"""

import io
import logging
import re
import zipfile

import numpy as np
from lxml import etree

logger = logging.getLogger(__name__)

# KML namespace
KML_NS = "{http://www.opengis.net/kml/2.2}"


def parse_kml_bytes(file_bytes: bytes, filename: str) -> dict:
    """
    Parse a KML or KMZ file and extract contour line data.

    Args:
        file_bytes: Raw file content as bytes
        filename: Original filename (used for format detection)

    Returns:
        dict with keys:
            - vertices: np.ndarray of shape (N, 3) → [lon, lat, elevation]
            - contour_lines: list of dicts with 'elevation' and 'coordinates'
            - elevation_range: (min_elev, max_elev)
            - contour_count: number of distinct contour lines
            - extent: (min_lon, max_lon, min_lat, max_lat)

    Raises:
        ValueError: If file cannot be parsed or contains no contour data
    """
    # Handle KMZ (ZIP containing KML)
    kml_content = _extract_kml_content(file_bytes, filename)

    # Parse XML
    try:
        root = etree.fromstring(kml_content)
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Invalid KML/XML syntax: {e}")

    # Extract contour lines from all Placemarks
    contour_lines = _extract_contour_lines(root)

    if not contour_lines:
        raise ValueError(
            "No contour lines found in the KML file. "
            "Expected Placemarks with LineString geometries and numeric elevation names."
        )

    # Build vertex array (lon, lat, elevation)
    all_vertices = []
    for contour in contour_lines:
        elev = contour["elevation"]
        for lon, lat in contour["coordinates"]:
            all_vertices.append([lon, lat, elev])

    vertices = np.array(all_vertices, dtype=np.float64)

    # Compute extent
    min_lon, max_lon = vertices[:, 0].min(), vertices[:, 0].max()
    min_lat, max_lat = vertices[:, 1].min(), vertices[:, 1].max()
    min_elev, max_elev = vertices[:, 2].min(), vertices[:, 2].max()

    logger.info(
        f"Parsed {len(contour_lines)} contour lines, "
        f"{len(vertices)} vertices, "
        f"elevation {min_elev}–{max_elev}m, "
        f"extent [{min_lon:.4f},{min_lat:.4f}]–[{max_lon:.4f},{max_lat:.4f}]"
    )

    return {
        "vertices": vertices,
        "contour_lines": contour_lines,
        "elevation_range": (float(min_elev), float(max_elev)),
        "contour_count": len(contour_lines),
        "extent": (float(min_lon), float(max_lon), float(min_lat), float(max_lat)),
    }


def contour_lines_to_geojson(contour_lines: list) -> dict:
    """
    Convert contour lines to a GeoJSON FeatureCollection.

    Args:
        contour_lines: List of dicts with 'elevation' and 'coordinates'

    Returns:
        GeoJSON FeatureCollection dict
    """
    features = []
    for contour in contour_lines:
        feature = {
            "type": "Feature",
            "properties": {
                "elevation": contour["elevation"],
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat] for lon, lat in contour["coordinates"]],
            },
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


# ─── Private Helpers ──────────────────────────────────────────────────────────


def _extract_kml_content(file_bytes: bytes, filename: str) -> bytes:
    """Extract KML XML content from either KML or KMZ file."""
    lower_name = filename.lower()

    if lower_name.endswith(".kmz"):
        return _extract_from_kmz(file_bytes)

    # Try as plain KML first
    if file_bytes[:4] in (b"<?xm", b"<kml", b"<Fol", b"\xef\xbb\xbf"):
        return file_bytes

    # Try as ZIP (might be KMZ with wrong extension)
    if file_bytes[:2] == b"PK":
        try:
            return _extract_from_kmz(file_bytes)
        except Exception:
            pass

    # Assume it's KML text
    return file_bytes


def _extract_from_kmz(file_bytes: bytes) -> bytes:
    """Extract the .kml file from a KMZ (ZIP) archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            # Look for .kml files in the archive
            kml_files = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_files:
                raise ValueError("KMZ archive does not contain any .kml files")
            # Use doc.kml if present (standard), otherwise the first .kml
            target = "doc.kml" if "doc.kml" in kml_files else kml_files[0]
            return zf.read(target)
    except zipfile.BadZipFile:
        raise ValueError("File appears to be KMZ but is not a valid ZIP archive")


def _extract_contour_lines(root: etree._Element) -> list:
    """
    Extract all contour lines from parsed KML XML.
    Walks all Placemark elements, looking for LineString geometries
    with a numeric elevation value in the <name> tag.
    """
    contour_lines = []

    # Find all Placemarks regardless of nesting depth
    # Handle both namespaced and non-namespaced KML
    placemarks = root.findall(f".//{KML_NS}Placemark")
    if not placemarks:
        placemarks = root.findall(".//Placemark")

    for pm in placemarks:
        # Extract elevation from <name> tag
        elevation = _extract_elevation(pm)
        if elevation is None:
            continue

        # Extract coordinates from <LineString>
        coords = _extract_linestring_coords(pm)
        if not coords:
            continue

        contour_lines.append({
            "elevation": elevation,
            "coordinates": coords,
        })

    return contour_lines


def _extract_elevation(placemark: etree._Element) -> float | None:
    """
    Extract elevation value from a Placemark.
    Tries: <name> tag, <description> tag, <SimpleData name="elevation">.
    Returns None if no numeric elevation found.
    """
    # Try <name> tag (most common for contour exports)
    for ns in [KML_NS, ""]:
        name_el = placemark.find(f"{ns}name")
        if name_el is not None and name_el.text:
            val = _try_parse_elevation(name_el.text.strip())
            if val is not None:
                return val

    # Try <description> tag
    for ns in [KML_NS, ""]:
        desc_el = placemark.find(f"{ns}description")
        if desc_el is not None and desc_el.text:
            val = _try_parse_elevation(desc_el.text.strip())
            if val is not None:
                return val

    # Try ExtendedData / SimpleData
    for ns in [KML_NS, ""]:
        for sd in placemark.findall(f".//{ns}SimpleData"):
            attr_name = sd.get("name", "").lower()
            if attr_name in ("elevation", "elev", "height", "contour", "z"):
                if sd.text:
                    val = _try_parse_elevation(sd.text.strip())
                    if val is not None:
                        return val

    return None


def _try_parse_elevation(text: str) -> float | None:
    """Try to parse a numeric elevation from a string."""
    # Direct float parse
    try:
        return float(text)
    except ValueError:
        pass

    # Extract number from strings like "contour_277" or "277m"
    match = re.search(r"[-+]?\d+\.?\d*", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            pass

    return None


def _extract_linestring_coords(placemark: etree._Element) -> list:
    """
    Extract (lon, lat) coordinate pairs from the first LineString in a Placemark.
    Handles the standard KML coordinate format: "lon,lat[,alt] lon,lat[,alt] ..."
    """
    for ns in [KML_NS, ""]:
        ls = placemark.find(f".//{ns}LineString/{ns}coordinates")
        if ls is not None and ls.text:
            return _parse_coordinate_string(ls.text)

    return []


def _parse_coordinate_string(coord_text: str) -> list:
    """
    Parse a KML coordinate string into (lon, lat) pairs.
    Format: "lon,lat[,alt] lon,lat[,alt] ..."
    """
    coords = []
    # Split by whitespace (spaces, newlines, etc.)
    tuples = coord_text.strip().split()

    for t in tuples:
        parts = t.split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                coords.append((lon, lat))
            except ValueError:
                continue

    return coords
