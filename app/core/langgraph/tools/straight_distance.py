"""Straight-line distance calculator tool for LangChain.

This module provides a straight-line (as the crow flies) distance calculator
using the Haversine formula. Works completely offline with no API required.
Supports both coordinate-based and city name-based input.
"""

import math
import re
from typing import Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type


# ============================================================================
# MAJOR US CITIES COORDINATES
# ============================================================================

MAJOR_US_CITIES: Dict[str, Tuple[float, float]] = {
    # Top 20 largest cities
    "new york": (40.7128, -74.0060),
    "new york city": (40.7128, -74.0060),
    "nyc": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "la": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "houston": (29.7604, -95.3698),
    "phoenix": (33.4484, -112.0740),
    "philadelphia": (39.9526, -75.1652),
    "san antonio": (29.4241, -98.4936),
    "san diego": (32.7157, -117.1611),
    "dallas": (32.7767, -96.7970),
    "san jose": (37.3382, -121.8863),
    "san francisco": (37.7749, -122.4194),
    "sf": (37.7749, -122.4194),
    "austin": (30.2672, -97.7431),
    "jacksonville": (30.3322, -81.6557),
    "fort worth": (32.7555, -97.3308),
    "columbus": (39.9612, -82.9988),
    "charlotte": (35.2271, -80.8431),
    "san francisco ca": (37.7749, -122.4194),
    
    # Additional major cities
    "boston": (42.3601, -71.0589),
    "seattle": (47.6062, -122.3321),
    "miami": (25.7617, -80.1918),
    "atlanta": (33.7490, -84.3880),
    "denver": (39.7392, -104.9903),
    "washington": (38.9072, -77.0369),
    "washington dc": (38.9072, -77.0369),
    "dc": (38.9072, -77.0369),
    "detroit": (42.3314, -83.0458),
    "minneapolis": (44.9778, -93.2650),
    "tampa": (27.9506, -82.4572),
    "st. louis": (38.6270, -90.1994),
    "st louis": (38.6270, -90.1994),
    "baltimore": (39.2904, -76.6122),
    "el paso": (31.7619, -106.4850),
    "milwaukee": (43.0389, -87.9065),
    "oakland": (37.8044, -122.2712),
    "arizona": (34.0489, -111.0937),
    "kansas city": (39.0997, -94.5786),
    "raleigh": (35.7796, -78.6382),
    "omaha": (41.2565, -95.9345),
    "miami fl": (25.7617, -80.1918),
    "miami florida": (25.7617, -80.1918),
    "long beach": (33.7701, -118.1937),
    "virginia beach": (36.8529, -75.9780),
    "oakland ca": (37.8044, -122.2712),
    "minneapolis mn": (44.9778, -93.2650),
    "tulsa": (36.1540, -95.9928),
    "tampa fl": (27.9506, -82.4572),
    "new orleans": (29.9511, -90.0715),
    "cleveland": (41.4993, -81.6944),
    "wichita": (37.6872, -97.3301),
    
    # West Coast
    "portland": (45.5152, -122.6784),
    "portland or": (45.5152, -122.6784),
    "sacramento": (38.5816, -121.4944),
    "las vegas": (36.1699, -115.1398),
    "vegas": (36.1699, -115.1398),
    "fresno": (36.7378, -119.7871),
    
    # East Coast
    "norfolk": (36.8468, -76.2852),
    "orlando": (28.5383, -81.3792),
    "buffalo": (42.8864, -78.8784),
    "newark": (40.7357, -74.1724),
    "lincoln": (40.8136, -96.7026),
    
    # Midwest
    "cincinnati": (39.1031, -84.5120),
    "pittsburgh": (40.4406, -79.9959),
    "indianapolis": (39.7684, -86.1581),
    "toledo": (41.6528, -83.5379),
    
    # South
    "nashville": (36.1627, -86.7816),
    "memphis": (35.1495, -90.0490),
    "oklahoma city": (35.4676, -97.5164),
    "okc": (35.4676, -97.5164),
    "louisville": (38.2527, -85.7585),
    "richmond": (37.5407, -77.4360),
    
    # Texas cities
    "fort worth tx": (32.7555, -97.3308),
    "el paso tx": (31.7619, -106.4850),
    "arlington": (32.7357, -97.1081),
    "corpus christi": (27.8006, -97.3964),
    "plano": (33.0198, -96.6989),
    "laredo": (27.5306, -99.4803),
    
    # California cities
    "fresno ca": (36.7378, -119.7871),
    "sacramento ca": (38.5816, -121.4944),
    "long beach ca": (33.7701, -118.1937),
    "oakland ca": (37.8044, -122.2712),
    
    # Florida cities
    "jacksonville fl": (30.3322, -81.6557),
    "tampa florida": (27.9506, -82.4572),
    "orlando fl": (28.5383, -81.3792),
    
    # New York area
    "brooklyn": (40.6782, -73.9442),
    "queens": (40.7282, -73.7949),
    "bronx": (40.8448, -73.8648),
    "manhattan": (40.7831, -73.9712),
}


# ============================================================================
# HAVERSINE FORMULA
# ============================================================================

def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> Tuple[float, float]:
    """Calculate great circle distance between two points using Haversine formula.
    
    The Haversine formula calculates the shortest distance over the earth's surface
    (as the crow flies) between two points on a sphere given their longitudes and latitudes.
    
    Formula:
        a = sin²(Δφ/2) + cos(φ1) * cos(φ2) * sin²(Δλ/2)
        c = 2 * atan2(√a, √(1-a))
        d = R * c
    
    where:
        φ1, φ2: latitude of point 1 and 2 (in radians)
        Δφ: difference in latitude
        Δλ: difference in longitude
        R: Earth's radius (3959 miles or 6371 km)
    
    Args:
        lat1: Latitude of first point (degrees)
        lon1: Longitude of first point (degrees)
        lat2: Latitude of second point (degrees)
        lon2: Longitude of second point (degrees)
        
    Returns:
        Tuple of (distance_in_miles, distance_in_km)
    """
    # Validate latitude/longitude ranges
    if not (-90 <= lat1 <= 90) or not (-90 <= lat2 <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not (-180 <= lon1 <= 180) or not (-180 <= lon2 <= 180):
        raise ValueError("Longitude must be between -180 and 180 degrees")
    
    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Earth's radius in miles and kilometers
    R_miles = 3959
    R_km = 6371
    
    distance_miles = R_miles * c
    distance_km = R_km * c
    
    return distance_miles, distance_km


# ============================================================================
# BEARING CALCULATION
# ============================================================================

def calculate_bearing(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> Tuple[float, str]:
    """Calculate initial bearing (compass direction) between two points.
    
    Formula:
        θ = atan2(sin(Δλ)*cos(φ2), cos(φ1)*sin(φ2) - sin(φ1)*cos(φ2)*cos(Δλ))
    
    where:
        φ1, φ2: latitude of point 1 and 2 (in radians)
        Δλ: difference in longitude (in radians)
    
    Args:
        lat1: Latitude of origin point (degrees)
        lon1: Longitude of origin point (degrees)
        lat2: Latitude of destination point (degrees)
        lon2: Longitude of destination point (degrees)
        
    Returns:
        Tuple of (bearing_in_degrees, compass_direction)
    """
    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    
    # Calculate bearing
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    )
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    
    # Normalize to 0-360 degrees
    bearing_deg = (bearing_deg + 360) % 360
    
    # Convert to compass direction
    compass_dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = int((bearing_deg + 22.5) / 45) % 8
    compass_dir = compass_dirs[index]
    
    return bearing_deg, compass_dir


# ============================================================================
# INPUT PARSING
# ============================================================================

def parse_coordinates(coord_str: str) -> Optional[Tuple[float, float]]:
    """Parse coordinate string into (latitude, longitude) tuple.
    
    Supports formats:
    - "40.7128,-74.0060"
    - "40.7128, -74.0060"
    - "40.7128, -74.0060"
    
    Args:
        coord_str: Coordinate string
        
    Returns:
        Tuple of (latitude, longitude) or None if invalid
    """
    # Remove whitespace
    coord_str = coord_str.strip()
    
    # Split by comma
    parts = [p.strip() for p in coord_str.split(",")]
    
    if len(parts) != 2:
        return None
    
    try:
        lat = float(parts[0])
        lon = float(parts[1])
        
        # Basic validation
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lat, lon)
        return None
    except (ValueError, TypeError):
        return None


def find_city_coordinates(city_name: str) -> Optional[Tuple[float, float]]:
    """Find coordinates for a city name (case-insensitive).
    
    Args:
        city_name: City name string
        
    Returns:
        Tuple of (latitude, longitude) or None if not found
    """
    city_key = city_name.lower().strip()
    
    # Direct lookup
    if city_key in MAJOR_US_CITIES:
        return MAJOR_US_CITIES[city_key]
    
    # Try with state abbreviations removed
    city_key_no_state = re.sub(r",?\s*(ca|ny|tx|fl|il|pa|ma|wa|co|az|nc|ga|mi|tn|nv|or|mn|mo|ok|la|ky|oh|in|ct|nj|sc|ks|md|wi|ct|va|ar|ut|nv|hi|ri|me|nh|id|de|sd|nd|mt|ak|wy|nm|vt|wv|dc)\b", "", city_key).strip()
    if city_key_no_state in MAJOR_US_CITIES:
        return MAJOR_US_CITIES[city_key_no_state]
    
    # Try fuzzy matching (simple substring match)
    for key, coords in MAJOR_US_CITIES.items():
        if city_key in key or key in city_key:
            return coords
    
    return None


def suggest_similar_cities(city_name: str, limit: int = 5) -> List[str]:
    """Suggest similar city names if exact match not found.
    
    Args:
        city_name: City name to find suggestions for
        limit: Maximum number of suggestions
        
    Returns:
        List of suggested city names
    """
    city_lower = city_name.lower().strip()
    suggestions = []
    
    # Simple substring matching
    for key in MAJOR_US_CITIES.keys():
        if city_lower in key or key in city_lower:
            suggestions.append(key.title())
            if len(suggestions) >= limit:
                break
    
    return suggestions[:limit]


def parse_input(input_str: str) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], str, str]:
    """Parse input string to extract origin and destination coordinates.
    
    Supports formats:
    - "lat1,lon1 to lat2,lon2"
    - "City1 to City2"
    - "from City1 to City2"
    - Coordinates and city names mixed
    
    Args:
        input_str: Input string
        
    Returns:
        Tuple of (origin_coords, dest_coords, origin_name, dest_name)
        Returns (None, None, "", "") if parsing fails
    """
    input_str = input_str.strip()
    
    # Remove "from" if present
    input_str = re.sub(r"^from\s+", "", input_str, flags=re.IGNORECASE)
    
    # Split by "to"
    if " to " not in input_str.lower():
        return (None, None, "", "")
    
    parts = re.split(r"\s+to\s+", input_str, flags=re.IGNORECASE)
    if len(parts) != 2:
        return (None, None, "", "")
    
    origin_str = parts[0].strip()
    dest_str = parts[1].strip()
    
    # Try to parse as coordinates first
    origin_coords = parse_coordinates(origin_str)
    dest_coords = parse_coordinates(dest_str)
    
    # If not coordinates, try city names
    origin_name = origin_str
    dest_name = dest_str
    
    if origin_coords is None:
        origin_coords = find_city_coordinates(origin_str)
    
    if dest_coords is None:
        dest_coords = find_city_coordinates(dest_str)
    
    return (origin_coords, dest_coords, origin_name, dest_name)


# ============================================================================
# TOOL IMPLEMENTATION
# ============================================================================

class StraightDistanceInput(BaseModel):
    """Input schema for straight distance calculator."""
    input_str: str = Field(description="Route query: 'City1 to City2' or 'lat1,lon1 to lat2,lon2'. Example: 'New York to Boston'")


class StraightDistanceTool(BaseTool):
    """Tool for calculating straight-line distance between two locations.
    
    Uses Haversine formula to calculate great circle distance (as the crow flies).
    Works completely offline with no API required.
    
    Supports both coordinate-based input (lat,lon) and city name input.
    Includes ~100 major US cities in the database.
    """
    
    name: str = "straight_distance"
    description: str = (
        "Calculates straight-line distance (as the crow flies) between two locations. "
        "Input: 'lat1,lon1 to lat2,lon2' or 'City1 to City2'. "
        "Returns distance in miles and kilometers, compass direction, and estimated flight time. "
        "Example: 'New York to Boston' or '40.7128,-74.0060 to 42.3601,-71.0589'. "
        "This calculates the shortest distance over the earth's surface, not driving distance."
    )
    args_schema: Type[BaseModel] = StraightDistanceInput
    
    def _run(self, input_str: str) -> str:
        """Synchronous execution (required by LangChain)."""
        return self._arun(input_str)
    
    async def _arun(self, input_str: str) -> str:
        """Asynchronous execution."""
        try:
            # Parse input
            origin_coords, dest_coords, origin_name, dest_name = parse_input(input_str)
            
            # Validate input
            if origin_coords is None:
                suggestions = suggest_similar_cities(origin_name)
                if suggestions:
                    return (
                        f"[ERROR] Could not find location: '{origin_name}'\n"
                        f"Did you mean: {', '.join(suggestions)}?"
                    )
                else:
                    return (
                        f"[ERROR] Could not parse origin location: '{origin_name}'\n"
                        f"Please provide coordinates (lat,lon) or a recognized city name."
                    )
            
            if dest_coords is None:
                suggestions = suggest_similar_cities(dest_name)
                if suggestions:
                    return (
                        f"[ERROR] Could not find location: '{dest_name}'\n"
                        f"Did you mean: {', '.join(suggestions)}?"
                    )
                else:
                    return (
                        f"[ERROR] Could not parse destination location: '{dest_name}'\n"
                        f"Please provide coordinates (lat,lon) or a recognized city name."
                    )
            
            lat1, lon1 = origin_coords
            lat2, lon2 = dest_coords
            
            # Check if same location
            if abs(lat1 - lat2) < 0.0001 and abs(lon1 - lon2) < 0.0001:
                return (
                    f"[INFO] Origin and destination are the same location.\n"
                    f"Distance: 0 miles (0 km)"
                )
            
            # Calculate distance
            try:
                distance_miles, distance_km = haversine_distance(lat1, lon1, lat2, lon2)
            except ValueError as e:
                return f"[ERROR] Invalid coordinates: {e}"
            
            # Calculate bearing
            bearing_deg, compass_dir = calculate_bearing(lat1, lon1, lat2, lon2)
            
            # Estimate flight time (assuming average commercial flight speed ~500 mph)
            # For short distances, assume slower speeds
            avg_speed_mph = 500 if distance_miles > 500 else 300
            flight_time_hours = distance_miles / avg_speed_mph
            flight_time_minutes = int(flight_time_hours * 60)
            
            # Format distance with appropriate precision
            if distance_miles < 1:
                dist_miles_str = f"{distance_miles * 5280:.0f} feet"
                dist_km_str = f"{distance_km * 1000:.0f} meters"
            elif distance_miles < 10:
                dist_miles_str = f"{distance_miles:.2f} miles"
                dist_km_str = f"{distance_km:.2f} km"
            else:
                dist_miles_str = f"{distance_miles:.1f} miles"
                dist_km_str = f"{distance_km:.1f} km"
            
            # Build output
            output_parts = []
            output_parts.append(
                f"Straight-line distance from {origin_name} to {dest_name}:"
            )
            output_parts.append("")
            output_parts.append(f"Distance: {dist_miles_str} ({dist_km_str})")
            output_parts.append(
                f"Direction: {compass_dir} ({bearing_deg:.1f} deg bearing)"
            )
            
            if flight_time_minutes > 0:
                if flight_time_minutes < 60:
                    output_parts.append(f"Estimated flight time: ~{flight_time_minutes} minutes")
                else:
                    hours = flight_time_minutes // 60
                    minutes = flight_time_minutes % 60
                    if minutes > 0:
                        output_parts.append(f"Estimated flight time: ~{hours}h {minutes}m")
                    else:
                        output_parts.append(f"Estimated flight time: ~{hours} hours")
            
            output_parts.append("")
            output_parts.append(
                "Note: This is straight-line distance (as the crow flies). "
                "Actual driving distance may be longer due to roads/terrain."
            )
            output_parts.append("Use the 'driving_distance' tool for road routing.")
            
            return "\n".join(output_parts)
        except Exception as e:
            return f"Error calculating straight distance: {str(e)}"


# Create tool instance
straight_distance_tool = StraightDistanceTool()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_straight_distance():
        """Test the straight distance calculator with various inputs."""
        tool = StraightDistanceTool()
        
        test_cases = [
            ("40.7128,-74.0060 to 42.3601,-71.0589", "NYC to Boston (coordinates)"),
            ("New York to Boston", "NYC to Boston (city names)"),
            ("New York to Los Angeles", "NYC to LA (city names)"),
            ("40.7128,-74.0060 to 34.0522,-118.2437", "NYC to LA (coordinates)"),
            ("Chicago to Miami", "Chicago to Miami"),
            ("Seattle to San Francisco", "Seattle to SF"),
        ]
        
        print("=" * 70)
        print("STRAIGHT DISTANCE CALCULATOR TEST")
        print("=" * 70)
        
        for input_str, description in test_cases:
            print(f"\n{description}: {input_str}")
            print("-" * 70)
            result = await tool._arun(input_str)
            print(result)
            print()
        
        print("\n" + "=" * 70)
        print("ERROR HANDLING TEST")
        print("=" * 70)
        
        error_cases = [
            ("Unknown City to Boston", "Unknown city"),
            ("New York", "Missing 'to' separator"),
            ("New York to New York", "Same origin and destination"),
        ]
        
        for input_str, description in error_cases:
            print(f"\n{description}: {input_str}")
            print("-" * 70)
            result = await tool._arun(input_str)
            print(result)
            print()
    
    asyncio.run(test_straight_distance())

