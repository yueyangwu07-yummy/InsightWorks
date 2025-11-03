"""Driving distance and route calculator tool for LangChain using Google Maps API.

This module provides driving distance calculations with real-time traffic information,
alternative routes, and turn-by-turn directions using Google Maps Directions API.
Falls back to mock data if API key is not configured.
"""

import hashlib
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

# Import straight distance tool for comparison
try:
    from .straight_distance import (
        find_city_coordinates,
        haversine_distance,
        parse_coordinates as parse_coords_straight,
    )
    STRAIGHT_DISTANCE_AVAILABLE = True
except ImportError:
    STRAIGHT_DISTANCE_AVAILABLE = False
    find_city_coordinates = None
    parse_coords_straight = None


# ============================================================================
# CONFIGURATION
# ============================================================================

GOOGLE_MAPS_API_URL = "https://maps.googleapis.com/maps/api/directions/json"
API_TIMEOUT = 10.0  # seconds
CACHE_TTL = 300  # 5 minutes in seconds

# Simple in-memory cache (in production, use Redis or similar)
_route_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def get_cache_key(origin: str, destination: str, mode: str) -> str:
    """Generate cache key for route request.
    
    Args:
        origin: Origin location
        destination: Destination location
        mode: Travel mode
        
    Returns:
        Cache key string
    """
    key_string = f"{origin}|{destination}|{mode}".lower().strip()
    return hashlib.md5(key_string.encode()).hexdigest()


def get_cached_route(origin: str, destination: str, mode: str) -> Optional[Dict[str, Any]]:
    """Get cached route if available and not expired.
    
    Args:
        origin: Origin location
        destination: Destination location
        mode: Travel mode
        
    Returns:
        Cached route data or None if not found/expired
    """
    cache_key = get_cache_key(origin, destination, mode)
    
    if cache_key in _route_cache:
        timestamp, data = _route_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return data
        else:
            # Remove expired entry
            del _route_cache[cache_key]
    
    return None


def cache_route(origin: str, destination: str, mode: str, data: Dict[str, Any]) -> None:
    """Cache route data.
    
    Args:
        origin: Origin location
        destination: Destination location
        mode: Travel mode
        data: Route data to cache
    """
    cache_key = get_cache_key(origin, destination, mode)
    _route_cache[cache_key] = (time.time(), data)


# ============================================================================
# GOOGLE MAPS API INTEGRATION
# ============================================================================

def parse_coordinates(location: str) -> Optional[str]:
    """Check if location string contains coordinates.
    
    Args:
        location: Location string
        
    Returns:
        Formatted coordinate string if valid, None otherwise
    """
    # Try to parse as "lat,lon" or "lat, lon"
    parts = [p.strip() for p in location.split(",")]
    if len(parts) == 2:
        try:
            lat = float(parts[0])
            lon = float(parts[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return f"{lat},{lon}"
        except (ValueError, TypeError):
            pass
    return None


async def fetch_google_maps_directions(
    origin: str,
    destination: str,
    mode: str = "driving",
    api_key: Optional[str] = None,
    alternatives: bool = True,
) -> Optional[Dict[str, Any]]:
    """Fetch directions from Google Maps Directions API.
    
    Args:
        origin: Starting location (address or coordinates)
        destination: Ending location (address or coordinates)
        mode: Travel mode (driving, walking, bicycling, transit)
        api_key: Google Maps API key
        alternatives: Whether to return alternative routes
        
    Returns:
        API response as dictionary or None if error
    """
    if not HTTPX_AVAILABLE:
        return None
    
    if not api_key:
        return None
    
    # Prepare parameters
    params = {
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "alternatives": "true" if alternatives else "false",
        "units": "imperial",  # miles, not km
        "departure_time": "now",  # for traffic-aware routing
        "traffic_model": "best_guess",  # best_guess, pessimistic, optimistic
        "key": api_key,
    }
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(GOOGLE_MAPS_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check API status
            status = data.get("status", "")
            if status == "OK":
                return data
            elif status == "ZERO_RESULTS":
                return {"status": "ZERO_RESULTS", "error": "No route found between these locations"}
            elif status == "INVALID_REQUEST":
                return {"status": "INVALID_REQUEST", "error": "Invalid request parameters"}
            elif status == "OVER_QUERY_LIMIT":
                return {"status": "OVER_QUERY_LIMIT", "error": "API query limit exceeded"}
            elif status == "REQUEST_DENIED":
                return {"status": "REQUEST_DENIED", "error": "API request denied (check API key)"}
            else:
                return {"status": status, "error": f"API error: {status}"}
    except httpx.TimeoutException:
        return {"status": "TIMEOUT", "error": "API request timeout"}
    except httpx.HTTPError as e:
        return {"status": "HTTP_ERROR", "error": f"HTTP error: {str(e)}"}
    except Exception as e:
        return {"status": "ERROR", "error": f"Unexpected error: {str(e)}"}


def parse_route(route_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse Google Maps route data into structured format.
    
    Args:
        route_data: Raw route data from Google Maps API
        
    Returns:
        List of parsed route dictionaries
    """
    routes = []
    
    if "routes" not in route_data:
        return routes
    
    for route in route_data["routes"]:
        leg = route["legs"][0]  # First leg (usually only one for point-to-point)
        
        # Extract distance and duration
        distance_text = leg["distance"]["text"]
        distance_value = leg["distance"]["value"]  # in meters
        
        duration_text = leg["duration"]["text"]
        duration_value = leg["duration"]["value"]  # in seconds
        
        # Check for traffic duration
        duration_in_traffic = None
        duration_in_traffic_text = None
        if "duration_in_traffic" in leg:
            duration_in_traffic = leg["duration_in_traffic"]["value"]
            duration_in_traffic_text = leg["duration_in_traffic"]["text"]
        
        # Extract route summary
        summary = route.get("summary", "Unknown route")
        
        # Extract step-by-step directions (highway names)
        steps = leg.get("steps", [])
        highways = []
        for step in steps:
            html_instructions = step.get("html_instructions", "")
            # Extract highway names from instructions
            # Simple extraction: look for "I-", "US ", "Route", etc.
            if "I-" in html_instructions or "US " in html_instructions or "Route" in html_instructions:
                # Extract highway reference
                import re
                highway_match = re.search(r"(I-\d+|US \d+|Route \d+)", html_instructions)
                if highway_match:
                    highway = highway_match.group(1)
                    if highway not in highways:
                        highways.append(highway)
        
        route_info = {
            "distance_text": distance_text,
            "distance_meters": distance_value,
            "distance_miles": distance_value / 1609.34,  # Convert meters to miles
            "duration_text": duration_text,
            "duration_seconds": duration_value,
            "duration_in_traffic_seconds": duration_in_traffic,
            "duration_in_traffic_text": duration_in_traffic_text,
            "summary": summary,
            "highways": highways if highways else [summary],
            "steps_count": len(steps),
        }
        
        routes.append(route_info)
    
    # Sort by duration (fastest first)
    routes.sort(key=lambda x: x.get("duration_in_traffic_seconds") or x["duration_seconds"])
    
    return routes


def resolve_location(location: str) -> str:
    """Resolve location to coordinates or address string.
    
    Args:
        location: Location string (coordinates or city name)
        
    Returns:
        Coordinate string if found, or original location string
    """
    # First try parsing as coordinates
    coords = parse_coordinates(location)
    if coords:
        return f"{coords[0]},{coords[1]}"
    
    # Try finding city coordinates
    if STRAIGHT_DISTANCE_AVAILABLE and find_city_coordinates:
        city_coords = find_city_coordinates(location)
        if city_coords:
            lat, lon = city_coords
            return f"{lat},{lon}"
    
    # Return original location (will be used as address by Google Maps)
    return location


def calculate_straight_distance(origin: str, destination: str) -> Optional[float]:
    """Calculate straight-line distance for comparison.
    
    Args:
        origin: Origin location
        destination: Destination location
        
    Returns:
        Distance in miles or None if calculation fails
    """
    if not STRAIGHT_DISTANCE_AVAILABLE:
        return None
    
    # Resolve locations to coordinates
    origin_resolved = resolve_location(origin)
    dest_resolved = resolve_location(destination)
    
    if not origin_resolved or not dest_resolved:
        return None
    
    # Check if resolved to coordinates
    origin_coords = parse_coords_straight(origin_resolved) if parse_coords_straight else None
    dest_coords = parse_coords_straight(dest_resolved) if parse_coords_straight else None
    
    if not origin_coords or not dest_coords:
        return None
    
    try:
        lat1, lon1 = origin_coords
        lat2, lon2 = dest_coords
        distance_miles, _ = haversine_distance(lat1, lon1, lat2, lon2)
        return distance_miles
    except Exception:
        return None


def generate_mock_data(origin: str, destination: str, mode: str = "driving") -> Dict[str, Any]:
    """Generate mock route data when API is unavailable.
    
    Uses straight-line distance * 1.3 as estimate for driving distance.
    
    Args:
        origin: Origin location
        destination: Destination location
        mode: Travel mode
        
    Returns:
        Mock route data
    """
    # Calculate straight distance for comparison (will resolve locations internally)
    straight_dist = calculate_straight_distance(origin, destination)
    
    if straight_dist:
        # Estimate: driving distance is ~30% longer than straight-line
        driving_dist_miles = straight_dist * 1.3
        driving_dist_meters = driving_dist_miles * 1609.34
        
        # Estimate duration: assume average speed based on mode
        if mode == "walking":
            avg_speed_mph = 3  # walking speed
        elif mode == "bicycling":
            avg_speed_mph = 12  # biking speed
        elif mode == "transit":
            avg_speed_mph = 35  # transit average
        else:  # driving
            # Highway vs city driving estimate
            if driving_dist_miles > 50:
                avg_speed_mph = 60  # highway
            elif driving_dist_miles > 10:
                avg_speed_mph = 45  # mixed
            else:
                avg_speed_mph = 30  # city
        
        duration_hours = driving_dist_miles / avg_speed_mph
        duration_minutes = int(duration_hours * 60)
        
        if duration_minutes < 60:
            duration_text = f"{duration_minutes} min"
        else:
            hours = duration_minutes // 60
            mins = duration_minutes % 60
            if mins > 0:
                duration_text = f"{hours}h {mins}min"
            else:
                duration_text = f"{hours} hour{'s' if hours > 1 else ''}"
        
        return {
            "routes": [
                {
                    "distance_text": f"{driving_dist_miles:.1f} mi",
                    "distance_meters": int(driving_dist_miles * 1609.34),
                    "distance_miles": driving_dist_miles,
                    "duration_text": duration_text,
                    "duration_seconds": duration_minutes * 60,
                    "duration_in_traffic_seconds": None,
                    "duration_in_traffic_text": None,
                    "summary": "Estimated route",
                    "highways": [],
                    "steps_count": 0,
                }
            ],
            "straight_distance": straight_dist,
            "mock": True,
        }
    else:
        # Fallback if can't calculate straight distance
        return {
            "routes": [
                {
                    "distance_text": "Unknown",
                    "distance_meters": 0,
                    "distance_miles": 0,
                    "duration_text": "Unknown",
                    "duration_seconds": 0,
                    "summary": "Route unavailable",
                    "highways": [],
                    "steps_count": 0,
                }
            ],
            "straight_distance": None,
            "mock": True,
        }


# ============================================================================
# TOOL IMPLEMENTATION
# ============================================================================

class DrivingDistanceInput(BaseModel):
    """Input schema for driving distance calculator."""
    input_str: str = Field(description="Route query: 'origin to destination' or 'origin, destination'. Example: 'New York to Boston'")


class DrivingDistanceTool(BaseTool):
    """Tool for calculating driving distance and routes using Google Maps API.
    
    Provides actual road distance, estimated duration with traffic,
    alternative routes, and turn-by-turn directions.
    Falls back to estimated data if API key is not configured.
    """
    
    name: str = "driving_distance"
    description: str = (
        "Calculates driving distance and route between two locations using Google Maps. "
        "Input: 'origin to destination' or 'origin, destination'. "
        "Returns actual road distance, estimated duration with traffic, route summary, "
        "and alternative routes if available. "
        "Example: 'New York to Boston' or '40.7128,-74.0060 to 42.3601,-71.0589'. "
        "Requires GOOGLE_MAPS_API_KEY environment variable for real-time data."
    )
    args_schema: Type[BaseModel] = DrivingDistanceInput
    
    def _run(self, input_str: str) -> str:
        """Synchronous execution (required by LangChain)."""
        import asyncio
        return asyncio.run(self._arun(input_str))
    
    async def _arun(self, input_str: str) -> str:
        """Asynchronous execution."""
        try:
            # Parse input
            # Format: "origin to destination" or "origin, destination"
            input_str = input_str.strip()
            mode = "driving"  # Default mode
            alternatives = True  # Default
            
            # Try to extract mode from input string if present
            input_lower = input_str.lower()
            if "mode=" in input_lower or "mode " in input_lower:
                # Parse mode from input
                mode_match = re.search(r"mode[=:]\s*(\w+)", input_lower)
                if mode_match:
                    mode = mode_match.group(1)
            
            # Try "to" separator first
            if " to " in input_str.lower():
                parts = input_str.split(" to ", 1)
                origin = parts[0].strip()
                destination = parts[1].strip()
                # Remove mode from destination if present
                if "mode" in destination.lower():
                    destination = re.sub(r"\s*mode[=:]\s*\w+", "", destination, flags=re.IGNORECASE).strip()
            elif ", " in input_str and input_str.count(", ") == 1:
                # Format: "origin, destination"
                parts = input_str.split(", ", 1)
                origin = parts[0].strip()
                destination = parts[1].strip()
                # Remove mode from destination if present
                if "mode" in destination.lower():
                    destination = re.sub(r"\s*mode[=:]\s*\w+", "", destination, flags=re.IGNORECASE).strip()
            else:
                return (
                    "[ERROR] Invalid input format. "
                    "Expected: 'origin to destination' or 'origin, destination'. "
                    f"Got: '{input_str}'"
                )
            
            if not origin or not destination:
                return "[ERROR] Both origin and destination are required."
            
            # Validate mode
            valid_modes = ["driving", "walking", "bicycling", "transit"]
            if mode not in valid_modes:
                mode = "driving"  # Fallback to default
            
            # Check cache first
            cached_data = get_cached_route(origin, destination, mode)
            if cached_data:
                return self._format_output(origin, destination, cached_data, mode)
            
            # Get API key
            api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
            
            # Resolve locations (convert city names to coordinates if possible, or use as-is)
            origin_resolved = resolve_location(origin)
            dest_resolved = resolve_location(destination)
            
            # Fetch directions
            if api_key:
                route_data = await fetch_google_maps_directions(
                    origin_resolved, dest_resolved, mode, api_key, alternatives
                )
            else:
                route_data = None
            
            # Handle API errors or missing key
            if not route_data:
                # Fallback to mock data
                parsed_routes = generate_mock_data(origin, destination, mode)
                parsed_routes["api_error"] = "No API key configured" if not api_key else "API request failed"
            elif "status" in route_data and route_data["status"] != "OK":
                error_msg = route_data.get("error", "Unknown error")
                if route_data["status"] == "ZERO_RESULTS":
                    return f"[ERROR] No route found between '{origin}' and '{destination}'."
                elif route_data["status"] == "OVER_QUERY_LIMIT":
                    # Fallback to mock data with warning
                    parsed_routes = generate_mock_data(origin, destination, mode)
                    parsed_routes["api_error"] = f"API quota exceeded: {error_msg}"
                else:
                    # Fallback to mock data
                    parsed_routes = generate_mock_data(origin, destination, mode)
                    parsed_routes["api_error"] = f"API error: {error_msg}"
            else:
                # Parse successful response
                parsed_routes = parse_route(route_data)
                # Calculate straight distance for comparison
                straight_dist = calculate_straight_distance(origin_resolved, dest_resolved)
                parsed_routes = {
                    "routes": parsed_routes,
                    "straight_distance": straight_dist,
                    "mock": False,
                }
            
            # Cache the result
            cache_route(origin, destination, mode, parsed_routes)
            
            return self._format_output(origin, destination, parsed_routes, mode)
        except Exception as e:
            return f"Error calculating driving distance: {str(e)}"
    
    def _format_output(
        self,
        origin: str,
        destination: str,
        route_data: Dict[str, Any],
        mode: str,
    ) -> str:
        """Format route data into readable output.
        
        Args:
            origin: Origin location
            destination: Destination location
            route_data: Parsed route data
            mode: Travel mode
            
        Returns:
            Formatted output string
        """
        output_parts = []
        output_parts.append(f"Driving distance from {origin} to {destination}:")
        output_parts.append("")
        
        routes = route_data.get("routes", [])
        if not routes:
            return "[ERROR] No routes found."
        
        # Show API warning if applicable
        if route_data.get("mock"):
            output_parts.append(
                "[INFO] Using estimated data. "
                "Set GOOGLE_MAPS_API_KEY for real-time route information."
            )
            if route_data.get("api_error"):
                output_parts.append(f"[WARNING] {route_data['api_error']}")
            output_parts.append("")
        
        # Display routes
        for i, route in enumerate(routes, 1):
            if i == 1:
                output_parts.append(f"Route {i} (Recommended): {route['summary']}")
            else:
                output_parts.append(f"Route {i} (Alternative): {route['summary']}")
            
            output_parts.append(f"  Distance: {route['distance_text']}")
            
            # Duration with traffic if available
            if route.get("duration_in_traffic_text"):
                base_duration = route["duration_text"]
                traffic_duration = route["duration_in_traffic_text"]
                traffic_seconds = route.get("duration_in_traffic_seconds", 0)
                base_seconds = route["duration_seconds"]
                delay_minutes = (traffic_seconds - base_seconds) // 60
                
                if delay_minutes > 0:
                    output_parts.append(
                        f"  Duration: {traffic_duration} (add {delay_minutes} min due to traffic)"
                    )
                else:
                    output_parts.append(f"  Duration: {traffic_duration} (current traffic)")
            else:
                output_parts.append(f"  Duration: {route['duration_text']}")
            
            # Route summary (highways)
            if route.get("highways"):
                highways_str = " → ".join(route["highways"])
                output_parts.append(f"  Route: {highways_str}")
            
            output_parts.append("")
        
        # Comparison with straight-line distance
        straight_dist = route_data.get("straight_distance")
        if straight_dist and routes:
            driving_dist = routes[0]["distance_miles"]
            if driving_dist > 0:
                diff_miles = driving_dist - straight_dist
                diff_percent = (diff_miles / straight_dist) * 100
                
                output_parts.append("Comparison:")
                output_parts.append(f"  Straight-line distance: {straight_dist:.1f} miles")
                output_parts.append(f"  Actual driving distance: {driving_dist:.1f} miles")
                output_parts.append(
                    f"  Difference: {diff_miles:.1f} miles ({diff_percent:.0f}% longer)"
                )
                output_parts.append(
                    "  Reason: Road network, avoiding water/obstacles, following highways"
                )
                output_parts.append("")
        
        # Traffic information
        if routes and routes[0].get("duration_in_traffic_text"):
            output_parts.append(
                "Traffic: Real-time traffic data included in duration estimates."
            )
        
        return "\n".join(output_parts)


# Create tool instance
driving_distance_tool = DrivingDistanceTool()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_driving_distance():
        """Test the driving distance calculator."""
        tool = DrivingDistanceTool()
        
        test_cases = [
            ("New York to Boston", "NYC to Boston"),
            ("Los Angeles to San Francisco", "LA to SF"),
            ("40.7128,-74.0060 to 42.3601,-71.0589", "NYC to Boston (coordinates)"),
        ]
        
        print("=" * 70)
        print("DRIVING DISTANCE CALCULATOR TEST")
        print("=" * 70)
        
        for input_str, description in test_cases:
            print(f"\n{description}: {input_str}")
            print("-" * 70)
            result = await tool._arun(input_str)
            print(result)
            print()
        
        print("\n" + "=" * 70)
        print("MOCK DATA TEST (no API key)")
        print("=" * 70)
        
        # Temporarily remove API key to test mock data
        original_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if original_key:
            del os.environ["GOOGLE_MAPS_API_KEY"]
        
        result = await tool._arun("New York to Boston")
        print(result)
        
        # Restore API key
        if original_key:
            os.environ["GOOGLE_MAPS_API_KEY"] = original_key
    
    asyncio.run(test_driving_distance())

