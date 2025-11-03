"""Traffic incident reporter tool for LangChain using HERE Traffic API.

This module provides real-time traffic incident information including accidents,
construction, road closures, and weather-related incidents. Uses HERE Traffic API
with support for area-based and route-based searches.
"""

import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

# Try to import driving distance tool for route integration
try:
    from .driving_distance import resolve_location
    DRIVING_DISTANCE_AVAILABLE = True
except ImportError:
    DRIVING_DISTANCE_AVAILABLE = False
    resolve_location = None


# ============================================================================
# CONFIGURATION
# ============================================================================

HERE_TRAFFIC_API_URL = "https://data.traffic.hereapi.com/v7/incidents"
API_TIMEOUT = 15.0  # seconds
CACHE_TTL = 120  # 2 minutes in seconds

# Simple in-memory cache
_incident_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


# ============================================================================
# INCIDENT TYPE MAPPING
# ============================================================================

INCIDENT_TYPE_MAP = {
    "ACCIDENT": "Accident",
    "CONGESTION": "Congestion",
    "CONSTRUCTION": "Construction",
    "ROAD_CLOSURE": "Road Closure",
    "ROAD_HAZARD": "Road Hazard",
    "WEATHER": "Weather",
    "DISABLED_VEHICLE": "Disabled Vehicle",
    "MASS_TRANSIT": "Transit Delay",
}

INCIDENT_TYPE_ICONS = {
    "ACCIDENT": "[!]",
    "CONGESTION": "[TRAFFIC]",
    "CONSTRUCTION": "[CONSTR]",
    "ROAD_CLOSURE": "[CLOSED]",
    "ROAD_HAZARD": "[HAZARD]",
    "WEATHER": "[WEATHER]",
    "DISABLED_VEHICLE": "[BROKEN]",
    "MASS_TRANSIT": "[TRANSIT]",
}

SEVERITY_MAP = {
    "CRITICAL": ("[CRITICAL]", "CRITICAL"),
    "MAJOR": ("[MAJOR]", "MAJOR"),
    "MODERATE": ("[MOD]", "MODERATE"),
    "MINOR": ("[MIN]", "MINOR"),
}


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def get_cache_key(location: str, radius: float, route: Optional[str] = None) -> str:
    """Generate cache key for incident request.
    
    Args:
        location: Location string
        radius: Search radius in miles
        route: Optional route string
        
    Returns:
        Cache key string
    """
    key_string = f"{location}|{radius}|{route or ''}".lower().strip()
    return key_string


def get_cached_incidents(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached incidents if available and not expired.
    
    Args:
        cache_key: Cache key
        
    Returns:
        Cached incident data or None if not found/expired
    """
    if cache_key in _incident_cache:
        timestamp, data = _incident_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return data
        else:
            del _incident_cache[cache_key]
    
    return None


def cache_incidents(cache_key: str, data: Dict[str, Any]) -> None:
    """Cache incident data.
    
    Args:
        cache_key: Cache key
        data: Incident data to cache
    """
    _incident_cache[cache_key] = (time.time(), data)


# ============================================================================
# COORDINATE PARSING
# ============================================================================

def parse_coordinates(location: str) -> Optional[Tuple[float, float]]:
    """Parse coordinates from location string.
    
    Args:
        location: Location string (e.g., "40.7128,-74.0060" or "New York, NY")
        
    Returns:
        Tuple of (latitude, longitude) or None
    """
    # Try parsing as coordinates
    parts = [p.strip() for p in location.split(",")]
    if len(parts) == 2:
        try:
            lat = float(parts[0])
            lon = float(parts[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)
        except (ValueError, TypeError):
            pass
    
    return None


def miles_to_meters(miles: float) -> float:
    """Convert miles to meters.
    
    Args:
        miles: Distance in miles
        
    Returns:
        Distance in meters
    """
    return miles * 1609.34


# ============================================================================
# HERE API INTEGRATION
# ============================================================================

async def fetch_traffic_incidents_area(
    lat: float,
    lon: float,
    radius_meters: float,
    api_key: str,
) -> Optional[Dict[str, Any]]:
    """Fetch traffic incidents in circular area from HERE API.
    
    Args:
        lat: Latitude
        lon: Longitude
        radius_meters: Search radius in meters
        api_key: HERE API key
        
    Returns:
        API response or None if error
    """
    if not HTTPX_AVAILABLE:
        return None
    
    params = {
        "in": f"circle:{lat},{lon};r={int(radius_meters)}",
        "apiKey": api_key,
    }
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(HERE_TRAFFIC_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            return data
    except httpx.TimeoutException:
        return {"error": "API request timeout"}
    except httpx.HTTPError as e:
        return {"error": f"HTTP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def fetch_traffic_incidents_route(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    width_meters: float,
    api_key: str,
) -> Optional[Dict[str, Any]]:
    """Fetch traffic incidents along route corridor from HERE API.
    
    Args:
        origin_lat: Origin latitude
        origin_lon: Origin longitude
        dest_lat: Destination latitude
        dest_lon: Destination longitude
        width_meters: Corridor width in meters
        api_key: HERE API key
        
    Returns:
        API response or None if error
    """
    if not HTTPX_AVAILABLE:
        return None
    
    params = {
        "in": f"corridor:{origin_lat},{origin_lon};{dest_lat},{dest_lon};w={int(width_meters)}",
        "apiKey": api_key,
    }
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(HERE_TRAFFIC_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            return data
    except httpx.TimeoutException:
        return {"error": "API request timeout"}
    except httpx.HTTPError as e:
        return {"error": f"HTTP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# ============================================================================
# INCIDENT PARSING
# ============================================================================

def parse_incident(incident_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse HERE API incident data into structured format.
    
    Args:
        incident_data: Raw incident data from HERE API
        
    Returns:
        Parsed incident dictionary
    """
    incident_type = incident_data.get("type", "UNKNOWN")
    severity_code = incident_data.get("severity", {}).get("value", 1)
    
    # Map severity code to level
    if severity_code >= 4:
        severity = "CRITICAL"
    elif severity_code == 3:
        severity = "MAJOR"
    elif severity_code == 2:
        severity = "MODERATE"
    else:
        severity = "MINOR"
    
    # Parse location
    location_info = incident_data.get("location", {})
    geometry = location_info.get("geometry", {})
    shapes = geometry.get("shapes", [])
    
    location_description = location_info.get("description", "Unknown location")
    
    # Parse times
    start_time = incident_data.get("startTime", "")
    end_time = incident_data.get("endTime", "")
    
    # Parse description
    description = incident_data.get("summary", {}).get("value", "No description available")
    
    # Parse road/lane information
    affected_roads = []
    if shapes:
        for shape in shapes:
            links = shape.get("links", [])
            for link in links:
                road_name = link.get("roadName", "")
                if road_name:
                    affected_roads.append(road_name)
    
    return {
        "type": incident_type,
        "type_display": INCIDENT_TYPE_MAP.get(incident_type, incident_type),
        "icon": INCIDENT_TYPE_ICONS.get(incident_type, "⚠️"),
        "severity": severity,
        "severity_code": severity_code,
        "location_description": location_description,
        "affected_roads": affected_roads,
        "start_time": start_time,
        "end_time": end_time,
        "description": description,
        "raw_data": incident_data,
    }


# ============================================================================
# MOCK DATA (for testing without API key)
# ============================================================================

def generate_mock_incidents(location: str) -> List[Dict[str, Any]]:
    """Generate mock incident data for testing.
    
    Args:
        location: Location string
        
    Returns:
        List of mock incident dictionaries
    """
    now = datetime.now()
    
    return [
        {
            "type": "ACCIDENT",
            "type_display": "Accident",
            "icon": "[!]",
            "severity": "CRITICAL",
            "severity_code": 4,
            "location_description": f"I-95 North near Exit 10 ({location})",
            "affected_roads": ["I-95"],
            "start_time": (now - timedelta(hours=1)).isoformat(),
            "end_time": None,
            "description": "Multi-vehicle collision, 3 lanes blocked",
            "raw_data": {},
        },
        {
            "type": "CONSTRUCTION",
            "type_display": "Construction",
            "icon": "[CONSTR]",
            "severity": "MODERATE",
            "severity_code": 2,
            "location_description": f"Main Street between 1st and 2nd Ave ({location})",
            "affected_roads": ["Main Street"],
            "start_time": (now - timedelta(hours=8)).isoformat(),
            "end_time": (now + timedelta(hours=6)).isoformat(),
            "description": "Road resurfacing work, left lane closed",
            "raw_data": {},
        },
        {
            "type": "WEATHER",
            "type_display": "Weather",
            "icon": "[WEATHER]",
            "severity": "MINOR",
            "severity_code": 1,
            "location_description": f"Hutchinson River Parkway ({location})",
            "affected_roads": ["Hutchinson River Parkway"],
            "start_time": (now - timedelta(minutes=30)).isoformat(),
            "end_time": None,
            "description": "Heavy rain, reduced visibility",
            "raw_data": {},
        },
    ]


# ============================================================================
# INPUT PARSING
# ============================================================================

def parse_natural_language(input_str: str) -> Dict[str, Any]:
    """Parse natural language input to extract traffic incident query parameters.
    
    Args:
        input_str: Natural language query
        
    Returns:
        Dictionary with parsed parameters
    """
    result = {
        "location": None,
        "radius": 25.0,  # Default 25 miles
        "route": None,
        "incident_types": None,
        "severity_filter": None,
    }
    
    input_lower = input_str.lower()
    
    # Extract route (origin to destination)
    route_patterns = [
        r"(?:from|route|to|between)\s+([A-Za-z0-9\s,]+)\s+(?:to|->|and)\s+([A-Za-z0-9\s,]+)",
        r"([A-Za-z0-9\s,]+)\s+(?:to|->)\s+([A-Za-z0-9\s,]+)",
    ]
    
    for pattern in route_patterns:
        match = re.search(pattern, input_str, re.IGNORECASE)
        if match:
            origin = match.group(1).strip()
            destination = match.group(2).strip()
            result["route"] = f"{origin} to {destination}"
            result["location"] = origin  # Use origin as location
            break
    
    # Extract location if not in route
    if not result["location"]:
        # Try to extract city/state or address
        location_patterns = [
            r"(?:near|in|at|around|on)\s+([A-Za-z\s,]+(?:,\s*[A-Z]{2})?)",
            r"([A-Za-z\s,]+(?:,\s*[A-Z]{2})?)(?:\s+traffic|\s+incidents)?",
        ]
        for pattern in location_patterns:
            match = re.search(pattern, input_lower)
            if match:
                result["location"] = match.group(1).strip()
                break
    
    # Extract radius
    radius_pattern = r"(\d+)\s*(?:mile|miles|mi|km|kilometer|kilometers)"
    match = re.search(radius_pattern, input_lower)
    if match:
        radius_value = float(match.group(1))
        if "km" in match.group(0).lower():
            result["radius"] = radius_value * 0.621371  # Convert km to miles
        else:
            result["radius"] = radius_value
    
    # Extract incident type filter
    if "accident" in input_lower or "crash" in input_lower or "collision" in input_lower:
        result["incident_types"] = ["ACCIDENT"]
    elif "construction" in input_lower or "roadwork" in input_lower:
        result["incident_types"] = ["CONSTRUCTION"]
    elif "closure" in input_lower or "closed" in input_lower:
        result["incident_types"] = ["ROAD_CLOSURE"]
    elif "weather" in input_lower:
        result["incident_types"] = ["WEATHER"]
    elif "hazard" in input_lower:
        result["incident_types"] = ["ROAD_HAZARD"]
    
    # Extract severity filter
    if "critical" in input_lower or "major" in input_lower:
        result["severity_filter"] = ["CRITICAL", "MAJOR"]
    elif "minor" in input_lower:
        result["severity_filter"] = ["MINOR"]
    
    return result


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_incident_report(
    incidents: List[Dict[str, Any]],
    location: str,
    radius: float,
    route: Optional[str] = None,
) -> str:
    """Format traffic incident report.
    
    Args:
        incidents: List of parsed incident dictionaries
        location: Location string
        radius: Search radius in miles
        route: Optional route string
        
    Returns:
        Formatted report string
    """
    if not incidents:
        location_str = f"for route {route}" if route else f"near {location} ({radius}-mile radius)"
        return (
            f"[OK] No traffic incidents reported {location_str}\n"
            "Roads appear clear at this time."
        )
    
    output_parts = []
    
    # Header
    if route:
        output_parts.append(f"TRAFFIC CHECK: {route.upper()}")
        output_parts.append("")
        if len(incidents) == 1:
            output_parts.append("Found 1 incident affecting your route:")
        else:
            output_parts.append(f"Found {len(incidents)} incidents affecting your route:")
    else:
        location_str = location.upper()
        output_parts.append(f"TRAFFIC INCIDENTS NEAR {location_str} ({radius}-mile radius)")
        output_parts.append("")
        if len(incidents) == 1:
            output_parts.append("Found 1 active incident:")
        else:
            output_parts.append(f"Found {len(incidents)} active incidents:")
    
    output_parts.append("")
    
    # Sort by severity (critical first)
    severity_order = {"CRITICAL": 4, "MAJOR": 3, "MODERATE": 2, "MINOR": 1}
    incidents_sorted = sorted(
        incidents,
        key=lambda x: severity_order.get(x["severity"], 0),
        reverse=True,
    )
    
    # Format each incident
    total_delay_minutes = 0
    
    for i, incident in enumerate(incidents_sorted, 1):
        icon = incident["icon"]
        severity = incident["severity"]
        severity_display = SEVERITY_MAP.get(severity, ("", severity))[1]
        severity_icon = SEVERITY_MAP.get(severity, ("", ""))[0]
        
        output_parts.append(f"{icon} INCIDENT #{i} - {severity_display}")
        output_parts.append(f"Type: {incident['type_display']}")
        output_parts.append(f"Location: {incident['location_description']}")
        output_parts.append(f"Severity: {severity_display} {severity_icon}")
        
        # Affected roads
        if incident["affected_roads"]:
            roads_str = ", ".join(incident["affected_roads"])
            output_parts.append(f"Affected Roads: {roads_str}")
        
        # Impact and delay
        delay_estimate = estimate_delay(incident)
        if delay_estimate:
            total_delay_minutes += delay_estimate
            if delay_estimate >= 30:
                delay_str = f"{delay_estimate}+ minutes"
            elif delay_estimate >= 15:
                delay_str = f"{delay_estimate}-{delay_estimate + 10} minutes"
            else:
                delay_str = f"{delay_estimate} minutes"
            output_parts.append(f"Delay: {delay_str}")
        
        # Time information
        if incident["start_time"]:
            start_dt = parse_time_string(incident["start_time"])
            if start_dt:
                time_str = format_incident_time(start_dt)
                output_parts.append(f"Started: {time_str}")
        
        if incident["end_time"]:
            end_dt = parse_time_string(incident["end_time"])
            if end_dt:
                time_str = format_incident_time(end_dt)
                output_parts.append(f"Duration: Until {time_str}")
        
        # Description
        if incident["description"]:
            output_parts.append(f"Description: {incident['description']}")
        
        output_parts.append("")
    
    # Summary for route-based search
    if route and total_delay_minutes > 0:
        output_parts.append(f"TOTAL ESTIMATED DELAY: +{total_delay_minutes} minutes")
        output_parts.append("")
        output_parts.append(
            "[TIP] Consider alternative route if delay is significant"
        )
    
    # Recommendations
    critical_incidents = [inc for inc in incidents_sorted if inc["severity"] == "CRITICAL"]
    if critical_incidents:
        output_parts.append("")
        output_parts.append("[RECOMMENDATION]:")
        for incident in critical_incidents[:3]:  # Show up to 3 recommendations
            location_desc = incident["location_description"]
            output_parts.append(f"- Avoid {location_desc} (major delays expected)")
    
    return "\n".join(output_parts)


def estimate_delay(incident: Dict[str, Any]) -> int:
    """Estimate delay in minutes based on incident type and severity.
    
    Args:
        incident: Incident dictionary
        
    Returns:
        Estimated delay in minutes
    """
    severity = incident["severity"]
    incident_type = incident["type"]
    
    # Base delay by severity
    base_delays = {
        "CRITICAL": 45,
        "MAJOR": 25,
        "MODERATE": 10,
        "MINOR": 5,
    }
    
    delay = base_delays.get(severity, 10)
    
    # Adjust by type
    if incident_type == "ROAD_CLOSURE":
        delay = int(delay * 1.5)  # Road closures cause longer delays
    elif incident_type == "CONSTRUCTION":
        delay = int(delay * 0.8)  # Construction is usually planned
    
    return delay


def parse_time_string(time_str: str) -> Optional[datetime]:
    """Parse time string from HERE API format.
    
    Args:
        time_str: Time string (ISO format)
        
    Returns:
        Datetime object or None
    """
    try:
        # Handle ISO format with timezone
        if "T" in time_str:
            if time_str.endswith("Z"):
                time_str = time_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(time_str, "%Y-%m-%d")
        return dt
    except Exception:
        return None


def format_incident_time(dt: datetime) -> str:
    """Format datetime for incident report.
    
    Args:
        dt: Datetime object
        
    Returns:
        Formatted time string
    """
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    dt_no_tz = dt.replace(tzinfo=None) if dt.tzinfo else dt
    now_no_tz = now.replace(tzinfo=None) if now.tzinfo else now
    diff = now_no_tz - dt_no_tz
    
    if diff.total_seconds() < 0:  # Future time (end time)
        return dt_no_tz.strftime("%B %d, %Y at %I:%M %p")
    elif diff.total_seconds() < 3600:  # Less than 1 hour
        minutes = int(diff.total_seconds() / 60)
        if minutes < 1:
            return "Just now"
        return f"{minutes} minutes ago"
    elif diff.days == 0:  # Today
        return f"Today at {dt_no_tz.strftime('%I:%M %p')}"
    elif diff.days == 1:
        return f"Yesterday at {dt_no_tz.strftime('%I:%M %p')}"
    else:
        return dt_no_tz.strftime("%B %d, %Y at %I:%M %p")


# ============================================================================
# TOOL IMPLEMENTATION
# ============================================================================

class TrafficIncidentInput(BaseModel):
    """Input schema for traffic incident tool."""
    input_str: str = Field(description="Traffic incident query: location, route, or highway. Example: 'Any incidents on I-95?' or 'Check traffic from NYC to Boston'")


class TrafficIncidentTool(BaseTool):
    """Tool for checking traffic incidents using HERE Traffic API.
    
    Provides real-time traffic incident information including accidents,
    construction, road closures, and weather-related incidents.
    Supports both area-based and route-based searches.
    """
    
    name: str = "traffic_incident"
    description: str = (
        "Checks for traffic incidents including accidents, construction, road closures, "
        "and weather-related incidents using HERE Traffic API. "
        "Input: location (city/address/coordinates) or route (origin to destination), "
        "optional radius in miles, optional incident type filter. "
        "Example: 'Any incidents on I-95?' or 'Check traffic from NYC to Boston'. "
        "Requires HERE_API_KEY (free tier: 250k requests/month)."
    )
    args_schema: Type[BaseModel] = TrafficIncidentInput
    
    def _run(self, input_str: str) -> str:
        """Synchronous execution (required by LangChain)."""
        import asyncio
        return asyncio.run(self._arun(input_str))
    
    async def _arun(self, input_str: str) -> str:
        """Asynchronous execution."""
        try:
            # Parse natural language input
            parsed = parse_natural_language(input_str)
            
            location = parsed.get("location")
            radius = parsed.get("radius", 25.0)
            route = parsed.get("route")
            incident_types = parsed.get("incident_types")
            
            # Check cache
            cache_key = get_cache_key(str(location) or "Unknown", radius or 25.0, route)
            cached_data = get_cached_incidents(cache_key)
            if cached_data:
                incidents = cached_data.get("incidents", [])
                return format_incident_report(incidents, location or "Unknown", radius or 25.0, route)
            
            # Get API key
            api_key = os.getenv("HERE_API_KEY", "")
            
            # If route provided, parse origin and destination
            if route:
                route_parts = route.split(" to ", 1)
                if len(route_parts) == 2:
                    origin = route_parts[0].strip()
                    destination = route_parts[1].strip()
                    
                    # Resolve locations to coordinates
                    if DRIVING_DISTANCE_AVAILABLE and resolve_location:
                        origin_coords = parse_coordinates(resolve_location(origin))
                        dest_coords = parse_coordinates(resolve_location(destination))
                        
                        if origin_coords and dest_coords:
                            # Fetch incidents along route
                            if api_key:
                                incidents_data = await fetch_traffic_incidents_route(
                                    origin_coords[0],
                                    origin_coords[1],
                                    dest_coords[0],
                                    dest_coords[1],
                                    5000.0,  # 5km corridor width
                                    api_key,
                                )
                            else:
                                incidents_data = None
                            
                            if incidents_data and "error" not in incidents_data:
                                incidents_list = incidents_data.get("items", [])
                                incidents = [parse_incident(inc) for inc in incidents_list]
                            else:
                                # Fallback to mock data
                                incidents = generate_mock_incidents(origin)
                            
                            # Filter by type if specified
                            if incident_types:
                                incidents = [inc for inc in incidents if inc["type"] in incident_types]
                            
                            cache_incidents(cache_key, {"incidents": incidents})
                            return format_incident_report(incidents, origin, radius or 25.0, route)
            
            # Area-based search
            if not location:
                return "[ERROR] Location required. Provide city/address or coordinates."
            
            # Resolve location to coordinates
            location_coords = parse_coordinates(location)
            
            if not location_coords:
                # Try to use location string directly (API might handle it)
                if DRIVING_DISTANCE_AVAILABLE and resolve_location:
                    resolved = resolve_location(location)
                    location_coords = parse_coordinates(resolved)
            
            if not location_coords:
                # Fallback: use mock data with warning
                incidents = generate_mock_incidents(location)
                return (
                    "[DEMO MODE] Using mock data (could not resolve location to coordinates)\n"
                    "Real-time traffic data requires valid location coordinates and HERE_API_KEY\n\n"
                    + format_incident_report(incidents, location, radius or 25.0, route)
                )
            
            lat, lon = location_coords
            radius_meters = miles_to_meters(radius or 25.0)
            
            # Fetch incidents
            if api_key:
                incidents_data = await fetch_traffic_incidents_area(
                    lat, lon, radius_meters, api_key
                )
            else:
                incidents_data = None
            
            if incidents_data and "error" not in incidents_data:
                incidents_list = incidents_data.get("items", [])
                incidents = [parse_incident(inc) for inc in incidents_list]
            else:
                # Fallback to mock data
                if incidents_data and "error" in incidents_data:
                    error_msg = incidents_data["error"]
                else:
                    error_msg = "No API key configured"
                
                incidents = generate_mock_incidents(location)
                warning = (
                    f"[DEMO MODE] Using mock data ({error_msg})\n"
                    "Real-time traffic data requires HERE_API_KEY\n"
                    "Sign up at: https://developer.here.com/\n\n"
                )
                result = format_incident_report(incidents, location, radius or 25.0, route)
                return warning + result
            
            # Filter by type if specified
            if incident_types:
                incidents = [inc for inc in incidents if inc["type"] in incident_types]
            
            # Cache results
            cache_incidents(cache_key, {"incidents": incidents})
            
            return format_incident_report(incidents, location or "Unknown", radius or 25.0, route)
        except Exception as e:
            return f"Error checking traffic incidents: {str(e)}"


# Create tool instance
traffic_incident_tool = TrafficIncidentTool()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_traffic_incident():
        """Test the traffic incident tool."""
        tool = TrafficIncidentTool()
        
        test_cases = [
            ("Any incidents near New York, NY?", "Area search"),
            ("Traffic from NYC to Boston", "Route search"),
            ("Accidents on I-95", "Type filter"),
        ]
        
        print("=" * 70)
        print("TRAFFIC INCIDENT TOOL TEST")
        print("=" * 70)
        
        for input_str, description in test_cases:
            print(f"\n{description}: {input_str}")
            print("-" * 70)
            result = await tool._arun(input_str)
            print(result)
            print()
    
    asyncio.run(test_traffic_incident())

