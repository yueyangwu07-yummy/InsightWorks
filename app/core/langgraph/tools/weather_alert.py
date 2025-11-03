"""Weather alert monitor tool for LangChain using Weather.gov API.

This module provides real-time weather alerts from the National Weather Service.
Completely free, no API key required. Covers entire United States.
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

WEATHER_GOV_ALERTS_URL = "https://api.weather.gov/alerts/active"
API_TIMEOUT = 10.0  # seconds
CACHE_TTL = 600  # 10 minutes in seconds

# Simple in-memory cache
_alert_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# US State abbreviations
STATE_ABBREVIATIONS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI", "GU", "AS", "MP",
}

# State name to abbreviation mapping
STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

# Alert severity order (higher number = more critical)
SEVERITY_ORDER = {
    "EXTREME": 4,
    "SEVERE": 3,
    "MODERATE": 2,
    "MINOR": 1,
    "UNKNOWN": 0,
}

# Alert type icons
ALERT_TYPE_ICONS = {
    "tornado": "[TORNADO]",
    "thunderstorm": "[STORM]",
    "winter": "[SNOW]",
    "flood": "[FLOOD]",
    "heat": "[HEAT]",
    "hurricane": "[HURRICANE]",
    "blizzard": "[BLIZZARD]",
    "ice": "[ICE]",
    "wind": "[WIND]",
    "fog": "[FOG]",
    "coastal": "[COASTAL]",
}


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def get_cache_key(location: str, severity_filter: Optional[List[str]] = None) -> str:
    """Generate cache key for alert request.
    
    Args:
        location: Location string
        severity_filter: Optional severity filter
        
    Returns:
        Cache key string
    """
    severity_str = ",".join(severity_filter or [])
    key_string = f"{location}|{severity_str}".lower().strip()
    return key_string


def get_cached_alerts(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached alerts if available and not expired.
    
    Args:
        cache_key: Cache key
        
    Returns:
        Cached alert data or None if not found/expired
    """
    if cache_key in _alert_cache:
        timestamp, data = _alert_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return data
        else:
            del _alert_cache[cache_key]
    
    return None


def cache_alerts(cache_key: str, data: Dict[str, Any]) -> None:
    """Cache alert data.
    
    Args:
        cache_key: Cache key
        data: Alert data to cache
    """
    _alert_cache[cache_key] = (time.time(), data)


# ============================================================================
# COORDINATE PARSING
# ============================================================================

def parse_coordinates(location: str) -> Optional[Tuple[float, float]]:
    """Parse coordinates from location string.
    
    Args:
        location: Location string
        
    Returns:
        Tuple of (latitude, longitude) or None
    """
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


def resolve_state_code(location: str) -> Optional[str]:
    """Resolve location to state code.
    
    Args:
        location: Location string (state name or abbreviation)
        
    Returns:
        State abbreviation or None
    """
    location_upper = location.upper().strip()
    location_lower = location.lower().strip()
    
    # Direct match with abbreviation
    if location_upper in STATE_ABBREVIATIONS:
        return location_upper
    
    # Match with state name
    if location_lower in STATE_NAMES:
        return STATE_NAMES[location_lower]
    
    # Try partial match
    for state_name, code in STATE_NAMES.items():
        if location_lower in state_name or state_name in location_lower:
            return code
    
    return None


# ============================================================================
# WEATHER.GOV API INTEGRATION
# ============================================================================

async def fetch_alerts_by_state(state_code: str) -> Optional[Dict[str, Any]]:
    """Fetch weather alerts for a state.
    
    Args:
        state_code: Two-letter state code
        
    Returns:
        API response or None if error
    """
    if not HTTPX_AVAILABLE:
        return None
    
    params = {"area": state_code}
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(WEATHER_GOV_ALERTS_URL, params=params)
            response.raise_for_status()
            data = response.json()
            return data
    except httpx.TimeoutException:
        return {"error": "API request timeout"}
    except httpx.HTTPError as e:
        return {"error": f"HTTP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def fetch_alerts_by_point(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Fetch weather alerts for a specific point.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        API response or None if error
    """
    if not HTTPX_AVAILABLE:
        return None
    
    params = {"point": f"{lat},{lon}"}
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(WEATHER_GOV_ALERTS_URL, params=params)
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
# ALERT PARSING
# ============================================================================

def parse_alert(alert_feature: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Weather.gov alert feature into structured format.
    
    Args:
        alert_feature: Alert feature from API
        
    Returns:
        Parsed alert dictionary
    """
    properties = alert_feature.get("properties", {})
    
    event_type = properties.get("event", "Unknown Alert")
    severity = properties.get("severity", "UNKNOWN").upper()
    urgency = properties.get("urgency", "Unknown").upper()
    certainty = properties.get("certainty", "Unknown").upper()
    
    headline = properties.get("headline", "")
    description = properties.get("description", "")
    instruction = properties.get("instruction", "")
    
    onset_str = properties.get("onset", "")
    expires_str = properties.get("expires", "")
    
    affected_areas = []
    geocode = properties.get("geocode", {})
    if "UGC" in geocode:
        affected_areas = geocode["UGC"]
    
    # Determine alert icon based on event type
    event_lower = event_type.lower()
    icon = "[ALERT]"
    for alert_key, alert_icon in ALERT_TYPE_ICONS.items():
        if alert_key in event_lower:
            icon = alert_icon
            break
    
    return {
        "event": event_type,
        "severity": severity,
        "urgency": urgency,
        "certainty": certainty,
        "headline": headline,
        "description": description,
        "instruction": instruction,
        "onset": onset_str,
        "expires": expires_str,
        "affected_areas": affected_areas,
        "icon": icon,
        "raw_data": properties,
    }


def prioritize_alerts(alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort alerts by priority (most critical first).
    
    Priority order:
    1. Extreme + Immediate
    2. Severe + Immediate
    3. Extreme + Expected
    4. Severe + Expected
    5. Moderate
    6. Minor
    
    Args:
        alerts: List of alert dictionaries
        
    Returns:
        Sorted list of alerts
    """
    def alert_priority(alert: Dict[str, Any]) -> Tuple[int, int]:
        severity_score = SEVERITY_ORDER.get(alert["severity"], 0)
        urgency_score = 2 if alert["urgency"] == "IMMEDIATE" else 1
        return (severity_score, urgency_score)
    
    return sorted(alerts, key=alert_priority, reverse=True)


# ============================================================================
# TRAVEL SAFETY ASSESSMENT
# ============================================================================

def assess_travel_safety(alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assess travel safety based on active alerts.
    
    Args:
        alerts: List of alert dictionaries
        
    Returns:
        Safety assessment dictionary
    """
    if not alerts:
        return {
            "status": "SAFE",
            "recommendation": "SAFE",
            "message": "No active weather alerts. Travel conditions are normal.",
            "risk_level": 1,
        }
    
    # Check for extreme/severe alerts
    extreme_alerts = [a for a in alerts if a["severity"] == "EXTREME"]
    severe_alerts = [a for a in alerts if a["severity"] == "SEVERE"]
    moderate_alerts = [a for a in alerts if a["severity"] == "MODERATE"]
    
    immediate_alerts = [a for a in alerts if a["urgency"] == "IMMEDIATE"]
    
    if extreme_alerts and immediate_alerts:
        return {
            "status": "DANGEROUS",
            "recommendation": "DO_NOT_TRAVEL",
            "message": "Extreme weather conditions with immediate impact. Travel is dangerous.",
            "risk_level": 5,
            "reasons": [
                f"{len(extreme_alerts)} extreme alert(s)",
                "Immediate impact expected",
                "High risk of accidents and getting stranded",
            ],
        }
    elif severe_alerts and immediate_alerts:
        return {
            "status": "DANGEROUS",
            "recommendation": "NOT_RECOMMENDED",
            "message": "Severe weather conditions make travel dangerous.",
            "risk_level": 4,
            "reasons": [
                f"{len(severe_alerts)} severe alert(s)",
                "Immediate impact expected",
                "Significant delays and road closures likely",
            ],
        }
    elif severe_alerts or (extreme_alerts and not immediate_alerts):
        return {
            "status": "CAUTION",
            "recommendation": "USE_CAUTION",
            "message": "Severe weather alerts in effect. Travel with caution.",
            "risk_level": 3,
            "reasons": [
                f"{len(severe_alerts + extreme_alerts)} severe/extreme alert(s)",
                "Possible delays and hazardous conditions",
            ],
        }
    elif moderate_alerts:
        return {
            "status": "CAUTION",
            "recommendation": "USE_CAUTION",
            "message": "Moderate weather alerts in effect. Exercise caution while traveling.",
            "risk_level": 2,
            "reasons": [
                f"{len(moderate_alerts)} moderate alert(s)",
                "Some weather-related delays possible",
            ],
        }
    else:
        return {
            "status": "SAFE",
            "recommendation": "CAUTION",
            "message": "Minor weather alerts only. Travel is generally safe but stay alert.",
            "risk_level": 1,
        }


# ============================================================================
# INPUT PARSING
# ============================================================================

def parse_natural_language(input_str: str) -> Dict[str, Any]:
    """Parse natural language input to extract weather alert query parameters.
    
    Args:
        input_str: Natural language query
        
    Returns:
        Dictionary with parsed parameters
    """
    result = {
        "location": None,
        "route": None,
        "severity_filter": None,
        "alert_types": None,
    }
    
    input_lower = input_str.lower()
    
    # Extract route (origin to destination)
    route_patterns = [
        r"(?:from|route|to|between|check\s+weather)\s+([A-Za-z0-9\s,]+)\s+(?:to|->|and)\s+([A-Za-z0-9\s,]+)",
        r"([A-Za-z0-9\s,]+)\s+(?:to|->)\s+([A-Za-z0-9\s,]+)",
    ]
    
    for pattern in route_patterns:
        match = re.search(pattern, input_str, re.IGNORECASE)
        if match:
            origin = match.group(1).strip()
            destination = match.group(2).strip()
            result["route"] = f"{origin} to {destination}"
            result["location"] = origin
            break
    
    # Extract location if not in route
    if not result["location"]:
        # Try to find state name or abbreviation
        # First try multi-word states (e.g., "New York", "North Carolina")
        for state_name, state_code in STATE_NAMES.items():
            if state_name in input_lower:
                result["location"] = state_code
                break
        
        # If not found, try single word states and abbreviations
        if not result["location"]:
            words = input_str.split()
            for i, word in enumerate(words):
                # Try single word state
                word_clean = re.sub(r'[^A-Za-z]', '', word)
                if word_clean and len(word_clean) >= 2:
                    state_code = resolve_state_code(word_clean)
                    if state_code:
                        result["location"] = state_code
                        break
    
    # Extract severity filter
    if "extreme" in input_lower:
        result["severity_filter"] = ["EXTREME"]
    elif "severe" in input_lower:
        result["severity_filter"] = ["SEVERE", "EXTREME"]
    elif "moderate" in input_lower:
        result["severity_filter"] = ["MODERATE"]
    elif "minor" in input_lower:
        result["severity_filter"] = ["MINOR"]
    
    # Extract alert type filter
    alert_types = []
    if "tornado" in input_lower:
        alert_types.append("tornado")
    if "winter" in input_lower or "snow" in input_lower:
        alert_types.append("winter")
    if "flood" in input_lower:
        alert_types.append("flood")
    if "heat" in input_lower:
        alert_types.append("heat")
    if "hurricane" in input_lower:
        alert_types.append("hurricane")
    
    if alert_types:
        result["alert_types"] = alert_types
    
    return result


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_alert_report(
    alerts: List[Dict[str, Any]],
    location: str,
    route: Optional[str] = None,
) -> str:
    """Format weather alert report.
    
    Args:
        alerts: List of parsed alert dictionaries
        location: Location string
        route: Optional route string
        
    Returns:
        Formatted report string
    """
    output_parts = []
    
    # Header
    if route:
        output_parts.append(f"WEATHER ALERTS: {route.upper()}")
        output_parts.append("")
    else:
        location_display = location.upper()
        output_parts.append(f"WEATHER ALERTS FOR {location_display}")
        output_parts.append("=" * 40)
        output_parts.append("")
    
    if not alerts:
        output_parts.append("[OK] No active weather alerts")
        output_parts.append("Travel conditions are normal.")
        return "\n".join(output_parts)
    
    # Alert count
    output_parts.append(f"[WARNING] {len(alerts)} ACTIVE ALERT(S)")
    output_parts.append("")
    
    # Sort by priority
    alerts_sorted = prioritize_alerts(alerts)
    
    # Format each alert
    for i, alert in enumerate(alerts_sorted, 1):
        icon = alert["icon"]
        severity = alert["severity"]
        severity_display = severity
        
        if severity == "EXTREME":
            severity_icon = "[CRITICAL]"
        elif severity == "SEVERE":
            severity_icon = "[WARNING]"
        elif severity == "MODERATE":
            severity_icon = "[CAUTION]"
        else:
            severity_icon = "[INFO]"
        
        output_parts.append(f"{icon} ALERT #{i} - {severity}")
        output_parts.append(f"Type: {alert['event']}")
        output_parts.append(f"Severity: {severity_display} {severity_icon}")
        if alert["urgency"] != "Unknown":
            output_parts.append(f"Urgency: {alert['urgency']}")
        output_parts.append("")
        
        # Time information
        if alert["onset"]:
            onset_dt = parse_time_string(alert["onset"])
            expires_dt = parse_time_string(alert["expires"]) if alert["expires"] else None
            
            if onset_dt and expires_dt:
                time_str = format_alert_time_range(onset_dt, expires_dt)
                output_parts.append(f"Effective: {time_str}")
            elif onset_dt:
                time_str = format_alert_time(onset_dt)
                output_parts.append(f"Effective: {time_str}")
        
        # Affected areas
        if alert["affected_areas"]:
            areas_str = ", ".join(alert["affected_areas"][:5])  # Limit to 5
            if len(alert["affected_areas"]) > 5:
                areas_str += f" and {len(alert["affected_areas"]) - 5} more"
            output_parts.append(f"Affected Areas: {areas_str}")
            output_parts.append("")
        
        # Description
        if alert["description"]:
            output_parts.append("DESCRIPTION:")
            desc_lines = alert["description"].split("\n")[:5]  # Limit lines
            for line in desc_lines:
                if line.strip():
                    output_parts.append(f"  {line.strip()}")
            output_parts.append("")
        
        # Instructions
        if alert["instruction"]:
            output_parts.append("INSTRUCTIONS:")
            inst_lines = alert["instruction"].split("\n")[:3]  # Limit lines
            for line in inst_lines:
                if line.strip():
                    output_parts.append(f"  {line.strip()}")
            output_parts.append("")
    
    # Travel safety assessment
    safety = assess_travel_safety(alerts)
    
    output_parts.append("")
    output_parts.append(f"TRAVEL RECOMMENDATION: {safety['recommendation']}")
    output_parts.append(safety["message"])
    
    if safety.get("reasons"):
        output_parts.append("")
        output_parts.append("Reasons:")
        for reason in safety["reasons"]:
            output_parts.append(f"  - {reason}")
    
    if safety["recommendation"] in ["DO_NOT_TRAVEL", "NOT_RECOMMENDED"]:
        output_parts.append("")
        output_parts.append("Alternatives:")
        output_parts.append("  - Postpone trip until weather improves")
        output_parts.append("  - Monitor conditions: https://weather.gov")
        output_parts.append("  - Check road conditions: 511")
    
    return "\n".join(output_parts)


def parse_time_string(time_str: str) -> Optional[datetime]:
    """Parse time string from Weather.gov API format.
    
    Args:
        time_str: Time string (ISO format)
        
    Returns:
        Datetime object or None
    """
    try:
        if "T" in time_str:
            if time_str.endswith("Z"):
                time_str = time_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(time_str, "%Y-%m-%d")
        return dt
    except Exception:
        return None


def format_alert_time(dt: datetime) -> str:
    """Format datetime for alert report.
    
    Args:
        dt: Datetime object
        
    Returns:
        Formatted time string
    """
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    dt_no_tz = dt.replace(tzinfo=None) if dt.tzinfo else dt
    now_no_tz = now.replace(tzinfo=None) if now.tzinfo else now
    diff = now_no_tz - dt_no_tz
    
    if diff.total_seconds() < 3600:  # Less than 1 hour
        minutes = int(diff.total_seconds() / 60)
        if minutes < 0:
            return f"In {abs(minutes)} minutes"
        return f"{minutes} minutes ago"
    elif diff.days == 0:  # Today
        return f"Today at {dt_no_tz.strftime('%I:%M %p')}"
    elif diff.days == 1:
        return f"Tomorrow at {dt_no_tz.strftime('%I:%M %p')}"
    else:
        return dt_no_tz.strftime("%B %d, %Y at %I:%M %p")


def format_alert_time_range(start_dt: datetime, end_dt: datetime) -> str:
    """Format time range for alert report.
    
    Args:
        start_dt: Start datetime
        end_dt: End datetime
        
    Returns:
        Formatted time range string
    """
    start_no_tz = start_dt.replace(tzinfo=None) if start_dt.tzinfo else start_dt
    end_no_tz = end_dt.replace(tzinfo=None) if end_dt.tzinfo else end_dt
    
    now = datetime.now()
    
    # Check if same day
    if start_no_tz.date() == end_no_tz.date():
        if start_no_tz.date() == now.date():
            return f"Today {start_no_tz.strftime('%I:%M %p')} - {end_no_tz.strftime('%I:%M %p')}"
        elif start_no_tz.date() == (now + timedelta(days=1)).date():
            return f"Tomorrow {start_no_tz.strftime('%I:%M %p')} - {end_no_tz.strftime('%I:%M %p')}"
        else:
            return f"{start_no_tz.strftime('%B %d')} {start_no_tz.strftime('%I:%M %p')} - {end_no_tz.strftime('%I:%M %p')}"
    else:
        return f"{format_alert_time(start_dt)} - {format_alert_time(end_dt)}"


# ============================================================================
# TOOL IMPLEMENTATION
# ============================================================================

class WeatherAlertInput(BaseModel):
    """Input schema for weather alert tool."""
    input_str: str = Field(description="Weather alert query: state, location, or route. Example: 'Any weather alerts in Massachusetts?' or 'Check weather from NYC to Boston'")


class WeatherAlertTool(BaseTool):
    """Tool for checking weather alerts using Weather.gov API.
    
    Provides real-time weather alerts from the National Weather Service.
    Completely free, no API key required. Covers entire United States.
    """
    
    name: str = "weather_alert"
    description: str = (
        "Checks for active weather alerts from the National Weather Service. "
        "Input: state abbreviation (e.g., 'MA', 'NY'), location, or route (origin to destination). "
        "Optional severity filter (extreme, severe, moderate, minor). "
        "Example: 'Any weather alerts in Massachusetts?' or 'Check weather from NYC to Boston'. "
        "Completely free, no API key required."
    )
    args_schema: Type[BaseModel] = WeatherAlertInput
    
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
            route = parsed.get("route")
            severity_filter = parsed.get("severity_filter")
            alert_types = parsed.get("alert_types")
            
            severity = severity_filter[0] if severity_filter else None
            
            # Check cache
            severity_filter_list = [severity] if severity else None
            cache_key = get_cache_key(str(location) or "Unknown", severity_filter_list)
            cached_data = get_cached_alerts(cache_key)
            if cached_data:
                alerts = cached_data.get("alerts", [])
                return format_alert_report(alerts, location or "Unknown", route)
            
            # Handle route-based search
            if route:
                route_parts = route.split(" to ", 1)
                if len(route_parts) == 2:
                    origin = route_parts[0].strip()
                    destination = route_parts[1].strip()
                    
                    # Get alerts for origin location
                    origin_state = resolve_state_code(origin) or origin
                    dest_state = resolve_state_code(destination) or destination
                    
                    # Fetch alerts for both locations
                    alerts = []
                    
                    if origin_state and origin_state in STATE_ABBREVIATIONS:
                        alerts_data = await fetch_alerts_by_state(origin_state)
                        if alerts_data and "error" not in alerts_data:
                            features = alerts_data.get("features", [])
                            alerts.extend([parse_alert(f) for f in features])
                    
                    if dest_state and dest_state in STATE_ABBREVIATIONS and dest_state != origin_state:
                        alerts_data = await fetch_alerts_by_state(dest_state)
                        if alerts_data and "error" not in alerts_data:
                            features = alerts_data.get("features", [])
                            alerts.extend([parse_alert(f) for f in features])
                    
                    # Remove duplicates
                    seen = set()
                    unique_alerts = []
                    for alert in alerts:
                        alert_key = alert["event"] + alert.get("headline", "")
                        if alert_key not in seen:
                            seen.add(alert_key)
                            unique_alerts.append(alert)
                    
                    alerts = unique_alerts
            else:
                # Single location search
                if not location:
                    return "[ERROR] Location required. Provide state abbreviation (e.g., 'MA', 'NY') or state name."
                
                # Resolve state code
                state_code = resolve_state_code(location)
                
                if state_code and state_code in STATE_ABBREVIATIONS:
                    alerts_data = await fetch_alerts_by_state(state_code)
                else:
                    # Try coordinates
                    coords = parse_coordinates(location)
                    if coords:
                        alerts_data = await fetch_alerts_by_point(coords[0], coords[1])
                    else:
                        return (
                            f"[ERROR] Invalid location: '{location}'. "
                            f"Provide state abbreviation (e.g., 'MA', 'NY') or state name."
                        )
                
                if alerts_data and "error" in alerts_data:
                    return f"[ERROR] {alerts_data['error']}"
                
                if not alerts_data:
                    return "[ERROR] Unable to fetch weather alerts. Please try again."
                
                features = alerts_data.get("features", [])
                alerts = [parse_alert(f) for f in features]
            
            # Filter by severity if specified
            if severity:
                severity_upper = severity.upper()
                alerts = [a for a in alerts if a["severity"] == severity_upper]
            
            # Filter by alert type if specified
            if alert_types:
                alerts = [
                    a for a in alerts
                    if any(alert_type in a["event"].lower() for alert_type in alert_types)
                ]
            
            # Cache results
            cache_alerts(cache_key, {"alerts": alerts})
            
            return format_alert_report(alerts, location or "Route", route)
        except Exception as e:
            return f"Error checking weather alerts: {str(e)}"


# Create tool instance
weather_alert_tool = WeatherAlertTool()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_weather_alert():
        """Test the weather alert tool."""
        tool = WeatherAlertTool()
        
        test_cases = [
            ("Any weather alerts in Massachusetts?", "State search"),
            ("Check weather from NYC to Boston", "Route search"),
            ("Severe weather alerts in NY", "Severity filter"),
        ]
        
        print("=" * 70)
        print("WEATHER ALERT TOOL TEST")
        print("=" * 70)
        
        for input_str, description in test_cases:
            print(f"\n{description}: {input_str}")
            print("-" * 70)
            result = await tool._arun(input_str)
            print(result)
            print()
    
    asyncio.run(test_weather_alert())

