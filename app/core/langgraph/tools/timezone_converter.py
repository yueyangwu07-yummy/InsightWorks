"""Timezone converter tool for LangChain.

This module provides timezone conversion capabilities for fleet management
and business operations. Works completely offline using Python's zoneinfo module.
Supports natural language input, business hours checking, and delivery time calculations.
"""

import re
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type


# ============================================================================
# TIMEZONE MAPPINGS
# ============================================================================

# US Timezone abbreviations to IANA timezone names
TIMEZONE_MAP: Dict[str, str] = {
    # Eastern Time
    "est": "America/New_York",
    "edt": "America/New_York",
    "et": "America/New_York",
    "eastern": "America/New_York",
    "new york": "America/New_York",
    "ny": "America/New_York",
    "nyc": "America/New_York",
    
    # Central Time
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
    "ct": "America/Chicago",
    "central": "America/Chicago",
    "chicago": "America/Chicago",
    
    # Mountain Time
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "mt": "America/Denver",
    "mountain": "America/Denver",
    "denver": "America/Denver",
    
    # Pacific Time
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "pt": "America/Los_Angeles",
    "pacific": "America/Los_Angeles",
    "los angeles": "America/Los_Angeles",
    "la": "America/Los_Angeles",
    
    # Alaska Time
    "akst": "America/Anchorage",
    "akdt": "America/Anchorage",
    "ak": "America/Anchorage",
    "alaska": "America/Anchorage",
    "anchorage": "America/Anchorage",
    
    # Hawaii Time
    "hst": "Pacific/Honolulu",
    "hawaii": "Pacific/Honolulu",
    "honolulu": "Pacific/Honolulu",
    "hi": "Pacific/Honolulu",
    
    # Arizona Time (no DST)
    "az": "America/Phoenix",
    "arizona": "America/Phoenix",
    "phoenix": "America/Phoenix",
}

# Major US cities to timezones
CITY_TIMEZONES: Dict[str, str] = {
    "new york": "America/New_York",
    "nyc": "America/New_York",
    "boston": "America/New_York",
    "washington": "America/New_York",
    "dc": "America/New_York",
    "miami": "America/New_York",
    "atlanta": "America/New_York",
    
    "chicago": "America/Chicago",
    "dallas": "America/Chicago",
    "houston": "America/Chicago",
    "minneapolis": "America/Chicago",
    
    "denver": "America/Denver",
    "phoenix": "America/Phoenix",  # No DST
    "salt lake city": "America/Denver",
    
    "los angeles": "America/Los_Angeles",
    "la": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "sf": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "portland": "America/Los_Angeles",
    
    "anchorage": "America/Anchorage",
    "honolulu": "Pacific/Honolulu",
}

# Standard business hours (9 AM - 5 PM local time)
BUSINESS_HOURS_START = time(9, 0)  # 9:00 AM
BUSINESS_HOURS_END = time(17, 0)   # 5:00 PM


# ============================================================================
# TIMEZONE UTILITIES
# ============================================================================

def resolve_timezone(tz_input: str) -> Optional[str]:
    """Resolve timezone input to IANA timezone name.
    
    Args:
        tz_input: Timezone abbreviation or city name
        
    Returns:
        IANA timezone name or None if not found
    """
    tz_lower = tz_input.lower().strip()
    
    # Direct lookup in timezone map
    if tz_lower in TIMEZONE_MAP:
        return TIMEZONE_MAP[tz_lower]
    
    # Lookup in city timezones
    if tz_lower in CITY_TIMEZONES:
        return CITY_TIMEZONES[tz_lower]
    
    # Try IANA timezone name directly (case-insensitive check)
    try:
        # Validate by trying to create ZoneInfo
        test_tz = ZoneInfo(tz_input)
        return tz_input
    except Exception:
        pass
    
    return None


def get_timezone_abbreviation(dt: datetime, tz: ZoneInfo) -> str:
    """Get timezone abbreviation for a datetime.
    
    Args:
        dt: Datetime object
        tz: Timezone
        
    Returns:
        Timezone abbreviation (EST, EDT, PST, etc.)
    """
    # zoneinfo doesn't provide abbreviation directly
    # Use strftime to get abbreviation
    try:
        return dt.strftime("%Z")
    except Exception:
        # Fallback: determine based on offset
        offset = dt.utcoffset()
        if offset:
            hours = int(offset.total_seconds() / 3600)
            # This is approximate, doesn't account for DST perfectly
            if hours == -5:
                return "EST"
            elif hours == -4:
                return "EDT"
            elif hours == -6:
                return "CST"
            elif hours == -5:
                return "CDT"
            elif hours == -7:
                return "MST"
            elif hours == -6:
                return "MDT"
            elif hours == -8:
                return "PST"
            elif hours == -7:
                return "PDT"
        return tz.key.split("/")[-1].upper()


def is_dst_active(dt: datetime, tz: ZoneInfo) -> bool:
    """Check if daylight saving time is active.
    
    Args:
        dt: Datetime object
        tz: Timezone
        
    Returns:
        True if DST is active
    """
    # Get offset for this datetime
    dt_tz = dt.astimezone(tz)
    offset = dt_tz.utcoffset()
    
    # Get offset for January (typically standard time)
    jan_dt = datetime(dt.year, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC")).astimezone(tz)
    jan_offset = jan_dt.utcoffset()
    
    # If current offset differs from January offset, DST is active
    return offset != jan_offset


def parse_time(time_str: str) -> Optional[time]:
    """Parse time string to time object.
    
    Supports formats:
    - "3:30 PM" or "3:30PM"
    - "15:30" (24-hour)
    - "3 PM" or "3PM"
    - "now" (returns None to indicate current time)
    
    Args:
        time_str: Time string
        
    Returns:
        time object or None for "now"
    """
    time_str = time_str.strip().lower()
    
    # Handle "now"
    if time_str == "now":
        return None
    
    # Pattern 1: 12-hour format with AM/PM
    pattern_12h = r"(\d{1,2}):?(\d{2})?\s*(am|pm)"
    match = re.search(pattern_12h, time_str, re.IGNORECASE)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        am_pm = match.group(3).lower()
        
        if am_pm == "pm" and hour != 12:
            hour += 12
        elif am_pm == "am" and hour == 12:
            hour = 0
        
        try:
            return time(hour, minute)
        except ValueError:
            return None
    
    # Pattern 2: 24-hour format
    pattern_24h = r"(\d{1,2}):(\d{2})"
    match = re.search(pattern_24h, time_str)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        
        try:
            return time(hour, minute)
        except ValueError:
            return None
    
    # Pattern 3: Just hour with AM/PM
    pattern_hour = r"(\d{1,2})\s*(am|pm)"
    match = re.search(pattern_hour, time_str, re.IGNORECASE)
    if match:
        hour = int(match.group(1))
        am_pm = match.group(2).lower()
        
        if am_pm == "pm" and hour != 12:
            hour += 12
        elif am_pm == "am" and hour == 12:
            hour = 0
        
        try:
            return time(hour, 0)
        except ValueError:
            return None
    
    return None


def is_business_hours(dt: datetime) -> bool:
    """Check if datetime is within business hours (9 AM - 5 PM).
    
    Args:
        dt: Datetime to check
        
    Returns:
        True if within business hours
    """
    current_time = dt.time()
    return BUSINESS_HOURS_START <= current_time <= BUSINESS_HOURS_END


def format_time_12h(dt: datetime, include_date: bool = False) -> str:
    """Format datetime in 12-hour format.
    
    Args:
        dt: Datetime to format
        include_date: Whether to include date
        
    Returns:
        Formatted time string
    """
    if include_date:
        return dt.strftime("%I:%M %p on %B %d, %Y")
    return dt.strftime("%I:%M %p")


def format_time_24h(dt: datetime, include_date: bool = False) -> str:
    """Format datetime in 24-hour format.
    
    Args:
        dt: Datetime to format
        include_date: Whether to include date
        
    Returns:
        Formatted time string
    """
    if include_date:
        return dt.strftime("%H:%M on %Y-%m-%d")
    return dt.strftime("%H:%M")


# ============================================================================
# NATURAL LANGUAGE PARSING
# ============================================================================

def parse_timezone_query(input_str: str) -> Optional[Dict[str, str]]:
    """Parse natural language timezone conversion query.
    
    Supports formats:
    - "What time is 3 PM EST in PST?"
    - "Convert 15:30 CST to EST"
    - "If it's 9 AM in New York, what time is it in LA?"
    - "Current time in Pacific timezone"
    
    Args:
        input_str: Natural language query
        
    Returns:
        Dictionary with parsed fields or None
    """
    input_lower = input_str.lower().strip()
    
    # Pattern 1: "What time is TIME TIMEZONE in TIMEZONE?"
    pattern1 = r"what\s+time\s+is\s+(.+?)\s+(?:in|at)\s+(.+?)(?:\?|$)"
    match = re.search(pattern1, input_lower)
    if match:
        time_part = match.group(1).strip()
        tz_part = match.group(2).strip()
        
        # Extract time and source timezone
        time_match = re.search(r"(\d{1,2}:?\d{0,2}\s*(?:am|pm)?|now)", time_part, re.IGNORECASE)
        if time_match:
            time_str = time_match.group(1)
            # Try to find source timezone in remaining text
            remaining = time_part.replace(time_str, "").strip()
            if remaining:
                source_tz = remaining
            else:
                # Assume EST/PST pattern
                tz_match = re.search(r"(est|pst|cst|mst|edt|pdt|cdt|mdt)", time_part, re.IGNORECASE)
                source_tz = tz_match.group(1) if tz_match else None
            target_tz = tz_part
            
            if source_tz and target_tz:
                return {
                    "time": time_str,
                    "source_timezone": source_tz,
                    "target_timezone": target_tz,
                }
    
    # Pattern 2: "Convert TIME TIMEZONE to TIMEZONE"
    pattern2 = r"convert\s+(.+?)\s+(?:to|in)\s+(.+?)(?:\?|$)"
    match = re.search(pattern2, input_lower)
    if match:
        source_part = match.group(1).strip()
        target_tz = match.group(2).strip()
        
        # Extract time
        time_match = re.search(r"(\d{1,2}:?\d{0,2}\s*(?:am|pm)?|now)", source_part, re.IGNORECASE)
        if time_match:
            time_str = time_match.group(1)
            source_tz = source_part.replace(time_str, "").strip()
            
            if source_tz and target_tz:
                return {
                    "time": time_str,
                    "source_timezone": source_tz,
                    "target_timezone": target_tz,
                }
    
    # Pattern 3: "If it's TIME in CITY, what time is it in CITY?"
    pattern3 = r"if\s+it'?s\s+(.+?)\s+in\s+(.+?),\s+what\s+time\s+is\s+it\s+in\s+(.+?)(?:\?|$)"
    match = re.search(pattern3, input_lower)
    if match:
        time_str = match.group(1).strip()
        source_location = match.group(2).strip()
        target_location = match.group(3).strip()
        
        return {
            "time": time_str,
            "source_timezone": source_location,
            "target_timezone": target_location,
        }
    
    # Pattern 3b: "What time is TIME in CITY if it is in CITY?"
    pattern3b = r"what\s+time\s+is\s+(.+?)\s+in\s+(.+?)\s+if\s+it\s+is\s+in\s+(.+?)(?:\?|$)"
    match = re.search(pattern3b, input_lower)
    if match:
        time_str = match.group(1).strip()
        source_location = match.group(2).strip()
        target_location = match.group(3).strip()
        
        return {
            "time": time_str,
            "source_timezone": source_location,
            "target_timezone": target_location,
        }
    
    # Pattern 4: "Current time in TIMEZONE"
    pattern4 = r"current\s+time\s+(?:in|at)\s+(.+?)(?:\?|$)"
    match = re.search(pattern4, input_lower)
    if match:
        tz = match.group(1).strip()
        # Remove "timezone" suffix if present
        tz = re.sub(r"\s+timezone$", "", tz, flags=re.IGNORECASE)
        return {
            "time": "now",
            "source_timezone": "UTC",  # Use UTC as source for "current time"
            "target_timezone": tz,
        }
    
    # Pattern 5: "TIME TIMEZONE to TIMEZONE" (simple format)
    pattern5 = r"(\d{1,2}:?\d{0,2}\s*(?:am|pm)?|now)\s+([a-z]+(?:\s+[a-z]+)?)\s+(?:to|in)\s+([a-z]+(?:\s+[a-z]+)?)"
    match = re.search(pattern5, input_lower)
    if match:
        return {
            "time": match.group(1).strip(),
            "source_timezone": match.group(2).strip(),
            "target_timezone": match.group(3).strip(),
        }
    
    return None


# ============================================================================
# DELIVERY TIME CALCULATION
# ============================================================================

def calculate_delivery_time(
    departure_time: datetime,
    departure_tz: ZoneInfo,
    arrival_tz: ZoneInfo,
    transit_hours: float,
) -> Dict[str, any]:
    """Calculate arrival time for shipment/delivery.
    
    Args:
        departure_time: Departure datetime
        departure_tz: Departure timezone
        arrival_tz: Arrival timezone
        transit_hours: Transit time in hours
        
    Returns:
        Dictionary with delivery timeline information
    """
    # Convert departure to timezone-aware
    if departure_time.tzinfo is None:
        departure_dt = departure_time.replace(tzinfo=departure_tz)
    else:
        departure_dt = departure_time.astimezone(departure_tz)
    
    # Calculate arrival time
    arrival_dt = departure_dt + timedelta(hours=transit_hours)
    arrival_dt_tz = arrival_dt.astimezone(arrival_tz)
    
    # Calculate time zones crossed
    departure_offset = departure_dt.utcoffset()
    arrival_offset = arrival_dt_tz.utcoffset()
    
    if departure_offset and arrival_offset:
        hours_diff = (arrival_offset.total_seconds() - departure_offset.total_seconds()) / 3600
        timezones_crossed = int(abs(hours_diff))
    else:
        timezones_crossed = 0
    
    # Check if same day
    same_day = departure_dt.date() == arrival_dt_tz.date()
    
    return {
        "departure": departure_dt,
        "arrival": arrival_dt_tz,
        "transit_hours": transit_hours,
        "timezones_crossed": timezones_crossed,
        "same_day": same_day,
    }


# ============================================================================
# TOOL IMPLEMENTATION
# ============================================================================

class TimezoneConverterInput(BaseModel):
    """Input schema for timezone converter."""
    input_str: str = Field(description="Timezone conversion query: '3 PM EST to PST' or 'What time is 3 PM EST in PST?'. Example: 'What time is 3 PM EST in PST?'")


class TimezoneConverterTool(BaseTool):
    """Tool for converting time between timezones.
    
    Supports US timezones with automatic DST handling, business hours checking,
    and delivery time calculations for fleet management.
    Works completely offline using Python's zoneinfo module.
    """
    
    name: str = "timezone_converter"
    description: str = (
        "Converts time between different timezones. "
        "Input: '3 PM EST to PST' or 'What time is 3 PM EST in PST?'. "
        "Supports US timezones (EST, PST, CST, MST) and major cities (NYC, LA, Chicago). "
        "Handles daylight saving time automatically. "
        "Can check business hours (9 AM - 5 PM) and calculate delivery times. "
        "Example: 'What time is 3 PM EST in PST?' or 'Current time in Pacific timezone'. "
        "Completely offline, no API required."
    )
    args_schema: Type[BaseModel] = TimezoneConverterInput
    
    def _run(self, input_str: str) -> str:
        """Synchronous execution (required by LangChain)."""
        import asyncio
        return asyncio.run(self._arun(input_str))
    
    async def _arun(self, input_str: str) -> str:
        """Asynchronous execution."""
        try:
            # Parse natural language input
            parsed = parse_timezone_query(input_str)
            
            if parsed:
                return self._convert(
                    parsed.get("time", "now"),
                    parsed.get("source_timezone", ""),
                    parsed.get("target_timezone", ""),
                    None,  # date not supported in single input
                )
            
            return (
                "[ERROR] Could not parse input. "
                "Expected format: '3 PM EST to PST' or 'What time is 3 PM EST in PST?'. "
                f"Got: '{input_str}'"
            )
        except Exception as e:
            return f"Error converting timezone: {str(e)}"
    
    def _convert(
        self,
        source_time: str,
        source_timezone: str,
        target_timezone: str,
        date: Optional[str] = None,
    ) -> str:
        """Perform timezone conversion.
        
        Args:
            source_time: Time string (e.g., "3:30 PM" or "now")
            source_timezone: Source timezone name or abbreviation
            target_timezone: Target timezone name or abbreviation
            date: Optional date string (default: today)
            
        Returns:
            Formatted conversion result
        """
        # Resolve timezones
        source_tz_name = resolve_timezone(source_timezone)
        target_tz_name = resolve_timezone(target_timezone)
        
        if not source_tz_name:
            suggestions = self._suggest_timezones(source_timezone)
            return (
                f"[ERROR] Invalid source timezone: '{source_timezone}'. "
                f"Valid timezones: {', '.join(suggestions[:5])}"
            )
        
        if not target_tz_name:
            suggestions = self._suggest_timezones(target_timezone)
            return (
                f"[ERROR] Invalid target timezone: '{target_timezone}'. "
                f"Valid timezones: {', '.join(suggestions[:5])}"
            )
        
        try:
            source_tz = ZoneInfo(source_tz_name)
            target_tz = ZoneInfo(target_tz_name)
        except Exception as e:
            return f"[ERROR] Failed to create timezone: {e}"
        
        # Parse time
        time_obj = parse_time(source_time)
        
        # Get date
        if date:
            try:
                date_obj = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                try:
                    date_obj = datetime.strptime(date, "%m/%d/%Y").date()
                except ValueError:
                    return f"[ERROR] Invalid date format: '{date}'. Use YYYY-MM-DD or MM/DD/YYYY"
        else:
            date_obj = datetime.now(source_tz).date()
        
        # Create datetime
        if time_obj is None:  # "now"
            source_dt = datetime.now(source_tz)
        else:
            # Create naive datetime first
            source_dt = datetime.combine(date_obj, time_obj)
            # Make timezone-aware using replace (zoneinfo doesn't have localize)
            source_dt = source_dt.replace(tzinfo=source_tz)
        
        # Ensure timezone-aware (convert if needed)
        if source_dt.tzinfo is None:
            source_dt = source_dt.replace(tzinfo=source_tz)
        else:
            source_dt = source_dt.astimezone(source_tz)
        
        # Convert to target timezone
        target_dt = source_dt.astimezone(target_tz)
        
        # Calculate time difference
        time_diff_seconds = (target_dt.utcoffset() - source_dt.utcoffset()).total_seconds()
        time_diff_hours = time_diff_seconds / 3600
        
        # Check if same day
        same_day = source_dt.date() == target_dt.date()
        
        # Get abbreviations
        source_abbr = get_timezone_abbreviation(source_dt, source_tz)
        target_abbr = get_timezone_abbreviation(target_dt, target_tz)
        
        # Check DST
        source_dst = is_dst_active(source_dt, source_tz)
        target_dst = is_dst_active(target_dt, target_tz)
        
        # Format output
        return self._format_output(
            source_dt,
            target_dt,
            source_tz_name,
            target_tz_name,
            source_abbr,
            target_abbr,
            time_diff_hours,
            same_day,
            source_dst,
            target_dst,
        )
    
    def _format_output(
        self,
        source_dt: datetime,
        target_dt: datetime,
        source_tz_name: str,
        target_tz_name: str,
        source_abbr: str,
        target_abbr: str,
        time_diff_hours: float,
        same_day: bool,
        source_dst: bool,
        target_dst: bool,
    ) -> str:
        """Format timezone conversion output.
        
        Args:
            source_dt: Source datetime
            target_dt: Target datetime
            source_tz_name: Source timezone name
            target_tz_name: Target timezone name
            source_abbr: Source timezone abbreviation
            target_abbr: Target timezone abbreviation
            time_diff_hours: Time difference in hours
            same_day: Whether same calendar day
            source_dst: Whether source has DST active
            target_dst: Whether target has DST active
            
        Returns:
            Formatted output string
        """
        output_parts = []
        
        # Main conversion
        source_time_str = format_time_12h(source_dt)
        target_time_str = format_time_12h(target_dt)
        
        output_parts.append("Timezone Conversion:")
        output_parts.append("")
        output_parts.append(f"Source: {source_time_str} {source_abbr} ({source_tz_name})")
        output_parts.append(f"Target: {target_time_str} {target_abbr} ({target_tz_name})")
        output_parts.append("")
        
        # Time difference
        if time_diff_hours > 0:
            diff_str = f"{abs(time_diff_hours):.1f} hours ahead"
        elif time_diff_hours < 0:
            diff_str = f"{abs(time_diff_hours):.1f} hours behind"
        else:
            diff_str = "same time"
        
        output_parts.append(f"Time difference: {diff_str}")
        output_parts.append(f"Same calendar day: {'Yes' if same_day else 'No'}")
        output_parts.append("")
        
        # Business hours check
        source_business = is_business_hours(source_dt)
        target_business = is_business_hours(target_dt)
        
        output_parts.append("Business hours check (9 AM - 5 PM):")
        source_indicator = "[OK]" if source_business else "[OUTSIDE]"
        target_indicator = "[OK]" if target_business else "[OUTSIDE]"
        output_parts.append(f"  {source_abbr}: {source_indicator} {'Within business hours' if source_business else 'Outside business hours (after hours)'}")
        output_parts.append(f"  {target_abbr}: {target_indicator} {'Within business hours' if target_business else 'Outside business hours (after hours)'}")
        output_parts.append("")
        
        # DST status
        output_parts.append("Daylight Saving Time:")
        source_dst_str = f"{source_abbr} (Daylight Saving {'active' if source_dst else 'inactive'})"
        target_dst_str = f"{target_abbr} (Daylight Saving {'active' if target_dst else 'inactive'})"
        output_parts.append(f"  {source_tz_name}: Currently {source_dst_str}")
        output_parts.append(f"  {target_tz_name}: Currently {target_dst_str}")
        output_parts.append("")
        
        # Meeting recommendation
        if source_business and target_business:
            output_parts.append("Good time for meeting: Yes, both locations are in business hours.")
        elif not source_business and not target_business:
            output_parts.append("Note: Both locations are outside business hours.")
        else:
            output_parts.append("Note: Only one location is in business hours.")
        
        return "\n".join(output_parts)
    
    def _suggest_timezones(self, invalid_tz: str) -> List[str]:
        """Suggest valid timezone names.
        
        Args:
            invalid_tz: Invalid timezone string
            
        Returns:
            List of suggested timezone names
        """
        suggestions = []
        invalid_lower = invalid_tz.lower()
        
        # Find similar timezones
        for tz_name in TIMEZONE_MAP.keys():
            if invalid_lower in tz_name or tz_name in invalid_lower:
                suggestions.append(tz_name.upper())
        
        # Add common US timezones
        if not suggestions:
            suggestions = ["EST", "PST", "CST", "MST", "EDT", "PDT"]
        
        return suggestions


# Create tool instance
timezone_converter_tool = TimezoneConverterTool()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_timezone_converter():
        """Test the timezone converter."""
        tool = TimezoneConverterTool()
        
        test_cases = [
            ("3 PM EST to PST", "Basic conversion"),
            ("What time is 15:30 CST in EST?", "24-hour format"),
            ("If it's 9 AM in New York, what time is it in LA?", "City names"),
            ("Current time in Pacific timezone", "Current time"),
        ]
        
        print("=" * 70)
        print("TIMEZONE CONVERTER TEST")
        print("=" * 70)
        
        for input_str, description in test_cases:
            print(f"\n{description}: {input_str}")
            print("-" * 70)
            result = await tool._arun(input_str)
            print(result)
            print()
    
    asyncio.run(test_timezone_converter())

