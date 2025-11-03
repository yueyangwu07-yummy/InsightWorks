"""Tire pressure monitor and recommendation tool for LangChain.

This module provides tire pressure recommendations based on vehicle type,
temperature, load, and current pressure. Works offline with optional weather
API integration for automatic temperature detection.
"""

import os
import re
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

# Try to import unit converter for pressure conversion
try:
    from .unit_converter import unit_converter_tool
    UNIT_CONVERTER_AVAILABLE = True
except ImportError:
    UNIT_CONVERTER_AVAILABLE = False
    unit_converter_tool = None


# ============================================================================
# TIRE PRESSURE DATABASE
# ============================================================================

TIRE_PRESSURE_SPECS: Dict[str, Dict[str, int]] = {
    "sedan": {
        "front": 32,  # PSI
        "rear": 32,
        "front_heavy": 35,
        "rear_heavy": 35,
        "front_max": 38,
        "rear_max": 40,
    },
    "suv": {
        "front": 35,
        "rear": 35,
        "front_heavy": 38,
        "rear_heavy": 40,
        "front_max": 40,
        "rear_max": 45,
    },
    "truck": {
        "front": 35,
        "rear": 35,
        "front_heavy": 40,
        "rear_heavy": 50,
        "front_max": 45,
        "rear_max": 55,
    },
    "sports_car": {
        "front": 33,
        "rear": 35,
        "front_heavy": 36,
        "rear_heavy": 38,
        "front_max": 38,
        "rear_max": 40,
    },
    "ev": {
        "front": 42,  # EVs typically higher due to weight
        "rear": 42,
        "front_heavy": 45,
        "rear_heavy": 45,
        "front_max": 47,
        "rear_max": 47,
    },
}

# Vehicle type aliases
VEHICLE_TYPE_ALIASES: Dict[str, str] = {
    "car": "sedan",
    "passenger car": "sedan",
    "vehicle": "sedan",
    "sport": "sports_car",
    "sports": "sports_car",
    "electric vehicle": "ev",
    "electric": "ev",
    "tesla": "ev",
    "pickup": "truck",
    "pickup truck": "truck",
}

# Standard temperature for tire pressure specs (70°F)
STANDARD_TEMPERATURE = 70.0  # Fahrenheit


# ============================================================================
# WEATHER API INTEGRATION (OPTIONAL)
# ============================================================================

OPENWEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
API_TIMEOUT = 10.0  # seconds


async def fetch_temperature_from_location(location: str, api_key: Optional[str] = None) -> Optional[float]:
    """Fetch current temperature from OpenWeatherMap API.
    
    Args:
        location: City/state or city name
        api_key: OpenWeatherMap API key
        
    Returns:
        Temperature in Fahrenheit or None if error
    """
    if not HTTPX_AVAILABLE or not api_key:
        return None
    
    params = {
        "q": location,
        "appid": api_key,
        "units": "imperial",  # Get temperature in Fahrenheit
    }
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(OPENWEATHER_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            temp = data.get("main", {}).get("temp")
            if temp is not None:
                return float(temp)
    except Exception:
        pass
    
    return None


# ============================================================================
# PRESSURE CALCULATIONS
# ============================================================================

def adjust_for_temperature(base_psi: float, current_temp_f: float) -> float:
    """Adjust tire pressure for temperature.
    
    Formula: For every 10°F change from 70°F, pressure changes by ~1 PSI.
    
    Args:
        base_psi: Base recommended pressure (at 70°F)
        current_temp_f: Current temperature in Fahrenheit
        
    Returns:
        Adjusted pressure in PSI
    """
    temp_diff = current_temp_f - STANDARD_TEMPERATURE
    adjustment = temp_diff / 10.0
    return round(base_psi + adjustment, 1)


def adjust_for_load(base_psi: float, load: str) -> float:
    """Adjust tire pressure for load.
    
    Args:
        base_psi: Base recommended pressure
        load: Load type ("normal", "heavy", "max")
        
    Returns:
        Adjusted pressure in PSI
    """
    load = load.lower()
    
    if load == "heavy":
        return base_psi + 3
    elif load == "max" or load == "maximum":
        return base_psi + 5
    elif load == "towing":
        return base_psi + 6
    else:  # normal
        return base_psi


def get_seasonal_recommendation(temp_f: float) -> str:
    """Get seasonal recommendation based on temperature.
    
    Args:
        temp_f: Temperature in Fahrenheit
        
    Returns:
        Seasonal recommendation message
    """
    if temp_f < 40:
        return (
            "[WINTER] Check pressure weekly, cold weather causes pressure drop. "
            "Tires lose ~1 PSI for every 10°F drop in temperature."
        )
    elif temp_f > 85:
        return (
            "[SUMMER] Monitor pressure, heat causes expansion - don't over-inflate. "
            "Pressure increases ~1 PSI for every 10°F rise in temperature."
        )
    return ""


def categorize_pressure_status(current_psi: float, recommended_psi: float) -> Tuple[str, str]:
    """Categorize tire pressure status.
    
    Args:
        current_psi: Current tire pressure
        recommended_psi: Recommended tire pressure
        
    Returns:
        Tuple of (status, message)
    """
    diff = current_psi - recommended_psi
    
    if diff < -10:
        return (
            "CRITICAL",
            "[CRITICAL] Significantly under-inflated (more than 10 PSI low). "
            "Check immediately - high risk of blowout and tire failure."
        )
    elif diff < -5:
        return (
            "WARNING",
            "[WARNING] Under-inflated (5-10 PSI low). "
            "Check and inflate to recommended pressure."
        )
    elif diff < -2:
        return (
            "LOW",
            "[INFO] Slightly under-inflated (2-5 PSI low). "
            "Consider adding air for optimal performance."
        )
    elif diff <= 2:
        return (
            "OK",
            "[OK] Tire pressure is within acceptable range."
        )
    elif diff <= 5:
        return (
            "HIGH",
            "[INFO] Slightly over-inflated (2-5 PSI high). "
            "May reduce comfort and tire contact patch."
        )
    else:  # diff > 5
        return (
            "WARNING",
            "[WARNING] Over-inflated (more than 5 PSI high). "
            "Reduce pressure for safety - risk of reduced traction and blowout."
        )


def calculate_fuel_waste(current_psi: float, recommended_psi: float) -> Dict[str, Any]:
    """Calculate estimated fuel waste from under-inflation.
    
    Args:
        current_psi: Current tire pressure
        recommended_psi: Recommended tire pressure
        
    Returns:
        Dictionary with fuel waste estimates
    """
    if current_psi >= recommended_psi:
        return {"waste_percent": 0, "annual_cost": 0}
    
    # Under-inflation causes 3-5% fuel efficiency loss per 5 PSI
    psi_diff = recommended_psi - current_psi
    efficiency_loss = min(psi_diff / 5.0 * 0.04, 0.10)  # Max 10% loss
    
    # Assume average annual mileage of 12,000 miles and $3.50/gal
    annual_miles = 12000
    mpg = 25  # Average MPG
    gas_price = 3.50
    annual_gallons = annual_miles / mpg
    waste_gallons = annual_gallons * efficiency_loss
    waste_cost = waste_gallons * gas_price
    
    return {
        "waste_percent": round(efficiency_loss * 100, 1),
        "annual_cost": round(waste_cost, 2),
        "waste_gallons": round(waste_gallons, 1),
    }


# ============================================================================
# INPUT PARSING
# ============================================================================

def parse_vehicle_type(input_str: str) -> Optional[str]:
    """Parse vehicle type from input string.
    
    Args:
        input_str: Input string
        
    Returns:
        Vehicle type or None
    """
    input_lower = input_str.lower()
    
    # Direct match
    if input_lower in TIRE_PRESSURE_SPECS:
        return input_lower
    
    # Check aliases
    if input_lower in VEHICLE_TYPE_ALIASES:
        return VEHICLE_TYPE_ALIASES[input_lower]
    
    # Partial match
    for vehicle_type in TIRE_PRESSURE_SPECS.keys():
        if vehicle_type in input_lower:
            return vehicle_type
    
    for alias, vehicle_type in VEHICLE_TYPE_ALIASES.items():
        if alias in input_lower:
            return vehicle_type
    
    return None


def parse_natural_language(input_str: str) -> Dict[str, Any]:
    """Parse natural language input to extract tire pressure parameters.
    
    Args:
        input_str: Natural language query
        
    Returns:
        Dictionary with parsed parameters
    """
    result = {
        "vehicle_type": None,
        "current_psi": None,
        "load": "normal",
        "temperature": None,
        "location": None,
    }
    
    input_lower = input_str.lower()
    
    # Extract vehicle type
    vehicle_type = parse_vehicle_type(input_str)
    if vehicle_type:
        result["vehicle_type"] = vehicle_type
    
    # Extract current PSI
    psi_pattern = r"(\d+(?:\.\d+)?)\s*(?:psi|psig)"
    match = re.search(psi_pattern, input_lower)
    if match:
        result["current_psi"] = float(match.group(1))
    
    # Extract temperature - check original string first for degree symbol
    # Pattern 1: "45°F" or "45 F" (check original string for ° symbol)
    # Try multiple patterns to handle different degree symbols
    temp_patterns = [
        r"(\d+(?:\.\d+)?)\s*[°]\s*[Ff]",  # "45°F"
        r"(\d+(?:\.\d+)?)\s+[Ff](?:\s|$)",  # "45 F" or "45F"
        r"(\d+(?:\.\d+)?)\s*[Ff]",  # "45F" or "45 F"
    ]
    
    match = None
    for pattern in temp_patterns:
        match = re.search(pattern, input_str)
        if match:
            break
    
    if match:
        result["temperature"] = float(match.group(1))
    else:
        # Pattern 2: "45 fahrenheit" or "45 degrees fahrenheit"
        temp_pattern = r"(\d+(?:\.\d+)?)\s*(?:f|fahrenheit|degrees?\s*(?:f|fahrenheit)?)"
        match = re.search(temp_pattern, input_lower)
        if match:
            result["temperature"] = float(match.group(1))
        else:
            # Pattern 3: "temperature is 45" or "it's 45 outside"
            temp_context_pattern = r"(?:temp|temperature|it'?s|is|outside|out)\s+(\d+(?:\.\d+)?)\s*[°]?[Ff]?"
            match = re.search(temp_context_pattern, input_lower)
            if match:
                # Assume Fahrenheit if no unit specified
                result["temperature"] = float(match.group(1))
            else:
                # Try Celsius conversion (rough)
                temp_c_pattern = r"(\d+(?:\.\d+)?)\s*°?\s*(?:c|celsius)"
                match = re.search(temp_c_pattern, input_lower)
                if match:
                    temp_c = float(match.group(1))
                    result["temperature"] = (temp_c * 9/5) + 32
    
    # Extract load type
    if any(word in input_lower for word in ["heavy", "max", "maximum", "full", "cargo"]):
        if any(word in input_lower for word in ["max", "maximum", "towing"]):
            result["load"] = "max"
        else:
            result["load"] = "heavy"
    elif "towing" in input_lower:
        result["load"] = "towing"
    
    # Extract location (city, state pattern)
    location_pattern = r"in\s+([A-Za-z]+(?:\s+[A-Za-z]+)?(?:,\s*[A-Z]{2})?)"
    match = re.search(location_pattern, input_lower)
    if match:
        result["location"] = match.group(1)
    
    return result


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_tire_pressure_output(
    vehicle_type: str,
    recommended_front: float,
    recommended_rear: float,
    current_temp_f: float,
    load: str,
    current_psi: Optional[float] = None,
) -> str:
    """Format tire pressure recommendation output.
    
    Args:
        vehicle_type: Vehicle type
        recommended_front: Recommended front tire pressure
        recommended_rear: Recommended rear tire pressure
        load: Load type
        current_temp_f: Current temperature in Fahrenheit
        current_psi: Current tire pressure (optional)
        
    Returns:
        Formatted output string
    """
    output_parts = []
    
    # Header
    if current_psi:
        output_parts.append("TIRE PRESSURE CHECK")
    else:
        output_parts.append("TIRE PRESSURE RECOMMENDATION")
    output_parts.append("")
    
    # Vehicle and conditions
    vehicle_display = vehicle_type.replace("_", " ").title()
    output_parts.append(f"Vehicle: {vehicle_display}")
    output_parts.append(f"Current Temperature: {current_temp_f:.1f}°F")
    output_parts.append(f"Load: {load.replace('_', ' ').title()}")
    output_parts.append("")
    
    # Recommendations
    output_parts.append("Recommended Pressure:")
    output_parts.append(f"  Front tires: {recommended_front:.1f} PSI")
    output_parts.append(f"  Rear tires: {recommended_rear:.1f} PSI")
    output_parts.append("")
    
    # Current pressure check
    if current_psi:
        output_parts.append(f"Your current pressure: {current_psi:.1f} PSI")
        avg_recommended = (recommended_front + recommended_rear) / 2
        output_parts.append(f"Recommended pressure: {avg_recommended:.1f} PSI (at {current_temp_f:.1f}°F)")
        output_parts.append("")
        
        # Status
        status, status_msg = categorize_pressure_status(current_psi, avg_recommended)
        output_parts.append(status_msg)
        output_parts.append("")
        
        # Adjustment needed
        diff = avg_recommended - current_psi
        if abs(diff) >= 2:
            if diff > 0:
                output_parts.append(
                    f"[ACTION NEEDED] Add {abs(diff):.1f} PSI to reach optimal pressure"
                )
            else:
                output_parts.append(
                    f"[ACTION NEEDED] Reduce {abs(diff):.1f} PSI to reach optimal pressure"
                )
            output_parts.append("")
        
        # Fuel waste calculation
        if current_psi < avg_recommended:
            fuel_waste = calculate_fuel_waste(current_psi, avg_recommended)
            if fuel_waste["waste_percent"] > 0:
                output_parts.append("Why it matters:")
                output_parts.append(
                    f"  - Under-inflated tires reduce fuel efficiency by {fuel_waste['waste_percent']:.1f}%"
                )
                output_parts.append(
                    f"  - Estimated annual fuel waste: ${fuel_waste['annual_cost']:.2f}"
                )
                output_parts.append("  - Increase tire wear and risk of blowout")
                output_parts.append("  - Poor handling and braking performance")
                output_parts.append("")
        
        # Steps
        output_parts.append("Steps:")
        output_parts.append("  1. Drive to nearest gas station with air pump")
        if diff > 0:
            output_parts.append(f"  2. Inflate to {avg_recommended:.1f} PSI (cold pressure)")
        else:
            output_parts.append(f"  2. Reduce pressure to {avg_recommended:.1f} PSI")
        output_parts.append(
            "  3. Recheck after driving (pressure may increase 2-4 PSI when warm)"
        )
        output_parts.append("")
    
    # Temperature adjustment explanation
    temp_diff = current_temp_f - STANDARD_TEMPERATURE
    if abs(temp_diff) >= 10:
        output_parts.append("Temperature Adjustment:")
        if temp_diff < 0:
            output_parts.append(
                f"Current temp ({current_temp_f:.1f}°F) is {abs(temp_diff):.1f}°F below standard (70°F)"
            )
            psi_change = abs(temp_diff) / 10.0
            output_parts.append(
                f"> Tires will lose ~{psi_change:.1f} PSI due to cold weather"
            )
            output_parts.append("> Check pressure and inflate if needed")
        else:
            output_parts.append(
                f"Current temp ({current_temp_f:.1f}°F) is {temp_diff:.1f}°F above standard (70°F)"
            )
            psi_change = temp_diff / 10.0
            output_parts.append(
                f"> Tires will gain ~{psi_change:.1f} PSI due to heat"
            )
            output_parts.append("> Monitor pressure, don't over-inflate")
        output_parts.append("")
    
    # Seasonal recommendation
    seasonal_msg = get_seasonal_recommendation(current_temp_f)
    if seasonal_msg:
        output_parts.append(seasonal_msg)
        output_parts.append("")
    
    # Safety note
    output_parts.append(
        "[NOTE] Tire pressure should be checked when tires are cold (before driving)"
    )
    
    # Additional safety warnings
    if current_psi:
        avg_recommended = (recommended_front + recommended_rear) / 2
        diff = abs(current_psi - avg_recommended)
        if diff > 5:
            output_parts.append("")
            output_parts.append(
                "[SAFETY WARNING] Significant pressure deviation detected. "
                "Check all tires and consult vehicle manual for specific recommendations."
            )
    
    if load != "normal":
        output_parts.append("")
        output_parts.append(
            f"[LOAD WARNING] {load.title()} load detected: "
            "Increase pressure as recommended above for safety and performance."
        )
    
    return "\n".join(output_parts)


# ============================================================================
# TOOL IMPLEMENTATION
# ============================================================================

class TirePressureInput(BaseModel):
    """Input schema for tire pressure tool."""
    input_str: str = Field(description="Tire pressure query with vehicle type, optional current PSI, load, and temperature. Example: 'What tire pressure for my SUV?' or 'Check tire pressure, current 28 PSI, 30°F'")


class TirePressureTool(BaseTool):
    """Tool for tire pressure monitoring and recommendations.
    
    Provides tire pressure recommendations based on vehicle type, temperature,
    load, and current pressure. Works offline with optional weather API integration.
    """
    
    name: str = "tire_pressure"
    description: str = (
        "Provides tire pressure recommendations based on vehicle type, temperature, and load. "
        "Input: vehicle type (sedan, SUV, truck, sports_car, EV), optional current PSI, "
        "load (normal/heavy/max), and temperature or location. "
        "Example: 'What tire pressure for my SUV?' or 'Check tire pressure, current 28 PSI, 30°F'. "
        "Works offline with optional weather API integration for automatic temperature detection."
    )
    args_schema: Type[BaseModel] = TirePressureInput
    
    def _run(self, input_str: str) -> str:
        """Synchronous execution (required by LangChain)."""
        import asyncio
        return asyncio.run(self._arun(input_str))
    
    async def _arun(self, input_str: str) -> str:
        """Asynchronous execution."""
        try:
            # Parse natural language input
            parsed = parse_natural_language(input_str)
            
            vehicle_type = parsed.get("vehicle_type")
            current_psi = parsed.get("current_psi")
            load = parsed.get("load", "normal")
            temperature = parsed.get("temperature")
            location = parsed.get("location")
            
            # Validate vehicle type
            if not vehicle_type:
                available_types = ", ".join(TIRE_PRESSURE_SPECS.keys())
                return (
                    f"[ERROR] Vehicle type required. "
                    f"Available types: {available_types}"
                )
            
            # Normalize vehicle type
            normalized_type = parse_vehicle_type(vehicle_type)
            if not normalized_type:
                available_types = ", ".join(TIRE_PRESSURE_SPECS.keys())
                return (
                    f"[ERROR] Unknown vehicle type: '{vehicle_type}'. "
                    f"Available types: {available_types}"
                )
            
            vehicle_type = normalized_type
            
            # Get temperature
            if not temperature:
                # Try to fetch from weather API if location provided
                if location:
                    api_key = os.getenv("OPENWEATHER_API_KEY", "")
                    if api_key:
                        temp = await fetch_temperature_from_location(location, api_key)
                        if temp:
                            temperature = temp
                        else:
                            return (
                                f"[ERROR] Could not fetch temperature for '{location}'. "
                                "Please provide temperature manually."
                            )
                    else:
                        return (
                            "[ERROR] Location provided but OPENWEATHER_API_KEY not set. "
                            "Please provide temperature manually or set API key."
                        )
                else:
                    # Use standard temperature as default
                    temperature = STANDARD_TEMPERATURE
            
            # Validate temperature
            if temperature < -50 or temperature > 150:
                return (
                    f"[ERROR] Invalid temperature: {temperature}°F. "
                    "Please provide a temperature between -50°F and 150°F."
                )
            
            # Validate load
            if load:
                load = load.lower()
                valid_loads = ["normal", "heavy", "max", "maximum", "towing"]
                if load not in valid_loads:
                    return (
                        f"[ERROR] Invalid load type: '{load}'. "
                        f"Valid types: {', '.join(valid_loads)}"
                    )
            else:
                load = "normal"
            
            # Validate current PSI if provided
            if current_psi is not None:
                if current_psi < 0 or current_psi > 100:
                    return (
                        f"[ERROR] Invalid current pressure: {current_psi} PSI. "
                        "Please provide a value between 0 and 100 PSI."
                    )
            
            # Get base pressure specs
            specs = TIRE_PRESSURE_SPECS[vehicle_type]
            
            # Determine which spec to use based on load
            if load == "max" or load == "maximum" or load == "towing":
                front_base = specs.get("front_max", specs["front"])
                rear_base = specs.get("rear_max", specs["rear_heavy"])
            elif load == "heavy":
                front_base = specs["front_heavy"]
                rear_base = specs["rear_heavy"]
            else:
                front_base = specs["front"]
                rear_base = specs["rear"]
            
            # Adjust for temperature
            recommended_front = adjust_for_temperature(front_base, temperature)
            recommended_rear = adjust_for_temperature(rear_base, temperature)
            
            # Adjust for load (additional adjustment on top of base)
            recommended_front = adjust_for_load(recommended_front, load)
            recommended_rear = adjust_for_load(recommended_rear, load)
            
            # Round to reasonable precision
            recommended_front = round(recommended_front, 1)
            recommended_rear = round(recommended_rear, 1)
            
            return format_tire_pressure_output(
                vehicle_type,
                recommended_front,
                recommended_rear,
                temperature,
                load,
                current_psi,
            )
        except Exception as e:
            return f"Error calculating tire pressure: {str(e)}"


# Create tool instance
tire_pressure_tool = TirePressureTool()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_tire_pressure():
        """Test the tire pressure tool."""
        tool = TirePressureTool()
        
        test_cases = [
            ("What tire pressure for my SUV?", "Basic recommendation"),
            ("Check tire pressure, current 28 PSI, 30°F", "With current pressure and temp"),
            ("Tire pressure for truck with heavy load", "Heavy load"),
            ("Sedan, normal load, 95°F", "Summer heat"),
        ]
        
        print("=" * 70)
        print("TIRE PRESSURE TOOL TEST")
        print("=" * 70)
        
        for input_str, description in test_cases:
            print(f"\n{description}: {input_str}")
            print("-" * 70)
            result = await tool._arun(input_str)
            print(result)
            print()
    
    asyncio.run(test_tire_pressure())

