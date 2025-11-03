"""Unit converter tool for LangChain.

This module provides comprehensive unit conversion capabilities for fleet management
and vehicle-related use cases. Works completely offline with no API required.
Supports natural language input and provides context-aware output with reference values.
"""

import re
from typing import Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type


# ============================================================================
# CONVERSION FACTORS
# ============================================================================

CONVERSIONS: Dict[str, Dict[str, float]] = {
    # Distance conversions
    "distance": {
        # Miles
        "mi_to_km": 1.60934,
        "mi_to_ft": 5280.0,
        "mi_to_m": 1609.34,
        "mi_to_yd": 1760.0,
        # Kilometers
        "km_to_mi": 1.0 / 1.60934,
        "km_to_m": 1000.0,
        "km_to_ft": 3280.84,
        "km_to_yd": 1093.61,
        # Feet
        "ft_to_mi": 1.0 / 5280.0,
        "ft_to_km": 1.0 / 3280.84,
        "ft_to_m": 0.3048,
        "ft_to_yd": 1.0 / 3.0,
        # Meters
        "m_to_mi": 1.0 / 1609.34,
        "m_to_km": 0.001,
        "m_to_ft": 3.28084,
        "m_to_yd": 1.09361,
        # Yards
        "yd_to_mi": 1.0 / 1760.0,
        "yd_to_km": 1.0 / 1093.61,
        "yd_to_m": 0.9144,
        "yd_to_ft": 3.0,
    },
    
    # Volume conversions (Fuel)
    "volume": {
        # Gallons (US)
        "gal_to_l": 3.78541,
        "gal_to_ml": 3785.41,
        # Liters
        "l_to_gal": 1.0 / 3.78541,
        "l_to_ml": 1000.0,
        # Milliliters
        "ml_to_gal": 1.0 / 3785.41,
        "ml_to_l": 0.001,
    },
    
    # Weight conversions
    "weight": {
        # Pounds
        "lb_to_kg": 0.453592,
        "lb_to_ton": 1.0 / 2000.0,
        "lb_to_tonne": 0.453592 / 1000.0,
        # Kilograms
        "kg_to_lb": 1.0 / 0.453592,
        "kg_to_ton": 1.0 / 2000.0 / 0.453592,
        "kg_to_tonne": 0.001,
        # Tons (US, 2000 lbs)
        "ton_to_lb": 2000.0,
        "ton_to_kg": 2000.0 * 0.453592,
        "ton_to_tonne": 0.907185,
        # Metric tons (tonnes)
        "tonne_to_lb": 2204.62,
        "tonne_to_kg": 1000.0,
        "tonne_to_ton": 1.0 / 0.907185,
    },
    
    # Pressure conversions (Tire)
    "pressure": {
        # PSI
        "psi_to_bar": 0.0689476,
        "psi_to_kpa": 6.89476,
        "psi_to_atm": 0.068046,
        # Bar
        "bar_to_psi": 1.0 / 0.0689476,
        "bar_to_kpa": 100.0,
        "bar_to_atm": 0.986923,
        # Kilopascals
        "kpa_to_psi": 1.0 / 6.89476,
        "kpa_to_bar": 0.01,
        "kpa_to_atm": 0.00986923,
        # Atmospheres
        "atm_to_psi": 1.0 / 0.068046,
        "atm_to_bar": 1.0 / 0.986923,
        "atm_to_kpa": 101.325,
    },
    
    # Speed conversions
    "speed": {
        # MPH
        "mph_to_kmh": 1.60934,
        "mph_to_ms": 0.44704,
        "mph_to_knots": 0.868976,
        # km/h
        "kmh_to_mph": 1.0 / 1.60934,
        "kmh_to_ms": 1.0 / 3.6,
        "kmh_to_knots": 0.539957,
        # m/s
        "ms_to_mph": 1.0 / 0.44704,
        "ms_to_kmh": 3.6,
        "ms_to_knots": 1.94384,
        # Knots
        "knots_to_mph": 1.0 / 0.868976,
        "knots_to_kmh": 1.852,
        "knots_to_ms": 0.514444,
    },
    
    # Torque conversions
    "torque": {
        "lbft_to_nm": 1.35582,
        "nm_to_lbft": 1.0 / 1.35582,
    },
}

# Unit aliases mapping (case-insensitive)
UNIT_ALIASES: Dict[str, Tuple[str, str]] = {
    # Distance
    "mi": ("distance", "mi"),
    "mile": ("distance", "mi"),
    "miles": ("distance", "mi"),
    "km": ("distance", "km"),
    "kilometer": ("distance", "km"),
    "kilometers": ("distance", "km"),
    "m": ("distance", "m"),
    "meter": ("distance", "m"),
    "meters": ("distance", "m"),
    "ft": ("distance", "ft"),
    "foot": ("distance", "ft"),
    "feet": ("distance", "ft"),
    "yd": ("distance", "yd"),
    "yard": ("distance", "yd"),
    "yards": ("distance", "yd"),
    
    # Volume
    "gal": ("volume", "gal"),
    "gallon": ("volume", "gal"),
    "gallons": ("volume", "gal"),
    "l": ("volume", "l"),
    "liter": ("volume", "l"),
    "liters": ("volume", "l"),
    "litre": ("volume", "l"),
    "litres": ("volume", "l"),
    "ml": ("volume", "ml"),
    "milliliter": ("volume", "ml"),
    "milliliters": ("volume", "ml"),
    "millilitre": ("volume", "ml"),
    "millilitres": ("volume", "ml"),
    
    # Weight
    "lb": ("weight", "lb"),
    "lbs": ("weight", "lb"),
    "pound": ("weight", "lb"),
    "pounds": ("weight", "lb"),
    "kg": ("weight", "kg"),
    "kilogram": ("weight", "kg"),
    "kilograms": ("weight", "kg"),
    "ton": ("weight", "ton"),
    "tons": ("weight", "ton"),
    "tonne": ("weight", "tonne"),
    "tonnes": ("weight", "tonne"),
    "metric ton": ("weight", "tonne"),
    "metric tons": ("weight", "tonne"),
    
    # Pressure
    "psi": ("pressure", "psi"),
    "bar": ("pressure", "bar"),
    "kpa": ("pressure", "kpa"),
    "kilopascal": ("pressure", "kpa"),
    "kilopascals": ("pressure", "kpa"),
    "atm": ("pressure", "atm"),
    "atmosphere": ("pressure", "atm"),
    "atmospheres": ("pressure", "atm"),
    
    # Speed
    "mph": ("speed", "mph"),
    "kmh": ("speed", "kmh"),
    "km/h": ("speed", "kmh"),
    "kph": ("speed", "kmh"),
    "km per hour": ("speed", "kmh"),
    "ms": ("speed", "ms"),
    "m/s": ("speed", "ms"),
    "meters per second": ("speed", "ms"),
    "knots": ("speed", "knots"),
    "knot": ("speed", "knots"),
    "kt": ("speed", "knots"),
    
    # Temperature
    "f": ("temperature", "f"),
    "fahrenheit": ("temperature", "f"),
    "c": ("temperature", "c"),
    "celsius": ("temperature", "c"),
    "k": ("temperature", "k"),
    "kelvin": ("temperature", "k"),
    
    # Torque
    "lb-ft": ("torque", "lbft"),
    "lbft": ("torque", "lbft"),
    "lb ft": ("torque", "lbft"),
    "foot-pound": ("torque", "lbft"),
    "foot-pounds": ("torque", "lbft"),
    "ft-lb": ("torque", "lbft"),
    "nm": ("torque", "nm"),
    "n⋅m": ("torque", "nm"),
    "n m": ("torque", "nm"),
    "newton-meter": ("torque", "nm"),
    "newton-meters": ("torque", "nm"),
    
    # Fuel efficiency
    "mpg": ("fuel_efficiency", "mpg"),
    "miles per gallon": ("fuel_efficiency", "mpg"),
    "l/100km": ("fuel_efficiency", "l100km"),
    "liters per 100km": ("fuel_efficiency", "l100km"),
    "litres per 100km": ("fuel_efficiency", "l100km"),
    "km/l": ("fuel_efficiency", "kmpl"),
    "km per liter": ("fuel_efficiency", "kmpl"),
    "km per litre": ("fuel_efficiency", "kmpl"),
    "kmpl": ("fuel_efficiency", "kmpl"),
}


# ============================================================================
# REFERENCE VALUES (for context)
# ============================================================================

REFERENCE_VALUES = {
    "pressure": {
        "tire_pressure": {
            "standard": "30-35 PSI (2.1-2.4 bar, 207-241 kPa)",
            "low": "< 28 PSI (< 1.9 bar, < 193 kPa)",
            "high": "> 40 PSI (> 2.8 bar, > 276 kPa)",
        },
    },
    "weight": {
        "sedan": "3,000-4,000 lbs (1,360-1,814 kg)",
        "suv": "4,000-6,000 lbs (1,814-2,722 kg)",
        "truck": "5,000-8,000 lbs (2,268-3,629 kg)",
    },
    "speed": {
        "highway": "65-75 mph (105-121 km/h)",
        "city": "25-35 mph (40-56 km/h)",
    },
    "fuel_efficiency": {
        "sedan": "25-35 MPG (9.4-13.1 L/100km)",
        "suv": "20-25 MPG (11.8-13.8 L/100km)",
        "truck": "15-20 MPG (14.7-18.8 L/100km)",
    },
}


# ============================================================================
# SPECIAL CONVERSIONS (formulas, not simple factors)
# ============================================================================

def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between Fahrenheit, Celsius, and Kelvin.
    
    Args:
        value: Temperature value
        from_unit: Source unit (f, c, k)
        to_unit: Target unit (f, c, k)
        
    Returns:
        Converted temperature value
    """
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    # First convert to Celsius
    if from_unit == "f":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "c":
        celsius = value
    elif from_unit == "k":
        celsius = value - 273.15
    else:
        raise ValueError(f"Invalid temperature unit: {from_unit}")
    
    # Then convert from Celsius to target
    if to_unit == "f":
        return celsius * 9 / 5 + 32
    elif to_unit == "c":
        return celsius
    elif to_unit == "k":
        return celsius + 273.15
    else:
        raise ValueError(f"Invalid temperature unit: {to_unit}")


def convert_fuel_efficiency(value: float, from_unit: str, to_unit: str) -> float:
    """Convert fuel efficiency between MPG, L/100km, and km/L.
    
    Args:
        value: Fuel efficiency value
        from_unit: Source unit (mpg, l100km, kmpl)
        to_unit: Target unit (mpg, l100km, kmpl)
        
    Returns:
        Converted fuel efficiency value
    """
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    # First convert to L/100km
    if from_unit == "mpg":
        l100km = 235.214 / value
    elif from_unit == "l100km":
        l100km = value
    elif from_unit == "kmpl":
        l100km = 100.0 / value
    else:
        raise ValueError(f"Invalid fuel efficiency unit: {from_unit}")
    
    # Then convert from L/100km to target
    if to_unit == "mpg":
        return 235.214 / l100km
    elif to_unit == "l100km":
        return l100km
    elif to_unit == "kmpl":
        return 100.0 / l100km
    else:
        raise ValueError(f"Invalid fuel efficiency unit: {to_unit}")


# ============================================================================
# UNIT NORMALIZATION
# ============================================================================

def normalize_unit(unit_str: str) -> Optional[Tuple[str, str]]:
    """Normalize unit string to (category, base_unit).
    
    Args:
        unit_str: Unit string (case-insensitive)
        
    Returns:
        Tuple of (category, base_unit) or None if not found
    """
    unit_lower = unit_str.lower().strip()
    
    # Handle special formats like "l/100km", "km/l", "l per 100km"
    # Normalize these formats
    unit_lower = unit_lower.replace(" per ", "/")
    unit_lower = unit_lower.replace(" ", "")
    
    # Direct lookup
    if unit_lower in UNIT_ALIASES:
        return UNIT_ALIASES[unit_lower]
    
    # Try common variations
    variations = [
        unit_lower,
        unit_lower.replace("/", ""),
        unit_lower.replace("-", ""),
        unit_lower.replace("_", ""),
    ]
    
    for variant in variations:
        if variant in UNIT_ALIASES:
            return UNIT_ALIASES[variant]
    
    # Try removing common suffixes
    suffixes = ["s", "es", "ed"]
    for suffix in suffixes:
        if unit_lower.endswith(suffix):
            base = unit_lower[:-len(suffix)]
            if base in UNIT_ALIASES:
                return UNIT_ALIASES[base]
    
    # Special handling for "l/100km" variations
    if "l100km" in unit_lower or "literper100km" in unit_lower or "litresper100km" in unit_lower:
        return ("fuel_efficiency", "l100km")
    
    # Special handling for "km/l" variations
    if "kmpl" in unit_lower or "kmperl" in unit_lower or "kmperliter" in unit_lower:
        return ("fuel_efficiency", "kmpl")
    
    return None


def get_conversion_factor(from_unit: str, to_unit: str, category: str) -> Optional[float]:
    """Get conversion factor between two units.
    
    Args:
        from_unit: Source unit
        to_unit: Target unit
        category: Unit category
        
    Returns:
        Conversion factor or None if not found
    """
    if category not in CONVERSIONS:
        return None
    
    conversion_key = f"{from_unit}_to_{to_unit}"
    
    if conversion_key in CONVERSIONS[category]:
        return CONVERSIONS[category][conversion_key]
    
    return None


# ============================================================================
# SMART ROUNDING
# ============================================================================

def smart_round(value: float) -> float:
    """Round value intelligently based on magnitude.
    
    Rules:
    - Small values (< 1): 3 decimal places
    - Medium values (1-1000): 2 decimal places
    - Large values (> 1000): 0 decimal places
    
    Args:
        value: Value to round
        
    Returns:
        Rounded value
    """
    abs_value = abs(value)
    
    if abs_value < 1:
        return round(value, 3)
    elif abs_value < 1000:
        return round(value, 2)
    else:
        return round(value, 0)


# ============================================================================
# NATURAL LANGUAGE PARSING
# ============================================================================

def parse_natural_language(input_str: str) -> Optional[Tuple[float, str, str]]:
    """Parse natural language input to extract value and units.
    
    Supports formats:
    - "convert 35 PSI to bar"
    - "50 miles in kilometers"
    - "100 km to mi"
    - "32 psi to kpa"
    - "30 mpg to l/100km"
    - "75 degrees Fahrenheit to Celsius"
    
    Args:
        input_str: Natural language input string
        
    Returns:
        Tuple of (value, from_unit, to_unit) or None if parsing fails
    """
    input_str = input_str.strip()
    
    # Pattern 1: "convert VALUE UNIT1 to UNIT2"
    # Support units with special characters like "l/100km"
    pattern1 = r"convert\s+([\d.]+)\s+([a-zA-Z0-9/]+(?:\s+[a-zA-Z0-9/]+)?)\s+to\s+([a-zA-Z0-9/]+(?:\s+[a-zA-Z0-9/]+)?)"
    match = re.search(pattern1, input_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        from_unit = match.group(2).strip()
        to_unit = match.group(3).strip()
        return (value, from_unit, to_unit)
    
    # Pattern 2: "VALUE UNIT1 in UNIT2" or "VALUE UNIT1 to UNIT2"
    pattern2 = r"([\d.]+)\s+([a-zA-Z0-9/]+(?:\s+[a-zA-Z0-9/]+)?)\s+(?:in|to)\s+([a-zA-Z0-9/]+(?:\s+[a-zA-Z0-9/]+)?)"
    match = re.search(pattern2, input_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        from_unit = match.group(2).strip()
        to_unit = match.group(3).strip()
        return (value, from_unit, to_unit)
    
    # Pattern 3: "VALUE UNIT1 UNIT2" (no connector)
    pattern3 = r"([\d.]+)\s+([a-zA-Z0-9/]+)\s+([a-zA-Z0-9/]+)"
    match = re.search(pattern3, input_str, re.IGNORECASE)
    if match:
        # Check if units look valid
        word2 = match.group(2).lower().replace("/", "").replace(" ", "")
        word3 = match.group(3).lower().replace("/", "").replace(" ", "")
        
        # Try to match normalized versions
        normalized_word2 = word2.replace("/", "")
        normalized_word3 = word3.replace("/", "")
        
        # Check if these look like units
        if any(normalized_word2 in alias.lower() or alias.lower() in normalized_word2 for alias in UNIT_ALIASES.keys()) or \
           any(normalized_word3 in alias.lower() or alias.lower() in normalized_word3 for alias in UNIT_ALIASES.keys()):
            value = float(match.group(1))
            from_unit = match.group(2)
            to_unit = match.group(3)
            return (value, from_unit, to_unit)
    
    # Pattern 4: Temperature with degree symbol
    pattern4 = r"([\d.]+)\s*°?\s*([fc])\s+(?:to|in)\s+°?\s*([fc])"
    match = re.search(pattern4, input_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        from_unit = match.group(2).upper()
        to_unit = match.group(3).upper()
        return (value, from_unit, to_unit)
    
    # Pattern 5: "VALUE degrees UNIT1 to UNIT2"
    pattern5 = r"([\d.]+)\s+degrees?\s+([a-zA-Z]+)\s+to\s+([a-zA-Z]+)"
    match = re.search(pattern5, input_str, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        from_unit = match.group(2)
        to_unit = match.group(3)
        return (value, from_unit, to_unit)
    
    return None


# ============================================================================
# TOOL IMPLEMENTATION
# ============================================================================

class UnitConverterInput(BaseModel):
    """Input schema for unit converter."""
    input_str: str = Field(description="Conversion query: 'convert VALUE UNIT1 to UNIT2' or 'VALUE UNIT1 to UNIT2'. Example: 'convert 35 PSI to bar'")


class UnitConverterTool(BaseTool):
    """Tool for converting units with focus on vehicle/fleet management.
    
    Supports distance, volume (fuel), weight, pressure (tire), speed,
    temperature, fuel efficiency, and torque conversions.
    Works completely offline with no API required.
    """
    
    name: str = "unit_converter"
    description: str = (
        "Converts between different units for vehicle/fleet management. "
        "Supports: distance (miles, km, ft, m), volume (gallons, liters), "
        "weight (lbs, kg, tons), pressure (PSI, bar, kPa), speed (mph, km/h), "
        "temperature (F, C, K), fuel efficiency (MPG, L/100km), torque (lb-ft, N⋅m). "
        "Input: 'convert VALUE UNIT1 to UNIT2' or 'VALUE UNIT1 to UNIT2'. "
        "Example: 'convert 35 PSI to bar' or '100 miles to km'. "
        "Completely offline, no API required."
    )
    args_schema: Type[BaseModel] = UnitConverterInput
    
    def _run(self, input_str: str) -> str:
        """Synchronous execution (required by LangChain)."""
        import asyncio
        return asyncio.run(self._arun(input_str))
    
    async def _arun(self, input_str: str) -> str:
        """Asynchronous execution."""
        try:
            # Parse natural language input
            parsed = parse_natural_language(input_str)
            
            if parsed:
                val, from_u, to_u = parsed
                return self._convert(val, from_u, to_u)
            
            # If parsing failed, try to extract from input string directly
            # This handles cases where input format is less structured
            parts = input_str.split()
            if len(parts) >= 3:
                try:
                    val = float(parts[0])
                    from_u = parts[1]
                    to_u = parts[-1]
                    return self._convert(val, from_u, to_u)
                except (ValueError, IndexError):
                    pass
            
            return (
                "[ERROR] Could not parse input. "
                "Expected format: 'convert VALUE UNIT1 to UNIT2' or 'VALUE UNIT1 to UNIT2'. "
                f"Got: '{input_str}'"
            )
        except Exception as e:
            return f"Error converting units: {str(e)}"
    
    def _convert(self, value: float, from_unit: str, to_unit: str) -> str:
        """Perform unit conversion.
        
        Args:
            value: Numeric value to convert
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            Formatted conversion result
        """
        # Validate value
        try:
            value = float(value)
        except (ValueError, TypeError):
            return "[ERROR] Value must be numeric"
        
        # Normalize units
        from_norm = normalize_unit(from_unit)
        to_norm = normalize_unit(to_unit)
        
        if not from_norm:
            # Suggest similar units
            suggestions = self._suggest_units(from_unit)
            return (
                f"[ERROR] Unsupported unit: '{from_unit}'. "
                f"Similar units: {', '.join(suggestions[:5])}"
            )
        
        if not to_norm:
            suggestions = self._suggest_units(to_unit)
            return (
                f"[ERROR] Unsupported unit: '{to_unit}'. "
                f"Similar units: {', '.join(suggestions[:5])}"
            )
        
        from_category, from_base = from_norm
        to_category, to_base = to_norm
        
        # Check compatibility
        if from_category != to_category:
            return (
                f"[ERROR] Cannot convert {from_category} to {to_category}. "
                f"Incompatible unit types."
            )
        
        # Special handling for temperature
        if from_category == "temperature":
            try:
                # Check for negative Kelvin
                if from_base == "k" and value < 0:
                    return "[ERROR] Kelvin cannot be negative (absolute temperature scale)"
                
                result = convert_temperature(value, from_base, to_base)
                return self._format_temperature_output(value, from_unit, result, to_unit)
            except ValueError as e:
                return f"[ERROR] {e}"
        
        # Special handling for fuel efficiency
        if from_category == "fuel_efficiency":
            try:
                if value <= 0:
                    return "[ERROR] Fuel efficiency must be positive"
                result = convert_fuel_efficiency(value, from_base, to_base)
                return self._format_fuel_efficiency_output(value, from_unit, result, to_unit)
            except ValueError as e:
                return f"[ERROR] {e}"
        
        # Standard conversion using factors
        conversion_key = f"{from_base}_to_{to_base}"
        factor = get_conversion_factor(from_base, to_base, from_category)
        
        if factor is None:
            return f"[ERROR] Conversion from {from_unit} to {to_unit} not supported"
        
        result = value * factor
        rounded_result = smart_round(result)
        
        return self._format_output(
            value, from_unit, rounded_result, to_unit, from_category
        )
    
    def _format_output(
        self,
        value: float,
        from_unit: str,
        result: float,
        to_unit: str,
        category: str,
    ) -> str:
        """Format conversion output with context.
        
        Args:
            value: Original value
            from_unit: Source unit
            result: Converted value
            to_unit: Target unit
            category: Unit category
            
        Returns:
            Formatted output string
        """
        output_parts = []
        
        # Main conversion
        output_parts.append(f"{value} {from_unit.upper()} = {result} {to_unit.upper()}")
        
        # Add related conversions for certain categories
        if category == "pressure":
            # Show all pressure units
            from_norm = normalize_unit(from_unit)
            if from_norm:
                _, from_base = from_norm
                
                # Convert to all pressure units
                all_units = ["psi", "bar", "kpa"]
                if from_base not in all_units:
                    all_units.append(from_base)
                
                related = []
                for unit in all_units:
                    if unit != from_base:
                        factor = get_conversion_factor(from_base, unit, "pressure")
                        if factor:
                            converted = smart_round(value * factor)
                            related.append(f"{converted} {unit.upper()}")
                
                if related:
                    output_parts.append("")
                    output_parts.append("All pressure units:")
                    output_parts.append(f"  = {' = '.join(related)}")
                
                # Add reference
                output_parts.append("")
                output_parts.append(
                    f"Common tire pressure range: {REFERENCE_VALUES['pressure']['tire_pressure']['standard']}"
                )
                
                # Warn about extreme values
                if from_base == "psi":
                    if value > 40:
                        output_parts.append(
                            f"[WARNING] {value} PSI is unusually high for car tires"
                        )
                    elif value < 20:
                        output_parts.append(
                            f"[WARNING] {value} PSI is unusually low for car tires"
                        )
        
        elif category == "weight":
            # Add vehicle weight references
            output_parts.append("")
            output_parts.append("Vehicle weight references:")
            output_parts.append(
                f"  Typical sedan: {REFERENCE_VALUES['weight']['sedan']}"
            )
            output_parts.append(
                f"  Typical SUV: {REFERENCE_VALUES['weight']['suv']}"
            )
            output_parts.append(
                f"  Typical truck: {REFERENCE_VALUES['weight']['truck']}"
            )
        
        elif category == "speed":
            # Add speed references
            output_parts.append("")
            output_parts.append("Speed references:")
            output_parts.append(f"  Highway: {REFERENCE_VALUES['speed']['highway']}")
            output_parts.append(f"  City: {REFERENCE_VALUES['speed']['city']}")
        
        elif category == "fuel_efficiency":
            # Add fuel efficiency references
            output_parts.append("")
            output_parts.append("Vehicle fuel efficiency references:")
            output_parts.append(
                f"  Typical sedan: {REFERENCE_VALUES['fuel_efficiency']['sedan']}"
            )
            output_parts.append(
                f"  Typical SUV: {REFERENCE_VALUES['fuel_efficiency']['suv']}"
            )
            output_parts.append(
                f"  Typical truck: {REFERENCE_VALUES['fuel_efficiency']['truck']}"
            )
        
        return "\n".join(output_parts)
    
    def _format_temperature_output(
        self, value: float, from_unit: str, result: float, to_unit: str
    ) -> str:
        """Format temperature conversion output.
        
        Args:
            value: Original temperature
            from_unit: Source unit
            result: Converted temperature
            to_unit: Target unit
            
        Returns:
            Formatted output string
        """
        # Format with degree symbol
        from_display = f"°{from_unit.upper()}" if from_unit.upper() in ["F", "C"] else from_unit.upper()
        to_display = f"°{to_unit.upper()}" if to_unit.upper() in ["F", "C"] else to_unit.upper()
        
        rounded_result = smart_round(result)
        
        return f"{value} {from_display} = {rounded_result} {to_display}"
    
    def _format_fuel_efficiency_output(
        self, value: float, from_unit: str, result: float, to_unit: str
    ) -> str:
        """Format fuel efficiency conversion output.
        
        Args:
            value: Original value
            from_unit: Source unit
            result: Converted value
            to_unit: Target unit
            
        Returns:
            Formatted output string
        """
        rounded_result = smart_round(result)
        
        # Format units properly
        from_display = from_unit.upper() if from_unit.lower() != "l100km" else "L/100km"
        to_display = to_unit.upper() if to_unit.lower() != "l100km" else "L/100km"
        if from_unit.lower() == "kmpl":
            from_display = "km/L"
        if to_unit.lower() == "kmpl":
            to_display = "km/L"
        
        return f"{value} {from_display} = {rounded_result} {to_display}"
    
    def _suggest_units(self, unit: str) -> List[str]:
        """Suggest similar unit names.
        
        Args:
            unit: Unit string
            
        Returns:
            List of suggested unit names
        """
        unit_lower = unit.lower()
        suggestions = []
        
        # Find units with similar names
        for alias, (category, _) in UNIT_ALIASES.items():
            if unit_lower in alias or alias in unit_lower:
                suggestions.append(alias)
        
        # Also check by category
        categories = {
            "distance": ["mi", "km", "m", "ft", "yd"],
            "volume": ["gal", "l", "ml"],
            "weight": ["lb", "kg", "ton", "tonne"],
            "pressure": ["psi", "bar", "kpa", "atm"],
            "speed": ["mph", "kmh", "ms", "knots"],
            "temperature": ["f", "c", "k"],
            "torque": ["lbft", "nm"],
            "fuel_efficiency": ["mpg", "l100km", "kmpl"],
        }
        
        # Add common units from all categories
        for category_units in categories.values():
            suggestions.extend(category_units)
        
        # Remove duplicates and limit
        return list(set(suggestions))[:10]


# Create tool instance
unit_converter_tool = UnitConverterTool()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_unit_converter():
        """Test the unit converter with various conversions."""
        tool = UnitConverterTool()
        
        test_cases = [
            ("convert 35 PSI to bar", "Tire pressure"),
            ("100 miles to km", "Distance"),
            ("15 gallons to liters", "Fuel volume"),
            ("75 f to c", "Temperature"),
            ("30 mpg to l/100km", "Fuel efficiency"),
            ("100 lb-ft to nm", "Torque"),
            ("32 psi to kpa", "Pressure with related units"),
        ]
        
        print("=" * 70)
        print("UNIT CONVERTER TEST")
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
            ("100 miles to PSI", "Incompatible units"),
            ("invalid value", "Invalid input"),
            ("-10 k", "Negative Kelvin"),
        ]
        
        for input_str, description in error_cases:
            print(f"\n{description}: {input_str}")
            print("-" * 70)
            result = await tool._arun(input_str)
            print(result)
            print()
    
    asyncio.run(test_unit_converter())

