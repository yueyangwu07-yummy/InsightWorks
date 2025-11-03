"""VIN (Vehicle Identification Number) decoder tool for LangChain.

This module provides a comprehensive VIN decoder with both offline and online capabilities.
Offline mode decodes basic VIN information without API calls, while online mode
fetches detailed vehicle specifications from the NHTSA API (free, no API key required).
"""

import asyncio
import re
from typing import Any, Dict, Optional, Tuple

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type


# ============================================================================
# VIN DECODING TABLES
# ============================================================================

# Country/Region codes (first character of VIN)
COUNTRY_CODES = {
    "1": "USA",
    "2": "Canada",
    "3": "Mexico",
    "4": "USA",
    "5": "USA",
    "J": "Japan",
    "K": "South Korea",
    "S": "United Kingdom",
    "T": "Czech Republic",
    "V": "France or Austria",
    "W": "Germany",
    "Y": "Sweden",
    "Z": "Italy",
}

# Common manufacturer codes (first 3 characters)
MANUFACTURER_CODES = {
    # General Motors
    "1G1": "Chevrolet",
    "1G6": "Cadillac",
    "1GC": "Chevrolet Truck",
    "1GM": "Pontiac",
    "1HD": "Harley-Davidson",
    # Ford
    "1FA": "Ford",
    "1FB": "Ford",
    "1FC": "Ford Truck",
    "1FD": "Ford Truck",
    "1FM": "Ford",
    "1FT": "Ford Truck",
    # Honda
    "1HG": "Honda",
    "1HT": "Honda Truck",
    "19U": "Acura",
    "19X": "Acura",
    # Toyota
    "4T1": "Toyota",
    "4T3": "Toyota",
    "4TA": "Toyota Truck",
    "4TF": "Toyota Truck",
    "JTD": "Toyota",
    "JTK": "Toyota",
    "JTM": "Toyota",
    # Hyundai
    "2HM": "Hyundai",
    "2HK": "Hyundai",
    "5NP": "Hyundai",
    "5N3": "Hyundai",
    "KMH": "Hyundai",
    "KM8": "Hyundai",
    # Volkswagen
    "3VW": "Volkswagen",
    "WVW": "Volkswagen",
    "1VW": "Volkswagen",
    # Tesla
    "5YJ": "Tesla",
    "5XK": "Tesla",
    # Mazda
    "JM1": "Mazda",
    "JM7": "Mazda",
    # BMW
    "WBA": "BMW",
    "WBS": "BMW",
    "5UX": "BMW",
    # Mercedes-Benz
    "WDD": "Mercedes-Benz",
    "WDF": "Mercedes-Benz",
    "WDY": "Mercedes-Benz",
    # Nissan
    "1N4": "Nissan",
    "1N6": "Nissan",
    "JN1": "Nissan",
    "JN8": "Nissan",
    # Subaru
    "JF1": "Subaru",
    "JF2": "Subaru",
}

# Year codes (position 10) - repeats every 30 years
YEAR_CODES = {
    "A": (1980, 2010),
    "B": (1981, 2011),
    "C": (1982, 2012),
    "D": (1983, 2013),
    "E": (1984, 2014),
    "F": (1985, 2015),
    "G": (1986, 2016),
    "H": (1987, 2017),
    "J": (1988, 2018),
    "K": (1989, 2019),
    "L": (1990, 2020),
    "M": (1991, 2021),
    "N": (1992, 2022),
    "P": (1993, 2023),
    "R": (1994, 2024),
    "S": (1995, 2025),
    "T": (1996, 2026),
    "V": (1997, 2027),
    "W": (1998, 2028),
    "X": (1999, 2029),
    "Y": (2000, 2030),
}

# Add numeric years (2001-2009, 2031-2039)
for i in range(1, 10):
    YEAR_CODES[str(i)] = (2000 + i, 2030 + i)


# ============================================================================
# CHECK DIGIT CALCULATION (ISO 3779 Standard)
# ============================================================================

def char_to_value(char: str) -> int:
    """Convert VIN character to numeric value for check digit calculation.
    
    ISO 3779 Standard mapping:
    - Numbers: 0=0, 1=1, ..., 9=9
    - Letters: 
      A=1, B=2, C=3, D=4, E=5, F=6, G=7, H=8
      J=1, K=2, L=3, M=4, N=5, P=7, R=9
      S=2, T=3, U=4, V=5, W=6, X=7, Y=8, Z=9
    - I, O, Q are not allowed in VINs (except in check digit position)
    """
    if char.isdigit():
        return int(char)
    elif char.isalpha():
        char_upper = char.upper()
        
        # Mapping table per ISO 3779 standard
        # Note: Values wrap around (J=1, S=2, etc.)
        letter_values = {
            "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
            "J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "P": 7, "R": 9,
            "S": 2, "T": 3, "U": 4, "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9,
        }
        
        if char_upper in letter_values:
            return letter_values[char_upper]
        elif char_upper in "IOQ":
            # I, O, Q are not allowed in VIN positions except check digit
            raise ValueError(f"{char_upper} is not allowed in VIN (confused with 1/0/O)")
        else:
            raise ValueError(f"Invalid letter in VIN: {char_upper}")
    else:
        raise ValueError(f"Invalid character in VIN: {char}")


def calculate_check_digit(vin: str) -> str:
    """Calculate the check digit for a VIN (ISO 3779 standard).
    
    Algorithm:
    1. Assign weights: [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]
    2. Convert each character to numeric value
    3. Multiply by corresponding weight
    4. Sum all products
    5. Modulo 11 gives check digit (10 is represented as "X")
    
    Args:
        vin: 17-character VIN string
        
    Returns:
        Check digit as string ("0"-"9" or "X")
    """
    weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
    
    if len(vin) != 17:
        raise ValueError(f"VIN must be exactly 17 characters, got {len(vin)}")
    
    total = 0
    for i, char in enumerate(vin):
        value = char_to_value(char)
        total += value * weights[i]
    
    remainder = total % 11
    
    # If remainder is 10, check digit is "X"
    if remainder == 10:
        return "X"
    else:
        return str(remainder)


def validate_vin(vin: str) -> Tuple[bool, Optional[str]]:
    """Validate VIN format and check digit.
    
    Args:
        vin: VIN string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check length
    if len(vin) != 17:
        return False, f"VIN must be exactly 17 characters, got {len(vin)}"
    
    vin_upper = vin.upper()
    
    # Check for invalid characters (I, O, Q)
    invalid_chars = []
    for i, char in enumerate(vin_upper):
        if char in "IOQ" and i != 8:  # Position 8 can be I, O, Q for check digit
            invalid_chars.append(f"{char} at position {i+1}")
        elif not (char.isalnum()):
            invalid_chars.append(f"{char} at position {i+1} (non-alphanumeric)")
    
    if invalid_chars:
        return False, f"VIN contains invalid characters: {', '.join(invalid_chars)}"
    
    # Validate check digit (position 9, index 8)
    try:
        expected_check = calculate_check_digit(vin_upper)
        actual_check = vin_upper[8]
        
        if actual_check != expected_check:
            return False, f"Invalid check digit. Expected '{expected_check}', got '{actual_check}'"
    except ValueError as e:
        return False, str(e)
    
    return True, None


# ============================================================================
# OFFLINE DECODING
# ============================================================================

def decode_vin_offline(vin: str) -> Dict[str, Any]:
    """Decode VIN in offline mode (no API required).
    
    Extracts:
    - VIN validity
    - Country of manufacture
    - Manufacturer (if known)
    - Model year
    - Check digit validation
    - Plant code
    - Serial number
    
    Args:
        vin: 17-character VIN string
        
    Returns:
        Dictionary with decoded information
    """
    vin_upper = vin.upper()
    
    # Validate VIN
    is_valid, error_msg = validate_vin(vin_upper)
    
    result = {
        "vin": vin_upper,
        "valid": is_valid,
        "error": error_msg,
    }
    
    if not is_valid:
        return result
    
    # Country code (position 1)
    country_char = vin_upper[0]
    country = COUNTRY_CODES.get(country_char, "Unknown")
    result["country"] = country
    
    # Manufacturer (positions 1-3)
    manufacturer_code = vin_upper[:3]
    manufacturer = MANUFACTURER_CODES.get(manufacturer_code, "Unknown")
    result["manufacturer_code"] = manufacturer_code
    result["manufacturer"] = manufacturer
    
    # Model year (position 10, index 9)
    year_char = vin_upper[9]
    year_info = YEAR_CODES.get(year_char)
    if year_info:
        # For vehicles after 2010, prefer the newer year
        # For older vehicles, use the earlier year
        # Simple heuristic: if we're past 2010, use the newer range
        year_candidates = year_info
        # Use more recent year if available
        result["year_code"] = year_char
        result["year"] = year_candidates[1] if year_candidates[1] <= 2030 else year_candidates[0]
        result["year_candidates"] = year_candidates
    else:
        result["year_code"] = year_char
        result["year"] = "Unknown"
    
    # Check digit (position 9, index 8)
    check_digit = vin_upper[8]
    expected_check = calculate_check_digit(vin_upper)
    result["check_digit"] = check_digit
    result["check_digit_valid"] = check_digit == expected_check
    
    # Plant code (position 11, index 10)
    result["plant_code"] = vin_upper[10]
    
    # Serial number (positions 12-17, indices 11-16)
    result["serial_number"] = vin_upper[11:17]
    
    return result


# ============================================================================
# ONLINE DECODING (NHTSA API)
# ============================================================================

async def fetch_nhtsa_data(vin: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """Fetch detailed VIN information from NHTSA API.
    
    NHTSA API is completely free, no API key required.
    Endpoint: https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{vin}?format=json
    
    Args:
        vin: 17-character VIN string
        timeout: Request timeout in seconds
        
    Returns:
        Parsed API response or None if error
    """
    if not HTTPX_AVAILABLE:
        return None
    
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{vin}?format=json"
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            # NHTSA API returns {"Results": [array of objects]}
            if "Results" in data and isinstance(data["Results"], list):
                return data["Results"]
            return None
    except httpx.TimeoutException:
        return None
    except httpx.HTTPError:
        return None
    except Exception:
        return None


def parse_nhtsa_results(nhtsa_results: list) -> Dict[str, Any]:
    """Parse NHTSA API results into structured format.
    
    Args:
        nhtsa_results: List of result objects from NHTSA API
        
    Returns:
        Dictionary with parsed vehicle information
    """
    if not nhtsa_results:
        return {}
    
    # Create lookup dictionary from NHTSA results
    nhtsa_dict = {}
    for item in nhtsa_results:
        variable = item.get("Variable", "")
        value = item.get("Value", "")
        if variable and value and value != "Not Applicable":
            nhtsa_dict[variable] = value
    
    result = {}
    
    # Extract key fields
    result["make"] = nhtsa_dict.get("Make", "")
    result["model"] = nhtsa_dict.get("Model", "")
    result["model_year"] = nhtsa_dict.get("Model Year", "")
    result["body_class"] = nhtsa_dict.get("Body Class", "")
    result["engine_configuration"] = nhtsa_dict.get("Engine Configuration", "")
    result["engine_cylinders"] = nhtsa_dict.get("Engine Cylinders", "")
    result["engine_displacement"] = nhtsa_dict.get("Displacement (L)", "")
    result["fuel_type_primary"] = nhtsa_dict.get("Fuel Type - Primary", "")
    result["manufacturer_name"] = nhtsa_dict.get("Manufacturer Name", "")
    result["plant_country"] = nhtsa_dict.get("Plant Country", "")
    result["vehicle_type"] = nhtsa_dict.get("Vehicle Type", "")
    result["drive_type"] = nhtsa_dict.get("Drive Type", "")
    result["transmission_style"] = nhtsa_dict.get("Transmission Style", "")
    
    # Clean up empty strings
    result = {k: v for k, v in result.items() if v}
    
    return result


# ============================================================================
# TOOL IMPLEMENTATION
# ============================================================================

class VINDecoderInput(BaseModel):
    """Input schema for VIN decoder."""
    input_str: str = Field(description="VIN string (17 characters) or 'VIN detailed' for detailed lookup. Example: '1HGBH41JXMN109186' or '1HGBH41JXMN109186 detailed'")


class VINDecoderTool(BaseTool):
    """Tool for decoding Vehicle Identification Numbers (VINs).
    
    Supports both offline decoding (basic information) and online decoding
    (detailed specifications from NHTSA API).
    
    Offline mode decodes:
    - VIN validity
    - Country of manufacture
    - Manufacturer code
    - Model year
    - Check digit validation
    - Plant code
    - Serial number
    
    Online mode additionally provides:
    - Make and model
    - Body type
    - Engine specifications
    - Fuel type
    - Manufacturer name
    - Plant country
    - Vehicle type
    """
    
    name: str = "vin_decoder"
    description: str = (
        "Decodes Vehicle Identification Numbers (VINs). "
        "Input: VIN string (17 characters) or 'VIN detailed' for detailed NHTSA API lookup. "
        "Offline mode provides basic decoding (country, year, manufacturer code). "
        "Online mode (add 'detailed') fetches detailed specs from NHTSA (make, model, engine, etc.). "
        "Example: '1HGBH41JXMN109186' or '1HGBH41JXMN109186 detailed'"
    )
    args_schema: Type[BaseModel] = VINDecoderInput
    
    def _run(self, input_str: str) -> str:
        """Synchronous execution (required by LangChain)."""
        return asyncio.run(self._arun(input_str))
    
    async def _arun(self, input_str: str) -> str:
        """Asynchronous execution."""
        try:
            # Parse input to extract VIN and detailed flag
            input_str = input_str.strip()
            detailed = "detailed" in input_str.lower()
            
            # Extract VIN (17 characters, alphanumeric)
            vin_match = re.search(r'([A-HJ-NPR-Z0-9]{17})', input_str.upper())
            if not vin_match:
                return "[ERROR] Invalid VIN format. VIN must be exactly 17 alphanumeric characters (no I, O, Q)."
            
            vin = vin_match.group(1)
            
            # Strip whitespace and convert to uppercase
            vin = vin.strip().upper()
            
            # Decode offline first
            offline_result = decode_vin_offline(vin)
            
            # If VIN is invalid, return error
            if not offline_result["valid"]:
                return f"[INVALID] Invalid VIN: {offline_result['error']}\n\nVIN: {vin}"
            
            # Build output string
            output_parts = []
            output_parts.append(f"VIN: {vin}")
            output_parts.append(f"Valid: [OK] Yes")
            output_parts.append("")
            
            # Basic information
            output_parts.append("Basic Information:")
            output_parts.append(f"  Country: {offline_result.get('country', 'Unknown')}")
            
            manufacturer = offline_result.get('manufacturer', 'Unknown')
            if manufacturer != "Unknown":
                output_parts.append(f"  Manufacturer: {manufacturer} ({offline_result.get('manufacturer_code', '')})")
            else:
                output_parts.append(f"  Manufacturer Code: {offline_result.get('manufacturer_code', 'Unknown')}")
            
            year = offline_result.get('year', 'Unknown')
            output_parts.append(f"  Model Year: {year} (Code: {offline_result.get('year_code', '')})")
            
            check_valid = offline_result.get('check_digit_valid', False)
            check_digit = offline_result.get('check_digit', '')
            output_parts.append(f"  Check Digit: {'[OK] Valid' if check_valid else '[ERROR] Invalid'} ({check_digit})")
            output_parts.append(f"  Plant Code: {offline_result.get('plant_code', '')}")
            output_parts.append(f"  Serial Number: {offline_result.get('serial_number', '')}")
            
            # Online mode: fetch from NHTSA
            if detailed:
                output_parts.append("")
                output_parts.append("Fetching detailed information from NHTSA API...")
                
                nhtsa_results = await fetch_nhtsa_data(vin)
                
                if nhtsa_results:
                    online_data = parse_nhtsa_results(nhtsa_results)
                    
                    if online_data:
                        output_parts.append("")
                        output_parts.append("Detailed Specifications:")
                        
                        if online_data.get("make"):
                            output_parts.append(f"  Make: {online_data['make']}")
                        if online_data.get("model"):
                            output_parts.append(f"  Model: {online_data['model']}")
                        if online_data.get("model_year"):
                            output_parts.append(f"  Model Year: {online_data['model_year']}")
                        if online_data.get("body_class"):
                            output_parts.append(f"  Body Type: {online_data['body_class']}")
                        if online_data.get("vehicle_type"):
                            output_parts.append(f"  Vehicle Type: {online_data['vehicle_type']}")
                        
                        # Engine information
                        engine_parts = []
                        if online_data.get("engine_configuration"):
                            engine_parts.append(online_data["engine_configuration"])
                        if online_data.get("engine_cylinders"):
                            engine_parts.append(f"{online_data['engine_cylinders']} cylinders")
                        if online_data.get("engine_displacement"):
                            engine_parts.append(f"{online_data['engine_displacement']}L")
                        
                        if engine_parts:
                            output_parts.append(f"  Engine: {' '.join(engine_parts)}")
                        
                        if online_data.get("fuel_type_primary"):
                            output_parts.append(f"  Fuel Type: {online_data['fuel_type_primary']}")
                        
                        if online_data.get("drive_type"):
                            output_parts.append(f"  Drive Type: {online_data['drive_type']}")
                        if online_data.get("transmission_style"):
                            output_parts.append(f"  Transmission: {online_data['transmission_style']}")
                        
                        if online_data.get("manufacturer_name"):
                            output_parts.append(f"  Manufacturer: {online_data['manufacturer_name']}")
                        if online_data.get("plant_country"):
                            output_parts.append(f"  Plant Country: {online_data['plant_country']}")
                    else:
                        output_parts.append("  [WARNING] No detailed data available from NHTSA API")
                else:
                    output_parts.append("  [WARNING] Could not fetch data from NHTSA API (network error or timeout)")
                    output_parts.append("  Using offline decoding results only")
            
            return "\n".join(output_parts)
        except Exception as e:
            return f"Error decoding VIN: {str(e)}"


# Create tool instance
vin_decoder_tool = VINDecoderTool()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_vin_decoder():
        """Test the VIN decoder with various VINs."""
        tool = VINDecoderTool()
        
        test_vins = [
            ("1HGBH41JXMN109186", "Valid Honda"),
            ("1FADP3F20FL190503", "Valid Ford"),
            ("5YJSA1E14HF200227", "Valid Tesla"),
            ("4T1BF1FK5CU123456", "Valid Toyota"),
        ]
        
        print("=" * 70)
        print("VIN DECODER TEST - OFFLINE MODE")
        print("=" * 70)
        
        for vin, description in test_vins:
            print(f"\n{description}: {vin}")
            print("-" * 70)
            result = await tool._arun(vin, detailed=False)
            print(result)
            print()
        
        print("\n" + "=" * 70)
        print("VIN DECODER TEST - ONLINE MODE (with NHTSA API)")
        print("=" * 70)
        
        # Test online mode with one VIN
        if test_vins:
            vin, desc = test_vins[0]
            print(f"\n{desc}: {vin}")
            print("-" * 70)
            result = await tool._arun(vin, detailed=True)
            print(result)
        
        print("\n" + "=" * 70)
        print("INVALID VIN TEST")
        print("=" * 70)
        
        invalid_vins = [
            ("1HGBH41JXMN109186X", "Too long"),
            ("1HGBH41JXMN10918", "Too short"),
            ("1HGBH41JXMN10918O", "Contains O"),
            ("1HGBH41JXMN10918Q", "Contains Q"),
        ]
        
        for vin, reason in invalid_vins:
            print(f"\nInvalid VIN ({reason}): {vin}")
            print("-" * 70)
            try:
                result = await tool._arun(vin, detailed=False)
                print(result)
            except Exception as e:
                print(f"Error: {e}")
    
    asyncio.run(test_vin_decoder())

