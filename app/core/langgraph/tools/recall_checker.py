"""Vehicle recall checker tool for LangChain using NHTSA API.

This module provides vehicle recall checking capabilities using the free NHTSA API.
Supports both VIN-based and Make/Model/Year queries. No API key required.
"""

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

# Try to import VIN decoder for enhanced functionality
try:
    from .vin_decoder import vin_decoder_tool
    VIN_DECODER_AVAILABLE = True
except ImportError:
    VIN_DECODER_AVAILABLE = False
    vin_decoder_tool = None


# ============================================================================
# CONFIGURATION
# ============================================================================

NHTSA_RECALLS_API_URL = "https://api.nhtsa.gov/recalls/recallsByVehicle"
NHTSA_VIN_API_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/decodevin"
API_TIMEOUT = 10.0  # seconds

# Risk level categorization keywords
RISK_KEYWORDS = {
    "high": [
        "airbag", "air bag", "brake", "braking", "steering", "seat belt",
        "seatbelt", "seat belt", "crash", "injury", "death", "fire",
        "explosion", "explode", "loss of control", "sudden acceleration",
    ],
    "medium": [
        "engine", "stall", "fuel", "electrical", "battery", "transmission",
        "drive", "power", "vehicle speed", "parking", "emergency",
    ],
    "low": [
        "cosmetic", "software", "update", "label", "sticker", "documentation",
        "manual", "information",
    ],
}


# ============================================================================
# API INTEGRATION
# ============================================================================

async def fetch_recalls_by_vehicle(
    make: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch recalls from NHTSA API by vehicle make, model, and year.
    
    Args:
        make: Vehicle make (e.g., "Honda")
        model: Vehicle model (e.g., "Accord")
        year: Model year (e.g., 2020)
        
    Returns:
        API response as dictionary or None if error
    """
    if not HTTPX_AVAILABLE:
        return {"error": "httpx not available"}
    
    if not make or not model or not year:
        return {"error": "Make, model, and year are required"}
    
    params = {
        "make": make,
        "model": model,
        "modelYear": str(year),
    }
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(NHTSA_RECALLS_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            return data
    except httpx.TimeoutException:
        return {"error": "API request timeout"}
    except httpx.HTTPError as e:
        return {"error": f"HTTP error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def decode_vin(vin: str) -> Optional[Dict[str, Any]]:
    """Decode VIN using NHTSA API to get vehicle details.
    
    Args:
        vin: 17-character VIN
        
    Returns:
        Decoded VIN data or None if error
    """
    if not HTTPX_AVAILABLE:
        return None
    
    if not vin or len(vin) != 17:
        return None
    
    params = {
        "vin": vin,
        "format": "json",
    }
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(NHTSA_VIN_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            return data
    except Exception:
        return None


def categorize_risk(recall: Dict[str, Any]) -> str:
    """Categorize recall risk level based on component and description.
    
    Args:
        recall: Recall data dictionary
        
    Returns:
        Risk level: "HIGH", "MEDIUM", or "LOW"
    """
    component = (recall.get("Component", "") or "").lower()
    summary = (recall.get("Summary", "") or "").lower()
    consequence = (recall.get("Consequence", "") or "").lower()
    
    text = f"{component} {summary} {consequence}"
    
    # Check for high-risk keywords
    for keyword in RISK_KEYWORDS["high"]:
        if keyword in text:
            return "HIGH"
    
    # Check for medium-risk keywords
    for keyword in RISK_KEYWORDS["medium"]:
        if keyword in text:
            return "MEDIUM"
    
    # Check for low-risk keywords
    for keyword in RISK_KEYWORDS["low"]:
        if keyword in text:
            return "LOW"
    
    # Default to medium if uncertain
    return "MEDIUM"


def format_date(date_str: Optional[str]) -> str:
    """Format date string to readable format.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Formatted date string
    """
    if not date_str:
        return "Date not available"
    
    try:
        # Try ISO format first
        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        
        return dt.strftime("%B %d, %Y")
    except Exception:
        # Return as-is if parsing fails
        return date_str


# ============================================================================
# INPUT PARSING
# ============================================================================

def extract_vin(input_str: str) -> Optional[str]:
    """Extract VIN from input string.
    
    Args:
        input_str: Input string
        
    Returns:
        VIN string if found, None otherwise
    """
    # VIN is 17 characters, alphanumeric (no I, O, Q)
    vin_pattern = r"\b([A-HJ-NPR-Z0-9]{17})\b"
    match = re.search(vin_pattern, input_str.upper())
    if match:
        return match.group(1)
    return None


def parse_make_model_year(input_str: str) -> Optional[Tuple[str, str, int]]:
    """Parse Make, Model, and Year from natural language input.
    
    Supports formats:
    - "2020 Honda Accord"
    - "Honda Accord 2020"
    - "Tesla Model 3 2021"
    - "2021 Ford F-150"
    - "Does 2021 Tesla Model 3 have any recalls?"
    
    Args:
        input_str: Natural language input
        
    Returns:
        Tuple of (make, model, year) or None if parsing fails
    """
    input_str = input_str.strip()
    
    # Remove question words and common phrases (be careful not to remove "Model")
    input_str = re.sub(r"^(does|do|check|recall|recalls|for|have|has|any|a|an)\s+", "", input_str, flags=re.IGNORECASE)
    input_str = re.sub(r"\s+(recalls|recall)\s*$", "", input_str, flags=re.IGNORECASE)
    input_str = re.sub(r"\?+$", "", input_str)
    input_str = input_str.strip()
    
    # Pattern 1: YEAR MAKE MODEL (e.g., "2021 Tesla Model 3")
    pattern1 = r"(\d{4})\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(.+?)(?:\s+\d{4})?(?:\s+.*)?$"
    match = re.search(pattern1, input_str)
    if match:
        year = int(match.group(1))
        make = match.group(2).strip()
        model = match.group(3).strip()
        # Clean up model (remove trailing question marks, etc.)
        model = re.sub(r"[?!.,;:]+$", "", model).strip()
        
        # Special handling for "Tesla Model 3" pattern
        # If make ends with "Model", it's actually part of the model name
        make_words = make.split()
        if len(make_words) > 1 and make_words[-1].lower() == "model":
            # "Tesla Model" -> make="Tesla", model="Model 3"
            make = " ".join(make_words[:-1])
            model = f"{make_words[-1]} {model}"
        
        return (make, model, year)
    
    # Pattern 2: MAKE MODEL YEAR
    pattern2 = r"([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(.+?)\s+(\d{4})"
    match = re.search(pattern2, input_str)
    if match:
        make = match.group(1).strip()
        model = match.group(2).strip()
        year = int(match.group(3))
        # Clean up model
        model = re.sub(r"[?!.,;:]+$", "", model).strip()
        # Handle special cases like "Tesla Model 3"
        if "Model" in model and model.split()[0].lower() == "model":
            # This is correct, Model is part of model name
            pass
        elif make.split()[-1].lower() == "model":
            # "Tesla Model" - move Model to model
            make_parts = make.split()
            make = " ".join(make_parts[:-1])
            model = f"{make_parts[-1]} {model}"
        return (make, model, year)
    
    # Pattern 2b: MAKE MODEL (extract year from elsewhere or use current)
    pattern2b = r"([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(.+?)(?:\s+\d{4})?(?:\s+.*)?$"
    match = re.search(pattern2b, input_str)
    if match:
        make = match.group(1).strip()
        model = match.group(2).strip()
        # Try to find year in the full string
        year_match = re.search(r"\b(19|20)\d{2}\b", input_str)
        if year_match:
            year = int(year_match.group())
        else:
            # Use current year as default
            year = datetime.now().year
        # Clean up model
        model = re.sub(r"[?!.,;:]+$", "", model).strip()
        # Handle "Tesla Model 3" pattern
        if len(make.split()) > 1 and make.split()[-1].lower() == "model":
            make_parts = make.split()
            make = " ".join(make_parts[:-1])
            model = f"{make_parts[-1]} {model}"
        return (make, model, year)
    
    return None


def parse_natural_language(input_str: str) -> Dict[str, Any]:
    """Parse natural language input to extract vehicle information.
    
    Args:
        input_str: Natural language query
        
    Returns:
        Dictionary with parsed information
    """
    result = {
        "vin": None,
        "make": None,
        "model": None,
        "year": None,
    }
    
    # Try to extract VIN first
    vin = extract_vin(input_str)
    if vin:
        result["vin"] = vin
        return result
    
    # Try to parse Make/Model/Year
    parsed = parse_make_model_year(input_str)
    if parsed:
        make, model, year = parsed
        result["make"] = make
        result["model"] = model
        result["year"] = year
        return result
    
    return result


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_recall_output(
    recalls: List[Dict[str, Any]],
    vehicle_info: Optional[str] = None,
) -> str:
    """Format recall data into readable output.
    
    Args:
        recalls: List of recall dictionaries
        vehicle_info: Vehicle information string (optional)
        
    Returns:
        Formatted output string
    """
    if not recalls:
        vehicle_str = vehicle_info or "this vehicle"
        return (
            f"[OK] No active recalls found for {vehicle_str}\n"
            "Vehicle appears safe based on NHTSA database."
        )
    
    output_parts = []
    
    # Header
    vehicle_str = vehicle_info or "vehicle"
    output_parts.append(
        f"[WARNING] {len(recalls)} ACTIVE RECALL(S) found for {vehicle_str}"
    )
    output_parts.append("")
    
    # Format each recall
    for i, recall in enumerate(recalls, 1):
        output_parts.append(f"RECALL #{i} (Campaign: {recall.get('NHTSACampaignNumber', 'N/A')})")
        
        # Date
        date_str = format_date(recall.get("ReportReceivedDate"))
        output_parts.append(f"Date: {date_str}")
        
        # Component
        component = recall.get("Component", "Unknown Component")
        output_parts.append(f"Component: {component}")
        
        # Issue/Summary
        summary = recall.get("Summary", recall.get("DefectDescription", "No description available"))
        output_parts.append(f"Issue: {summary}")
        
        # Risk level
        risk = categorize_risk(recall)
        consequence = recall.get("Consequence", "")
        if consequence:
            output_parts.append(f"Risk: {risk} - {consequence}")
        else:
            output_parts.append(f"Risk: {risk}")
        
        # Solution/Remedy
        remedy = recall.get("Remedy", recall.get("RemedyDescription", "Remedy information not available"))
        output_parts.append(f"Solution: {remedy}")
        
        # Status
        status = recall.get("Status", "Unknown")
        output_parts.append(f"Status: {status}")
        
        output_parts.append("")
    
    # Action required notice
    output_parts.append(
        "[WARNING] ACTION REQUIRED: Contact your dealer immediately"
    )
    output_parts.append("NHTSA Hotline: 1-888-327-4236")
    output_parts.append(
        "Visit https://www.nhtsa.gov/recalls for more information"
    )
    
    return "\n".join(output_parts)


# ============================================================================
# TOOL IMPLEMENTATION
# ============================================================================

class RecallCheckerInput(BaseModel):
    """Input schema for recall checker."""
    input_str: str = Field(description="VIN number (17 characters) or 'Make Model Year' (e.g., '2020 Honda Accord')")


class RecallCheckerTool(BaseTool):
    """Tool for checking vehicle recalls using NHTSA API.
    
    Supports both VIN-based and Make/Model/Year queries.
    Provides detailed recall information including risk levels and remedies.
    Completely free, no API key required.
    """
    
    name: str = "recall_checker"
    description: str = (
        "Checks for vehicle recalls using the free NHTSA database. "
        "Input: VIN (17 characters) or 'Make Model Year' (e.g., '2020 Honda Accord'). "
        "Returns recall details including date, component, issue, risk level, and remedy. "
        "Example: 'Check recalls for VIN 1HGBH41JXMN109186' or 'Does 2020 Honda Accord have any recalls?'. "
        "Completely free, no API key required."
    )
    args_schema: Type[BaseModel] = RecallCheckerInput
    
    def _run(self, input_str: str) -> str:
        """Synchronous execution (required by LangChain)."""
        import asyncio
        return asyncio.run(self._arun(input_str))
    
    async def _arun(self, input_str: str) -> str:
        """Asynchronous execution."""
        try:
            # Parse natural language input
            parsed = parse_natural_language(input_str)
            
            if parsed["vin"]:
                return await self._check_by_vin(parsed["vin"])
            elif parsed["make"] and parsed["model"] and parsed["year"]:
                return await self._check_by_vehicle(
                    parsed["make"], parsed["model"], parsed["year"]
                )
            
            return (
                "[ERROR] Could not parse input. "
                "Expected format: 'VIN 1HGBH41JXMN109186' or '2020 Honda Accord'. "
                f"Got: '{input_str}'"
            )
        except Exception as e:
            return f"Error checking recalls: {str(e)}"
    
    async def _check_by_vin(self, vin: str) -> str:
        """Check recalls by VIN.
        
        Args:
            vin: 17-character VIN
            
        Returns:
            Formatted recall information
        """
        # Validate VIN format
        if not vin or len(vin) != 17:
            return (
                "[ERROR] Invalid VIN format. "
                "VIN must be exactly 17 characters (alphanumeric, no I, O, Q)."
            )
        
        # Try to decode VIN first for better context
        vehicle_info = None
        make, model, year = None, None, None
        
        if VIN_DECODER_AVAILABLE and vin_decoder_tool:
            try:
                # Decode VIN to get make/model/year
                decoded_result = await vin_decoder_tool._arun(vin)
                # Extract make/model/year from decoded result if possible
                # This is a simplified extraction - in production, parse the full result
                if "Make:" in decoded_result:
                    # Try to extract make/model/year from the decoded output
                    # For now, we'll use the API directly
                    pass
            except Exception:
                pass
        
        # Try NHTSA VIN decode API
        vin_data = await decode_vin(vin)
        if vin_data and "Results" in vin_data:
            results = vin_data["Results"]
            for result in results:
                variable = result.get("Variable", "")
                value = result.get("Value", "")
                
                if variable == "Make":
                    make = value
                elif variable == "Model":
                    model = value
                elif variable == "Model Year":
                    try:
                        year = int(value) if value and value.isdigit() else None
                    except ValueError:
                        year = None
            
            if make and model and year:
                vehicle_info = f"{year} {make} {model}"
        
        # If we have make/model/year, use the vehicle-based API for more detailed recalls
        if make and model and year:
            recalls_data = await fetch_recalls_by_vehicle(make, model, year)
        else:
            # Fallback: use VIN directly if available in API
            # Note: NHTSA Recalls API primarily uses Make/Model/Year
            # For VIN-only, we might need to use a different endpoint
            return (
                f"[ERROR] Could not decode VIN {vin}. "
                "Please try using Make/Model/Year format instead (e.g., '2020 Honda Accord')."
            )
        
        if not recalls_data:
            return "[ERROR] Unable to fetch recall data. Please try again later."
        
        if "error" in recalls_data:
            return f"[ERROR] {recalls_data['error']}"
        
        recalls = recalls_data.get("results", [])
        
        return format_recall_output(recalls, vehicle_info or f"VIN {vin}")
    
    async def _check_by_vehicle(
        self, make: str, model: str, year: int
    ) -> str:
        """Check recalls by vehicle make, model, and year.
        
        Args:
            make: Vehicle make
            model: Vehicle model
            year: Model year
            
        Returns:
            Formatted recall information
        """
        # Validate year
        current_year = datetime.now().year
        if year < 1900 or year > current_year + 1:
            return (
                f"[ERROR] Invalid year: {year}. "
                f"Year must be between 1900 and {current_year + 1}."
            )
        
        # Fetch recalls
        recalls_data = await fetch_recalls_by_vehicle(make, model, year)
        
        if not recalls_data:
            return "[ERROR] Unable to fetch recall data. Please try again later."
        
        if "error" in recalls_data:
            return f"[ERROR] {recalls_data['error']}"
        
        recalls = recalls_data.get("results", [])
        
        vehicle_info = f"{year} {make} {model}"
        return format_recall_output(recalls, vehicle_info)


# Create tool instance
recall_checker_tool = RecallCheckerTool()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_recall_checker():
        """Test the recall checker tool."""
        tool = RecallCheckerTool()
        
        test_cases = [
            ("Check recalls for 2020 Honda Accord", "Make/Model/Year"),
            ("Does 2021 Tesla Model 3 have any recalls?", "Make/Model/Year with question"),
            ("Recall check: 2019 Ford F-150", "Natural language"),
        ]
        
        print("=" * 70)
        print("RECALL CHECKER TEST")
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
            ("Invalid VIN format", "Error handling"),
        ]
        
        for input_str, description in error_cases:
            print(f"\n{description}: {input_str}")
            print("-" * 70)
            result = await tool._arun("VIN INVALID123")
            print(result)
            print()
    
    asyncio.run(test_recall_checker())

