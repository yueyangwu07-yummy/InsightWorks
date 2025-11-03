"""LangGraph tools for enhanced language model capabilities.

This package contains custom tools that can be used with LangGraph to extend
the capabilities of language models. Currently includes tools for web search,
VIN decoding, and other external integrations.
"""

from langchain_core.tools.base import BaseTool

from .driving_distance import driving_distance_tool
from .duckduckgo_search import duckduckgo_search_tool
from .recall_checker import recall_checker_tool
from .road_condition import road_condition_tool
from .straight_distance import straight_distance_tool
from .timezone_converter import timezone_converter_tool
from .tire_pressure import tire_pressure_tool
from .traffic_incident import traffic_incident_tool
from .unit_converter import unit_converter_tool
from .vin_decoder import vin_decoder_tool
from .weather_alert import weather_alert_tool

tools: list[BaseTool] = [
    duckduckgo_search_tool,
    straight_distance_tool,
    driving_distance_tool,
    timezone_converter_tool,
    unit_converter_tool,
    tire_pressure_tool,
    traffic_incident_tool,
    road_condition_tool,
    weather_alert_tool,
    vin_decoder_tool,
    recall_checker_tool,
]
