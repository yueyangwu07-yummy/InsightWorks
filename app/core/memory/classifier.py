"""Memory classifier for detecting long-term memory information in user messages."""

from typing import Dict, Optional

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logging import logger


class MemoryClassifier:
    """Classifier to detect if user input contains long-term memory information.
    
    Uses LLM to classify whether a message contains stable facts that should be
    persisted for long-term memory (vehicle VIN, preferences, personal info, etc.).
    """

    def __init__(self):
        """Initialize the memory classifier with configured LLM."""
        if not settings.FEATURE_MEMORY_CLASSIFIER:
            logger.info("memory_classifier_disabled")
            return
        
        try:
            self.llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                temperature=0.0,  # Deterministic classification
                api_key=settings.LLM_API_KEY,
                max_tokens=500,
            )
            logger.info("memory_classifier_initialized", model=settings.LLM_MODEL)
        except Exception as e:
            logger.error("memory_classifier_init_failed", error=str(e))
            self.llm = None

    async def classify(self, user_message: str) -> Dict[str, any]:
        """Classify if the message contains long-term memory information.
        
        Args:
            user_message: The user's message to classify
            
        Returns:
            Dictionary with classification results:
            {
                "is_memory": bool,
                "memory_type": Optional[str],
                "confidence": float,
                "extracted_facts": Optional[Dict]
            }
        """
        # Feature flag check
        if not settings.FEATURE_MEMORY_CLASSIFIER:
            return {
                "is_memory": False,
                "memory_type": None,
                "confidence": 0.0,
                "extracted_facts": None,
            }
        
        if not self.llm:
            logger.warning("memory_classifier_not_available")
            return {
                "is_memory": False,
                "memory_type": None,
                "confidence": 0.0,
                "extracted_facts": None,
            }

        try:
            classification_prompt = f"""Analyze this user message and determine if it contains long-term memory information that should be remembered.

Long-term memory includes:
- Vehicle information (VIN, make, model, year)
- User preferences (units, timezone, language)
- Personal facts (name, location, contact info)
- Stable relationships or context

User message: "{user_message}"

Respond with a JSON object with these fields:
{{
    "is_memory": true/false,
    "memory_type": "vehicle_info" | "preferences" | "personal_info" | "context" | null,
    "confidence": 0.0-1.0,
    "extracted_facts": {{}} or null
}}

If is_memory is true, extract the facts as a structured dictionary in extracted_facts.
Only return the JSON object, no other text."""

            response = await self.llm.ainvoke(classification_prompt)
            classification_text = response.content.strip()
            
            # Parse JSON response
            import json
            classification = json.loads(classification_text)
            
            logger.info(
                "memory_classified",
                is_memory=classification.get("is_memory", False),
                memory_type=classification.get("memory_type"),
                confidence=classification.get("confidence", 0.0),
            )
            
            return classification
            
        except Exception as e:
            logger.error(
                "memory_classification_failed",
                error=str(e),
                exc_info=True,
            )
            return {
                "is_memory": False,
                "memory_type": None,
                "confidence": 0.0,
                "extracted_facts": None,
            }


# Create singleton instance
memory_classifier = MemoryClassifier()

