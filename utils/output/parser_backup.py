import json
import re
from typing import List
from langchain_core.messages import AIMessage
from utils.schema.detection_schema import DetectionSchema
from utils.schema.generation_schema import StatisticalMeasureList
from langchain_core.output_parsers import PydanticOutputParser

detect_pydantic_parser = PydanticOutputParser(pydantic_object=DetectionSchema)
generation_pydantic_parser = PydanticOutputParser(pydantic_object=StatisticalMeasureList)

def json_tag_parser(message: AIMessage) -> List[dict]:
    """
    Extracts JSON content from a string where JSON is embedded between "<json>\s*(.*?)\s*</json>" tags or no tags as fallback.

    Parameters:
        message (AIMessage): The message object containing the text with JSON content.

    Returns:
        list: A list of extracted JSON objects.
    """
    text = message.content

    # Regex patterns to extract JSON within tags or code blocks
    patterns = [r"<json>\s*(.*?)\s*</json>", r"```json\s*(.*?)\s*```"]

    # Extract matches
    matches = [re.findall(p, text, re.DOTALL) for p in patterns]

    # Combine matches from both patterns, flatten the list of lists
    all_matches = [m for mm in matches for m in mm]

    # Fallback: If no <json> tags or json markdown blocks, consider the entire text
    if not all_matches:
        all_matches = [text]

    parsed = []

    for match in all_matches:
        try:
            # Attempt to parse each match as JSON
            parsed.append(json.loads(match.strip()))
        except json.JSONDecodeError as e:
            # Handle invalid JSON with meaningful error messages
            raise ValueError(
                f"Failed to parse JSON from message: {match.strip()}\nError: {e}"
            )

    return parsed