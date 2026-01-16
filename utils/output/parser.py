import json
import re
from typing import List
from langchain_core.messages import AIMessage
from utils.schema.detection_schema import DetectionSchema
from utils.schema.generation_schema import StatisticalMeasureList
from langchain_core.output_parsers import PydanticOutputParser

detect_pydantic_parser = PydanticOutputParser(pydantic_object=DetectionSchema)
generation_pydantic_parser = PydanticOutputParser(pydantic_object=StatisticalMeasureList)

def json_tag_parser(message: AIMessage, error_return=False) -> List[dict]:
    """
    Extracts JSON content from a string where JSON is embedded between <json>...</json> or ```json``` code blocks.

    Parameters:
        message (AIMessage): The message object containing the response content.
        error_return: whether to return the error or not

    Returns:
        List[dict]: Parsed list of dictionaries from the JSON content.
    """
    text = message.content.strip()

    # Try to extract contents between <json>...</json> or ```json ... ```
    patterns = [
        r"<json>\s*(.*?)\s*</json>",
        r"```json\s*(.*?)\s*```"
    ]

    matches = []
    for p in patterns:
        found = re.findall(p, text, re.DOTALL)
        if found:
            matches.extend(found)

    # Fallback: if nothing matched, use the whole text
    if not matches:
        matches = [text]

    parsed = []

    for match in matches:
        candidate = match.strip()

        # Fix common formatting issues
        candidate = candidate.replace("\n", "").replace("\r", "").strip()

        # Remove trailing commas before closing list or dict
        candidate = re.sub(r",\s*([\]}])", r"\1", candidate)

        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                parsed.extend(data)
            elif isinstance(data, dict):
                parsed.append(data)
            else:
                raise ValueError("Parsed JSON is neither a list nor a dict.")
        except json.JSONDecodeError as e:
            error_info = f"[JSON Parse Error] Could not decode the following content:\n---\n{candidate}\n---\nError: {e}"
            if error_return:
                parsed.extend({"parse_error":error_info})
            else:
                raise ValueError(error_info)

    return parsed