from utils.data.stats import stats_classifier_single, stats_with_threshold
from langchain_core.tools import tool
from utils.schema.detection_schema import AutoArgsFileFaultDect, AIArgsFileFaultDect, AutoArgsRawFaultDect, AIArgsRawFaultDect
from utils.data.formats import train_test_formater
from typing import Annotated, List, Dict, Optional
from pydantic import FilePath

from utils.output.file import json_file
# https://python.langchain.com/docs/how_to/custom_tools/
################# Tools with no arguments: https://github.com/langchain-ai/langchain/issues/7685

# ----------- Utility Functions -----------
def prepare_examples(examples: Optional[List[Dict]]) -> Optional[List[Dict]]:
    return train_test_formater(examples, use="tool_SmartRaw") if examples else None

@tool("ai_args_file_fault_detect", args_schema=AIArgsFileFaultDect, return_direct=True)
def ai_args_file_fault_detect(signal_path: FilePath, examples_path: Optional[FilePath]=None) -> int:
    """
    The tool is designed for machine fault detection, i.e., classifying the machine signal into 0 (healthy) or 1 (faulty).
    The tool requires the AI agent to call the tool name and extract the file path from human inputs as tool arguments.

    Parameters:
    - signal_path (required): the file path for loading the machine signal, e.g. r"./algorithms/args/test_signal_0.json"
    - examples_path (optional): the file path for loading the few-shot examples, e.g., r"./algorithms/args/examples.json"

    Returns:
    - Predicted label (0 or 1)
    """

    # Read JSON files
    x = json_file(signal_path)
    if examples_path is not None:
        examples = json_file(examples_path)
    else:
        examples = None
    return stats_classifier_single(stats=stats_with_threshold(x), examples=examples)

@tool("auto_args_file_fault_detect", args_schema=AutoArgsFileFaultDect, return_direct=True)
def auto_args_file_fault_detect(signal_path: Optional[FilePath], examples_path: Optional[FilePath]=None) -> int:
    """
    The tool is designed for machine fault detection, i.e., classifying the machine signal into 0 (healthy) or 1 (faulty).
    The tool only requires the AI agent to call the tool name while parsing the tool arguments is done by the tool itself.

    Parameters:
    - signal_path (optional): the file path for loading the machine signal, e.g. r"./algorithms/args/test_signal_0.json"
    - examples_path (optional): the file path for loading the few-shot examples, e.g., r"./algorithms/args/examples.json"

    Returns:
    - Predicted label (0 or 1)
    """

    # Read JSON files
    x = json_file(signal_path)
    if examples_path is not None:
        examples = json_file(examples_path)
    else:
        examples = None
    return stats_classifier_single(stats=stats_with_threshold(x), examples=examples)

@tool("ai_args_raw_fault_detect", args_schema=AIArgsRawFaultDect, return_direct=True)
def ai_args_raw_fault_detect(x: List[float], examples: Optional[List[Dict]] = None) -> int:
    """
    The tool is designed for machine fault detection, i.e., classifying the machine signal x into 0 (healthy) or 1 (faulty).
    The tool requires the AI agent to call the tool name and accurately parse the tool arguments from the context.

    Parameters:
    - x (required):  machine signal, which is a list of floats
    - examples (optional): list of dicts and each dict is as {"signal": a list of floats, "label": int (0 or 1)}

    Returns:
    - Predicted label (0 or 1)
    """
    return stats_classifier_single(stats=stats_with_threshold(x),  examples=prepare_examples(examples))

@tool("auto_args_raw_fault_detect", args_schema=AutoArgsRawFaultDect, return_direct=True)
def auto_args_raw_fault_detect(x: Optional[List[float]], examples: Optional[List[Dict]] = None) -> int:
    """
    The tool is designed for machine fault detection, i.e., classifying the machine signal x into 0 (healthy) or 1 (faulty).
    The tool only requires the AI agent to call the tool name while parsing the tool arguments is done by the tool itself.

    Parameters:
    - x (optional): machine signal, which is a list of floats
    - examples (optional): list of dicts and each dict is as {"signal": a list of floats, "label": int (0 or 1)}

    Returns:
    - Predicted label (0 or 1)
    """
    return stats_classifier_single(stats=stats_with_threshold(x), examples=prepare_examples(examples))