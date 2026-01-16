from pydantic import BaseModel, Field, FilePath
from enum import Enum
from typing import List, Dict, Optional

# -- define a schema for the LLM to extraction the label information
# - Define Label as enumeration type so that the label should only be chosen from [0, 1]
class DetectionLabel(int, Enum):
    """labels to be chosen for machine fault detection"""
    healthy_sate = 0
    faulty_state = 1

class DetectionSchema(BaseModel):
    """label and the reason of obtaining such label for machine fault detection"""
    label: DetectionLabel = Field(..., description="The fault detection label of the machine signal.")
    reason: str = Field(..., description="""Explain the method you use to obtain the fault detection label, e.g., the tool, <name of the tool>, is used and the output is, <output of the tool>.""")

class SmartArgsSchema(BaseModel):
    """ArgsSchema for Smart"""
    signal_path:FilePath = Field(..., description="""The file path for loading the machine signal, e.g. r"./algorithms/args/test_signal_0.json" """)

class AIArgsFileFaultDect(BaseModel):
    """arguments information for ai_args_file_fault_detect"""
    signal_path:FilePath = Field(..., description="""The file path for loading the machine signal, e.g. r"./algorithms/args/test_signal_0.json" """)
    examples_path: Optional[FilePath] = Field(default=None,
                                description="""The file path for loading the few-shot examples, e.g., r"./algorithms/args/examples.json" """)

class AutoArgsFileFaultDect(BaseModel):
    """arguments information for auto_args_file_fault_detect"""
    signal_path: Optional[FilePath] = Field(...,
                                       description="""The file path for loading the machine signal, e.g. r"./algorithms/args/test_signal_0.json" """)
    examples_path: Optional[FilePath] = Field(default=None,
                                              description="""The file path for loading the few-shot examples, e.g., r"./algorithms/args/examples.json" """)

class AIArgsRawFaultDect(BaseModel):
    """arguments information for ai_args_raw_fault_detect"""
    x: List[float] = Field(..., description="The machine signal for fault detection given in Human inputs")
    examples: Optional[List[Dict]] = Field(default=None,
    description="""Few shot examples, possibly given in Human inputs, with each example to be formated as as {"signal": a list of floats, "label": int (0 or 1)}""")

class AutoArgsRawFaultDect(BaseModel):
    """arguments information for auto_args_raw_fault_detect"""
    x:Optional[List[float]] = Field(default=None, description="The machine signal for fault detection given in Human inputs")
    examples: Optional[List[Dict]] = Field(default=None,
    description="""Few shot examples, possibly given in Human inputs, with each example to be formated as as {"signal": a list of floats, "label": int (0 or 1)}""")
