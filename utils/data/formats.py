from utils.data.stats import stats_with_threshold, common_health_indicators
from utils.output.file import json_file
import numpy as np
from typing import Union
import json
import torch

def np2str(array, decimals=2):
    """
    Converts a NumPy value (scalar, array, or matrix) to string with specified decimal places.

    Parameters:
    - array: NumPy scalar, ndarray, or compatible object
    - decimals: Number of decimal places to keep

    Returns:
    - A string or list of strings with formatted numbers
    """
    # Define the format string
    format_str = f"{{:.{decimals}f}}"

    if np.isscalar(array):
        # Format scalar
        return format_str.format(array)
    else:
        # Apply formatting to each element in the array
        vectorized_format = np.vectorize(lambda x: format_str.format(x))
        return vectorized_format(array)

def str2np(string_input):
    """
    Converts a string, list of strings, or nested list of strings back to NumPy float values.

    Parameters:
    - string_input: A string, list of strings, or nested structure representing numbers

    Returns:
    - NumPy float scalar or ndarray
    """
    if isinstance(string_input, str):
        # Single string to float
        return float(string_input)
    elif isinstance(string_input, (list, tuple)):
        # Convert list or tuple to NumPy array of floats
        try:
            return np.array(string_input, dtype=float)
        except ValueError:
            # In case of nested lists (e.g., 2D)
            return np.array([[float(x) for x in row] for row in string_input])
    elif isinstance(string_input, np.ndarray):
        # If it's already an ndarray of strings
        return string_input.astype(float)
    else:
        raise TypeError("Input must be a string, list of strings, or a NumPy array of strings.")


def str2float(string_input: Union[str, list, tuple, np.ndarray]):
    """
    Converts a string, list of strings, or nested list/array of strings into float(s) or a NumPy array of floats.
    [It is used when np.arrary are not JSON-serializable and not supported in its built-in schema ecosystem.]
    Parameters:
    - string_input: A string, list of strings, tuple, or NumPy array containing string representations of numbers.

    Returns:
    - A float if input is a single string, or a list of floats for list/tuple/ndarray string inputs.
    """
    if isinstance(string_input, str):
        # Convert a single string to float
        return float(string_input)

    elif isinstance(string_input, (list, tuple)):
        try:
            # Try to convert directly to a flat array
            return np.array(string_input, dtype=float).tolist()
        except ValueError:
            # If it's nested (e.g., 2D), convert element-wise
            return np.array([[float(x) for x in row] for row in string_input]).tolist()

    elif isinstance(string_input, np.ndarray):
        # Convert ndarray of strings to float type
        return string_input.astype(float).tolist()

    else:
        raise TypeError("Input must be a string, list, tuple, or NumPy array of strings.")

def train_test_formater(input, use, device):
    """
    format the input to specific prompt elements according to the use
    :param input: numerical input
    :param use: decides how to format the input
    :return: elements for prompting
    """

    #--------- standard in-context learning
    #-- for the few-shot training set of the standard in-context learning
    if use in ["train_InContextHIs", "train_InContext"]:
        # input is the training data dictionary
        # return a list of dictionaries as input-output examples
        # to a LLM using raw signals
        return [{"input": json.dumps(common_health_indicators(data)), "output": json.dumps({"label":str(label),
                                                              "reason":"The health indicators show Gaussian distribution signs, indicating healthy " if label==0 else
                                                              "The health indicators DO NOT show Gaussian distribution signs, indicating faulty"}) } for data, label in zip(input["data"], input["label"])]

    # -- for the test set of the standard in-context learning
    elif use in ["test_InContextHIs", "test_InContext"]:
        # input is a test data batch: a list of lists
        return [{"input": json.dumps(common_health_indicators(row)) } for row in input.tolist()]

    if use in ["train_InContextRaw"]:
        # input is the training data dictionary
        # return a list of dictionaries as input-output examples
        # to a LLM using raw signals
        return [{"input": np2str(data), "output": json.dumps({"label":str(label),
                                                              "reason":"The signal shows Gaussian distribution signs, indicating healthy " if label==0 else
                                                              "The signal DO NOT show Gaussian distribution signs, indicating faulty"}) } for data, label in zip(input["data"], input["label"])]

    # -- for the test set of the standard in-context learning
    elif use in ["test_InContextRaw"]:
        # input is a test data batch: a list of lists
        return [{"input": np2str(row) } for row in input.tolist()]

    elif use in ["test_SmileAgent"]:
        # input is a test data batch: a list of lists
        prefix = "./algorithms/temp/args/test_signal_"
        suffix = ".json"
        # convert to a list of dictionary
        metalist=[dict(zip(input[1].keys(), values)) for values in zip(*input[1].values())]
        file_paths = [{"input": json_file(prefix + str(i) + suffix, obj={"data":[float(x) for x in rowdata], "metadata":meta})} for i, (rowdata, meta) in
                      enumerate( zip(input[0].tolist(), metalist) )]

        return file_paths


    elif use in ["train_Smile", "train_SmileAgent", "train_SmileAblation", "train_AFTD"]:
        return [{"stats": stats_with_threshold(d), "label": l, "metadata":m} for d, l, m in zip(input["data"], input["label"], input["metadata"])]

    elif use in ["test_Smile", "test_Smile", "test_SmileAblation", "test_AFTD"]:
        return [[float(x) for x in row] for row in input.tolist()]

    else:
        env_list = [int(m["env"]) for m in input["metadata"]]
        return {"data":torch.tensor(np.array(input["data"]), dtype=torch.float32).to(device),
                "label":torch.tensor(np.array(input["label"])).long().to(device),
                "env":torch.tensor(np.array(env_list)).long().to(device)}

    # else:
    #     raise ValueError("Formats are not found !")
