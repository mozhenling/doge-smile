import json
import numpy as np
import os
import shutil
import torch

def tozip(output_name, dir_to_zip):
    """
    e.g.
    OUTPUT_NAME ='0-FIRMseg_Prostate_sweep_seed0' or a directory
    DIRECTORY_TO_ZIP = r'./outputs'
    """
    shutil.make_archive(output_name, 'zip', dir_to_zip)

class NumpyEncoder(json.JSONEncoder):
    """
     The json.dump() function is designed to work with native Python data types,
     and NumPy's ndarray/torch.tensor is not one of them. To resolve this issue, we need
     to convert the ndarray to a Python list or another JSON-serializable
     data type before saving it to a JSON file.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return json.JSONEncoder.default(self, obj)

def json_file(file_path, obj=None, jsonl=False, readnum = None):
    """Read or write a JSON or JSONL file depending on parameters.

    Args:
        file_path (str): Path to the file.
        obj (object, optional): Object to write. If None, the function will read the file.
        jsonl (bool): If True, use JSON Lines format.
        readnum: number of read JSON lines

    Returns:
        object or str: Loaded object if reading, or file path if writing.
    """
    # Adjust file extension
    ext = '.jsonl' if jsonl else '.json'
    if not str(file_path).endswith(ext):
        file_path = os.path.splitext(file_path)[0] + ext

    # Read
    if obj is None:
        with open(file_path, 'r', encoding='utf-8') as f:
            if jsonl:
                if readnum is not None:
                    return [json.loads(line) for i, line in enumerate(f) if line.strip() and i < readnum]
                else:
                    return [json.loads(line) for line in f if line.strip()]
            else:
                return json.load(f)

    # Write
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            if jsonl:
                for item in obj:
                    f.write(json.dumps(item, cls=NumpyEncoder) + '\n')
            else:
                json.dump(obj, f, cls=NumpyEncoder, indent=2)
        return file_path



