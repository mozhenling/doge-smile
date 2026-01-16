
import json
import copy
import os
import warnings
import shutil
import numpy as np
from typing import Optional
from pydantic import FilePath

from utils.output.file import json_file
from utils.data.distances import clark_distance
from utils.schema.detection_schema import SmartArgsSchema
from utils.data.stats import stats_with_threshold, stats_classifier_single

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from langchain_core.callbacks import CallbackManagerForToolRun

class Smile(BaseTool):
    name: str = "smile"
    description: str = """
    'smile', is short for 'statistical measure integrated learner'.
    This tool is designed for rotating machine/bearing fault detection, i.e., classifying the machine/bearing signal into 0 (healthy) or 1 (faulty).
    This tool is useful for fault detections of rotating machines/bearings, especially when data is skewed, limited, and out-of-dsitributed.
    When using this tool, it is assumed that the signal has been preprocessed, including de-trending, pre-whitening, and z-score normalization.
    When called by an AI agent, the argument, 'signal_path', should be provided by the AI agent for this tool to load and classify the signal.
    """
    args_schema: Optional[ArgsSchema] = SmartArgsSchema
    return_direct: bool = True

    class Config:
        extra = 'allow'

    def __init__(self, config, train_examples=(), metadatabase_preview_retriever=None,
                 adapt_examples=None, agent_part=False):
        super(Smile, self).__init__()
        # use to differentiate LLM and non-LLM algorithms
        self.is_llm = False
        self.train_required = False

        self.config = config
        self.database_path = config["database_path"]

        if self.config["start_new"] and not agent_part:
            self.remove_database(self.database_path)

        self.metadatabase_preview_retriever = metadatabase_preview_retriever

        self.local_top_k = self.config["local_top_k"] if "local_top_k" in self.config else 10
        self.eps = 10 ** -16
        """
        The databases should be a list of dicts, with each dict is an 'example' as follows:
         {"stats":stats of a signal, 
         "label": 0 or 1, 
         "metadata":{"id": unique value for this instance
                    "source":data source, e.g., name of the dataset,
                    "sensor": sensor types, such as vibration, 
                    "time":time of adding the data to databases,
                    "freq":sampling frequency,
                    "length":signal length,
                    "ref": a link or citation for further details}
                    }
        """
        if not agent_part:
            self.database = []
            self.database_ids = []
            if train_examples is not None:
                self.add_entries(train_examples)
            if adapt_examples is not None:
                self.add_entries(adapt_examples)
        else:
            self.database = train_examples
            self.database_ids = self.get_databaes_idx()

        # Store those that given by the single group NN classifier. The dictionary will change in-place
        self.test_time_extra_database = {"f_examples":[], "n_examples":[]}

    def _run(self, signal_path: FilePath, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""

        # Read JSON file for a single instance
        x = json_file(signal_path)

        # Add databases selection
        if self.metadatabase_preview_retriever is not None and isinstance(x, dict):
            query = json.dumps(x['metadata'])
            doc = self.metadatabase_preview_retriever.invoke(query)
            # if len(doc)>1:
            #     database_paths = [d.metadata["source"] for d in doc]
            #     databases = self.concat_databases(database_paths=database_paths)
            # else:
            file_path = doc[0].metadata["source"]
            print(f"Select: {file_path}")
            database = json_file(file_path=file_path, jsonl=True)
            tool_output = self.predict( x["data"] , is_batch=False, database=database)
        else:
            tool_output = self.predict(x, is_batch=False)

        tool_content = json.dumps({
            "tool_output": tool_output,
            "tool_name": "smart",
            "tool_aim": "classify the machine signal into one of the two labels: 0 (healthy) or 1 (faulty)",
            "tool_run": "successful!" if tool_output is not None else "unsuccessful!"
        })

        return tool_content

    def predict(self, x, is_batch=True, database=None, database_path=None):
        """
        prediction
        :param x: an instance or a batch of test data, a list of lists with each inner list as a list of floats
        :return: a batch of predicted labels
        """
        database = self.database if database is None else database
        database_path = self.database_path if database_path is None else database_path

        if is_batch:
            stats_batch = [stats_with_threshold(xi) for xi in x]
            exampls_batch = [self.similar_examples(stats, database, database_path=database_path) for stats in stats_batch]
            return [stats_classifier_single(stats, examples=egs,
                                            test_time_extra_database=self.test_time_extra_database,
                                            test_ulb_max=self.config["smile_test_ulb_max"],
                                            f_sensitivity=self.config["f_sensitivity"],
                                            ) for stats, egs in zip(stats_batch, exampls_batch) ]
        else:
            stats = stats_with_threshold(x)
            egs = self.similar_examples(stats, database, database_path=database_path)
            return stats_classifier_single(stats, examples=egs,
                                           test_time_extra_database=self.test_time_extra_database,
                                           test_ulb_max=self.config["smile_test_ulb_max"],
                                           f_sensitivity=self.config["f_sensitivity"])

    def similar_examples(self, stats, database, database_path=None):
        database = self.database if database is None else database
        database_path = self.database_path if database_path is None else database_path

        f_len = len(self.test_time_extra_database["f_examples"])
        n_len = len(self.test_time_extra_database["n_examples"])

        if f_len >0 and f_len < self.config["smile_test_ulb_max"]:
            self.add_entries(self.test_time_extra_database["f_examples"],
                             database_path, remark=f"\n\tRemark: current test_time_f_examples num = {f_len}")

        if n_len >0 and n_len < self.config["smile_test_ulb_max"]:
            self.add_entries(self.test_time_extra_database["n_examples"],
                             database_path, remark=f"\n\tRemark: current test_time_n_examples num= {n_len}")

        if len(database) >= self.local_top_k:
            # Define keys explicitly from stats
            keys = sorted(stats.keys())  # sorted ensures consistent ordering

            # Extract stats values according to defined keys
            sv = np.array([stats[k]["value"] for k in keys])  # shape: (stats_num,)

            # Extract database values according to the same keys
            dv = np.array([
                [e["stats"][k]["value"] for k in keys] for e in database
            ])  # shape: (N, stats_num)

            # Compute chi-square distances
            distances = np.array([clark_distance(sv, d) for d in dv])

            # Sort indices from most similar (smallest distance) to least
            ids = np.argsort(distances)

            return [database[i]for i in ids[:self.local_top_k]]

        elif self.train_examples is not None:
            return copy.deepcopy(self.train_examples)
        else:
            return None

    def remove_database(self, database_path):
        if os.path.exists(database_path):
            try:
                # remove file
                os.remove(database_path)
                print(f"Remove: {database_path}")
            except:
                # remove directory
                shutil.rmtree(database_path)
                print(f"Remove: {database_path}")

    def get_databaes_idx(self, database=None):
        database = self.database if database is None else database
        if len(database) > 0:
            return [i["metadata"]["id"] for i in database]
        else:
            return []

    def add_entries(self, entries, database_path=None, remark=None):
        etrs_id_verified = []
        database_idx = self.get_databaes_idx()
        for e in entries:
            if e["metadata"]["id"] not in database_idx:
                self.database_ids.append(e["metadata"]["id"])
                etrs_id_verified.append(e)
        if len(etrs_id_verified) > 0:
            database_path = self.database_path if database_path is None else database_path
            if os.path.exists(database_path):
                # read the old
                self.database = json_file(database_path, jsonl=True)
                # update database
                self.database.extend(etrs_id_verified)
                # write to file
                _ = json_file(database_path, obj=self.database, jsonl=True)
                msg = f"Add {len(etrs_id_verified)} examples to the current database!"
                if remark is not None:
                    msg += remark
                print(msg)
            else:
                # update database
                self.database = copy.deepcopy(entries)
                # write to file
                _ = json_file(database_path, obj=self.database, jsonl=True)
                msg = f"Add {len(etrs_id_verified)} examples to the current database!"
                if remark is not None:
                    msg += remark
                print(msg)
        else:
            msg = "No new entries to be added to the current database!"
            if remark is not None:
                msg += remark
            print(msg)
            warnings.warn("msg")

