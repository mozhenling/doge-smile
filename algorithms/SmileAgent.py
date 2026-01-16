import os
import copy
import random
import json
import shutil
import warnings

from datetime import datetime
from algorithms.SmileBase import SmileBase
from algorithms.Smile import Smile
from torch.utils.data import DataLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.vectorstores import InMemoryVectorStore

from algorithms.BaseChain import BaseChain
from utils.data.datasets import ListDataset
from utils.output.file import json_file
from utils.output.seed import seed_hash
from utils.data.distances import clark_distance_for_check
from utils.output.parser import generation_pydantic_parser


class SmartAgent(SmileBase):
    def __init__(self, config, train_examples, adapt_examples=None):
        self.config=config
        # -- Database management: for database selection
        self.database_path = self.config[
            "database_path"] if "database_path" in self.config else r"./datasets/database/smile_default_database.jsonl"
        self.database = []
        self.database_ids = []

        if self.config["start_new"]:
            self.remove_database(self.database_path)

        # -------------------- Initialize the database
        if os.path.exists(self.database_path):
            # by the existing one
            self.database = json_file(self.database_path, jsonl=True)
            self.database_ids = self.get_databaes_idx()
            print(f"The current database has {len(self.database)} samples.")
            # # concat multiple database
            # if "database_concat_paths" in config:
            #     self.concat_databases(config["database_paths"], add_to_current=True)

            if train_examples is not None:
                self.add_entries(entries=train_examples)
                print(f"Examples (len={len(train_examples)}) have been added to the current database at:\n {self.database_path}")

        # elif "database_concat_paths" in config:
        #     self.concat_databases(config["database_paths"], add_to_current=True)

        # -- Add train_examples to the database if train_examples is not None
        elif train_examples is not None:
            print(f"The current database is empty. Try to add {len(train_examples)} examples to it.")
            self.add_entries(entries=train_examples)
            print(f"Examples (len={len(train_examples)}) have been added to the current database at:\n {self.database_path}")
        # if previous initializations fail, simulate the database by standard Gaussian
        else:
            warnings.warn(f"{self.database_path} not exits, no examples to be added, the current database is empty!")

        if not self.config["init_dataset_only"]:
            # -------------------- Initialize the LLM
            dir_system = "./messages/smileagent_system.txt"
            dir_human = "./messages/smileagent_human.txt"
            dir_ai = "./messages/smileagent_ai.txt"
            super(SmartAgent, self).__init__(config, dir_system, dir_human, dir_ai, agent_part=True)

            # -------------------- Balance and augment the current database
            if "blc_augs" in self.config and adapt_examples is not None:
                if self.config["blc_augs"] is not None:
                    assert self.config["blc_augs"] >= 1 , "blc_augs should be equal to or larger than 1"
                    # inplace-add augmented samples to the current dataset
                    self.data_balance_augmentation(adapt_examples)

            # -------------------- Initialize the meta-database and the vector store for database selection
            self.meta_db_paths = self.config["meta_db_paths"] if "meta_db_paths" in self.config else []
            if len(self.meta_db_paths) > 0:
                # Add the current database to meta-database for selection
                if len(self.database) > 0 and self.config["database_path"] not in self.meta_db_paths:
                    self.meta_db_paths.append(self.config["database_path"])
                self.meta_db_preview = [json_file(p, jsonl=True, readnum=1)[0] for p in self.meta_db_paths]

                # use the metadata of an instance of the database as the page content for selecting a database from the metadatabase
                self.meta_db_preview_docs = [Document(page_content=json.dumps(d["metadata"]), metadata={"source": p}, )
                                             for d, p in zip(self.meta_db_preview, self.meta_db_paths)]

                embeddings = OllamaEmbeddings(model=config["model_params"]["model"])
                # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                # embeddings = OllamaEmbeddings(model="bge-m3")
                # embeddings = OpenAIEmbeddings()
                self.meta_db_preview_vector_store = FAISS.from_documents(self.meta_db_preview_docs, embeddings)

                self.meta_db_preview_retriever = self.meta_db_preview_vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": config["metadatabase_top_k"]},
                )
            else:
                self.meta_db_preview_retriever = None

            self.smile = Smile(config, self.database, self.meta_db_preview_retriever, agent_part=True)

            self.tool_call_records = {"tool_ai": [], "tool_default": []}
            self.toolkit_ai = {"smile": self.smile}
            self.toolkit_all = {**self.toolkit_ai}
            self.tools = list(self.toolkit_all.values())
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            self.chain_with_tools = self.prompt_chat | self.llm_with_tools

    def data_balance_augmentation(self, dataset, few_shot_group_limit=10):
        """
        First balance the dataset, and augment it to config["blc_augs"] times of the number of the  largest class
        """

        def _check_distance_error(i,j,k,label, sample, min_representatives,
                                  maj_representatives, distance_inter_group, intra_c=0.5, inter_c=1):
            """
            Check for data generation quality based on chi-square distance.

            :param label: label of generated sample
            :param sample: generated sample dictionary
            :param min_representatives: list of few-shot dicts for minority class
            :param maj_representatives: list of few-shot dicts for majority class
            :param distance_inter_group: precomputed average chi-square distance between group representatives
            :return: error message or None
            """
            print("Check up distance errors...")
            min_label = min_representatives[0]["label"]
            random.seed(seed_hash(i,j,k, "select a valid example"))
            if label == min_label:
                num_rpst = len(min_representatives)
                data_eg_distance = clark_distance_for_check(sample, min_representatives)
                distance_intra_group = clark_distance_for_check(min_representatives[:num_rpst // 2],
                                                                   min_representatives[num_rpst // 2:])
                
                smp_eg=random.sample(min_representatives, k=1)
            else:
                num_rpst = len(maj_representatives)
                data_eg_distance = clark_distance_for_check(sample, maj_representatives)
                distance_intra_group = clark_distance_for_check(maj_representatives[:num_rpst // 2],
                                                                   maj_representatives[num_rpst // 2:])
                smp_eg=random.sample(maj_representatives, k=1)

            # intra_group distance should be < inter_group distance
            if distance_intra_group > distance_inter_group:
                warnings.warn(
                    f"Dataset Distance Warning: distance_intra_group={distance_intra_group:.4f} > distance_inter_group={distance_inter_group:.4f}!")
                return None
            else:
                # # new distance should be larger than intra-group distance (diversity)
                # if data_eg_distance < intra_c*distance_intra_group:
                #     print(f"Intra-Group Distance Error Found!\n\tdata-eg-dist={data_eg_distance:.4f}\n\tintra-eg-eg-dist={intra_c*distance_intra_group:.4f}")
                #     distance_error = f"""
                #     [Intra-Group Distance Error] The generated data is too close to the train_examples of its group. Please increase generation diversity.
                #     Intra-Group Data-Example Clark Distance: {data_eg_distance:.4f} should be **larger** than Intra-Group Example-Example Clark Distance: {intra_c*distance_intra_group:.4f}.
                #     An example of valid generated data is : <json>{json.dumps(smp_eg)}</json>.\n"""
                # new distance should be smaller than inter-group distance (class consistency)
                if data_eg_distance > inter_c*distance_inter_group:
                    print(f"Inter-Group Distance Error Found!\n\tdata-eg-dist={data_eg_distance:.4f}\n\tinter-eg-eg-dist={inter_c*distance_inter_group:.4f}")
                    distance_error = f"""
                    [Inter-Group Distance Error] The generated data is too far from the train_examples of its group. It may resemble other classes. Please decrease generation diversity.
                    Intra-Group Data-Example Clark Distance: {data_eg_distance:.4f} should be **smaller** than Inter-Group Example-Example Clark Distance: {inter_c*distance_inter_group:.4f}.
                    An example of valid generated data is : <json>{json.dumps(smp_eg)}</json>.\n"""
                else:
                    print("No Distance Error Found!")
                    distance_error = None

                return distance_error

        # --------------
        normal_org_smps = [d for i, d in enumerate(dataset) if d["label"] == 0]
        faulty_org_smps = [d for i, d in enumerate(dataset) if d["label"] == 1]
        # Create ID by hashing the stats dictionary
        existing_stats_ids = [str(seed_hash(json.dumps(d["stats"], sort_keys=True))) for d in dataset]
        aug_valid_data = []

        assert len(normal_org_smps) > 1, "At least, 2 normal samples should be provided!"
        assert len(faulty_org_smps) > 1, "At least, 2 faulty samples should be provided!"

        _max_num = max(len(normal_org_smps), len(faulty_org_smps))
        max_num = int(_max_num * self.config["blc_augs"])

        n_gap_nums = max_num - len(normal_org_smps)
        f_gap_nums = max_num - len(faulty_org_smps)

        assert n_gap_nums >= 0 and f_gap_nums >= 0, "blc_augs should be equal to or larger than 1"

        if n_gap_nums == 0:
            print("Normal samples are enough!")
        if f_gap_nums == 0:
            print("Faulty samples are enough!")

        gap_labs = [0 for _ in range(n_gap_nums)] + [1 for _ in range(f_gap_nums)]

        gap_datasets = ListDataset(gap_labs)
        gap_loader = DataLoader(gap_datasets, batch_size=self.config["aug_batch_size"], shuffle=True)

        aug_dir_system = "./messages/smile_b_aug_system.txt"
        aug_dir_human = "./messages/smile_b_aug_human.txt"
        aug_dir_ai = "./messages/smile_b_aug_ai.txt"

        if n_gap_nums < f_gap_nums:
            majority_group, minority_group = normal_org_smps, faulty_org_smps
        else:
            majority_group, minority_group = faulty_org_smps, normal_org_smps

        metadata_base = copy.deepcopy(majority_group[0]["metadata"])

        # Limit the context length
        few_shot_group_num = len(minority_group) if len(minority_group) < few_shot_group_limit else few_shot_group_limit

        aug_config = copy.deepcopy(self.config)
        # Set model temperature to augmentation temperature for data generation
        aug_config["model_params"]["temperature"] = aug_config["aug_temperature"]
        format_instructions = generation_pydantic_parser.get_format_instructions()

        n_count = len(normal_org_smps)
        f_count = len(faulty_org_smps)

        batch_len = len(gap_loader)
        for i, _lab_batch in enumerate(gap_loader):
            print(
                f"Batch progress: batch({(i + 1)}/{batch_len}), normal({n_count}/{max_num}), faulty({f_count}/{max_num}).")

            lab_batch = _lab_batch.cpu().numpy()
            info_batch = [{"info": f"""The {k + 1}-th label at the {i + 1}-th batch is {lab}."""} for k, lab in
                          enumerate(lab_batch)]
            valid_data_batch = [{"stats": None, "label": lab, "metadata": copy.deepcopy(metadata_base)} for lab in
                                lab_batch]

            # Varying samples for more diversity
            # balanced few-shots: sample without replacement
            random.seed(seed_hash(i, "selection from the minority"))
            min_representatives = random.sample(minority_group, k=few_shot_group_num)
            # few_shots_raw = random.sample(minority_group, k=few_shot_group_num) + random.sample(majority_group, k=few_shot_group_num)

            random.seed(seed_hash(i, "selection from the majority"))
            maj_representatives = random.sample(majority_group, k=few_shot_group_num)
            # inter-group distance
            distance_inter_group = clark_distance_for_check(min_representatives, maj_representatives)

            few_shots_raw = min_representatives + maj_representatives
            # random.seed(seed_hash(i, "shuffle selections"))
            # random.shuffle(few_shots_raw) # in-place shuffling

            few_shots = [{"info": f"""The {k + 1}-th example label at the {i + 1}-th batch is {d["label"]}.""",
                          "output": f"""<json> {json.dumps(d["stats"])} </json>"""} for k, d in
                         enumerate(few_shots_raw)]

            # Varying seeds for more diversity
            aug_config["model_params"]["seed"] = seed_hash(i, "llm augmentation")
            # Instantiate a LLM chain
            aug_chain = BaseChain(aug_config, aug_dir_system, aug_dir_human,
                                  aug_dir_ai, train_examples=few_shots, format_instructions=format_instructions)

            for j in range(aug_config["aug_max_run"]):
                # Check incompletes
                info_comb_batch_incomplete = [(k, info, lab) for k, (info, lab, valid)
                                              in enumerate(zip(info_batch, lab_batch, valid_data_batch)) if
                                              valid["stats"] is None]

                if j == aug_config["aug_max_run"] - 1:
                    raise RuntimeError("Run out of aug_max_run but still unable to generate enough valid data!")

                if len(info_comb_batch_incomplete) > 0:
                    info_batch_incomplete = [info for _, info, _ in info_comb_batch_incomplete]
                    print(
                        f"Retry Progress: The {j + 1}-th try to generate data for the {i + 1}-th batch (batch_size={len(_lab_batch)}).")
                    rsp_content_list = aug_chain.llm_batch(info_batch_incomplete, parser="json_tag_parser",
                                                           error_return=True)
                    # Check for duplicates by stats
                    for rsp, (k, info, lab) in zip(rsp_content_list, info_comb_batch_incomplete):
                        rsp_dict = rsp[0]
                        smp_id = str(seed_hash(json.dumps(rsp_dict, sort_keys=True)))

                        invalid_smp_init_check = smp_id in existing_stats_ids or "parse_error" in rsp_dict
                        if not invalid_smp_init_check:
                            distance_error = _check_distance_error(i,j,k, lab, rsp_dict, min_representatives,
                                                                     maj_representatives, distance_inter_group)
                        else:
                            distance_error = None

                        error_info = ""
                        if invalid_smp_init_check or distance_error is not None:
                            if distance_error is not None:
                                error_info += distance_error

                            if smp_id in existing_stats_ids:
                                print(
                                    f"Duplicated data (id={smp_id}) for the {k + 1}-th label at the {i + 1}-th batch (label={lab}).")
                                error_info += f"""[Duplication Error] The following data was already generated at the {j + 1}-th try:
                                \n{json.dumps(rsp_dict)}\n"""

                            if "parse_error" in rsp_dict:
                                print(
                                    f"""Encounter a parse error for the {k + 1}-th label at the {i + 1}-th batch (label={lab}).\n{json.dumps(rsp_dict["parse_error"])}\n""")
                                error_info += json.dumps(rsp_dict["parse_error"])

                            # Define new info for generation with a feedback.
                            info_batch[k]["info"] = f"""The {k + 1}-th label at the {i + 1}-th batch is {lab}. 
                            Now, it is the {j + 2}-th try to generate the data for this label.
                            Please generate a new valid data considering the errors in your {j + 1}-th try:\n {error_info}"""
                        else:
                            existing_stats_ids.append(smp_id)
                            valid_data_batch[k]["stats"] = rsp_dict
                            valid_data_batch[k]["metadata"]["source"] = metadata_base["source"] + " (LLM-generated)"
                            valid_data_batch[k]["metadata"]["id"] = f"{metadata_base['id']}_gen{smp_id}"
                            valid_data_batch[k]["metadata"]["time"] = datetime.now().isoformat(timespec='seconds')
                            if lab == 0:
                                n_count += 1
                            else:
                                f_count += 1
                else:
                    aug_valid_data.extend(valid_data_batch)
                    break
        print(f"Batch progress: normal({n_count}/{max_num}), faulty({f_count}/{max_num}).")

        # add to the current database (in-place operation)
        self.add_entries(aug_valid_data)
        # dataset.extend(aug_valid_data)
        # json_file(file_path=self.database_path, obj=aug_valid_data, jsonl=True)

        if "aug_backup_path" in self.config:
            if self.config["aug_backup_path"] is not None:
                time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                file_path = self.config["aug_backup_path"] + "_" + time_now + ".jsonl"
                print(f"Backup balance-augmented dataset of length {len(self.database)} to:\n {file_path}")
                json_file(file_path=file_path, obj=self.database, jsonl=True)

    def general_assistant(self, query):
        # General Q and A
        return self.llm.invoke(query)

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

    def concat_databases(self, database_paths, add_to_current=False):
        database = []
        for p in database_paths:
            database.extend(json_file(p, jsonl=True))
        if add_to_current:
            self.add_entries(database)
        else:
            return database

    def get_databaes_idx(self, database=None):
        database = self.database if database is None else database
        if len(database) > 0:
            return [i["metadata"]["id"] for i in database]
        else:
            return []

    def add_entries(self, entries, database_path=None):
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
            else:
                # update database
                self.database = copy.deepcopy(entries)
                # write to file
                _ = json_file(database_path, obj=self.database, jsonl=True)
        else:
            warnings.warn("New entries not found and to be added to the database based on the id-check!")

    def update_database(self, exact_filters=None, range_filters=None, remove="matching_both"):
        """
        Update the database by removing entries based on exact_filters and range_filters.

        Parameters:
            exact_filters (dict): Metadata keys and exact values to match. None: matched!
            range_filters (dict): Metadata keys and (min, max) tuple ranges. None: matched!
            remove (str): Which entries to remove:
                          "matching_both", "matching_one", or "matching_none".
        """
        match_none_db, match_both_db, match_one_db = self.filter_entries(
            self.database, exact_filters, range_filters
        )

        if remove == "matching_both":
            self.database = match_none_db + match_one_db
        elif remove == "matching_none":
            self.database = match_both_db + match_one_db
        elif remove == "matching_one":
            self.database = match_none_db + match_both_db
        else:
            raise ValueError("remove should be one of: 'matching_both', 'matching_one', or 'matching_none'")

    def filter_entries(self, database=None, exact_filters=None, range_filters=None):
        """
        Filtering entries from the database based on exact matches and range-based metadata criteria.

        Parameters:
            database (list of dict): Your database, each dict containing a 'metadata' key.
            exact_filters (dict or None): Metadata keys and exact values to match.
                                          If None or empty, all entries match automatically.
            range_filters (dict or None): Metadata keys and (min, max) tuple ranges.
                                          Use None for open-ended ranges.
                                          If None or empty, all entries match automatically.

        Returns:
            match_none_db: entries matching neither exact nor range conditions.
            match_both_db: entries matching both exact and range conditions.
            match_one_db:  entries matching exactly one of the conditions.
        """
        exact_filters = exact_filters or {}
        range_filters = range_filters or {}

        match_none_db = []
        match_both_db = []
        match_one_db = []

        database = copy.deepcopy(self.database) if database is None else database

        for entry in database:
            metadata = entry.get('metadata', {})

            # Check exact match filters
            if not exact_filters:  # No exact filter provided, auto-match
                exact_match = True
            else:
                exact_match = all(metadata.get(k) == v for k, v in exact_filters.items())

            # Check range filters
            if not range_filters:  # No range filter provided, auto-match
                range_match = True
            else:
                range_match = True
                for k, (min_val, max_val) in range_filters.items():
                    val = metadata.get(k)
                    if val is None:
                        range_match = False
                        break

                    if k == 'time':
                        try:
                            val_dt = datetime.fromisoformat(val)
                            min_dt = datetime.fromisoformat(min_val) if min_val else None
                            max_dt = datetime.fromisoformat(max_val) if max_val else None
                            if (min_dt and val_dt < min_dt) or (max_dt and val_dt > max_dt):
                                range_match = False
                                break
                        except ValueError:
                            range_match = False
                            break
                    else:
                        try:
                            val_num = float(val)
                            if (min_val is not None and val_num < min_val) or (
                                    max_val is not None and val_num > max_val):
                                range_match = False
                                break
                        except (ValueError, TypeError):
                            range_match = False
                            break

            # Categorize entry clearly
            if exact_match and range_match:
                match_both_db.append(entry)
            elif exact_match or range_match:
                match_one_db.append(entry)
            else:
                match_none_db.append(entry)

        return match_none_db, match_both_db, match_one_db


