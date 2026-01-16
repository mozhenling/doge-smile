from pathlib import Path
from utils.output.seed import seed_hash
import numpy as np
from datetime import datetime

def set_alg_configs(case_config, default=True):
    """set the configration for an algorithm"""
    algorithm = case_config["algorithm"]
    config={}

    def _add_config(name, default_val, random_val_fn=None):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert (name not in config)
        if random_val_fn is None or default:
            config[name] = default_val
        else:
            random_state = np.random.RandomState(seed_hash(case_config["seed"], name))
            config[name] = random_val_fn(random_state)

    if algorithm in ["Smile","SmileAgent", "SmileAblation", "AFTD"]:
        adapt_cls01_smp = case_config["adapt_cls01_smp"]
        tr_cls01_smp = case_config["train_cls01_smp"]

        config = {
            "algorithm": case_config["algorithm"],
            "smile_test_ulb_max":case_config["smile_test_ulb_max"],
            "start_new": True,
            "local_top_k":5,
            "f_sensitivity":0.5,

            "database_path": f"""./datasets/databases/smart_{case_config["data_name"]}_test-env{case_config["test_envs"]}_tr-cls01-smp{tr_cls01_smp}.jsonl""",
            # "database_concat_paths":[
            #     r"""./algorithms/databases/smile_Database_CWRU_Bearing01_100.jsonl""",
            #     r"""./algorithms/databases/smile_Database_Simulated_Gaussian0_100.jsonl"""
            # ],
            }

        if algorithm in ["SmileAgent"]:
            _add_config("init_dataset_only", case_config["init_dataset_only"]) # Only initialize and save certain datasets

            _add_config("max_runs", 10) # maximum runs to obtain valid detection results by the LLM
            _add_config("force_tool_use", True)
            # # data augmentation/generation
            # _add_config("blc_augs", None) # how many times should we increase
            # if config["blc_augs"] is not None:
            #     _add_config("aug_temperature", 0.8)
            #     _add_config("aug_max_run", 10)
            #     _add_config("aug_batch_size", 4) # small batch is recommended to avoid duplications.
            #     _add_config("aug_backup_path",
            #                 f"""./datasets/backup/smart_{case_config["data_name"]}_test-env{case_config["test_envs"]}_adapt-cls01-smp{adapt_cls01_smp}_blc_aug_times{config['blc_augs']}""")
            # _add_config("try_multiple_gap_num", 3) # try_count limit based on the gap of a group to the target number of augmentation.
            # _add_config("few_shot_org_num", 2) # initial sampling on the original data

            _add_config("metadatabase_top_k", 3)
            _add_config("model_provider", None)
            _add_config("api_key", None)
            _add_config("parser", "detect_json_parser")
            _add_config("with_reason", True)

            _add_config("model_params", {"model": "llama3.2",
                                        "temperature": 0.,
                                        "seed": case_config["seed"]})

            # Define the folder path of metadatabase
            folder_path = Path("./datasets/databases")
            # Get all .jsonl files
            jsonl_files = list(folder_path.glob("*.jsonl"))
            # Convert to string paths
            jsonl_file_paths = [str(file) for file in jsonl_files]
            _add_config("meta_db_paths", jsonl_file_paths)

        if algorithm in ["SmileAblation","AFTD"]:
            _add_config("single_group_classifier", True)
            _add_config("similarity_adjustment", True)
            _add_config("approximation_adjustment", True)
            _add_config("knn", False)  #
            _add_config("stats_single_name", "gk_JB_stats") #

    elif algorithm in ["InContext", "InContextHIs", "InContextRaw"]:
        _add_config("max_runs", 3)
        _add_config("model_provider", None)
        _add_config("api_key", None)
        _add_config("parser", "detect_json_parser")
        _add_config("with_reason", True)
        _add_config("model_params", {"model": "llama3.2",
                                     "temperature": 0.,
                                     "seed": case_config["seed"]})

    elif algorithm in ["iBRF"]:
        _add_config("valid_ratio", None)

    elif algorithm in ["ERM", "ERMpp","VNE", "MixStyle","RIDG", "URM", "AGLU", "RDR",  "ELMloss"]:
        _add_config("train_steps", 500)
        _add_config("weight_decay", 0)
        _add_config("valid_ratio", None)
        _add_config("batch_size", case_config["batch_size"])
        _add_config("model_path", None)
        _add_config("patience", 20)
        _add_config("device", case_config["device"])
        _add_config("lr", 0.01)
        _add_config("check_freq", 10)

        if algorithm =="ERMpp":
            _add_config('lars', False)
            _add_config('freeze_bn', False)
            _add_config('linear_steps', 10)
            _add_config('linear_lr', 5e-5)

        if algorithm == "VNE":
            _add_config('vne_coef', 0.001)

        if algorithm == "RIDG":
            _add_config("momentum", 0.001)
            _add_config("ridg_reg", 0.01)

        if algorithm == "MixStyle":
            _add_config("mixup_alpha", 0.2)

        if algorithm == "RDR":
            _add_config("rdr_momentum", 0.9)
            _add_config("warmup", 10)

        if algorithm == "URM":
            _add_config('urm', 'adversarial')  # 'adversarial'
            _add_config('urm_adv_lambda', 0.1)
            _add_config('urm_discriminator_label_smoothing',0)
            _add_config('urm_discriminator_optimizer', 'adam')
            _add_config('urm_discriminator_hidden_layers', 1)
            _add_config('urm_generator_output', 'relu')
            _add_config('urm_discriminator_lr', 1e-3)

        
    return config
