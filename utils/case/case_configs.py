from utils.output.seed import seed_hash
import torch
def set_case_configs(case_base_config):
    """set the configration for a case study"""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    data_name = case_base_config["data_name"]
    config=case_base_config
    config["device"]=device
    config["batch_size"]=16
    config["seed"]=seed_hash(0, "case_seed")
    config["batch_eval_names"]=["f1_score", "accuracy"]
    config["trial_eval_names"]=["precision", "recall", "f1_score", "accuracy", "matthews_corrcoef", "geometric_mean"]
    config["case_eval_names"]=["precision", "recall", "f1_score", "accuracy", "matthews_corrcoef", "geometric_mean"]
    config["output_dir"]=r"./outputs"

    if data_name in ["KAIST_Motor", "CWRU_Bearing",]:
        config["test_envs"]=(1,2)
        config["train_cls01_smp"]=[20, 5]
        config["adapt_cls01_smp"] = case_base_config["adapt_cls01_smp"]
        config["test_cls01_smp"]= ["All", "All"]

    elif data_name in ["PU_Bearing", "SCA_PulpMill",]:
        config["test_envs"] = (1,)
        config["train_cls01_smp"] = [20, 5]
        config["adapt_cls01_smp"] = case_base_config["adapt_cls01_smp"]
        config["test_cls01_smp"] = ["All", "All"]

    else:
        raise ValueError("Dataset for config not found!")

    return config
