import os
import sys
import time
import numpy as np
from utils.output.seed import seed_everything
from utils.data import diag_datasets

from torch.utils.data import  DataLoader
from utils.data.formats import train_test_formater
from utils.log import logger
from utils.output.evaluation import binary_eval_metrics
import algorithms

def run_single_case(alg_config, case_config):

    # ---------------------------- Initialization
    job_start_time = time.time()
    algorithm = case_config["algorithm"]
    init_dataset_only = case_config["init_dataset_only"]
    data_name = case_config["data_name"]
    test_envs = case_config["test_envs"]
    train_cls01_smp = case_config["train_cls01_smp"]
    test_cls01_smp = case_config["test_cls01_smp"]
    adapt_cls01_smp = case_config["adapt_cls01_smp"]
    batch_size = case_config["batch_size"]
    data_dir = case_config["data_dir"]
    device = case_config["device"]
    trials = case_config["trials"]
    seed = case_config["seed"]
    batch_eval_names = case_config["batch_eval_names"]
    trial_eval_names = case_config["trial_eval_names"]
    case_eval_names = case_config["case_eval_names"]
    output_dir = case_config["output_dir"]


    # -- run case studies
    case_eval_list = []
    train_num = 0
    test_num = 0
    adapt_num =0
    _train_cls01_smp = train_cls01_smp if train_cls01_smp is not None else ("All", "All")
    _test_cls01_smp = test_cls01_smp if test_cls01_smp is not None else ("All", "All")

    # -- Set a "logger" that sends a standard outputs (print) to a file
    if "model_params" in alg_config:
        llm_name = "_"+alg_config["model_params"]["model"]
    else:
        llm_name = ''

    if adapt_cls01_smp is not None:
        case_folder = data_name+f"_test-env{test_envs}_tr-cls01-smp{train_cls01_smp}_te-cls01-smp{test_cls01_smp}_ad-cls01-smp{adapt_cls01_smp}_"+algorithm+llm_name
    else:
        case_folder = data_name + f"_test-env{test_envs}_tr-cls01-smp{train_cls01_smp}_te-cls01-smp{test_cls01_smp}_" + algorithm+llm_name

    if ":" in case_folder:
        case_folder = case_folder.replace(":", "-") # ensure a valid folder name

    log_file_path = os.path.join(output_dir, case_folder)
    os.makedirs(log_file_path, exist_ok=True)
    sys.stdout = logger.Tee(os.path.join(log_file_path, 'log.txt'))
    sys.stderr = logger.Tee(os.path.join(log_file_path, 'err.txt'))

    # ---------------------------- Case Study
    for i in range(trials):
        trial_seed = seed_everything(seed=i, remark=f"run a trial of the case study with initial seed: {seed}")

        #---- prepare data
        dataset = vars(diag_datasets)[data_name](test_envs, data_dir, data_name,
                                                 train_cls01_smp, test_cls01_smp, adapt_cls01_smp=adapt_cls01_smp)
        # add data info.
        alg_config["input_shape"] = dataset.input_shape
        alg_config["num_classes"] = dataset.num_classes
        alg_config["num_domains"] = dataset.num_domains

        print("#" * 10)
        print(f"folder: {case_folder}")
        print(f"Trial: {i}")
        print(f"case seed: {seed}")
        print(f"trial seed: {trial_seed}")
        print('alg_config:')
        for k, v in sorted(alg_config.items()):
            print('\t{}: {}'.format(k, v))

        data_test = diag_datasets.Dataset(dataset.data_test)
        data_test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

        # -- few-shot train_examples
        train_num = len(dataset.data_train["label"])
        test_num = len(dataset.data_test["label"])
        adapt_num = len(dataset.data_adapt["label"])

        if train_num > 0:
            train_examples = train_test_formater(dataset.data_train, use="train_" + algorithm, device=device)
        else:
            train_examples = None

        if adapt_num > 0:
            adapt_examples = train_test_formater(dataset.data_adapt, use="train_" + algorithm, device=device)
        else:
            adapt_examples = None

        print(f"Train with {train_num} sample(s) and train_cls01_smp: {_train_cls01_smp}")
        if adapt_cls01_smp is not None:
            print(f"Adapt with {adapt_num} sample(s) and adapt_cls01_smp: {adapt_cls01_smp}")
        print(f"Test with {test_num} sample(s) and test_cls01_smp: {_test_cls01_smp}")
        

        trial_label_true = []
        trial_label_pred = []

        batch_checks = {}
        # instantiate a model
        model = vars(algorithms)[algorithm](alg_config, train_examples, adapt_examples=adapt_examples)
        if init_dataset_only:
            break
        # ---------------------------- Training Stage
        if model.train_required:
            model.train_val()

        # ---------------------------- Testing Stage
        for j, (batch_data, batch_label_true, batch_meta) in enumerate(data_test_loader):

            if model.is_llm or algorithm in ["Smile","SmileAblation", "AFTD", "SmileAgent"]:
                # --convert tensor data to strings for the LLM
                if algorithm == "SmileAgent":
                    batch_data = train_test_formater((batch_data, batch_meta), use="test_" + algorithm, device=device)
                else:
                    batch_data = train_test_formater(batch_data, use="test_" + algorithm, device=device)

            batch_label_true = [int(l) for l in batch_label_true.tolist()]
            batch_env = [int(e) for e in  batch_meta["env"]]

            alg_start_time = time.time()
            # ask the LLM to output the prediction
            pred_batch = model.predict(batch_data)
            alg_stop_time = time.time()

            if isinstance(pred_batch[0], tuple):
                # Outputs of LLMs with reasons
                batch_label_pred = [int(l[0]) for l in pred_batch]
                batch_pred_reasons = [l[1] for l in pred_batch]
            else:
                # Outputs of other methods
                batch_label_pred = [int(l) for l in pred_batch]
                batch_pred_reasons = None

            trial_label_true.extend(batch_label_true)
            trial_label_pred.extend(batch_label_pred)

            batch_eval_dict = binary_eval_metrics(batch_label_true, batch_label_pred, batch_eval_names)

            batch_checks["invoke"] = j
            batch_checks["batch_env"] = batch_env
            batch_checks["batch_label_true"] = batch_label_true
            batch_checks["batch_label_pred"] = batch_label_pred
            batch_checks["batch_pred_reasons"] = batch_pred_reasons
            batch_checks.update(batch_eval_dict)
            batch_checks["batch_check_time"] = alg_stop_time - alg_start_time
            print("-" * 10)
            for k, v in batch_checks.items():
                if k in list(batch_eval_dict.keys()) + ["batch_check_time"]:
                    strings = f"\t{k}" + ":\t{:.4f}".format(v)
                    if k == "batch_check_time":
                        strings += ' s'
                else:
                    strings = f"{k}: " + str(v)
                print(strings)
            print("-" * 10)

        trial_eval_dict = binary_eval_metrics(trial_label_true, trial_label_pred, trial_eval_names)
        case_eval_list.append(trial_eval_dict)
        print("*" * 10)
        print("trial: ", i)
        for mn in trial_eval_names:
            ps =  "\ttrial_" + mn + ":\t{:.4f}".format(trial_eval_dict[mn])
            print(ps)
        print("*" * 10)

    job_stop_time = time.time()
    if not init_dataset_only:
        sys.stdout = logger.Tee(os.path.join(log_file_path, 'case_results.txt'))
        print("#" * 10)
        print('alg_config:')
        for k, v in sorted(alg_config.items()):
            print('\t{}: {}'.format(k, v))
        print("#" * 10)
        print(f"case: {log_file_path}")
        print(f"Train with {train_num} sample(s) and train_cls01_smp: {_train_cls01_smp}")
        if adapt_cls01_smp is not None:
            print(f"Adapt with {adapt_num} sample(s) and adapt_cls01_smp: {adapt_cls01_smp}")
        print(f"Test with {test_num} sample(s) and test_cls01_smp: {_test_cls01_smp}")

        for mn in case_eval_names:
            m_list = [d[mn] for d in case_eval_list]
            ps = "\t" + "case_" + mn + ":\t{:.4f}".format(np.mean(m_list)) + " +/- {:.4f}".format(np.std(m_list))
            print(ps)
    print("#" * 10, ' total_time = {:.2f} s '.format((job_stop_time - job_start_time)), "#" * 10)


