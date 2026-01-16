
from algorithms.alg_configs import set_alg_configs
from utils.case.case_configs import set_case_configs
from utils.case.run import run_single_case


if __name__ == '__main__':

    case_base_config = {
                   "init_dataset_only":False,
                   "adapt_cls01_smp":None,#, # [10, 10]
                   "smile_test_ulb_max":0, # 10

                   "algorithm":"AFTD",
                   "data_name":"SCA_PulpMill", # KAIST_Motor, CWRU_Bearing, PU_Bearing, SCA_PulpMill
                   "data_dir": r"C:\Users\MSI-NB\Desktop\Smile_prj\0-experiments\datasets",
                   "trials":3,
                   }

    case_config = set_case_configs(case_base_config)
    alg_config = set_alg_configs(case_config)
    run_single_case(alg_config, case_config)
