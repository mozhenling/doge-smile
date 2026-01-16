import os
from datetime import datetime
import numpy as np
import torch
from scipy.io import loadmat
from utils.data.preprocess import *
from utils.data.stats import stats_with_threshold
from utils.output.file import json_file
from utils.output.seed import seed_hash

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dataset, with_meta=True, device=None):
      'Initialization'
      self.dataset = dataset
      self.with_meta = with_meta
      self.device = device

  def __len__(self):
      'Denotes the total number of samples'
      return len(self.dataset["data"])

  def __getitem__(self, index):
      data = torch.tensor(self.dataset["data"][index], dtype=torch.float32)
      label = torch.tensor(self.dataset["label"][index], dtype=torch.long)
      # env = torch.tensor(self.dataset["env"][index], dtype=torch.long)
      if self.device is not None:
          data.to(self.device)
          label.to(self.device)
          # env.to(self.device)
      if self.with_meta:
          meta = self.dataset["metadata"][index]
          return data, label, meta
      else:
          return data, label

def data_transform(x):
    # x: 1-d signal, np.array
    # de-trend
    x = detrend(x)
    # pre-whiten
    x = pre_whiten(x)
    # z-score normalization
    x = standardization(x)
    return x

def shuffle_with_idx_limits(data_all, label_all, train_cls01_smp, test_cls01_smp, env_id, test_envs):
    # -- shuffle data and labels for random sampling
    # Generate a random permutation of indices
    indices = np.random.permutation(len(label_all))
    shuffled_data = data_all[indices]
    shuffled_labels = label_all[indices]

    # if Nones, set a higher numbers to use all data available
    n_idx_limit = []
    f_idx_limit = []
    if env_id in test_envs:
        if test_cls01_smp[0] == "All":
            n_idx_limit = [i for i, l in enumerate(shuffled_labels) if l == 0]
        if test_cls01_smp[1] == "All":
            f_idx_limit = [i for i, l in enumerate(shuffled_labels) if l == 1]
    else:
        if train_cls01_smp[0] == "All":
            n_idx_limit = [i for i, l in enumerate(shuffled_labels) if l == 0]
        if train_cls01_smp[1] == "All":
            f_idx_limit = [i for i, l in enumerate(shuffled_labels) if l == 1]

    sample_limit = test_cls01_smp if env_id in test_envs else train_cls01_smp

    n_idx = [i for i, l in enumerate(shuffled_labels) if l == 0]
    f_idx = [i for i, l in enumerate(shuffled_labels) if l == 1]

    if len(n_idx_limit)==0:
        if sample_limit[0] != "All":
            n_idx_limit = n_idx[:sample_limit[0]]
        else:
            n_idx_limit = []

    if len(f_idx_limit)==0:
        if sample_limit[1] != "All":
            f_idx_limit = f_idx[:sample_limit[1]]
        else:
            f_idx_limit = []

    idx_limits = n_idx_limit + f_idx_limit
    return shuffled_data, shuffled_labels, idx_limits

class PU_Bearing():
    ENVIRONMENTS = ['N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04']

    def __init__(self,  test_envs, data_dir, data_name,
                        train_cls01_smp, test_cls01_smp, save_dataset=False, adapt_cls01_smp=None):

        # -- sample points
        self.source = "Paderborn University"
        self.freqHz = 64000
        self.seg_len = 8192
        self.instance_size = 124 # per class
        self.sig_type = 'vibration'
        self.obj_type = "6203 ball bearing"
        self.ref = "https://doi.org/10.36001/phme.2016.v3i1.1577"

        self.class_name_list = ['Normal K006',
                                'Inner&Outer KB27'
                                    ]

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = [ 'N15_M01_F10', 'N15_M07_F04']
        self.env_list = [i for i in range(len(self.environments))]
        self.num_envs = len(self.env_list)

        self.input_shape = [1, self.seg_len] # may be updated later
        self.num_classes = 2
        self.num_domains = len(self.environments)

        self.data_train = {"data": [], "label": [],  "metadata": []}
        self.data_adapt= {"data": [], "label": [],  "metadata": []}
        self.data_test = {"data": [], "label": [],  "metadata": []}
        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            file_path = os.path.join(data_dir, data_name, env_name + '.mat')
            data_dict = loadmat(file_path)
            data_all, label_all = data_dict['data'].reshape(-1, self.seg_len), data_dict['labels'].reshape(-1)

            shuffled_data, shuffled_labels, idx_limits  = shuffle_with_idx_limits(data_all, label_all,
                                                                                  train_cls01_smp, test_cls01_smp,
                                                                                  env_id, test_envs)

            for i, idx in enumerate(idx_limits):
                data, label = shuffled_data[idx], int(shuffled_labels[idx])
                data = data_transform(data)
                self.input_shape[1] = len(data)
                current_datetime = datetime.now().isoformat(timespec='seconds')
                m = {"id": f"PU_Bearing_env{env_id}_cls{label}_@datahash{seed_hash(data)}",
                     "source": self.source,
                     "object": self.obj_type,
                     "sensor": self.sig_type,
                     "time": current_datetime,
                     "freqHz": self.freqHz,
                     "env":env_id,
                     "length": len(data),
                     "ref": self.ref}
                if env_id in test_envs:
                    self.data_test["data"].append(data)
                    self.data_test["label"].append(label)
                    self.data_test["metadata"].append(m)
                else:
                    # if label == 0:
                    self.data_train["data"].append(data)
                    self.data_train["label"].append(label)
                    self.data_train["metadata"].append(m)


        if adapt_cls01_smp is not None:
            indices = np.random.permutation(len(self.data_test["label"]))
            shuffled_test_data = [self.data_test["data"][i] for i in indices]
            shuffled_test_labels = [self.data_test["label"][i] for i in indices]
            shuffled_test_meta = [self.data_test["metadata"][i] for i in indices]
            test_n_idx = [i for i, l in enumerate(shuffled_test_labels) if l == 0]
            test_f_idx = [i for i, l in enumerate(shuffled_test_labels) if l == 1]
            n_idx_limit = test_n_idx[:adapt_cls01_smp[0]]
            f_idx_limit = test_f_idx[:adapt_cls01_smp[1]]
            adapt_dx_limits = n_idx_limit+f_idx_limit
            self.data_adapt["data"]=[shuffled_test_data[i] for i in adapt_dx_limits]
            self.data_adapt["label"] = [shuffled_test_labels[i] for i in adapt_dx_limits]
            self.data_adapt["metadata"] = [shuffled_test_meta[i] for i in adapt_dx_limits]

        if save_dataset:
            current_datetime = datetime.now().isoformat(timespec='seconds')
            json_file(file_path=os.path.join(data_dir, data_name,current_datetime+"used_dataset.json"),
                      obj={"train":self.data_train, "test":self.data_test, "adapt":self.data_adapt})

class CWRU_Bearing():
    ENVIRONMENTS = ['0hp_1797rpm', '2hp_1750rpm', '3hp_1730rpm']

    def __init__(self, test_envs, data_dir, data_name,
                 train_cls01_smp, test_cls01_smp, save_dataset=False, adapt_cls01_smp=None):

        # -- sample points
        self.source = "Case Western Reserve University"
        self.obj_type = "6205-2RS JEM SKF, deep groove ball bearing"
        self.sig_type = "vibration"
        self.freqHz = 12000
        self.seg_len = 2048
        self.instance_size = 50
        self.ref = "https://engineering.case.edu/bearingdatacenter/download-data-file"

        self.class_name_list = ['normal',
                                'outer',
                                'inner']

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = ['0hp_1797rpm', '2hp_1750rpm', '3hp_1730rpm']
        self.env_list = [i for i in range(len(self.environments))]
        self.num_envs = len(self.env_list)

        self.input_shape = [1, self.seg_len]  # may be updated later
        self.num_classes = 2
        self.num_domains = len(self.environments)

        self.data_train = {"data": [], "label": [], "metadata": []}
        self.data_test = {"data": [], "label": [],  "metadata": []}
        self.data_adapt = {"data": [], "label": [], "metadata": []}
        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            file_path = os.path.join(data_dir, data_name, 'CWRU_DE_'+env_name+'_seg.mat')
            data_dict = loadmat(file_path)
            data_all, label_all = data_dict['data'], data_dict['labels'].squeeze()
            label_all = np.array([int(l>0) for l in label_all])

            shuffled_data, shuffled_labels, idx_limits = shuffle_with_idx_limits(data_all, label_all,
                                                                                 train_cls01_smp, test_cls01_smp,
                                                                                 env_id, test_envs)
            for i, idx in enumerate(idx_limits):
                data, label = shuffled_data[idx], int(shuffled_labels[idx])
                data = data_transform(data)
                self.input_shape[1] = len(data)
                current_datetime = datetime.now().isoformat(timespec='seconds')
                m = {"id": f"CWRU_Bearing_env{env_id}_cls{label}_@datahash{seed_hash(data)}",
                     "source": self.source,
                     "object": self.obj_type,
                     "sensor": self.sig_type,
                     "time": current_datetime,
                     "freqHz": self.freqHz,
                     "env": env_id,
                     "length": len(data),
                     "ref": self.ref}
                if env_id in test_envs:
                    self.data_test["data"].append(data)
                    self.data_test["label"].append(label)
                    self.data_test["metadata"].append(m)
                else:
                    # if label == 0:
                    self.data_train["data"].append(data)
                    self.data_train["label"].append(label)
                    self.data_train["metadata"].append(m)

        if adapt_cls01_smp is not None:
            indices = np.random.permutation(len(self.data_test["label"]))
            shuffled_test_data = [self.data_test["data"][i] for i in indices]
            shuffled_test_labels = [self.data_test["label"][i] for i in indices]
            shuffled_test_meta = [self.data_test["metadata"][i] for i in indices]
            test_n_idx = [i for i, l in enumerate(shuffled_test_labels) if l == 0]
            test_f_idx = [i for i, l in enumerate(shuffled_test_labels) if l == 1]
            n_idx_limit = test_n_idx[:adapt_cls01_smp[0]]
            f_idx_limit = test_f_idx[:adapt_cls01_smp[1]]
            adapt_dx_limits = n_idx_limit + f_idx_limit
            self.data_adapt["data"] = [shuffled_test_data[i] for i in adapt_dx_limits]
            self.data_adapt["label"] = [shuffled_test_labels[i] for i in adapt_dx_limits]
            self.data_adapt["metadata"] = [shuffled_test_meta[i] for i in adapt_dx_limits]

        if save_dataset:
            json_file(file_path=os.path.join(data_dir, data_name, "used_dataset.json"),
                      obj={"train": self.data_train, "test": self.data_test})
class KAIST_Motor():
    ENVIRONMENTS = ['0Nm', '2Nm', '4Nm']

    def __init__(self,  test_envs, data_dir, data_name, 
                        train_cls01_smp, test_cls01_smp, save_dataset=False, adapt_cls01_smp=None):

        # -- sample points
        self.source = "Korea Advanced Institute of Science and Technology"
        self.freqHz = 25600
        self.seg_len = 2048
        self.instance_size = 150 # per class
        self.sig_type = 'vibration'
        self.obj_type = "NSK 6205 DDU bearing"
        self.ref = "https://data.mendeley.com/datasets/ztmf3m7h5x/6"

        self.class_name_list = ['Normal',
                                'BPFO_10'
                                    ]

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = ['0Nm', '2Nm', '4Nm']
        self.env_list = [i for i in range(len(self.environments))]
        self.num_envs = len(self.env_list)

        self.input_shape = [1, self.seg_len] # may be updated later
        self.num_classes = 2
        self.num_domains = len(self.environments)

        self.data_train = {"data": [], "label": [],  "metadata": []}
        self.data_test = {"data": [], "label": [],  "metadata": []}
        self.data_adapt = {"data": [], "label": [], "metadata": []}
        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            file_path = os.path.join(data_dir, data_name,'Xvib_'+ env_name + '_data.mat')
            data_dict = loadmat(file_path)
            data_all, label_all = data_dict['env_data'].reshape(-1, self.seg_len), data_dict['env_label'].reshape(-1)

            shuffled_data, shuffled_labels, idx_limits  = shuffle_with_idx_limits(data_all, label_all,
                                                                                  train_cls01_smp, test_cls01_smp,
                                                                                  env_id, test_envs)

            for i, idx in enumerate(idx_limits):
                data, label = shuffled_data[idx], int(shuffled_labels[idx])
                data = data_transform(data)
                self.input_shape[1] = len(data)
                current_datetime = datetime.now().isoformat(timespec='seconds')
                m = {"id": f"KAIST_Motor_env{env_id}_cls{label}_@datahash{seed_hash(data)}",
                     "source": self.source,
                     "object": self.obj_type,
                     "sensor": self.sig_type,
                     "time": current_datetime,
                     "freqHz": self.freqHz,
                     "env":env_id,
                     "length": len(data),
                     "ref": self.ref}
                if env_id in test_envs:
                    self.data_test["data"].append(data)
                    self.data_test["label"].append(label)
                    self.data_test["metadata"].append(m)
                else:
                    # if label == 0:
                    self.data_train["data"].append(data)
                    self.data_train["label"].append(label)
                    self.data_train["metadata"].append(m)

        if adapt_cls01_smp is not None:
            indices = np.random.permutation(len(self.data_test["label"]))
            shuffled_test_data = [self.data_test["data"][i] for i in indices]
            shuffled_test_labels = [self.data_test["label"][i] for i in indices]
            shuffled_test_meta = [self.data_test["metadata"][i] for i in indices]
            test_n_idx = [i for i, l in enumerate(shuffled_test_labels) if l == 0]
            test_f_idx = [i for i, l in enumerate(shuffled_test_labels) if l == 1]
            n_idx_limit = test_n_idx[:adapt_cls01_smp[0]]
            f_idx_limit = test_f_idx[:adapt_cls01_smp[1]]
            adapt_dx_limits = n_idx_limit + f_idx_limit
            self.data_adapt["data"] = [shuffled_test_data[i] for i in adapt_dx_limits]
            self.data_adapt["label"] = [shuffled_test_labels[i] for i in adapt_dx_limits]
            self.data_adapt["metadata"] = [shuffled_test_meta[i] for i in adapt_dx_limits]

        if save_dataset:
            current_datetime = datetime.now().isoformat(timespec='seconds')
            json_file(file_path=os.path.join(data_dir, data_name,current_datetime+"used_dataset.json"),
                      obj={"train":self.data_train, "test":self.data_test})

class SCA_PulpMill():
    def __init__(self, test_envs, data_dir, data_name,
                 train_cls01_smp, test_cls01_smp, save_dataset=False, adapt_cls01_smp=None):

        # -- sample points
        self.source = "Svenska Cellulosa Aktiebolaget"
        self.freqHz = [8192, 12800]
        self.seg_len = 1024
        self.instance_size = "N.A."
        self.sig_type = 'vibration'
        self.obj_type = ["SKF NU328 E bearing", "SKF 7312 BEAP bearing"]
        self.bearing_id = ["4", "5"]
        self.ref = "https://doi.org/10.3390/data8070115"

        self.class_name_list = ['Normal',
                                'Faulty' #"4": Inner fault, "5":Ball fault
                                ]

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = self.bearing_id
        self.env_list = [i for i in range(len(self.environments))]
        self.num_envs = len(self.env_list)

        self.input_shape = [1, self.seg_len]  # may be updated later
        self.num_classes = 2
        self.num_domains = len(self.environments)

        self.data_train = {"data": [], "label": [],  "metadata": []}
        self.data_test = {"data": [], "label": [], "metadata": []}
        self.data_adapt = {"data": [], "label": [], "metadata": []}

        for env_id, env_name in enumerate(self.environments):
            file_path = os.path.join(data_dir, data_name, "bearing_" + env_name + "_" + 'test.mat')
            data_dict = loadmat(file_path)
            data_all, label_all = data_dict['test_data'].reshape(-1, self.seg_len), data_dict['test_label'].reshape(-1)

            label_all_binary = np.array([int(label > 0) for label in label_all])
            shuffled_data, shuffled_labels, idx_limits = shuffle_with_idx_limits(data_all, label_all_binary,
                                                                                 train_cls01_smp, test_cls01_smp,
                                                                                 env_id, test_envs)

            for i, idx in enumerate(idx_limits):
                data, label = shuffled_data[idx], int(shuffled_labels[idx])
                data = data_transform(data)
                self.input_shape[1] = len(data)
                current_datetime = datetime.now().isoformat(timespec='seconds')
                m = {"id": f"SCA_PulpMil_env{env_id}_cls{label}_@datahash{seed_hash(data)}",
                     "source": self.source,
                     "object": self.obj_type[env_id],
                     "sensor": self.sig_type,
                     "time": current_datetime,
                     "freqHz": self.freqHz[env_id],
                     "env": env_id,
                     "length": len(data),
                     "ref": self.ref}
                if env_id in test_envs:
                    self.data_test["data"].append(data)
                    self.data_test["label"].append(label)
                    self.data_test["metadata"].append(m)
                else:
                    # if label == 0:
                    self.data_train["data"].append(data)
                    self.data_train["label"].append(label)
                    self.data_train["metadata"].append(m)

        if adapt_cls01_smp is not None:
            indices = np.random.permutation(len(self.data_test["label"]))
            shuffled_test_data = [self.data_test["data"][i] for i in indices]
            shuffled_test_labels = [self.data_test["label"][i] for i in indices]
            shuffled_test_meta = [self.data_test["metadata"][i] for i in indices]
            test_n_idx = [i for i, l in enumerate(shuffled_test_labels) if l == 0]
            test_f_idx = [i for i, l in enumerate(shuffled_test_labels) if l == 1]
            n_idx_limit = test_n_idx[:adapt_cls01_smp[0]]
            f_idx_limit = test_f_idx[:adapt_cls01_smp[1]]
            adapt_dx_limits = n_idx_limit + f_idx_limit
            self.data_adapt["data"] = [shuffled_test_data[i] for i in adapt_dx_limits]
            self.data_adapt["label"] = [shuffled_test_labels[i] for i in adapt_dx_limits]
            self.data_adapt["metadata"] = [shuffled_test_meta[i] for i in adapt_dx_limits]

        if save_dataset:
            current_datetime = datetime.now().isoformat(timespec='seconds')
            json_file(file_path=os.path.join(data_dir, data_name, current_datetime + "used_dataset.json"),
                      obj={"train": self.data_train, "test": self.data_test})


if __name__=="__main__":
    from utils.output.seed import seed_everything

    seed_everything(0, remark="diag_data")
    test_envs = (1, 2)
    train_cls01_smp = [25, 0]
    test_cls01_smp = ["All", "All"]
    data_dir = r"C:\Users\MSI-NB\Desktop\Smart_prj\0-exp\datasets"
    data_name = "SCA_PulpMill"
    dataset = KAIST_Motor(test_envs, data_dir, data_name,
                 train_cls01_smp, test_cls01_smp)

    #-- CWRU_Bearing
    # train_cls01_smp = [20, 5]
    # data_dir = r"C:\Users\MSI-NB\Desktop\Smart_prj\0-exp\datasets"
    # data_name = "Database_CWRU_Bearing"
    # save_path = f"../../algorithms/temp/databases/smart_Database_CWRU_Bearing01_{train_cls01_smp}.json"
    # dataset_CWRU = Database_CWRU_Bearing(data_dir, data_name, train_cls01_smp, save_path)

    # #-- Simulated Gaussian
    # save_path = f"../../algorithms/temp/databases/smart_Database_Simulated_Gaussian0_{train_cls01_smp}.json"
    # dataset_Gaussian = Database_Simulated_Gaussian(train_cls01_smp=train_cls01_smp, save_path=save_path)

    