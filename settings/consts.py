from typing import Any
import torch
import os
from os import path as osp
from collections import defaultdict
import hydra
# donot import numpy here as we have to set np threads later


IMPORTANT_INFO_SUFFIX = "*"*10
WARN_INFO_SUFFIX = "!"*10

NODE_FEATURES = ['type_0', 'type_1', 'type_2', 'type_3', 'coef_normalized', 'has_lb', 'has_ub', 'sol_is_at_lb', 'sol_is_at_ub', 'sol_frac', 'basis_status_0', 'basis_status_1', 'basis_status_2', 'basis_status_3', 'reduced_cost', 'age', 'sol_val', 'inc_val', 'avg_inc_val']
CONSTRAINT_FEATURES = ['obj_cosine_similarity', 'bias', 'is_tight', 'age', 'dualsol_val_normalized']
EDGE_FEATURES = ['coef_normalized']
GRAPH_NAMES = CONSTRAINT_FEATURES + EDGE_FEATURES + NODE_FEATURES


DEVICE = torch.device("cpu")
torch.set_default_device(DEVICE)

TRAIN_NAME_DICT = defaultdict(lambda : "")
TRAIN_NAME_DICT.update({"setcover": "500r_1000c_0.05d", "indset": "750_4", "facilities": "100_100_5", "cauctions": "100_500"})



WORK_DIR = osp.dirname(osp.dirname(__file__))
DATA_DIR = osp.join(WORK_DIR, "data")
RESULT_BASE_DIR = osp.join(WORK_DIR, "results")
INSTANCE_DIR = osp.join(DATA_DIR, "instances")
SAMPLE_DIR = osp.join(DATA_DIR, "samples")
RESULT_DIR = osp.join(RESULT_BASE_DIR, "results")



ITER_DICT = defaultdict(lambda : 3000)
GLOBAL_INFO_DICT = {}


SAFE_EPSILON = 1e-6
DETAILED_LOG_FREQ = 100
DETAILED_LOG = True

STATUS_DICT = {"optimal": 0, "timelimit":-1, "infeasible":-2, "unbounded":-3, "userinterrupt":-4, "unknown":-5}
STATUS_INDEX_DICT = {v:k for k,v in STATUS_DICT.items()}