import numpy as np
from os import path as osp
import numpy as np
import scipy.sparse as sp
import pyscipopt as scip
import time, random, torch, os, git, gzip, pickle

from torch_scatter import scatter_max as scatter_max_raw, scatter_min as scatter_min_raw

import utils.logger as logger
import settings.consts as consts



def scatter_min(src, index):
    return scatter_min_raw(src, index)[0]

def scatter_max(src, index):
    return scatter_max_raw(src, index)[0]

def normalize_features(features):
    features -= features.min(axis=0, keepdims=True)
    max_val = features.max(axis=0, keepdims=True)
    max_val[max_val == 0] = 1
    features /= max_val
    return features

def init_scip_params(model, seed, heuristics=True, presolving=True, separating=True, conflict=True):

    seed = seed % 2147483648  # SCIP seed range

    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    # separation only at root node
    model.setIntParam('separating/maxrounds', 0)

    # no restart
    model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable presolving
    if not presolving:
        model.setIntParam('presolving/maxrounds', 0)
        model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable separating (cuts)
    if not separating:
        model.setIntParam('separating/maxroundsroot', 0)

    # if asked, disable conflict analysis (more cuts)
    if not conflict:
        model.setBoolParam('conflict/enable', False)

    # if asked, disable primal heuristics
    if not heuristics:
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)


def extract_state(model, buffer=None):
    """
    Compute a bipartite graph representation of the solver. In this
    representation, the variables and constraints of the MILP are the
    left- and right-hand side nodes, and an edge links two nodes iff the
    variable is involved in the constraint. Both the nodes and edges carry
    features.

    Parameters
    ----------
    model : pyscipopt.scip.Model
        The current model.
    buffer : dict
        A buffer to avoid re-extracting redundant information from the solver
        each time.
    Returns
    -------
    variable_features : dictionary of type {'names': list, 'values': np.ndarray}
        The features associated with the variable nodes in the bipartite graph.
    edge_features : dictionary of type ('names': list, 'indices': np.ndarray, 'values': np.ndarray}
        The features associated with the edges in the bipartite graph.
        This is given as a sparse matrix in COO format.
    constraint_features : dictionary of type {'names': list, 'values': np.ndarray}
        The features associated with the constraint nodes in the bipartite graph.
    """
    if buffer is None or model.getNNodes() == 1:
        buffer = {}

    # update state from buffer if any
    s = model.getState(buffer['scip_state'] if 'scip_state' in buffer else None)

    if 'state' in buffer:
        obj_norm = buffer['state']['obj_norm']
    else:
        obj_norm = np.linalg.norm(s['col']['coefs'])
        obj_norm = 1 if obj_norm <= 0 else obj_norm

    row_norms = s['row']['norms']
    row_norms[row_norms == 0] = 1

    # Column features
    n_cols = len(s['col']['types'])

    if 'state' in buffer:
        col_feats = buffer['state']['col_feats']
    else:
        col_feats = {}
        col_feats['type'] = np.zeros((n_cols, 4))  # BINARY INTEGER IMPLINT CONTINUOUS
        col_feats['type'][np.arange(n_cols), s['col']['types']] = 1
        col_feats['coef_normalized'] = s['col']['coefs'].reshape(-1, 1) / obj_norm

    col_feats['has_lb'] = ~np.isnan(s['col']['lbs']).reshape(-1, 1)
    col_feats['has_ub'] = ~np.isnan(s['col']['ubs']).reshape(-1, 1)
    col_feats['sol_is_at_lb'] = s['col']['sol_is_at_lb'].reshape(-1, 1)
    col_feats['sol_is_at_ub'] = s['col']['sol_is_at_ub'].reshape(-1, 1)
    col_feats['sol_frac'] = s['col']['solfracs'].reshape(-1, 1)
    col_feats['sol_frac'][s['col']['types'] == 3] = 0  # continuous have no fractionality
    col_feats['basis_status'] = np.zeros((n_cols, 4))  # LOWER BASIC UPPER ZERO
    col_feats['basis_status'][np.arange(n_cols), s['col']['basestats']] = 1
    col_feats['reduced_cost'] = s['col']['redcosts'].reshape(-1, 1) / obj_norm
    col_feats['age'] = s['col']['ages'].reshape(-1, 1) / (s['stats']['nlps'] + 5)
    col_feats['sol_val'] = s['col']['solvals'].reshape(-1, 1)
    col_feats['inc_val'] = s['col']['incvals'].reshape(-1, 1)
    col_feats['avg_inc_val'] = s['col']['avgincvals'].reshape(-1, 1)

    col_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in col_feats.items()]
    col_feat_names = [n for names in col_feat_names for n in names]
    col_feat_vals = np.concatenate(list(col_feats.values()), axis=-1)

    variable_features = {
        'names': col_feat_names,
        'values': col_feat_vals,}

    # Row features

    if 'state' in buffer:
        row_feats = buffer['state']['row_feats']
        has_lhs = buffer['state']['has_lhs']
        has_rhs = buffer['state']['has_rhs']
    else:
        row_feats = {}
        has_lhs = np.nonzero(~np.isnan(s['row']['lhss']))[0]
        has_rhs = np.nonzero(~np.isnan(s['row']['rhss']))[0]
        row_feats['obj_cosine_similarity'] = np.concatenate((
            -s['row']['objcossims'][has_lhs],
            +s['row']['objcossims'][has_rhs])).reshape(-1, 1)
        row_feats['bias'] = np.concatenate((
            -(s['row']['lhss'] / row_norms)[has_lhs],
            +(s['row']['rhss'] / row_norms)[has_rhs])).reshape(-1, 1)

    row_feats['is_tight'] = np.concatenate((
        s['row']['is_at_lhs'][has_lhs],
        s['row']['is_at_rhs'][has_rhs])).reshape(-1, 1)

    row_feats['age'] = np.concatenate((
        s['row']['ages'][has_lhs],
        s['row']['ages'][has_rhs])).reshape(-1, 1) / (s['stats']['nlps'] + 5)

    tmp = s['row']['dualsols'] / (row_norms * obj_norm)
    row_feats['dualsol_val_normalized'] = np.concatenate((
            -tmp[has_lhs],
            +tmp[has_rhs])).reshape(-1, 1)

    row_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in row_feats.items()]
    row_feat_names = [n for names in row_feat_names for n in names]
    row_feat_vals = np.concatenate(list(row_feats.values()), axis=-1)

    constraint_features = {
        'names': row_feat_names,
        'values': row_feat_vals,}

    # Edge features
    if 'state' in buffer:
        edge_row_idxs = buffer['state']['edge_row_idxs']
        edge_col_idxs = buffer['state']['edge_col_idxs']
        edge_feats = buffer['state']['edge_feats']
    else:
        coef_matrix = sp.csr_matrix(
            (s['nzrcoef']['vals'] / row_norms[s['nzrcoef']['rowidxs']],
            (s['nzrcoef']['rowidxs'], s['nzrcoef']['colidxs'])),
            shape=(len(s['row']['nnzrs']), len(s['col']['types'])))
        coef_matrix = sp.vstack((
            -coef_matrix[has_lhs, :],
            coef_matrix[has_rhs, :])).tocoo(copy=False)

        edge_row_idxs, edge_col_idxs = coef_matrix.row, coef_matrix.col
        edge_feats = {}

        edge_feats['coef_normalized'] = coef_matrix.data.reshape(-1, 1)

    edge_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in edge_feats.items()]
    edge_feat_names = [n for names in edge_feat_names for n in names]
    edge_feat_indices = np.vstack([edge_row_idxs, edge_col_idxs])
    edge_feat_vals = np.concatenate(list(edge_feats.values()), axis=-1)

    edge_features = {
        'names': edge_feat_names,
        'indices': edge_feat_indices,
        'values': edge_feat_vals,}

    if 'state' not in buffer:
        buffer['state'] = {
            'obj_norm': obj_norm,
            'col_feats': col_feats,
            'row_feats': row_feats,
            'has_lhs': has_lhs,
            'has_rhs': has_rhs,
            'edge_row_idxs': edge_row_idxs,
            'edge_col_idxs': edge_col_idxs,
            'edge_feats': edge_feats,
        }

    return constraint_features, edge_features, variable_features


def extract_khalil_variable_features(model, candidates, root_buffer):
    """
    Extract features following Khalil et al. (2016) Learning to Branch in Mixed Integer Programming.

    Parameters
    ----------
    model : pyscipopt.scip.Model
        The current model.
    candidates : list of pyscipopt.scip.Variable's
        A list of variables for which to compute the variable features.
    root_buffer : dict
        A buffer to avoid re-extracting redundant root node information (None to deactivate buffering).

    Returns
    -------
    variable_features : 2D np.ndarray
        The features associated with the candidate variables.
    """
    # update state from state_buffer if any
    scip_state = model.getKhalilState(root_buffer, candidates)

    variable_feature_names = sorted(scip_state)
    variable_features = np.stack([scip_state[feature_name] for feature_name in variable_feature_names], axis=1)

    return variable_features

class BranchDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_num, raw_dir_name, processed_suffix):
        super().__init__()
        self.root, self.data_num = root, data_num

        self.raw_dir = osp.join(self.root, raw_dir_name)
        self.processed_dir = self.raw_dir + processed_suffix

        assert osp.exists(self.raw_dir) or osp.exists(self.processed_dir)

        if data_num > 0:
            self.load()
        else:
            self._data_list = []

    def load(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        info_dict_path = osp.join(self.processed_dir, "info_dict.pt")

        if osp.exists(info_dict_path):
            info_dict = torch.load(info_dict_path)
            file_names = info_dict["file_names"]
            processed_files = info_dict["processed_files"]
        else:
            info_dict = {}
            raw_file_names = os.listdir(self.raw_dir)
            random.shuffle(raw_file_names)
            file_names = [osp.join(self.raw_dir, raw_file_name) for raw_file_name in raw_file_names]
            file_names = [x for x in file_names if not osp.isdir(x)]
            processed_files = []
            info_dict.update(processed_files=processed_files, file_names=file_names)

        if self.data_num > len(processed_files):
            for file_name in file_names[len(processed_files):self.data_num]:
                with gzip.open(file_name, 'rb') as f:
                    sample = pickle.load(f)
                processed_file = self.process_sample(sample)
                processed_files.append(processed_file)
            self._data_list = processed_files
            
            torch.save(info_dict, info_dict_path)
        else:
            self._data_list = processed_files[:self.data_num]
    
    def process_sample(self, sample):
        raise NotImplementedError

    @property
    def data(self):
        return self._data_list

    def __len__(self):
        return len(self._data_list)
    
    def __getitem__(self, idx):
        return self._data_list[idx]


def valid_seed(seed):
    assert seed >= 0
    seed = seed % 65536  # 2^16

    np.random.seed(seed)    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return seed


def initial_logdir_and_get_seed(exp_type, instance_type, exp_name, **kwargs):
    log_dir = get_log_dir(exp_type, instance_type, exp_name, **kwargs)
    return initial_logger_and_seed(log_dir)


def get_log_dir(exp_type, instance_type, exp_name, **kwargs):
    try:
        git_hash = git.Repo(search_parent_directories=True).head.object.hexsha[:7]
    except:
        git_hash = "no_git"

    time_str_git_hash = f"{time.strftime('%m%d%H%M%S')}_{git_hash}"
    exp_str = "_".join([f"{k}-{v}" for k,v in kwargs.items()])
    dir_name = f"{time_str_git_hash}_{exp_str}" if exp_str else time_str_git_hash

    log_dir_now = osp.join(consts.RESULT_DIR, exp_type, instance_type, exp_name, dir_name)
    return log_dir_now


def initial_logger_and_seed(log_dir, ith_exp=0, conf=None, original_seed=0):
    if original_seed < 0 or original_seed is None:
        original_seed = np.random.randint(low=0, high=4096)
    seed = original_seed + ith_exp * 1000
    seed = valid_seed(seed)
    log_dir = osp.join(log_dir, f"{ith_exp}_{seed}")

    logger.configure(log_dir)

    if conf:
        conf["seed"] = seed
        logger.write_json("configs", conf)
    return log_dir, seed
