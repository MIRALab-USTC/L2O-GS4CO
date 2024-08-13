import os, random, gzip, argparse
import os.path as osp, torch.nn as nn
from torch import as_tensor
from torch_scatter import scatter_sum, scatter_mean
import torch_scatter

import torch, pickle
import numpy as np
import torch_geometric

from utils.utilities import normalize_features, scatter_max, scatter_min, scatter_max_raw

import settings.consts as consts
torch.set_default_device(consts.DEVICE)

class Expression(nn.Module):
    def __init__(self, expression):
        super().__init__()

        assert ";;" in expression
        self.alloc_expression, self.expression = expression.split(";;")

    def forward(self, constraint, variable, cv_edge_index, edge_attr, cand_mask):
        c_edge_index, v_edge_index = cv_edge_index
        exec(self.alloc_expression)
        result = eval(self.expression)
        return result[cand_mask]

class BipartiteNodeData(torch_geometric.data.Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "x_cv_edge_index":
            return torch.tensor(
                [[self.x_constraint.size(0)], [self.x_variable.size(0)]]
            )
        if key == "y_cand_mask":
            return self.x_variable.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def process_from_file_path(file_path, khali=False): # normalize=True, 
    with gzip.open(file_path, 'rb') as f:
        sample = pickle.load(f)
    # (c, e, v), state_khalil, _, sample_cands, cand_scores = sample['data']
    obss = sample["obss"]

    vars_all, cons_feature, edge = obss[0][0], obss[0][1], obss[0][2]
    depth = obss[2]["depth"]

    scores = obss[2]["scores"]
    vars_feature, indices = vars_all[:,:19], vars_all[:,-1].astype(bool)
    indices = np.where(indices)[0]
    scores = scores[indices]
    labels = scores >= scores.max()

    scores = normalize_features(scores)

    other_features = {}
    if khali:
        khali_feature = vars_all[:,19:-1]
        khali_feature = khali_feature[indices]
        khali_feature = normalize_features(khali_feature)
        other_features["x_khali"] = as_tensor(khali_feature, dtype=torch.float, device="cpu")

    data = BipartiteNodeData(x_constraint=as_tensor(cons_feature, dtype=torch.float, device="cpu"), x_variable=as_tensor(vars_feature, dtype=torch.float, device="cpu"),
                             x_cv_edge_index=as_tensor(edge['indices'], dtype=torch.long, device="cpu"), x_edge_attr=as_tensor(edge['values'].squeeze(1), dtype=torch.float, device="cpu"),
                             y_cand_mask=as_tensor(indices, dtype=torch.long, device="cpu"), y_cand_score=as_tensor(scores, dtype=torch.float, device="cpu"), y_cand_label=as_tensor(labels, dtype=torch.bool, device="cpu"),
                             depth=depth,
                             **other_features)
    return data

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_num, raw_dir_name="train", processed_suffix="_processed"):
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

        if self.data_num > len(processed_files):
            for file_name in file_names[len(processed_files):self.data_num]:
                processed_file = process_from_file_path(file_name)
                processed_files.append(processed_file)
            self._data_list = processed_files
            info_dict.update(processed_files=processed_files, file_names=file_names)
            torch.save(info_dict, info_dict_path)
        else:
            self._data_list = processed_files[:self.data_num]
    
    def __len__(self):
        return len(self._data_list)
    
    def __getitem__(self, idx):
        return self._data_list[idx]

def get_all_dataset(instance_type, dataset_prefix, dataset_type=None, number=5000, batch_size=1000, shuffle=False):
    file_dir = osp.join(consts.SAMPLE_DIR, instance_type, consts.TRAIN_NAME_DICT[instance_type] if dataset_type is None else dataset_type)
    dataset = GraphDataset(file_dir, number, raw_dir_name=dataset_prefix)
    loader = torch_geometric.loader.DataLoader(dataset, batch_size, shuffle=shuffle, follow_batch=["y_cand_mask"])

    return loader

def get_batch_score_precision(model, batch):
    data = batch.x_constraint, batch.x_variable, batch.x_cv_edge_index, batch.x_edge_attr, batch.y_cand_mask
    pred_y = model(*data)
    _, where_max = scatter_max_raw(pred_y, batch.y_cand_mask_batch)
    where_max_illegal = where_max==len(batch.y_cand_label)
    where_max[where_max_illegal] = 0
    real_label = batch.y_cand_label[where_max]
    real_label[where_max_illegal] = False
    return real_label

@torch.no_grad()
def get_precision_iteratively(model, data, partial_sample=None, score_func_name="precision"):
    score_func = globals()[f"get_batch_score_{score_func_name}"]

    scores_sum, data_sum = 0, 0
    if partial_sample is None:
        partial_sample = len(data)
    for batch in data:
        batch = batch.to(consts.DEVICE)
        batch_labels = score_func(model, batch)
        scores_sum += batch_labels.sum(dim=-1)
        data_sum += len(batch)
        if data_sum >= partial_sample:
            break
    result = scores_sum / data_sum
    return result

def get_expression(dataset_name):
    with open(osp.join("./expressions", dataset_name), "r") as txt:
        expression = next(txt)
    return expression

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    args = parser.parse_args()
    
    dataloader = get_all_dataset(args.problem, "transfer")
    model = Expression(get_expression(args.problem))

    precision = get_precision_iteratively(model, dataloader)
    print(f"the imitation learning accuracy of {args.problem} is: {precision:.2f}")