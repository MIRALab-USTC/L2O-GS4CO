import os
from os import path as osp
import pickle
import gzip
import torch
from torch import as_tensor
import numpy as np
from time import time as time
import random
from copy import deepcopy

import settings.consts as consts
import utils.logger as logger
import utils.utilities as utilities

import torch
import torch_geometric
from torch_scatter import scatter_mean, scatter_max, scatter_sum

import dso_utils_graph
from utils.rl_algos import PPOAlgo


class BipartiteNodeData(torch_geometric.data.Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "x_cv_edge_index":
            return torch.tensor(
                [[self.x_constraint.size(0)], [self.x_variable.size(0)]]
            )
        if key == "y_cand_mask":
            return self.x_variable.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(utilities.BranchDataset):
    def __init__(self, root, data_num, raw_dir_name="train", processed_suffix="_processed"):
        super().__init__(root, data_num, raw_dir_name, processed_suffix)

    def process_sample(self, sample):
        obss = sample["obss"]

        vars_all, cons_feature, edge = obss[0][0], obss[0][1], obss[0][2]
        depth = obss[2]["depth"]

        scores = obss[2]["scores"]
        vars_feature, indices = vars_all[:,:19], vars_all[:,-1].astype(bool)
        indices = np.where(indices)[0]
        scores = scores[indices]
        labels = scores >= scores.max()

        scores = utilities.normalize_features(scores)

        data = BipartiteNodeData(x_constraint=as_tensor(cons_feature, dtype=torch.float, device="cpu"), x_variable=as_tensor(vars_feature, dtype=torch.float, device="cpu"),
                                x_cv_edge_index=as_tensor(edge['indices'], dtype=torch.long, device="cpu"), x_edge_attr=as_tensor(edge['values'].squeeze(1), dtype=torch.float, device="cpu"),
                                y_cand_mask=as_tensor(indices, dtype=torch.long, device="cpu"), y_cand_score=as_tensor(scores, dtype=torch.float, device="cpu"), y_cand_label=as_tensor(labels, dtype=torch.bool, device="cpu"),
                                depth=depth,)
        return data



def get_all_dataset(instance_type, dataset_type=None, train_num=150000, valid_num=100000, batch_size_train=400, batch_size_valid=400, get_train=True, get_valid=True):
    file_dir = osp.join(consts.SAMPLE_DIR, instance_type, consts.TRAIN_NAME_DICT[instance_type] if dataset_type is None else dataset_type)
    if get_train:
        train_dataset = GraphDataset(file_dir, train_num)
        train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size_train, shuffle=True, follow_batch=["y_cand_mask"], generator=torch.Generator(device=consts.DEVICE))
    else:
        train_loader = None

    if get_valid:
        valid_dataset = GraphDataset(file_dir, valid_num, raw_dir_name="valid")
        valid_loader = torch_geometric.loader.DataLoader(valid_dataset, batch_size_valid, shuffle=False, follow_batch=["y_cand_mask"])
    else:
        valid_loader = None
    return train_loader, valid_loader



def get_batch_score_precision(model, batch):
    pred_y = model(batch, train_mode=False)
    _, where_max = scatter_max(pred_y, batch.y_cand_mask_batch)
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



class TrainDSOAgent(object):
    def __init__(self, 

                 seed=0,

                 batch_size=1024, # number of generated expressions 
                 data_batch_size=2000, # number of data to evaluate fitness
                 eval_expression_num=48, # number of active expressions
                 score_func_name='precision',

                 record_expression_num=16, # top k expressions from fitness evaluation to evaluate on valid dataset
                 record_expression_freq=10, # evaluation frequency

                 early_stop=1000,

                 total_iter=None,

                 # env args
                 instance_kwargs={},

                 # expression
                 expression_kwargs={},

                 # agent
                 dso_agent_kwargs={},

                 # rl_algo
                 rl_algo_kwargs={},

                 ):
        self.batch_size, self.data_batch_size, self.eval_expression_num, self.seed = batch_size, data_batch_size, eval_expression_num, seed
        self.score_func_name = score_func_name
        self.early_stop, self.current_early_stop = early_stop, 0

        self.record_expression_num, self.record_expression_freq = record_expression_num, record_expression_freq
        self.instance_type = instance_kwargs["instance_type"]

        self.total_iter = consts.ITER_DICT[self.instance_type] if total_iter is None else total_iter
        self.train_data, self.valid_data = get_all_dataset(**instance_kwargs)

        # expression
        self.operators = dso_utils_graph.Operators(**expression_kwargs)

        # dso agent
        self.state_dict_dir, = logger.create_and_get_subdirs("state_dict")

        self.agent = dso_utils_graph.TransformerDSOAgent(self.operators, **dso_agent_kwargs["transformer_kwargs"])


        # rl algo
        self.rl_algo = PPOAlgo(agent=self.agent, **rl_algo_kwargs["kwargs"])

        # algo process variables
        self.train_iter = 0
        self.best_performance = - float("inf")
        self.best_writter = open(osp.join(logger.get_dir(), "best.txt"), "w")

    def process(self):
        start_time = time()
        for self.train_iter in range(self.total_iter+1):
            if self.current_early_stop > self.early_stop:
                break
            iter_start_time = time()

            sequences, all_lengths, log_probs, (scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, silbing_pairs, silbing_length) = self.agent.sample_sequence_eval(self.batch_size)
            expression_list = [dso_utils_graph.Expression(sequence[1:length+1], scatter_degree_now[:length], self.operators) for sequence, length, scatter_degree_now in zip(sequences, all_lengths, scatter_degree)]

            expression_generation_time = time() - iter_start_time

            eval_expression_start_time = time()

            # train
            ensemble_expressions = dso_utils_graph.EnsemBleExpression(expression_list)
            precisions = get_precision_iteratively(ensemble_expressions, self.train_data, self.data_batch_size, score_func_name=self.score_func_name)

            eval_expression_time = time() - eval_expression_start_time

            rl_start_time = time()
            returns, indices = torch.topk(precisions, self.eval_expression_num, sorted=False)

            sequences, all_lengths, log_probs = sequences[indices], all_lengths[indices], log_probs[indices]
            scatter_degree, all_counters_list, scatter_parent_where_seq = scatter_degree[indices],\
                                                                        [all_counters[indices] for all_counters in all_counters_list],\
                                                                        scatter_parent_where_seq[indices]
            parent_useful_index = torch.any(parent_child_pairs[:,0][:, None] == indices[None,:], dim=1)
            parent_child_pairs = parent_child_pairs[parent_useful_index]
            parent_useful_cumsum = torch.cumsum(parent_useful_index.long(),dim=0)
            parent_child_length[1:] = parent_useful_cumsum[parent_child_length[1:]-1]
            parent_new_index0 = torch.full((self.batch_size,), fill_value=-1,dtype=torch.long)
            parent_new_index0[indices] = torch.arange(len(indices))
            parent_child_pairs[:, 0] = parent_new_index0[parent_child_pairs[:, 0]]


            silbling_useful_index = torch.any(silbing_pairs[:,0][:, None] == indices[None,:], dim=1)
            silbing_pairs = silbing_pairs[silbling_useful_index]
            silbing_useful_cumsum = torch.cumsum(silbling_useful_index.long(), dim=0)
            where_start_positive = torch.where(silbing_length > 0)[0][0]
            silbing_length[where_start_positive:] = silbing_useful_cumsum[silbing_length[where_start_positive:]-1]
            silbing_new_index0 = torch.full((self.batch_size,), fill_value=-1,dtype=torch.long)
            silbing_new_index0[indices] = torch.arange(len(indices))
            silbing_pairs[:, 0] = silbing_new_index0[silbing_pairs[:, 0]]

            assert (silbing_pairs[:, 0].min() == parent_child_pairs[:, 0].min() == 0) and (silbing_pairs[:, 0].max() == parent_child_pairs[:, 0].max() ==  len(indices) - 1)

            index_useful = (torch.arange(sequences.shape[1]-1, dtype=torch.long)[None, :] < all_lengths[:, None]).type(torch.float32)

            results_rl = self.rl_algo.train(sequences, all_lengths, log_probs, index_useful, (scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, silbing_pairs, silbing_length), returns=returns, train_iter=self.train_iter)


            iter_end_time = time()
            rl_time = iter_end_time - rl_start_time
            iter_time = iter_end_time - iter_start_time

            ## tensorboard record
            total_time = iter_end_time - start_time
            results = {"train/batch_best_loss": returns.max().item(),
                       "train/batch_topk_mean_loss": returns.mean(),
                       "train/batch_topk_var_loss": returns.std(),
                       "train/batch_all_mean_loss": precisions.mean(),
                       "train/batch_all_var_loss": precisions.std(),

                       "train/train_iteration": self.train_iter,
                       "train/iter_time": iter_time,
                       "train/iter_time_generation": expression_generation_time,
                       "train/iter_time_evaluation": eval_expression_time,
                       "train/iter_time_rl": rl_time,
                       "train/total_time": total_time
                       }
            results.update(results_rl)

            ## save expressions and models
            if self.train_iter % self.record_expression_freq == 0:
                _, where_to_valid = torch.topk(precisions, self.record_expression_num, sorted=True)

                expressions_to_valid = [expression_list[i.item()] for i in where_to_valid]
                ensemble_expressions_valid = dso_utils_graph.EnsemBleExpression(expressions_to_valid)

                loss_valid = get_precision_iteratively(ensemble_expressions_valid, self.valid_data, score_func_name=self.score_func_name)
                precisions_valid = get_precision_iteratively(ensemble_expressions_valid, self.valid_data, score_func_name="precision")

                where_to_record = torch.where(loss_valid > self.best_performance)[0]
                if len(where_to_record) > 0:
                    self.current_early_stop = 0
                    pairs = [(expressions_to_valid[i], loss_valid[i].item(), precisions_valid[i]) for i in where_to_record]
                    pairs.sort(key=lambda x: x[1])
                    self.best_performance = pairs[-1][1]
                    for (exp, value, precision_value) in pairs:
                        best = f"iteration:{self.train_iter}_loss:{round(value, 4)}_precision:{round(precision_value.item(), 4)}\t{exp.get_nlp()}\t{exp.get_expression()}\n"
                        self.best_writter.write(best)
                    logger.log(best)
                    self.best_writter.flush()
                    os.fsync(self.best_writter.fileno())
                else:
                    self.current_early_stop += self.record_expression_freq
                results.update({
                    "valid/overall_best_loss": self.best_performance,

                    "valid/valid_best_loss": loss_valid.max().item(),
                    "valid/valid_all_mean_loss": loss_valid.mean(),
                    "valid/valid_all_var_loss": loss_valid.std(),

                    "valid/valid_best_loss_precision": precisions_valid[torch.argmax(loss_valid)].item(),
                    "valid/valid_best_precision": precisions_valid.max().item(),
                    "valid/valid_all_mean_precision": precisions_valid.mean(),
                    "valid/valid_all_var_precision": precisions_valid.std(),

                    "valid/valid_iteration": self.train_iter,
                })

                state_dict = self.agent.state_dict()
                state_dict_save_path = osp.join(self.state_dict_dir, f"train_iter_{self.train_iter}_precision_{round(value, 4)}.pkl")
                torch.save(state_dict, state_dict_save_path)

            logger.logkvs_tb(results)
            logger.dumpkvs_tb()
