"""
File adapted from https://github.com/dandip/DSRPytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_scatter import scatter_max as scatter_max_raw, scatter_min as scatter_min_raw

from collections import defaultdict

from contextlib import contextmanager

import settings.consts as consts
import dso_utils_graph.operators as operators_module

def scatter_min(src, index):
    return scatter_min_raw(src, index)[0]

def scatter_max(src, index):
    return scatter_max_raw(src, index)[0]

@contextmanager
def set_train_mode(network, train_mode=True):
    if train_mode:
        yield
    else:
        with torch.no_grad():
            network.eval()
            yield
            network.train()

class OperatorNode:
    def __init__(self, operator, operators, scatter_degree, parent=None):
        """Description here
        """
        self.operator = operator.item()
        self.operators = operators
        self.operator_str = operators.operator_list[operator]
        self.arity = operators.arity_list[operator]
        self.is_var = operator < operators.variable_end
        self.is_scatter = operator >= operators.scatter_begin

        self.parent = parent
        self.left_child = None
        self.right_child = None
        self.scatter_degree = scatter_degree

    def add_child(self, node):
        if (self.left_child is None):
            self.left_child = node
        elif (self.right_child is None):
            self.right_child = node
        else:
            raise RuntimeError("Both children have been created.")

    def set_parent(self, node):
        self.parent = node

    def remaining_children(self):
        if (self.arity == 0):
            return False
        elif (self.arity == 1 and self.left_child is not None):
            return False
        elif (self.arity == 2 and self.left_child is not None and self.right_child is not None):
            return False
        return True

    def print(self, operator_dict, unary_func, use_tensor, tree_dict, tree_list):
        if (self.arity == 2):
            left_print = self.left_child.print(operator_dict, unary_func, use_tensor, tree_dict, tree_list)
            right_print = self.right_child.print(operator_dict, unary_func, use_tensor, tree_dict, tree_list)
            return operator_dict.get(self.operator_str, operators_module.binary_f)(self.operator_str, left_print, right_print)
        elif (self.arity == 1):
            left_print = self.left_child.print(operator_dict, unary_func, use_tensor, tree_dict, tree_list)
            scatter_func = operators_module.scatter_func if use_tensor else operators_module.scatter_func_nlp
            value = scatter_func(self.operator_str, left_print, self.scatter_degree+1, self.operators.scatter_max_degree) if self.is_scatter else operator_dict.get(self.operator_str, unary_func)(self.operator_str, left_print) # +1 since the scatter degree did not add at current function
            return value
        else:
            assert self.arity == 0
            if self.is_var:
                # return variable_func(self.operator_str)
                if use_tensor:
                    key_name = f"{int(self.scatter_degree>0)}_{self.operator}"
                    x_name = tree_dict.get(key_name, None)
                    if x_name is None:
                        if self.operators.variable_constraint_begin <= self.operator < self.operators.variable_constraint_end:
                            x_name = f"constraint_{key_name}"
                            x_value = f"constraint[:,{self.operator}]"
                        elif self.operators.variable_variable_begin <= self.operator < self.operators.variable_variable_end:
                            x_name = f"variable_{key_name}"
                            x_value = f"variable[:, {self.operator - self.operators.variable_variable_begin}]"
                        else:
                            x_name = "edge_attr"
                            x_value = ""
                        if x_value:
                            if self.scatter_degree > 0:
                                scatter_index = ("[v_edge_index]") if (x_name[0] == "v") else ("[c_edge_index]")
                                x_value += scatter_index
                            tree_list.append(f"{x_name}={x_value}")
                        tree_dict[key_name] = x_name
                    return x_name
                else:
                    return self.operator_str  # f"inputs[:,{self.operator}]" if use_tensor else 
            else:
                if use_tensor:
                    return f"torch.tensor({self.operator_str}, dtype=torch.float, device=consts.DEVICE)"
                else:
                    return self.operator_str


def construct_tree(operators, sequence, scatter_degree):

    root = OperatorNode(sequence[0], operators, scatter_degree[0])
    past_node = root
    for operator, scatter_degree_i in zip(sequence[1:], scatter_degree[1:]):
        curr_node = OperatorNode(operator, operators, scatter_degree_i, parent=past_node)
        past_node.add_child(curr_node)
        past_node = curr_node
        while not past_node.remaining_children():
            past_node = past_node.parent
            if (past_node is None):
                assert operator == sequence[-1]
                break
    return root


class Expression(nn.Module):
    def __init__(self, sequence, scatter_degree, operators, from_expression=None):
        super().__init__()

        if from_expression is None:
            self.sequence = sequence

            self.root = construct_tree(operators, sequence, scatter_degree)
            tree_list = []
            self.expression = self.root.print(operators_module.TORCH_OPERATOR_DICT, operators_module.unary_f, True, {}, tree_list)
            self.alloc_expression = ";".join(tree_list)

        else:
            assert ";;" in from_expression
            self.alloc_expression, self.expression = from_expression.split(";;")

    def get_nlp(self):
        return self.root.print(operators_module.NLP_OPERATOR_DICT, operators_module.unary_f_nlp, False, {}, [])

    def get_expression(self, record_alloc=True):
        return (self.alloc_expression + ";;" + self.expression) if record_alloc else self.expression

    def forward(self, constraint, variable, cv_edge_index, edge_attr, cand_mask):
        c_edge_index, v_edge_index = cv_edge_index
        exec(self.alloc_expression)
        result = eval(self.expression)
        return result[cand_mask]

class EnsemBleExpression(nn.Module):
    def __init__(self, models, max_parallel=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_num = len(self.models)
        self.max_parallel = max_parallel
        self.train()

    def forward(self, batch, train_mode=False):
        with set_train_mode(self, train_mode):
            if self.max_parallel is None:
                futures = [torch.jit.fork(model, batch.x_constraint, batch.x_variable, batch.x_cv_edge_index, batch.x_edge_attr, batch.y_cand_mask) for model in self.models]
                results = [torch.jit.wait(fut) for fut in futures]
            else:
                results = []
                for i in range(0, self.ensemble_num, self.max_parallel):
                    futures = [torch.jit.fork(model, batch.x_constraint, batch.x_variable, batch.x_cv_edge_index, batch.x_edge_attr, batch.y_cand_mask) for model in self.models[i:i+self.max_parallel]]
                    results += [torch.jit.wait(fut) for fut in futures]
                    del futures
                    torch.cuda.empty_cache()
            return torch.stack(results)
