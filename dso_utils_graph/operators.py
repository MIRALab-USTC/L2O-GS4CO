"""
File adapted from https://github.com/dandip/DSRPytorch
"""

import torch
from collections import OrderedDict, defaultdict

import settings.consts as consts
import utils.logger as logger

MATH_ARITY = OrderedDict([ # arity 2 -> arity 1
    ('+',2),
    ('-',2),
    ('*',2),
    ('/',2), 
    ('^', 2),
    ('exp',1),
    ('log',1),
    ('scatter_sum',1),
    ('scatter_mean',1),
    ('scatter_max',1),
    ('scatter_min',1),
])
CONSTANT_OPERATORS = ["2.0", "5.0", "10.0", "0.1", "0.2", "0.5"] # @ means a place holder which will be optimized in the inner loop
INVERSE_OPERATOR_DICT = {"exp": "log", "log": "exp", "sqrt": "square", "square": "sqrt"}

def binary_f(x,a,b):
    return f"({a} {x} {b})"
def unary_f(x,a):
    return f"torch.{x}({a})"
def unary_f_nlp(x,a):
    return f"{x}({a})"
def nlp_power(x,a,b):
    return f"{a}^{b}"
def nlp_square(x,a):
    return f"{a}^2"
def nlp_sqrt(x,a):
    return f"{a}^0.5"

def scatter_func(x, a, scatter_degree, max_degree):
    assert 0 < scatter_degree <= max_degree

    b = "c_edge_index" if (scatter_degree % 2) == 0 else "v_edge_index"
    output = f"{x}({a},{b})"
    if scatter_degree > 1:
        output += f"[{b}]"
    return output

def scatter_func_nlp(x, a, scatter_degree, max_degree):
    return f"{x}({a})"

def power(x,a,b):
    return f"torch.pow({a}, {b})"

TORCH_OPERATOR_DICT = {
    "^": power,
}

NLP_OPERATOR_DICT = {
    "^": nlp_power,
}


class Operators:

    def __init__(self, const_list=None, math_list="simple", var_list="graph", scatter_max_degree=2):
        """
        order: vars, consts, arity_two_operators, arity_one_operators
        """
        if var_list == "graph":
            self.var_operators = consts.GRAPH_NAMES
        else:
            raise NotImplementedError

        if const_list is None:
            self.constant_operators = CONSTANT_OPERATORS[:]
        else:
            self.constant_operators = const_list

        if math_list == "simple":
            self.math_operators = ['+', '-', '*', 'scatter_sum', 'scatter_mean', 'scatter_max', 'scatter_min'] # , 
        else:
            assert math_list == 'all'
            self.math_operators = list(MATH_ARITY.keys())
        self.scatter_max_degree = scatter_max_degree


        self.operator_list = self.var_operators + self.constant_operators + self.math_operators
        self.operator_dict = {k:i for i,k in enumerate(self.operator_list)}
        self.operator_length = len(self.operator_list)

        arity_dict = defaultdict(int, MATH_ARITY)
        self.arity_list = [arity_dict[operator] for operator in self.operator_list]
        self.arity_tensor = torch.tensor(self.arity_list, dtype=torch.long)

        self.zero_arity_mask = torch.tensor([True if arity_dict[x]==0 else False for x in self.operator_list], dtype=torch.bool)[None, :]
        self.nonzero_arity_mask = torch.tensor([True if arity_dict[x]!=0 else False for x in self.operator_list], dtype=torch.bool)[None, :]

        self.have_inverse = torch.tensor([((operator in INVERSE_OPERATOR_DICT) and (INVERSE_OPERATOR_DICT[operator] in self.operator_dict)) for operator in self.operator_list], dtype=torch.bool)
        self.where_inverse = torch.full(size=(self.operator_length,), fill_value=int(1e5), dtype=torch.long)
        self.where_inverse[self.have_inverse] = torch.tensor([self.operator_dict[INVERSE_OPERATOR_DICT[operator]] for i, operator in enumerate(self.operator_list) if self.have_inverse[i]], dtype=torch.long)

        variable_mask = torch.zeros(len(self.operator_list), dtype=torch.bool)
        variable_mask[:len(self.var_operators)] = True
        self.variable_mask = variable_mask[None, :]
        self.non_variable_mask = torch.logical_not(self.variable_mask)

        const_mask = torch.zeros(len(self.operator_list), dtype=torch.bool)
        const_mask[len(self.var_operators):-len(self.math_operators)] = True
        self.const_mask = const_mask[None, :]
        self.non_const_mask = torch.logical_not(self.const_mask)

        self.scatter_num = sum([("scatter" in x) for x in self.math_operators])
        assert self.scatter_num > 0
        scatter_mask = torch.zeros(len(self.operator_list), dtype=torch.bool)
        scatter_mask[-self.scatter_num:] = True
        self.scatter_mask = scatter_mask[None, :]
        self.non_scatter_mask = torch.logical_not(self.scatter_mask)

        num_math_arity_two = sum([1 for x in self.math_operators if MATH_ARITY[x]==2])
        num_math_arity_one = len(self.math_operators) - num_math_arity_two
        self.arity_zero_begin, self.arity_zero_end = 0, len(self.var_operators) + len(self.constant_operators)
        self.arity_two_begin, self.arity_two_end = len(self.var_operators) + len(self.constant_operators), len(self.var_operators) + len(self.constant_operators) + num_math_arity_two
        self.arity_one_begin, self.arity_one_end = len(self.operator_list) - num_math_arity_one, len(self.operator_list)

        self.variable_begin, self.variable_end = 0, len(self.var_operators)
        self.scatter_begin, self.scatter_end = len(self.operator_list) - self.scatter_num, len(self.operator_list)

        self.variable_constraint_begin, self.variable_constraint_end = 0, len(consts.CONSTRAINT_FEATURES)
        self.variable_variable_begin, self.variable_variable_end = self.variable_constraint_end + len(consts.EDGE_FEATURES), self.variable_constraint_end + len(consts.EDGE_FEATURES) + len(consts.NODE_FEATURES)

        scatter_degree_0_mask = torch.zeros(len(self.operator_list), dtype=torch.bool)
        scatter_degree_0_mask[:self.variable_variable_begin] = True
        self.scatter_degree_0_mask = scatter_degree_0_mask[None, :]

        scatter_degree_1_mask = torch.zeros(len(self.operator_list), dtype=torch.bool)
        scatter_degree_1_mask[self.variable_constraint_end:-len(self.math_operators)] = True
        self.scatter_degree_1_mask = scatter_degree_1_mask[None, :]


        scatter_degree_2_mask = scatter_degree_0_mask.clone()
        scatter_degree_2_mask[self.variable_variable_end:-len(self.math_operators)] = True
        self.scatter_degree_2_mask = scatter_degree_2_mask[None, :]


    def is_var_i(self, i):
        return self.variable_begin <= i < self.variable_end
