import os.path as osp
import os
import argparse
import csv
import time
import pyscipopt as scip

import torch
from torch_scatter import scatter_mean, scatter_sum

from utils.utilities import extract_state, init_scip_params, scatter_max, scatter_min
import settings.consts as consts


def get_graph_policy_from_dataset_name(dataset_name):
    if dataset_name == "indset":
        dataset_name = "cauctions" # As mentioned in Table C6, we use the policy trained from Cauctions for Indset as it achieves higher performance.
    with open(osp.join("./expressions", dataset_name), "r") as txt:
        expression = next(txt)
    variable_allocation_exp, calculation_exp = expression.split(";;")
    def get_logits(state):
        constraint, cv_edge_index, edge_attr, variable = state
        c_edge_index, v_edge_index = cv_edge_index
        exec(variable_allocation_exp)
        result = eval(calculation_exp)
        return result
    return get_logits

class PolicyBranching(scip.Branchrule):

    def __init__(self, dataset_name, device, depth=25):
        super().__init__()
        self.device = device
        self.policy = get_graph_policy_from_dataset_name(dataset_name)
        self.depth = depth

    def branchinitsol(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.state_buffer = {}
        self.khalil_root_buffer = {}

    @torch.no_grad()
    def branchexeclp(self, allowaddcons):
        if self.model.getDepth() < self.depth:
            candidate_vars, *_ = self.model.getPseudoBranchCands()
            candidate_mask = [var.getCol().getIndex() for var in candidate_vars]
            state = extract_state(self.model, self.state_buffer)
            c,e,v =  state

            state = (
                torch.as_tensor(c['values'], dtype=torch.float32, device=self.device),
                torch.as_tensor(e['indices'], dtype=torch.long, device=self.device),
                torch.as_tensor(e['values'], dtype=torch.float32, device=self.device).reshape(-1),
                torch.as_tensor(v['values'], dtype=torch.float32, device=self.device),
            )
            var_logits = self.policy(state)
            candidate_scores = var_logits[candidate_mask]
            best_var = candidate_vars[candidate_scores.argmax()]

            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED
        else:
            result = self.model.executeBranchRule("relpscost", allowaddcons)

        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}

def get_expression_result(instance, time_limit, seed, dataset_name, device, name, **kwargs):
    m = scip.Model()
    m.setIntParam('display/verblevel', 0)
    m.readProblem(f"{instance['path']}")
    init_scip_params(m, seed=seed)
    m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
    m.setRealParam('limits/time', time_limit)

    brancher = PolicyBranching(dataset_name, device, **kwargs)
    m.includeBranchrule(
        branchrule=brancher,
        name=name,
        desc=f"Custom MLPOpt branching policy.",
        priority=666666, maxdepth=-1, maxbounddist=1)

    walltime = time.perf_counter()
    proctime = time.process_time()

    m.optimize()

    walltime = time.perf_counter() - walltime
    proctime = time.process_time() - proctime

    stime = m.getSolvingTime()
    nnodes = m.getNNodes()
    nlps = m.getNLPs()
    gap = m.getGap()
    status = m.getStatus()
    ndomchgs = brancher.ndomchgs
    ncutoffs = brancher.ncutoffs

    result = {
        'policy': name,
        'seed': seed,
        'type': instance['type'],
        'instance': instance['path'],
        'nnodes': nnodes,
        'nlps': nlps,
        'stime': stime,
        'gap': gap,
        'status': status,
        'ndomchgs': ndomchgs,
        'ncutoffs': ncutoffs,
        'walltime': walltime,
        'proctime': proctime,
        'problem':dataset_name,
        'device': device
    }
    m.freeProb()
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '-s', '--seed',
        help='seed for parallelizing the evaluation. Uses all seeds if not provided.',
        type=int,
        default=-1
    )
    parser.add_argument(
        '-l', '--level',
        help='size of instances to evaluate. Default is all.',
        type=str,
        default='all',
        choices=['all', 'small', 'medium', 'big']
    )

    args = parser.parse_args()

    instances = []
    seed = 0
    time_limit = 3000
    evaluate_instance_num = 50
    

    result_dir = f"eval_results/{args.problem}"
    os.makedirs(result_dir, exist_ok=True)
    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"


    if args.problem == 'setcover':
        instances += [{'type': 'small', 'path': f"data/instances/setcover/transfer_500r_1000c_0.05d/instance_{i+1}.lp"} for i in range(evaluate_instance_num)]
        instances += [{'type': 'medium', 'path': f"data/instances/setcover/transfer_1000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(evaluate_instance_num)]
        instances += [{'type': 'big', 'path': f"data/instances/setcover/transfer_2000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(evaluate_instance_num)]

    elif args.problem == 'cauctions':
        instances += [{'type': 'small', 'path': f"data/instances/cauctions/transfer_100_500/instance_{i+1}.lp"} for i in range(evaluate_instance_num)]
        instances += [{'type': 'medium', 'path': f"data/instances/cauctions/transfer_200_1000/instance_{i+1}.lp"} for i in range(evaluate_instance_num)]
        instances += [{'type': 'big', 'path': f"data/instances/cauctions/transfer_300_1500/instance_{i+1}.lp"} for i in range(evaluate_instance_num)]

    elif args.problem == 'facilities':
        instances += [{'type': 'small', 'path': f"data/instances/facilities/transfer_100_100_5/instance_{i+1}.lp"} for i in range(evaluate_instance_num)]
        instances += [{'type': 'medium', 'path': f"data/instances/facilities/transfer_200_100_5/instance_{i+1}.lp"} for i in range(evaluate_instance_num)]
        instances += [{'type': 'big', 'path': f"data/instances/facilities/transfer_400_100_5/instance_{i+1}.lp"} for i in range(evaluate_instance_num)]

    elif args.problem == 'indset':
        instances += [{'type': 'small', 'path': f"data/instances/indset/transfer_750_4/instance_{i+1}.lp"} for i in range(evaluate_instance_num)]
        instances += [{'type': 'medium', 'path': f"data/instances/indset/transfer_1000_4/instance_{i+1}.lp"} for i in range(evaluate_instance_num)]
        instances += [{'type': 'big', 'path': f"data/instances/indset/transfer_1500_4/instance_{i+1}.lp"} for i in range(evaluate_instance_num)]

    else:
        raise NotImplementedError

    ### SEEDS TO EVALUATE ###
    if args.seed != -1:
        seed = args.seed
    torch.manual_seed(seed)

    ### PROBLEM SIZES TO EVALUATE ###
    if args.level != "all":
        instances = [x for x in instances if x['type'] == args.level]

    # load and assign tensorflow models to policies (share models and update parameters)
    loaded_models = {}

    fieldnames = [
        'problem',
        'device',
        'policy',
        'seed',
        'type',
        'instance',
        'nnodes',
        'nlps',
        'stime',
        'gap',
        'status',
        'ndomchgs',
        'ncutoffs',
        'walltime',
        'proctime',
    ]


    with open(osp.join(result_dir, f"gs4co_{time.strftime('%Y%m%d-%H%M%S')}"), 'w', newline='') as csvfile_gs4co, open(osp.join(result_dir, f"rpb_{time.strftime('%Y%m%d-%H%M%S')}"), 'w', newline='') as csvfile_rpb:
        writer_gs4co, writer_rpb = csv.DictWriter(csvfile_gs4co, fieldnames=fieldnames), csv.DictWriter(csvfile_rpb, fieldnames=fieldnames)
        writer_gs4co.writeheader()
        writer_rpb.writeheader()

        for instance in instances:
            print(f"{instance['type']}: {instance['path']}...")

            # gs4co
            result = get_expression_result(instance, time_limit, seed, args.problem, device, "gs4co")
            writer_gs4co.writerow(result)
            csvfile_gs4co.flush()
            print(f"gs4co_{args.problem}_{instance['path'].split('/')[-1]}:\t {result['nnodes']} nodes {result['nlps']} lps {result['stime']:.2f}s stime {result['status']} status")

            # rpb
            result = get_expression_result(instance, time_limit, seed, args.problem, device, "rpb", depth=0)
            writer_rpb.writerow(result)
            csvfile_rpb.flush()
            print(f"rpb_{args.problem}_{instance['path'].split('/')[-1]}:\t {result['nnodes']} nodes {result['nlps']} lps {result['stime']:.2f}s stime {result['status']} status")

