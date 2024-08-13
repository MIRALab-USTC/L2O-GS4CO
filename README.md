# Towards General Algorithm Discovery for Combinatorial Optimization: Learning Symbolic Branching Policy from Bipartite Graph


This is the code of paper *Towards General Algorithm Discovery for Combinatorial Optimization: Learning Symbolic Branching Policy from Bipartite Graph*. [[Paper](https://openreview.net/forum?id=ULleq1Dtaw&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2024%2FConference%2FAuthors%23your-submissions))]


## Installation

### Revidsed scip 6.0.1

We revise the official scip (version 6.0.1) to add interfaces for Symb4CO. To install the revised scip:
1. Download the official scipoptsuite 6.0.1 from [here](https://scipopt.org/download.php?fname=scipoptsuite-6.0.1.tgz).
2. Replace the folder `./scipoptsuite-6.0.1/scip` (i.e., the folder of scip) with our revised version provided [here](https://github.com/MIRALab-USTC/scip/tree/symb4co-iclr2024). 
> Attention! The revised scip is in the `symb4co-iclr2024` branch, **not the `main`**!
3. Installing the revised scip 6.0.1 via cmake following the official [instructions](https://scipopt.org/doc/html/md_INSTALL.php).

P.S.: You can refer to the git commits of our revised scip 6.0.1 to identify the changes, compared with the original scip 6.0.1 and the revised version provided in [learn2branch](https://github.com/ds4dm/learn2branch/blob/master/scip_patch/vanillafullstrong.patch).

### Revised PySCIPOpt

We revise the PySCIPOpt to add interfaces for Symb4CO. To install the revised PySCIPOpt:
1. Specify the installation path of scip 6.0.1
```bash
export SCIPOPTDIR='/path/to/where/you/installation'
```
2. Install the revised PySCIPOpt
```bash
pip install git+https://github.com/MIRALab-USTC/PySCIPOpt.git@symb4co-iclr2024
```
> Attention! The revised PySCIPOpt is in the `symb4co-iclr2024` branch, **not the `master`**!

P.S.: You can refer to the last git commit of our revised PySCIPOpt to identify what has been revised, compared with the version provided in [learn2branch](https://github.com/ds4dm/PySCIPOpt/tree/ml-branching).

### Python Environment

We list the required python 3.9 packages in `./requirement.txt`. 

P.S.: We found inconsistent Pytorch might lead to unknown RuntimeError.

## Instructions for Execution

### Generating Instances and Datasets
The instance and the dataset generation is based on the [codes](https://github.com/pg2455/Hybrid-learn2branch) implemented by Gupta et al. To generate them, run

```bash
# generate CO instances
python 01_generate_instances.py indset
# generate full strong branching datasets
python 02_generate_dataset.py indset
# indset can be replaced to cauctions, setcover, and facilities
```

### Inference

If you do not want to train new symbolic policies, you can directly use our trained ones in `./expressions` via
```bash
# To evaluate the imitation learning accuracy
python 04_test_gs4co.py indset
# To evaluate the end-to-end performance of learned graph symbolic policies
python 05_evaluate_gs4co.py indset
```


### Training
To train symbolic policies, run

```bash
python 03_train_gs4co.py instance_kwargs.instance_type=indset
```
