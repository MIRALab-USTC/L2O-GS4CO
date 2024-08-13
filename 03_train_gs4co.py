import settings.consts as consts


from os import path as osp
import hydra
from omegaconf import OmegaConf

from train_utils_gs4co import TrainDSOAgent
import utils.utilities as utilities


@hydra.main(config_path='settings', config_name='train_gs4co', version_base=None)
def main(conf):
    log_dir = utilities.get_log_dir(exp_type="train_graph", instance_type=conf.instance_kwargs.instance_type, exp_name=conf.exp_name)

    for i in range(conf.exp_num):
        new_conf = OmegaConf.to_container(conf, resolve=True)
        train_kwargs, instance_kwargs, expression_kwargs, dso_agent_kwargs, rl_algo_kwargs= new_conf["train_kwargs"], new_conf["instance_kwargs"], new_conf["expression_kwargs"], new_conf["dso_agent_kwargs"], new_conf["rl_algo_kwargs"] # , inner_loop_kwargs , new_conf["inner_loop_kwargs"]

        logdir, _ = utilities.initial_logger_and_seed(log_dir, i, new_conf)
        train_agent = TrainDSOAgent(**train_kwargs, instance_kwargs=instance_kwargs, expression_kwargs=expression_kwargs, dso_agent_kwargs=dso_agent_kwargs, rl_algo_kwargs=rl_algo_kwargs)
        train_agent.process()
        del train_agent

if __name__ == "__main__":
    main()
