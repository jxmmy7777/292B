from configs.config import Dict
from copy import deepcopy


class TrainConfig(Dict):
    def __init__(self):
        super(TrainConfig, self).__init__()
        self.logging.terminal_output_to_txt = True  # whether to log stdout to txt file
        self.logging.log_tb = True  # enable tensorboard logging
        self.logging.log_wandb = False  # enable wandb logging
        self.logging.log_every_n_steps = 10
        self.logging.flush_every_n_steps = 100

        ## save config - if and when to save model checkpoints ##
        self.save.enabled = True  # whether model saving should be enabled or disabled
        self.save.every_n_steps = 100  # save model every n epochs
        self.save.best_k = 5
        self.save.save_best_rollout = False
        self.save.save_best_validation = True

        ## evaluation rollout config ##
        self.rollout.save_video = True
        self.rollout.enabled = False  # enable evaluation rollouts
        self.rollout.every_n_steps = 1000  # do rollouts every @rate epochs
        self.rollout.warm_start_n_steps = 1  # number of steps to wait before starting rollouts


        ## training config
        self.training.batch_size = 256
        self.training.num_steps = 200000
        self.training.num_data_workers = 0

        ## validation config
        self.validation.enabled = True
        self.validation.batch_size = 100
        self.validation.num_data_workers = 0
        self.validation.every_n_steps = 1000
        self.validation.num_steps_per_epoch = 100
        

        ## Training parallelism (e.g., multi-GPU)
        self.parallel_strategy = "ddp_spawn"


class EnvConfig(Dict):
    def __init__(self):
        super(EnvConfig, self).__init__()
        self.name = "my_env"


class AlgoConfig(Dict):
    def __init__(self):
        super(AlgoConfig, self).__init__()
        self.name = "my_algo"
