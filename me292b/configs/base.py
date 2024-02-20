from me292b.configs.config import Dict
from copy import deepcopy
import math

class TrainConfig(Dict):
    def __init__(self):
        super(TrainConfig, self).__init__()
        
        self.dataset_path = "MODIFY_ME"
        self.seed = 0
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
        self.training.num_data_workers = 4

        ## validation config
        self.validation.enabled = True
        self.validation.batch_size = 100
        self.validation.num_data_workers = 4
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
# ----------------------------config --------------------------------

class BehaviorCloningConfig(AlgoConfig):
    def __init__(self):
        super(BehaviorCloningConfig, self).__init__()
     
        self.name = "bc"
        self.raster_size = 224
        self.pixel_size = 0.5
          
        self.model_architecture = "mobilenet_v2" #mobilenet_v2 resnet18
        self.map_feature_dim = 256
        self.history_num_frames = 9
        self.history_num_frames_ego = 9
        self.history_num_frames_agents = 9
        self.future_num_frames = 30
        self.step_time = 0.1
        self.render_ego_history = False

        self.decoder.layer_dims = ()
        self.decoder.state_as_input = True

        self.dynamics.type = "Unicycle"
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph


        self.loss_weights.prediction_loss = 1.0
        self.loss_weights.goal_loss = 0.0
        self.loss_weights.collision_loss = 0.0
        self.loss_weights.yaw_reg_loss = 0.001

        self.optim_params.policy.learning_rate.initial = 1e-3  # policy learning rate
        self.optim_params.policy.learning_rate.decay_factor = (
            0.1  # factor to decay LR by (if epoch schedule non-empty)
        )
        self.optim_params.policy.learning_rate.epoch_schedule = (
            []
        )  # epochs where LR decay occurs
        self.optim_params.policy.regularization.L2 = 0.00  # L2 regularization strength
