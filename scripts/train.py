import argparse
import sys
import os
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from utils.log_utils import PrintLogger


#import trajdata module
from trajdata_datamodules import UnifiedDataModule

from algos import BehaviorCloning

def main(cfg, debug=False):
    pl.seed_everything(cfg.seed)
    
    torch.set_float32_matmul_precision('medium')



    logger = PrintLogger(os.path.join(log_dir, "log.txt"))
    sys.stdout = logger
    sys.stderr = logger

  
    # -----Dataset------------------
    datamodule = UnifiedDataModule
    datamodule.setup()

   
    # ------ Model ---------------------
    model = algo_factory(
        config=cfg,
        modality_shapes=datamodule.modality_shapes
    )
    ## Set CUDA visible devices:
  
    # Checkpointing
    if cfg.train.validation.enabled and cfg.train.save.save_best_validation:
        assert (
            cfg.train.save.every_n_steps > cfg.train.validation.every_n_steps
        ), "checkpointing frequency needs to be greater than validation frequency"
        for metric_name, metric_key in model.checkpoint_monitor_keys.items():
            print(
                "Monitoring metrics {} under alias {}".format(metric_key, metric_name)
            )
            ckpt_valid_callback = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="iter{step}_ep{epoch}_%s{%s:.2f}" % (metric_name, metric_key),
                # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
                auto_insert_metric_name=False,
                save_top_k=cfg.train.save.best_k,  # save the best k models
                monitor=metric_key,
                mode="min",
                every_n_train_steps=cfg.train.save.every_n_steps,
                verbose=True,
            )
            train_callbacks.append(ckpt_valid_callback)

    # a ckpt monitor to save at fixed interval
    ckpt_fixed_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="iter{step}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        monitor=None,
        every_n_train_steps=10000,
        verbose=True,
    )
    train_callbacks.append(ckpt_fixed_callback)
    
    
    # Logging
    assert not (cfg.train.logging.log_tb and cfg.train.logging.log_wandb)
    logger = None

    logger = TensorBoardLogger(
        save_dir=root_dir, version=version_key, name=None, sub_dir="logs/"
    )
    print("Tensorboard event will be saved at {}".format(logger.log_dir))
   

    # Train
    # Common trainer options
    trainer_options = {
        "default_root_dir": root_dir,
        # Checkpointing
        "enable_checkpointing": cfg.train.save.enabled,
        # Logging
        "logger": logger,
        "log_every_n_steps": cfg.train.logging.log_every_n_steps,
        # Training
        "max_steps": cfg.train.training.num_steps,
        # Validation
        "val_check_interval": cfg.train.validation.every_n_steps,
        "limit_val_batches": cfg.train.validation.num_steps_per_epoch,
        # All callbacks
        "callbacks": train_callbacks,
        # Device & distributed training setup
        "gpus": cfg.devices.num_gpus,
        "strategy": cfg.train.parallel_strategy,
        "inference_mode": not cfg.eval,
    }

    # Add debug-specific options
    if debug:
        trainer_options.update({
            "overfit_batches": 1,
            # Uncomment if you want to use fast_dev_run for a quick complete run (training+validation+test)
            # "fast_dev_run": True,
            "val_check_interval": 1
        })

    # Initialize the trainer with the configured options
    trainer = pl.Trainer(**trainer_options)

    trainer.fit(model=model, datamodule=datamodule,ckpt_path=checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config

    # Experiment Name (for tensorboard, saving models, etc.)

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset root path",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="(optional) if provided, load from checkpoint",
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory of training output (checkpoints, visualization, tensorboard log, etc.)",
    )
    
    parser.add_argument(
        "--eval",
        action="store_true",
        help="whether to evaluate the model ",
    )

    parser.add_argument(
        "--remove_exp_dir",
        action="store_true",
        help="Whether to automatically remove existing experiment directory of the same name (remember to set this to "
        "True to avoid unexpected stall when launching cloud experiments).",
    )


    parser.add_argument(
        "--debug", action="store_true", help="Debug mode, suppress wandb logging, etc."
    )

    args = parser.parse_args()

    if args.config_name is not None:
        default_config = get_registered_experiment_config(args.config_name)
    elif args.config_file is not None:
        # Update default config with external json file
        default_config = get_experiment_config_from_file(args.config_file, locked=False)
    else:
        raise Exception(
            "Need either a config name or a json file to create experiment config"
        )
    if args.checkpoint:
        path = Path(args.checkpoint)

        # Navigate up the tree until you find a parent directory starting with 'run'
        prev_config_path = path.parent
        while not prev_config_path.name.startswith('run'):
            prev_config_path = prev_config_path.parent
        prev_config = get_experiment_config_from_file(str(prev_config_path / "config.json"), locked=False)
        default_config.algo.update(prev_config.algo)
    if args.name is not None:
        default_config.name = args.name

    if args.dataset_path is not None:
        default_config.train.dataset_path = args.dataset_path

    if args.output_dir is not None:
        default_config.root_dir = os.path.abspath(args.output_dir)

    if args.wandb_project_name is not None:
        default_config.train.logging.wandb_project_name = args.wandb_project_name

    default_config.eval = args.eval
  
    main(default_config, auto_remove_exp_dir=args.remove_exp_dir, debug=args.debug)
