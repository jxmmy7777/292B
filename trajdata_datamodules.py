import os
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.custom_func import get_actions_inversdyn, is_stationary, get_lane_info

class UnifiedDataModule(pl.LightningDataModule):
    def __init__(self, data_config, train_config):
        super(UnifiedDataModule, self).__init__()
        self._data_config = data_config
        self._train_config = train_config
        self.train_dataset = None
        self.valid_dataset = None

    @property
    def modality_shapes(self):
        # TODO: better way to figure out channel size?
        return dict(
            image=(3 + self._data_config.history_num_frames + 1,  # semantic map + num_history + current
                   self._data_config.raster_size,
                   self._data_config.raster_size),
            static=(3,self._data_config.raster_size,self._data_config.raster_size),
            dynamic=(self._data_config.history_num_frames + 1,self._data_config.raster_size,self._data_config.raster_size)

        )

    def setup(self, stage = None):
        data_cfg = self._data_config
        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance
        kwargs = dict(
            centric = data_cfg.centric,
            desired_data=[data_cfg.trajdata_source_train],
            # desired_data=["lyft_val"],
            desired_dt=data_cfg.step_time,
            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs={
                # data_cfg.trajdata_source_root: data_cfg.dataset_path,
                # "waymo_train":"/mnt/hdd2/waymo",
                # "waymo_val":"/mnt/hdd2/waymo",
                
                # "waymo_train": "/home/msc_lab/datasets/waymo_open_dataset_motion_v_1_1_0/debug",
                # "waymo_val": "/home/msc_lab/datasets/waymo_open_dataset_motion_v_1_1_0/debug",
                "lyft_train": "~/datasets/lyft-prediction-dataset/scenes/train_full.zarr",
                "lyft_val": "~/datasets/lyft-prediction-dataset/scenes/validate.zarr",
                
            },
            only_types=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=True,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center,
                "original_format": True,
            },
            cache_location= "~/.unified_data_cache",
            verbose=False,
            max_agent_num = 1+data_cfg.other_agents_num,
            # max_neighbor_num = data_cfg.other_agents_num,
            extras = {
                "target_actions":get_actions_inversdyn,
                "is_stationary":is_stationary,
            },
            num_workers=os.cpu_count(),
            # save_index = True
            # ego_only = self._train_config.ego_only,
            # filter_fn = self.stationary_filter,
        )
        print(kwargs)
        self.train_dataset = UnifiedDataset(**kwargs)
        # save_path = "train_stationary_filter"
        # self.train_dataset.apply_filter(self.stationary_filter,save_path=save_path, num_workers=4)
        kwargs["desired_data"] = [data_cfg.trajdata_source_valid]
        kwargs["rebuild_cache"] = False
        self.valid_dataset = UnifiedDataset(**kwargs)

    def train_dataloader(self, return_dict = True):
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self._train_config.training.batch_size,
            num_workers=self._train_config.training.num_data_workers,
            drop_last=True,
            collate_fn=self.train_dataset.get_collate_fn(return_dict=return_dict),
            persistent_workers=True if self._train_config.training.num_data_workers>0 else False
            
        )

    def val_dataloader(self, return_dict = True):
        return DataLoader(
            dataset=self.valid_dataset,
            shuffle=True,
            batch_size=self._train_config.validation.batch_size,
            num_workers=self._train_config.validation.num_data_workers,
            drop_last=True,
            collate_fn=self.valid_dataset.get_collate_fn(return_dict=return_dict),
            persistent_workers=True if self._train_config.validation.num_data_workers>0 else False
        )

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass
    
    @staticmethod
    def stationary_filter(elem) -> bool:
        return elem.agent_meta_dict["is_stationary"]
