from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.augmentation import NoiseHistories
from trajdata.visualization.vis import plot_agent_batch

INTERACTIONS_DATASETS_PATH = "/mnt/hdd2/interaction_dataset/single"

def main():
    # noise_hists = NoiseHistories()

    dataset = UnifiedDataset(
        # desired_data=["lyft_val"],
        # desired_data=["waymo_train"],
        desired_data=["val"],
        centric="agent",
        desired_dt=0.1,
        history_sec=(0.9, 0.9),
        future_sec=(3.0, 3.0),
        only_predict=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_robot_future=False,
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
        },
        num_workers=0,
        obs_format="x,y,xd,yd,xdd,ydd,s,c",
        verbose=True,
        data_dirs={  # Remember to change this to match your filesystem!
            # "waymo_val": "/home/msc_lab/datasets/waymo_open_dataset_motion_v_1_1_0/debug",
            # "waymo_train": "/mnt/hdd2/waymo",
            # "waymo_val":"/mnt/hdd2/waymo",
            # "lyft_val": "~/datasets/lyft-prediction-dataset",
            "interaction_single":INTERACTIONS_DATASETS_PATH
        },
        # extras = {
        #     "target_actions": get_actions_inversdyn,
        #     "is_stationary": is_stationary,
        # },
        cache_location= "/mnt/hdd2/weijer/.unified_data_cache/",
        save_index = False,
        # filter_fn = stationary_filter
    )
    # dataset.apply_filter(filter_fn=stationary_filter, num_workers=4)

    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=16,
    )

    batch: AgentBatch
    import matplotlib.pyplot as plt
    i=0
    for batch in tqdm(dataloader):
        i=i+1
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax = plot_agent_batch(batch, batch_idx=0, ax=ax, close=False)
        plt.savefig(f"/home/msc_lab/ctang/weijer/292B/visualizations/{i}_interaction_batch_example.png")
        if i>=20:
            break
# def stationary_filter(elem) -> bool:
#     return elem.agent_meta_dict["is_stationary"]


if __name__ == "__main__":
    main()
