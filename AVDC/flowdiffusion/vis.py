import lightning as L
from tqdm import tqdm
from minestudio.data import MineDataModule
from minestudio.data.minecraft.utils import visualize_dataloader
import time
import random
import torch

t = int(time.time())
random.seed(t)
torch.manual_seed(t)
data_module = MineDataModule(
    data_params=dict(
        mode='raw',
        dataset_dirs=[
            '/nfs-shared-2/data/contractors/dataset_10xx',
        ],
        frame_width=224,
        frame_height=224,
        win_len=128,
        split_ratio=0.8,
        # event_regex='minecraft.mine_block:.*diamond.*',
    ),
    shuffle_episodes=True,
    batch_size=2,
)
data_module.setup()
dataloader = data_module.val_dataloader()

visualize_dataloader(
    dataloader, 
    num_samples=5, 
    resolution=(640, 360), 
    legend=True,  # print action, contractor info, and segment info ... in the video
    save_fps=30, 
    output_dir="./"
)