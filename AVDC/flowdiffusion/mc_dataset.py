from torch.utils.data import Dataset, DataLoader
from minestudio.data import load_dataset
from einops import rearrange

"""
vpt contractor data:
6xx: free play, very diveserse, AGI level tasks/beahviors
7xx: early game, half of the data from early 30 minutes, half like 6xx 
8xx: house building 
9xx: House Building from Random Starting Materials Task
10xx: obtain diamond pickaxe
"""
"""
this is a wrapper for load_dataset, to return a 
dataset object that meets the requirements of the AVDC model
"""
class MCdataset(Dataset):
    def __init__(
        self,
        mode = 'raw',
        dataset_dirs = [], 
        enable_video = True, 
        enable_action = False, 
        frame_width = 128, 
        frame_height = 128, 
        win_len = 128, 
        skip = 1,
        split = 'train', 
        split_ratio = 0.9, 
        verbose = True, 
        enable_contractor_info = True
    ):
        self.mode = mode
        self.dataset_dirs = dataset_dirs
        self.enable_video = enable_video
        self.enable_action = enable_action
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.win_len = win_len
        self.skip = skip
        self.split = split
        self.split_ratio = split_ratio
        self.verbose = verbose
        self.enable_contractor_info = enable_contractor_info
        self.dataset = load_dataset(
            mode=self.mode, 
            dataset_dirs=self.dataset_dirs, 
            enable_video=self.enable_video,
            enable_action=self.enable_action,
            frame_width=self.frame_width, 
            frame_height=self.frame_height,
            win_len=self.win_len*self.skip, # sample (win_len*skip) frames
            split=self.split, 
            split_ratio=self.split_ratio, 
            verbose=self.verbose,
            enable_contractor_info=self.enable_contractor_info, # add text data 
        )
    
    def __len__(self):
        return len(self.dataset)
    """
    returns:
    x (f,c) h w  [0,1] f: frames, c: channels, h: height, w: width
    x_cond c h w
    task ...
    """
    def __getitem__(self, idx):
        img = self.dataset[idx]['image'] / 255 # normalize to [0, 1]
        x_cond = img[0,:]
        x = img[self.skip::self.skip,:]
        x = rearrange(x, 'f h w c -> (f c) h w')
        x_cond = rearrange(x_cond, 'h w c -> c h w')
        x = x.float()
        x_cond = x_cond.float()
        if self.enable_contractor_info:
            task = self.dataset[idx]['contractor_info']
        else :
            task = ""
        return x, x_cond, task

if __name__ == "__main__":
    dataset = MCdataset(
        mode='raw', 
        dataset_dirs=['/nfs-shared-2/data/contractors/dataset_7xx'], 
        enable_video=True,
        enable_action=False,
        frame_width=128,  # 640
        frame_height=128, # 360
        win_len=8, 
        skip=1,
        split='train', 
        split_ratio=0.9, 
        verbose=True,
        enable_contractor_info=True, # condition
    )
    print("dataset_len",len(dataset))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for item in dataloader:
        x, x_cond, task = item
        print(x.shape, x_cond.shape, task)
        x = x[0]
        x = rearrange(x, '(f c) h w -> f h w c', f=7).numpy() * 255
        import cv2
        import numpy as np
        all_x = np.hstack(x)
        # import ipdb; ipdb.set_trace()
        output_file = "tmp.jpg"
        all_x = cv2.cvtColor(all_x, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_file, all_x)
        import ipdb; ipdb.set_trace()

