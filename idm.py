import torch
import torch.nn as nn
from einops import rearrange
import argparse
from tqdm import tqdm
from datetime import datetime
import cv2
import av
import numpy as np
import os
from unipy import Unipy, save_image
from minestudio.data import load_dataset
from torch.utils.data import DataLoader

def main(args):
    # or load the policy from the Hugging Face model hub
    policy = Unipy(
        checkpoint_path=args.checkpoint_path,
    )
    dataset = load_dataset(
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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    pred_frame_list = []
    tot_action_list = []
    gt_frame_list = []
    data_iter = iter(dataloader)
    for id in range(args.num):
        x = next(data_iter)
        x = x['image'][0].numpy() # from batch=1,shape (f,h,w,c)
        x_cond = x[0,:] # shape (1,h,w)
        actions, pred_frames = policy.get_action(x_cond, [x_cond])
        pred_frame_list.append(pred_frames)
        cur_action_list = []
        for i in range(args.frames - 1):
            action = {}
            for k,v in actions.items():
                action[k] = v[0][i]
            cur_action_list.append(action)
        tot_action_list.append(cur_action_list)
        gt_frame_list.append(x)
    dt = datetime.now().strftime('%m%d-%H%M%S')
    output_path = f"./rollout/{dt}"
    for i in range(args.num):
        cur_output_path = os.path.join(output_path, str(i))
        os.makedirs(cur_output_path, exist_ok=True)
        save_image(gt_frame_list[i], pred_frame_list[i], tot_action_list[i], cur_output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, default="./AVDC/results/mc/02-01-22-07-00/model-13.pt") # set to checkpoint number to resume training or generate samples
    parser.add_argument('-t', '--text', type=str, default=None) # set to text to generate samples
    parser.add_argument('-n', '--sample_steps', type=int, default=100) # set to number of steps to sample
    parser.add_argument('-b', '--batch_size', type=int, default=2) # set to batch size
    parser.add_argument('-l','--learning_rate', type=float, default=1e-4) # set to learning rate
    parser.add_argument('-cond', '--condition', action='store_true') # set to True to use condition
    parser.add_argument('-log', '--log', action='store_true') # set to True to use wandb
    parser.add_argument('-valid_n', '--valid_n', type=int, default=4) # set to number of validation samples
    parser.add_argument('-f', '--frames', type=int, default=8) # set to number of samples per sequence
    parser.add_argument('-prci', '--precision', type=str, default='fp16') # set to True to use mixed precision
    parser.add_argument('-r', '--resolution', type=str, default="128,128") # set to resolution
    parser.add_argument('-num', '--num', type=int, default=3) # set to number of samples
    args = parser.parse_args()
    args.resolution = tuple(map(int, args.resolution.split(',')))
    main(args)