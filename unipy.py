import torch
import torch.nn as nn
from minestudio.models.base_policy import MinePolicy
from einops import rearrange
import argparse

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback
from AVDC.flowdiffusion.unet import UnetThor as Unet
from transformers import CLIPTextModel, CLIPTokenizer
from AVDC.flowdiffusion.goal_diffusion import GoalGaussianDiffusion, Trainer
from accelerate import Accelerator
import pickle
from vpt.inverse_dynamics_model import IDMAgent
from tqdm import tqdm
from datetime import datetime
import imageio
import cv2
import av
import numpy as np
import os
class Unipy():
    def __init__(self, args):
        self.frames = args.frames
        self.target_size = args.resolution
        unet = Unet()
        if args.condition:
            pretrained_model = "openai/clip-vit-base-patch32"
            tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
            text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
            text_encoder.requires_grad_(False)
            text_encoder.eval()
        else:
            tokenizer = None
            text_encoder = None
        accelerator = Accelerator(
            split_batches = True,
            mixed_precision = args.precision,
        )
        diffusion = GoalGaussianDiffusion(
            channels=3*(self.frames-1),
            model=unet,
            image_size=(self.target_size[1], self.target_size[0]), # height=imgsz[0], width = imgsz[1]
            timesteps=args.sample_steps, # difussion steps, like 100
            sampling_timesteps=args.sample_steps,
            loss_type='l2',
            objective='pred_v',
            beta_schedule = 'cosine',
            min_snr_loss_weight = True,
        )
        self.trainer = Trainer(
            diffusion_model=diffusion,
            tokenizer=tokenizer, 
            text_encoder=text_encoder,
            accelerator=accelerator,
            rollout_mode=True,
            cond = args.condition,
        )
        self.trainer.load_model(args.checkpoint_path)
        agent_parameters = pickle.load(open("/nfs-shared/liankewei/vptidm/4x_idm.model", "rb"))
        net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
        self.IDMAgent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs)
        self.IDMAgent.load_weights("/nfs-shared/liankewei/vptidm/4x_idm.weights")

    """
    obs: np.array(h,w,c)
    predicted_frames: list of np.array(h,w,c)
    """
    def get_action(self, obs, predicted_frames): 
        obs = rearrange(obs, 'h w c -> c h w') 
        obs = torch.from_numpy(obs).float()
        batched = obs.unsqueeze(0) # add batch dimension [1,c,h,w]
        frames = self.trainer.sample(x_conds = batched, batch_text = None, batch_size = 1)
        frames = frames.reshape(-1,3, *self.target_size) # [frames,3,h,w]
        frames = rearrange(frames, 'n c h w -> n h w c').cpu().numpy()
        for i in range(frames.shape[0]):
            predicted_frames.append(frames[i])
        predicted_actions = self.IDMAgent.predict_actions(frames)
        return predicted_actions, predicted_frames
    
def main(args):
    # or load the policy from the Hugging Face model hub
    policy = Unipy(args=args)
    env = MinecraftSim(
        obs_size=(128, 128), 
        callbacks=[RecordCallback(record_path="./output", fps=30, frame_type="pov")]
    )
    obs, info = env.reset() # obs (h,w,c)

    start = obs['image']
    obs_list = [start]
    predicted_frames = [start]
    action_list = []
    # actual number of frames is num_steps * frames(8)
    for _ in range(1):
        actions, predicted_frames = policy.get_action(start, predicted_frames)
        action_list.append(actions)
        print("actions",len(actions))
        for i in range(args.frames - 1):
            action = {}
            for k,v in actions.items():
                action[k] = np.array(v[0][i]).expand_dims(0)
            import ipdb; ipdb.set_trace()
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(np.random.rand(128,128,3))
        start = obs_list[-1]
    # env.close()
    print("Rollout finished")
    assert len(obs_list) == len(predicted_frames)
    dt = datetime.now().strftime('%m%d-%H%M%S')
    output_path = f"./rollout"
    os.makedirs(output_path, exist_ok=True)
    output_path = f"{output_path}/rollout_{dt}.mp4"
    h,w = args.resolution
    output_container = av.open(output_path, mode='w')
    stream = output_container.add_stream('libx264')
    stream.width = w
    stream.height = h
    stream.pix_fmt = 'yuv420p'

    # 遍历并写入帧
    for gt_frame, pred_frame in zip(obs_list, predicted_frames):
        separator = np.zeros((10, w, 3), dtype=np.uint8)  # Black separator
        stacked_frame = np.concatenate([gt_frame, separator, pred_frame],axis=0)  # 将两张图像拼接成一张
        import ipdb; ipdb.set_trace()
        stacked_frame = (stacked_frame * 255).astype(np.uint8)
        # 将numpy数组转换为av的图像帧
        # 注意：av要求图像为YUV格式，需要转换为yuv420p
        frame = av.VideoFrame.from_ndarray(stacked_frame, format='rgb24')

        # 编码并写入视频流
        for packet in stream.encode(frame):
            output_container.mux(packet)

    for packet in stream.encode():
        output_container.mux(packet)

    output_container.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, default="./AVDC/results/mc/02-01-22-07-00/model-13.pt") # set to checkpoint number to resume training or generate samples
    parser.add_argument('-t', '--text', type=str, default=None) # set to text to generate samples
    parser.add_argument('-n', '--sample_steps', type=int, default=100) # set to number of steps to sample
    parser.add_argument('-g', '--guidance_weight', type=int, default=0) # set to positive to use guidance
    parser.add_argument('-b', '--batch_size', type=int, default=1) # set to batch size
    parser.add_argument('-l','--learning_rate', type=float, default=1e-4) # set to learning rate
    parser.add_argument('-cond', '--condition', action='store_true') # set to True to use condition
    parser.add_argument('-log', '--log', action='store_true') # set to True to use wandb
    parser.add_argument('-valid_n', '--valid_n', type=int, default=4) # set to number of validation samples
    parser.add_argument('-f', '--frames', type=int, default=8) # set to number of samples per sequence
    parser.add_argument('-prci', '--precision', type=str, default='fp16') # set to True to use mixed precision
    parser.add_argument('-r', '--resolution', type=str, default="128,128") # set to resolution
    args = parser.parse_args()
    args.resolution = tuple(map(int, args.resolution.split(',')))
    main(args)