import torch
import torch.nn as nn
from einops import rearrange
import argparse
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
import pickle
from datetime import datetime
import cv2
import av
import numpy as np
import os

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback
from AVDC.flowdiffusion.unet import UnetThor as Unet
from AVDC.flowdiffusion.goal_diffusion import GoalGaussianDiffusion, Trainer

from vpt.inverse_dynamics_model import IDMAgent
from minestudio.data.minecraft.utils import visualize_dataloader

class Unipy():
    def __init__(self,
            checkpoint_path,
            idm_model_path = "/nfs-shared/liankewei/vptidm/4x_idm.model",
            idm_weights_path = "/nfs-shared/liankewei/vptidm/4x_idm.weights",
            frames = 8,
            resolution = (128,128),         
            condition = False,
            precision = 'fp16',
            sample_steps = 100,
        ):
        self.frames = frames
        self.target_size = resolution
        unet = Unet()
        if condition:
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
            mixed_precision = precision,
        )
        diffusion = GoalGaussianDiffusion(
            channels=3*(self.frames-1),
            model=unet,
            image_size=(self.target_size[1], self.target_size[0]), # height=imgsz[0], width = imgsz[1]
            timesteps=sample_steps, # difussion steps, like 100
            sampling_timesteps=sample_steps, # this can be less than timesteps (DDIM sampling)
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
            cond = condition,
        )
        self.trainer.load_model(checkpoint_path)
        agent_parameters = pickle.load(open(idm_model_path, "rb"))
        net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
        self.IDMAgent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs)
        self.IDMAgent.load_weights(idm_weights_path)
    
    """
    IDM requirement:
    - frames: np.array (n,h,w,c)
    - RGB frames
    - [0,255]
    - best practice: [640,360]->[128,128] (I guess)
    """
    def predict_actions(self,frames):
        return self.IDMAgent.predict_actions(frames)
    
    """
    obs: np.array (h,w,c) \in [0,255]
    predicted_frames: list of np.array(h,w,c)
    """
    def get_action(self, obs, predicted_frames):
        obs = rearrange(obs, 'h w c -> c h w') 
        obs = torch.from_numpy(obs).float()
        batched = obs.unsqueeze(0) # add batch dimension [1,c,h,w]
        """
        [0-255] -> [0-1],for input of generative model
        """
        batched = batched / 255.0
        frames = self.trainer.sample(x_conds = batched, batch_text = None, batch_size = 1)
        frames = frames.reshape(-1,3, *self.target_size) # [frames,3,h,w]
        frames = rearrange(frames, 'n c h w -> n h w c').cpu().numpy()
        frames = frames * 255
        for i in range(frames.shape[0]):
            predicted_frames.append(frames[i])

        predicted_actions = self.IDMAgent.predict_actions(frames)
        return predicted_actions, predicted_frames
"""
create list of np.array (128,w,c) as image printed with action
"""
def create_action_frames(action_list, num_frames):
    action_frames = []
    action_image = np.zeros((256, 128, 3), dtype=np.uint8)  # 创建一个空的图像行
    action_frames.append(action_image)
    for i in range(num_frames-1):
        action_image = np.zeros((256, 128, 3), dtype=np.uint8)  # 创建一个空的图像行
        for j,item in enumerate(action_list[i].items()):
            key,value = item
            action_text = f"{key}: {value}"  # 将动作信息转换为字符串
            cv2.putText(action_image, action_text, (0, 10 + 12*j), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        action_frames.append(action_image)
    return action_frames

"""
save result as image
obs_list: list of np.array (h,w,c) 
predicted_frames: list of np.array (h,w,c)
action_list: env_action, list of dict , dict:{action_name: action_value}
"""     
def save_image(obs_list, predicted_frames, action_list, output_path, gt_action=None):
    # 将图像按列拼接
    top_row = np.hstack(predicted_frames)  
    action_frames = create_action_frames(action_list, len(predicted_frames))
    action_image = np.hstack(action_frames)
    bottom_row = np.hstack(obs_list)  
    images = np.vstack((top_row, action_image, bottom_row)) 

    if gt_action:
        action_frames = create_action_frames(gt_action, len(predicted_frames))
        action_image = np.hstack(action_frames)
        images = np.vstack((images, action_image))
    # 输出路径
    output_file = os.path.join(output_path, "image.jpg")
    images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)  # 转换为BGR格式
    # 保存图片
    cv2.imwrite(output_file, images)
    print(f"Image saved at {output_file}")
"""
same as save_image
"""
def save_video(obs_list, predicted_frames, action_list, output_path, gt_action=None):
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, "video.mp4")
    h,w = predicted_frames[0].shape[:2]
    with av.open(output_path, mode='w', format="mp4") as output_container:
        action_frames = create_action_frames(action_list, len(predicted_frames))
        # import ipdb; ipdb.set_trace()
        stream = output_container.add_stream('libx264',rate=30)
        stream.width = w*2
        stream.height = obs_list[0].shape[0] + predicted_frames[0].shape[0]
        stream.pix_fmt = 'yuv420p'
        for i in range(len(predicted_frames)):
            frame = np.concatenate((predicted_frames[i], obs_list[i]),axis=0)
            frame = np.concatenate((frame, action_frames[i]),axis=1)
            frame = frame.astype(np.uint8)
            # assert frame.shape[1] == stream.width and frame.shape[0] == stream.height, f"frame shape not match"
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            for packet in stream.encode(frame):
                output_container.mux(packet)
        for packet in stream.encode(): # important step...
            output_container.mux(packet)

    print(f"Video saved at {output_path}")
def main(args):
    # or load the policy from the Hugging Face model hub
    policy = Unipy(
        checkpoint_path=args.checkpoint_path,
        frames=args.frames,
        resolution=args.resolution,
        condition=args.condition,
        precision=args.precision,
        sample_steps=args.sample_steps
    )
    env = MinecraftSim(
        action_type="env",
        obs_size=(128, 128), 
        callbacks=[RecordCallback(record_path="./output", fps=30, frame_type="pov")]
    )
    obs, info = env.reset() # obs (h,w,c)

    start = obs['image']
    obs_list = [start]
    predicted_frames = [start]
    action_list = []
    # actual number of frames is num_steps * frames(8)
    for chunk in range(args.steps):
        actions, predicted_frames = policy.get_action(start, predicted_frames)
        for i in range(args.frames - 1):
            action = {}
            for k,v in actions.items():
                action[k] = v[0][i]
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs['image'])
            action_list.append(action)
        start = obs_list[-1]
    env.close()
    print("Rollout finished")
    assert len(obs_list) == len(predicted_frames)

    dt = datetime.now().strftime('%m%d-%H%M%S')
    output_path = f"./rollout/{dt}"
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, "log.txt")
    with open(log_file, "w") as f:
        #output dict args
        f.write(str(args))
    # save_image(obs_list, predicted_frames, action_list, output_path)
    save_video(obs_list, predicted_frames, action_list, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, default="./AVDC/results/mc/02-01-22-54-16/model-11.pt") # set to checkpoint number to resume training or generate samples
    parser.add_argument('-s', '--steps', type=int, default=1) # set to number of rollout steps, total steps = steps*8
    parser.add_argument('-t', '--text', type=str, default=None) # set to text to generate samples
    parser.add_argument('-n', '--sample_steps', type=int, default=100) # set to number of steps to sample
    parser.add_argument('-b', '--batch_size', type=int, default=2) # set to batch size
    parser.add_argument('-cond', '--condition', action='store_true') # set to True to use condition
    parser.add_argument('-log', '--log', action='store_true') # set to True to use wandb
    parser.add_argument('-valid_n', '--valid_n', type=int, default=4) # set to number of validation samples
    parser.add_argument('-f', '--frames', type=int, default=8) # set to number of samples per sequence
    parser.add_argument('-prci', '--precision', type=str, default='fp16') # set to True to use mixed precision
    parser.add_argument('-r', '--resolution', type=str, default="128,128") # set to resolution
    parser.add_argument('-steps', '--steps', type=int, default=1) # set to number of rollout steps, total steps = steps*8
    args = parser.parse_args()
    args.resolution = tuple(map(int, args.resolution.split(',')))
    main(args)

    # number = args.steps
    # obs_list = [np.random.randint(0,255,(128,128,3), dtype=np.uint8) for _ in range(number)]
    # predicted_frames = [np.random.randint(0,255,(128,128,3), dtype=np.uint8) for _ in range(number)]
    # action_list = [{'attack': 0, 'back': 0, 'forward': 1, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0, 'use': 0, 'drop': 0, 
    #                 'inventory': 0, 'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0, 'hotbar.5': 0, 
    #                 'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0, 'camera': [0., 0.]} for _ in range(number)]
    # output_path = "./output/tmp"
    # # os.makedirs(output_path, exist_ok=True)
    # save_video(obs_list, predicted_frames, action_list, output_path)