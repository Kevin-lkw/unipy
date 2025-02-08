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

def create_action_image(action_list, num_frames):
    action_image = np.zeros((256, 128*num_frames, 3), dtype=np.uint8)  # 创建一个空的图像行
    for i in range(num_frames-1):
        for j,item in enumerate(action_list[i].items()):
            key,value = item
            action_text = f"{key}: {value}"  # 将动作信息转换为字符串
            cv2.putText(action_image, action_text, (128*(i+1), 10 + 12*j), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return action_image

"""
save result as image
obs_list: list of np.array (h,w,c) 
predicted_frames: list of np.array (h,w,c)
action_list: env_action, list of dict , dict:{action_name: action_value}
"""     
def save_image(obs_list, predicted_frames, action_list, output_path, gt_action=None):
    # 将图像按列拼接
    top_row = np.hstack(predicted_frames)  
    action_image = create_action_image(action_list, len(predicted_frames))
    bottom_row = np.hstack(obs_list)  
    images = np.vstack((top_row, action_image, bottom_row)) 

    if gt_action:
        action_image = create_action_image(gt_action, len(predicted_frames))
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
def save_video(obs_list, predicted_frames, action_list, output_path, resolution):
    output_path = f"{output_path}/video.mp4"
    h,w = resolution
    output_container = av.open(output_path, mode='w')
    stream = output_container.add_stream('libx264')
    stream.width = w
    stream.height = h
    stream.pix_fmt = 'yuv420p'

    # 遍历并写入帧
    for gt_frame, pred_frame in zip(obs_list, predicted_frames):
        separator = np.zeros((10, w, 3), dtype=np.uint8)  # Black separator
        stacked_frame = np.concatenate([gt_frame, separator, pred_frame],axis=0)  # 将两张图像拼接成一张
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
    for chunk in range(1):
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
    
    # save image
    save_image(obs_list, predicted_frames, action_list, output_path)
    # save_video(obs_list, predicted_frames, action_list, output_path, args.resolution)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, default="./AVDC/results/mc/02-01-22-54-16/model-11.pt") # set to checkpoint number to resume training or generate samples
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
    args = parser.parse_args()
    args.resolution = tuple(map(int, args.resolution.split(',')))
    main(args)
    # obs_list = [np.random.randint(0,255,(128,128,3), dtype=np.uint8) for _ in range(10)]
    # predicted_frames = [np.random.randint(0,255,(128,128,3), dtype=np.uint8) for _ in range(10)]
    # action_list = [{"action1":0,"action2":0} for _ in range(10)]
    # output_path = "./output/tmp"
    # os.makedirs(output_path, exist_ok=True)
    # save_image(obs_list, predicted_frames, action_list, output_path, gt_action=action_list)