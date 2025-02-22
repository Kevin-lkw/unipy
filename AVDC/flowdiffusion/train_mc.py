import argparse
from goal_diffusion import GoalGaussianDiffusion, Trainer
from unet import UnetThor as Unet
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Subset
import wandb
from accelerate import Accelerator
from datetime import datetime
from .mc_dataset import MCdataset

def main(args):

    accelerator = Accelerator(
        split_batches = True,
        mixed_precision = args.precision,
    )
    accelerator.native_amp = True # it seems that this is necessary ?
    wandbname = None
    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    if args.log and accelerator.is_main_process:
        wandb.init(
            project="mc-unipy",  # 项目名称
            config={
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "num_steps": args.num_steps,
                "sample_steps": args.sample_steps,
                "guidance_weight": args.guidance_weight,
                "dataset": args.dataset,
                "condition": args.condition,
                "valid_n": args.valid_n,
                "frames": args.frames, # samples per sequence
                "skip": args.skip,
                "precision": args.precision,
                "resolution": args.resolution,
            },
            name=f"{args.dataset}-bs{args.batch_size}-skip{args.skip}-{timestamp}",
        )
        wandbname = wandb.run.name

    valid_n = args.valid_n
    sample_per_seq = args.frames
    target_size = args.resolution
    train_set = MCdataset(
        mode='raw', 
        dataset_dirs=[f'/nfs-shared-2/data/contractors/dataset_{args.dataset}'], 
        enable_video=True,
        enable_action=False,
        frame_width=target_size[0],  # 640
        frame_height=target_size[1], # 360
        win_len=sample_per_seq, 
        skip=args.skip,
        split='train', 
        split_ratio=0.9, 
        verbose=True,
        # event_regex='minecraft.kill_entity:.*',
        enable_contractor_info=args.condition, # add text data 
    )
    valid_inds = [i for i in range(0, len(train_set), len(train_set)//valid_n)][:valid_n]
    valid_set = Subset(train_set, valid_inds)
    
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
    diffusion = GoalGaussianDiffusion(
        channels=3*(sample_per_seq-1),
        model=unet,
        image_size=(target_size[1], target_size[0]), # height=imgsz[0], width = imgsz[1]
        timesteps=args.sample_steps, # difussion steps, like 100
        sampling_timesteps=args.sample_steps, # can be less than timesteps (DDIM sampling)
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
    )
    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=train_set,
        valid_set=valid_set,
        train_lr=args.learning_rate,
        train_num_steps=args.num_steps, # training steps, like 80000
        save_and_sample_every = args.save_steps,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size = args.batch_size,
        valid_batch_size = args.batch_size,
        gradient_accumulate_every = 1,
        num_samples=valid_n, 
        wandb_name = wandbname,
        results_folder =f'../results/mc/{timestamp}',
        accelerator = accelerator,
        cond = args.condition,
        log = args.log,
    )

    if args.checkpoint_num is not None:
        trainer.load(args.checkpoint_num)
    
    if args.mode == 'train':
        trainer.train()
    else:
        raise NotImplementedError
        # from PIL import Image
        # from torchvision import transforms
        # import imageio
        # import torch
        # from os.path import splitext
        # text = args.text
        # image = Image.open(args.inference_path)
        # batch_size = 1
        # transform = transforms.Compose([
        #     transforms.Resize(target_size),
        #     transforms.ToTensor(),
        # ])
        # image = transform(image)
        # output = trainer.sample(image.unsqueeze(0), [text], batch_size).cpu()
        # output = output[0].reshape(-1, 3, *target_size)
        # output = torch.cat([image.unsqueeze(0), output], dim=0)
        # root, ext = splitext(args.inference_path)
        # output_gif = root + '_out.gif'
        # output = (output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8')
        # imageio.mimsave(output_gif, output, duration=200, loop=1000)
        # print(f'Generated {output_gif}')
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'inference']) # set to 'inference' to generate samples
    parser.add_argument('-c', '--checkpoint_num', type=int, default=None) # set to checkpoint number to resume training or generate samples
    parser.add_argument('-p', '--inference_path', type=str, default=None) # set to path to generate samples
    parser.add_argument('-t', '--text', type=str, default=None) # set to text to generate samples
    parser.add_argument('-s', '--num_steps', type=int, default=80000) # set to number of steps to train
    parser.add_argument('-n', '--sample_steps', type=int, default=100) # set to number of steps to sample
    parser.add_argument('-g', '--guidance_weight', type=int, default=0) # set to positive to use guidance
    parser.add_argument('-b', '--batch_size', type=int, default=2) # set to batch size
    parser.add_argument('-l','--learning_rate', type=float, default=1e-4) # set to learning rate
    parser.add_argument('-ss', '--save_steps', type=int, default=5000) # set to number of steps to save and sample
    parser.add_argument('-cond', '--condition', action='store_true') # set to True to use condition
    parser.add_argument('-d', '--dataset', type=str, default='10xx') # set to dataset
    parser.add_argument('-log', '--log', action='store_true') # set to True to use wandb
    parser.add_argument('-valid_n', '--valid_n', type=int, default=4) # set to number of validation samples
    parser.add_argument('-f', '--frames', type=int, default=8) # set to number of samples per sequence
    parser.add_argument('-prci', '--precision', type=str, default='fp16') # set to True to use mixed precision
    parser.add_argument('-r', '--resolution', type=str, default="128,128") # set to resolution
    parser.add_argument('-skip', '--skip', type=int, default=1) # enable more information in training: sample (skip*args) from dataset  
    args = parser.parse_args()
    args.resolution = tuple(map(int, args.resolution.split(',')))
    if args.mode == 'inference':
        assert args.checkpoint_num is not None
        assert args.inference_path is not None
        assert args.text is not None
        assert args.sample_steps <= 100
    main(args)