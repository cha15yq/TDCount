from utils.regression_trainer import Reg_Trainer
import argparse
import os
import torch
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--content', default='depth_thermal_simi',
                        help='Exp name')
    parser.add_argument('--data-dir', default="RGBT-CC",
                        help='training data directory')
    parser.add_argument('--save-dir', default="history",
                        help='directory to save models.')
    parser.add_argument('--seed', default=15, type=int,
                        help='seed number')

    parser.add_argument('--lr', type=float, default=5e-5,
                        help='the initial learning rate')
    parser.add_argument('--pretrained-model-name', default="runwayml/stable-diffusion-v1-5",
                        help='the pretrained name of the diffusion model')
    parser.add_argument('--control-net-path', default="outputs/trained/checkpoint-8500/controlnet",
                        help='the path to the pretrained ControlNet')
    parser.add_argument('--lcm-lora-path', default="pytorch_lora_weights.safetensors",
                        help='the path to the pretrained LCM lora')

    parser.add_argument('--num-steps', default=1, type=int,
                        help='the number of inference steps to the middle part of the diffusion process')
    parser.add_argument('--weight-decay', type=float, default=5e-5,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--density-unet-path', default='',
                        help='the path to the pretrained density lora, Must if resume')
    parser.add_argument('--max-num', type=int, default=2,
                        help='max models num to save ')
    parser.add_argument('--epochs', type=int, default=400,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='the num of steps to log training information')
    parser.add_argument('--start-val', type=int, default=20,
                        help='the epoch start to val')

    parser.add_argument('--batch-size', type=int, default=4,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--crop-size', type=int, default=384,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = Reg_Trainer(args)
    trainer.setup()
    trainer.train()