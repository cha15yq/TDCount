from models.model import vgg19
import argparse
import torch
from datasets.dataset import Crowd
from torch.utils.data import DataLoader
import random
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms
from utils.helper import extract_patches, reassemble_patches
from utils.regression_trainer import setup_seed


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


depth_transform = transforms.Compose([
    transforms.ToTensor(),
])

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.452016860247, 0.447249650955, 0.431981861591],
                                 [0.23242045939, 0.224925786257, 0.221840232611])
        ])


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--pretrained-model-name', default="runwayml/stable-diffusion-v1-5",
                        help='the pretrained name of the diffusion model')
    parser.add_argument('--control-net-path', default="outputs/checkpoint-24500/controlnet",
                        help='the path to the pretrained ControlNet')
    parser.add_argument('--lcm-lora-path', default="pytorch_lora_weights.safetensors",
                        help='the path to the pretrained LCM lora')
    parser.add_argument('--data-dir', default="RGBT-CC",
                        help='training data directory')

    parser.add_argument('--num-steps', default=1, type=int,
                        help='the number of inference steps to the middle part of the diffusion process')
    parser.add_argument('--pretrained', default='best_model_148/best_model_148.pth',
                        help='the path of resume training model')
    parser.add_argument('--density-unet-path', default='best_model_148/unet_148',
                        help='the path to the pretrained density lora, Must if resume')

    parser.add_argument('--batch-size', type=int, default=4,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')

    parser.add_argument('--crop-size', type=int, default=384,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=8,
                        help='downsample ratio')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda')
    model = vgg19(args)
    setup_seed(15)
    model.load_state_dict(torch.load(args.pretrained))

    model.to(device)
    model.depth_feat.pipeline.to(device)
    model.eval()
    model.depth_feat.setup_eval()

    data = Crowd(args, method='test')
    dataloader = DataLoader(data, batch_size=1, shuffle=False)

    file = open('result2.txt', 'w')
    mae = []
    for thermal_inputs, thermal_d_inputs, target, name, token_ids in dataloader:
        thermal_inputs = thermal_inputs.to(device)
        thermal_d_inputs = thermal_d_inputs.to(device)
        target = target.to(device)
        token_ids = token_ids.to(device)
        cropped_t, num_h, num_w = extract_patches(thermal_inputs, patch_size=384, stride=384)
        cropped_t_d, num_h, num_w = extract_patches(thermal_d_inputs, patch_size=384, stride=384)
        outputs = []
        outputs_seg = []
        with torch.set_grad_enabled(False):
            num_chunks = (cropped_t.size(0) + args.batch_size - 1) // args.batch_size
            for i in range(num_chunks):
                start_idx = i * args.batch_size
                end_idx = min((i + 1) * args.batch_size, cropped_t.size(0))
                outputs_partial, outputs_seg_partial = model(cropped_t[start_idx:end_idx],
                                                             cropped_t_d[start_idx:end_idx],
                                                             token_ids.repeat((end_idx - start_idx), 1, 1))

                outputs.append(outputs_partial)
                outputs_seg.append(
                    outputs_seg_partial.transpose(1, 2).reshape(outputs_seg_partial.shape[0], 6, 48, 48).float())
            pred_den = reassemble_patches(torch.cat(outputs, dim=0), num_h, num_w, thermal_inputs.size(2),
                                          thermal_inputs.size(3),
                                          patch_size=args.crop_size, stride=args.crop_size)
            import matplotlib.pyplot as plt
            mae.append(abs(torch.sum(pred_den).item() - target.sum().item()))
            import numpy as np
            print(pred_den.sum(), target.sum(), name, np.array(mae).mean())
            file.write('{} Pred: {} GT: {} Diff:{}\n'.format(name[0], torch.sum(pred_den).item(), target.sum().item(), abs(torch.sum(pred_den).item() - target.sum().item())))
