import torch
from torch.utils.data import Dataset
from glob import glob
import os
import numpy as np
import torchvision.transforms.functional as F
import random
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


class Crowd(Dataset):
    def __init__(self, args, method='train'):
        super(Crowd, self).__init__()
        self.im_list = sorted(glob(os.path.join(args.data_dir, method, 'T/*.jpg')))
        self.root = os.path.join(args.data_dir, method)
        assert args.crop_size % args.downsample_ratio == 0
        assert method in ['train', 'val', 'test'], f"Invalid method: {method}. Must be 'train', 'val', or 'test'."
        self.crop_size = args.crop_size
        self.down_ratio = args.downsample_ratio
        self.method = method
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name,
            subfolder="tokenizer",
            use_fast=False,
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.452016860247, 0.447249650955, 0.431981861591],
                                 [0.23242045939, 0.224925786257, 0.221840232611])
        ])
        self.depth_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        thermal_path = self.im_list[item]
        thermal_d_path = os.path.join(self.root, 'depth_t', os.path.basename(thermal_path))

        thermal = Image.open(thermal_path).convert('RGB')
        thermal_d = Image.open(thermal_d_path).convert('RGB')

        points_path = os.path.join(self.root, 'points', os.path.basename(thermal_path).replace('jpg', 'npy'))
        points = np.load(points_path)
        pt_map = np.zeros((thermal.size[1], thermal.size[0]), dtype=np.float32)
        for pt in points:
            pt_map[min(int(pt[1]), thermal.size[1] - 1), min(int(pt[0]), thermal.size[0] - 1)] = 1
        if self.method == 'train':
            den_path = os.path.join(self.root, 'density_map', os.path.basename(thermal_path).replace('jpg', 'npy'))
            den_map = np.load(den_path)
            thermal, thermal_d, den_map, token_ids, pt_map = self.train_transform_density(thermal, thermal_d, den_map, pt_map)
            return thermal, thermal_d, den_map, token_ids, pt_map
        else:
            caption = 'a photo of a crowd of people'
            return self.transform(thermal), self.depth_transform(thermal_d), pt_map, os.path.basename(thermal_path).split('_')[0], \
                   self.tokenizer(caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

    def train_transform_density(self, thermal, thermal_d, den_map, pt_map):
        wd, ht = thermal.size

        i, j, h, w = random_crop(ht, wd, self.crop_size, self.crop_size)
        thermal = F.crop(thermal, i, j, h, w)
        thermal_d = F.crop(thermal_d, i, j, h, w)
        den_map = den_map[i: (i + h), j: (j + w)]
        den_map = den_map.reshape(
            [h // self.down_ratio, self.down_ratio, w // self.down_ratio, self.down_ratio]).sum(
            axis=(1, 3))
        pt_map = pt_map[i: (i + h), j: (j + w)]
        pt_map = pt_map.reshape([h // self.down_ratio, self.down_ratio, w // self.down_ratio, self.down_ratio]).sum(
            axis=(1, 3))

        if random.random() > 0.5:
            thermal = F.hflip(thermal)
            thermal_d = F.hflip(thermal_d)
            den_map = np.fliplr(den_map)
            pt_map = np.fliplr(pt_map)

        if random.random() > 0.5:
            caption = 'a photo of a crowd of people'
        else:
            caption = ''
        pt_map[(pt_map >= 5)] = 5
        return self.transform(thermal), self.depth_transform(thermal_d), torch.from_numpy(den_map.copy()).float().unsqueeze(0), \
               self.tokenizer(caption, max_length=self.tokenizer.model_max_length, padding="max_length",
                              truncation=True, return_tensors="pt").input_ids, torch.from_numpy(pt_map.copy()).long()