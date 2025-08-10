import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from timm.models.layers import trunc_normal_
import copy
from models.depth_model import Depth2RGB
from transformers import AutoTokenizer
import numpy as np
from models.cross_attn import CrossAttentionTransformer


model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}



class VGG(nn.Module):
    def __init__(self, features, args):
        super(VGG, self).__init__()
        self.features = features
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(nn.Conv2d(512, 512, 1, bias=False))
        self.conv2 = nn.Sequential(nn.Conv2d(320, 512, 1, bias=False),
                                   nn.ReLU(inplace=True))
        self.reg_head = nn.Sequential(nn.Conv2d(1024, 512, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, 256, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, 128, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 1, 1, 1),
                                      nn.ReLU())
        self.text_projection = nn.Linear(768, 512, bias=False)
        self.x2y = CrossAttentionTransformer(embed_dim=512, depth=1, num_heads=8)
        self.y2x = CrossAttentionTransformer(embed_dim=512, depth=1, num_heads=8)
        self.apply(self._init_weights)
        self.depth_feat = Depth2RGB(pretrained_model_name=args.pretrained_model_name,
                                    control_net_path=args.control_net_path,
                                    lcm_lora_path=args.lcm_lora_path,
                                    weight_dtype=torch.float32,
                                    num_steps=args.num_steps,
                                    density_unet_path=args.density_unet_path)

        latents = torch.randn((1, self.depth_feat.pipeline.unet.in_channels, args.crop_size // 8, args.crop_size // 8))
        self.register_buffer('diffusion_constant', latents)
        prompt_embeds = self.get_prompt_embeds()
        self.register_buffer('prompt_embeds', prompt_embeds)  # B, 768
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward(self, thermal, thermal_depth, token_ids):
        thermal_feat = self.features(thermal)  # B, 512, 24, 24
        thermal_depth_feat = self.depth_feat(thermal_depth, token_ids, self.diffusion_constant)  # B, 320, 48, 48
        thermal_feat = self.upsample(thermal_feat)  # B, 512, 48, 48
        thermal_depth_feat = self.conv2(thermal_depth_feat)

        text_feat = self.text_projection(self.prompt_embeds)  # B_t, 512
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
        image_feat = self.conv(thermal_feat).flatten(2).transpose(1, 2)  # B, HW, 512
        image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)

        sim_score = self.logit_scale.exp().clamp(max=100.0) * torch.matmul(image_feat, text_feat.transpose(0, 1))  # B, HW, B_t

        enhanced_thermal = self.x2y(thermal_feat, thermal_depth_feat)
        enhanced_thermal_depth = self.y2x(thermal_depth_feat, thermal_feat)

        fused_feat = torch.concat([enhanced_thermal, enhanced_thermal_depth], dim=1)
        pred_den = self.reg_head(fused_feat)
        return pred_den, sim_score

    def get_prompt_embeds(self):
        prompt = [
            "There is no one in the area.",
            "There is one person in the area.",
            "There are two people in the area.",
            "There are three people in the area.",
            "There are four people in the area.",
            "There are at least five people in the area.",
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="tokenizer",
            use_fast=False,
        )
        input_ids = tokenizer(prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                              return_tensors="pt").input_ids
        eos_token_id = tokenizer.eos_token_id
        first_eos = (input_ids == eos_token_id).float().masked_fill(~(input_ids == eos_token_id), float('inf')).argmin(
            dim=1)
        prompt_embed = self.depth_feat.pipeline.text_encoder(input_ids, return_dict=False)[0]
        del tokenizer
        return prompt_embed[torch.arange(len(prompt)), first_eos]


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}


def vgg19(args):
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), args)
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

