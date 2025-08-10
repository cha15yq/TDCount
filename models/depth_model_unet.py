from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler, UNet2DConditionModel
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import gc


class Depth2RGB(nn.Module):
    def __init__(self, pretrained_model_name, control_net_path, lcm_lora_path, weight_dtype, num_steps,
                 density_unet_path=None):
        super(Depth2RGB, self).__init__()
        unet = UNet2DConditionModel.from_pretrained(density_unet_path if density_unet_path else pretrained_model_name,
                                                    subfolder=None if density_unet_path else "unet")
        controlnet = ControlNetModel.from_pretrained(control_net_path, torch_dtype=weight_dtype)
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    pretrained_model_name,
                    controlnet=controlnet,
                    unet=unet,
                    torch_dtype=weight_dtype,
            )

        if density_unet_path is None:
            self.pipeline.load_lora_weights(lcm_lora_path)
            self.pipeline.fuse_lora()
            self.pipeline.unload_lora_weights()

        self.num_steps = num_steps
        self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.scheduler.set_timesteps(7)
        self.freeze_pipeline()
        self.pipeline.unet.requires_grad_(True)

        # if density_lora_path is None:
        #     lora_target_modules = [
        #         "to_q",
        #         "to_k",
        #         "to_v",
        #         "to_out.0",
        #         "proj_in",
        #         "proj_out",
        #         "ff.net.0.proj",
        #         "ff.net.2",
        #         "conv1",
        #         "conv2",
        #         "conv_shortcut",
        #         "downsamplers.0.conv",
        #         "upsamplers.0.conv",
        #         "time_emb_proj",
        #     ]
        #     lora_config = LoraConfig(
        #         r=64,
        #         target_modules=lora_target_modules,
        #         lora_alpha=64,
        #     )
        #     self.pipeline.unet = get_peft_model(self.pipeline.unet, lora_config)
        # else:
        #     self.pipeline.unet.load_lora_weights(density_lora_path)

    def freeze_pipeline(self):
        for module_name in ["unet", "vae", "text_encoder", "controlnet"]:
            module = getattr(self.pipeline, module_name, None)
            if module is not None:
                module.requires_grad_(False)

    def setup_train(self):
        self.pipeline.unet.train()
        self.pipeline.controlnet.eval()
        self.pipeline.text_encoder.eval()
        self.pipeline.vae.eval()

    def setup_eval(self):
        self.pipeline.unet.eval()
        self.pipeline.controlnet.eval()
        self.pipeline.text_encoder.eval()
        self.pipeline.vae.eval()

    def forward(self, x, text_ids, latents):
        B, C, H, W = x.shape

        latents = latents.repeat(B, 1, 1, 1)
        latents = latents * self.pipeline.scheduler.init_noise_sigma
        captured_features = {}

        encoder_hidden_states = self.pipeline.text_encoder(text_ids, return_dict=False)[0]
        def hook_fn(module, input, output):
            captured_features["features"] = output

        for i in range(0, self.num_steps):
            curr_t = self.pipeline.scheduler.timesteps[i]
            if i == (self.num_steps - 1):
                hook_handle = self.pipeline.unet.up_blocks[-1].register_forward_hook(hook_fn)
            down_block_res_samples, mid_block_res_sample = self.pipeline.controlnet(
                latents,
                curr_t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=x,
                return_dict=False,
            )

            noise_pred = self.pipeline.unet(
                latents,
                curr_t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

            latents = self.pipeline.scheduler.step(noise_pred, curr_t, latents).prev_sample
        hook_handle.remove()

        return captured_features["features"] #self.pipeline.vae.decode(1 / 0.18215 * latents).sample#captured_features["features"]

    def to(self, device):
        super().to(device)
        self.pipeline = self.pipeline.to(device)
        return self


if __name__ == '__main__':
    model = Depth2RGB(pretrained_model_name="runwayml/stable-diffusion-v1-5",
                      control_net_path='/home/yq1/local_raid/yq1/rgbt_cc/outputs/trained/checkpoint-8500/controlnet',
                      lcm_lora_path='/home/yq1/local_raid/yq1/rgbt_cc/pytorch_lora_weights.safetensors',
                      weight_dtype=torch.float32,
                      num_steps=1,
                      density_unet_path=None).to('cuda')
    for name, param in model.pipeline.unet.named_parameters():
        if param.requires_grad:
            print(f'{name}, shape:{param.shape}')
    model.setup_eval()
    import random
    from PIL import Image
    import torchvision.transforms.functional as F
    from torchvision import transforms

    def random_crop(im_h, im_w, crop_h, crop_w):
        res_h = im_h - crop_h
        res_w = im_w - crop_w
        i = random.randint(0, res_h)
        j = random.randint(0, res_w)
        return i, j, crop_h, crop_w

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    control_image = Image.open('/home/yq1/local_raid/yq1/rgbt_cc/RGBT-CC/test/depth_t/1571.jpg').convert('RGB')
    original_imgae = Image.open('/home/yq1/local_raid/yq1/rgbt_cc/RGBT-CC/test/RGB/1571.jpg').convert('RGB')
    w, h = control_image.size
    i, j, h, w = random_crop(h, w, 384, 384)
    pil_control_image = F.crop(control_image, i, j, h, w)
    original_image = F.crop(original_imgae, i, j, h, w)

    transformed_image = (transform(pil_control_image) / 255.0).unsqueeze(0).to('cuda')

    latents = torch.randn((1, model.pipeline.unet.in_channels, 384 // 8, 384 // 8)).to('cuda')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer",
        use_fast=False,
    )
    caption = 'a photo of a crowd of people'
    tokens = (tokenizer(caption, max_length=tokenizer.model_max_length, padding="max_length",
                                                 truncation=True, return_tensors="pt").input_ids).unsqueeze(0)
    # caption2 = ''
    # tokens2 = (tokenizer(caption2, max_length=tokenizer.model_max_length, padding="max_length",
    #                     truncation=True, return_tensors="pt").input_ids).unsqueeze(0)
    attention_maps = []
    def register_attention_hooks(model):
        def save_attention_hook(module, input, output):
            # output[1] 是 attention weights, shape: (B, num_heads, N, M)
            # N = image tokens (H*W), M = text tokens
            if isinstance(output, tuple) and len(output) > 1:
                attention_maps.append(output[1].detach().cpu())

        for name, module in model.named_modules():
            if module.__class__.__name__ == "CrossAttention":
                module.register_forward_hook(save_attention_hook)


    register_attention_hooks(model.pipeline.unet)

    with torch.no_grad():
        feat1 = model(transformed_image, tokens.to('cuda'), latents).detach().cpu()
        print(attention_maps)
        attn = attention_maps[0]  # shape: (B, heads, HxW, tokens)
        attn_avg = attn.mean(dim=1)[0]  # 平均多个 heads → (HxW, tokens)

        # 选一个词（比如 "dog"）的 index
        tokens = model.pipeline.tokenizer.tokenize(caption)
        token_index = tokens.index("crowd")

        # reshape 到空间分布
        token_attn_map = attn_avg[:, token_index].reshape(48, 48)
        # feat2 = model(transformed_image, tokens2.to('cuda'), latents).detach().cpu()
    # diff = feat1 - feat2  # shape: (1, C, H, W)
    # spatial_diff = torch.norm(diff, dim=1)
    #
    # spatial_diff_norm = (spatial_diff - spatial_diff.min()) / (spatial_diff.max() - spatial_diff.min())

    # 转为 numpy 再画图
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(5, 2))

    axs[0].imshow(original_image)
    axs[0].axis('off')  # 不显示坐标轴
    axs[0].set_title('RGB')

    im = axs[1].imshow(token_attn_map.cpu().numpy(), cmap='jet')
    axs[1].axis('off')
    axs[1].set_title("Feature Difference Heatmap")
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    plt.show()
    # image = (image / 2 + 0.5).clamp(0, 1)
    # image = image.cpu().permute(0, 2, 3, 1).numpy()
    # image = (image * 255).round().astype("uint8")
    # pil_image = Image.fromarray(image[0])
    # pil_image.show()


