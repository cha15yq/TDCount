from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler, UNet2DConditionModel
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
import gc


class Depth2RGB(nn.Module):
    def __init__(self, pretrained_model_name, control_net_path, lcm_lora_path, weight_dtype, num_steps,
                 density_unet_path=None):
        super(Depth2RGB, self).__init__()

        controlnet = ControlNetModel.from_pretrained(control_net_path, torch_dtype=weight_dtype)
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    pretrained_model_name,
                    controlnet=controlnet,
                    torch_dtype=weight_dtype,
            )

        self.pipeline.load_lora_weights(lcm_lora_path)
        self.pipeline.fuse_lora()
        self.pipeline.unload_lora_weights()

        self.num_steps = num_steps
        self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.scheduler.set_timesteps(7)
        self.freeze_pipeline()
        self.pipeline.unet.requires_grad_(True)

        if not density_unet_path:
            lora_target_modules = [
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "conv1",
                "conv2",
                "conv_shortcut",
                "downsamplers.0.conv",
                "upsamplers.0.conv",
                "time_emb_proj",
            ]
            lora_config = LoraConfig(
                r=64,
                target_modules=lora_target_modules,
                lora_alpha=64,
            )
            self.pipeline.unet = get_peft_model(self.pipeline.unet, lora_config)
        else:
            self.pipeline.unet = PeftModel.from_pretrained(self.pipeline.unet, density_unet_path)

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

        return captured_features["features"] #self.pipeline.vae.decode(1 / 0.18215 * latents).sample

    def to(self, device):
        super().to(device)
        self.pipeline = self.pipeline.to(device)
        return self


