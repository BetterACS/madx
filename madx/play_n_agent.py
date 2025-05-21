from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.actor_critic import ActorCritic, ActorCriticConfig, ActorCriticLossConfig
from omegaconf import DictConfig, OmegaConf
import yaml
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm
from dataclasses import dataclass
from PIL import Image
from typing import Optional
from torchvision import transforms
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# from models.diffusion.inner_model import InnerModel, InnerModelConfig
from models.diffusion import DiffusionSampler, DiffusionSamplerConfig

# from models.diffusion.denoiser_nagent import Denoiser as DenoiserNagent

from pathlib import Path
import gymnasium
from envs.atari_preprocessing import AtariPreprocessing
from supersuit import frame_skip_v0, resize_v1, frame_stack_v1, reshape_v0

import pygame
import numpy as np
import time

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# from models.diffusion.inner_model import InnerModel, InnerModelConfig

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import Conv3x3, FourierFeatures, GroupNorm, UNet


from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor


@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None


class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.noise_emb = FourierFeatures(cfg.cond_channels)

        # self.act_emb = nn.Sequential(
        #     nn.Embedding(cfg.num_actions, cfg.cond_channels // cfg.num_steps_conditioning),
        #     nn.Flatten(),  # b t e -> b (t e)
        # )

        self.act_emb = nn.Sequential(
            nn.Embedding(cfg.num_actions, cfg.cond_channels // cfg.num_steps_conditioning),
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.cond_channels // cfg.num_steps_conditioning,
            nhead=4,
            dim_feedforward=cfg.cond_channels // cfg.num_steps_conditioning * 2,
        )

        self.player_attn = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.pool = nn.Linear(cfg.cond_channels, cfg.cond_channels)

        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        action_emb = self.act_emb(act)
        b, n_players, seq_len, emb_dim = action_emb.shape
        action_emb = action_emb.view(seq_len, b * n_players, emb_dim)
        action_emb = self.player_attn(action_emb)
        action_emb = action_emb.view(b, seq_len, n_players, emb_dim)  # .reshape(t, b, p, D).transpose(0, 1)   # [b, t, p, D]
        action_emb = action_emb.mean(dim=2)
        action_emb = action_emb.view(b, seq_len * emb_dim)
        action_emb = self.pool(action_emb)

        cond = self.cond_proj(self.noise_emb(c_noise) + action_emb)
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float


class DenoiserNagent(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner_model = InnerModel(cfg.inner_model)
        self.sample_sigma_training = None

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device):
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma

    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        b, c, _, _ = x.shape
        offset_noise = sigma_offset_noise * torch.randn(b, c, 1, 1, device=self.device)
        return x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim)

    def compute_conditioners(self, sigma: Tensor) -> Conditioners:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        return Conditioners(*(add_dims(c, n) for c, n in zip((c_in, c_out, c_skip, c_noise), (4, 4, 4, 1, 1))))

    def compute_model_output(self, noisy_next_obs: Tensor, obs: Tensor, act: Tensor, cs: Conditioners) -> Tensor:
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        return self.inner_model(rescaled_noise, cs.c_noise, rescaled_obs, act)

    @torch.no_grad()
    def wrap_model_output(self, noisy_next_obs: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d

    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised

    def forward(self, prev_actions: Tensor, frames: Tensor):
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = frames.size(1) - n

        all_obs = frames.clone()
        loss = 0

        for i in range(seq_length):
            obs = all_obs[:, i : n + i]
            next_obs = all_obs[:, n + i]
            act = prev_actions[:, :, i : n + i]  # B, p, s

            # OLD (Diamond)
            # mask = batch.mask_padding[:, n + i]

            b, t, c, h, w = obs.shape
            obs = obs.reshape(b, t * c, h, w)

            sigma = self.sample_sigma_training(b, self.device)
            noisy_next_obs = self.apply_noise(next_obs, sigma, self.cfg.sigma_offset_noise)

            cs = self.compute_conditioners(sigma)
            model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)

            target = (next_obs - cs.c_skip * noisy_next_obs) / cs.c_out

            # loss += F.mse_loss(model_output[mask], target[mask]) # OLD (Diamond)
            loss += F.mse_loss(model_output, target)

            denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
            all_obs[:, n + i] = denoised

        loss /= seq_length
        return loss, {"loss_denoising": loss.detach()}


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1


class DiffusionSampler:
    def __init__(self, denoiser: DenoiserNagent, cfg: DiffusionSamplerConfig) -> None:
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser.device)

    @torch.no_grad()
    def sample(self, prev_obs: Tensor, prev_act: Tensor) -> Tuple[Tensor, List[Tensor]]:
        device = prev_obs.device
        b, t, c, h, w = prev_obs.size()
        prev_obs = prev_obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        x = torch.randn(b, c, h, w, device=device)
        trajectory = [x]
        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x) * self.cfg.s_noise
                x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
            denoised = self.denoiser.denoise(x, sigma, prev_obs, prev_act)
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.cfg.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, prev_obs, prev_act)
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
            trajectory.append(x)
        return x, trajectory


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))


def get_env_from_id(id: str):
    if id == "boxing":
        from pettingzoo.atari import boxing_v2

        return boxing_v2.parallel_env

    else:
        raise ValueError(f"Environment {id} not found")


def env_fn(env_func):
    env = env_func(render_mode="rgb_array")
    env = frame_skip_v0(env, num_frames=4)
    env = resize_v1(env, x_size=64, y_size=64)

    return env


def get_action_from_key(player=0):
    """Map keyboard keys to all 18 actions in Atari Boxing"""
    # Boxing action meanings:
    # 0: NOOP
    # 1: Fire
    # 2: Up
    # 3: Right
    # 4: Left
    # 5: Down
    # 6: Up-Right
    # 7: Up-Left
    # 8: Down-Right
    # 9: Down-Left
    # 10: Fire-Up
    # 11: Fire-Right
    # 12: Fire-Left
    # 13: Fire-Down
    # 14: Fire-Up-Right
    # 15: Fire-Up-Left
    # 16: Fire-Down-Right
    # 17: Fire-Down-Left

    keys = pygame.key.get_pressed()
    if player == 0:
        # Player 1 controls
        up = keys[pygame.K_w]
        down = keys[pygame.K_s]
        left = keys[pygame.K_a]
        right = keys[pygame.K_d]
        fire = keys[pygame.K_SPACE]
    elif player == 1:
        # Player 2 controls
        up = keys[pygame.K_UP]
        down = keys[pygame.K_DOWN]
        left = keys[pygame.K_LEFT]
        right = keys[pygame.K_RIGHT]
        fire = keys[pygame.K_SLASH]
    # # Handle combined key presses
    # up = keys[pygame.K_w]
    # down = keys[pygame.K_s]
    # left = keys[pygame.K_a]
    # right = keys[pygame.K_d]
    # fire = keys[pygame.K_SPACE]

    if not fire:

        # Basic movement
        if up and not down and not left and not right:
            action = 2
        elif down and not up and not left and not right:
            action = 5
        elif left and not right and not up and not down:
            action = 4
        elif right and not left and not up and not down:
            action = 3
        elif up and right and not down and not left:
            action = 6
        elif up and left and not down and not right:
            action = 7
        elif down and right and not up and not left:
            action = 8
        elif down and left and not up and not right:
            action = 9
        else:
            action = 0
    else:
        if fire and not up and not down and not left and not right:
            action = 1
        elif fire and up and not down and not left and not right:
            action = 10
        elif fire and right and not left and not up and not down:
            action = 11
        elif fire and left and not right and not up and not down:
            action = 12
        elif fire and down and not up and not left and not right:
            action = 13
        elif fire and up and right and not down and not left:
            action = 14
        elif fire and up and left and not down and not right:
            action = 15
        elif fire and down and right and not up and not left:
            action = 16
        elif fire and down and left and not up and not right:
            action = 17
        else:
            action = 0

    return action


def _to_tensor(x):
    return torch.tensor(x, device="cuda").div(255).mul(2).sub(1).permute(2, 0, 1).contiguous()


if __name__ == "__main__":
    seq = 4

    frames = []
    prev_actions = [[], []]

    config_path = "/home/monsh/works/image/madx/madx/config/trainer.yaml"

    # model_weight_path = "/home/monsh/works/image/madx/madx/Boxing.pt"
    # model_weight_path = "/home/monsh/works/image/madx/madx/weights/n-agent-boxing-0100-second.pt"
    model_weight_path = "/home/monsh/works/image/madx/madx/weights/n-players-attention-015000.pt"

    use_vae = True

    if use_vae:
        from diffusers import AutoencoderKL

        device = "cuda:0"
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        vae.load_state_dict(torch.load("/home/monsh/works/image/madx/madx/weights/vae_finetuned_boxing_best.pt"))
        vae = vae.to(device)
        vae.eval()

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        cfg["agent"]["denoiser"]["inner_model"]["num_actions"] = 18

    cfg = OmegaConf.structured(cfg)

    denoiser = DenoiserNagent(cfg.agent.denoiser).to("cuda:0")

    # checkpoint = torch.load(model_weight_path)
    # denoiser_state_dict = {k.replace("denoiser.", ""): v for k, v in checkpoint.items() if k.startswith("denoiser.")}
    # denoiser.load_state_dict(denoiser_state_dict)

    denoiser_state_dict = torch.load(model_weight_path)
    denoiser.load_state_dict(denoiser_state_dict)
    denoiser.eval()

    sampler = DiffusionSampler(denoiser=denoiser, cfg=DiffusionSamplerConfig(num_steps_denoising=4))
    env = env_fn(env_func=get_env_from_id("boxing"))
    observations, infos = env.reset()

    # Initialize pygame
    pygame.init()
    display_scale = 8  # Scale up the 64x64 image
    display_size = (64 * display_scale, 64 * display_scale)
    screen = pygame.display.set_mode(display_size)
    pygame.display.set_caption("Boxing Game")
    clock = pygame.time.Clock()

    running = True

    while running and env.agents:
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get actions for all agents
        actions = {}
        keys = pygame.key.get_pressed()

        if len(frames) < seq:
            for no_player, (agent) in enumerate(env.agents):
                # a = random.randint(0, 17)
                # a = 0
                actions[agent] = 0
                prev_actions[no_player].append(0)

            # prev_actions.append([act[0], act[1]])
            observations, rewards, terminations, truncations, infos = env.step(actions)
            frames.append(observations[env.agents[0]])
            continue

        if isinstance(prev_actions, list):
            # (b, seq, players)
            prev_actions = torch.tensor(prev_actions, dtype=torch.long).to("cuda:0")

        frames_tensor = [_to_tensor(frame) for frame in frames]
        frames_tensor = torch.stack(frames_tensor).to("cuda:0")

        x = frames_tensor.unsqueeze(0)
        act = prev_actions.unsqueeze(0)
        # act = act.reshape(1, 2, seq)
        # print(act)
        out = sampler.sample(x, act)

        frames.pop(0)
        prev_actions = prev_actions[:, 1:]

        img = Image.fromarray(out[0][0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
        pygame_image = np.array(img.resize((64, 64), resample=Image.NEAREST))

        if use_vae:
            vae.eval()

            posterior = vae.encode(transform(pygame_image).unsqueeze(0).to("cuda")).latent_dist
            z = posterior.sample()
            reconstruction = vae.decode(z).sample
            pygame_image = reconstruction[0].cpu().permute(1, 2, 0) * 0.5 + 0.5  # Denormalize
            pygame_image = pygame_image.detach().numpy().clip(0, 1)
            pygame_image = pygame_image * 255

        frames.append(pygame_image)

        # Flip the image vertically
        pygame_image = np.flip(pygame_image, axis=0)
        # Rotate the image 90 degrees clockwise
        pygame_image = np.rot90(pygame_image, k=-1)

        # Create RGB surface
        surf = pygame.surfarray.make_surface(pygame_image)  # np.repeat(display_frame[:, :, np.newaxis], 3, axis=2))
        surf = pygame.transform.scale(surf, display_size)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(30)

        # Default action is NOOP (0)
        player_action = get_action_from_key(player=0)
        second_player_action = get_action_from_key(player=1)
        prev_actions = torch.cat([prev_actions, torch.tensor([[player_action], [second_player_action]]).to("cuda:0")], dim=1)
        print("prev_actions", prev_actions)

        # prev_actions[0][0] = torch.cat([prev_actions[0][0], torch.tensor([player_action], dtype=torch.long).to("cuda:0")])
        # prev_actions[0][1] = torch.cat([prev_actions[0][1], torch.tensor([second_player_action], dtype=torch.long).to("cuda:0")])
    # Clean up
    pygame.quit()
