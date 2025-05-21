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

from models.diffusion.inner_model import InnerModel, InnerModelConfig
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig

from pathlib import Path
import gymnasium
from envs.atari_preprocessing import AtariPreprocessing
from supersuit import frame_skip_v0, resize_v1, frame_stack_v1, reshape_v0

import pygame
import numpy as np
import time


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
    prev_actions = []

    config_path = "/home/monsh/works/image/madx/madx/config/trainer.yaml"

    # model_weight_path = "/home/monsh/works/image/madx/madx/Boxing.pt"
    # model_weight_path = "/home/monsh/works/image/madx/madx/weights/n-agent-boxing-0100-second.pt"
    model_weight_path = "/home/monsh/works/image/madx/madx/weights/naive-065000.pt"
    model_weight_path = "/home/monsh/works/image/madx/madx/weights/single-agent-boxing-0050.pt"

    single_agent = True
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
        cfg["agent"]["denoiser"]["inner_model"]["num_actions"] = 324 if not single_agent else 18

    cfg = OmegaConf.structured(cfg)

    denoiser = Denoiser(cfg.agent.denoiser).to("cuda:0")

    # checkpoint = torch.load(model_weight_path)
    # denoiser_state_dict = {k.replace("denoiser.", ""): v for k, v in checkpoint.items() if k.startswith("denoiser.")}
    # denoiser.load_state_dict(denoiser_state_dict)

    denoiser_state_dict = torch.load(model_weight_path)
    denoiser.load_state_dict(denoiser_state_dict)
    denoiser.eval()

    sampler = DiffusionSampler(denoiser=denoiser, cfg=DiffusionSamplerConfig(num_steps_denoising=5))
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
            act = []
            for agent in env.agents:
                # a = random.randint(0, 17)
                a = 0
                actions[agent] = a
                act.append(a)

            if not single_agent:
                prev_actions.append(act[0] * 18 + act[1])
            else:
                prev_actions.append(act[0])

            observations, rewards, terminations, truncations, infos = env.step(actions)
            frames.append(observations[env.agents[0]])
            continue

        if isinstance(prev_actions, list):
            prev_actions = torch.tensor(prev_actions, dtype=torch.long).to("cuda:0")

        frames_tensor = [_to_tensor(frame) for frame in frames]
        frames_tensor = torch.stack(frames_tensor).to("cuda:0")

        x = frames_tensor.unsqueeze(0)
        act = prev_actions.unsqueeze(0)
        print(act)
        out = sampler.sample(x, act)

        frames.pop(0)
        prev_actions = prev_actions[1:]

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

        if single_agent:
            prev_actions = torch.cat([prev_actions, torch.tensor([player_action], dtype=torch.long).to("cuda:0")])
        else:
            second_player_action = get_action_from_key(player=1)
            # prev_actions = torch.cat(
            #     [prev_actions, torch.tensor([player_action * 18 + random.randint(0, 17)], dtype=torch.long).to("cuda:0")]
            # )
            prev_actions = torch.cat(
                [prev_actions, torch.tensor([player_action * 18 + second_player_action], dtype=torch.long).to("cuda:0")]
            )

    # Clean up
    pygame.quit()
