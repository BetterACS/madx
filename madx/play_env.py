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
from typing import Optional

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


def get_action_from_key(key, action_space):
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

    # Handle combined key presses
    up = keys[pygame.K_w]
    down = keys[pygame.K_s]
    left = keys[pygame.K_a]
    right = keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

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
    elif fire and not up and not down and not left and not right:
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


if __name__ == "__main__":
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

        # The player controls the first agent
        first_agent = env.agents[0]

        # Default action is NOOP (0)
        player_action = get_action_from_key(pygame.key.get_pressed(), env.action_space(first_agent))

        actions[first_agent] = player_action

        # For other agents, either use AI or random actions
        for agent in env.agents[1:]:
            actions[agent] = env.action_space(agent).sample()
            # TODO: You can use your trained model here to generate AI actions

        # Step the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Get observation for first agent (or any agent you want to display)
        obs = observations[env.agents[0]]
        # Flip the image vertically
        obs = np.flip(obs, axis=0)
        # Rotate the image 90 degrees clockwise
        obs = np.rot90(obs, k=-1)

        # Create RGB surface
        surf = pygame.surfarray.make_surface(obs)  # np.repeat(display_frame[:, :, np.newaxis], 3, axis=2))
        surf = pygame.transform.scale(surf, display_size)
        screen.blit(surf, (0, 0))

        # Display rewards
        font = pygame.font.Font(None, 36)
        reward_text = f"Reward: {rewards.get(first_agent, 0):.2f}"
        text_surface = font.render(reward_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()

        # Cap the framerate
        clock.tick(30)

        # Check for game over
        if any(terminations.values()) or any(truncations.values()):
            observations, infos = env.reset()

    # Clean up
    pygame.quit()
