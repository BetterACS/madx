from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
from omegaconf import OmegaConf
import yaml
import torch
import random
from PIL import Image
from torchvision import transforms
import torch
from torch import Tensor
from supersuit import frame_skip_v0, resize_v1

import pygame
import numpy as np


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

    config_path = "./config/trainer.yaml"
    use_vae = True
    device = "cuda:0"

    if use_vae:
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        vae.load_state_dict(torch.load("./weights/vae_finetuned_boxing_best.pt"))
        vae = vae.to(device)
        vae.eval()

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        cfg["agent"]["denoiser"]["inner_model"]["num_actions"] = 18
        cfg["agent"]["denoiser"]["inner_model"]["cond_channels"] = 768

    cfg = OmegaConf.structured(cfg)

    model_weights = "./weights/denoiser_finetune_movement_nagent-transformer_90.pt"
    denoiser = Denoiser(cfg.agent.denoiser).to("cuda:0")
    denoiser.load_state_dict(torch.load(model_weights))
    denoiser.eval()

    sampler = DiffusionSampler(denoiser=denoiser, cfg=DiffusionSamplerConfig(num_steps_denoising=2))
    env = env_fn(env_func=get_env_from_id("boxing"))
    observations, infos = env.reset()

    # Initialize pygame
    pygame.init()
    display_scale = 8  # Scale up the 64x64 image
    display_size = (64 * display_scale, 64 * display_scale)
    screen = pygame.display.set_mode(display_size)
    pygame.display.set_caption("Boxing Game")
    clock = pygame.time.Clock()

    # Initialize the first frame
    running = True
    frames = []
    prev_actions = [[], []]

    while running and env.agents:
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get actions for all agents
        actions = {}

        if len(frames) < seq and isinstance(prev_actions, list):
            # Random start
            for no_player, (agent) in enumerate(env.agents):
                a = random.randint(0, 8)

                actions[agent] = a
                prev_actions[no_player].append(a)

            observations, rewards, terminations, truncations, infos = env.step(actions)
            frames.append(observations[env.agents[0]])
            continue

        # Convert frames (numpy arrays) to tensors. (Running once)
        if isinstance(prev_actions, list):
            frames = frames[-seq:]
            prev_actions = [prev_actions[0][-seq:], prev_actions[1][-seq:]]
            prev_actions = torch.tensor(prev_actions, dtype=torch.long).to("cuda:0")

        frames_tensor = [_to_tensor(frame) for frame in frames]
        frames_tensor = torch.stack(frames_tensor).to("cuda:0")

        x = frames_tensor.unsqueeze(0)
        act = prev_actions.unsqueeze(0)

        # Sample from the denoiser.
        out = sampler.sample(x, act)

        frames.pop(0)
        prev_actions = prev_actions[:, 1:]

        # Convert the output tensor to a numpy array and then to a PIL image.
        img = Image.fromarray(out[0][0].add(1).div(2).mul(255).byte().permute(1, 2, 0).cpu().numpy())
        pygame_image = np.array(img.resize((64, 64), resample=Image.NEAREST))

        # Use VAE to decode the image if required.
        if use_vae:
            vae.eval()

            posterior = vae.encode(transform(pygame_image).unsqueeze(0).to("cuda")).latent_dist
            z = posterior.sample()
            reconstruction = vae.decode(z).sample

            pygame_image = reconstruction[0].cpu().permute(1, 2, 0) * 0.5 + 0.5  # Denormalize
            pygame_image = pygame_image.detach().numpy().clip(0, 1)
            pygame_image = pygame_image * 255

        frames.append(pygame_image)

        # For rendering in pygame, we need to convert the image to a format pygame can use.
        pygame_image = np.flip(pygame_image, axis=0)
        pygame_image = np.rot90(pygame_image, k=-1)

        # Create RGB surface
        surf = pygame.surfarray.make_surface(pygame_image)  # np.repeat(display_frame[:, :, np.newaxis], 3, axis=2))
        surf = pygame.transform.scale(surf, display_size)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(60)

        # Get player actions from keyboard input.
        player_action = get_action_from_key(player=0)
        second_player_action = get_action_from_key(player=1)
        # Update the actions for the players.
        prev_actions = torch.cat(
            [prev_actions, torch.tensor([[player_action], [second_player_action]], dtype=torch.long).to("cuda:0")], dim=1
        )

    # Clean up
    pygame.quit()
