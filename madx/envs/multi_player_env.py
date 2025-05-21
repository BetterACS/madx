from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import ale_py
import gymnasium
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import torch
from torch import Tensor

from supersuit import frame_skip_v0
# from .atari_preprocessing import AtariPreprocessing

def get_env_from_id(id: str):
    if id == "boxing":
        from pettingzoo.atari import boxing_v2
        return boxing_v2.env
    
    else:
        raise ValueError(f"Environment {id} not found")



def make_atari_env(
    id: str,
    num_envs: int,
    device: torch.device,
    done_on_life_loss: bool,
    size: int,
    max_episode_steps: Optional[int],
):
    
    def env_fn(env_func):
        env = env_func(render_mode="rgb_array")
        env = frame_skip_v0(env, num_frames=4)
        
        return env

    env = env_fn(env_func=get_env_from_id(id))

    # def env_fn():
    #     env = gymnasium.make(
    #         id,
    #         full_action_space=False,
    #         frameskip=1,
    #         render_mode="rgb_array",
    #         max_episode_steps=max_episode_steps,
    #     )
    #     env = AtariPreprocessing(
    #         env=env,
    #         noop_max=30,
    #         frame_skip=4,
    #         screen_size=size,
    #     )
    #     return env

    # env = AsyncVectorEnv([env_fn for _ in range(num_envs)])

    # The AsyncVectorEnv resets the env on termination, which means that it will
    # reset the environment if we use the default AtariPreprocessing of gymnasium with
    # terminate_on_life_loss=True (which means that we will only see the first life).
    # Hence a separate wrapper for life_loss, coming after the AsyncVectorEnv.

    # if done_on_life_loss:
    #     env = DoneOnLifeLoss(env)

    # env = TorchEnv(env, device)

    return env

if __name__ == "__main__":
    env = make_atari_env("boxing", 1, torch.device("cpu"), False, 84, None)
    print(env)
