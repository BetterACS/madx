from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import Conv3x3, FourierFeatures, GroupNorm, UNet


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
        #     nn.Embedding(cfg.num_actions, cfg.cond_channels // (cfg.num_steps_conditioning * 2)),
        #     nn.Flatten(),
        # )

        action_embedding_dim = cfg.cond_channels // (cfg.num_steps_conditioning * 2)
        self.action_embedding = nn.Embedding(cfg.num_actions, action_embedding_dim)

        # Positional Embedding (usually needed for Transformers)
        self.positional_embedding = nn.Embedding(cfg.num_steps_conditioning * 2, action_embedding_dim)

        # Transformer Encoder Layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=action_embedding_dim,
            nhead=min(action_embedding_dim // 32, 8),  # Example number of heads
            dim_feedforward=action_embedding_dim * 4,  # Example feedforward dimension
            batch_first=True,
        )
        self.action_transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)  # Use one layer for simplicity

        # Linear layer to project the sequence output to a single vector
        # You could pool the sequence output or use a specific token's output
        # Let's pool for simplicity: (B, n, embedding_dim) -> (B, embedding_dim)
        self.action_sequence_pool = nn.AdaptiveAvgPool1d(1)

        # Project the pooled vector to the desired size
        self.action_transformer_proj = nn.Linear(action_embedding_dim, cfg.cond_channels)

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
        # action_emb = self.act_emb(act)
        # act shape: (B, n) - integers
        embedded_actions = self.action_embedding(act)  # Shape: (B, players, n, action_embedding_dim)
        embedded_actions = embedded_actions.view(
            embedded_actions.size(0), -1, embedded_actions.size(-1)
        )  # Shape: (B, players * n, action_embedding_dim)

        # Create position IDs and add positional embedding
        position_ids = torch.arange(0, 8, device=act.device).unsqueeze(0).expand(act.shape[0], 8)  # Shape: (B, n)
        embedded_positions = self.positional_embedding(position_ids)  # Shape: (B, n, action_embedding_dim)
        # Add positional embedding
        embedded_actions_with_pos = embedded_actions + embedded_positions  # Shape: (B, n, action_embedding_dim)
        # Pass through Transformer Encoder
        transformer_output = self.action_transformer(embedded_actions_with_pos)  # Shape: (B, n, action_embedding_dim)
        # Pool across the sequence dimension
        # Need to permute for AdaptiveAvgPool1d: (B, action_embedding_dim, n)
        pooled_output = self.action_sequence_pool(transformer_output.permute(0, 2, 1)).squeeze(-1)  # Shape: (B, action_embedding_dim)
        # Project to the desired size
        action_emb = self.action_transformer_proj(pooled_output)  # Shape: (B, cfg.cond_channels)

        cond = self.cond_proj(self.noise_emb(c_noise) + action_emb)
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x
