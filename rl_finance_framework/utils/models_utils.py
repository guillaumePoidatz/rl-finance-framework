import torch
import numpy as np
import scipy.signal

from ray.rllib.core.columns import Columns

from trading_enhance_software.rl_training.models.one_asset_trade_transformer import (
    oneAssetTradeTransformer,
)


def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute discounted cumulative sums of vectors.
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def compute_gae(
    model: oneAssetTradeTransformer,
    batch: dict,
    rewards: torch.Tensor,
    gamma: float,
    _lambda: float,
) -> dict:
    """
    Computes GAE advantages and value targets given a batch and the model.

    Args:
        model: your trained oneAssetTradeTransformer module.
        batch: dict with at least `Columns.OBS` (flat history obs).
        rewards: tensor of shape [Batch, Time] with rewards per time step.
        gamma: discount factor.
        _lambda: lambda for GAE.

    Returns:
        batch with added keys `advantages` and `value_targets` (both torch tensors).
    """
    device = next(model.model.parameters()).device

    obs_flat = batch[Columns.OBS].to(device)  # shape: [B, T_flat]

    # reshape into [batch*time, D_flat] if needed — here assumed 1 step per sample
    # if batch already contains sequences, adjust accordingly
    logits, values = model.model(obs_flat)  # [B, num_outputs], [B, 1]

    # `values` here is the predicted V(s_t), shape: [B, 1]
    # reshape to [B], then assume last_r = 0 (or can be passed)
    values = values.squeeze(-1).detach().cpu().numpy()  # [B]
    rewards = rewards.detach().cpu().numpy()  # [B]

    # Append last value (bootstrap) for terminal state
    vpreds = np.concatenate([values, np.array([0.0])])

    # compute deltas
    deltas = rewards + gamma * vpreds[1:] - vpreds[:-1]

    advantages = discount_cumsum(deltas, gamma * _lambda)

    value_targets = advantages + values

    # add to batch
    # batch["value_targets"] = torch.tensor(value_targets, dtype=torch.float32).to(device)

    return advantages
