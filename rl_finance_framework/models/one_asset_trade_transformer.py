import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Any, NewType

from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.typing import TensorType
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT, Model


logging.getLogger(__name__)


# the input layer is transforming a flat encoding vector in different features tensors to train the model
class InputSplit(nn.Module):
    def __init__(
        self, num_obs_in_history: int, d_obs: int, d_obs_logical_segments: List[int]
    ):
        super(InputSplit, self).__init__()
        self.num_obs_in_history = (
            num_obs_in_history  # one week hourly is for example 24*7 = 168
        )
        self.d_obs = d_obs  # dimension of the observation space which is the number of features used for the asset management (38)
        self.d_obs_logical_segments = d_obs_logical_segments  # list with dimension of all the observational subspaces (2,2,34)

        assert sum(d_obs_logical_segments) == d_obs, (
            "The sum of each element of d_obs_logical_segments must be equal to d_obs"
        )

    def forward(
        self, obs_history_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param obs_history_flat: Tensor [batch_size, time, d_flat], with d_flat = num_obs_in_history * d_obs
        :return: tuple of 3 tensors splitted following the logical segments of the observation
        """
        batch_size = obs_history_flat.shape[0]

        # reshape in [batch, time, d_obs] = [B, T=168, D=38]
        obs_history = obs_history_flat.view(
            batch_size, self.num_obs_in_history, self.d_obs
        )

        # split following observational subspace dimensions  (ex. [2, 2, 34])
        obs_logical_segments = torch.split(
            obs_history, self.d_obs_logical_segments, dim=-1
        )

        obs_history_time = obs_logical_segments[0]
        obs_history_account = obs_logical_segments[1]
        obs_history_candlesticks_btc = obs_logical_segments[2]

        return obs_history_time, obs_history_account, obs_history_candlesticks_btc


# each time information is encoding in a higher dimensional space in order to have richer information
class TimeEncoding(nn.Module):
    def __init__(
        self,
        num_obs_in_history: int = 168,
        d_obs_time: int = 2,
        d_time_enc: int = 255 // 3,
    ):
        super(TimeEncoding, self).__init__()

        self.num_obs_in_history = num_obs_in_history
        self.d_time_enc = d_time_enc

        # input size is 2 with the date and its index
        self.dense_time_encoding = nn.Linear(
            in_features=d_obs_time + 1, out_features=self.d_time_enc
        )

        self.layer_norm_time = nn.LayerNorm(self.d_time_enc)

    def forward(self, obs_history_time: torch.Tensor) -> torch.Tensor:
        # Shape: [batch_size, time, DObsTime], where DObsTime is typically 2 (e.g., hour_of_day, day_of_week)

        # Step 1: Create time_abs_scaled (normalized absolute time) tensor
        # the nn need to now the relative position of each time in the sequence
        batch_size = obs_history_time.size(0)
        time_abs_scaled = torch.linspace(
            0.0, 1.0, self.num_obs_in_history, device=obs_history_time.device
        )
        time_abs_scaled = time_abs_scaled.view(1, self.num_obs_in_history, 1).repeat(
            batch_size, 1, 1
        )

        # Step 2: Prepare the time_calendar_clock_scaled (the time features)
        time_calendar_clock_scaled = obs_history_time

        # Step 3: Concatenate time_abs_scaled and time_calendar_clock_scaled
        time_concat = torch.cat([time_abs_scaled, time_calendar_clock_scaled], dim=-1)

        # Step 4: Pass through the dense layer
        obs_history_time_enc = self.dense_time_encoding(time_concat)

        # Step 5: Apply Layer Normalization
        obs_history_time_enc = self.layer_norm_time(obs_history_time_enc)

        return obs_history_time_enc


class AccountEncoding(nn.Module):
    def __init__(self, d_account_enc: int = 256 // 8):
        super(AccountEncoding, self).__init__()

        self.d_account_enc = d_account_enc

        self.dense_account_encoding = nn.Linear(
            in_features=2, out_features=self.d_account_enc
        )
        self.layer_norm_account = nn.LayerNorm(self.d_account_enc)

    def forward(self, obs_history_account: torch.Tensor) -> torch.Tensor:
        obs_history_account_enc = self.dense_account_encoding(obs_history_account)
        obs_history_account_enc = self.layer_norm_account(obs_history_account_enc)

        return obs_history_account_enc


class ObsEncodingInternal(nn.Module):
    def __init__(
        self,
        num_obs_in_history: int = 168,
        d_obs_time: int = 2,
        d_time_enc: int = 256 // 8,
        d_account_enc: int = 256 // 8,
    ):
        """
        obs_history_time:      shape [Batch, Time, DObsTime]
        obs_history_account:   shape [Batch, Time, DObsAccount]
        returns:               shape [Batch, Time, DTimeEnc + DAccountEnc]
        """

        super(ObsEncodingInternal, self).__init__()

        self.num_obs_in_history = num_obs_in_history
        self.d_time_enc = d_time_enc
        self.d_account_enc = d_account_enc

        self.time_encoding = TimeEncoding(
            num_obs_in_history=self.num_obs_in_history,
            d_obs_time=d_obs_time,
            d_time_enc=self.d_time_enc,
        )
        self.account_encoding = AccountEncoding(d_account_enc=self.d_account_enc)

    def forward(
        self, obs_history_time: torch.Tensor, obs_history_account: torch.Tensor
    ) -> torch.Tensor:
        obs_history_time_enc = self.time_encoding(
            obs_history_time=obs_history_time
        )  # [B, T, DTimeEnc]

        obs_history_account_enc = self.account_encoding(
            obs_history_account=obs_history_account
        )  # [B, T, DAccountEnc]

        obs_history_internal_enc = torch.cat(
            [obs_history_time_enc, obs_history_account_enc], dim=-1
        )  # [B, T, DSum]

        return obs_history_internal_enc


class CandlesticksEncoding(nn.Module):
    def __init__(
        self,
        d_candlesticks_btc: int = 34,
        d_candlesticks_enc: int = 256 * 3 // 8,
    ):
        super(CandlesticksEncoding, self).__init__()

        self.d_candlesticks_btc = d_candlesticks_btc
        self.d_candlesticks_enc = d_candlesticks_enc

        self.d_candlesticks_enc_btc = self.d_candlesticks_enc * 4 // 3

        self.dense_candlesticks_btc = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.d_candlesticks_btc,
                out_features=self.d_candlesticks_enc_btc,
            ),
            torch.nn.GELU(),
        )

        self.dense_candlesticks_encoding = nn.Linear(
            in_features=self.d_candlesticks_enc_btc,
            out_features=self.d_candlesticks_enc,
        )
        self.layer_norm_candlesticks = nn.LayerNorm(self.d_candlesticks_enc)

    def forward(self, obs_history_candlesticks_btc: torch.Tensor) -> torch.Tensor:
        obs_history_candlesticks_enc_btc = self.dense_candlesticks_btc(
            obs_history_candlesticks_btc
        )

        obs_history_candlesticks_enc = self.dense_candlesticks_encoding(
            obs_history_candlesticks_enc_btc
        )

        obs_history_candlesticks_enc = self.layer_norm_candlesticks(
            obs_history_candlesticks_enc
        )

        return obs_history_candlesticks_enc


class ObsEncodingExternal(nn.Module):
    def __init__(
        self,
        d_candlesticks_btc: int = 34,
        d_candlesticks_enc: int = 256 * 3 // 8,
    ):
        super(ObsEncodingExternal, self).__init__()

        self.d_candlesticks_btc = d_candlesticks_btc
        self.d_candlesticks_enc = d_candlesticks_enc

        self.candlesticks_encoding = CandlesticksEncoding(
            d_candlesticks_btc=self.d_candlesticks_btc,
            d_candlesticks_enc=self.d_candlesticks_enc,
        )

    def forward(self, obs_history_candlesticks_btc: torch.Tensor) -> torch.Tensor:
        obs_history_candlesticks_enc = self.candlesticks_encoding(
            obs_history_candlesticks_btc=obs_history_candlesticks_btc
        )

        obs_external_enc = obs_history_candlesticks_enc

        return obs_external_enc


class Stem(nn.Module):
    def __init__(
        self,
        num_obs_in_history: int = 168,
        d_obs_time: int = 2,
        d_time_enc: int = 256 // 8,
        d_account_enc: int = 256 // 8,
        d_candlesticks_btc: int = 34,
        d_candlesticks_enc: int = 256 * 3 // 8,
    ):
        super(Stem, self).__init__()

        self.num_obs_in_history = num_obs_in_history
        self.d_time_enc = d_time_enc
        self.d_account_enc = d_account_enc

        self.d_candlesticks_btc = d_candlesticks_btc
        self.d_candlesticks_enc = d_candlesticks_enc

        self.obs_encoding_internal = ObsEncodingInternal(
            num_obs_in_history=self.num_obs_in_history,
            d_obs_time=d_obs_time,
            d_time_enc=self.d_time_enc,
            d_account_enc=self.d_account_enc,
        )

        self.obs_encoding_external = ObsEncodingExternal(
            d_candlesticks_btc=self.d_candlesticks_btc,
            d_candlesticks_enc=self.d_candlesticks_enc,
        )

    def forward(
        self,
        obs_history_time: torch.Tensor,
        obs_history_account: torch.Tensor,
        obs_history_candlesticks_btc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tensor3[float32, Batch, Time, DObsInternalEnc],
        Tensor3[float32, Batch, Time, DObsExternalEnc]
        """

        obs_history_internal_enc = self.obs_encoding_internal(
            obs_history_time=obs_history_time,
            obs_history_account=obs_history_account,
        )

        obs_history_external_enc = self.obs_encoding_external(
            obs_history_candlesticks_btc=obs_history_candlesticks_btc
        )

        return obs_history_internal_enc, obs_history_external_enc


class StemOutputConcatenation(nn.Module):
    def __init__(self):
        super(StemOutputConcatenation, self).__init__()

    # obs_history_internal_enc [Batch, Time, DObsInternalEnc],
    # obs_history_external_enc: [Batch, Time, DObsExternalEnc]
    def forward(
        self, obs_history_internal_enc, obs_history_external_enc
    ) -> torch.Tensor:
        obs_history_stem_enc = torch.cat(
            [obs_history_internal_enc, obs_history_external_enc], dim=-1
        )

        return obs_history_stem_enc


class AttentionBlock(nn.Module):
    def __init__(
        self,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        d_obs_internal_enc: int = 256 // 4,
        d_obs_external_enc: int = 256 * 3 // 4,
    ):
        super(AttentionBlock, self).__init__()

        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.d_obs_internal_enc = d_obs_internal_enc
        self.d_obs_external_enc = d_obs_external_enc

        self.d_model = self.d_obs_internal_enc + self.d_obs_external_enc
        self.d_ff_intermediate = self.d_model * 3 // 2

        self.attn_mha = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=self.num_heads, batch_first=True
        )
        self.attn_dense = nn.Linear(
            in_features=self.d_model, out_features=self.d_obs_external_enc
        )
        self.attn_layer_norm = nn.LayerNorm(self.d_obs_external_enc)

        self.ff_dense_1 = nn.Linear(
            in_features=self.d_model, out_features=self.d_ff_intermediate
        )
        self.ff_dense_2 = nn.Linear(
            in_features=self.d_ff_intermediate, out_features=self.d_obs_external_enc
        )
        self.ff_dropout = nn.Dropout(p=self.dropout_rate)
        self.ff_layer_norm = nn.LayerNorm(normalized_shape=self.d_obs_external_enc)

    def forward(
        self,
        obs_history_internal_enc: torch.Tensor,  # [B, T, D_int]
        obs_history_external_enc: torch.Tensor,  # [B, T, D_ext]
    ) -> torch.Tensor:  # [float32, Batch, Time, DObsExternalEnc]
        # Multi-head attention (Self-attention)
        attn_block_input = obs_history_external_enc  # [B, T, D_ext]
        obs_history_enc = torch.cat(
            [obs_history_internal_enc, attn_block_input], dim=-1
        )  # [B, T, D_model]
        attn_mha_output, attn_mha_output_weights = self.attn_mha(
            query=obs_history_enc, key=obs_history_enc, value=obs_history_enc
        )  # [B, T, DObsStemEnc]
        attn_dense_output = self.attn_dense(attn_mha_output)  # [B, T, D_ext]
        # addition + normalization
        attn_add_output = attn_block_input + attn_dense_output  # Residual
        attn_block_output = self.attn_layer_norm(attn_add_output)  # [B, T, D_ext]

        # Feedforward block
        ff_block_input = attn_block_output  # [Batch, Time, D_ext]
        obs_history_enc = torch.cat(
            [obs_history_internal_enc, ff_block_input], dim=-1
        )  # [B, T, D_model]
        ff_dense_1_output = F.gelu(
            self.ff_dense_1(obs_history_enc)
        )  # [B, T, D_ff_inter]
        ff_dense_2_output = self.ff_dense_2(ff_dense_1_output)  # [B, T, D_ext]

        # dropout
        ff_dropout_output = self.ff_dropout(ff_dense_2_output)  # [Batch, Time, D_ext]

        # addition + nprmalization
        ff_add_output = ff_block_input + ff_dropout_output
        ff_block_output = self.ff_layer_norm(ff_add_output)  # [B, T, D_ext]

        return ff_block_output


class TransformerOutputTimePooling(nn.Module):
    def __init__(self):
        super(TransformerOutputTimePooling, self).__init__()

    def forward(
        self,
        obs_history_stem_enc: torch.Tensor,
        obs_history_transformer_enc: torch.Tensor,
    ) -> torch.Tensor:
        # obs_history_stem_enc [Batch, Time, D_model]
        # obs_history_transformer_enc [Batch, Time, D_ext]

        last_obs_stem_enc = obs_history_stem_enc[:, -1, :]
        last_obs_transformer_enc = obs_history_transformer_enc[:, -1, :]

        last_obs_enc = torch.cat([last_obs_stem_enc, last_obs_transformer_enc], dim=-1)

        return last_obs_enc


class Transformer(nn.Module):
    def __init__(
        self,
        d_history_flat: int = 168 * 38,
        num_obs_in_history: int = 168,
        d_obs: int = 38,
        d_time: int = 2,
        d_account: int = 2,
        d_candlesticks_btc: int = 34,
        d_obs_enc: int = 256,
        num_attn_blocks: int = 3,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        num_outputs: int = 4,
    ):
        super(Transformer, self).__init__()

        self.d_history_flat = d_history_flat
        self.num_obs_in_history = num_obs_in_history
        self.d_obs = d_obs

        self.d_time = d_time
        self.d_account = d_account
        self.d_candlesticks_btc = d_candlesticks_btc

        self.d_obs_enc = d_obs_enc
        self.num_attn_blocks = num_attn_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.num_outputs = num_outputs

        self.d_time_enc = self.d_obs_enc // 8
        self.d_account_enc = self.d_obs_enc // 8
        self.d_candlesticks_enc = self.d_obs_enc * 3 // 8

        self.d_obs_internal_enc = self.d_time_enc + self.d_account_enc
        self.d_obs_external_enc = self.d_candlesticks_enc  # + self.d_santiment_enc

        self.d_obs_logical_segments = [
            self.d_time,
            self.d_account,
            self.d_candlesticks_btc,
        ]

        assert self.d_obs_enc % 8 == 0
        assert self.d_history_flat == self.num_obs_in_history * self.d_obs
        assert self.d_obs == sum(self.d_obs_logical_segments)

        self.input_split = InputSplit(
            num_obs_in_history=self.num_obs_in_history,
            d_obs=self.d_obs,
            d_obs_logical_segments=self.d_obs_logical_segments,
        )

        self.stem = Stem(
            num_obs_in_history=self.num_obs_in_history,
            d_obs_time=self.d_time,
            d_time_enc=self.d_time_enc,
            d_account_enc=self.d_account_enc,
            d_candlesticks_btc=self.d_candlesticks_btc,
            d_candlesticks_enc=self.d_candlesticks_enc,
        )

        self.stem_output_concatenation = StemOutputConcatenation()

        self.attention_blocks: List[AttentionBlock] = [
            AttentionBlock(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                d_obs_internal_enc=self.d_obs_internal_enc,
                d_obs_external_enc=self.d_obs_external_enc,
            )
            for i in range(self.num_attn_blocks)
        ]

        self.transformer_output_time_pooling = TransformerOutputTimePooling()

        """
        self.action_branch = ActionBranch(
            num_inputs=2 * self.d_obs_external_enc + self.d_obs_internal_enc,
            num_actions=self.num_outputs,
        )
        self.value_branch = ValueBranch(
            num_inputs=2 * self.d_obs_external_enc + self.d_obs_internal_enc
        )
        """

    def forward(self, obs_history_flat: torch.Tensor):
        # [Batch, DFlatHistory]
        obs_history_flat = obs_history_flat

        obs_logical_segments = self.input_split(obs_history_flat=obs_history_flat)

        # [Batch, Time, DObsTime]
        obs_history_time = obs_logical_segments[0]
        obs_history_account = obs_logical_segments[1]  # [Batch, Time, DObsAccount]
        obs_history_candlesticks_btc = obs_logical_segments[
            2
        ]  # [Batch, Time, DObsCandlesticksBTC]

        obs_logical_segments_enc = self.stem(
            obs_history_time=obs_history_time,
            obs_history_account=obs_history_account,
            obs_history_candlesticks_btc=obs_history_candlesticks_btc,
        )

        obs_history_internal_enc = obs_logical_segments_enc[
            0
        ]  # [Batch, Time, DObsInternalEnc]
        obs_history_external_enc = obs_logical_segments_enc[
            1
        ]  # [Batch, Time, DObsExternalEnc]

        # [Batch, Time, DObsStemEnc]
        obs_history_stem_enc = self.stem_output_concatenation(
            obs_history_internal_enc=obs_history_internal_enc,
            obs_history_external_enc=obs_history_external_enc,
        )

        # [Batch, Time, DObsExternalEnc]
        for i in range(self.num_attn_blocks):
            obs_history_external_enc = self.attention_blocks[i](
                obs_history_internal_enc=obs_history_internal_enc,
                obs_history_external_enc=obs_history_external_enc,
            )

        obs_history_transformer_enc = (
            obs_history_external_enc  # [Batch, Time, DObsExternalEnc]
        )

        # [Batch, DObsTranfEnc]
        current_obs_transformer_enc = self.transformer_output_time_pooling(
            obs_history_stem_enc=obs_history_stem_enc,
            obs_history_transformer_enc=obs_history_transformer_enc,
        )

        """
        actions_logits = self.action_branch(
            inputs=current_obs_transformer_enc
        )  # [Batch, NOutputs (actions)]
        values = self.value_branch(
            inputs=current_obs_transformer_enc
        )  # [Batch, Const1]
        """
        return {
            ENCODER_OUT: {
                ACTOR: current_obs_transformer_enc,
                CRITIC: current_obs_transformer_enc,
            }
        }


# layer connected to the output of the pool layer to take the decision
# input: [Batch, DObsTranfEnc = D_model + D_ext]
# output: [Batch, NOutputs (actions)]
class ActionBranch(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int):
        super(ActionBranch, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.dense = nn.Linear(
            in_features=self.num_inputs, out_features=self.num_actions
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            inputs (torch[float32, Batch, DObsTranfEnc]): _description_

        Returns:
            torch[float32, Batch, NOutputs]: _description_
        """

        act_logits = self.dense(inputs)

        return act_logits


# layer for computing the the value of taking some action
class ValueBranch(nn.Module):
    def __init__(self, num_inputs):
        super(ValueBranch, self).__init__()

        self.num_inputs = num_inputs
        self.dense = nn.Linear(in_features=self.num_inputs, out_features=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = self.dense(inputs)

        return value


class oneAssetTradeTransformer(TorchRLModule, DefaultPPORLModule):
    def setup(self):
        self.d_history_flat = self.model_config["d_history_flat"]
        self.num_obs_in_history = self.model_config["num_obs_in_history"]
        self.d_obs = self.model_config["d_obs"]
        self.d_time = self.model_config["d_time"]
        self.d_account = self.model_config["d_account"]
        self.d_candlesticks_btc = self.model_config["d_candlesticks_btc"]
        self.d_obs_enc = self.model_config["d_obs_enc"]

        self.num_attn_blocks = self.model_config["num_attn_blocks"]
        self.num_heads = self.model_config["num_heads"]
        self.dropout_rate = self.model_config["dropout_rate"]

        self.num_outputs = self.model_config["num_outputs"]

        self.framework = "torch"

        self.d_time_enc = self.d_obs_enc // 8
        self.d_account_enc = self.d_obs_enc // 8
        self.d_candlesticks_enc = self.d_obs_enc * 3 // 8
        self.d_obs_external_enc = self.d_candlesticks_enc  # + self.d_santiment_enc

        self.d_obs_internal_enc = self.d_time_enc + self.d_account_enc

        self.encoder = Transformer(
            d_history_flat=self.d_history_flat,
            num_obs_in_history=self.num_obs_in_history,
            d_obs=self.d_obs,
            d_time=self.d_time,
            d_account=self.d_account,
            d_candlesticks_btc=self.d_candlesticks_btc,
            d_obs_enc=self.d_obs_enc,
            num_attn_blocks=self.num_attn_blocks,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            num_outputs=self.num_outputs,
        )

        self.pi = self.build_pi_head(framework=self.framework)
        self.vf = self.build_vf_head(framework=self.framework)

        """
        self.action_branch = ActionBranch(
            num_inputs=2 * self.d_obs_external_enc + self.d_obs_internal_enc,
            num_actions=self.num_outputs,
        )
        self.value_branch = ValueBranch(
            num_inputs=2 * self.d_obs_external_enc + self.d_obs_internal_enc
        )

        """
        """
    def _forward(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        # batch Dict[str, TensorType]
        # state List[Tensor2[float32, Batch, Depth]]
        # seq_lens Tensor1[int32, Batch]

        # output
        # Tuple[
        # Tensor2[float32, Batch, NOutputs],
        # List[Tensor2[float32, Batch, Depth]]]

        # [Batch, DFlatHistory]
        obs_history_flat = batch[Columns.OBS]

        # [Batch, NOutputs]
        action_logits: torch.Tensor
        # [float32, Batch, Const1]
        values_logits: torch.Tensor

        action_logits, values_logits = self.model(obs_history_flat=obs_history_flat)

        # [Batch, Const1]
        self._value_out: torch.Tensor
        self._value_out = values_logits

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.VF_PREDS: values_logits,
        }

        """

    def build_pi_head(self, framework: str) -> Model:
        """
        Builds the policy head (pi) using a custom ActionBranch.

        Args:
            framework: The framework to use. Must be "torch" here.

        Returns:
            The policy head as nn.Module.
        """
        assert framework == "torch", "This custom head is only implemented for torch."

        pi_head = ActionBranch(
            num_inputs=2 * self.d_obs_external_enc + self.d_obs_internal_enc,
            num_actions=self.num_outputs,
        )

        return pi_head

    def build_vf_head(self, framework: str) -> Model:
        """
        Args:
            framework: The framework to use. Either "torch" or "tf2".

        Returns:
            The value function head.
        """
        vf_head_config = ValueBranch(
            num_inputs=2 * self.d_obs_external_enc + self.d_obs_internal_enc
        )
        return vf_head_config

    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Default forward pass (used for inference and exploration)

        Args:
            batch (Dict[str, Any]): [Batch, DFlatHistory]

        Returns:
            Dict[str, Any]: _description_
        """
        obs_history_flat = batch[Columns.OBS]

        output = {}
        # Encoder forward pass.
        encoder_outs = self.encoder(obs_history_flat=obs_history_flat)
        # Stateful encoder?
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]

        # Pi head.
        output[Columns.ACTION_DIST_INPUTS] = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        return output

    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Train forward pass (keep embeddings for possible shared value func. call).

        Args:
            batch (Dict[str, Any]): _description_

        Returns:
            Dict[str, Any]: _description_
        """
        output = {}
        obs_history_flat = batch[Columns.OBS]
        encoder_outs = self.encoder(obs_history_flat=obs_history_flat)
        output[Columns.EMBEDDINGS] = encoder_outs[ENCODER_OUT][CRITIC]
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]
        output[Columns.ACTION_DIST_INPUTS] = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        return output

    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        if embeddings is None:
            # Separate vf-encoder.
            if hasattr(self.encoder, "critic_encoder"):
                batch_ = batch
                if self.is_stateful():
                    # The recurrent encoders expect a `(state_in, h)`  key in the
                    # input dict while the key returned is `(state_in, critic, h)`.
                    batch_ = batch.copy()
                    obs_history_flat = batch[Columns.OBS]
                    batch_[Columns.STATE_IN] = batch[Columns.STATE_IN][CRITIC]
                embeddings = self.encoder.critic_encoder(
                    obs_history_flat=obs_history_flat
                )[ENCODER_OUT]
            # Shared encoder.
            else:
                obs_history_flat = batch[Columns.OBS]
                embeddings = self.encoder(obs_history_flat=obs_history_flat)[
                    ENCODER_OUT
                ][CRITIC]

        # Value head.
        vf_out = self.vf(embeddings)
        # Squeeze out last dimension (single node value head).
        return vf_out.squeeze(-1)
