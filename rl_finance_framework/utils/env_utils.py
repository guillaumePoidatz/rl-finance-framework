import numpy as np
import os
from gymnasium import spaces
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict, Any


def calculate_margin_isolated(
    coins_short=0,
    average_price_short=0,
    coins_long=0,
    average_price_long=0,
    leverage=0,
    close_fee=0,
    maintenance_margin_percentage=0,
):
    position_value_short = -coins_short * average_price_short
    position_value_long = coins_long * average_price_long

    initial_margin_short = position_value_short / leverage
    initial_margin_long = position_value_long / leverage

    fee_to_close_short = position_value_short * close_fee
    fee_to_close_long = position_value_long * close_fee

    margin_short = (
        initial_margin_short
        + maintenance_margin_percentage * position_value_short
        + fee_to_close_short
    )
    margin_long = (
        initial_margin_long
        + maintenance_margin_percentage * position_value_long
        + fee_to_close_long
    )

    return (
        margin_short,
        margin_long,
        initial_margin_short,
        initial_margin_long,
        position_value_short,
        position_value_long,
    )


class GymSpaceBuilderHedge:
    def __init__(self, observation_dim: int = 241) -> None:
        self.observation_dim = observation_dim

        self.action_space = spaces.Tuple([spaces.Discrete(4), spaces.Discrete(4)])

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

    def get_spaces(self) -> Tuple[spaces.Space, spaces.Space]:
        return self.observation_space, self.action_space


class GymSpaceBuilderOneWay:
    def __init__(self, observation_dim: int = 168 * 241) -> None:
        self.observation_dim = observation_dim
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

    def get_spaces(self) -> Tuple[spaces.Space, spaces.Space]:
        return self.observation_space, self.action_space


class GymSpaceBuilder:
    def __init__(self, observation_dim: int = 567) -> None:
        self.observation_dim = observation_dim

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

    def get_spaces(self) -> Tuple[spaces.Space, spaces.Space]:
        return self.observation_space, self.action_space


class DiskDataLoader:
    def __init__(self, directoryName, dataset_name: str = "dataset") -> None:
        super().__init__()
        self.dataset_name = dataset_name

        self.path_root = directoryName + "/data"

        self.path_dataset = os.path.join(self.path_root, self.dataset_name)

        try:
            self.price_array_total = np.load(
                os.path.join(self.path_dataset, "price_outfile.npy"), mmap_mode="r"
            )
        except:
            print("price_outfile.npy does not exist in {}".format(self.path_dataset))
            self.price_array_total = None

        try:
            self.tech_array_total = np.load(
                os.path.join(self.path_dataset, "metrics_outfile.npy"),
                mmap_mode="r",
            )
        except:
            print("metrics_outfile.npy does not exist in {}".format(self.path_dataset))
            self.tech_array_total = None

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.price_array_total, self.tech_array_total

    from typing import Dict, Any


class StatisticsRecorder:
    def __init__(self, record_statistics: bool = True) -> None:
        self.record_statistics = record_statistics

        self.buy_markers_time_short = []
        self.sell_markers_time_short = []
        self.markers_amount_short = []

        self.buy_markers_time_long = []
        self.sell_markers_time_long = []
        self.markers_amount_long = []

        self.reward_realized_pnl_short = []
        self.reward_realized_pnl_long = []

        self.unrealized_pnl_short = []
        self.unrealized_pnl_long = []

        self.equity_list = []
        self.wallet_balance_list = []
        self.action_list = []
        self.reward_list = []

        self.average_price_short_list = []
        self.average_price_long_list = []

    def update(
        self,
        action,
        reward,
        reward_realized_pnl_short,
        reward_realized_pnl_long,
        unrealized_pnl_short,
        unrealized_pnl_long,
        margin_short_start,
        margin_short_end,
        margin_long_start,
        margin_long_end,
        num_steps,
        coins_short,
        coins_long,
        equity,
        wallet_balance,
        average_price_short,
        average_price_long,
    ) -> None:
        if self.record_statistics:
            margin_short_delta = margin_short_end - margin_short_start
            margin_long_delta = margin_long_end - margin_long_start

            if margin_short_delta > 0:
                self.sell_markers_time_short.append(num_steps)
                self.markers_amount_short.append(coins_short)

            if margin_short_delta < 0:
                self.buy_markers_time_short.append(num_steps)
                self.markers_amount_short.append(coins_short)

            if margin_long_delta < 0:
                self.sell_markers_time_long.append(num_steps)
                self.markers_amount_long.append(coins_long)

            if margin_long_delta > 0:
                self.buy_markers_time_long.append(num_steps)
                self.markers_amount_long.append(coins_long)

            self.reward_realized_pnl_short.append(reward_realized_pnl_short)
            self.reward_realized_pnl_long.append(reward_realized_pnl_long)

            self.unrealized_pnl_short.append(unrealized_pnl_short)
            self.unrealized_pnl_long.append(unrealized_pnl_long)

            self.equity_list.append(equity)
            self.wallet_balance_list.append(wallet_balance)
            self.action_list.append(action)
            self.reward_list.append(reward)

            self.average_price_short_list.append(average_price_short)
            self.average_price_long_list.append(average_price_long)

    def get(self) -> Dict[str, Any]:
        info = {}

        if self.record_statistics:
            info = {
                "buy_markers_time_short": self.buy_markers_time_short,
                "sell_markers_time_short": self.sell_markers_time_short,
                "buy_markers_time_long": self.buy_markers_time_long,
                "sell_markers_time_long": self.sell_markers_time_long,
                "markers_amount_short": self.markers_amount_short,
                "markers_amount_long": self.markers_amount_long,
                "reward_realized_pnl_short": self.reward_realized_pnl_short,
                "reward_realized_pnl_long": self.reward_realized_pnl_long,
                "unrealized_pnl_short": self.unrealized_pnl_short,
                "unrealized_pnl_long": self.unrealized_pnl_long,
                "equity": self.equity_list,
                "wallet_balance": self.wallet_balance_list,
                "action": self.action_list,
                "reward": self.reward_list,
                "average_price_short": self.average_price_short_list,
                "average_price_long": self.average_price_long_list,
            }

        return info


class Scaler:
    def __init__(
        self, min_quantile: int = 1, max_quantile: int = 99, scale_coef: float = 1e3
    ) -> None:
        self.transformer = None  # <class 'sklearn.preprocessing._data.RobustScaler'>
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile
        self.scale_coef = scale_coef

    def reset(self, state_array, reset_array):
        # don't apply scaler to day, hour, unrealized_pnl and available_balance columns
        self.transformer = RobustScaler(
            quantile_range=(self.min_quantile, self.max_quantile)
        ).fit(reset_array[:, 4:])
        scaled_np_array = self.step(state_array)

        return scaled_np_array

    def step(self, state_array):
        day_column = state_array[:, [0]]
        hour_column = state_array[:, [1]]
        available_balance = state_array[:, [2]] / self.scale_coef
        unrealized_pnl = state_array[:, [3]] / self.scale_coef
        transformed_indicators = np.clip(
            self.transformer.transform(state_array[:, 4:]), a_min=-10.0, a_max=10.0
        )
        scaled_np_array = np.hstack(
            (
                day_column,
                hour_column,
                available_balance,
                unrealized_pnl,
                transformed_indicators,
            )
        ).astype(np.float32)
        return scaled_np_array
