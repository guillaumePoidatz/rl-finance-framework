import gymnasium as gym
import numpy as np
import collections


from rl_finance_framework.utils.env_utils import GymSpaceBuilder
from rl_finance_framework.utils.env_utils import DiskDataLoader
from rl_finance_framework.utils.env_utils import StatisticsRecorder
from rl_finance_framework.utils.env_utils import Scaler
from rl_finance_framework.utils.env_utils import calculate_margin_isolated

import random

class LearningCryptoEnv(gym.Env):
    def __init__(self, **env_config):
        super().__init__()

        directory_name = env_config.get("logdir", "")
        self.lookback_window_len = env_config.get("lookback_window_len")
        dataset_name = env_config.get("dataset_name")
        self.open_fee: float = env_config.get("open_fee")
        self.close_fee: float = env_config.get("close_fee")
        self.maintenance_margin_percentage: float = env_config.get(
            "maintenance_margin_percentage"
        )
        self.initial_balance: float = env_config.get("initial_capital")
        self.leverage: int = env_config.get("leverage")
        self.episode_maxstep_achieved: bool = False
        self.liquidation: bool = False
        self.record_stats: bool = env_config.get("record_stats")
        self.regime = env_config.get("regime")
        self.train_start = env_config.get("train_start")
        self.train_end = env_config.get("train_end")
        self.episode_max_len = env_config.get("episode_max_len")
        self.initial_random_allocated = env_config.get("initial_random_allocated")
        self.order_size = env_config.get("order_size")

        self.price_array, self.tech_array_total = DiskDataLoader(
            directoryName=directory_name, dataset_name=dataset_name
        ).load_dataset()

        self.scaler = Scaler(
            min_quantile=0.5, max_quantile=99.5, scale_coef=self.initial_balance
        )

        # for custom transformer
        observation_dim = (
            self.tech_array_total.shape[1] + 2
        ) * self.lookback_window_len  # hardcoded, added 2 parameters from exchange

        if self.regime == "training":
            random_interval = np.random.randint(len(self.train_start))
            self.max_step = self.episode_max_len - 1

            # Episode beginning random sampling
            self.time_absolute = np.random.randint(
                self.train_start[random_interval],
                self.train_end[random_interval] - self.max_step - 1,
            )

        elif self.regime == "evaluation":
            random_interval = np.random.randint(len(self.test_start))
            self.max_step = (
                self.episode_max_len - 1
            )  # self.test_end[random_interval] - self.test_start[random_interval] - 1
            self.time_absolute = np.random.randint(
                self.test_start[random_interval],
                self.test_end[random_interval] - self.max_step - 1,
            )

        elif self.regime == "backtesting":
            random_interval = 0
            self.max_step = (
                self.episode_max_len - 1
            )  # self.test_end[random_interval] - self.test_start[random_interval] - 1
            self.time_absolute = self.test_start[random_interval]

        else:
            raise ValueError(
                f"Invalid regime: '{self.regime}'. Allowed values are 'training', 'evaluation', or 'backtesting'."
            )

        self.observation_space, self.action_space = GymSpaceBuilder(
            observation_dim=observation_dim
        ).get_spaces()

        self._reset_env_state()

    def _reset_env_state(self):
        self.statistics_recorder = StatisticsRecorder(
            record_statistics=self.record_stats
        )
        self.state_que = collections.deque(maxlen=self.lookback_window_len)
        self.reset_que = collections.deque(
            maxlen=self.lookback_window_len * 4
        )  # dataframe to fit scaler is 4 times longer than lookback_window_len

        self.time_relative = 0  # steps played in the current episode
        self.wallet_balance = self.initial_balance

        self.liquidation = False
        self.episode_maxstep_achieved = False

        # fixed bid/ask spread
        self.price_bid = self.price_array[self.time_absolute, 0] * (1 - self.open_fee)
        self.price_ask = self.price_array[self.time_absolute, 0] * (1 + self.open_fee)

        # open position at the episode start can be only long or only short
        random_initial_position = random.choice(
            [True, False]
        )  # used if self.initial_random_allocated > 0
        self._reset_env_state_short(random_initial_position)
        self._reset_env_state_long(
            not random_initial_position
        )  # invert random_initial_position

        (
            self.margin_short,
            self.margin_long,
            self.initial_margin_short,
            self.initial_margin_long,
            self.position_value_short,
            self.position_value_long,
        ) = calculate_margin_isolated(
            coins_long=self.coins_long,
            average_price_long=self.average_price_long,
            leverage=self.leverage,
            close_fee=self.close_fee,
            maintenance_margin_percentage=self.maintenance_margin_percentage,
        )

        self.available_balance = max(
            self.wallet_balance - self.margin_short - self.margin_long, 0
        )

        if self.regime == "training":
            random_interval = np.random.randint(len(self.train_start))
            self.max_step = self.episode_max_len - 1

            # Episode beginning random sampling
            self.time_absolute = np.random.randint(
                self.train_start[random_interval],
                self.train_end[random_interval] - self.max_step - 1,
            )

            ## Sample more recent timesteps more often
            # sample_list = np.linspace(-2, 3, self.train_end[random_interval]-self.train_start[random_interval]-self.max_step)
            # cdf = ss.norm.cdf(sample_list, loc=0, scale=1)
            # self.time_absolute_step_array = np.arange(self.train_start[random_interval], self.train_end[random_interval]-self.max_step)

            # sum_cdf = sum(cdf)
            # self.probability_distribution = [float(i)/sum_cdf for i in cdf]

            # self.time_absolute = np.random.choice(self.time_absolute_step_array, 1, p=self.probability_distribution)[0]
        elif self.regime == "evaluation":
            random_interval = np.random.randint(len(self.test_start))
            self.max_step = (
                self.episode_max_len - 1
            )  # self.test_end[random_interval] - self.test_start[random_interval] - 1
            self.time_absolute = np.random.randint(
                self.test_start[random_interval],
                self.test_end[random_interval] - self.max_step - 1,
            )

        elif self.regime == "backtesting":
            random_interval = 0
            self.max_step = (
                self.test_end[random_interval] - self.test_start[random_interval] - 1
            )
            self.time_absolute = self.test_start[random_interval]

        else:
            raise ValueError(
                f"Invalid regime: '{self.regime}'. Allowed values are 'training', 'evaluation', or 'backtesting'."
            )

        self.unrealized_pnl_short = -self.coins_short * (
            self.average_price_short - self.price_ask
        )  # - self.fee_to_close_short
        self.unrealized_pnl_long = self.coins_long * (
            self.price_bid - self.average_price_long
        )  # - self.fee_to_close_long

        # equity at the episode's beginning
        self.equity = (
            self.wallet_balance + self.unrealized_pnl_short + self.unrealized_pnl_long
        )

    # self.coins_short is negative
    def _reset_env_state_short(self, random_open_position):
        # Start episode with already open SHORT position
        if self.regime == "training" and random_open_position:
            # sample average_price from past 24 hours
            self.average_price_short = random.uniform(
                self.price_array[self.time_absolute - 24, 0],
                self.price_array[self.time_absolute, 0],
            )
            self.position_value_short = random.uniform(
                0.0, self.initial_random_allocated
            )
            self.coins_short = (
                self.position_value_short / self.average_price_short * (-1)
            )
        else:
            self.average_price_short = self.price_array[self.time_absolute, 0]
            self.position_value_short = 0.0
            self.coins_short = 0.0

    def _reset_env_state_long(self, random_open_position):
        # Start episode with already open LONG position
        if self.regime == "training" and random_open_position:
            # sample average_price from past 24 hours
            self.average_price_long = random.uniform(
                self.price_array[self.time_absolute - 24, 0],
                self.price_array[self.time_absolute, 0],
            )
            self.position_value_long = random.uniform(
                0.0, self.initial_random_allocated
            )
            self.coins_long = self.position_value_long / self.average_price_long
        else:
            self.average_price_long = self.price_array[self.time_absolute, 0]
            self.position_value_long = 0.0
            self.coins_long = 0.0

    def reset(self, seed=None, options={}):
        self._reset_env_state()
        state_array, reset_array = self._get_observation_reset()
        scaled_obs_reset = self.scaler.reset(state_array, reset_array).flatten()

        # return scaled_obs_reset
        return scaled_obs_reset, {}

    def step(self, action: int):
        assert action in [0, 1, 2, 3], action
        ## prevent random actions with not initialized LSTM hidden state, applied if "use_lstm" in PPO config
        # if self.time_relative < self.cold_start_steps:
        #     action = 0

        # price = self.self.price_array[self.time_absolute, 0]
        self.price_bid = self.price_array[self.time_absolute, 0] * (1 - self.open_fee)
        self.price_ask = self.price_array[self.time_absolute, 0] * (1 + self.open_fee)

        margin_short_start = self.margin_short
        margin_long_start = self.margin_long

        reward_realized_pnl_short = 0.0
        reward_realized_pnl_long = 0.0

        # Oneway actions
        if action == 0:  # do nothing
            reward_realized_pnl_long = 0.0
            reward_realized_pnl_short = 0.0

        # similar to "BUY" button
        if action == 1:  # open/increace long position by self.order_size
            if self.coins_long >= 0:
                if self.available_balance > self.order_size:
                    buy_num_coins = self.order_size / self.price_ask
                    self.average_price_long = (
                        self.position_value_long + buy_num_coins * self.price_ask
                    ) / (self.coins_long + buy_num_coins)
                    self.initial_margin_long += (
                        buy_num_coins * self.price_ask / self.leverage
                    )
                    self.coins_long += buy_num_coins

        # similar to "SELL" button
        if action == 2:  # close/reduce long position by self.order_size
            if self.coins_long > 0:
                sell_num_coins = min(self.coins_long, self.order_size / self.price_ask)
                self.initial_margin_long *= (
                    max((self.coins_long - sell_num_coins), 0.0) / self.coins_long
                )
                self.coins_long = max(
                    self.coins_long - sell_num_coins, 0
                )  # cannot be negative
                realized_pnl = sell_num_coins * (
                    self.price_bid - self.average_price_long
                )
                self.wallet_balance += realized_pnl
                reward_realized_pnl_long = realized_pnl

        self.liquidation = (
            -self.unrealized_pnl_long - self.unrealized_pnl_short
            > self.margin_long + self.margin_short
        )
        self.episode_maxstep_achieved = self.time_relative == self.max_step

        # CLOSE entire position or LIQUIDATION
        if action == 3 or self.liquidation or self.episode_maxstep_achieved:
            # close LONG position
            if self.coins_long > 0:
                sell_num_coins = self.coins_long
                # becomes zero
                self.initial_margin_long *= (
                    max((self.coins_long - sell_num_coins), 0.0) / self.coins_long
                )
                # becomes zero
                self.coins_long = max(self.coins_long - sell_num_coins, 0)
                realized_pnl = sell_num_coins * (
                    self.price_bid - self.average_price_long
                )
                self.wallet_balance += realized_pnl
                reward_realized_pnl_long = realized_pnl

        (
            self.margin_short,
            self.margin_long,
            self.initial_margin_short,
            self.initial_margin_long,
            self.position_value_short,
            self.position_value_long,
        ) = calculate_margin_isolated(
            coins_long=self.coins_long,
            average_price_long=self.average_price_long,
            leverage=self.leverage,
            close_fee=self.close_fee,
            maintenance_margin_percentage=self.maintenance_margin_percentage,
        )

        self.available_balance = max(
            self.wallet_balance - self.margin_short - self.margin_long, 0
        )
        self.unrealized_pnl_short = -self.coins_short * (
            self.average_price_short - self.price_ask
        )  # self.coins_short is negatve
        self.unrealized_pnl_long = self.coins_long * (
            self.price_bid - self.average_price_long
        )  # - self.fee_to_close_long
        next_equity = (
            self.wallet_balance + self.unrealized_pnl_short + self.unrealized_pnl_long
        )

        done = (
            self.episode_maxstep_achieved or self.liquidation
        )  # end of episode or liquidation event

        # reward function
        # normalize rewards to fit [-10:10] range
        reward = (
            reward_realized_pnl_short + reward_realized_pnl_long
        ) / self.initial_balance
        # reward = (next_equity - self.equity) / self.initial_balance # reward function for equity changes

        self.equity = next_equity

        margin_short_end = self.margin_short
        margin_long_end = self.margin_long

        obs_step = self._get_observation_step(self.time_absolute)
        obs = self.scaler.step(obs_step).flatten()

        self.statistics_recorder.update(
            action=action,
            reward=reward,
            reward_realized_pnl_short=reward_realized_pnl_short,
            reward_realized_pnl_long=reward_realized_pnl_long,
            unrealized_pnl_short=self.unrealized_pnl_short,
            unrealized_pnl_long=self.unrealized_pnl_long,
            margin_short_start=margin_short_start,
            margin_long_start=margin_long_start,
            margin_short_end=margin_short_end,
            margin_long_end=margin_long_end,
            num_steps=self.time_relative,
            coins_short=self.coins_short,
            coins_long=self.coins_long,
            equity=self.equity,
            wallet_balance=self.wallet_balance,
            average_price_short=self.average_price_short,
            average_price_long=self.average_price_long,
        )

        self.time_absolute += 1
        self.time_relative += 1

        info = self.statistics_recorder.get()

        return obs, reward, done, False, info

    def _get_observation_reset(self):
        for current_time_absolute in range(
            self.time_absolute - self.lookback_window_len * 4, self.time_absolute
        ):
            self._get_observation_step(current_time_absolute)

        return np.array(self.state_que), np.array(self.reset_que)

    def _get_observation_step(self, current_time):
        input_array = self.tech_array_total[current_time]

        day_column = input_array[0]
        hour_column = input_array[1]
        available_balance = self.available_balance
        unrealized_pnl = self.unrealized_pnl_long + self.unrealized_pnl_short

        current_observation = np.hstack(
            (
                day_column,
                hour_column,
                available_balance,
                unrealized_pnl,
                input_array[2:],
            )
        ).astype(np.float32)
        self.state_que.append(current_observation)
        self.reset_que.append(current_observation)

        return np.array(self.state_que)
