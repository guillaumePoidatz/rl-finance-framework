import numpy as np
import gymnasium as gym
import logging

log = logging.getLogger(__name__)


class BitcoinEnv(gym.Env):  # custom env
    def __init__(
        self,
        config,
        data_cwd=None,
        initial_amount=1e2,
        transaction_fee_percent=1e-3,
        mode="train",
        gamma=0.99,
    ):
        self.price_ary = config["price_array"]
        self.tech_ary = config["tech_array"]
        self.stock_dim = 1
        self.initial_amount = initial_amount
        self.transaction_fee_percent = transaction_fee_percent
        self.max_stock = 1
        self.gamma = gamma
        self.mode = mode

        # reset
        self.day = 0
        self.initial_amount__reset = self.initial_amount
        self.amount = self.initial_amount__reset
        self.day_price = self.price_ary[self.day]
        self.day_tech = self.tech_ary[self.day]
        self.stocks = 0.0  # multi-stack

        self.total_asset = self.amount + self.day_price[0] * self.stocks
        self.episode_return = 0.0
        self.gamma_return = 0.0
        self.initial_total_asset = None

        """env information"""
        self.env_name = "BitcoinEnv4"
        self.state_dim = 1 + self.price_ary.shape[1] + self.tech_ary.shape[1]
        self.action_dim = 1
        self.if_discrete = False
        self.target_return = 10
        self.max_step = self.price_ary.shape[0]

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> np.ndarray:
        self.day = 0
        self.day_price = self.price_ary[self.day]
        self.day_tech = self.tech_ary[self.day]
        self.initial_amount__reset = self.initial_amount  # reset()
        self.amount = self.initial_amount__reset
        self.stocks = 0.0
        self.total_asset = self.amount + self.day_price[0] * self.stocks
        self.initial_total_asset = self.total_asset

        normalized_tech = [
            self.day_tech[0] * 2**-1,
            self.day_tech[1] * 2**-15,
            self.day_tech[2] * 2**-15,
            self.day_tech[3] * 2**-6,
            self.day_tech[4] * 2**-6,
            self.day_tech[5] * 2**-15,
            self.day_tech[6] * 2**-15,
        ]

        state = np.hstack(
            (
                self.amount * 2**-18,
                self.day_price * 2**-15,
                normalized_tech,
                self.stocks * 2**-4,
            )
        ).astype(np.float32)
        return state, {}

    def step(self, action) -> (np.ndarray, float, bool, None):
        stock_action = action[0]
        """buy or sell stock"""
        adj = self.day_price[0]
        if stock_action < 0:
            stock_action = min(self.stocks, -stock_action)
            self.amount += adj * stock_action * (1 - self.transaction_fee_percent)
            self.stocks -= stock_action
        elif stock_action > 0:
            max_amount_btc = self.amount / (adj * (1 + self.transaction_fee_percent))
            stock_action = min(stock_action, max_amount_btc)
            self.amount = self.amount - adj * stock_action * (
                1 + self.transaction_fee_percent
            )
            self.stocks += stock_action

        """update day"""
        self.day += 1
        self.day_price = self.price_ary[self.day]
        self.day_tech = self.tech_ary[self.day]
        done = (self.day + 1) == self.max_step
        normalized_tech = [
            self.day_tech[0] * 2**-1,
            self.day_tech[1] * 2**-15,
            self.day_tech[2] * 2**-15,
            self.day_tech[3] * 2**-6,
            self.day_tech[4] * 2**-6,
            self.day_tech[5] * 2**-15,
            self.day_tech[6] * 2**-15,
        ]
        state = np.hstack(
            (
                self.amount * 2**-18,
                self.day_price * 2**-15,
                normalized_tech,
                self.stocks * 2**-4,
            )
        ).astype(np.float32)

        next_total_asset = self.amount + self.day_price[0] * self.stocks
        reward = (next_total_asset - self.total_asset) / self.total_asset * 100
        self.total_asset = next_total_asset

        self.gamma_return = self.gamma_return * self.gamma + reward
        if done:
            reward += self.gamma_return
            self.gamma_return = 0.0
            self.episode_return = next_total_asset / self.initial_amount
        return state, reward, done, False, dict()
