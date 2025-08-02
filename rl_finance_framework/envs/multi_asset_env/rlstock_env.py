import numpy as np
import pandas as pd
from gymnasium.utils import seeding
import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt

df = pd.read_csv('/Users/guillaumepoidatz/Documents/VENV/BackTest/lib/python3.12/site-packages/gymnasium/envs/rlstock/Data_Daily_Stock_Dow_Jones_30/dow_jones_30_daily_price.csv')

def data_preprocess_train(df):
    data_1 = df.copy()
    equal_4711_list = list(data_1.tic.value_counts() == 4711)
    names = data_1.tic.value_counts().index

    select_stocks_list = list(names[equal_4711_list]) + ['NKE', 'KO']
    data_2 = data_1[data_1.tic.isin(select_stocks_list)][~data_1.datadate.isin(['20010912', '20010913'])]
    data_3 = data_2[['iid', 'datadate', 'tic', 'prccd', 'ajexdi']]
    data_3['adjcp'] = data_3['prccd'] / data_3['ajexdi']

    train_data = data_3[(data_3.datadate > 20090000) & (data_3.datadate < 20160000)]
    train_daily_data = [train_data[train_data.datadate == date] for date in np.unique(train_data.datadate)]

    return train_daily_data

train_daily_data = data_preprocess_train(df)


class StockEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, day=0, money=10000):
        self.day = day

        self.action_space = spaces.Box(low=-5, high=5, shape=(28,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(57,))

        self.data = train_daily_data[self.day]
        self.terminal = False
        self.state = [money] + self.data.adjcp.values.tolist() + [0 for i in range(28)]
        self.reward = 0
        self.asset_memory = [money]

        self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        if self.state[index + 29] > 0:
            self.state[0] += self.state[index + 1] * min(abs(action), self.state[index + 29])
            self.state[index + 29] -= min(abs(action), self.state[index + 29])

    def _buy_stock(self, index, action):
        available_amount = self.state[0] // self.state[index + 1]
        self.state[0] -= self.state[index + 1] * min(available_amount, action)
        self.state[index + 29] += min(available_amount, action)

    def step(self, actions):
        self.terminal = self.day >= 1761

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            plt.savefig('result_training.png')
            plt.close()
            return self.state, self.reward, self.terminal, False, {}

        else:
            begin_total_asset = self.state[0] + sum(np.array(self.state[1:29]) * np.array(self.state[29:]))
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = train_daily_data[self.day]
            self.state = [self.state[0]] + self.data.adjcp.values.tolist() + list(self.state[29:])
            end_total_asset = self.state[0] + sum(np.array(self.state[1:29]) * np.array(self.state[29:]))

            self.reward = end_total_asset - begin_total_asset
            self.asset_memory.append(end_total_asset)

        return self.state, self.reward, self.terminal, False, {}

    def reset(self, seed=None, options=None):
        self.asset_memory = [10000]
        self.day = 0
        self.data = train_daily_data[self.day]
        self.state = [10000] + self.data.adjcp.values.tolist() + [0 for i in range(28)]
        return self.state, {}

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
