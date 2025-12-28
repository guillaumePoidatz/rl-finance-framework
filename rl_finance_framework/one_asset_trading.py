import os
import sys
from pathlib import Path
from argparse import ArgumentParser
import logging

from rl_finance_framework.config import DATA_SAVE_DIR
from rl_finance_framework.config import ERL_PARAMS
from rl_finance_framework.config import INDICATORS
from rl_finance_framework.config import RESULTS_DIR
from rl_finance_framework.config import TENSORBOARD_LOG_DIR
from rl_finance_framework.config import TEST_END_DATE
from rl_finance_framework.config import TEST_START_DATE
from rl_finance_framework.config import TRAIN_END_DATE
from rl_finance_framework.config import TRAIN_START_DATE
from rl_finance_framework.config import TRAINED_MODEL_DIR
from rl_finance_framework.config_tickers import DOW_30_TICKER
from rl_finance_framework.envs.stock_trading_env import (
    StockTradingEnv,
)

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, project_root.as_posix())


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data backtest",
        metavar="MODE",
        default="train",
    )

    parser.add_argument(
        "--num-steps", type=int, default=3e7, help="Number of training iterations"
    )
    return parser


def check_and_make_directories(directories: list[str]):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def main() -> int:
    parser = build_parser()
    options = parser.parse_args()
    check_and_make_directories(
        [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
    )

    if options.mode == "train":
        from rl_finance_framework.train import train

        env = StockTradingEnv

        kwargs = {}
        train(
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="ccxt",
            time_interval="1d",
            technical_indicator_list=INDICATORS,
            drl_lib="elegantrl",
            env=env,
            model_name="ppo",
            cwd="./test_ppo",
            erl_params=ERL_PARAMS,
            break_step=options.num_steps,
            kwargs=kwargs,
        )
    elif options.mode == "test":
        from rl_finance_framework.test import test

        env = StockTradingEnv

        kwargs = {}

        account_value_erl = test(  # noqa
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="ccxt",
            time_interval="1d",
            technical_indicator_list=INDICATORS,
            drl_lib="elegantrl",
            env=env,
            model_name="ppo",
            cwd="./test_ppo",
            net_dimension=512,
            kwargs=kwargs,
        )
    else:
        raise ValueError("Wrong mode.")
    return 0


# Users can input the following command in terminal
# python main.py --mode=train
# python main.py --mode=test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
