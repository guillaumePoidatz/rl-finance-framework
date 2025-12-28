import numpy as np
from pathlib import Path
import logging

log = logging.getLogger(__name__)


def test(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    **kwargs,
):
    path = Path("datasets/crypto/test_price_array.npy")

    if path.exists():
        price_array = np.load("datasets/crypto/test_price_array.npy")
        tech_array = np.load("datasets/crypto/test_tech_array.npy")
        turbulence_array = np.load("datasets/crypto/test_turbulence_array.npy")
    else:
        from rl_finance_framework.data_processors.meta_processor import MetaProcessor

        dp = MetaProcessor(data_source, **kwargs)
        data = dp.download_data(ticker_list, start_date, end_date, time_interval)
        data = dp.clean_data(data)
        data = dp.add_technical_indicator(data, technical_indicator_list)
        if if_vix:
            data = dp.add_vix(data)
        price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)
        np.save("datasets/crypto/test_price_array.npy", price_array)
        np.save("datasets/crypto/test_tech_array.npy", tech_array)
        np.save("datasets/crypto/test_turbulence_array.npy", turbulence_array)

    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    }
    env_instance = env(config=env_config)

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2**7)
    cwd = kwargs.get("cwd", "./" + str(model_name))
    log.info(f"price_array: {len(price_array)}")

    if drl_lib == "elegantrl":
        from rl_finance_framework.models.elegantRL_model import (
            DRLAgent as DRLAgent_erl,
        )

        env_args = {
            "price_array": price_array,
            "tech_array": tech_array,
            "turbulence_array": turbulence_array,
        }

        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=cwd,
            net_dimension=net_dimension,
            environment=env_instance,
            env_args=env_args,
        )
        return episode_total_assets
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")
