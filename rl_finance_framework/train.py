from pathlib import Path
import numpy as np


def train(
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
        price_array = np.load("datasets/crypto/train_price_array.npy")
        tech_array = np.load("datasets/crypto/train_tech_array.npy")
        turbulence_array = np.load("datasets/crypto/train_turbulence_array.npy")
    else:
        from rl_finance_framework.data_processors.meta_processor import MetaProcessor

        dp = MetaProcessor(data_source, **kwargs)
        data = dp.download_data(ticker_list, start_date, end_date, time_interval)
        data = dp.clean_data(data)
        data = dp.add_technical_indicator(data, technical_indicator_list)
        if if_vix:
            data = dp.add_vix(data)
        price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)
        np.save("datasets/crypto/train_price_array.npy", price_array)
        np.save("datasets/crypto/train_tech_array.npy", tech_array)
        np.save("datasets/crypto/train_turbulence_array.npy", turbulence_array)

    cwd = kwargs.get("cwd", "./" + str(model_name))

    if drl_lib == "elegantrl":
        from rl_finance_framework.models.elegantRL_model import DRLAgent as DRLAgent_erl

        break_step = kwargs.get("break_step", 1e10)
        erl_params = kwargs.get("erl_params")
        agent = DRLAgent_erl(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )
        model = agent.get_model(model_name, model_kwargs=erl_params)
        agent.train_model(model=model, cwd=cwd, total_timesteps=break_step)

    else:
        raise ValueError("DRL library input is NOT supported. Please check.")
