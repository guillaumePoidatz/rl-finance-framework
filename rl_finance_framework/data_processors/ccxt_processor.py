from __future__ import annotations

import calendar
from datetime import datetime
from datetime import timezone
import ccxt
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
import logging

log = logging.getLogger(__name__)


class CCXTEngineer:
    def __init__(self):
        self.binance = ccxt.binance()

    def download_data(self, ticker_list, start_date, end_date, time_interval="1m"):
        def min_ohlcv(dt, tic, limit):
            since = calendar.timegm(dt.utctimetuple()) * 1000
            ohlcv = self.binance.fetch_ohlcv(
                symbol=tic, timeframe="1m", since=since, limit=limit
            )
            return ohlcv

        def ohlcv(dt, tic, time_interval="1d"):
            ohlcv = []
            limit = 1000
            if time_interval == "1m":
                limit = 720
            elif time_interval == "1d":
                limit = 1
            elif time_interval == "1h":
                limit = 24
            elif time_interval == "5m":
                limit = 288
            for i in dt:
                start_dt = i
                since = calendar.timegm(start_dt.utctimetuple()) * 1000
                if time_interval == "1m":
                    ohlcv.extend(min_ohlcv(start_dt, tic, limit))
                else:
                    ohlcv.extend(
                        self.binance.fetch_ohlcv(
                            symbol=tic,
                            timeframe=time_interval,
                            since=since,
                            limit=limit,
                        )
                    )

            self.start = start_date
            self.end = end_date
            self.time_interval = time_interval

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = [
                datetime.fromtimestamp(float(timestamp) / 1000)
                for timestamp in df["timestamp"]
            ]
            df["open"] = df["open"].astype(np.float64)
            df["high"] = df["high"].astype(np.float64)
            df["low"] = df["low"].astype(np.float64)
            df["close"] = df["close"].astype(np.float64)
            df["volume"] = df["volume"].astype(np.float64)
            return df

        df = pd.DataFrame()
        for tic in ticker_list:
            start_dt = datetime.strptime(start_date, "%Y%m%d %H:%M:%S")
            end_dt = datetime.strptime(end_date, "%Y%m%d %H:%M:%S")
            start_timestamp = calendar.timegm(start_dt.utctimetuple())
            end_timestamp = calendar.timegm(end_dt.utctimetuple())
            if time_interval == "1m":
                date_list = [
                    datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    for timestamp in range(start_timestamp, end_timestamp, 60 * 720)
                ]
            else:
                date_list = [
                    datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    for timestamp in range(start_timestamp, end_timestamp, 60 * 1440)
                ]
            ohlcv_df = ohlcv(date_list, tic, time_interval)

            tic_df = pd.DataFrame({"tic": [tic] * len(ohlcv_df)})
            ohlcv_with_tic_df = pd.concat([ohlcv_df, tic_df], axis=1)
            df = pd.concat([df, ohlcv_with_tic_df], ignore_index=True)

        log.info(f"Actual end timestamp: {df['timestamp'].values[-1]}")
        df = df.reset_index(drop=True)
        df.columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tic",
        ]
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.sort_values(["tic", "timestamp"])
        cleaned_df.drop_duplicates(
            subset=["tic", "timestamp"], keep="last", inplace=True
        )
        log.info(f"Cleaned data: {cleaned_df}")
        return cleaned_df

    def add_technical_indicator(
        self, data: pd.DataFrame, tech_indicator_list: list[str]
    ):
        """
        calculate technical indicators
        use stockstats package to add technical indicators
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """

        df = data.copy()
        df = df.sort_values(by=["tic", "timestamp"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                temp_indicator = pd.DataFrame(temp_indicator)
                temp_indicator["tic"] = unique_ticker[i]
                temp_indicator["timestamp"] = df[df.tic == unique_ticker[i]][
                    "timestamp"
                ].to_list()
                indicator_df = pd.concat(
                    [indicator_df, temp_indicator], ignore_index=False, axis=1
                )

            df = df.merge(
                indicator_df[["tic", "timestamp", indicator]],
                on=["tic", "timestamp"],
                how="left",
            )
        df = df.sort_values(by=["timestamp", "tic"])
        return df

    def add_vix(self, data: pd.DataFrame, window=30) -> pd.DataFrame:
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """

        def compute_vix(df: pd.DataFrame) -> pd.DataFrame:
            performance = np.log(df["close"]).diff()
            df["VIXY"] = performance.rolling(window).std() * np.sqrt(365) * 100
            return df

        data = data.groupby("tic", group_keys=False).apply(compute_vix)
        data = data[window:]
        data = data.sort_values(by=["timestamp", "tic"])
        log.info(f"data with vix: {data}")
        return data

    def calculate_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="timestamp", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.timestamp.unique()
        # start after a fixed timestamp period
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"timestamp": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def add_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(
        self, df: pd.DataFrame, tech_indicator_list: list[str], if_vix: bool
    ) -> list[np.ndarray]:
        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["VIXY"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        log.info(f"price_array: {price_array}")
        log.info(f"tech_array: {tech_array}")
        log.info(f"turbulence_array: {turbulence_array}")
        return price_array, tech_array, turbulence_array
