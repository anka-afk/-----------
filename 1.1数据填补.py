import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# 加载数据
data = pd.read_csv("环杭州湾1.csv", encoding="utf-8-sig")

# 创建填补后的数据框
data_filled_grouped = data.copy()

# 按地区分组填补缺失值
for region, group in data.groupby("地区"):
    # 对每一列计算缺失比例并选择填补方式
    for column in group.columns:
        # 跳过非数值列
        if column in ["年份", "地区"]:
            continue

        # 确保数据类型为数值型
        try:
            series = pd.to_numeric(group[column], errors="coerce")
        except:
            continue

        # 如果该列全是缺失值，跳过
        if series.isnull().all():
            continue

        missing_ratio = series.isnull().mean()

        if missing_ratio > 0:
            filled_series = series.copy()

            # 1. 对缺失比例较低的数据使用akima插值
            if missing_ratio <= 0.2:
                try:
                    filled_series = series.interpolate(method="akima", order=3)
                except:
                    pass

            # 2. 如果仍有缺失，且缺失比例适中，使用ARIMA
            if filled_series.isnull().any() and missing_ratio <= 0.5:
                try:
                    series_clean = filled_series.dropna()
                    if len(series_clean) >= 3:  # 确保有足够的数据点
                        index = pd.date_range(
                            start="1990", periods=len(series_clean), freq="YE"
                        )
                        series_clean.index = index

                        # 尝试不同的ARIMA参数
                        orders = [(1, 1, 0), (1, 0, 0), (0, 1, 1)]
                        best_aic = float("inf")
                        best_forecast = None

                        for order in orders:
                            try:
                                model = ARIMA(series_clean, order=order)
                                fitted_model = model.fit(method="css-mle", maxiter=500)
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    forecast = fitted_model.predict(
                                        start=0, end=len(series) - 1
                                    )
                                    best_forecast = forecast
                            except:
                                continue

                        if best_forecast is not None:
                            filled_series.loc[filled_series.isnull()] = (
                                best_forecast.loc[filled_series.isnull()]
                            )
                except:
                    pass

            # 3. 如果仍有缺失，使用线性插值
            if filled_series.isnull().any():
                try:
                    filled_series = filled_series.interpolate(method="linear")
                except:
                    pass

            # 4. 如果仍有缺失，使用前向和后向填充
            if filled_series.isnull().any():
                try:
                    filled_series = filled_series.ffill().bfill()
                except:
                    pass

            # 5. 如果仍有缺失，使用该列的均值填充
            if filled_series.isnull().any():
                try:
                    mean_value = filled_series.mean()
                    if not pd.isna(mean_value):
                        filled_series = filled_series.fillna(mean_value)
                except:
                    pass

            # 更新数据
            data_filled_grouped.loc[group.index, column] = filled_series

# 检查最终填补后的缺失值情况
missing_values_check_final_grouped = data_filled_grouped.isnull().sum()
print("缺失值填补后的检查结果：")
print(missing_values_check_final_grouped)


# 保存填补后的数据
data_filled_grouped.to_csv("环杭州湾2.csv", index=False, encoding="gbk")
