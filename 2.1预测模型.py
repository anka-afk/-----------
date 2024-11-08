import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    MultiHeadAttention,
    LayerNormalization,
    Add,
)
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from kerastuner.tuners import BayesianOptimization
import kerastuner as kt
import os
import shutil
import tempfile
import itertools
from sklearn.metrics import mean_squared_error

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据加载和准备
data = pd.read_csv("环杭州湾3.csv", encoding="gbk")
feature_importance = pd.read_csv(
    "环杭州湾feature_importance_regularization.csv", index_col=0, encoding="utf-8-sig"
)

# 创建重命名映射字典
column_mapping = {
    col: col.replace("(", "_")
    .replace(")", "_")
    .replace("，", "_")
    .replace("／", "_")
    .replace("%", "pct")
    .replace("‰", "permille")
    .replace(" ", "_")
    .strip("_")
    .lower()
    for col in feature_importance.index
}
column_mapping["地区生产总值(万元)"] = "gdp"
column_mapping["地区"] = "region"

data = data.rename(columns=column_mapping)
data = data.sort_values(by=["region", "年份"]).reset_index(drop=True)
selected_features = [column_mapping[col] for col in feature_importance.index]
target = "gdp"


# 创建多变量时间序列数据集
def create_multivariate_dataset(dataset, time_step=10):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i : (i + time_step), :-1])
        y.append(dataset[i + time_step, -1])
    return np.array(X), np.array(y)


# 定义模型构建函数，用于超参数搜索
def build_model(hp):
    input_layer = Input(shape=(time_step, X.shape[2]))
    x = input_layer
    for i in range(hp.Int("num_lstm_layers", 2, 4)):
        units = hp.Int(f"lstm_units_{i}", min_value=32, max_value=128, step=32)
        x = Bidirectional(LSTM(units, return_sequences=True))(x)
        x = Dropout(hp.Float(f"dropout_{i}", 0.2, 0.5, step=0.1))(x)

    attention_output = MultiHeadAttention(
        num_heads=hp.Int("num_heads", 2, 8, step=2),
        key_dim=hp.Int("key_dim", 32, 128, step=32),
    )(x, x)
    attention_output = Add()([x, attention_output])
    attention_output = LayerNormalization()(attention_output)

    attention_output = tf.reduce_mean(attention_output, axis=1)
    residual_output = Dense(64, activation="relu")(attention_output)
    residual_output = Dropout(0.2)(residual_output)
    output_layer = Dense(1)(residual_output)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# 使用临时目录
temp_dir = tempfile.mkdtemp()
project_name = "gdp_forecast"

# 遍历每个地区并绘图
plt.figure(figsize=(12, 8))
colors = itertools.cycle(plt.cm.tab10.colors)
regions = data["region"].unique()

# 存储每个地区的评价结果和预测数据
evaluation_results = {}
all_forecasts = []

for region in regions:
    region_data = data[data["region"] == region]
    region_data = region_data[["年份", "gdp"] + selected_features].dropna()
    region_data = region_data.sort_values(by="年份").reset_index(drop=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(region_data[selected_features + ["gdp"]])

    time_step = 10
    X, y = create_multivariate_dataset(scaled_data, time_step)

    tuner = BayesianOptimization(
        build_model,
        objective="loss",
        max_trials=20,
        executions_per_trial=1,
        directory=temp_dir,
        project_name=project_name,
        overwrite=True,
    )

    early_stopping = EarlyStopping(
        monitor="loss", patience=10, restore_best_weights=True
    )
    tuner.search(X, y, epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)
    best_model = tuner.get_best_models(num_models=1)[0]

    # 保存模型
    model_filename = f"{region}_best_model.h5"
    best_model.save(model_filename)

    # 预测未来10年 GDP
    temp_input = scaled_data[-time_step:, :-1]
    future_gdp_scaled = []
    for i in range(10):
        x_input = temp_input.reshape((1, time_step, X.shape[2]))
        pred = best_model.predict(x_input, verbose=0)[0]
        future_gdp_scaled.append(pred)
        new_row = np.append(temp_input[-1, :-1], pred)
        temp_input = np.append(temp_input[1:], [new_row], axis=0)

    future_gdp_scaled = np.array(future_gdp_scaled)
    future_gdp = scaler.inverse_transform(
        np.concatenate(
            [
                np.zeros((10, scaled_data.shape[1] - 1)),
                future_gdp_scaled.reshape(-1, 1),
            ],
            axis=1,
        )
    )[:, -1]

    # 保存预测数据
    last_year = region_data["年份"].max()
    future_years = np.array(range(last_year + 1, last_year + 11))
    forecast_df = pd.DataFrame(
        {"年份": future_years, "地区": region, "预测GDP": future_gdp}
    )
    all_forecasts.append(forecast_df)

    # 绘图
    color = next(colors)
    plt.plot(
        region_data["年份"], region_data["gdp"], label=f"{region} 历史GDP", color=color
    )
    plt.plot(
        future_years, future_gdp, label=f"{region} 预测GDP", color=color, linestyle="--"
    )
    plt.plot(
        [region_data["年份"].iloc[-1], future_years[0]],
        [region_data["gdp"].iloc[-1], future_gdp[0]],
        color=color,
        linestyle=":",
        linewidth=1,
    )

    # 计算评价指标
    predicted_train = best_model.predict(X, verbose=0).flatten()
    y_actual = scaler.inverse_transform(
        np.concatenate(
            [np.zeros((len(y), scaled_data.shape[1] - 1)), y.reshape(-1, 1)], axis=1
        )
    )[:, -1]
    y_pred = scaler.inverse_transform(
        np.concatenate(
            [
                np.zeros((len(predicted_train), scaled_data.shape[1] - 1)),
                predicted_train.reshape(-1, 1),
            ],
            axis=1,
        )
    )[:, -1]
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    evaluation_results[region] = {"MSE": mse, "RMSE": rmse}

# 导出预测数据
all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
all_forecasts_df.to_csv("all_region_forecasts.csv", index=False, encoding="utf-8-sig")

# 图表美化
plt.title(
    "各地区未来10年GDP预测 - 深度残差LSTM + Multi-Head Attention优化模型", fontsize=16
)
plt.xlabel("年份", fontsize=12)
plt.ylabel("GDP (万元)", fontsize=12)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

# 输出并保存每个地区的评价结果
with open("evaluation_results.txt", "w", encoding="utf-8") as f:
    f.write("各地区GDP预测模型评价指标\n")
    f.write("=" * 50 + "\n\n")

    for region, metrics in evaluation_results.items():
        result_text = (
            f"{region} - MSE: {metrics['MSE']:.2f}, RMSE: {metrics['RMSE']:.2f}"
        )
        print(result_text)  # 控制台输出
        f.write(result_text + "\n")  # 写入文件

    # 可以添加时间戳
    from datetime import datetime

    f.write(f"\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 清理临时目录
shutil.rmtree(temp_dir, ignore_errors=True)

# 模型使用说明：
# 1. 加载模型：使用 `tf.keras.models.load_model("地区名称_best_model.h5")` 加载保存的模型文件。
# 2. 数据预处理：在预测新数据时，需按相同方式进行数据标准化（MinMaxScaler），并生成时间序列格式数据。
# 3. 预测未来值：使用模型的 `predict` 方法输入处理好的数据，并将预测结果逆标准化，恢复到原始单位。
