import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from scipy.interpolate import interp1d

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

# 加载数据
data = pd.read_csv("环杭州湾2.csv", encoding="gbk")

# 分离标识列和数值列
id_columns = ["年份", "地区"]
id_data = data[id_columns].copy()
numeric_data = data.drop(columns=id_columns)
numeric_columns = numeric_data.select_dtypes(include=[np.number]).columns


# 1. 使用 LOF 识别异常值
def handle_outliers_lof(df, columns, n_neighbors=20, contamination=0.01):
    # 排除地区生产总值列
    columns_to_process = [col for col in columns if col != "地区生产总值(万元)"]

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outliers = lof.fit_predict(df[columns_to_process])
    # 将异常值设置为 NaN，准备后续插值
    for i, col in enumerate(columns_to_process):
        df[col] = np.where(outliers == -1, np.nan, df[col])
    return df


# 调整异常值检测的严格性，通过较小的 contamination 只识别极端异常的值
numeric_data = handle_outliers_lof(numeric_data, numeric_columns, contamination=0.01)


# 2. 使用三次样条插值替换 NaN 值
def cubic_spline_interpolation(df, columns):
    for col in columns:
        # 找到非 NaN 值的索引和数据
        valid_index = df[col].dropna().index
        valid_data = df[col].dropna().values

        # 创建三次样条插值函数
        if len(valid_index) > 1:  # 确保至少有两个点才能进行插值
            cubic_interp = interp1d(
                valid_index, valid_data, kind="cubic", fill_value="extrapolate"
            )
            # 填充 NaN 值
            df[col] = df[col].fillna(pd.Series(cubic_interp(df.index), index=df.index))
    return df


numeric_data = cubic_spline_interpolation(numeric_data, numeric_columns)


# 4. 相关性分析和特征选择

# 4.1 相关性矩阵和热图
correlation_matrix = numeric_data[numeric_columns].corr()
plt.figure(figsize=(16, 12))  # 调整图形尺寸
sns.heatmap(
    correlation_matrix,
    annot=False,
    cmap="coolwarm",
    square=True,
    cbar_kws={"shrink": 0.8},
    linewidths=0.5,
    linecolor="gray",
)
plt.title("相关性矩阵热图")
plt.xticks(rotation=45, ha="right")  # x轴标签旋转45度并靠右对齐
plt.yticks(rotation=0)  # y轴标签保持水平
plt.tight_layout()  # 自动调整子图参数，使得子图适应到整个图像区域
plt.show()

# 5. 标准化
scaler = StandardScaler()
numeric_data[numeric_columns] = scaler.fit_transform(numeric_data[numeric_columns])

# 合并标识列和处理后的数据
final_data = pd.concat([id_data, numeric_data], axis=1)

# 保存最终预处理后的数据
final_data.to_csv("环杭州湾3.csv", index=False, encoding="gbk")
print("数据已保存至 '环杭州湾3.csv'")
