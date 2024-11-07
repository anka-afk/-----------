import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 加载并准备数据
data = pd.read_csv("环杭州湾3.csv", encoding="gbk")

# 读取特征重要性文件
feature_importance = pd.read_csv(
    "环杭州湾feature_importance_regularization.csv", index_col=0, encoding="utf-8-sig"
)

# 创建重命名映射字典，进一步清理特殊字符
column_mapping = {
    col: col.replace("(", "_")
    .replace(")", "_")
    .replace("，", "_")
    .replace("／", "_")
    .replace("%", "pct")  # 替换百分号
    .replace("‰", "permille")  # 替换千分号
    .replace(" ", "_")  # 替换空格
    .strip("_")
    .lower()
    for col in feature_importance.index
}
column_mapping["地区生产总值(万元)"] = "gdp"
column_mapping["地区"] = "region"

# 重命名数据列
data = data.rename(columns=column_mapping)

# 从特征重要性文件中获取特征列表（按重要性排序）
selected_features = [column_mapping[col] for col in feature_importance.index]

# 数据标准化
X = data[selected_features]
y = data["gdp"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=selected_features)

# 合并标准化后的数据
data_scaled = pd.concat([X_scaled, data[["gdp", "region"]]], axis=1)

# 构建多层次模型公式
formula = "gdp ~ " + " + ".join(selected_features)
model = smf.mixedlm(formula, data_scaled, groups=data_scaled["region"])
result = model.fit()

# 在输出模型结果之前，添加交叉验证部分
# 准备交叉验证
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_scores = {"r2": [], "rmse": []}

# 执行交叉验证
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
    # 准备训练集和测试集
    X_train = X_scaled.iloc[train_idx]
    X_test = X_scaled.iloc[test_idx]
    y_train = data_scaled.iloc[train_idx]["gdp"]
    y_test = data_scaled.iloc[test_idx]["gdp"]
    groups_train = data_scaled.iloc[train_idx]["region"]

    # 训练模型
    model_cv = smf.mixedlm(
        formula,
        data=pd.concat(
            [X_train, pd.DataFrame({"gdp": y_train, "region": groups_train})], axis=1
        ),
        groups=groups_train,
    )
    result_cv = model_cv.fit()

    # 预测并计算性能指标
    y_pred = result_cv.predict(X_test)
    cv_scores["r2"].append(r2_score(y_test, y_pred))
    cv_scores["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))

# 输出交叉验证结果
print("\n=== 交叉验证结果 ===")
print(f"平均 R² 得分: {np.mean(cv_scores['r2']):.4f} (±{np.std(cv_scores['r2']):.4f})")
print(f"平均 RMSE: {np.mean(cv_scores['rmse']):.4f} (±{np.std(cv_scores['rmse']):.4f})")

# 输出模型结果
print(result.summary())

# 假设 result 是你的回归结果对象
coefficients = pd.DataFrame(
    {
        "Feature": result.params.index,
        "Coefficient": result.params.values,
        "P-value": result.pvalues.values,
    }
)

# 按系数绝对值排序
coefficients["Abs_Coefficient"] = coefficients["Coefficient"].abs()
coefficients = coefficients.sort_values(by=["Abs_Coefficient"], ascending=False)


# 定义分级
def classify_factor(row):
    if row["Abs_Coefficient"] > 0.1 and row["P-value"] < 0.01:
        return "核心因素"
    elif row["Abs_Coefficient"] > 0.05 and row["P-value"] < 0.05:
        return "重要因素"
    else:
        return "次要因素"


coefficients["Level"] = coefficients.apply(classify_factor, axis=1)
print(coefficients[["Feature", "Coefficient", "P-value", "Level"]])
