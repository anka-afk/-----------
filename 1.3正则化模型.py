import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler

# 加载数据和标准化
data = pd.read_csv("3.csv", encoding="gbk")
X = data.drop(
    columns=["地区生产总值(万元)", "年份", "地区"]
)  # 删除非数值列（年份和地区）以及目标变量
y = data["地区生产总值(万元)"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso 模型
lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
lasso_coefs = pd.Series(np.abs(lasso.coef_), index=X.columns)
lasso_selected_features = lasso_coefs[lasso_coefs > 0].sort_values(ascending=False)

# ElasticNet 模型
elastic_net = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=42).fit(X_scaled, y)
elastic_net_coefs = pd.Series(np.abs(elastic_net.coef_), index=X.columns)
elastic_net_selected_features = elastic_net_coefs[elastic_net_coefs > 0].sort_values(
    ascending=False
)

# 合并和排序特征后保存结果
combined_features = pd.DataFrame(
    {
        "Lasso Importance": lasso_selected_features,
        "ElasticNet Importance": elastic_net_selected_features,
    }
)
combined_features["Average Importance"] = combined_features.mean(axis=1)
combined_features = combined_features.sort_values(
    by="Average Importance", ascending=False
)

# 保存结果到CSV文件
combined_features.to_csv("feature_importance_regularization.csv", encoding="utf-8-sig")
print("合并和排序后的特征:\n", combined_features)
print("\n特征重要性已保存到 'feature_importance_regularization.csv'")
