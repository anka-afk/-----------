import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 加载数据
gdm_bay = pd.read_csv("2.csv", encoding="gbk")

# 设置要分析的城市
cities = ["深圳", "广州", "东莞", "佛山"]

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

# 设置颜色
population_color = "#3498db"
growth_rate_color = "#e74c3c"

# 为每个城市创建子图
for idx, city in enumerate(cities):
    city_data = gdm_bay[gdm_bay["地区"] == city]

    # 创建双轴
    ax1 = axes[idx]
    ax2 = ax1.twinx()

    # 绘制人口数量（左轴）
    line1 = ax1.plot(
        city_data["年份"],
        city_data["年平均人口(万人)"],
        color=population_color,
        linewidth=2,
        marker="o",
        label="年平均人口",
    )
    ax1.set_xlabel("年份")
    ax1.set_ylabel("年平均人口（万人）", color=population_color)
    ax1.tick_params(axis="y", labelcolor=population_color)

    # 绘制自然增长率（右轴）
    line2 = ax2.plot(
        city_data["年份"],
        city_data["自然增长率(‰)"],
        color=growth_rate_color,
        linewidth=2,
        marker="s",
        label="自然增长率",
    )
    ax2.set_ylabel("自然增长率（‰）", color=growth_rate_color)
    ax2.tick_params(axis="y", labelcolor=growth_rate_color)

    # 设置标题
    ax1.set_title(f"{city}人口数量及自然增长率变化趋势", pad=10)

    # 添加网格线
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 设置x轴标签旋转
    ax1.tick_params(axis="x", rotation=45)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    if idx == 0:  # 只在第一个子图显示图例
        ax1.legend(lines, labels, loc="upper left")

# 调整布局
plt.suptitle("粤港澳大湾区主要城市人口变化趋势", fontsize=14, y=1.02)
plt.tight_layout()

# 保存图片
plt.savefig("population_growth.png", dpi=300, bbox_inches="tight")
plt.show()
