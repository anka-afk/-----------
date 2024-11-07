import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

# Load the CSV data to examine its structure and confirm column names and data types.
data_path = "1.csv"
data = pd.read_csv(data_path)

# Set up Chinese font
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# Plot 1: GDP and its three industries' contributions over time
plt.figure(figsize=(10, 6))
for column, label in zip(
    [
        "地区生产总值(万元)",
        "第一产业增加值(万元)",
        "第二产业增加值(万元)",
        "第三产业增加值(万元)",
    ],
    ["GDP总量", "第一产业", "第二产业", "第三产业"],
):
    plt.plot(data["年份"], data[column], marker="o", label=label)
plt.xlabel("年份")
plt.ylabel("金额（万元）")
plt.title("各年GDP总量及三大产业增加值趋势")
plt.legend()
plt.grid()
plt.show()

# Plot 2: Percentage of each industry's contribution to GDP over time
plt.figure(figsize=(10, 6))
for column, label in zip(
    [
        "第一产业增加值占GDP比重(%)",
        "第二产业增加值占GDP比重(%)",
        "第三产业增加值占GDP比重(%)",
    ],
    ["第一产业占比", "第二产业占比", "第三产业占比"],
):
    plt.plot(data["年份"], data[column], marker="o", label=label)
plt.xlabel("年份")
plt.ylabel("占比（%）")
plt.title("各年三大产业对GDP的贡献比重")
plt.legend()
plt.grid()
plt.show()

# Plot 3: Population changes (户籍人口, 非农业人口, and 年平均人口) over time
plt.figure(figsize=(10, 6))
for column, label in zip(
    ["户籍人口(万人)", "非农业人口数(万人)", "年平均人口(万人)"],
    ["户籍人口", "非农业人口", "年平均人口"],
):
    plt.plot(data["年份"], data[column], marker="o", label=label)
plt.xlabel("年份")
plt.ylabel("人口数（万人）")
plt.title("各年户籍人口、非农业人口及年平均人口趋势")
plt.legend()
plt.grid()
plt.show()

# Plot 4: Financial data (Loans and Deposits) over time
plt.figure(figsize=(10, 6))
for column, label in zip(
    [
        "年末金融机构各项贷款余额(万元)",
        "年末金融机构存款余额(万元)",
        "城乡居民储蓄年末余额(万元)",
    ],
    ["金融机构贷款余额", "金融机构存款余额", "城乡居民储蓄余额"],
):
    plt.plot(data["年份"], data[column], marker="o", label=label)
plt.xlabel("年份")
plt.ylabel("金额（万元）")
plt.title("各年金融机构贷款、存款及城乡居民储蓄余额趋势")
plt.legend()
plt.grid()
plt.show()

# Plot 5: Science and Education spending, with authorized patents over time
plt.figure(figsize=(10, 6))
for column, label in zip(
    ["科学支出(万元)", "教育支出(万元)", "专利授权数(件)"],
    ["科学支出", "教育支出", "专利授权数"],
):
    plt.plot(data["年份"], data[column], marker="o", label=label)
plt.xlabel("年份")
plt.ylabel("数量（万元/件）")
plt.title("各年科学、教育支出及专利授权数趋势")
plt.legend()
plt.grid()
plt.show()
