import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 路径
# =========================================================

CSV_FILE = (
    r"D:\PycharmProjects\computer_controller"
    r"\experiment_results\experiment_results.csv"
)

OUTPUT_DIR = (
    r"D:\PycharmProjects\computer_controller"
    r"\experiment_results"
)


# =========================================================
# 中文字体
# =========================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS"
]

plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 读取 CSV
# =========================================================

print("=" * 100)
print("读取实验数据")
print("=" * 100)

print(CSV_FILE)

if not os.path.exists(CSV_FILE):
    print()
    print("错误：CSV 文件不存在")
    print(CSV_FILE)
    raise SystemExit


# ---------------------------------------------------------
# 你的 CSV 可能存在旧数据列数不一致的问题
#
# 使用 Python engine，并跳过异常行
# ---------------------------------------------------------

try:

    df = pd.read_csv(
        CSV_FILE,
        encoding="utf-8-sig",
        engine="python",
        on_bad_lines="skip"
    )

except Exception as e:

    print()
    print("读取 CSV 失败：")
    print(e)

    raise SystemExit


print()
print("读取完成")
print("数据行数：", len(df))
print("列数：", len(df.columns))

print()
print("CSV 列：")

for i, col in enumerate(df.columns):

    print(
        f"{i:02d} : {col}"
    )


# =========================================================
# 检查需要的列
# =========================================================

required_columns = [

    "实验编号",
    "输入次数",

    "单次输入X",
    "总输入X",

    "初始目标X",

    "移动前目标X",
    "移动后目标X",

    "本次目标变化X",
    "目标总变化X",

    "当前误差X",
    "当前误差距离",

    "单次Kx",
    "累计Kx"

]


missing = [

    col
    for col in required_columns
    if col not in df.columns

]


if missing:

    print()
    print("=" * 100)
    print("缺少以下列：")
    print("=" * 100)

    for col in missing:
        print(col)

    raise SystemExit


# =========================================================
# 转换数值
# =========================================================

numeric_columns = [

    "实验编号",
    "输入次数",

    "单次输入X",
    "总输入X",

    "初始目标X",

    "移动前目标X",
    "移动后目标X",

    "本次目标变化X",
    "目标总变化X",

    "当前误差X",
    "当前误差距离",

    "单次Kx",
    "累计Kx"

]


for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "实验编号",
        "输入次数",
        "移动后目标X"
    ]
)


# =========================================================
# 只分析 X 轴
# =========================================================

print()
print("=" * 100)
print("X 轴数据分析")
print("=" * 100)


# =========================================================
# 拟合函数
# =========================================================

def polynomial_fit(x, y, degree):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    coefficients = np.polyfit(
        x,
        y,
        degree
    )

    polynomial = np.poly1d(
        coefficients
    )

    y_pred = polynomial(x)

    # -----------------------------------------------------
    # R²
    # -----------------------------------------------------

    ss_res = np.sum(
        (y - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if ss_tot == 0:

        r2 = 1.0

    else:

        r2 = (
            1
            - ss_res / ss_tot
        )

    # -----------------------------------------------------
    # RMSE
    # -----------------------------------------------------

    rmse = np.sqrt(
        np.mean(
            (y - y_pred) ** 2
        )
    )

    return (
        polynomial,
        r2,
        rmse
    )


# =========================================================
# 格式化多项式
# =========================================================

def polynomial_formula(
    polynomial,
    x_name="x"
):

    coefficients = polynomial.coefficients
    degree = len(coefficients) - 1

    parts = []

    for i, coefficient in enumerate(coefficients):

        power = degree - i

        if abs(coefficient) < 1e-12:
            continue

        if power == 0:

            term = (
                f"{abs(coefficient):.10f}"
            )

        elif power == 1:

            term = (
                f"{abs(coefficient):.10f}"
                f"·{x_name}"
            )

        else:

            term = (
                f"{abs(coefficient):.10f}"
                f"·{x_name}^{power}"
            )

        if not parts:

            if coefficient < 0:

                term = "- " + term

            else:

                term = term

        else:

            if coefficient < 0:

                term = "- " + term

            else:

                term = "+ " + term

        parts.append(term)

    return " ".join(parts)


# =========================================================
# 自动比较 1/2/3 次拟合
# =========================================================

def analyze_fit(
    x,
    y,
    title
):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    results = []

    for degree in [1, 2, 3]:

        try:

            polynomial, r2, rmse = polynomial_fit(
                x,
                y,
                degree
            )

            formula = polynomial_formula(
                polynomial
            )

            results.append(
                {
                    "degree": degree,
                    "poly": polynomial,
                    "r2": r2,
                    "rmse": rmse,
                    "formula": formula
                }
            )

            print()
            print(
                f"{degree} 次拟合"
            )

            print(
                f"R²   = {r2:.12f}"
            )

            print(
                f"RMSE = {rmse:.12f}"
            )

            print(
                f"公式 = y = {formula}"
            )

        except Exception as e:

            print(
                f"{degree} 次拟合失败：{e}"
            )


    # -----------------------------------------------------
    # 按 R² 最大选择
    # -----------------------------------------------------

    best = max(
        results,
        key=lambda item: item["r2"]
    )

    print()
    print("-" * 100)
    print("最佳拟合")
    print("-" * 100)

    print(
        f"阶数 : {best['degree']}"
    )

    print(
        f"R²   : {best['r2']:.12f}"
    )

    print(
        f"RMSE : {best['rmse']:.12f}"
    )

    print(
        f"公式 : y = {best['formula']}"
    )

    return best, results


# =========================================================
# 图 1
#
# 鼠标累计输入 X
# ↓
# 移动后目标 X
# =========================================================

x1 = df["总输入X"].values
y1 = df["移动后目标X"].values


best1, results1 = analyze_fit(
    x1,
    y1,
    "图1：累计鼠标输入 X → 目标位置 X"
)


# ---------------------------------------------------------
# 绘图
# ---------------------------------------------------------

plt.figure(
    figsize=(12, 7)
)

plt.scatter(
    x1,
    y1,
    s=15,
    alpha=0.6,
    label="实验数据"
)


x_plot = np.linspace(
    np.min(x1),
    np.max(x1),
    1000
)

y_plot = best1["poly"](
    x_plot
)


plt.plot(
    x_plot,
    y_plot,
    linewidth=2,
    label=(
        f"{best1['degree']}次拟合\n"
        f"R²={best1['r2']:.8f}\n"
        f"RMSE={best1['rmse']:.4f}"
    )
)


plt.xlabel(
    "累计鼠标输入 X"
)

plt.ylabel(
    "移动后目标 X"
)

plt.title(
    "累计鼠标输入 X 与目标位置 X 的关系"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


output1 = os.path.join(
    OUTPUT_DIR,
    "01_累计鼠标输入X_目标位置X.png"
)

plt.savefig(
    output1,
    dpi=200
)

plt.show()


# =========================================================
# 图 2
#
# 移动前目标 X
# ↓
# 本次目标变化 X
# =========================================================

x2 = df["移动前目标X"].values
y2 = df["本次目标变化X"].values


best2, results2 = analyze_fit(
    x2,
    y2,
    "图2：移动前目标 X → 单次目标变化 ΔX"
)


plt.figure(
    figsize=(12, 7)
)

plt.scatter(
    x2,
    y2,
    s=15,
    alpha=0.6,
    label="实验数据"
)


x_plot = np.linspace(
    np.min(x2),
    np.max(x2),
    1000
)

y_plot = best2["poly"](
    x_plot
)


plt.plot(
    x_plot,
    y_plot,
    linewidth=2,
    label=(
        f"{best2['degree']}次拟合\n"
        f"R²={best2['r2']:.8f}\n"
        f"RMSE={best2['rmse']:.4f}"
    )
)


plt.xlabel(
    "移动前目标 X"
)

plt.ylabel(
    "本次目标变化 ΔX"
)

plt.title(
    "目标当前位置与单次鼠标输入造成的目标位移关系"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


output2 = os.path.join(
    OUTPUT_DIR,
    "02_目标X_单次目标变化.png"
)

plt.savefig(
    output2,
    dpi=200
)

plt.show()


# =========================================================
# 图 3
#
# 输入次数
# ↓
# 目标 X
# =========================================================

x3 = df["输入次数"].values
y3 = df["移动后目标X"].values


best3, results3 = analyze_fit(
    x3,
    y3,
    "图3：鼠标输入次数 → 目标位置 X"
)


plt.figure(
    figsize=(12, 7)
)

plt.scatter(
    x3,
    y3,
    s=15,
    alpha=0.6,
    label="实验数据"
)


x_plot = np.linspace(
    np.min(x3),
    np.max(x3),
    1000
)

y_plot = best3["poly"](
    x_plot
)


plt.plot(
    x_plot,
    y_plot,
    linewidth=2,
    label=(
        f"{best3['degree']}次拟合\n"
        f"R²={best3['r2']:.8f}\n"
        f"RMSE={best3['rmse']:.4f}"
    )
)


plt.xlabel(
    "鼠标输入次数"
)

plt.ylabel(
    "目标 X"
)

plt.title(
    "鼠标移动次数与目标 X 的关系"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


output3 = os.path.join(
    OUTPUT_DIR,
    "03_鼠标输入次数_目标X.png"
)

plt.savefig(
    output3,
    dpi=200
)

plt.show()


# =========================================================
# 图 4
#
# 当前目标 X
# ↓
# 当前误差 X
# =========================================================

x4 = df["移动后目标X"].values
y4 = df["当前误差X"].values


best4, results4 = analyze_fit(
    x4,
    y4,
    "图4：目标位置 X → 中心误差 X"
)


plt.figure(
    figsize=(12, 7)
)

plt.scatter(
    x4,
    y4,
    s=15,
    alpha=0.6,
    label="实验数据"
)


x_plot = np.linspace(
    np.min(x4),
    np.max(x4),
    1000
)

y_plot = best4["poly"](
    x_plot
)


plt.plot(
    x_plot,
    y_plot,
    linewidth=2,
    label=(
        f"{best4['degree']}次拟合\n"
        f"R²={best4['r2']:.8f}\n"
        f"RMSE={best4['rmse']:.4f}"
    )
)


plt.axhline(
    0,
    linewidth=1
)

plt.xlabel(
    "目标 X"
)

plt.ylabel(
    "中心误差 X"
)

plt.title(
    "目标位置 X 与中心误差 X 的关系"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()


output4 = os.path.join(
    OUTPUT_DIR,
    "04_目标X_中心误差X.png"
)

plt.savefig(
    output4,
    dpi=200
)

plt.show()


# =========================================================
# 输出总结
# =========================================================

print()
print()
print("=" * 100)
print("全部分析完成")
print("=" * 100)

print()

print("【图1】累计输入 X → 目标 X")
print(
    f"最佳阶数 : {best1['degree']}"
)
print(
    f"R²       : {best1['r2']:.12f}"
)
print(
    f"RMSE     : {best1['rmse']:.12f}"
)
print(
    f"公式     : y = {best1['formula']}"
)

print()

print("【图2】移动前目标 X → 单次目标变化 ΔX")
print(
    f"最佳阶数 : {best2['degree']}"
)
print(
    f"R²       : {best2['r2']:.12f}"
)
print(
    f"RMSE     : {best2['rmse']:.12f}"
)
print(
    f"公式     : y = {best2['formula']}"
)

print()

print("【图3】输入次数 → 目标 X")
print(
    f"最佳阶数 : {best3['degree']}"
)
print(
    f"R²       : {best3['r2']:.12f}"
)
print(
    f"RMSE     : {best3['rmse']:.12f}"
)
print(
    f"公式     : y = {best3['formula']}"
)

print()

print("【图4】目标 X → 中心误差 X")
print(
    f"最佳阶数 : {best4['degree']}"
)
print(
    f"R²       : {best4['r2']:.12f}"
)
print(
    f"RMSE     : {best4['rmse']:.12f}"
)
print(
    f"公式     : y = {best4['formula']}"
)

print()
print("=" * 100)

print("图片保存位置：")
print(OUTPUT_DIR)

print("=" * 100)