import pandas as pd
import matplotlib.pyplot as plt

# 读取训练日志
df = pd.read_csv("/workspace/classification_projects/classification_task11/results.csv")

# 简单平滑函数（移动平均）
def smooth(x, k=5):
    return x.rolling(k, min_periods=1).mean()

# 图像大小
plt.figure(figsize=(12, 10))

# -------------------- 1. train/loss --------------------
plt.subplot(3, 1, 1)
plt.plot(df["epoch"], df["train/loss"], label="results", marker="o")
plt.plot(df["epoch"], smooth(df["train/loss"]), "--", label="smooth")
plt.title("train/loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# -------------------- 2. metrics/accuracy --------------------
plt.subplot(3, 1, 2)
plt.plot(df["epoch"], df["metrics/accuracy_top1"], label="results", marker="o")
plt.plot(df["epoch"], smooth(df["metrics/accuracy_top1"]), "--", label="smooth")
plt.title("metrics/accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# -------------------- 3. val/loss --------------------
plt.subplot(3, 1, 3)
plt.plot(df["epoch"], df["val/loss"], label="results", marker="o")
plt.plot(df["epoch"], smooth(df["val/loss"]), "--", label="smooth")
plt.title("val/loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("results_plots.png", dpi=300)

