from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path


import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 路径
# =========================

HISTORY_FILE = project_path('results/metrics/07_cnn/17_cnn_history.csv')


OUTPUT_DIR = project_path('results/figures/07_cnn')


# =========================
# 读取训练记录
# =========================

print("读取CNN训练历史...")


history = pd.read_csv(
    HISTORY_FILE
)


print(history.head())


print()

print(
    "训练轮数：",
    len(history)
)


# =========================
# Accuracy曲线
# =========================


plt.figure(
    figsize=(8,5)
)


plt.plot(
    history["accuracy"],
    label="Training Accuracy"
)


plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)


plt.title(
    "CNN Accuracy Curve"
)


plt.legend()


plt.grid()


plt.savefig(
    output_path('results/figures/07_cnn', "18_cnn_accuracy_curve.png"),
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# =========================
# Loss曲线
# =========================


plt.figure(
    figsize=(8,5)
)


plt.plot(
    history["loss"],
    label="Training Loss"
)


plt.plot(
    history["val_loss"],
    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "CNN Loss Curve"
)


plt.legend()


plt.grid()


plt.savefig(
    output_path('results/figures/07_cnn', "18_cnn_loss_curve.png"),
    dpi=300,
    bbox_inches="tight"
)


plt.close()



print()

print(
    "CNN训练曲线保存完成"
)

print(
    OUTPUT_DIR
)