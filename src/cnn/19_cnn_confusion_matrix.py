from pathlib import Path
import sys

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.paths import project_path, output_path


import pandas as pd
import matplotlib.pyplot as plt


# ==================================
# 路径
# ==================================

INPUT_FILE = project_path('results/confusion_matrix/07_cnn/17_cnn_confusion_matrix.csv')


OUTPUT_DIR = project_path('results/confusion_matrix/07_cnn')



# ==================================
# 读取混淆矩阵
# ==================================

print("读取CNN混淆矩阵...")


cm = pd.read_csv(
    INPUT_FILE,
    index_col=0
)


print()

print(cm)



# ==================================
# 绘制热图
# 不使用seaborn
# ==================================

plt.figure(
    figsize=(8,6)
)


plt.imshow(
    cm.values
)


plt.colorbar()



plt.xticks(
    range(len(cm.columns)),
    cm.columns,
    rotation=45,
    ha="right"
)


plt.yticks(
    range(len(cm.index)),
    cm.index
)



# 添加数字

for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            cm.iloc[i,j],
            ha="center",
            va="center"
        )



plt.xlabel(
    "Predicted Activity"
)


plt.ylabel(
    "True Activity"
)


plt.title(
    "CNN Confusion Matrix"
)



plt.tight_layout()



plt.savefig(
    output_path('results/confusion_matrix/07_cnn', "19_cnn_confusion_matrix.png"),
    dpi=300
)


plt.close()



print()

print(
    "混淆矩阵图片保存完成"
)

print(
    output_path('results/confusion_matrix/07_cnn', "19_cnn_confusion_matrix.png")
)