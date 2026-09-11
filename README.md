# WISDM Feature Extraction and CNN Classification

基于 WISDM 三轴加速度数据的活动识别实验：数据统计、200 点窗口、43 维特征、受试者独立划分、KNN / RandomForest 比较，以及融合三轴 CNN 与统计特征的分类模型。

## 当前流程范围与复现限制

当前仓库覆盖：从已有清洗 CSV 开始的数据统计、时间戳检查、窗口划分、43 维特征提取、受试者独立划分、KNN / RandomForest 实验、CNN 输入准备与训练、训练曲线和混淆矩阵绘制。

原始清洗源码尚未包含在仓库中，当前本地 WISDM 数据的精确版本及来源记录也尚未确认。**不保证从原始 WISDM 数据完全复现实验结果，不声称完全端到端复现。** 下载官方原始数据不能替代缺失的原清洗流程。

## 数据集

使用 Walking、Jogging、Upstairs、Downstairs、Sitting、Standing 六类标签。来源及引用见 [WISDM 官方数据页面](https://www.cis.fordham.edu/wisdm/dataset.php) 和 [数据准备说明](data/README.md)。本地文件的精确来源版本仍需作者确认。

## 环境与安装

推荐 Python **3.11 64-bit**、TensorFlow **2.16.1**、Keras **3.3.3**。主要依赖固定于 `requirements.txt`，包含 SciPy。现有 `.venv` 是 Python 3.14 数据处理环境，保持不变；现有 `cnn_env` 是推荐 CNN 环境。

已有本地环境无需重新安装，直接使用 `cnn_env/Scripts/python.exe`。新使用者在仓库根目录执行：

```powershell
py -3.11 -m venv .venv-public
.\.venv-public\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python -c "import tensorflow as tf; from tensorflow.keras import Model; print(tf.__version__)"
```

默认镜像失败时安装命令增加 `--index-url https://pypi.org/simple`。`requirements-cnn.txt` 保留为此前的 CNN 依赖清单；完整推荐安装入口为 `requirements.txt`。

TensorFlow **2.21.0 / Keras 3.15.1** 是作者的个人测试环境，不作为默认依赖或已验证的推荐环境，需独立保留。推荐环境始终为 `requirements.txt` 中的 **2.16.1 / 3.3.3**。

未升级或删除任何现有环境。本次检查未在项目内定位到 2.21.0 / 3.15.1 环境，因此没有验证其安装位置或运行兼容性，也没有新建或改动该环境。`cnn_env_old_20260911` 登记的是 2.16.1 / 3.3.3，不应当作该个人测试环境。

## 项目结构

```text
src/
  paths.py                 项目根目录与输出路径解析
  preprocessing/           02、03、04：数据统计、时间戳、窗口统计
  feature_extraction/      05、07～12：特征工程及检查
  analysis/                13～15：划分与传统模型；20：历史汇总
  cnn/                     16～19：CNN 输入、训练、绘图
    legacy/                另一版 CNN 脚本，保留原名
tools/                    历史目录整理工具
data/
   raw/                    原始文件（忽略）
   processed/              清洗、特征、划分、NPZ（忽略）
   sample/                 样例说明
models/                   本地模型（模型文件忽略）
results/
   figures/                训练曲线
   metrics/                统计、预测、训练历史、分类指标
   confusion_matrix/       混淆矩阵 CSV 和图片
logs/                     既有日志（忽略）
```

`results/`、`logs/` 和环境目录仅在本地保留。Git 不保存空目录；输出父目录由脚本按需创建。旧 `output/` 可能保留空目录，已不属于当前数据流程且被 Git 忽略。

## 数据处理与特征提取

当前流水线从 `data/processed/01_cleaning/01_clean_data.csv` 开始。原始清洗源码缺失，不能声称支持从原始下载文件端到端复现。`tools/feature_extraction.py` 只是历史文件整理工具，不执行特征提取或清洗；其工作目录为 `data/processed/legacy_output/`，无需在主流程运行。

在已激活推荐环境后，从项目根目录依次执行：

```powershell
python src/preprocessing/02_statistics.py
python src/preprocessing/03_timestamp_check.py
python src/preprocessing/04_windows.py
python src/feature_extraction/05_feature_mean.py
python src/feature_extraction/07_feature_abso.py
python src/feature_extraction/08_feature_resultant.py
python src/feature_extraction/09_feature_bins.py
python src/feature_extraction/10_feature_peak.py
python src/feature_extraction/11_feature_quality_check.py
python src/feature_extraction/12_peak_missing_analysis.py
python src/analysis/13_subject_window_distribution.py
python src/analysis/14_subject_independent_split.py
python src/analysis/15_nan_strategy_comparison.py
```

保留原有编号；不补造 01、06 算法。200 点窗口按原脚本规则构建，提取均值、绝对偏差、标准差、合加速度、分箱及峰值相关特征，形成 43 维输入。固定测试受试者为 5、6、7、13、24、33、34。15 比较既有缺失值策略及 KNN / RandomForest；缺失值拟合使用训练数据。

## CNN 训练与可视化

```powershell
python src/cnn/16_prepare_cnn_data.py
python src/cnn/17_train_cnn.py
python src/cnn/18_cnn_training_curve.py
python src/cnn/19_cnn_confusion_matrix.py
```

16 从清洗数据及最终特征重新构建对应窗口，输出 `data/processed/07_cnn/16_cnn_{train,test}_data.npz`。17 保留三轴卷积分支与 43 维统计输入融合，训练 50 epochs、batch size 32，输出 Accuracy、Macro Recall、Macro F1 和分类报告。

- 模型：`models/17_cnn_model.keras`
- 历史：`results/metrics/07_cnn/17_cnn_history.csv`
- 混淆矩阵：`results/confusion_matrix/07_cnn/17_cnn_confusion_matrix.csv`
- 曲线：`results/figures/07_cnn/18_cnn_*_curve.png`
- 混淆矩阵图片：`results/confusion_matrix/07_cnn/19_cnn_confusion_matrix.png`

已有历史和矩阵时，可只运行 18、19，无需训练。整理本身没有运行 15、16、17、20，没有重新训练或改写现有结果。脚本路径通过自身位置解析，直接运行脚本不依赖工作目录；PyCharm 的 CNN 运行配置已同步新脚本位置。

## 模型兼容性与复现边界

现有 `.keras` 元数据记录 Keras 3.3.3，已在推荐环境加载成功；输入为三个 `(200, 1)` 轴分支和一个 `(43,)` 统计分支，输出 6 类。迁移保持文件字节不变，没有进行格式转换。未验证 2.21.0 / 3.15.1 加载行为，不宣称兼容。

`src/analysis/20_final_results_summary.py` 保留原文件作为历史版本，并在文件开头及 CNN 指标定义处明确标注。Accuracy=0.88、Macro Recall=0.86、Macro F1=0.84 是**历史硬编码值**，不是当前训练的实测指标。该脚本及其汇总输出**不作为当前实验结果来源**，不属于推荐运行流程。当前结果应追溯到对应训练运行的实际输出与产物；不得将历史硬编码值替代实测值。历史备用 CNN 脚本也不作为默认入口，运行可能覆盖同名模型及结果。

原 CNN 脚本只设置 NumPy 随机种子，未完整固定 TensorFlow 随机性；重新训练不保证逐位复现。原流程将测试集传入训练的 validation_data；本次未调整实验设计。Python 3.14 数据处理环境与推荐 Python 3.11 环境的完整流水线数值等价性未重新验证。

## GitHub 上传前检查

- [x] 环境、IDE、原始/处理后数据、结果及日志已设置忽略。
- [x] 源码保留，路径改为仓库相对定位，模型未转换。
- [ ] 补充原始清洗源码并确认数据版本，才能声称端到端复现。
- [ ] 由作者确认 20 的真实结果来源；当前不应引用其硬编码 CNN 指标。
- [x] 添加 MIT LICENSE，适用于仓库代码；数据集及第三方组件遵循各自许可。
- [ ] 补充论文及作者引用信息。
- [ ] 首次提交前检查暂存清单、敏感信息和大文件；不要强制提交环境或数据。
- [ ] 如需公开模型或结果，单独决定 Git LFS / Release 分发及数据使用条款。

本次仅整理本地仓库内容，未创建远程仓库或执行上传。

## 许可证

仓库代码采用 [MIT License](LICENSE)。该许可不替代 WISDM 数据集和第三方依赖各自的使用与分发条款。
