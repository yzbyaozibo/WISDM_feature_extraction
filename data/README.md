# 数据准备

数据来源：[WISDM Activity Prediction](https://www.cis.fordham.edu/wisdm/dataset.php)。请核对原始下载包的版本及说明；本地文件名 WISDM_raw.txt 不能单独证明版本。

- `raw/WISDM_raw.txt`：原始数据，仅本地保留。
- `processed/01_cleaning/01_clean_data.csv`：目前流程的必需输入。
- `processed/04_features/`：中间特征与最终 43 维特征。
- `processed/05_split/`：受试者独立划分。
- `processed/07_cnn/`：CNN 训练和测试 NPZ。
- `sample/`：格式说明，不包含真实数据或可用于训练的样本。

当前仓库未找到生成清洗 CSV 的源码，因此仅下载原始数据不能直接开始完整流程。需要补充原清洗实现，或自行提供与现有实验完全一致的清洗 CSV；不要用未经验证的新清洗规则替代原流程。

原始与处理后数据默认不提交 Git。引用数据集时请引用 Kwapisz、Weiss、Moore 的论文 “Activity Recognition using Cell Phone Accelerometers”（2010）。官方要求再分发数据时包含原 readme.txt。
