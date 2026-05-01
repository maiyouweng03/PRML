# 基于多变量 LSTM 的空气污染时间序列预测

本项目基于 `LSTM-Multivariate_pollution.csv` 空气质量数据集，使用多变量长短期记忆网络（LSTM）对未来 1 小时的 PM2.5 浓度进行预测。项目包含原始数据、训练与预测代码、模型权重、结果图表以及课程报告文档，适合作为时间序列预测课程作业或复现实验项目使用。

## 项目目标

- 使用污染浓度与气象变量联合建模，而不是仅依赖单变量污染历史序列。
- 利用过去 `24` 小时的多变量观测序列预测下一小时的 PM2.5 浓度。
- 输出模型评估指标、预测结果表以及关键可视化图表。
- 支持课程报告撰写与实验结果复现。

## 项目结构

```text
D:\PRML\3
├── LSTM-Multivariate_pollution.csv        # 训练数据集
├── pollution_test_data1.csv               # 外部测试数据集
├── 代码.py                                 # 主训练与预测脚本
├── PRML报告3.docx                         # Word 版课程报告
├── PRML报告3.pdf                          # PDF 版课程报告
├── README.md                              # 项目说明文档
└── 模型及结果图表
    ├── lstm_pollution_model.pt            # 训练好的 LSTM 模型参数
    ├── holdout_predictions.csv            # 验证集预测结果
    ├── pollution_test_predictions.csv     # 外部测试集预测结果
    ├── training_loss_curve.png            # 训练/验证损失曲线
    ├── holdout_prediction_curve.png       # 验证集最后 168 小时预测对比图
    ├── test_prediction_curve.png          # 外部测试集预测对比图
    ├── prediction_scatter.png             # 真实值与预测值散点图
    └── prediction_error_histogram.png     # 预测误差分布直方图
```

## 数据说明

### 1. 训练数据 `LSTM-Multivariate_pollution.csv`

训练集包含逐小时空气质量与气象观测信息，主要字段如下：

- `date`：日期时间
- `pollution`：PM2.5 浓度
- `dew`：露点
- `temp`：气温
- `press`：气压
- `wnd_dir`：风向
- `wnd_spd`：风速
- `snow`：累计降雪时长
- `rain`：累计降雨时长

### 2. 测试数据 `pollution_test_data1.csv`

该文件作为外部测试集使用，用于检验模型在新样本上的泛化表现。字段与训练集基本一致。

## 模型说明

项目采用基于 PyTorch 实现的多变量 LSTM 回归模型，核心结构如下：

- 输入：过去 `24` 小时的多变量时序样本
- LSTM 层数：`2`
- 隐藏单元数：`64`
- Dropout：`0.2`
- 输出层：全连接层，输出下一小时 PM2.5 预测值

### 模型输入形式

模型输入张量形状为：

```text
(batch_size, 24, feature_dim)
```

其中：

- `24` 表示时间窗口长度
- `feature_dim` 表示污染浓度与气象特征拼接后的输入维度

### 主要训练参数

- 预测步长：`1` 小时
- 批量大小：`64`
- 训练轮数：`15`
- 学习率：`1e-3`
- 训练集划分比例：`80%`

## 代码流程说明

主脚本为 [代码.py](/D:/PRML/3/代码.py)，运行后会自动完成以下步骤：

1. 读取训练集与测试集。
2. 对 `pollution`、`dew`、`temp`、`press`、`wnd_spd`、`snow`、`rain` 等数值列做类型转换。
3. 将训练集中的 `pollution=0` 视作缺失值，并进行前向/后向填充。
4. 对风向 `wnd_dir` 做独热编码。
5. 对数值特征执行标准化。
6. 用长度为 `24` 的滑动窗口构造监督学习样本。
7. 训练双层 LSTM 模型。
8. 在验证集和外部测试集上进行预测并计算评估指标。
9. 导出预测结果表、模型权重以及关键图表。

## 如何运行

在当前项目目录下，使用可用的 Python 环境运行：

```powershell
D:\anaconda3\python.exe D:\PRML\3\代码.py
```

运行完成后，脚本会在目录中生成：

- 模型参数文件
- 验证集预测结果
- 外部测试集预测结果
- 五张关键结果图

## 输出结果说明

### 1. 模型文件

- `lstm_pollution_model.pt`

说明：

- 保存训练好的模型参数
- 同时保存输入特征名、时间窗口长度、预测步长以及标准化所需的均值和标准差

### 2. 预测结果表

- `holdout_predictions.csv`
  - 验证集预测结果
  - 包含 `date`、`actual_pollution`、`predicted_pollution`

- `pollution_test_predictions.csv`
  - 外部测试集预测结果
  - 包含 `test_row_index`、`actual_pollution`、`predicted_pollution`

## 图表说明

### `training_loss_curve.png`

训练集损失与验证集损失曲线。

用途：

- 用于观察模型收敛过程
- 判断训练是否稳定
- 辅助分析是否存在明显过拟合

### `holdout_prediction_curve.png`

验证集最后 `168` 小时真实值与预测值对比图。

用途：

- 观察模型在训练集后段时间上的拟合效果
- 查看模型对污染变化趋势的跟踪能力

### `test_prediction_curve.png`

外部测试集真实值与预测值走势对比图。

用途：

- 评估模型在新样本上的泛化效果
- 观察测试集上整体趋势拟合是否稳定

### `prediction_scatter.png`

真实 PM2.5 与预测 PM2.5 的散点图，并带有理想对角线。

用途：

- 判断预测值与真实值的一致性
- 观察高污染样本区域是否存在偏差增大现象

### `prediction_error_histogram.png`

预测误差分布直方图。

用途：

- 观察误差是否集中在 0 附近
- 判断是否存在较明显的偏差方向或长尾误差样本

## 评估指标说明

项目使用以下回归指标评价模型性能：

- `RMSE`：均方根误差，反映整体误差水平
- `MAE`：平均绝对误差，反映平均预测偏差
- `MAPE`：平均绝对百分比误差，反映相对误差大小

根据当前实验结果：

- 验证集：`RMSE=20.507`，`MAE=11.693`，`MAPE=23.07%`
- 外部测试集：`RMSE=26.496`，`MAE=14.864`，`MAPE=28.74%`

## 文档说明

- [PRML报告3.docx](/D:/PRML/3/PRML报告3.docx)
  - 项目的 Word 课程报告
  - 包含研究背景、方法、实验结果和结论

- [PRML报告3.pdf](/D:/PRML/3/PRML报告3.pdf)
  - 课程报告的 PDF 版本
  - 便于提交和打印

## 适用场景

本项目适用于：

- 机器学习/模式识别课程作业
- 时间序列预测实验
- LSTM 多变量建模入门案例
- 空气质量预测任务的基线方法参考

## 后续可改进方向

- 增加更长时间窗口，观察长依赖对预测效果的影响
- 改为多步预测，而不仅预测下一小时
- 对比 GRU、BiLSTM、Transformer 等时序模型
- 引入更多外部变量，例如季节、节假日、空气质量等级等
- 使用早停、学习率调度等训练策略进一步优化模型
