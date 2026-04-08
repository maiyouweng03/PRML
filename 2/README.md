# PRML Homework 2: 3D 数据集分类实验

本项目围绕一个程序生成的三维双月形分类数据集展开，比较了 `Decision Tree`、`AdaBoost + Decision Tree` 与 `SVM` 在该任务上的分类性能，并给出了测试结果、混淆矩阵与可视化分析图。

## 项目内容

- 数据集由 [3D数据集.py](D:\PRML\2\3D数据集.py) 生成
- 训练与评估脚本为 [训练代码evaluate_models.py](D:\PRML\2\训练代码evaluate_models.py)
- 实验结果汇总保存在 [训练结果evaluation_results.json](D:\PRML\2\训练结果evaluation_results.json)
- 最终报告为 [报告.pdf](D:\PRML\2\报告.pdf)
- 可视化图片位于 [visualizations可视化结果图](D:\PRML\2\visualizations可视化结果图)

## 数据集说明

原始生成函数 `make_moons_3d` 会为每个类别分别生成 `n_samples` 个点，因此总样本数为 `2 * n_samples`。

本实验按照作业要求使用：

- 训练集：1000 个样本
  - `C0` 类 500 个
  - `C1` 类 500 个
- 测试集：500 个样本
  - `C0` 类 250 个
  - `C1` 类 250 个

数据的生成过程具有明显的非线性特征：

- 第一类样本由 `(1.5 cos t, sin t, sin 2t)` 构成
- 第二类样本由 `(-1.5 cos t, sin t - 1, -sin 2t)` 构成
- 再叠加高斯噪声 `noise = 0.2`

因此，该任务是一个典型的非线性分类问题。

## 模型设置

本项目比较了以下模型：

- `Decision Tree`
- `AdaBoost + Decision Tree`
- `SVM (linear)`
- `SVM (poly)`
- `SVM (rbf)`
- `SVM (sigmoid)`

其中：

- 决策树搜索了 `criterion`、`max_depth`、`min_samples_split`、`min_samples_leaf`
- AdaBoost 搜索了基学习器深度、`n_estimators` 与 `learning_rate`
- SVM 结合 `Pipeline` 与 `StandardScaler`，并分别对不同核函数搜索关键超参数
- 所有模型均采用 `5-fold Stratified Cross Validation`

## 主要结果

按测试集准确率排序，结果如下：

| 模型 | CV Accuracy | Test Accuracy | Macro-F1 |
|---|---:|---:|---:|
| SVM (rbf) | 0.9830 | 0.9780 | 0.9780 |
| SVM (sigmoid) | 0.9490 | 0.9780 | 0.9780 |
| AdaBoost + Decision Tree | 0.9810 | 0.9760 | 0.9760 |
| SVM (poly) | 0.9800 | 0.9760 | 0.9760 |
| Decision Tree | 0.9580 | 0.9520 | 0.9520 |
| SVM (linear) | 0.6780 | 0.6820 | 0.6820 |

结论上可以看到：

- `RBF` 核 SVM 表现最好
- `AdaBoost + Decision Tree` 明显优于单棵决策树
- `linear` 核 SVM 表现最差，说明该问题并非线性可分

## 可视化结果

项目已经生成了多种便于分析和写报告的图片：

- [dataset_distribution.png](D:\PRML\2\visualizations可视化结果图\dataset_distribution.png)
  - 展示训练集与测试集的三维数据分布
- [model_metric_comparison.png](D:\PRML\2\visualizations可视化结果图\model_metric_comparison.png)
  - 对比各模型的交叉验证准确率、测试准确率和 Macro-F1
- [confusion_matrices_overview.png](D:\PRML\2\visualizations可视化结果图\confusion_matrices_overview.png)
  - 汇总展示所有模型的混淆矩阵
- [top3_prediction_quality.png](D:\PRML\2\visualizations可视化结果图\top3_prediction_quality.png)
  - 展示前三名模型在测试集上的预测正确与错误分布

此外，还导出了每个模型单独的混淆矩阵图，例如：

- [confusion_matrix_svm_rbf.png](D:\PRML\2\visualizations可视化结果图\confusion_matrix_svm_rbf.png)
- [confusion_matrix_adaboost_plus_decision_tree.png](D:\PRML\2\visualizations可视化结果图\confusion_matrix_adaboost_plus_decision_tree.png)
- [confusion_matrix_decision_tree.png](D:\PRML\2\visualizations可视化结果图\confusion_matrix_decision_tree.png)

## 运行方法

建议使用你当前的 Anaconda Python 环境：

```powershell
D:\anaconda3\python.exe D:\PRML\2\训练代码evaluate_models.py
```

脚本运行后会完成以下工作：

- 生成训练集和测试集
- 对各模型进行网格搜索和交叉验证
- 在测试集上输出准确率、Macro-F1 和混淆矩阵
- 生成 `evaluation_results.json`
- 生成若干可视化图片

## 文件说明

- [3D数据集.py](D:\PRML\2\3D数据集.py)
  - 原始 3D 数据集生成脚本
- [训练代码evaluate_models.py](D:\PRML\2\训练代码evaluate_models.py)
  - 训练、调参与可视化主脚本
- [训练结果evaluation_results.json](D:\PRML\2\训练结果evaluation_results.json)
  - 实验指标、最优参数、预测结果与混淆矩阵汇总
- [visualizations可视化结果图](D:\PRML\2\visualizations可视化结果图)
  - 图像化结果输出目录
- [报告.pdf](D:\PRML\2\报告.pdf)
  - 最终实验报告

## 说明

当前脚本中的输出目录变量仍写为 `visualizations`，而项目中保留的结果目录名称为 `visualizations可视化结果图`。这说明结果文件后续经过了人工整理或重命名。若你重新运行脚本，默认可能会重新生成一个新的 `visualizations` 文件夹；如果希望与当前目录保持一致，可以将脚本中的：

```python
OUTPUT_DIR = Path("visualizations")
```

改为：

```python
OUTPUT_DIR = Path("visualizations可视化结果图")
```

## 实验结论

对于该三维双月数据集，分类性能更好的方法通常具备更强的非线性建模能力。

- `RBF` 核 SVM 最适合刻画平滑而复杂的非线性边界
- `AdaBoost + Decision Tree` 通过集成学习提升了决策树的泛化能力
- 单棵决策树虽然能处理非线性，但性能略弱
- 线性核 SVM 无法有效处理该类弯曲分布，因此表现明显较差

这也说明，在带噪声且边界非线性的分类任务中，模型表达能力对最终效果影响很大。
