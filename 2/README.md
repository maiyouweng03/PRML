# PRML Homework 2: 3D 数据集分类实验

本项目围绕一个程序生成的三维双月形分类数据集展开，比较了 `Decision Tree`、`AdaBoost + Decision Tree` 与 `SVM` 在该任务上的分类性能，并给出了测试结果、混淆矩阵与可视化分析图。

## 项目内容

- 数据集由 [3D数据集.py](D:\PRML\2\3D数据集.py) 生成
- 训练与评估脚本为 [训练代码evaluate_models.py](D:\PRML\2\训练代码evaluate_models.py)
- 实验结果汇总保存在 [训练结果evaluation_results.json](D:\PRML\2\训练结果evaluation_results.json)
- 最终报告为 [报告.pdf](D:\PRML\2\报告.pdf)
- 可视化图片位于 [visualizations可视化结果图](D:\PRML\2\visualizations可视化结果图)


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


