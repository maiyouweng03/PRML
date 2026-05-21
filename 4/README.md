# Transformer 复现与残差连接消融实验

本项目围绕经典论文 `Attention Is All You Need` 展开，实现一个小型 Transformer 编码器-解码器模型，并在合成序列转导任务上验证模型结构。项目分为两部分：一是标准 Transformer 复现，二是残差连接消融实验。

## 文件结构

```text
D:\PRML\4
├── 原论文.pdf
├── PRML报告4.docx
├── PRML报告4.pdf
├── 代码
│   ├── reproduce_transformer.py
│   └── residual_ablation_experiment.py
├── 结果数据
│   ├── results_reproduction
│   │   ├── config.json
│   │   ├── history.csv
│   │   ├── results.json
│   │   └── reproduction_training_curve.png
│   ├── results_residual_ablation
│   │   ├── config.json
│   │   ├── history.csv
│   │   ├── results.json
│   │   └── residual_ablation_curves.png
│   ├── results_reproduction_quick
│   └── results_residual_ablation_quick
└── README.md
```

`__pycache__` 为 Python 运行时自动生成的缓存目录，可忽略。

## 代码说明

`代码/reproduce_transformer.py` 用于复现标准 Transformer。脚本实现词嵌入、sinusoidal 位置编码、多头注意力、前馈网络、残差连接、LayerNorm、编码器层和解码器层。运行后会训练标准模型，并自动保存训练曲线图。

`代码/residual_ablation_experiment.py` 用于研究残差连接的作用。该脚本在相同任务和超参数下比较两个模型：

- `transformer`：标准结构，子层输出为 `LayerNorm(x + Sublayer(x))`
- `no_residual`：移除残差支路，子层输出为 `LayerNorm(Sublayer(x))`

两份脚本均使用同一个 reverse-copy 合成序列转导任务。输入为随机 token 序列，输出为输入序列的反转结果并追加 `EOS`。

```latex
x=[x_1,x_2,\ldots,x_n],\quad y=[x_n,x_{n-1},\ldots,x_1,\mathrm{EOS}]
```

## 环境依赖

主要依赖如下：

- Python 3.11 或兼容版本
- PyTorch
- Pillow



## 运行方式

进入代码目录：

```powershell
cd D:\PRML\4\代码
```

快速检查标准 Transformer 复现流程：

```powershell
& 'C:\Users\Lu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' reproduce_transformer.py --quick --out-dir ..\结果数据\results_reproduction_quick
```

正式运行标准 Transformer 复现：

```powershell
& 'C:\Users\Lu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' reproduce_transformer.py --steps 400 --seq-len 5 --out-dir ..\结果数据\results_reproduction
```

快速检查残差连接消融流程：

```powershell
& 'C:\Users\Lu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' residual_ablation_experiment.py --quick --out-dir ..\结果数据\results_residual_ablation_quick
```

正式运行残差连接消融实验：

```powershell
& 'C:\Users\Lu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' residual_ablation_experiment.py --steps 400 --seq-len 5 --out-dir ..\结果数据\results_residual_ablation
```

## 输出文件说明



- `config.json`：实验配置，包括序列长度、模型维度、层数、学习率等
- `history.csv`：训练过程记录，包括训练损失、评估损失、token accuracy 和 sequence accuracy
- `results.json`：最终评估结果
- `*.png`：训练过程曲线图

标准 Transformer 复现图为：

```text
结果数据\results_reproduction\reproduction_training_curve.png
```

残差连接消融对比图为：

```text
结果数据\results_residual_ablation\residual_ablation_curves.png
```

## 当前正式结果

标准 Transformer 复现结果：

| 模型 | Loss | Token Accuracy | Sequence Accuracy | 参数量 |
| --- | ---: | ---: | ---: | ---: |
| Transformer | 0.3020 | 89.19% | 46.25% | 257,312 |

残差连接消融结果：

| 模型 | Loss | Token Accuracy | Sequence Accuracy | 参数量 |
| --- | ---: | ---: | ---: | ---: |
| Transformer | 0.3020 | 89.19% | 46.25% | 257,312 |
| No Residual | 2.6280 | 17.50% | 0.00% | 257,312 |

结果显示，标准 Transformer 能够稳定学习 reverse-copy 序列映射；移除残差连接后，模型仍能学习到少量局部 token 规律，但无法形成稳定的序列级反转能力。

## 报告文件

`PRML报告4.docx` 和 `PRML报告4.pdf` 为最终报告文件，内容包括论文背景、模型原理、复现流程、实验设计、结果图表和结果讨论。
