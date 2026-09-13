# SFT + LoRA

目标：在同一模型和独立测试集上比较微调前后的表现，验证监督微调的效果。

- 框架：MLX / MLX LM。
- 方法：SFT + LoRA。
- 基础模型、数据集、评价指标：待定。
- Python 环境：复用仓库根目录的 uv 环境。

从仓库根目录查看训练工具的参数：

```bash
uv run mlx_lm.lora --help
```

确定模型和数据集后，在此目录添加数据准备、训练配置和评估代码。首次实验应保留固定的训练、验证和测试划分，并用相同的评估设置比较微调前后的结果。

参考：[MLX LM LoRA 文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)。
