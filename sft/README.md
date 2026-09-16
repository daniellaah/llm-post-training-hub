# SFT + LoRA

目标：在同一模型和独立测试集上比较微调前后的表现，验证监督微调的效果。

- 框架：MLX / MLX LM。
- 方法：SFT + LoRA，只对答案部分计算 loss。
- 实验：[gsm8k/](gsm8k/) —— Qwen3-0.6B-Base 在 GSM8K 上的微调，训练数据为 Qwen3-4B 蒸馏的分步解答。
- Python 环境：复用仓库根目录的 uv 环境。

评测原则：Base 模型用 few-shot 提示作为公平基线（零样本时它不会输出答案格式，分数只反映格式而非能力），微调后的模型用零样本对比；两者使用相同的解码设置、答案抽取规则和全部测试集。

参考：[MLX LM LoRA 文档](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)。
