# LLM Post-Training Hub

用于验证 LLM post-training 效果的小实验。第一个实验采用 MLX + SFT + LoRA，模型和数据集待定。

## 环境

- 平台：Apple Silicon Mac（原生 arm64），macOS 14 或更新版本。
- Python：3.14.6，由 `.python-version` 固定；项目限定为 Python 3.14 系列。
- 环境与依赖管理：uv；虚拟环境位于仓库根目录的 `.venv/`。
- 训练依赖：`mlx` 和 `mlx-lm[train]`，后者包含数据处理所需的 `datasets` 等依赖。
- `uv.lock` 固定实际解析的依赖版本，应纳入版本控制。

Python 3.14 是当前稳定维护版本，MLX 提供 CPython 3.14 的 macOS ARM64 安装包。
参考：[Python 版本状态](https://devguide.python.org/versions/) · [MLX 安装说明](https://ml-explore.github.io/mlx/build/html/install.html)。

## 使用

在仓库根目录执行：

```bash
uv sync --locked
uv run python --version
uv run mlx_lm.lora --help
```

`uv run` 会使用项目虚拟环境，无需手动激活。需要在交互式终端激活时可执行：

```bash
source .venv/bin/activate
```

快速检查 Metal GPU 是否可用：

```bash
uv run python -c 'import mlx.core as mx; print("Metal available:", mx.metal.is_available())'
```

## 实验目录

- [`sft/`](sft/README.md)：第一个 SFT + LoRA 实验。

模型、数据和训练产物由 `.gitignore` 排除；实验代码、配置及结果说明应纳入版本控制。
