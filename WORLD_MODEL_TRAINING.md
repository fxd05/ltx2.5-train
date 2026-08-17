# 白膜机械臂到 RGB 的 IC-LoRA 训练

该训练链路以 `head_color.mp4` 作为 RGB 目标视频、`white_robot.mp4` 作为参考控制视频、
`instruction.txt` 作为文本条件，训练 LTX-2.5 的视频到视频 IC-LoRA。

## 训练环境

```bash
cd /lpai/volumes/mind-eb-ali-sh/dulingyi/fang/LTX-2
bash scripts/setup_world_model_training_env.sh
```

环境复用服务器预装的 CUDA PyTorch。它不安装不兼容的 `torchaudio`，因此预处理和训练均为
视频-only 模式，必须保留 `--skip-audio`。

## 数据预处理

训练 manifest 已生成到 `data/white_robot_iclora_train.jsonl`，共包含 `3283` 条配对样本。
完整预处理会生成目标 latent、文本条件和白膜参考 latent：

```bash
bash scripts/prepare_white_robot_iclora_data.sh
```

建议先进行小规模冒烟测试，并使用单独的输出目录：

```bash
MAX_SAMPLES=8 \
PREPROCESSED_ROOT=/lpai/volumes/mind-eb-ali-sh/dulingyi/fang/LTX-2/data/smoke_preprocessed \
bash scripts/prepare_white_robot_iclora_data.sh
```

完整预处理的默认输出为 `data/white_robot_iclora_preprocessed_512x384x121`，训练数据规格为
`512x384`、`121` 帧、`24 FPS`。

## 训练

配置文件为 `configs/white_robot_to_rgb_iclora.yaml`：LTX-2.5、rank 32、BF16、3000 steps，
使用 `reference_latents` 作为概率 1.0 的 IC-LoRA 控制条件。

```bash
NUM_PROCESSES=8 bash scripts/train_white_robot_iclora.sh
```

脚本使用 `packages/ltx-trainer/configs/accelerate/fsdp.yaml` 和 8 张 RTX 5090。训练输出写入
`outputs/white_robot_to_rgb_iclora`。

训练完成后，将导出的 IC-LoRA checkpoint 放入：

```text
/lpai/models/robot_world_model/white_robot_to_rgb_iclora.safetensors
```

然后使用 `scripts/infer_white_robot_iclora.py` 以白膜视频驱动 RGB 世界模型推理。
