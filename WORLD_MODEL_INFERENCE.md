# LTX-2.5 白膜机械臂到 RGB 世界模型推理

入口为 `scripts/infer_white_robot_iclora.py`。它用首帧 RGB 图像固定初始观测，用 `white_robot.mp4` 作为时序动作控制条件，通过 LTX-2.5 生成 RGB 视频。

## 前提：需要专用 IC-LoRA

基础 LTX-2.5 没有学习“白膜机械臂轨迹 -> RGB 机械臂和场景”的映射。官方 `ICLoraPipeline` 需要加载针对白膜条件训练的 **IC-LoRA**。已有的 `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` 仅用于蒸馏采样，**不能**传给 `--ic-lora-path`。

请先训练或获取类似下列 checkpoint：

```text
/lpai/models/robot_world_model/white_robot_to_rgb_iclora.safetensors
```

没有该 checkpoint 时，白膜视频不会成为可靠运动控制，生成结果不适合用作 VLA 世界模型训练数据。

## 配置环境

```bash
cd /lpai/volumes/mind-eb-ali-sh/dulingyi/fang/LTX-2
bash scripts/setup_world_model_env.sh
```

脚本会创建复用服务器预装 CUDA PyTorch 的 `.venv`，仅安装 LTX 运行所缺的 Python
依赖，并检查 CUDA、PyAV 和 `ICLoraPipeline`。它不会重新下载 PyTorch 或安装与当前
PyTorch 版本不匹配的 `natten` 或 `torchaudio`。因此此环境面向 RGB 视频推理；若后续需要
音频输出，必须另行安装与服务器 PyTorch ABI 完全匹配的 `torchaudio`。LTX-2.5 组件默认来自：

```text
/lpai/models/Lightricks__LTX-2_5/26-08-11-2005
```

## RoboTwin 单条推理

`--episode` 会自动读取 `head_color.mp4`（抽首帧）、`white_robot.mp4`（控制视频）及 `instruction.txt`（指令）。

```bash
cd /lpai/volumes/mind-eb-ali-sh/dulingyi/fang/LTX-2
.venv/bin/python scripts/infer_white_robot_iclora.py \
  --episode blocks_ranking_size/aloha-agilex_clean_50/episode30 \
  --ic-lora-path /lpai/models/robot_world_model/white_robot_to_rgb_iclora.safetensors \
  --output-dir outputs/blocks_ranking_size/episode30 \
  --height 384 --width 512 --num-frames 121 --frame-rate 24 --seed 42
```

LTX 帧数必须是 `8k+1`，例如 `121`、`241` 或 `361`；两阶段 IC-LoRA 的输出宽高均必须为
`64` 的倍数。450 帧源视频请按 `121` 或 `241` 帧滑动窗口生成，再在后处理中衔接，不能直接填 `450`。

## 自定义输入

```bash
.venv/bin/python scripts/infer_white_robot_iclora.py \
  --first-frame /path/to/first_rgb.png \
  --reference-video /path/to/white_robot.mp4 \
  --prompt "Move the robotic arm toward the object while keeping the workspace unchanged." \
  --ic-lora-path /lpai/models/robot_world_model/white_robot_to_rgb_iclora.safetensors \
  --output-path outputs/custom_rgb.mp4
```

也可用 `--rgb-video` 自动抽取首帧。添加 `--dry-run` 可先检查即将调用的官方 LTX 命令。

## 显存建议

服务器有 8 张 32GB RTX 5090。默认使用 `--quantization fp8-cast --offload cpu`，适合单卡安全启动。显存充足时可加 `--quantization none`；调试时可加入 `--skip-stage-2`。
