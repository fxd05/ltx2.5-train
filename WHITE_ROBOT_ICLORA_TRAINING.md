# 白膜机械臂到 RGB 世界模型训练指南

本指南说明如何在 LTX-2.5 上训练一个 **V2V IC-LoRA**：输入 RGB 首帧、白膜机械臂动作参考视频和文本指令，输出 RGB 机械臂视频。本文档对应当前仓库中已经验证过的 RoboTwin 数据、L20X 单卡训练和本地 TensorBoard 工作流。

## 1. 任务定义

每个训练样本包含三种输入：

- **目标 RGB 视频**：`head_color.mp4`。训练损失只在该视频的可生成 latent token 上计算。
- **白膜参考视频**：`white_robot.mp4`。作为 IC-LoRA 的冻结参考条件，用来传递机器人动作轨迹；它本身不参与重建损失。
- **文本指令**：`instruction.txt`。作为文本条件，例如抓取瓶子的描述。

推理时还额外使用目标 RGB 视频的首帧作为 image condition。训练后的 IC-LoRA 才能把 `white_robot.mp4` 解释为 RGB 机器人的动作控制；原始 LTX-2.5 基座模型不具备这个白膜视频条件能力。

## 2. 路径与分辨率约束

当前 LTX 服务器约定路径：

```bash
export REPO=/lpai/volumes/mind-eb-ali-sh/dulingyi/fang/LTX-2
export MODELS=/lpai/models/Lightricks__LTX-2_5/26-08-11-2005
export DATA=/lpai/volumes/mind-eb-ali-sh/dulingyi/sunyilu/RoboTwin2_0_cleaned_full/white
cd "$REPO"
```

训练数据目录中，每个 episode 至少需要：

```text
<DATA>/<task>/<variant>/episode<N>/
├── head_color.mp4     # RGB target video
├── white_robot.mp4    # white robot reference/control video
└── instruction.txt    # prompt
```

本项目训练目标为：

- 空间尺寸：`640×480`（宽 × 高）
- 时间长度：`121` 帧
- 帧率：`24 fps`

LTX 视频 VAE 的时间约束是 `num_frames % 8 == 1`，因此 `121` 是合法值。训练预处理可以使用 `640x480x121`；视频宽高均需能被 VAE 空间压缩因子 `32` 整除。

> 原生双阶段 IC-LoRA 推理器要求最终推理尺寸可被 `64` 整除，`640×480` 中的 `480` 不满足这一限制。当前推理流程会先生成 `640×512`，再无重采样地裁剪上下各 `16` 像素，得到 `640×480` 输出。

## 3. 环境准备

服务器已经安装 CUDA PyTorch 时，使用系统 PyTorch 创建带本地源码路径的虚拟环境：

```bash
cd "$REPO"
bash scripts/setup_world_model_training_env.sh
```

快速检查：

```bash
.venv/bin/python - <<'PY'
import torch
import accelerate
import bitsandbytes
import peft
from ltx_trainer.config import LtxTrainerConfig
from torch.utils.tensorboard import SummaryWriter

print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print("training environment ready")
PY
```

如果服务器预装 PyTorch 不在默认 `python3` 中，显式指定：

```bash
PYTHON_BIN=/path/to/python bash scripts/setup_world_model_training_env.sh
```

## 4. 预处理配对数据

`prepare_white_robot_iclora_data.sh` 会：

1. 调用 `build_white_robot_iclora_manifest.py` 扫描 `head_color.mp4`、`white_robot.mp4` 与 `instruction.txt`；
2. 生成 JSONL manifest；
3. 用 LTX VAE 编码目标视频和白膜参考视频，并编码文本；
4. 写出 `latents/`、`reference_latents/`、`conditions/`。

### 4.1 小批量 smoke / 过拟合数据

先用 8 个 episode 验证数据和训练链路：

```bash
cd "$REPO"
CUDA_VISIBLE_DEVICES=2 \
DEVICE=cuda:0 \
MAX_SAMPLES=8 \
RESOLUTION_BUCKETS=640x480x121 \
PREPROCESSED_ROOT="$REPO/data/white_robot_iclora_smoke_preprocessed_640x480x121" \
MANIFEST_PATH="$REPO/data/white_robot_iclora_smoke_8samples.jsonl" \
OVERWRITE=1 \
bash scripts/prepare_white_robot_iclora_data.sh
```

检查三个数据源的样本数是否一致：

```bash
for directory in latents reference_latents conditions; do
  printf '%s: ' "$directory"
  find "$REPO/data/white_robot_iclora_smoke_preprocessed_640x480x121/$directory" \
    -type f -name '*.pt' | wc -l
done
```

每个有效样本会在三个目录各有一个 `.pt` 文件；8 样本时总数应为 `24`。

### 4.2 全量 `640×480×121` 数据

全量预处理使用单张空闲 GPU，避免同时与训练争抢显存：

```bash
cd "$REPO"
CUDA_VISIBLE_DEVICES=2 \
DEVICE=cuda:0 \
RESOLUTION_BUCKETS=640x480x121 \
PREPROCESSED_ROOT="$REPO/data/white_robot_iclora_preprocessed_640x480x121" \
MANIFEST_PATH="$REPO/data/white_robot_iclora_train_640x480x121.jsonl" \
bash scripts/prepare_white_robot_iclora_data.sh
```

默认情况下脚本不会覆盖已有 latent。确需重新编码时才加 `OVERWRITE=1`。

### 4.3 多卡预处理

全量数据可用 `scripts/prepare_white_robot_iclora_data_multigpu.sh` 并行预处理。它使用
`accelerate launch`，每个进程处理同一 manifest 的交错分片，因此**不需要手动拆分 JSONL**。
所有 rank 会写入同一个 `PREPROCESSED_ROOT`；已存在的 `.pt` 会跳过，意外中断后可直接重跑。

脚本要求显式传入 `GPU_IDS`，并默认检查每张卡至少有 `100 GiB` 空闲显存，避免在共享 L20X
服务器上抢占其他作业。下面以四张非连续 GPU 为例：

```bash
cd "$REPO"
GPU_IDS=2,4,5,6 \
MIN_FREE_GB=100 \
RESOLUTION_BUCKETS=640x480x121 \
MANIFEST_PATH="$REPO/data/white_robot_iclora_train_640x480x121.jsonl" \
PREPROCESSED_ROOT="$REPO/data/white_robot_iclora_preprocessed_640x480x121" \
bash scripts/prepare_white_robot_iclora_data_multigpu.sh
```

小样本并行验证可附加 `MAX_SAMPLES=8`，并使用新的 manifest 和输出目录：

```bash
cd "$REPO"
GPU_IDS=2,4 \
MAX_SAMPLES=8 \
RESOLUTION_BUCKETS=640x480x121 \
MANIFEST_PATH="$REPO/data/white_robot_iclora_smoke_8samples_multigpu.jsonl" \
PREPROCESSED_ROOT="$REPO/data/white_robot_iclora_smoke_preprocessed_640x480x121_multigpu" \
OVERWRITE=1 \
bash scripts/prepare_white_robot_iclora_data_multigpu.sh
```

`NUM_PROCESSES` 默认等于 `GPU_IDS` 中 GPU 数；若手动设置，两者必须严格一致。
每个 rank 会自动绑定其对应的 `LOCAL_RANK`（即 `CUDA_VISIBLE_DEVICES` 中的相对序号），不要传
`DEVICE=cuda:0`。预处理仍会在每张选中卡加载一份文本编码器和 VAE，故 `BATCH_SIZE` 推荐保持为 `1`。
日志默认写入 `logs/prepare_multigpu_<timestamp>.log`。仅在确认选中 GPU 归自己所有时，才可用
`SKIP_GPU_CHECK=1` 跳过显存预检。

### 4.4 训练完成后的跨任务 checkpoint 评测

`scripts/watch_white_robot_checkpoint_evaluation.sh` 会每 5 分钟检查一次训练状态。只有最终
`step 3000` checkpoint 和训练状态文件都已保存、训练进程已退出、并且 GPU 5 至少有 `100 GiB`
空闲显存时，才会调用 `scripts/run_white_robot_checkpoint_evaluation.sh`。评测会以固定随机种子，
对 `blocks_ranking_size`、`open_laptop`、`stack_blocks_three` 三个 task 的固定 episode，比较每
`500` steps 的 checkpoint；结果与逐项日志分别写入：

```text
outputs/white_robot_to_rgb_iclora/checkpoint_eval/REPORT.md
logs/checkpoint_eval/
```

评测的最终输出为 `640×480×121`。两阶段 IC-LoRA 内部要求空间尺寸为 `64` 的倍数，
推理脚本会先以同纵横比的 `512×384×121` 生成，再无裁剪缩放为最终 `640×480` MP4。启动 watchdog：

```bash
cd "$REPO"
EVAL_GPU_ID=5 MIN_FREE_GB=100 POLL_SECONDS=300 \
  nohup bash scripts/watch_white_robot_checkpoint_evaluation.sh \
  > logs/checkpoint_eval/watchdog.stdout.log 2>&1 &
```

## 5. L20X 训练

### 5.1 为什么使用单卡 FSDP

当前 `LTX-2.5` transformer 很大。仓库默认训练脚本使用：

```text
packages/ltx-trainer/configs/accelerate/fsdp_size_wrap.yaml
```

该配置包含细粒度 `SIZE_BASED_WRAP`、activation checkpointing 和 BF16。它可在单张 140 GiB L20X 上跑通 `640×480×121`、batch size `1` 的 IC-LoRA。

已验证的单卡峰值显存约为 `88–92 GB`。请在启动前检查 GPU：

```bash
nvidia-smi
```

不要在共享环境中使用已有大量外部显存占用的卡。若 `nvidia-smi` 显示空闲，但训练仍在启动阶段 OOM，优先确认是否有其他容器/作业抢占显存；不要删除不属于自己的进程。

### 5.2 多卡 FSDP 训练

多卡训练使用 `scripts/train_white_robot_iclora_multigpu.sh`。脚本会：

1. 读取显式指定的物理 GPU 列表；
2. 令 `NUM_PROCESSES` 与 GPU 数量严格一致；
3. 复用 `fsdp_size_wrap.yaml` 的 `SIZE_BASED_WRAP`、BF16 和 activation checkpointing；
4. 默认检查每张选中卡至少有 `100 GiB` 空闲显存；
5. 将主进程输出写到单独的日志文件。

**不要**在共享服务器上省略 `GPU_IDS`。必须先通过 `nvidia-smi` 选定一组同时空闲的卡。下面使用四张卡的例子：

```bash
cd "$REPO"
GPU_IDS=0,1,2,3 \
MIN_FREE_GB=100 \
CONFIG_PATH="$REPO/configs/white_robot_to_rgb_iclora.yaml" \
PREPROCESSED_ROOT="$REPO/data/white_robot_iclora_preprocessed_640x480x121" \
bash scripts/train_white_robot_iclora_multigpu.sh
```

`PREPROCESSED_ROOT` 会覆盖原始 YAML 中的 `data.preprocessed_data_root`：启动器会在
`logs/runtime_configs/` 生成一份运行时配置，而不会改写 `CONFIG_PATH` 指向的原始文件。
因此，可以直接指定任意已完成预处理的数据目录，例如 `data/white_1000x121`。

默认白膜 IC-LoRA 配置将 `checkpoints.keep_last_n` 设为 `-1`，关闭自动清理并保留所有
按 `checkpoints.interval` 保存的 LoRA checkpoint，便于训练后进行跨 checkpoint 对比推理。
如果磁盘空间受限，再将该值改成正整数以仅保留最近若干个 checkpoint。

选择非连续卡也可以，例如：

```bash
GPU_IDS=2,4,5,6 \
MIN_FREE_GB=100 \
CONFIG_PATH="$REPO/configs/white_robot_to_rgb_iclora.yaml" \
PREPROCESSED_ROOT="$REPO/data/white_robot_iclora_preprocessed_640x480x121" \
bash scripts/train_white_robot_iclora_multigpu.sh
```

脚本会自动导出 `CUDA_VISIBLE_DEVICES`，并把进程 `local_rank=0..N-1` 映射到所给 GPU 列表。不要额外设置一个与 GPU 数量不一致的 `NUM_PROCESSES`。如已确认调度系统保留了对应 GPU、但 `nvidia-smi` 空闲显存低于阈值，可以明确跳过预检：

```bash
GPU_IDS=0,1,2,3 SKIP_GPU_CHECK=1 bash scripts/train_white_robot_iclora_multigpu.sh
```

仅在你确认这些 GPU 的占用归属时使用 `SKIP_GPU_CHECK=1`；它不会释放或终止其他任务。多卡能提高吞吐，但要求所有参与卡在整个作业期间都可用。此前共享卡被外部进程抢占会导致单个 rank OOM，并使所有 rank 停止。

### 5.3 8 样本、300 step 过拟合测试

该配置用于确认 loss 可下降、checkpoint 能保存、TensorBoard 能写入。它不是泛化训练。

```bash
cd "$REPO"
CUDA_VISIBLE_DEVICES=2 \
NUM_PROCESSES=1 \
ACCELERATE_CONFIG="$REPO/packages/ltx-trainer/configs/accelerate/fsdp_size_wrap.yaml" \
CONFIG_PATH="$REPO/configs/white_robot_to_rgb_iclora_overfit_8samples_640x480x121.yaml" \
PREPROCESSED_ROOT="$REPO/data/white_robot_iclora_smoke_preprocessed_640x480x121" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/train_white_robot_iclora.sh 2>&1 | tee logs/train_overfit_8samples_640x480x121.log
```

该配置的关键参数：

- 8 个样本，`batch_size: 1`
- 总 `300` step（约 37.5 次数据遍历）
- `rank: 8`、`alpha: 8`
- `adamw8bit`、BF16、gradient checkpointing
- 固定学习率 `2e-4`
- 每 50 step 保存 checkpoint
- `no_resume: true`，每次从 step 0 开始

成功时应在：

```text
outputs/white_robot_to_rgb_iclora_overfit_8samples_640x480x121/checkpoints/
```

看到 `lora_weights_step_00300.safetensors`。

### 5.4 从 step 300 续训到总 1000 step

不要把 `steps` 理解成“额外训练步数”；它是**总目标 step**。下列配置会加载 step 300 checkpoint，并恢复 scheduler/RNG，继续到 step 1000：

```bash
cd "$REPO"
CUDA_VISIBLE_DEVICES=2 \
NUM_PROCESSES=1 \
ACCELERATE_CONFIG="$REPO/packages/ltx-trainer/configs/accelerate/fsdp_size_wrap.yaml" \
CONFIG_PATH="$REPO/configs/white_robot_to_rgb_iclora_overfit_8samples_640x480x121_step1000.yaml" \
PREPROCESSED_ROOT="$REPO/data/white_robot_iclora_smoke_preprocessed_640x480x121" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/train_white_robot_iclora.sh 2>&1 | tee logs/train_overfit_8samples_640x480x121_step1000.log
```

启动日志应出现：

```text
Loading checkpoint from ...lora_weights_step_00300.safetensors
LoRA checkpoint loaded successfully
Loaded training state from ...training_state_step_00300.pt
Resuming from step 300
```

若只希望加载 LoRA 权重而不恢复 step/scheduler，请把配置中的 `checkpoints.no_resume` 设为 `true`。这样训练会从 step 0 计数，通常不适合严格的续训实验。

### 5.5 全量训练

在全量 `640×480×121` 预处理数据完成后，使用：

```bash
cd "$REPO"
CUDA_VISIBLE_DEVICES=2 \
NUM_PROCESSES=1 \
ACCELERATE_CONFIG="$REPO/packages/ltx-trainer/configs/accelerate/fsdp_size_wrap.yaml" \
CONFIG_PATH="$REPO/configs/white_robot_to_rgb_iclora.yaml" \
PREPROCESSED_ROOT="$REPO/data/white_robot_iclora_preprocessed_640x480x121" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/train_white_robot_iclora.sh 2>&1 | tee logs/train_white_robot_640x480x121.log
```

`configs/white_robot_to_rgb_iclora.yaml` 的默认设置是 3000 step。全量训练前先确认其 `data.preprocessed_data_root` 已存在且三类 latent 文件匹配。

## 6. 监控与 checkpoint

训练器会把以下指标写入 TensorBoard：

- `train/loss`
- `train/learning_rate`
- `train/step_time`
- `train/loss_sigma_*`（不同 flow-matching 噪声区间）
- `system/gpu_memory_allocated_gb`
- `system/gpu_memory_reserved_gb`
- `system/gpu_memory_peak_allocated_gb`
- 训练结束后的 `stats/*`

服务器已有 TensorBoard 服务时，在本机建立端口转发：

```bash
ssh -L 6006:127.0.0.1:6006 ltx
```

然后打开：

```text
http://localhost:6006
```

当前配置将 event 文件写在 `/lpai/tensorboard/` 下。相关目录：

```text
/lpai/tensorboard/white_robot_to_rgb_iclora_overfit_8samples_640x480x121
/lpai/tensorboard/white_robot_to_rgb_iclora_overfit_8samples_640x480x121_step1000
/lpai/tensorboard/white_robot_to_rgb_iclora
```

禁用 Rich progress bar 后，训练日志每 20 step 打印一次 loss、学习率和时间。实时查看：

```bash
tail -f "$REPO/logs/train_overfit_8samples_640x480x121_step1000.log"
watch -n 2 nvidia-smi
```

flow-matching 的单 step loss 会因随机噪声时间步而出现尖峰；判断训练是否稳定时，应观察滑动均值、分段均值或 TensorBoard 总体趋势，而不是要求每一步都单调下降。

## 7. 训练后 IC-LoRA 推理

使用 `scripts/infer_white_robot_iclora.py` 进行原生两阶段 IC-LoRA 推理。`--episode` 会自动读取：

- `head_color.mp4` 的首帧；
- `white_robot.mp4` 作为参考控制视频；
- `instruction.txt` 作为 prompt。

示例（先生成兼容原生两阶段 pipeline 的 `640×512` 视频）：

```bash
cd "$REPO"
CUDA_VISIBLE_DEVICES=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/infer_white_robot_iclora.py \
  --episode adjust_bottle/aloha-agilex_clean_50/episode0 \
  --ic-lora-path "$REPO/outputs/white_robot_to_rgb_iclora_overfit_8samples_640x480x121_step1000/checkpoints/lora_weights_step_01000.safetensors" \
  --height 512 --width 640 --num-frames 121 --frame-rate 24 --seed 42 \
  --offload cpu --quantization fp8-cast \
  --output-path "$REPO/outputs/world_model/episode0_step1000_raw_640x512.mp4"
```

将该视频中心裁剪为要求的 `640×480`：

```bash
cd "$REPO"
.venv/bin/python - <<'PY'
from pathlib import Path
import av

source = Path("outputs/world_model/episode0_step1000_raw_640x512.mp4")
target = Path("outputs/world_model/episode0_step1000_640x480.mp4")
with av.open(source) as input_container:
    stream = input_container.streams.video[0]
    with av.open(target, "w") as output_container:
        output_stream = output_container.add_stream("libx264", rate=stream.average_rate or 24)
        output_stream.width, output_stream.height, output_stream.pix_fmt = 640, 480, "yuv420p"
        for frame in input_container.decode(stream):
            cropped = frame.to_image().crop((0, 16, 640, 496))
            for packet in output_stream.encode(av.VideoFrame.from_image(cropped)):
                output_container.mux(packet)
        for packet in output_stream.encode():
            output_container.mux(packet)
print(target)
PY
```

对于过拟合实验，应先用训练集 episode 验证管线，再选择未训练的 episode 或任务衡量泛化。

## 8. 常见问题

### OOM：`Tried to allocate 300 MiB`

L20X 单卡确实可以跑该设置，但共享服务器的外部任务可能占掉大量显存。先检查：

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

优先换到真正空闲的 GPU，不要随意终止其他用户的进程。保持：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 训练开始后没有 loss 曲线

确认配置包含：

```yaml
tensorboard:
  enabled: true
  log_dir: "/lpai/tensorboard/<run_name>"
```

并确认训练日志中有：

```text
TensorBoard logging enabled: ...
```

旧训练不会回填历史 TensorBoard event；需要重新训练才会记录逐 step 指标。

### 训练配置显示数据目录不存在

`data.preprocessed_data_root` 必须与预处理命令的 `PREPROCESSED_ROOT` 一致。若训练配置指向 `...preprocessed_640x480x121`，先运行对应的全量预处理命令。

### 原始基座模型与 IC-LoRA 结果不能直接等价比较

原始 LTX-2.5 基座可做 RGB 首帧 + 文本的 I2V，但无法接收本项目的白膜视频控制。IC-LoRA 结果有额外的动作轨迹条件，因此对比时应明确基座结果是无参考条件的 I2V baseline。
