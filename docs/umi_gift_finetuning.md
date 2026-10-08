# 仅微调 gift 数据集

训练配置：`configs/vla/umi/gift.yaml`；统计配置：`configs/vla/umi/gift_norm.yaml`。
数据列表 `assets/training_data/umi_gift.txt` 仅包含 `/mnt/dataset/gift_overall_20260812`，
使用 `configs/robot_configs/umi_gift.yaml`，归一化文件为 `assets/norm_stats/umi_gift.json`。
沿用 `umi_split.json` 中的 627 条训练轨迹和 70 条验证轨迹。
`gift.yaml` 每 100 个优化器更新步，在验证集固定的 1,024 个采样起点上计算动作 loss。

每条轨迹首尾各裁剪 30 帧，动作已有下一帧标签，因此有效采样起点数为 N-61。
训练集共 799,722 个采样起点。统计只使用训练集，并排除填充动作。

英文指令读取 `meta/episodes.jsonl` 的 `task_annotation`，传入模型的 prompt。
全部 697 条轨迹中，688 条为 `Pack the gift and cover the box`，9 条为
`Pack the gift and cover the box lid`。不修改原始标注。

当前 `observation.state` 是模型输入。动作监督来自 Parquet 的 `action` 列，取双手的
前 16 维（两组位置、四元数、夹爪），忽略最后 7 维 ego。映射后，动作位姿转换为
当前状态坐标系下的相对位姿，夹爪保留绝对宽度，组成 50 步动作块；有效维度填入
55 维统一空间，其余维度和时间填充位置均屏蔽动作损失。深度和视频 teacher 另提供辅助监督。

默认采用 `meanstd`：`(x - mean) / (std + 1e-6)`。状态和动作分别统计；动作统计在
四元数规范化和相对位姿转换之后计算。推理必须使用相同映射、统计文件及归一化方法。

仅计算统计，不训练，也可以单进程使用 CPU（4 个数据工作进程）：

```bash
LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 OMP_NUM_THREADS=1 \
python scripts/compute_norm_stats.py configs/vla/umi/gift_norm.yaml \
  --train.global_batch_size 32
```

统计 JSON 包含 mean、std、min、max 和近似 q01/q99、q02/q98，供不同归一化方式使用。

### GPU 统计（仅计算 JSON）

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
bash train.sh scripts/compute_norm_stats.py configs/vla/umi/gift_norm.yaml \
  --data.norm_device cuda --data.norm_batch_size 2048
```

只用一张卡时改为 `CUDA_VISIBLE_DEVICES=0`，并加 `--train.global_batch_size 32`。
`norm_batch_size` 是统计时每批采样起点数，与模型训练 batch 无关；显存不足可减到 512。
CUDA 路径按 episode 分配给各卡，GPU 批量执行相对位姿转换、矩统计和直方图统计；
CPU 读取 Parquet，不加载视频或模型权重，不启动训练。该路径不使用 `data.num_workers`。
每次只将一条轨迹和一个动作块批次放到 GPU。

GPU 路径支持 `local_v21`、显式 action 列、完整训练划分以及合并动作块维度的统计。
第一遍使用 FP64 稳定矩计算 mean/std 和全局 min/max；第二遍使用 5000 个固定全局
直方图区间计算近似分位数。CPU 原实现会动态调整直方图，因此分位数可能有小幅差异；
mean/std/min/max 在浮点误差范围内一致。训练和推理继续读取相同 JSON 格式。

在项目根目录执行：

```bash
source .venv/bin/activate

# 先生成统计；已有成功生成的 gift 统计时可以跳过。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash train.sh scripts/compute_norm_stats.py configs/vla/umi/gift_norm.yaml \
  --data.norm_device cuda

wandb login

# 8 卡，每卡 batch 4，累积 1，全局 batch 32。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/umi/gift.yaml \
  --train.output_dir output/umi_gift_smoke \
  --train.enable_resume false \
  --train.max_steps 20 --train.save_steps 20

# 检查 loss、显存和 checkpoint 保存后正式训练。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/umi/gift.yaml
```

正式输出目录为 `output/umi_gift`。没有已有 checkpoint 时从 base 初始化；重复启动会自动续训。
z-score 配置训练 30,000 步，每 5,000 步保存，每 10 步记录当前训练指标。
每卡 batch=4，累积=1，八卡全局 batch=32。

周期性验证的 `validation/vla_loss` 是按有效动作元素加权的 Flow Matching loss，
不是实机成功率，也不是多步去噪后的动作误差。每次使用相同验证子集和相同噪声/时间
随机数种子，不做图像增强，不计算 teacher 辅助 loss，不更新参数，并恢复训练随机数状态
和模型模式。子集索引保存在输出目录的 `eval_subset.json`。八卡时每次每卡验证 32 个 batch。
现有 z-score 统计直接用于验证，不重新计算验证集统计。

如需试跑时覆盖首次验证，将上面的 smoke 命令改为 `--train.max_steps 100 --train.save_steps 100`。
周期性验证只支持当前带 `split_file` 的 local_v21 数据和 LingbotVLAV2Config，纯数据并行。
`eval_samples` 必须能被 `卡数 × 每卡 batch` 整除；`--train.eval_steps 0` 可禁用周期性验证。

## 对比 z-score 与 min-max

`meanstd` 就是 z-score。原配置 `gift.yaml` 和 `assets/norm_stats/umi_gift.json` 保留不变。
新增 `gift_minmax.yaml` 使用 `minmax`（映射到 [-1, 1] 并截断），通过独立的
`umi_gift_minmax` 机器人配置读取 `assets/norm_stats/umi_gift_minmax.json`，训练输出到
`output/umi_gift_minmax`，W&B 名称为 `umi-gift-minmax`。

统计 JSON 同时保存 mean/std/min/max/分位数，不是已经归一化的数据。
因此 `umi_gift_minmax.json` 初始是现有 gift 统计的独立副本，不需要重新计算；
实际归一化方式由训练配置中的 `data.norm_type` 决定。
如需单独重新计算 min-max 实验的统计，只会写入新文件：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
bash train.sh scripts/compute_norm_stats.py configs/vla/umi/gift_minmax_norm.yaml \
  --data.norm_device cuda --data.norm_batch_size 2048
```

之后对比训练时分别使用 `gift.yaml` 和 `gift_minmax.yaml`。保持 seed、batch、学习率、
训练步数和数据划分一致，使用反归一化后的误差或任务成功率比较，不能直接比较不同尺度的训练 loss。

## 分位数归一化实验

`configs/vla/umi/gift_quantile.yaml` 使用 `bounds_99_woclip`，公式为
`2 * (x - q01) / (q99 - q01 + 1e-6) - 1`。q01/q99 是 GPU 直方图估计的
1%/99% 分位数；超出该区间的值允许超出 [-1, 1]，不做截断。
这是按分位数确定缩放范围，不是将整个分布转换成均匀或正态分布的 QuantileTransformer。

通过独立机器人配置 `umi_gift_quantile.yaml` 读取现有
`assets/norm_stats/umi_gift_minmax.json`，无需重算或修改 JSON。
训练 30,000 步，每 5,000 步保存、每 10 步记录训练指标、每 100 步验证固定 1,024 个样本；
八卡每卡 batch=4，累积=1。输出目录为 `output/umi_gift_quantile`，首次从 base 初始化。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/umi/gift_quantile.yaml
```

验证和推理需使用同一分位数归一化配置及同一统计文件。
