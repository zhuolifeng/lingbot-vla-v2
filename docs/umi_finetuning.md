# UMI 双腕微调

所有命令在项目根目录、激活 `.venv` 后运行。数据上游已完成坐标处理，本适配不添加外参转换。

## 下载权重

```bash
source .venv/bin/activate
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE

python scripts/download_hf_model.py \
  --repo_id robbyant/lingbot-vla-v2-6b --local_dir checkpoints \
  --endpoint https://huggingface.co
python scripts/download_hf_model.py \
  --repo_id Qwen/Qwen3-VL-4B-Instruct --local_dir checkpoints \
  --endpoint https://huggingface.co
python scripts/download_hf_model.py \
  --repo_id Ruicheng/moge-2-vitb-normal --local_dir checkpoints \
  --endpoint https://huggingface.co
```

`--endpoint` 显式覆盖终端里的 `HF_ENDPOINT`，保留现有 HTTP/HTTPS/ALL_PROXY 网络代理。
遇到 `FileMetadataError: Distant resource does not seem to be on huggingface.co` 时，当前 Hub 库的具体触发条件是响应缺少 `x-repo-commit`，并不等于已经确认 SSL 证书有问题。可先在第一条命令后加 `--include config.json`，仅下载小配置文件验证，再去掉该选项下载完整模型。

下载脚本会自动在 `--local_dir` 下加模型名称。第一条同时下载 base、`depth/model.pt`、`dino_video/teacher_step_10000.pth` 和 `dino_video/config.yaml`。
官方权重位于 `checkpoints/lingbot-vla-v2-6b/`，没有 `hf_ckpt/` 子目录。MoGe 的文件名是 `model.pt`。
第二条下载完整 Qwen 仓库，操作简单；本项目使用其中的配置、tokenizer 和 processor。

官方发布的 `config.json` 当前只记录 `vlm_family: qwen3_vl`，没有动作块长度。准备脚本会明确提示并读取本仓库配置类的默认值（当前 50），而非使用 StarVLA 的 60。
若 checkpoint 明确记录了 `chunk_size`，则以该值为准，并检查 `n_action_steps` 一致性。

来源：[LingBot 文件](https://huggingface.co/robbyant/lingbot-vla-v2-6b/tree/main)、[官方 config](https://huggingface.co/robbyant/lingbot-vla-v2-6b/blob/main/config.json)、[MoGe 文件](https://huggingface.co/Ruicheng/moge-2-vitb-normal/tree/main)。

## 准备配置

```bash
python scripts/prepare_umi.py \
  --checkpoint checkpoints/lingbot-vla-v2-6b \
  --tokenizer checkpoints/Qwen3-VL-4B-Instruct \
  --depth-root checkpoints/lingbot-vla-v2-6b/depth \
  --moge-checkpoint checkpoints/moge-2-vitb-normal/model.pt \
  --video-root checkpoints/lingbot-vla-v2-6b/dino_video
```

输出：`assets/training_data/umi.txt`、`umi_split.json`、`configs/vla/umi/umi.yaml`、`norm.yaml`。
默认每个数据集约 90% episode 训练、10% 验证，seed=42；不按帧拆分。重新运行同一命令会重写这些生成文件，结果保持确定性。
不提供 `--checkpoint` 时，只生成数据划分，不生成训练配置。

## 数据和边界约定

- 使用项目内 `local_v21_dataset.py`，绕开安装的 LeRobot v3 元数据加载器；无需降级 `.venv`，无需转换或复制视频。
- `configs/robot_configs/umi.yaml` 显式选择此读取器，其他机器人保持原加载路径。
- 原始每条轨迹 N 帧，仅保留 `[30, N-30)`；不修改原始 Parquet、视频或时间戳。
- 已有 `action[t] = state[t+1]`，不再次移动标签。最后一个保留状态只能作为动作目标；可采样起点数为 `N-61`。
- 动作块只读取裁剪范围内的标签；越界位置重复最后一个有效标签并屏蔽损失，归一化统计也排除这些填充标签。未来图像在保留的视频帧范围内截断。
- 英文指令：`meta/episodes.jsonl` 的 `task_annotation`，按 `episode_index` 查找。抽查的 `action_config[].action_text` 也有英文指令，但此适配使用 episode 的 `task_annotation`。`meta/tasks.jsonl` 是中文采集名称。
- 仅解码 `wrist_image_1`（左）和 `wrist_image_2`（右）；忽略头部视频及最后 7 维 ego。

## 位姿、归一化与 q / -q

原始每只手为 `[xyz, qw,qx,qy,qz, gripper]`。映射到统一动作空间时使用两个 `xyz+xyzw` 位姿（`end.position`，14 维）和两个夹爪（`effector.position`，2 维）。55 维内部结构保持不变。

当前状态保持绝对位姿。动作位姿按每只手分别计算 `inverse(T_state[t]) @ T_action[t+k]`，整个动作块共用同一个参考状态。夹爪保持绝对宽度。

UMI 沿用 real_robot 模板的 `meanstd`：`(x - mean) / (std + 1e-6)`。状态和动作分别计算统计量，三个数据集共享同一套训练统计。统计应在裁剪、划分、四元数规范化和相对动作计算之后生成。RoboTwin 模板使用的是 `bounds_99_woclip`（1%/99% 分位数缩放），并非所有配置都统一采用 minmax。

UMI 输入状态和动作四元数在数值归一化之前先单位化、统一符号：w>0；精确 w=0 时，以 x/y/z 的首个非零分量为正打破平局。因此数据中 q 和 -q 得到相同表示。这不代表数值 Flow Matching 损失本身具有旋转等价性，也不消除 180 度附近的表示不连续。

推理先反归一化，再用生成动作块时保存的状态恢复 `T_target = T_state[t] @ T_relative`，最后恢复原始 wxyz 排列。输入状态也必须使用相同的 UMI FeatureTransform。

## 统计和训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash train.sh scripts/compute_norm_stats.py configs/vla/umi/norm.yaml

wandb login

# 先运行少量步骤检查显存、loss、保存及 teacher 路径。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash train.sh tasks/vla/train_lingbotvla.py configs/vla/umi/umi.yaml \
  --train.output_dir output/umi_smoke --train.max_steps 20 --train.save_steps 20

# 正式运行，使用独立输出目录。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash train.sh tasks/vla/train_lingbotvla.py configs/vla/umi/umi.yaml
```

生成配置从 real_robot 模板派生：每卡 batch=1、梯度累积=4、8 卡全局 batch=32，启用梯度检查点和 W&B，默认 60,000 步。步数是起始设置，不代表已完成充分训练。显存允许后再调整 batch，并根据验证结果选择 checkpoint。
训练脚本会同步 TensorBoard 指标到 W&B。teacher 保持冻结的现有流程；当前深度/视频 teacher 使用相机列表第一路，即左腕。
验证数据可用 `--data.episode_split val` 读取，但现有训练循环不会因此自动增加离线验证；不能把验证集传给训练命令继续优化。

## 验证适配

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_umi_data.py -v
```

测试覆盖首尾裁剪、下一帧标签边界、两相机时间戳、英文 prompt、episode 划分、四元数正负号、局部平移旋转、绝对/相对位姿往返，以及填充动作损失屏蔽。
