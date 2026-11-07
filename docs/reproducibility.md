# 复现指南

## CPU：立即可运行

```bash
python -m venv .venv-cpu
source .venv-cpu/bin/activate
pip install -r requirements-cpu.txt
python examples/demo.py
make check
pip install -r requirements-viz.txt
make figures
```

## GPU：准备条件

修复训练使用 Torch2.8.0 / Transformers4.56.1 / vLLM0.11.0 的本地 STAR 环境。`requirements-repair.txt` 是修复阶段的版本参考，CUDA/FlashAttention 需要平台匹配。原 CKD/Sim-RL 还需要 STAR 上游环境与带 CKD 补丁的 OpenRLHF。

```bash
bash scripts/bootstrap_upstream.sh
# 在单独 GPU 环境中安装；不与其他项目的训练环境混装。
python -m venv .venv
source .venv/bin/activate
pip install -r upstream/requirements_uv.txt
pip install -e openrlhf --no-deps
```

上游 source 固定到已记录提交并应用官方 CKD patch。修复评分器依赖 `upstream/simrl.py`；CPU gate/demo 不需要克隆上游。

## 输入约定

训练任务 JSONL 每行含 `id`、`inputs`、`targets`，可选 `split: train`。`inputs` 为带 `<tools>` 定义和原 user 消息的 STAR rendered prompt，末尾处于 `<think>` 内；`targets` 是以 `</think>` 开头的 continuation。父预测 JSONL 包含相同 `id` 与 `output`。

`build_training_pairs.py` 只接受训练数据，生成同 ID/gold 的 identity/repair JSONL 与 SHA256 manifest。构建脚本可用于新实验，不保证自动重建历史 590 行；先对照历史 `configs/training-manifest.json` 检查选择结果。

```bash
python scripts/train_real_draft_repair.py --base models/simrl-parent \
  --data data/pairs --output results/runs/preflight42 --seed 42 --preflight
```

真正训练仍使用固定 64 updates、batch16、micro4、rank16。输出必须是新目录。三个 seed 应分别运行，均保留 identity 与 repair。新实验的训练记录、输入 hash 与环境必须另存。

## 基线训练

`prepare_public_mixed.py` 需要自行取得官方 Hammer 并按 `split_hammer.py` 准备负例，再加入上游 1024 条正向示例。设置 `STUDENT_MODEL` / `TEACHER_MODEL` 后，按 `prepare_ckd_data.sh` → `train_ckd.sh` → `train_sim_rl.sh` 执行。训练脚本需要一个由操作者管理的 Ray head。两张卡、依赖和完整教师轨迹是前置条件。

历史 Sim-RL 父模型和历史 split/cohort 不包含在此仓库。使用 Raw Qwen3-0.6B 做修复父模型会改变实验，不能报告为历史 37.20% 的复跑。

## Evidence and release boundaries

The reported GPU experiments were run in October 2026. This public export adds
portable paths, CPU demos, documentation, charts and validation workflows. Those
packaging changes are not a new GPU replication.

`results/source-manifest.json` records SHA256 hashes of the selected original
workspace files and their initial sanitized exports. `publication-manifest.json`
records the final public source, evidence and figure hashes. These hashes support
artifact traceability; they do not independently verify that a GPU run happened.

Only code, configuration and aggregate evidence are included. Model weights,
raw datasets, full prompts and raw model transcripts are obtained separately.
Exact historical replay requires the original frozen inputs and checkpoint
artifacts; a new dataset or checkpoint creates a new experiment. Existing test
cohorts have been observed and must not be used as new blind tests.

CPU checks validate scoring, guards and released aggregates. They do not certify
GPU dependency compatibility or reproduce training accuracy. Follow the recorded
environment and use a new run directory for GPU experiments. Final-only saves
avoid redundant base and optimizer copies.
