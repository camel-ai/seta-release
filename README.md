# SETA: Scaling Environments for Terminal Agents

This repository is the official implementation of **SETA: Scaling Environments for Terminal Agents**.


## Requirements

Run the setup script:

```bash
bash setup.sh
```

This creates a `terminal_agent` conda environment and installs all dependencies, including:
- [CAMEL](external/camel) — agent framework
- [Harbor](external/harbor) — Docker-based task execution
- [AReaL](external/areal) — distributed RL/SFT training framework
- FlashAttention, Transformers, Datasets
- Docker (installed automatically if not present)

> 📋 The setup script also configures Docker's network address pool for concurrent container execution. Run it once on each machine.

## Data Synthesis

SETA provides two pipelines under `datasynth/` for generating training tasks.

### SETA-Synth Pipeline

Synthesizes terminal-agent tasks from seed data (Stack Overflow, Unix StackExchange, Kaggle notebooks). Located at `datasynth/seed2synth_pipeline/`.

```bash
cd datasynth/seed2synth_pipeline

# Set required environment variables
export HF_TOKEN=<your-hf-token>
export SGLANG_URL=http://localhost:8000   # only needed for rollout

# Download seed metadata (~5 sec)
python hf_utils.py --config configs/kaggle_base.yaml download-metadata

# Synthesis only
python run_orchestrator.py configs/kaggle_base.yaml --synth-only

# Dry-run to preview the queue
python run_orchestrator.py configs/kaggle_base.yaml --dry-run

# Full pipeline (synthesis + rollout validation)
python run_orchestrator.py configs/kaggle_base.yaml
```

To create custom splits:

```bash
python hf_utils.py --config configs/kaggle_base.yaml check-and-split \
  --sources kaggle_notebook --n-parts 8 --output-dir configs/filters/
```

Other useful commands:

```bash
# Override worker count
python run_orchestrator.py configs/kaggle_base.yaml --synth-only --n-synth-workers 4

# Regenerate summary.csv from completed tasks
python run_orchestrator.py configs/kaggle_base.yaml --generate-summary

# Monitor progress
cat outputs/synth_data/kaggle_notebook/summary.csv | head -20
grep "pass" outputs/synth_data/kaggle_notebook/summary.csv | wc -l
```

### SETA-Evol Pipeline

Evolves existing tasks into harder variants or ports them to new technology domains. Located at `datasynth/evol_pipeline/`.

```bash
cd datasynth/evol_pipeline

# Copy and edit a config
cp configs/config.example.yaml configs/my_run.yaml

# Preview the queue (dry-run)
python run_evol_orchestrator.py configs/my_run.yaml --dry-run

# Run
python run_evol_orchestrator.py configs/my_run.yaml
```

Two evolution strategies are available (one per config run):
- **INCREASE_DIFFICULTY** — harder version with more steps, edge cases, or tighter constraints
- **DECREASE_DIFFICULTY** — harder version with more steps, edge cases, or tighter constraints
- **CHANGE_CONTEXT** — same complexity ported to a different technology (e.g. nginx → apache2)

Chain multiple evolution turns by pointing the next run's `input_dir` at the previous run's `output_dir`:

```bash
# Turn 1: increase difficulty
python run_evol_orchestrator.py configs/increase_difficulty.yaml

# Turn 2: port to new domains
python run_evol_orchestrator.py configs/change_context_chained.yaml
```

For distributed multi-machine runs:

```bash
python run_evol_orchestrator.py configs/my_run.yaml --generate-filters --n-parts 4
# Distribute each part's config to a separate machine
```

Other CLI flags:

```
--dry-run              Preview task queue
--n-evol-workers N     Override parallelism
--generate-summary     Rebuild summary.csv from completed tasks
--generate-filters     Split tasks for multi-machine runs
--upload               Push results to HuggingFace
```

## Evaluation

Evaluation uses Docker to run tasks in isolated containers. Run from the repo root.

```bash
# 1. Start a model server
python -m sglang.launch_server --model Qwen/Qwen3-8B --port 30000

# 2. Run evaluation (dataset auto-downloads on first use)
python scripts/evaluation/eval.py --config scripts/evaluation/configs/eval_default_qwen3_8b.yaml
```

Override config fields on the command line (Hydra-style):

```bash
python scripts/evaluation/eval.py \
    --config scripts/evaluation/configs/eval_default_qwen3_8b.yaml \
    terminal_env.model.model_type=Qwen/Qwen3-32B \
    terminal_env.model.url=http://localhost:30000/v1 \
    workers=8 \
    dataset=seta-env
```

The `seta-env` dataset auto-downloads from HuggingFace on first use (registry in `seta_env/dataset/datasets.yaml`).

### Collecting and Filtering Results

After evaluation, collect results and filter out uninformative tasks before RL training:

```bash
# Collect results into evaluated_tasks.csv
python -m seta_env.utils.collect_results /path/to/eval_run/trials \
    --output /path/to/eval_run

# Merge results from multiple resumed runs
python -m seta_env.utils.collect_results --merge \
    /path/to/eval_run /path/to/eval_run_resume \
    --output /path/to/merged

# Filter dataset — drop tasks that are too easy, too hard, or missing
python -m seta_env.dataset.filter_tasks \
    --csv /path/to/merged/evaluated_tasks.csv \
    --dataset dataset/Anonymous_Submission_Release \
    --drop-missing --drop-too-hard --drop-too-easy
```

This writes `dataset/Anonymous_Submission_Release/task_filter.txt`, which the dataset loader automatically honors on subsequent runs.

## Training

### RL Training

SETA uses the [AReaL](external/areal) framework for distributed RL training. All scripts are under `scripts/areal/`. Run from the repo root.

```bash
python -m areal.launcher.local \
    scripts/areal/rl_train.py \
    --config scripts/areal/configs/config_train_local_seta_env.yaml
```

Override fields on the command line:

```bash
python -m areal.launcher.local \
    scripts/areal/rl_train.py \
    --config scripts/areal/configs/config_train_local_seta_env.yaml \
    actor.path=Qwen/Qwen3-8B \
    cluster.n_gpus_per_node=8 \
    allocation_mode=sglang:d8p1t1+fsdp:c2t2
```

For multi-node training on Ray, swap `areal.launcher.local` for `areal.launcher.ray` and add `cluster.n_nodes=N`.

The dataset is loaded from `dataset/Anonymous_Submission_Release/SETA_Synth` (set via `train_dataset.path` in the config). Run the AReaL evaluation-only pass with:

```bash
python -m areal.launcher.local \
    scripts/areal/eval.py \
    --config scripts/areal/configs/config_eval_local_seta_env.yaml
```

### SFT Data Collection

Collect SFT trajectories from a strong teacher model, then build a training JSONL:

```bash
# 1. Run evaluation with a teacher model to collect trajectories
python scripts/evaluation/eval.py \
    --config scripts/evaluation/configs/eval_default.yaml \
    terminal_env.model.model_type=<teacher-model>

# 2. Merge resumed runs (if any)
python -m seta_env.utils.collect_results --merge \
    outputs/eval/<trial_name> \
    outputs/eval/<trial_name>_resume \
    --output outputs/eval/<trial_name>_merged \
    --collect-trials move

# 3. Build SFT JSONL (tokenized under Qwen3-8B chat template)
#    Add --debug to write per-trial boundary inspection artifacts
python -m seta_env.utils.sft_utils.build_sft_dataset \
    --trials-dir outputs/eval/<trial_name>_merged/trials \
    --output     outputs/eval/<trial_name>_merged/sft.jsonl \
    --model      Qwen/Qwen3-8B

# 3b. No-thinking variant (prune <think> blocks, append /no_think to system prompt)
python -m seta_env.utils.sft_utils.build_sft_dataset \
    --trials-dir outputs/eval/<trial_name>_merged/trials \
    --output     outputs/eval/<trial_name>_merged/sft_no_thinking.jsonl \
    --no-thinking

# 4. (Optional) Push to HuggingFace Hub
export HF_TOKEN=<token>
python -m seta_env.utils.sft_utils.build_sft_dataset \
    --trials-dir outputs/eval/<trial_name>_merged/trials \
    --output     outputs/eval/<trial_name>_merged/sft.jsonl \
    --model      Qwen/Qwen3-8B \
    --push-to-hub <your-org/dataset-name>
```

By default, trials with no verifier output (timeouts / errors) are dropped. Pass `--min-reward 1.0` to keep only fully-passing rollouts.

### SFT Training

Train on an SFT dataset from HuggingFace (or a local `.jsonl`). Run from the repo root.

```bash
# Thinking variant (preserves <think>...</think> reasoning blocks)
python -m areal.launcher.local \
    scripts/areal_sft/sft_train.py \
    --config scripts/areal_sft/configs/seta_kimi_qwen3_sft_thinking.yaml

# No-thinking variant
python -m areal.launcher.local \
    scripts/areal_sft/sft_train.py \
    --config scripts/areal_sft/configs/seta_kimi_qwen3_sft_nothink.yaml
```

Default config runs on 2 GPUs (`cluster.n_gpus_per_node=2`, `allocation_mode=d2p1t1`). Override on the command line:

```bash
python -m areal.launcher.local \
    scripts/areal_sft/sft_train.py \
    --config scripts/areal_sft/configs/seta_kimi_qwen3_sft_thinking.yaml \
    train_dataset.batch_size=16 \
    total_train_epochs=1 \
    cluster.n_gpus_per_node=8 \
    allocation_mode=d8p1t1 \
    cluster.fileroot=outputs/areal/my_run
```

Checkpoints and logs land under `${cluster.fileroot}/<experiment>/<trial>/`. Resume an interrupted run with `recover.mode=auto`.

To use a local JSONL instead of the default dataset path, set `train_dataset.path=outputs/eval/<trial>_merged/sft_thinking.jsonl`.

## Pre-trained Models

You can download pretrained models here:

- [seta-env-rl](https://huggingface.co/AnonymousSubmissionUnderDouble-BlindRevi/seta-env-rl) — Qwen3-8B fine-tuned with RL on the SETA training set
- [seta-sft-kimi-qwen3](https://huggingface.co/AnonymousSubmissionUnderDouble-BlindRevi/seta-sft-kimi-qwen3) — Qwen3-8B fine-tuned with SFT on Kimi-K2.5 trajectories
