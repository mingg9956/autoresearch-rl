# autoresearch-rl

Autonomous RL post-training research. Give an AI agent a real RL training setup and let it experiment autonomously overnight. It modifies the config, trains for 10 minutes, checks if the result improved, keeps or discards, and repeats.

RL post-training is hard. Training is unstable, reward hacking is real, and small hyperparameter changes can completely tank a run. It's the kind of problem where you end up babysitting experiments for hours, tweaking one knob at a time. When I saw Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) for pretraining, the first thing I thought was: this is exactly what RL post-training needs.

The idea is simple: instead of a human sitting around tweaking hyperparameters, let an AI agent do it. The agent reads the current config, proposes a change (learning rate, batch size, loss masking, regularization, etc.), runs a full training loop, evaluates, and decides whether to keep or discard the result. Over dozens of experiments it systematically explores the search space and converges on a strong configuration — all without human intervention.

This applies the autoresearch philosophy to **RL post-training** using [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) — my favourite RL post-training framework — and [verifiers](https://github.com/PrimeIntellect-ai/verifiers).

Shoutout to [@willccbb](https://x.com/willccbb) for creating verifiers.

## Progress

![Autoresearch-RL Progress](progress.png)

## How it works

The repo has four files that matter:

- **`prepare.py`** — fixed constants, one-time setup (downloads base model, verifies GPUs). Not modified.
- **`train.toml`** — the single file the agent edits. Contains the full RL training configuration: optimizer, learning rate, loss function, environments, rollout settings, etc. **This file is edited and iterated on by the agent**.
- **`run.py`** — experiment runner. Launches prime-rl, enforces the time budget, extracts metrics. Not modified.
- **`program.md`** — instructions for the agent. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 10-minute time budget**. The metric is **eval_score** (average pass@1 across environments) — higher is better.

## Quick start

**Requirements:** 2 NVIDIA GPUs, Python 3.12+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. One-time setup (download model, verify GPUs)
uv run prepare.py

# 4. Run a single experiment (~12 min)
uv run run.py
```

## Running the agent

Spin up Claude/Codex in this repo, then prompt:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

## Design choices

- **Single file to modify.** The agent only touches `train.toml`. Diffs are easy to review.
- **Fixed time budget.** Training always runs for 10 minutes, making experiments directly comparable.
- **2-GPU setup.** GPU 0 runs vLLM inference, GPU 1 runs the RL trainer. No memory contention.
- **Multiple environments.** GSM8K by default. Composite metric prevents overfitting to one task.
- **Built on prime-rl.** Production-ready async RL framework with GRPO/IPO/DPPO support.

## License

MIT
