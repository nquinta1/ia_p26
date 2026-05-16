# Deep RL — CartPole Lab

Live demo and image generation scripts for Module 23 (Reinforcement Learning).
Trains three methods — SARSA, Q-learning, DQN — on CartPole-v1 and lets you
watch them learn in real time.

---

## Prerequisites

- Python 3.9 or later
- No GPU needed — everything runs on CPU
- A working display for the live window (X11 or Wayland on Linux, native on macOS/Windows)

---

## Setup (one time)

```bash
# 1. Navigate here
cd clase/23_reinforcement_learning/deep_rl

# 2. Create the virtual environment and install dependencies (~2-5 min first time)
./setup.sh

# 3. Activate the environment
source .venv/bin/activate

# You should see (.venv) in your prompt. You're ready.
```

To deactivate when you're done:
```bash
deactivate
```

### What gets installed

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.5.1 (CPU) | Neural network for DQN |
| `gymnasium[classic-control]` | 1.0.0 | CartPole-v1 environment |
| `numpy` | 1.26.4 | Arrays and discretization |
| `matplotlib` | 3.9.4 | Live window and plot generation |
| `tqdm` | 4.67.1 | Terminal progress bars |

---

## Live demo: `demo_cartpole.py`

### Quick start

```bash
# Watch DQN learn (recommended first run)
python demo_cartpole.py --method dqn

# Compare all 3 methods, then see a final summary plot
python demo_cartpole.py --compare
```

### All flags

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--method` | `sarsa` `qlearning` `dqn` | `dqn` | Algorithm to train |
| `--compare` | — | off | Run all 3 methods sequentially, show final comparison |
| `--episodes` | any integer | `500` | Number of training episodes |
| `--speed` | `fast` `normal` `slow` | `normal` | How often to update the animation window |

Speed controls how many training steps occur between window refreshes:

| Speed | Steps per frame | Best for |
|-------|----------------|---------|
| `slow` | 1 | Watching every single action |
| `normal` | 5 | Balanced — smooth animation |
| `fast` | 20 | Getting to convergence quickly, less animation detail |

### Common commands

```bash
# Watch DQN converge over 500 episodes (default)
python demo_cartpole.py --method dqn

# Faster run — 200 episodes, update window every 20 steps
python demo_cartpole.py --method dqn --episodes 200 --speed fast

# Watch a tabular method (SARSA or Q-learning)
python demo_cartpole.py --method sarsa
python demo_cartpole.py --method qlearning

# Slow motion — see every step clearly
python demo_cartpole.py --method dqn --episodes 100 --speed slow

# Compare all 3 methods with 300 episodes each
python demo_cartpole.py --compare --episodes 300

# Quick comparison (150 episodes per method, ~2-3 min total)
python demo_cartpole.py --compare --episodes 150
```

### The window layout

When you run a single method, a window opens with 4 panels and a status bar:

```
┌─────────────────────────┬──────────────────────────┐
│                         │                          │
│   CartPole animation    │   Reward per episode     │
│   (live rendering)      │   + 50-ep rolling avg    │
│                         │   + solved line (475)    │
├─────────────────────────┼──────────────────────────┤
│                         │                          │
│   DQN: MSE loss curve   │   DQN: Q-value           │
│   Tabular: ε decay      │        distribution      │
│                         │   Tabular: Q-table       │
│                         │           heatmap        │
├─────────────────────────┴──────────────────────────┤
│   Status bar: episode | ε | avg reward | loss      │
└────────────────────────────────────────────────────┘
```

**Top-left — CartPole animation:**
The physical simulation rendered in real time. Early on (ε high) the agent acts randomly and the pole falls immediately. As ε decays and the agent learns, the cart starts making smooth corrective movements to keep the pole up.

**Top-right — Reward per episode:**
Each episode ends when the pole falls or reaches 500 steps. The thin line is the raw per-episode reward; the thick line is the 50-episode rolling average. The dashed line at 475 is the "solved" threshold. You want the thick line to cross that dashed line.

**Bottom-left — Loss (DQN) or ε decay (tabular):**
- *DQN*: MSE loss between the network's prediction and the TD target. Starts high (network predicts random values), decreases as the network learns. A loss that doesn't decrease suggests the learning rate is too high.
- *Tabular methods*: ε decays from 1.0 (pure exploration) to 0.05 (mostly greedy). The shape is always exponential — this panel confirms the schedule is working.

**Bottom-right — Q-value distribution (DQN) or Q-table heatmap (tabular):**
- *DQN*: Histogram of Q-values sampled from recent states. Early training: narrow cluster near 0. After learning: spread out with higher mean as the network correctly estimates long-horizon returns.
- *Tabular methods*: Heatmap showing max-Q averaged over the pole-angle × angular-velocity dimensions. Cells that are visited more often get higher values. After convergence you'll see a pattern — near-zero angle states have high value (good), extreme-angle states have low value (about to fall).

**Status bar:**
Shows at a glance: current episode, current ε, 50-episode average reward, and (for DQN) the most recent loss value.

### What to watch for

**With DQN (`--method dqn`):**
1. **Episodes 1-50**: Pole falls in ~10-20 steps. Animation looks chaotic.
2. **Episodes 50-150**: Buffer fills up, loss starts dropping, reward slowly climbs.
3. **Episodes 150-300**: ε is now below 0.3 — you can see the agent start correcting. Reward spikes up.
4. **Episodes 300-500**: Rolling average crosses 475. Episodes take longer and longer (more steps = more reward = more time to render).

The per-episode time in the terminal is itself a signal: if episodes are taking 2-4 seconds each, the pole is surviving for hundreds of steps — the agent is solving it.

**With tabular methods (`--method sarsa/qlearning`):**
The reward curve climbs a bit early (the agent learns to avoid the worst actions) but then plateaus around 50-80. The discretization loses too much information — nearby continuous states get treated as unrelated. The Q-table heatmap shows which state regions got visited, but the agent can never generalize across bins.

**With `--compare`:**
The 3 methods run sequentially in the terminal (with tqdm progress bars). At the end, a single window shows all 3 convergence curves overlaid. The DQN line should climb clearly above the two flat tabular lines.

### Headless environments (no display)

If you're on a server without a display, the script detects this automatically:

```
No display found. Summary saved to demo_results.png
```

It trains the model, then saves a PNG with the reward curve. All training logic is identical.

---

## Image generation: `lab_deep_rl.py`

Generates the 6 static images used in the course pages (08–13).

```bash
python lab_deep_rl.py
```

Output goes to `../images/`. Expected runtime: ~5-8 minutes on a modern CPU.

| Image | File | What it shows |
|-------|------|---------------|
| 08 | `08_dqn_architecture.png` | Network diagram: 4→64→64→2 with ReLU |
| 09 | `09_experience_replay.png` | Replay buffer: store arrow + random sample arrow |
| 10 | `10_target_network.png` | Online vs frozen target network, copy arrow, TD equation |
| 11 | `11_convergence_comparison.png` | All 3 methods over 500 episodes with rolling averages |
| 12 | `12_loss_curve.png` | DQN MSE loss vs training step with buffer-full annotation |
| 13 | `13_cartpole_frames.png` | 4 frames from a solved episode (pole stays vertical) |

---

## Files in this directory

```
deep_rl/
├── README.md            ← this file
├── setup.sh             ← creates .venv/ and installs dependencies
├── requirements.txt     ← pinned versions (CPU torch, gymnasium, numpy, matplotlib, tqdm)
├── demo_cartpole.py     ← live interactive demo
├── lab_deep_rl.py       ← static image generation for course pages
└── .venv/               ← virtual environment (gitignored, created by setup.sh)
```

---

## Troubleshooting

**`./setup.sh: Permission denied`**
```bash
chmod +x setup.sh && ./setup.sh
```

**`ModuleNotFoundError: No module named 'gymnasium'`**
The venv is not active. Run `source .venv/bin/activate` first.

**Window doesn't open (Linux)**
Check that `$DISPLAY` is set: `echo $DISPLAY`. If empty, you're in a headless session — the script will fall back to saving a PNG automatically.

**Window opens but animation is blank/gray**
The first few episodes can have blank frames if CartPole terminates in 1 step (the very first reset). This clears up after episode 2-3.

**Training is very slow**
With `--speed slow` every step triggers a window refresh (matplotlib `pause(0.001)`), which is expensive. Use `--speed fast` if you mainly care about the learning curves, not the animation.

**`UserWarning: This figure includes Axes that are not compatible with tight_layout`**
Harmless — the colorbar for the Q-table heatmap causes this. Doesn't affect output.

**Want to reproduce exact results from the course images**
Use the same seeds as `lab_deep_rl.py`:
```python
import numpy, random, torch
numpy.random.seed(42); torch.manual_seed(42); random.seed(42)
```
