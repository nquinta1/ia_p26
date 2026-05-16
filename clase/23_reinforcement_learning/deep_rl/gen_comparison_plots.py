"""Generate extra diagnostic images for the comparison page."""
import json, collections, random, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch, torch.nn as nn, torch.optim as optim
import gymnasium as gym

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)

COLORS = {"blue":"#2E86AB","red":"#E94F37","green":"#27AE60","gray":"#7F8C8D",
          "orange":"#F39C12","purple":"#8E44AD","dark":"#2C3E50","teal":"#1ABC9C"}
METHOD_COLORS = {"qtable":COLORS["gray"],"sarsa":COLORS["blue"],
                 "qlearning":COLORS["green"],"dqn":COLORS["red"]}
METHOD_LABELS = {"qtable":"Q-tabla","sarsa":"SARSA","qlearning":"Q-learning","dqn":"DQN"}
plt.style.use("seaborn-v0_8-whitegrid")
IMAGES_DIR = Path("../images")

def _save(fig, name):
    fig.savefig(IMAGES_DIR/name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {name}")

# ── helpers ───────────────────────────────────────────────────────────────────
def make_bins():
    return (np.linspace(-2.4,2.4,11),np.linspace(-3.0,3.0,11),
            np.linspace(-0.2,0.2,11),np.linspace(-3.0,3.0,11))
def disc(obs,bins):
    return tuple(max(0,min(9,int(np.digitize(obs[i],bins[i]))-1)) for i in range(4))

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,x): return self.net(x)

class ReplayBuffer:
    def __init__(self,cap=10_000): self.buf=collections.deque(maxlen=cap)
    def push(self,*t): self.buf.append(t)
    def sample(self,n):
        b=random.sample(self.buf,n); s,a,r,ns,d=zip(*b)
        return (torch.tensor(np.array(s),dtype=torch.float32),
                torch.tensor(a,dtype=torch.long),
                torch.tensor(r,dtype=torch.float32),
                torch.tensor(np.array(ns),dtype=torch.float32),
                torch.tensor(d,dtype=torch.float32))
    def __len__(self): return len(self.buf)

def run_tabular_detailed(use_sarsa=False, n=500, alpha=0.1, gamma=0.99):
    np.random.seed(SEED)
    bins=make_bins(); Q=np.zeros((10,10,10,10,2)); eps=1.0; rets=[]; visited_by_ep=[]
    visited=set(); env=gym.make("CartPole-v1")
    for ep in range(n):
        obs,_=env.reset(); s=disc(obs,bins)
        if use_sarsa:
            a=env.action_space.sample() if np.random.random()<eps else int(np.argmax(Q[s]))
        done=False; G=0.0
        while not done:
            if not use_sarsa:
                a=env.action_space.sample() if np.random.random()<eps else int(np.argmax(Q[s]))
            obs2,r,te,tr,_=env.step(a); done=te or tr; G+=r; ns=disc(obs2,bins)
            visited.add(s+(a,))
            if use_sarsa:
                if done: na,nq=0,0.0
                else:
                    na=env.action_space.sample() if np.random.random()<eps else int(np.argmax(Q[ns]))
                    nq=float(Q[ns][na])
                Q[s][a]+=alpha*(r+gamma*nq-Q[s][a]); a=na
            else:
                best=float(np.max(Q[ns])) if not done else 0.0; Q[s][a]+=alpha*(r+gamma*best-Q[s][a])
            s=ns
        eps=max(0.05,eps*0.995); rets.append(G); visited_by_ep.append(len(visited))
    env.close()
    return rets, visited_by_ep, Q

def run_dqn_detailed(n=500):
    np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
    env=gym.make("CartPole-v1")
    online=DQN(); target=DQN(); target.load_state_dict(online.state_dict()); target.eval()
    opt=optim.Adam(online.parameters(),lr=1e-3)
    buf=ReplayBuffer(10_000); eps=1.0; rets=[]; losses=[]; eps_hist=[]; step=0
    for ep in range(n):
        obs,_=env.reset(); s=np.array(obs,dtype=np.float32); done=False; G=0.0
        while not done:
            if np.random.random()<eps: a=env.action_space.sample()
            else:
                with torch.no_grad(): a=int(online(torch.tensor(s).unsqueeze(0)).argmax().item())
            o2,r,te,tr,_=env.step(a); done=te or tr; ns=np.array(o2,dtype=np.float32); G+=r
            buf.push(s,a,r,ns,float(done)); s=ns; step+=1
            if len(buf)>=64:
                sb,ab,rb,nsb,db=buf.sample(64)
                with torch.no_grad(): tgt=rb+0.99*target(nsb).max(1).values*(1-db)
                cq=online(sb).gather(1,ab.unsqueeze(1)).squeeze(1)
                loss=nn.functional.mse_loss(cq,tgt)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append((step,float(loss.item())))
            if step%50==0: target.load_state_dict(online.state_dict())
        eps=max(0.05,eps*0.995); rets.append(G); eps_hist.append(eps)
    env.close()
    return rets, losses, eps_hist

# ── run ───────────────────────────────────────────────────────────────────────
print("Running Q-tabla..."); r_qt, v_qt, Q_qt = run_tabular_detailed(False)
print("Running SARSA...");   r_sa, v_sa, Q_sa = run_tabular_detailed(True)
print("Running Q-learning...");r_ql, v_ql, Q_ql = run_tabular_detailed(False)
print("Running DQN...");     r_dqn, losses_dqn, eps_dqn = run_dqn_detailed()

# ─────────────────────────────────────────────────────────────────────────────
# Image 14: Coverage + visited cells + Q-table density
# ─────────────────────────────────────────────────────────────────────────────
def plot_coverage():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: visited cells over episodes
    ax = axes[0]
    ep = np.arange(1,501)
    ax.plot(ep, v_qt, color=METHOD_COLORS["qtable"], lw=2, label="Q-tabla")
    ax.plot(ep, v_sa, color=METHOD_COLORS["sarsa"],  lw=2, label="SARSA")
    ax.plot(ep, v_ql, color=METHOD_COLORS["qlearning"], lw=2, label="Q-learning")
    ax.axhline(20_000, color=COLORS["dark"], ls="--", lw=1.2, alpha=0.5, label="Total (20 000)")
    ax.set_xlabel("Episodio"); ax.set_ylabel("Pares (s,a) únicos visitados")
    ax.set_title("Cobertura acumulada de la tabla Q")
    ax.legend(fontsize=8); ax.set_ylim(0, 21_000)

    # Panel 2: bar chart final coverage %
    ax2 = axes[1]
    methods = ["qtable","sarsa","qlearning"]
    vals    = [v_qt[-1], v_sa[-1], v_ql[-1]]
    pcts    = [v/20_000*100 for v in vals]
    colors  = [METHOD_COLORS[m] for m in methods]
    bars = ax2.bar([METHOD_LABELS[m] for m in methods], pcts, color=colors, width=0.5)
    for bar, pct, v in zip(bars, pcts, vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("% de la tabla visitado")
    ax2.set_title("Cobertura final (500 episodios)")
    ax2.set_ylim(0, 10)
    ax2.axhline(100, color=COLORS["dark"], ls="--", alpha=0.3)

    # Panel 3: Q-table heatmap for best tabular (q-learning)
    ax3 = axes[2]
    q_max = np.max(Q_ql, axis=4)          # max over actions -> (10,10,10,10)
    q_2d  = q_max.mean(axis=(0,1))        # avg over cart dims -> (10,10) pole dims
    # mask unvisited
    visited_mask = (np.max(np.abs(Q_ql),axis=4) > 0).mean(axis=(0,1))
    q_display = np.where(visited_mask>0, q_2d, np.nan)
    im = ax3.imshow(q_display, aspect="auto", cmap="RdYlGn",
                    origin="lower", interpolation="nearest")
    ax3.set_xlabel("Vel. angular (índice bin)"); ax3.set_ylabel("Ángulo (índice bin)")
    ax3.set_title("Q-learning: max-Q por región\n(blanco = nunca visitado)")
    plt.colorbar(im, ax=ax3, fraction=0.046)

    fig.suptitle("Métodos tabulares: cobertura del espacio de estados",
                 fontsize=12, color=COLORS["dark"])
    plt.tight_layout()
    _save(fig, "14_coverage_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# Image 15: DQN learning phases annotated
# ─────────────────────────────────────────────────────────────────────────────
def plot_dqn_phases():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: reward with phase annotations
    ax = axes[0]
    ep  = np.arange(1,501)
    ax.plot(ep, r_dqn, color=COLORS["red"], alpha=0.25, lw=0.7)
    if len(r_dqn)>=50:
        roll = np.convolve(r_dqn, np.ones(50)/50, mode="valid")
        ax.plot(np.arange(50,501), roll, color=COLORS["red"], lw=2.5, label="Media 50 ep.")
    ax.axhline(475, color=COLORS["dark"], ls="--", lw=1.2, alpha=0.6, label="Resuelto (475)")

    # Phase shading
    ax.axvspan(1,   50,  alpha=0.07, color=COLORS["gray"],   label="Fase 1: exploración pura")
    ax.axvspan(50, 280,  alpha=0.07, color=COLORS["orange"], label="Fase 2: aprendizaje lento")
    ax.axvspan(280,500,  alpha=0.07, color=COLORS["green"],  label="Fase 3: convergencia")

    # Annotations
    ax.annotate("Buffer lleno\n(ep. 4)", xy=(4,10), xytext=(30,60),
                fontsize=8, color=COLORS["orange"],
                arrowprops=dict(arrowstyle="->",color=COLORS["orange"],lw=1))
    ax.annotate("Primera vez\n≥200 (ep. 289)", xy=(289, 200), xytext=(200,250),
                fontsize=8, color=COLORS["green"],
                arrowprops=dict(arrowstyle="->",color=COLORS["green"],lw=1))
    ax.annotate("Mejor avg50\n(ep. ~460, avg=412)", xy=(460,412), xytext=(360,440),
                fontsize=8, color=COLORS["dark"],
                arrowprops=dict(arrowstyle="->",color=COLORS["dark"],lw=1))

    ax.set_xlabel("Episodio"); ax.set_ylabel("Recompensa")
    ax.set_title("DQN: fases del aprendizaje"); ax.set_ylim(0,520)
    ax.legend(fontsize=7.5, loc="upper left")

    # Right: epsilon decay overlay
    ax2 = axes[1]
    ep_arr = np.arange(1, len(eps_dqn)+1)
    color2 = COLORS["purple"]
    ax2.plot(ep_arr, eps_dqn, color=color2, lw=2, label="ε")
    ax2.axhline(0.5, color=color2, ls=":", alpha=0.5)
    ax2.axhline(0.1, color=color2, ls=":", alpha=0.5)
    ax2.text(510, 0.51, "ε=0.5", fontsize=8, color=color2)
    ax2.text(510, 0.11, "ε=0.1", fontsize=8, color=color2)
    ax2.set_xlabel("Episodio"); ax2.set_ylabel("ε (tasa de exploración)", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax3 = ax2.twinx()
    roll_d = np.convolve(r_dqn, np.ones(50)/50, mode="valid") if len(r_dqn)>=50 else []
    if len(roll_d)>0:
        ax3.plot(np.arange(50,501), roll_d, color=COLORS["red"], lw=2, alpha=0.8, label="Media 50 ep.")
    ax3.set_ylabel("Recompensa (media 50 ep.)", color=COLORS["red"])
    ax3.tick_params(axis='y', labelcolor=COLORS["red"])
    ax3.set_ylim(0,520)
    ax2.set_title("DQN: decaimiento de ε vs. recompensa")

    lines1 = [plt.Line2D([0],[0],color=color2,lw=2),
              plt.Line2D([0],[0],color=COLORS["red"],lw=2)]
    ax2.legend(lines1, ["ε","Media 50 ep."], fontsize=9, loc="center left")
    fig.suptitle("DQN — anatomía del entrenamiento",fontsize=12,color=COLORS["dark"])
    plt.tight_layout()
    _save(fig, "15_dqn_phases.png")

# ─────────────────────────────────────────────────────────────────────────────
# Image 16: Loss curve re-annotated explaining why it grows
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_explained():
    if not losses_dqn: return
    steps = np.array([s for s,_ in losses_dqn])
    vals  = np.array([v for _,v in losses_dqn])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: raw loss + smoothed
    ax = axes[0]
    ax.plot(steps, vals, color=COLORS["blue"], alpha=0.18, lw=0.6)
    win=300
    if len(vals)>=win:
        smooth = np.convolve(vals, np.ones(win)/win, mode="valid")
        ax.plot(steps[win-1:], smooth, color=COLORS["blue"], lw=2.2, label="Suavizado (v=300)")
    ax.set_xlabel("Paso de entrenamiento"); ax.set_ylabel("Pérdida MSE")
    ax.set_title("Pérdida DQN: escala cruda")
    ax.legend(fontsize=9)

    # Right: log scale + episode reward annotation
    ax2 = axes[1]
    ax2.semilogy(steps, vals, color=COLORS["blue"], alpha=0.18, lw=0.6)
    if len(vals)>=win:
        smooth2 = np.convolve(vals, np.ones(win)/win, mode="valid")
        ax2.semilogy(steps[win-1:], smooth2, color=COLORS["blue"], lw=2.2, label="Suavizado (log)")

    # Overlay avg reward (rescaled to loss range) as reference
    avg_reward = np.array(r_dqn)
    # approximate step for each episode
    cumsteps = np.cumsum([int(r) for r in r_dqn])
    ax2b = ax2.twinx()
    ax2b.plot(cumsteps, avg_reward, color=COLORS["red"], lw=1.5, alpha=0.7, label="Recompensa ep.")
    ax2b.set_ylabel("Recompensa por episodio", color=COLORS["red"])
    ax2b.tick_params(axis='y', labelcolor=COLORS["red"])

    ax2.set_xlabel("Paso de entrenamiento"); ax2.set_ylabel("Pérdida MSE (escala log)")
    ax2.set_title("Pérdida (log) + recompensa superpuesta\n→ la pérdida sube porque los Q-valores crecen")
    lines=[plt.Line2D([0],[0],color=COLORS["blue"],lw=2),
           plt.Line2D([0],[0],color=COLORS["red"],lw=1.5)]
    ax2.legend(lines,["Pérdida (log, suavizada)","Recompensa"],fontsize=8,loc="upper left")

    fig.suptitle("Por qué la pérdida DQN aumenta conforme el agente mejora",
                 fontsize=11, color=COLORS["dark"])
    plt.tight_layout()
    _save(fig, "16_loss_explained.png")

plot_coverage()
plot_dqn_phases()
plot_loss_explained()
print("\nAll images saved.")
