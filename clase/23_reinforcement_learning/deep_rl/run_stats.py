"""Runs all 4 methods and saves detailed statistics to stats.json."""
import json, collections, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)

# ── shared helpers ────────────────────────────────────────────────────────────
def make_bins():
    return (np.linspace(-2.4,2.4,11), np.linspace(-3.0,3.0,11),
            np.linspace(-0.2,0.2,11), np.linspace(-3.0,3.0,11))

def disc(obs, bins):
    return tuple(max(0,min(9,int(np.digitize(obs[i],bins[i]))-1)) for i in range(4))

def rolling50(arr):
    if len(arr) < 50: return []
    return list(np.convolve(arr, np.ones(50)/50, mode='valid'))

def first_cross(roll50, threshold):
    for i,v in enumerate(roll50):
        if v >= threshold: return i + 50   # episode number (1-indexed offset)
    return None

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
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

# ── tabular ───────────────────────────────────────────────────────────────────
def run_tabular(use_sarsa=False, n=500, alpha=0.1, gamma=0.99):
    np.random.seed(SEED)
    bins=make_bins(); Q=np.zeros((10,10,10,10,2)); eps=1.0; rets=[]
    env=gym.make("CartPole-v1")
    visited=set()
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
                best=float(np.max(Q[ns])) if not done else 0.0
                Q[s][a]+=alpha*(r+gamma*best-Q[s][a])
            s=ns
        eps=max(0.05,eps*0.995); rets.append(G)
    env.close()
    nonzero=int(np.sum(np.any(Q!=0,axis=4)))
    return rets, nonzero, len(visited)

# ── dqn ───────────────────────────────────────────────────────────────────────
def run_dqn(n=500):
    np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
    env=gym.make("CartPole-v1")
    online=DQN(); target=DQN(); target.load_state_dict(online.state_dict()); target.eval()
    opt=optim.Adam(online.parameters(),lr=1e-3)
    buf=ReplayBuffer(10_000); eps=1.0; rets=[]; losses=[]; step=0
    buffer_full_ep=None
    for ep in range(n):
        obs,_=env.reset(); s=np.array(obs,dtype=np.float32); done=False; G=0.0
        while not done:
            if np.random.random()<eps: a=env.action_space.sample()
            else:
                with torch.no_grad(): a=int(online(torch.tensor(s).unsqueeze(0)).argmax().item())
            o2,r,te,tr,_=env.step(a); done=te or tr; ns=np.array(o2,dtype=np.float32); G+=r
            buf.push(s,a,r,ns,float(done)); s=ns; step+=1
            if len(buf)>=64:
                if buffer_full_ep is None: buffer_full_ep=ep
                sb,ab,rb,nsb,db=buf.sample(64)
                with torch.no_grad(): tgt=rb+0.99*target(nsb).max(1).values*(1-db)
                cq=online(sb).gather(1,ab.unsqueeze(1)).squeeze(1)
                loss=nn.functional.mse_loss(cq,tgt)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss.item()))
            if step%50==0: target.load_state_dict(online.state_dict())
        eps=max(0.05,eps*0.995); rets.append(G)
    env.close()
    return rets, losses, buffer_full_ep

# ── run all ───────────────────────────────────────────────────────────────────
print("Running Q-table (500 eps)...")
r_qt,  nz_qt,  vis_qt  = run_tabular(use_sarsa=False)
print("Running SARSA (500 eps)...")
r_sa,  nz_sa,  vis_sa  = run_tabular(use_sarsa=True)
print("Running Q-learning (500 eps)...")
r_ql,  nz_ql,  vis_ql  = run_tabular(use_sarsa=False)
print("Running DQN (500 eps)...")
r_dqn, losses, buf_full_ep = run_dqn()

# ── statistics ────────────────────────────────────────────────────────────────
def stats(rets):
    roll = rolling50(rets)
    return {
        "final_avg50":   round(float(np.mean(rets[-50:])),1) if len(rets)>=50 else None,
        "max_single_ep": round(float(max(rets)),1),
        "max_avg50":     round(float(max(roll)),1) if roll else None,
        "ep_cross200":   first_cross(roll, 200),
        "ep_cross300":   first_cross(roll, 300),
        "ep_cross400":   first_cross(roll, 400),
        "ep_cross475":   first_cross(roll, 475),
        "first10_avg":   round(float(np.mean(rets[:10])),1),
        "last50_avg":    round(float(np.mean(rets[-50:])),1) if len(rets)>=50 else None,
    }

results = {
    "qtable":    {**stats(r_qt),  "nonzero_cells": nz_qt,  "visited_pairs": vis_qt},
    "sarsa":     {**stats(r_sa),  "nonzero_cells": nz_sa,  "visited_pairs": vis_sa},
    "qlearning": {**stats(r_ql),  "nonzero_cells": nz_ql,  "visited_pairs": vis_ql},
    "dqn": {
        **stats(r_dqn),
        "buffer_full_episode": buf_full_ep,
        "loss_at_start":   round(float(np.mean(losses[:50])),4)  if len(losses)>=50  else None,
        "loss_at_end":     round(float(np.mean(losses[-200:])),4) if len(losses)>=200 else None,
        "loss_reduction_pct": round((1 - np.mean(losses[-200:])/np.mean(losses[:50]))*100,1)
                              if len(losses)>=200 and np.mean(losses[:50])>0 else None,
        "total_steps":     sum(int(r) for r in r_dqn),
    },
}

Path("stats.json").write_text(json.dumps(results, indent=2))
print("\nDone. stats.json written.")
print(json.dumps(results, indent=2))
