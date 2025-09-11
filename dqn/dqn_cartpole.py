# -*- coding: utf-8 -*-
import os, argparse, random, math, numpy as np
import torch, torch.nn as nn, torch.optim as optim

try:
    import gymnasium as gym
    GYMN = True
except Exception:
    import gym
    GYMN = False

class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x): return self.net(x)

class Replay:
    def __init__(self, capacity=50000):
        self.capacity = capacity; self.buf=[]; self.pos=0
    def push(self, exp):
        if len(self.buf) < self.capacity: self.buf.append(exp)
        else: self.buf[self.pos] = exp
        self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), size=batch_size, replace=False)
        return [self.buf[i] for i in idx]
    def __len__(self): return len(self.buf)

def make_env(seed):
    env = gym.make("CartPole-v1")
    if GYMN: env.reset(seed=seed)
    else: env.seed(seed)
    return env

def run_once(seed, episodes, target_update_ep, batch_size, gamma=0.99, lr=1e-3, eps_start=1.0, eps_end=0.05, eps_decay=300):
    rng = np.random.default_rng(seed)
    random.seed(seed); torch.manual_seed(seed)
    env = make_env(seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    q = QNet(obs_dim, n_actions); q_t = QNet(obs_dim, n_actions); q_t.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)
    replay = Replay(50000)

    def epsilon(ep):
        return eps_end + (eps_start - eps_end) * math.exp(-ep / eps_decay)

    rewards = []
    for ep in range(episodes):
        obs = env.reset(seed=(seed+ep))[0] if GYMN else env.reset()
        done = False; total = 0.0
        while not done:
            if rng.random() < epsilon(ep): a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = int(torch.argmax(q(torch.tensor(obs, dtype=torch.float32))).item())
            step_out = env.step(a)
            if GYMN:
                obs2, r, terminated, truncated, _ = step_out; done = terminated or truncated
            else:
                obs2, r, done, _ = step_out
            total += r
            replay.push((obs, a, r, obs2, done)); obs = obs2
            if len(replay) >= batch_size:
                batch = replay.sample(batch_size)
                b_obs  = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
                b_a    = torch.tensor([b[1] for b in batch], dtype=torch.int64).unsqueeze(1)
                b_r    = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1)
                b_obs2 = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
                b_done = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1)
                q_pred = q(b_obs).gather(1, b_a)
                with torch.no_grad():
                    q_next = q_t(b_obs2).max(1, keepdim=True)[0]
                    target = b_r + gamma * (1.0 - b_done) * q_next
                loss = nn.functional.mse_loss(q_pred, target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        rewards.append(total)
        if (ep + 1) % target_update_ep == 0: q_t.load_state_dict(q.state_dict())
    env.close()
    return np.array(rewards, dtype=float)

def moving_average(x, k=25):
    if len(x) < k: return x
    c = np.cumsum(np.insert(x, 0, 0)); return (c[k:] - c[:-k]) / k

def main():
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--target_update_ep", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()
    os.makedirs("figs", exist_ok=True)
    curves = []
    for s in range(args.seeds):
        r = run_once(seed=1234+s, episodes=args.episodes, target_update_ep=args.target_update_ep, batch_size=args.batch_size)
        curves.append(moving_average(r, k=25))
    L = min(len(c) for c in curves)
    ys = np.stack([c[:L] for c in curves], axis=0).mean(axis=0)
    xs = np.arange(L)
    plt.figure()
    plt.plot(xs, ys, label=f"TU={args.target_update_ep}, B={args.batch_size}")
    plt.xlabel("Episodes"); plt.ylabel("最近 25 回合平均回饋"); plt.legend()
    out = "figs/dqn_single.png"; plt.savefig(out, dpi=150, bbox_inches="tight"); print("圖已輸出：", out)

if __name__ == "__main__":
    main()
