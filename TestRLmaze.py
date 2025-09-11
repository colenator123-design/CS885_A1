# -*- coding: utf-8 -*-
import argparse, os, json, numpy as np
from RL import RLProblem, make_grid_maze

def run_trials(epsilon, episodes, steps, trials, seed_base=0):
    mdp, s_of, coords = make_grid_maze(gamma=0.99)
    avg_rewards_per_ep = np.zeros(episodes, dtype=float)
    for k in range(trials):
        rng = np.random.default_rng(seed_base + k)
        rl = RLProblem(mdp=mdp, alpha=0.1, gamma=0.99, epsilon=epsilon, rng=rng)
        Q0 = np.zeros((mdp.nStates, mdp.nActions))
        per_ep = np.zeros(episodes, dtype=float)
        for ep in range(episodes):
            s = 0
            disc = 1.0
            total = 0.0
            for t in range(steps):
                a = rl.act_eps_greedy(Q0, s)
                s2, r = rl.step(s, a)
                td_target = r + rl.gamma * np.max(Q0[s2])
                Q0[s, a] = (1 - rl.alpha) * Q0[s, a] + rl.alpha * td_target
                total += disc * r
                disc *= rl.gamma
                s = s2
            per_ep[ep] = total
        avg_rewards_per_ep += per_ep
    avg_rewards_per_ep /= trials
    return avg_rewards_per_ep.tolist()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()

    os.makedirs("figs", exist_ok=True)
    result = {}
    for eps in [0.05, 0.1, 0.3, 0.5]:
        print(f"跑 epsilon={eps} ...")
        result[str(eps)] = run_trials(eps, args.episodes, args.steps, args.trials)

    with open("figs/qlearning_maze_results.json", "w", encoding="utf-8") as f:
        json.dump({"episodes": args.episodes, "curves": result}, f, ensure_ascii=False, indent=2)
    print("完成，結果儲存到 figs/qlearning_maze_results.json")
