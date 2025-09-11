# -*- coding: utf-8 -*-
import argparse, os, csv, numpy as np
import matplotlib.pyplot as plt
from dqn_cartpole import run_once, moving_average

def scan_target(episodes, seeds):
    os.makedirs("figs", exist_ok=True)
    settings = [1, 10, 50, 100]
    all_curves = {}
    for tu in settings:
        curves = []
        for s in range(seeds):
            r = run_once(seed=2025+s, episodes=episodes, target_update_ep=tu, batch_size=64)
            curves.append(moving_average(r, k=25))
        L = min(len(c) for c in curves)
        ys = np.stack([c[:L] for c in curves], axis=0).mean(axis=0)
        all_curves[str(tu)] = ys

    # Save results to CSV
    csv_out = "../part3_target_update.csv"
    with open(csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'target_update_frequency', 'average_reward'])
        for tu, ys in all_curves.items():
            for episode, reward in enumerate(ys):
                writer.writerow([episode, tu, reward])
    print("已輸出 CSV:", csv_out)

    plt.figure()
    for tu, ys in all_curves.items():
        xs = np.arange(len(ys)); plt.plot(xs, ys, label=f"TargetUpdate={tu}")
    plt.xlabel("Episodes"); plt.ylabel("最近 25 回合平均回饋（5 seeds 平均）"); plt.legend()
    out = "figs/dqn_target_update_scan.png"; plt.savefig(out, dpi=150, bbox_inches="tight"); print("已輸出：", out)

def scan_batch(episodes, seeds):
    os.makedirs("figs", exist_ok=True)
    settings = [1, 10, 50, 100]
    all_curves = {}
    for bs in settings:
        curves = []
        for s in range(seeds):
            r = run_once(seed=3000+s, episodes=episodes, target_update_ep=10, batch_size=bs)
            curves.append(moving_average(r, k=25))
        L = min(len(c) for c in curves)
        ys = np.stack([c[:L] for c in curves], axis=0).mean(axis=0)
        all_curves[str(bs)] = ys

    # Save results to CSV
    csv_out = "../part3_batch_size.csv"
    with open(csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'batch_size', 'average_reward'])
        for bs, ys in all_curves.items():
            for episode, reward in enumerate(ys):
                writer.writerow([episode, bs, reward])
    print("已輸出 CSV:", csv_out)

    plt.figure()
    for bs, ys in all_curves.items():
        xs = np.arange(len(ys)); plt.plot(xs, ys, label=f"Batch={bs}")
    plt.xlabel("Episodes"); plt.ylabel("最近 25 回合平均回饋（5 seeds 平均）"); plt.legend()
    out = "figs/dqn_batch_size_scan.png"; plt.savefig(out, dpi=150, bbox_inches="tight"); print("已輸出：", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan-target", "scan-batch"])
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    if args.mode == "scan-target":
        scan_target(args.episodes, args.seeds)
    else:
        scan_batch(args.episodes, args.seeds)
