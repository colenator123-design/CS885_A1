# -*- coding: utf-8 -*-
import argparse, numpy as np
import MDP
from mazes import make_gridworld_4x3
import csv

def fmt_grid(vals, coords, W=4, H=3):
    grid = [["{:6.2f}".format(0.0) for _ in range(W)] for _ in range(H)]
    for s,v in enumerate(vals):
        x,y = coords[s]
        grid[y][x] = "{:6.2f}".format(v)
    return "\n".join(" ".join(row) for row in grid)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tol", type=float, default=0.01)
    args = ap.parse_args()

    T,R,gamma,coords = make_gridworld_4x3(gamma=args.gamma)
    mdp = MDP.MDP(T,R,gamma)
    
    summary_lines = []

    # --- Value Iteration ---
    print("[Value Iteration] 從 V=0, tol=%.3f" % args.tol)
    V,it,eps = mdp.valueIteration(initialV=np.zeros(mdp.nStates), tolerance=args.tol)
    pi_vi = mdp.extractPolicy(V)
    print("iters:", it, "eps:", eps)
    print("pi*:", pi_vi)
    print("V*:\n", fmt_grid(V, coords))
    
    summary_lines.append(f"Value Iteration Iterations: {it}\n")
    
    with open('part1_value_iteration.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['state', 'value', 'policy_action'])
        for s in range(mdp.nStates):
            writer.writerow([s, V[s], pi_vi[s]])
    print("... Value Iteration results saved to part1_value_iteration.csv")

    # --- Policy Iteration ---
    print("\n[Policy Iteration] 從全 0 策略起始")
    pi,V_pi,it_pi = mdp.policyIteration(np.zeros(mdp.nStates, dtype=int))
    print("iters:", it_pi)
    print("π:", pi)
    print("Vπ:\n", fmt_grid(V_pi, coords))

    summary_lines.append(f"Policy Iteration Iterations: {it_pi}\n")

    with open('part1_policy_iteration.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['state', 'value', 'policy_action'])
        for s in range(mdp.nStates):
            writer.writerow([s, V_pi[s], pi[s]])
    print("... Policy Iteration results saved to part1_policy_iteration.csv")

    # --- Modified Policy Iteration ---
    print("\n[Modified Policy Iteration] k=1..10 外圈次數：")
    mpi_results = []
    for k in range(1, 11):
        pi_k,V_k,it_k,tol_k = mdp.modifiedPolicyIteration(np.zeros(mdp.nStates, dtype=int),
                                                          np.zeros(mdp.nStates), k=k, tolerance=args.tol)
        print("k=%2d → iters=%d, tol=%.4f" % (k, it_k, tol_k))
        mpi_results.append([k, it_k])

    with open('part1_modified_policy_iteration.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['k', 'total_iterations'])
        writer.writerows(mpi_results)
    print("... Modified Policy Iteration results saved to part1_modified_policy_iteration.csv.")

    # --- Write Summary File ---
    with open('part1_summary.txt', 'w') as f:
        f.writelines(summary_lines)
    print("... Iteration counts saved to part1_summary.txt")
