# -*- coding: utf-8 -*-
"""
RL.py — Tabular Q-learning（自包含版本，與 Part I 分離）
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class RLProblem:
    mdp: any
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    rng: np.random.Generator = np.random.default_rng(0)

    def reset(self, s0: int = 0):
        return s0

    def step(self, s: int, a: int) -> Tuple[int, float]:
        Psa = self.mdp.P[a, s]  # [S]
        s_next = self.rng.choice(self.mdp.nStates, p=Psa)
        r = self.mdp.R[s, a]
        return s_next, r

    def act_eps_greedy(self, Q: np.ndarray, s: int) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.mdp.nActions)
        return int(np.argmax(Q[s]))

    def q_learning(self, s0: int, initialQ: np.ndarray, nEpisodes: int, nSteps: int):
        S, A = self.mdp.nStates, self.mdp.nActions
        Q = initialQ.copy()
        for ep in range(nEpisodes):
            s = self.reset(s0)
            for t in range(nSteps):
                a = self.act_eps_greedy(Q, s)
                s2, r = self.step(s, a)
                td_target = r + self.gamma * np.max(Q[s2])
                Q[s, a] = (1 - self.alpha) * Q[s, a] + self.alpha * td_target
                s = s2
        policy = np.argmax(Q, axis=1)
        return Q, policy

# 迷宮產生器（與 Part I 無關，僅供 RL 使用）：返回 (mdp_like, s_of, coords)
def make_grid_maze(width=4, height=3, terminal_dict={(3,0): +1.0, (3,1): -1.0}, walls={(1,1)},
                   slip_prob=0.2, step_reward=-0.04, gamma=0.99):
    import numpy as np
    from dataclasses import dataclass
    A = 4
    coords = [(x,y) for y in range(height) for x in range(width) if (x,y) not in walls]
    S = len(coords)
    idx = {xy:i for i,xy in enumerate(coords)}
    def in_bounds(x,y): return 0 <= x < width and 0 <= y < height and (x,y) not in walls
    actions = {0:(0,-1),1:(1,0),2:(0,1),3:(-1,0)}
    def move(x,y,a):
        dx,dy = actions[a]
        nx,ny = x+dx, y+dy
        return (x,y) if not in_bounds(nx,ny) else (nx,ny)
    left = {0:3,1:0,2:1,3:2}
    right= {0:1,1:2,2:3,3:0}
    P = np.zeros((A, S, S), dtype=float)
    R = np.full((S, A), step_reward, dtype=float)
    terminals = {idx[xy]: r for xy,r in terminal_dict.items() if xy in idx}
    for s, (x,y) in enumerate(coords):
        if s in terminals:
            for a in range(A):
                P[a, s, s] = 1.0
                R[s, a] = terminals[s]
            continue
        for a in range(A):
            s_main = idx[move(x,y,a)]
            s_left = idx[move(x,y,left[a])]
            s_right= idx[move(x,y,right[a])]
            P[a, s, s_main] += (1.0 - slip_prob)
            P[a, s, s_left] += slip_prob/2.0
            P[a, s, s_right]+= slip_prob/2.0
    @dataclass
    class M:
        nStates:int; nActions:int; P:any; R:any
    return M(S,4,P,R), (lambda x,y: idx[(x,y)]), coords
