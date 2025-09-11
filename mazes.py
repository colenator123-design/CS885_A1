# -*- coding: utf-8 -*-
"""
mazes.py — 產生 4x3 迷宮的 MDP：回傳 (T, R, gamma, coords)
- State 編號跳過牆 (1,1)，終點 (+1) 在 (3,0)，壞終點 (-1) 在 (3,1)
- 動作: 0=up,1=right,2=down,3=left；滑移 slip_prob 到左右各一半
- R 的形狀為 [A,S]，以 step_reward 為基礎，terminal 上覆寫
"""
import numpy as np

def make_gridworld_4x3(gamma=0.99, slip_prob=0.2, step_reward=-0.04,
                       width=4, height=3, walls={(1,1)}, terminals={(3,0):+1.0,(3,1):-1.0}):
    A = 4
    coords = [(x,y) for y in range(height) for x in range(width) if (x,y) not in walls]
    S = len(coords)
    idx = {xy:i for i,xy in enumerate(coords)}
    def in_bounds(x,y): return 0 <= x < width and 0 <= y < height and (x,y) not in walls
    moves = {0:(0,-1),1:(1,0),2:(0,1),3:(-1,0)}
    left = {0:3,1:0,2:1,3:2}
    right= {0:1,1:2,2:3,3:0}
    def step(x,y,a):
        dx,dy=moves[a]
        nx,ny=x+dx,y+dy
        return (x,y) if not in_bounds(nx,ny) else (nx,ny)
    T = np.zeros((A,S,S), dtype=float)
    R = np.full((A,S), step_reward, dtype=float)
    terminal_ids = {idx[xy]:r for xy,r in terminals.items() if xy in idx}
    for s,(x,y) in enumerate(coords):
        if s in terminal_ids:
            for a in range(A):
                T[a,s,s] = 1.0
                R[a,s] = terminal_ids[s]
            continue
        for a in range(A):
            s_main = idx[step(x,y,a)]
            s_left = idx[step(x,y,left[a])]
            s_right= idx[step(x,y,right[a])]
            T[a,s,s_main] += (1.0 - slip_prob)
            T[a,s,s_left] += slip_prob/2.0
            T[a,s,s_right]+= slip_prob/2.0
    return T, R, gamma, coords
