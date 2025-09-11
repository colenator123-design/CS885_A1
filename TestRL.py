# -*- coding: utf-8 -*-
import numpy as np
from RL import RLProblem, make_grid_maze

if __name__ == "__main__":
    mdp, s_of, coords = make_grid_maze(width=2, height=2, terminal_dict={}, walls=set(), slip_prob=0.0, step_reward=-0.01)
    rl = RLProblem(mdp=mdp, alpha=0.5, gamma=0.95, epsilon=0.2)
    Q0 = np.zeros((mdp.nStates, mdp.nActions))
    Q, pi = rl.q_learning(s0=0, initialQ=Q0, nEpisodes=50, nSteps=50)
    print("Q 形狀 =", Q.shape)
    print("policy =", pi)
