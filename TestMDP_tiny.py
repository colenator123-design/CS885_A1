# -*- coding: utf-8 -*-
import numpy as np
import MDP

def tiny_mdp():
    # 3 狀態、2 動作（參考講義風格）
    S, A = 3, 2
    T = np.zeros((A,S,S))
    R = np.zeros((A,S))
    # s=0
    T[0,0,0]=0.7; T[0,0,1]=0.3; R[0,0]=0.0
    T[1,0,2]=1.0; R[1,0]=1.0
    # s=1
    T[0,1,2]=1.0; R[0,1]=0.5
    T[1,1,0]=1.0; R[1,1]=0.0
    # s=2 終點
    T[:,2,2]=1.0; R[:,2]=0.0
    gamma = 0.95
    return T,R,gamma

if __name__ == "__main__":
    T,R,gamma = tiny_mdp()
    mdp = MDP.MDP(T,R,gamma)

    print("=== Value Iteration ===")
    V,it,eps = mdp.valueIteration(initialV=np.zeros(mdp.nStates), tolerance=1e-6)
    print("iters:", it, "eps:", eps, "\nV*:", V)
    print("pi*:", mdp.extractPolicy(V))

    print("\n=== Policy Iteration ===")
    pi,V_pi,it_pi = mdp.policyIteration(np.zeros(mdp.nStates, dtype=int))
    print("iters:", it_pi, "\nVπ:", V_pi, "\nπ:", pi)

    print("\n=== MPI (k=3) ===")
    pi2,V2,it2,tol = mdp.modifiedPolicyIteration(np.zeros(mdp.nStates, dtype=int), np.zeros(mdp.nStates), k=3)
    print("iters:", it2, "tol:", tol, "\nV~:", V2, "\nπ~:", pi2)
