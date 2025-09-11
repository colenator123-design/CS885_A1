# -*- coding: utf-8 -*-
"""
MDP.py — 作業骨架相容版
- 建構：MDP(T, R, discount)，T:[A,S,S']，R:[A,S]
- 方法：valueIteration / extractPolicy / evaluatePolicy / policyIteration /
        evaluatePolicyPartially / modifiedPolicyIteration
"""
from __future__ import annotations
import numpy as np

class MDP:
    def __init__(self, T: np.ndarray, R_as: np.ndarray, discount: float):
        assert T.ndim == 3
        assert R_as.ndim == 2
        A, S, S2 = T.shape
        assert S == S2, "T 維度不符"
        assert R_as.shape == (A, S), "R 應為 [A,S]"
        assert 0.0 <= discount < 1.0
        self.T = T.astype(float)
        self.R = R_as.astype(float)
        self.discount = float(discount)
        self.nActions = A
        self.nStates  = S

    def _Q_from_V(self, V):
        return self.R + self.discount * np.einsum('ass,s->as', self.T, V)

    def valueIteration(self, initialV=None, tolerance: float = 1e-2, maxIterations: int = 10000):
        V = np.zeros(self.nStates) if initialV is None else initialV.astype(float).copy()
        it = 0
        eps = np.inf
        while it < maxIterations:
            Q = self._Q_from_V(V)       # [A,S]
            V_new = Q.max(axis=0)       # [S]
            eps = float(np.max(np.abs(V_new - V)))
            V = V_new
            it += 1
            if eps < tolerance:
                break
        return [V, it, eps]

    def extractPolicy(self, V):
        Q = self._Q_from_V(V)
        return Q.argmax(axis=0).astype(int)

    def evaluatePolicy(self, policy, tolerance: float = 1e-10, maxIterations: int = 100000):
        V = np.zeros(self.nStates, dtype=float)
        for _ in range(maxIterations):
            V_new = np.empty_like(V)
            for s in range(self.nStates):
                a = int(policy[s])
                V_new[s] = self.R[a, s] + self.discount * (self.T[a, s] @ V)
            if np.max(np.abs(V_new - V)) < tolerance:
                V = V_new
                break
            V = V_new
        return V

    def policyIteration(self, init_policy, tolerance: float = 1e-10, maxIterations: int = 1000):
        policy = init_policy.astype(int).copy()
        it = 0
        while it < maxIterations:
            V = self.evaluatePolicy(policy, tolerance=tolerance)
            new_policy = self.extractPolicy(V)
            it += 1
            if np.all(new_policy == policy):
                policy = new_policy
                break
            policy = new_policy
        V = self.evaluatePolicy(policy, tolerance=tolerance)
        return [policy, V, it]

    def evaluatePolicyPartially(self, policy, V0, k: int = 1):
        V = V0.astype(float).copy()
        eps = None
        for _ in range(k):
            V_new = np.empty_like(V)
            for s in range(self.nStates):
                a = int(policy[s])
                V_new[s] = self.R[a, s] + self.discount * (self.T[a, s] @ V)
            eps = float(np.max(np.abs(V_new - V)))
            V = V_new
        return [V, k, 0.0 if eps is None else eps]

    def modifiedPolicyIteration(self, init_policy, V0, k: int = 1, tolerance: float = 1e-10, maxIterations: int = 1000):
        policy = init_policy.astype(int).copy()
        V = V0.astype(float).copy()
        it = 0
        tol_last = np.inf
        while it < maxIterations:
            V, _, tol_last = self.evaluatePolicyPartially(policy, V, k=k)
            new_policy = self.extractPolicy(V)
            it += 1
            if np.all(new_policy == policy):
                policy = new_policy
                break
            policy = new_policy
        V = self.evaluatePolicy(policy, tolerance=tolerance)
        Q = self._Q_from_V(V)
        V_next = Q.max(axis=0)
        tol_last = float(np.max(np.abs(V_next - V)))
        return [policy, V, it, tol_last]
