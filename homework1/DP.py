#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Yifan WANG <yifanwang1993@gmail.com>
#
# Distributed under terms of the MIT license.

import numpy as np

"""

"""


class MDP(object):
    def __init__(self, proba, reward, gamma):
        self.proba = proba
        self.reward = reward
        self.gamma = gamma

        self.ns = self.proba.shape[0]
        self.na = self.proba.shape[1]
        self.states = range(self.ns)
        self.actions = range(self.na)
        self.V = np.zeros(self.ns)
        self.Q = np.zeros((self.ns, self.na))


    def proba(self, s0,  a,  s1):
        return self.proba[s0, a, s1]

    def proba(self, s0, a):
        return self.proba[s0, a, :]

    def reward(self, s0, a):
        return self.reward[s0, a]

    def value_iteration(self, get_values=False):
        V = self.V.copy()
        turn = 0
        if get_value
        while np.abs(V - self.V).max() > 0.01 or turn == 0:
            V = self.V.copy()
            turn += 1
            for s in self.states:
                for a in self.actions:
                    self.Q[s, a] = self.reward[s, a] + self.gamma * self.proba[s, a, :].dot(V)
                self.V[s] = np.max(self.Q[s, :])

    def policy_evaluation(self, policy):
        
        pass






reward = np.zeros((3, 2))
reward[0,0] = -0.4
reward[0,1] = 0
reward[1, 0] = 2
reward[1, 1] = 0
reward[2, 0] = -1
reward[2, 1] = -0.5

proba = np.zeros((3,2,3))
proba[0,0,0] = 0.45
proba[0,0,2] = 0.55
proba[0,1,2] = 1
proba[1,0,2] = 1
proba[1,1,0] = 0.5
proba[1,1,1] = 0.4
proba[1,1,2] = 0.1
proba[2,0,0] = 0.6
proba[2,0,1] = 0.4
proba[2,1,1] = 0.9
proba[2,1,2] = 0.1

mdp = MDP(proba, reward, gamma=0.95)
mdp.value_iteration()
print mdp.V


