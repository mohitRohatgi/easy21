#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:29:51 2017

@author: m0r00ds
"""

import numpy as np
from Player import Player
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname, abspath
    path.append(dirname(abspath(__file__)) + "/..")
    from dealer.Dealer import Dealer

class MCPlayer(Player):
    def __init__(self):
        self.player_total = self._get_first_card()
        self.dealer = Dealer()
        self.action_state = np.zeros((21, 2))
        self.times_state_visited = np.zeros(21)
        self.times_state_action_visited = np.zeros((21, 2))
        self.N0 = 100.0
        self.step_size = self._get_step_size()
    
    def _get_first_card(self):
        return np.random.randint(1, 11)
    
    def _hit(self):
        self.player_total += self.dealer.hit()
        if self.player_total > 21.0:
            return -1.0
        return None
    
    def _get_eps_m(self):
        return self.N0 / (self.N0 + self.times_state_visited) / len(self.action_state[0])
    
    def _get_action_idx(self, state_idx):
        return int(np.random.random() / self.eps_m)
    
    def _get_step_size(self):
        return 1.0 / self.times_state_visited
    
    def _stick(self):
        self.dealer.play_move()
        if self.dealer.dealer_total > 21.0:
            return 1.0
        return self._get_reward()
    
    def _get_reward(self):
        if self.dealer.dealer_total > self.player_total:
            return -1.0
        elif np.abs(self.dealer.dealer_total - self.player_total) < 1e-6:
            return 0.0
        else:
            return 1.0
    
    def learn(self):
        pass
    
    def _perform_episode(self):
        self.times_state_visited[self.player_total] += 1
        self.eps_m = self._get_eps_m()
        action = self._get_action_idx(self.player_total)
    