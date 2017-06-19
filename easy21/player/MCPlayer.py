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
        self.action_state = np.zeros((10, 2))
        self.times_state_visited = np.zeros(10)
        self.state_visited = []
        self.action_taken = []
        self.N0 = 100.0
    
    def learn(self):
        for i in range(10000):
            self._start_episode()
            reward = self._perform_episode()
            self._update_action_state(reward)
            self._end_episode()
        print (self.action_state)
    
    def _start_episode(self):
        self.player_total = self._get_first_card()
        self.dealer = Dealer()
        self.dealer_showing = self.dealer.dealer_total
    
    def _perform_episode(self):
        self.eps = self._get_eps()
        reward = None
        while (reward == None):
            state_idx = self._get_state_idx()
            action = self._get_action_idx(state_idx)
            reward = self._perform_action(action)
        return reward
    
    def _update_action_state(self, reward):
        error = self._get_error(reward)
        step_size = self._get_step_size()
        self.action_state[self.state_visited, self.action_taken] += error * step_size
    
    def _end_episode(self):
        self.state_visited = []
        self.action_taken = []
    
    def _get_error(self, reward):
        return reward - self.action_state[self.state_visited, self.action_taken]
    
    def _get_eps(self):
        return self.N0 / (self.N0 + self.times_state_visited)
    
    def _get_state_idx(self):
        while (self.player_total < 12):
            self._hit()
        state_idx = self.player_total - 12
        self.state_visited.append(state_idx)
        self.times_state_visited[state_idx] += 1
        return state_idx
    
    # assuming 0 to stick and 1 to be hit index
    def _get_action_idx(self, state_idx):
        max_arg = np.argmax(self.action_state[state_idx])
        prob = np.random.random()
        if prob > 1.0 - self.eps[state_idx] / len(self.action_state[state_idx]):
            action = max_arg
        else:
            action = self._choose_random_action(max_arg)
        self.action_taken.append(action)
        return action
    
    def _perform_action(self, action_idx):
        if (action_idx == 0):
            return self._stick()
        return self._hit()
    
    def _choose_random_action(self, max_arg):
        action = np.random.randint(0, len(self.action_state[0]))
        while (action == int(max_arg)):
            action = np.random.randint(0, len(self.action_state[0]))
        return action
    
    def _get_step_size(self):
        return 1.0 / self.times_state_visited[self.state_visited]
    
    def _get_first_card(self):
        return np.random.randint(1, 11)
    
    def _hit(self):
        self.player_total += self.dealer.hit()
        if self.player_total > 21.0:
            return -1.0
        return None
    
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

if __name__ == '__main__':
    MCPlayer().learn()