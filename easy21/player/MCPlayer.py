#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:29:51 2017

@author: m0r00ds
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Player import Player

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname, abspath
    path.append(dirname(abspath(__file__)) + "/..")
    from dealer.Dealer import Dealer

class MCPlayer(Player):
    def __init__(self):
        self.action_state = np.zeros((10, 10, 2))
        self.times_state_action_visited = np.zeros((10, 10, 2))
        self.state_visited = []
        self.action_taken = []
        self.N0 = 100.0
    
    def learn(self):
        for i in range(100):
            self._start_episode()
            reward = self._perform_episode()
            self._update_action_state(reward)
            self._end_episode()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.action_state[:,0], self.action_state[:,1], 
                self.action_state[:,2])
        plt.show()
    
    def _start_episode(self):
        self.player_total = self._get_first_card()
        self.dealer = Dealer()
        self.dealer_showing = self.dealer.dealer_total
    
    def _perform_episode(self):
        reward = None
        while (reward == None):
            state_idx = self._get_state_idx()
            action = self._get_action_idx(state_idx)
            reward = self._perform_action(action)
        return reward
    
    def _update_action_state(self, reward):
        dealer_idx = np.ones_like(self.state_visited) * self.dealer_showing - 1
        idx = self.state_visited, dealer_idx, self.action_taken
        error = reward - self.action_state[idx]
        step_size = 1.0 / self.times_state_action_visited[idx]
        print ("step_size = ", step_size)
        print ("times_state_action_visited = ", self.times_state_action_visited[idx])
        self.action_state[idx] += error * step_size
    
    def _end_episode(self):
        self.state_visited = []
        self.action_taken = []
    
    def _get_state_idx(self):
        while (self.player_total < 12):
            self._hit()
        player_idx = self.player_total - 12
        self.state_visited.append(player_idx)
        state_idx = player_idx, self.dealer_showing - 1
        return state_idx
    
    # assuming 0 to stick and 1 to be hit index
    def _get_action_idx(self, state_idx):
        max_arg = np.argmax(self.action_state[state_idx])
        prob = np.random.random()
        N_st = np.sum(self.times_state_action_visited, axis=2)
        self.eps = self.N0 / (self.N0 + N_st)
        if prob > 1.0 - self.eps[state_idx] / len(self.action_state[state_idx]):
            action = max_arg
        else:
            action = self._choose_random_action(max_arg, state_idx)
        self.action_taken.append(action)
        idx = state_idx[0], state_idx[1], action
        self.times_state_action_visited[idx] += 1
        return action
    
    def _perform_action(self, action_idx):
        if (action_idx == 0):
            return self._stick()
        return self._hit()
    
    def _choose_random_action(self, max_arg, state_idx):
        nactions = len(self.action_state[state_idx])
        action = np.random.randint(0, nactions)
        while (action == int(max_arg)):
            action = np.random.randint(0, nactions)
        return action
    
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