#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:59:34 2017

@author: m0r00ds
"""
import numpy as np
from sys import path
from os.path import dirname as dir


class Dealer:
    def __init__(self):
        self.dealer_total = np.random.randint(1, 11)
    
    def play_move(self):
        while self.dealer_total < 17 :
            self.dealer_total += self.hit()
    
    def hit(self):
        if (np.random.random() < 1.0 / 3.0):
            return -np.random.randint(1, 11)
        else:
            return np.random.randint(1, 11)

if __name__ == '__main__':
    path.append(dir(path[0]))
    __package__ = "examples"