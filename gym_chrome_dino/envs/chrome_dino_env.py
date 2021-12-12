#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import base64
import io
import numpy as np
import os
from collections import deque
from PIL import Image, ImageOps

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_chrome_dino.game import DinoGame
from gym_chrome_dino.utils.helpers import rgba2rgb

from matplotlib import pyplot as plt

from time import sleep

class ChromeDinoEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, render, accelerate, autoscale):
        self.game = DinoGame(render, accelerate)
        self.image_size = self._observe().shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(80, 80, 4), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(2)
        self.gametime_reward = 1
        self.gameover_penalty = -5
        self.current_frame = self.observation_space.low
        self._action_set = [0, 1]
        self.observation_buffer = []
        self.observations = []
    
    def _observe(self):
        s = self.game.get_canvas()
        b = io.BytesIO(base64.b64decode(s))
        i = Image.open(b).resize((80, 80))
        i = rgba2rgb(i)
        a = np.array(ImageOps.grayscale(i))
        a = a.reshape(1,80,80,1)
        self.current_frame = a
        """
        try:
            self.observations.append(a)
        except:
            self.observations = [a]
        """
        
        try:
            self.observation_buffer.append(a)
        except:
            self.observation_buffer = [a] * 4
        
        if len(self.observation_buffer) > 4:
            self.observation_buffer.pop(0)

        if len(self.observation_buffer) < 4:
            self.observation_buffer = [a] * 4

        arr = np.array(self.observation_buffer).reshape(1,80,80,4)
        return arr
    
    def step(self, action):
        reward = self.gametime_reward
        if action == 0:
            """Do Nothing"""
            sleep(0.1)
        elif action == 1:
            self.game.press_up()
            sleep(0.55)

        next_state = self._observe()
        
        done = False
        info = {}
        if self.game.is_crashed():
            reward = self.gameover_penalty
            done = True

        return next_state, reward, done, info
    
    def reset(self, record=False):
        self.game.restart()
        self.observation_buffer = []
        #self.observations = []
        return self._observe()
    
    def render(self, mode='rgb_array', close=False):
        #assert mode=='rgb_array', 'Only supports rgb_array mode.'
        plt.imshow(self.current_frame.reshape((80,80)), interpolation='nearest', cmap="Greys")
        plt.show()
    
    def close(self):
        self.game.close()
    
    def get_score(self):
        return self.game.get_score()
    
    def set_acceleration(self, enable):
        if enable:
            self.game.restore_parameter('config.ACCELERATION')
        else:
            self.game.set_parameter('config.ACCELERATION', 0)
