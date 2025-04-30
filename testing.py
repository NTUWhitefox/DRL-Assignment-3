import numpy as np
import gym
import gym_super_mario_bros 
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import torch
from collections import deque
import random
import cv2
import pickle
import time

import student_agent

def convert_to_opencv(frame):
    # Verify frame has 3 dimensions
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must be a 3D array with 3 channels (RGB).")
    # Convert RGB to BGR
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return bgr_frame

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

total_test_reward = 0
for t in range(1):
    done = False
    test_reward = 0
    state = env.reset()
    A = student_agent.Agent()
    A.agent.eval()
    for step in range(5500):
        if done:
            break
        action = A.act(state)
        #if (step+1) % 60 == 0:
        #    action = 0 #try insert noop to unstuck
        if (step+1) % 10000 == 0:
            frame = env.env.render(mode = 'rgb_array')  
            opencv_frame = convert_to_opencv(frame) 
            cv2.imshow("OpenCV Image", opencv_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        state, reward, done, info = env.step(action) 
        test_reward += reward              
    print(f"test : {t} :", test_reward)
    total_test_reward += test_reward
print(total_test_reward / 10)