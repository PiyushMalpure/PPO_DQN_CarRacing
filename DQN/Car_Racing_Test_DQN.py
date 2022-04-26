#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from queue import PriorityQueue
import random
import gym
from collections import deque
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
import torchvision

import gym
import numpy as np
from numpy import moveaxis
from torch.autograd import Variable,grad

from collections import deque,namedtuple

import copy
import PIL
from PIL import Image,ImageOps

from torch import grid_sampler_3d
import time
import os




def convertImagetotensor(current_state):
    state_image = PIL.Image.fromarray(current_state)
    state_image = PIL.ImageOps.grayscale(state_image) 
    state_image = state_image.crop((6, 12, 90, 96)) 
    state_image = PIL.ImageOps.equalize(state_image, mask=None)
    width, height = state_image.size
    to_tensor = torchvision.transforms.ToTensor()
    state_tensor = to_tensor(state_image).unsqueeze(0)
    state_tensor.requires_grad = True
    
    return state_tensor

class DQN(nn.Module):
    """Initialize a deep Q-learning network
    """

    def __init__(self):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(4, 4),stride=4)
        self.conv2 = nn.Conv2d(6, 24, kernel_size=(4, 4),stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = torch.nn.Linear(9*9*24, 1000)
        self.fc2 = torch.nn.Linear(1000, 256)
        self.fc3 = torch.nn.Linear(256, 4)


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(self.pool1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def make_action(dqn,observation):
    """
    Return predicted action of your agent
    Input:
        observation: np.array
            stack 4 last preprocessed frames, shape: (84, 84, 4)
    Return:
        action: int
            the predicted action from trained model
    """
    ###########################
    # YOUR IMPLEMENTATION HERE 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_value = dqn.forward(observation).to(device)
    action  = torch.argmax(q_value)
    action = int(action.item())

                
    ###########################
    return action
    
def test():
    ############## Hyperparameters ##############
    env_name = "CarRacing-v0"
    env = gym.make(env_name)
    
    n_episodes = 20        # num of episodes to run
    max_timesteps = 1500    # max timesteps in one episode
    render = True          # render the environment
    save_gif = False     # png images are saved in gif folder
    possible_actions = np.array([[0.0, 1.0, 0.0], [1.0, 0.3, 0], [-1.0, 0.3, 0.0], [0.0, 0.0, 0.8]])
    parent_dir = '.\gif'
    
    # filename and directory to load model from
    # replace it with your own trained model
    filename = '.\CarRacing_DQN.pth'
    
    # Loading Model
    checkpoint = torch.load(filename)

    #############################################
    

    dqn = DQN()
    dqn.load_state_dict(checkpoint)
    
    
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        state = convertImagetotensor(state)
        
        reward_each_episode = 0
        time_frame_counter = 0
        negative_reward_counter = 0
        current_episode_frame = 0
        done = False
        
        while not done:
            current_episode_frame+=1
              
            action = make_action(dqn,state)
            action = possible_actions[action]
            if render:
                
                env.render()
              

            
            next_state,reward,done,_ = env.step(action)
            

            next_state = convertImagetotensor(next_state)
      
            reward_each_episode+=reward
            
            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0
            time_frame_counter+=1
              
              
            state = next_state
            
            
            
            if save_gif:
                if current_episode_frame==1:
                    directory = 'Run{}'.format(ep)
                    path = os.path.join(parent_dir, directory)
                    os.mkdir(path)
                    
                img = env.render(mode='rgb_array')
                img = Image.fromarray(img)
                img.save('./gif/Run{}/{}.jpg'.format(ep,current_episode_frame))
            
            if current_episode_frame>1500 or negative_reward_counter>25 or reward_each_episode<0:
                current_episode_frame = 0
                break
            
            
        print("Episode Number : ",ep, "  Current Score : ", reward_each_episode,"  Avergae Score : ",reward_each_episode,"  Epsilon : ", 0)
        
    env.close()
        

    
if __name__ == '__main__':
    test()
    