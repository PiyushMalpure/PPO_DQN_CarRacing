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
import torchvision
import matplotlib.pyplot as plt

import gym
import numpy as np
from numpy import moveaxis
from torch.autograd import Variable,grad
import os,csv
from torch.utils.tensorboard import SummaryWriter

from collections import deque,namedtuple

import copy
import PIL
from PIL import Image

from torch import grid_sampler_3d
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def state_image_preprocess(state_image):
    state_image = state_image.transpose((2,0,1))
    state_image = np.ascontiguousarray(state_image, dtype=np.float32) / 255
    state_image = torch.from_numpy(state_image)
    return state_image.unsqueeze(0).to(device)

def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state

def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
    return np.transpose(frame_stack, (1, 2, 0))

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

from torch._C import dtype
class Agent_DQN():
  def __init__(self, env):
      """
      Initialize everything you need here.
      For example: 
          paramters for neural network  
          initialize Q net and target Q net
          parameters for repaly buffer
          parameters for q-learning; decaying epsilon-greedy
          ...
      """

      ###########################
      # YOUR IMPLEMENTATION HERE #
      
      self.CUDA_available = torch.cuda.is_available()
      # self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if self.CUDA_available else autograd.Variable(*args, **kwargs)
      self.Variable = Variable
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.env = env
      self.obs = env.reset()
      self.obs = convertImagetotensor(self.obs)
      self.possible_actions = np.array([[0.0, 1.0, 0.0], [1.0, 0.3, 0], [-1.0, 0.3, 0.0], [0.0, 0.0, 0.8]])
      self.actionSpace = [0, 1, 2 , 3]
      
      self.channel_input = self.obs.shape
      self.num_actions = len(self.actionSpace)
      
      
      # Model Parameters
      self.current_model = DQN()
      self.target_model = DQN()

      self.target_model.load_state_dict(self.current_model.state_dict())
      self.target_model.eval()
      
      # if self.CUDA_available:
      #     self.current_model = self.current_model.cuda()
      #     self.target_model = self.target_model.cuda()
      
      # Hyper Parameters
      self.epsilon_start = 1
      self.epsilon_goal = 0.05
      self.train_episodes = 300 # 300000
      self.gamma = 0.95
      self.learning_rate = 0.0001
      self.learn_start = 50000 # 5000
      self.training_freq = 4
      self.update_target_model_num = 10000
      self.batch_size = 64
      self.max_len_buffer = 100000
      self.stabilizer = 0.01
      self.gradient_momentum = 0.95
      
      
      
      self.optimizer = optim.RMSprop(self.current_model.parameters(), lr = self.learning_rate, alpha = self.gradient_momentum, 
                  eps = self.stabilizer)
      
      
      self.current_model.train()
      self.target_model.eval()
      
      self.loss_criterion = nn.SmoothL1Loss()


      
      # Action Counter
      self.step = 0
      self.epsilon_step = 30000
      self.max_time_step = 1500
      
      self.epsilon_index = lambda index: self.epsilon_start - ((self.epsilon_start-self.epsilon_goal)/(self.episilon_linear))*index
      
      
      self.current_train =0
      self.current = 0
      self.current_target =0
      self.current_episode_frame =0
      
      self.best_mean = 0
      
      
      self.buffer = deque(maxlen=self.max_len_buffer)
      self.reward_list = []
      self.reward_average = []

      self.current_time = time.time()
      self.time_base = self.current_time
      self.index = 0
      self.writer = SummaryWriter(comment = 'dqn')
      self.load = True


      # Load File
      if self.load:
        with open('datafile.csv', mode ='r')as file: 
          temp = csv.reader(file)
          for i in temp:
            self.epsilon = float(i[0])
            self.index = 225
            checkpoint = torch.load('./DQN/CarRacing-v1_{}.pth'.format(self.index))
            self.current_model.load_state_dict(checkpoint)
            print("-------------------Loading Trained Model-------------------")
            self.current_model.eval()
            
            

      
 

      
      
  def make_action(self, observation, test=False):
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
      # YOUR IMPLEMENTATION HERE #
      
  
      
      random_num = np.random.uniform(0,1)
      
    
      
      if random_num<self.epsilon:
          action = random.randrange(self.num_actions)
      else:
          q_value = self.current_model.forward(observation).to(self.device)
          action  = torch.argmax(q_value)
          action = int(action.item())
  
                    
      ###########################
      return action
  
  def optimize(self):
      
      
      state, action, reward, next_state, done = self.replay_buffer()
      target_q_values = []
      current_q_values = []

  


      for i in range(self.batch_size):
        temp_target = self.target_model(state[i])
        temp_current = self.current_model(state[i])

        target_max = max(temp_target)
        current_max = max(temp_current)

        target_q_values.append(target_max)
        current_q_values.append(current_max)

      target_q_values = torch.stack(target_q_values).cuda()
      current_q_values = torch.stack(current_q_values).cuda()
      done = torch.FloatTensor(done).cuda()
      reward = torch.FloatTensor(reward).cuda()


      target_q_values = target_q_values * self.gamma * (1-done)

      loss = self.loss_criterion(current_q_values, reward + target_q_values)
      self.optimizer.zero_grad()
      loss.backward()
      for param in self.current_model.parameters():
        param.grad.data.clamp_(-1, 1)
      self.optimizer.step()

      return loss.cpu().detach().numpy()
      

  
  
  def push(self,state,action,reward,next_state,done):
      """ You can add additional arguments as you need. 
      Push new data to buffer and remove the old one if the buffer is full.
      
      Hints:
      -----
          you can consider deque(maxlen = 10000) list
      """
      ###########################
      # YOUR IMPLEMENTATION HERE #
      self.buffer.append((state,action,reward,next_state,done))
      
      ###########################
      
  def update_target_model(self):
      self.target_model.load_state_dict(self.current_model.state_dict())    
      
      
      
  def replay_buffer(self):
      """ Select batch from buffer.
      Input:
          None:
              buffer from class is used
      Output:
          state:
              current state
          action:
              action taken
          reward:
              reward for action from current state
          next state:
              next state due to current state,action
          done:
              check termination
              
      
      """
      ###########################
      # YOUR IMPLEMENTATION HERE #
      
      batch = random.sample(self.buffer,self.batch_size)
      obs, action, reward, next_state, done = [], [], [], [], []
      
      for num in batch:
          obs_num, action_num, reward_num, next_state_num, done_num = num
          obs.append(obs_num)
          action.append(action_num)
          reward.append(reward_num)
          next_state.append(next_state_num)
          done.append(done_num)
          
      ###########################
      return obs, action, reward, next_state, done
  
  def get_epsilon(self):
      if self.current <= self.learn_start:
          return self.epsilon_start
      
      steps = self.step - self.learn_start
      self.epsilon = self.epsilon_start - steps * (self.epsilon_start - self.epsilon_goal) / self.epsilon_step
      
      self.epsilon = max(self.epsilon,self.epsilon_goal)
      
      
      return self.epsilon
      

  def train(self):
      """
      Implement your training algorithm here
      """
      ###########################
      # YOUR IMPLEMENTATION HERE #
      state = self.env.reset()
      state = convertImagetotensor(state)
      
    
      reward_list = []
      reward_avg = 0
      loss = 0

    
      
      
      # self.calculate_loss()
      index = self.index
      
      while True:
          state = self.env.reset()
          state = convertImagetotensor(state)
          
          done = False
          reward_each_episode = 0
          time_frame_counter = 0
          negative_reward_counter = 0
          
          if self.current>self.learn_start:      
            index+=1
          
          while not done:
              self.step+=1
              
              self.epsilon = self.get_epsilon()
              
              
              action = self.make_action(state)
              action = self.possible_actions[action]
              

            
              next_state,reward,done,_ = self.env.step(action)
            

              next_state = convertImagetotensor(next_state)
      
              reward_each_episode+=reward
                            
              self.push(state,action,reward,next_state,done)

              negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0
              time_frame_counter+=1
              
              
              state = next_state

            
              
              self.current+= 1
              self.current_train += 1
              self.current_target += 1
              self.current_episode_frame +=1
              
              if self.current>self.learn_start and self.current_train%self.training_freq==0:
                loss = self.optimize()
                self.current_train=0
                  
          
              if self.current_target>self.update_target_model_num and self.current>self.learn_start:
                print("---------------------Updating Targel Model---------------------------")
                self.update_target_model()
                self.current_target=0

              if self.current_episode_frame>self.max_time_step or negative_reward_counter>25 or reward_each_episode<0:
                self.current_episode_frame = 0
                break

              self.writer.add_scalar('Loss', loss,self.current)

                      
  
  
          reward_avg = float(np.mean(reward_list[-10:])) 
                            
          reward_list.append(reward_each_episode)

          self.current_time = time.time() - self.time_base


          # Logging        
          if self.current>self.learn_start:
            print("Episode Number : ",index, "  Current Score : ", reward_each_episode,"  Avergae Score : ",reward_avg,"  Epsilon : ", self.epsilon, "Loss : ",loss, "  Current Step : ", self.step, "  Time : {:0.2f}".format(self.current_time))    
            self.time_base = time.time()

            savefile = [self.epsilon,self.current,self.index,reward_avg,reward_each_episode,loss]

            with open('datafile.csv','a') as f:
              writer=csv.writer(f)
              writer.writerow(savefile)

            self.writer.add_scalar("epsilon", self.epsilon, index)
            self.writer.add_scalar("mean_reward", reward_avg, index)
            self.writer.add_scalar("reward", reward_each_episode, index)

          if len(reward_list)>25:
            reward_list = reward_list[-25:]

          

      
                  
          if index%5==0 and self.current>self.learn_start:
              print("------------------------Saving Model---------------------------------")
              torch.save(self.current_model.state_dict(), './DQN/CarRacing-v1_{}.pth'.format(index))
              
              
              
if __name__ == '__main__':
    env_name = "CarRacing-v0"
    environment  = gym.make(env_name)
    AAA = Agent_DQN(env = environment)


    AAA.train()


  
   
                                


