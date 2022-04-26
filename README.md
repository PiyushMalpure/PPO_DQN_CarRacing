# PPO_DQN_CarRacing

CarRacing is a gym environment available in the OpenAI gym library. This environment has 96x96x3 pixel observation space and 3 actions: discrete steering (-1 for left steer and +1 for right steer), Acceleration and Breaking.
This project worked with this environment to compare the effect of discrete, and continuous action spaces on the effectiveness of reinforcement learning. This project implemented two reinforcement learning algorithms: Deep Q Network Algorithm  (DQN) and Continuous Proximal Policy Optimization (PPO) Reinforcement Learning (RL) Algorithm. 
It was found that PPO algorithm performed better than DQN algorithm in playing the Car Racing game by a huge margin. PPO based agent got an average reward of 217.38 over 10 episode while DQN based agent got an average of 92.

# Environment setup

#### Gym installation
pip install gym[box2d]
#### Install swig
pip install swig
conda install -c anaconda swig

#### Install box2d
pip install -e ".box2d" 

# DQN


# PPO
To train PPO file

cd PPO
python3 PPO.py

To test the trained file with the pretrained weights

python3 test_car.py
