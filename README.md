<img src="images/agent.gif" width="400" > <img src="images/agents.gif" width="400" >

# OverView
Navigation and obstacle avoidance for an agent in unknown environments, implementing with DDPG.

# RL
- [Input]: global coordinates of target and robot, velocity of robot, virtual radar information(distance information, in the heading direction of robot with fixed angle)

- [Output]: force in eight direction

- [Reward]: 1. distance with target; 2. radar information compared with threshold distance with obstacles

# Installation Dependencies
* Python3
* pygame
* keras

# How to Run
* Test
```
python demo.py	(Note: def train(flag_train = False, flag_display = True))
```
* Train 
```
python demo.py	(Note: def train(flag_train = True, flag_display = False))
```
