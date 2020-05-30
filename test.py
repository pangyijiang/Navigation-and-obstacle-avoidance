import numpy as np

action_obstacle = [0 ,1]

for action in action_obstacle:
        degree_f_obstacle = np.pi/4 + np.pi/2*(1 - 2*action)
        p_force = np.array([np.cos(degree_f_obstacle), np.sin(degree_f_obstacle)])
        print(p_force)