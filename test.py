import numpy as np

for action in range(8):

    degree_f = np.pi/4*action
            # degree_f = self.degree + np.pi/4*action
    p_force = np.array([np.cos(degree_f), np.sin(degree_f)])

    print(p_force)