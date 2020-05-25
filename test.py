import numpy as np

x_old = 1
y_old = 1
degree = np.arctan(1/1)
r = 1


point_1_o_x = x_old + np.cos(np.pi/2 - degree)*r
point_1_o_y = y_old - np.sin(np.pi/2 - degree)*r

point_2_o_x = x_old + np.cos(np.pi/2 + degree - np.arctan(2))*r*5**0.5
point_2_o_y = y_old + np.sin(np.pi/2 + degree - np.arctan(2))*r*5**0.5
#new coordinate system 
point_1_x = point_1_o_x*np.cos(-degree) - point_1_o_y*np.sin(-degree)
point_1_y = point_1_o_y*np.cos(-degree) + point_1_o_x*np.sin(-degree)

point_2_x = point_2_o_x*np.cos(-degree) - point_2_o_y*np.sin(-degree)
point_2_y = point_2_o_y*np.cos(-degree) + point_2_o_x*np.sin(-degree)

print(point_1_o_x, point_1_o_y)
print(point_2_o_x, point_2_o_y)