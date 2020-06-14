import pygame as pg
import numpy as np
import random
from PIL import Image
import math
import copy
import time

class SWARM():
    uavs = []
    r_comm = 200 #communication radius
    flag_show_comm = False   #show comm radius
    flag_show_data = False   #show data
    robot_size = 10
    
    def __init__(self, map, num_uavs = 5):
        self.map = map
        self.num_uavs = num_uavs
        self.init_swarm()

    def init_swarm(self):
        # Initialise swarm
        self.uavs = []
        state = []
        self.trans = Transmitter(self.r_comm)
        pose_set,vel_set = self._init_pos_vel(self.num_uavs)
        for i in range(self.num_uavs):
            name = "robot_%d" % i
            self.uavs.append(ROBOT(self.map, self.trans, name, self.robot_size, self.map.gold.pos, pose_set[i], vel_set[i]))
            state.append(self.uavs[i].state_cal())
        return state[0]

    def _init_pos_vel(self, num_uavs):
        while True:
            pose_set = [np.array([np.random.randint(0, self.map.MAP_SIZE[0]), np.random.randint(0, self.map.MAP_SIZE[1])]) for i in range(num_uavs)]
            colllision_dectector = []
            for robot_pose in pose_set:
                for obs_pos, obs_size in zip(self.map.obstacles.pos, self.map.obstacles.size):
                    r = np.sqrt(np.sum(np.square(robot_pose - obs_pos)))
                    if r < self.robot_size + obs_size:
                        colllision_dectector.append(1)
                r = np.sqrt(np.sum(np.square(robot_pose - self.map.gold.pos)))
                if r < self.robot_size + self.map.gold.size:
                    colllision_dectector.append(1)
            if 1 not in colllision_dectector:
                break
        vel_set = [np.array([np.random.randint(0, 70), np.random.randint(0, 70)]) for i in range(num_uavs)]
        return pose_set,vel_set

    def swarm_step(self, actions):
        #actions: [np.ones(4),...]
        assert len(actions) == len(self.uavs)
        #step 1
        for uav,action in zip(self.uavs, actions):
            uav.step_1(action)
        #update necessary graph
        self.map.update_screen_1()
        #step 2
        for uav in self.uavs:
            state, reward, done, flag, robot_pose, radar_end = uav.step_2()
        
        #update necessary graph
        self.map.update_screen_2(robot_pose, radar_end)

        return state, reward, done, flag

class ROBOT():
    info_pkg_s = {}
    info_pkg_r = {}
    comm_delay = 0.5    #delay time of communication
    robot_color  = (0, 0, 255)  #blue
    flag_collision = {"uav":True, "obstacle":False, "gold":False}
    radius_obs = 64
    #physical state
    time_clk = 0.05
    damping = 0.25
    mass = 1.0
    vel_max = 50    #linear velocity
    p_force_gain = 200
    radar_degree  = [ i*np.pi/18 for i in [-6, -4.5, -3, -1.5, 0, 1.5, 3, 4.5, 6]]
    radar_dis = np.zeros(9)
    

    def __init__(self, map, transmitter, robot_name, robot_size, robot_goal, robot_pose = np.zeros(2), robot_vel = np.zeros(2)):
        #calculation
        self.map = map
        self.transmitter = transmitter
        self.robot_pose = robot_pose
        self.robot_pose_prv = robot_pose 
        self.robot_vel = robot_vel
        self.robot_name = robot_name
        self.robot_goal = robot_goal
        self.robot_size = robot_size
        self.robot_degree = np.arctan2(robot_vel[1], robot_vel[0])
        self.robot_obs_rect = [ np.zeros(2) for i in range(4)]
        # self.radar_dis_threshold  = np.array([ abs(robot_size*3/2/np.sin(d)/self.radius_obs) for d in self.radar_degree])
        self.radar_dis_threshold  = np.array([ abs(robot_size*3/2/np.sin(d)) for d in self.radar_degree])
        self.radar_dis_threshold[int(len(self.radar_dis_threshold)/2)] = self.radius_obs
        radar_end = np.array([self.robot_pose for i in range(9)])

    def step_1(self, action):
        #exchange info with neighbors
        self._exchange_info()
        #take action
        self._status_update(action)
        self._collision_detection()

    def step_2(self):
        #cal "state, reward, done"
        state = self.state_cal()
        reward, done, flag = self.reward_cal()
        return state, reward, done, flag, self.robot_pose, self.radar_end

    def reward_cal(self):
        reward = 0.0
        reward_t = 0.0
        reward_obs = 0.0
        done = False
        flag = "TryOut"
        #distance with target 
        # r = np.sqrt(np.sum(np.square(self.robot_goal - self.robot_pose)))
        # reward += -r/(np.sum(self.map.MAP_SIZE)/2*1.414)*5.0
        r = np.sqrt(np.sum(np.square(self.robot_goal - self.robot_pose)))
        r_p = np.sqrt(np.sum(np.square(self.robot_goal - self.robot_pose_prv)))
        reward += -(r - r_p)/5.0
        for th, dis in zip(self.radar_dis_threshold, self.radar_dis):
            if dis < th:
                 reward += -(th - dis)/(th - self.robot_size)*1.0
                 if dis < self.robot_size*2:
                      reward += - 2.0
        # reward = reward_obs if reward_obs < 0.0 else reward_t
        if self.flag_collision["uav"] | self.flag_collision["obstacle"]:
            reward += -2.0
            flag = "loser"
        if self.flag_collision["gold"]:
            reward += 2.0
            flag = "winner"
        if True in self.flag_collision.values():
            done = True
        return reward, done, flag

    def state_cal(self):
        map_mid = np.array(self.map.MAP_SIZE)/2
        # nei_region = self._obs()
        radar = self._radar()/self.radius_obs
        # obs_pos = (np.array(self.map.obstacles.pos).flatten() - 250)/250
        # obs_size = (np.array(self.map.obstacles.size).flatten() - 15)/15
        #position
        dir = (self.robot_goal - self.robot_pose)
        dir = dir/np.sqrt(np.sum(np.square(dir)))  #scale
        pos_robot = (self.robot_pose - map_mid)/map_mid   #scale
        pos_goal = (self.robot_goal - map_mid)/map_mid    #scale
        vel = self.robot_vel/self.vel_max   ##scale
        # pos_related = np.hstack((dir, pos_goal, pos_robot, vel))
        # state = [pos_related, nei_region]
        state = np.hstack((dir, pos_goal, pos_robot, vel, radar))
        return state

    def _status_update(self, action):
        assert action in [i for i in range(self.map.n_action)]
        degree_f = np.pi/4*action
        p_force = np.array([np.cos(degree_f), np.sin(degree_f)])   
        self._robot_clk(p_force*self.p_force_gain)
        
    def _robot_clk(self, p_force):
        # integrate physical state
        self.robot_vel = self.robot_vel * (1 - self.damping)
        self.robot_vel += (p_force/ self.mass) * self.time_clk
        speed = np.sqrt(np.square(self.robot_vel[0]) + np.square(self.robot_vel[1]))
        if speed > self.vel_max:
            self.robot_vel = self.robot_vel / np.sqrt(np.square(self.robot_vel[0]) + np.square(self.robot_vel[1])) * self.vel_max

        self.robot_degree = np.arctan2(self.robot_vel[1], self.robot_vel[0])  #update UAV heading direction
        self.robot_pose_prv = self.robot_pose
        self.robot_pose = self.robot_pose + self.robot_vel*self.time_clk

    def _radar(self):
        
        data = pg.image.tostring(self.map.screen, 'RGB')
        screen = Image.frombytes('RGB', self.map.MAP_SIZE, data)
        screen = screen.convert('L')
        screen = np.array(screen)
        # screen = screen[:, :, np.newaxis]
        # screen = (screen - 127.0)/255.0 #scale
        
        self.radar_dis = np.zeros(9)
        self.radar_end = np.array([self.robot_pose for i in range(9)])    #np.zeros((9,2))
        for j, degree in enumerate(self.radar_degree):
            for i in range(self.radius_obs):
                x = int(self.robot_pose[0] + np.cos(self.robot_degree + degree)*i)
                y = int(self.robot_pose[1] + np.sin(self.robot_degree + degree)*i)
                if y in range(0,screen.shape[0]) and x in range(0,screen.shape[1]):
                    if screen[y][x] != np.average(self.map.obstacles.color):
                        self.radar_dis[j] += 1
                        self.radar_end[j][0] = x
                        self.radar_end[j][1] = y
                    else:
                        break
                else:
                    break
        # return self.radar_dis.flatten()/self.radius_obs
        return self.radar_dis.flatten()
            
    def _obs(self):
        data = pg.image.tostring(self.map.screen, 'RGB')
        screen = Image.frombytes('RGB', self.map.MAP_SIZE, data)
        screen = screen.convert('L')
        screen = np.array(screen)
        screen = screen[:, :, np.newaxis]
        screen = (screen - 127.0)/255.0 #scale
        
        r_half = int(self.radius_obs/2)
        obs_img = np.zeros([self.radius_obs, self.radius_obs, 1])
        for i in range(self.radius_obs):
            c_x = self.robot_pose[0] + np.cos(self.robot_degree)*i
            c_y = self.robot_pose[1] + np.sin(self.robot_degree)*i
            for j in range(r_half):
                point_1_x = int( c_x + np.cos(np.pi/2 + self.robot_degree)*j )
                point_1_y = int( c_y + np.sin(np.pi/2 + self.robot_degree)*j )
                point_2_x = int( c_x + np.cos(self.robot_degree - np.pi/2)*j )
                point_2_y = int( c_y + np.sin(self.robot_degree - np.pi/2)*j )

                if point_2_y in range(0,screen.shape[0]) and point_2_x in range(0,screen.shape[1]):
                    obs_img[r_half - j][i][:] = screen[point_2_y][point_2_x][:]
                else:
                    obs_img[r_half - j][i][:] = 0.0
                if point_1_y in range(0,screen.shape[0]) and point_1_x in range(0,screen.shape[1]):
                    obs_img[r_half + j][i][:] = screen[point_1_y][point_1_x][:]
                else:
                    obs_img[r_half + j][i][:] = 0.0

            if i == 0 and j == r_half-1:
                self.robot_obs_rect[0] = np.array([point_1_x, point_1_y])
                self.robot_obs_rect[1] = np.array([point_2_x, point_2_y])
            if i == self.radius_obs-1 and j == r_half-1:
                self.robot_obs_rect[2] = np.array([point_2_x, point_2_y])
                self.robot_obs_rect[3] = np.array([point_1_x, point_1_y])
                
        #test obs
        # im=Image.fromarray(obs_img)
        # im = im.convert("RGB").tobytes("raw", 'RGB')
        # im = pg.image.fromstring(im, (self.radius_obs, self.radius_obs), "RGB")
        # self.map.screen.blit(im, (0,0))
        # flag = False
        # if flag:
        #     import matplotlib.pyplot as plt
        #     fig = plt.figure()
        #     subplt0 = fig.add_subplot(121)
        #     subplt0.imshow(obs_img)
        #     subplt1 = fig.add_subplot(122)
        #     subplt1.imshow(screen)
        #     plt.show()
        obs_img = obs_img.flatten()
        return obs_img


    def _collision_detection(self):
        self.flag_collision = {"uav":False, "obstacle":False, "gold":False}

        for uav in self.info_pkg_r:
            r = np.sqrt(np.sum(np.square(self.robot_pose - uav["pose"])))
            if r <= self.robot_size*2:
                self.flag_collision["uav"] = True
                break
        for obs_pos, obs_size in zip(self.map.obstacles.pos, self.map.obstacles.size):
            r = np.sqrt(np.sum(np.square(self.robot_pose - obs_pos)))
            if r <= (self.robot_size + obs_size):
                self.flag_collision["obstacle"] = True
                break
        #for gold_pos in self.map.gold.pos:
        r = np.sqrt(np.sum(np.square(self.robot_pose - self.map.gold.pos)))
        if r <= (self.robot_size + self.map.gold.size):
            self.flag_collision["gold"] = True

        if self.flag_collision["uav"] | self.flag_collision["obstacle"]:
            self.robot_color = (255, 0, 0)
        else:
            self.robot_color = (0, 0, 255)


    def _exchange_info(self):
        #send info to neighbors, recv info of neighbors
        self.info_pkg_s = {"name":self.robot_name, "pose":self.robot_pose, \
                    "vel":self.robot_vel}
        self.transmitter.Send(self.info_pkg_s)
        self.info_pkg_r = self.transmitter.Recv(self.robot_name)
        #limit_linear_vel

    def _limit_linear_vel(self, vel, max_velocity):
        vel_linear = np.sqrt(np.sum(np.square(vel)))
        if(vel_linear > max_velocity):
            vel = vel*max_velocity/vel_linear
        return vel


class Transmitter:
    def __init__(self, radius = 10.0):
        # manager = Manager()
        self.database = {}
        self.communication_radius = radius
    def Send(self, info_pkg):
        name = info_pkg["name"]
        self.database[name] = info_pkg
    def Recv(self, name):
        queue_d = []
        neibs_info_pkg = []
        query = self.database[name]
        neib_names = list(self.database)    #avoid multi process changing dic in the same time
        for neib_name in neib_names:
            neib_info_pkg = self.database[neib_name]
            if(neib_info_pkg["name"] != query["name"]):
                d = np.sqrt(np.sum(np.square(neib_info_pkg["pose"][0:2] - query["pose"][0:2])))
                if (d <= self.communication_radius):
                    neibs_info_pkg.append(neib_info_pkg)
                    queue_d.append(d)

        neibs_info_pkg = self._sort_queue(neibs_info_pkg, queue_d)
        return neibs_info_pkg
    def _sort_queue(self, neibs_info_pkg, queue_d):
        neibs_info_pkg_sort = copy.copy(neibs_info_pkg)
        queue_d_sort = copy.copy(queue_d)
        queue_d_sort.sort()
        for i in range(len(queue_d)):
            neibs_info_pkg_sort[i] = neibs_info_pkg[queue_d.index(queue_d_sort[i])]
        return neibs_info_pkg_sort