import pygame as pg
import numpy as np
import random
from PIL import Image
import math
import copy
import time

class SWARM():
    uavs = []
    uav_group = pg.sprite.Group()
    r_comm = 200 #communication radius
    flag_show_comm = False   #show comm radius
    flag_show_data = False   #show data
    
    def __init__(self, map, num_uavs = 5):
        self.map = map
        self.num_uavs = num_uavs
        self.init_swarm()

    def init_swarm(self):
        # Initialise swarm
        self.uavs = []
        state = []
        self.uav_group = pg.sprite.Group()
        self.trans = Transmitter(self.r_comm)
        pose_set,vel_set = self._init_pos_vel(self.num_uavs)
        for i in range(self.num_uavs):
            name = "robot_%d" % i
            self.uavs.append(ROBOT(self.map, self.trans, name, self.map.gold.pos, pose_set[i], vel_set[i]))
            state.append(self.uavs[i].state_cal())
            self.uav_group.add(self.uavs[i])
        return state[0]

    def _init_pos_vel(self, num_uavs):
        circle_point = np.arange(0 , 2*math.pi , 2*math.pi/num_uavs)
        r = 30.0
        centre_x = self.map.MAP_SIZE[0]/2
        centre_y = self.map.MAP_SIZE[1]/2
        pose_set = [np.array([math.sin(circle_point[i])*r + centre_x, math.cos(circle_point[i])*r + centre_y]) for i in range(num_uavs)]
        #vel_set = [np.array([math.sin(circle_point[i]), math.cos(circle_point[i])])*np.random.uniform(8.0, 20.0) for i in range(num_uavs)]
        #vel_set = [np.array([np.random.uniform(-50.0, 50.0), np.random.uniform(-50.0, 50.0)]) for i in range(num_uavs)]
        vel_set = [np.array([0.0, 0.0]) for i in range(num_uavs)]
        return pose_set,vel_set

    def swarm_step(self, actions):
        #actions: [np.ones(4),...]
        assert len(actions) == len(self.uavs)
        for uav,action in zip(self.uavs, actions):
            state, reward, done = uav.step(action)

        return state, reward, done

class ROBOT(pg.sprite.Sprite):
    info_pkg_s = {}
    info_pkg_r = {}
    comm_delay = 0.5    #delay time of communication
    action_delay = 0.01 #in case uav.step too fast
    time_current_action = 0.0
    time_last_action = 0.0
    time_end = 0
    robot_color  = (0, 0, 255)  #blue
    acc = 20 #acceleration
    flag_collision = {"uav":True, "obstacle":False, "gold":False}
    radius_obs = 128
    #physical state
    time_clk = 0.1
    damping = 0.25
    mass = 1.0
    vel_max = 100    #linear velocity
    p_force_gain = 200

    def __init__(self, map, transmitter, robot_name, robot_goal, robot_pose = np.zeros(2), robot_vel = np.zeros(2)):
        #calculation
        self.map = map
        self.transmitter = transmitter
        self.robot_pose = robot_pose
        self.robot_pose_prv = robot_pose 
        self.robot_vel = robot_vel
        self.robot_name = robot_name
        self.robot_goal = robot_goal
        #pygame - animation
        pg.sprite.Sprite.__init__(self) # for collision dectection
        #for collision dectection
        self.radius_size = 15   
        self.rect = pg.Rect(0, 0, self.radius_size*2, self.radius_size*2)
        self.rect.center = self.robot_pose

    def step(self, action):
        #exchange info with neighbors
        self._exchange_info()
        #take action
        self._status_update(action)
        self._collision_detection()
        #cal "state, reward, done"
        state = self.state_cal()
        reward, done = self.reward_cal()
        return state, reward, done

    def reward_cal(self):
        reward = 0.0
        #distance with target 
        # r = np.sqrt(np.sum(np.square(self.robot_goal - self.robot_pose))) - np.sqrt(np.sum(np.square(self.robot_goal - self.robot_pose_prv)))
        # reward += -r*0.01
        r = np.sqrt(np.sum(np.square(self.robot_goal - self.robot_pose)))
        reward += -r*0.01
        #out of map
        # if self.robot_pose[0] > self.map.MAP_SIZE[0] or self.robot_pose[0] < 0 or self.robot_pose[1] > self.map.MAP_SIZE[1] or self.robot_pose[1] < 0:
        #     reward += -np.sqrt(np.sum(np.square(self.robot_pose - np.array(self.map.MAP_SIZE)/2)))/(np.sum(self.map.MAP_SIZE)/2)
        # #collision with obstacle
        # if self.flag_collision["uav"] | self.flag_collision["obstacle"]:
        #     reward += -1.0
        # #collision with target
        # if self.flag_collision["gold"]:
        #     reward += 10.0
        #third: done
        if True in self.flag_collision.values():
            done = True
        else:
            done = False
        return reward, done
    def state_cal(self):
        nei_region = self._obs()
        #position
        dir = (self.robot_goal - self.robot_pose)
        dir = dir/np.sqrt(np.sum(np.square(dir)))  #scale
        map_mid = np.array(self.map.MAP_SIZE)/2
        pos_robot = (self.robot_pose - map_mid)/map_mid   #scale
        pos_goal = (self.robot_goal - map_mid)/map_mid    #scale
        vel = self.robot_vel/self.vel_max   ##scale
        pos_related = np.hstack((dir, pos_goal, pos_robot, vel))
        state = [pos_related, nei_region]
        return state

    def _status_update(self, action):
        # p_force = np.array([0.0/4.0, 1.0/4.0, 2.0/4.0, 3.0/4.0, 4.0/4.0, 5.0/4.0, 6.0/4.0, 7.0/4.0])*math.pi
        #action: np.zeros(4) - (up, down, left, right)
        assert action in [i for i in range(self.map.n_action)]
        action =  np.eye(self.map.n_action)[action]
        if action[0] == 1.0:
            p_force = np.array([0.0, 1.0])
        elif action[1] == 1.0:
            p_force = np.array([0.0, -1.0])
        elif action[2] == 1.0:
            p_force = np.array([-1.0, 0.0])
        elif action[3] == 1.0:
            p_force = np.array([1.0, 0.0])
        elif action[4] == 1.0:
            p_force = np.array([0.707, 0.707])
        elif action[5] == 1.0:
            p_force = np.array([-0.707, 0.707])
        elif action[6] == 1.0:
            p_force = np.array([-0.707, -0.707])
        elif action[7] == 1.0:
            p_force = np.array([0.707, -0.707])
        self._robot_clk(p_force*self.p_force_gain)
        
    def _robot_clk(self, p_force):
        # self.time_current_action = time.time()
        # time_clk = self.time_current_action - self.time_last_action
        # if self.time_last_action == 0.0:
        #     self.time_last_action = self.time_current_action
        #     return 
        # integrate physical state
        self.robot_vel = self.robot_vel * (1 - self.damping)
        self.robot_vel += (p_force/ self.mass) * self.time_clk
        speed = np.sqrt(np.square(self.robot_vel[0]) + np.square(self.robot_vel[1]))
        if speed > self.vel_max:
            self.robot_vel = self.robot_vel / np.sqrt(np.square(self.robot_vel[0]) + np.square(self.robot_vel[1])) * self.vel_max

        self.robot_pose_prv = self.robot_pose
        self.robot_pose = self.robot_pose + self.robot_vel*self.time_clk #*self.action_delay
        #update rect for collision dectection
        self.rect.center = self.robot_pose

    def _obs(self):
        data = pg.image.tostring(self.map.screen, 'RGB')
        screen = Image.frombytes('RGB', self.map.MAP_SIZE, data)
        #nei_size = [max(self.robot_pose[0] - self.radius_obs, 0), max(self.robot_pose[1] - self.radius_obs, 0), min(self.robot_pose[0] + self.radius_obs, self.map.MAP_SIZE[0]), min(self.robot_pose[1] + self.radius_obs, self.map.MAP_SIZE[1])]
        nei_size = [self.robot_pose[0] - self.radius_obs, self.robot_pose[1] - self.radius_obs, self.robot_pose[0] + self.radius_obs, self.robot_pose[1] + self.radius_obs]
        nei_region = screen.crop(nei_size)
        # screen_region = nei_region.resize((128,128))
        # screen_region = screen_region.tobytes("raw", 'RGB')
        # screen_region = pg.image.fromstring(screen_region, (128,128), "RGB")
        nei_region = nei_region.convert('L')    #gray image
        # flag = False
        # if flag == True:
        #     nei_region.save("123.png")
        #obs = np.array(obs)
        nei_region = np.array(nei_region)
        nei_region = nei_region[:, :, np.newaxis]
        nei_region = (nei_region - 127.0)/255.0 #scale
        #nei_region = nei_region[np.newaxis, :, :, :]
        return nei_region


    def _collision_detection(self):
        self.flag_collision = {"uav":False, "obstacle":False, "gold":False}

        for uav in self.info_pkg_r:
            r = np.sqrt(np.sum(np.square(self.robot_pose - uav["pose"])))
            if r <= self.radius_size*2:
                self.flag_collision["uav"] = True
                break
        for obstacle_pos in self.map.obstacles.pos:
            r = np.sqrt(np.sum(np.square(self.robot_pose - obstacle_pos)))
            if r <= (self.radius_size + self.map.obstacles.size):
                self.flag_collision["obstacle"] = True
                break
        #for gold_pos in self.map.gold.pos:
        r = np.sqrt(np.sum(np.square(self.robot_pose - self.map.gold.pos)))
        if r <= (self.radius_size + self.map.gold.size):
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