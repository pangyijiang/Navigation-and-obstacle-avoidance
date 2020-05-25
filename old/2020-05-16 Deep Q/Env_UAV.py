import pygame as pg
import numpy as np
import random
from PIL import Image
import math
import copy
import time
from Swarm import SWARM


class ENV():
    #pg related para
    screen_frequency = 60
    color_bg = [255,255,255]
    MAP_SIZE = [1000, 600]
    n_actions = 4
    
    def __init__(self, flag_display = True):
        self._init_pgscreen(flag_display)
        self._init_env()
        self.swarm = SWARM(self, num_uavs = 1)

    def _init_env(self):
        class Obstacles():
            def __init__(self, map_size, num = 5):
                self.num = num
                self.size = 30
                self.color = (211, 211, 211)    #light grey
                self.pos = [np.array([np.random.randint(0, map_size[0]), np.random.randint(0, map_size[1])]) for i in range(num)]
        
        class Gold():
            def __init__(self, map_size):
                self.size = 20
                self.color = (105, 105, 105)    #dim grey
                self.pos = np.array([np.random.randint(0, map_size[0]), np.random.randint(0, map_size[1])])
        while True:
            self.obstacles = Obstacles(self.MAP_SIZE, 15)
            self.gold = Gold(self.MAP_SIZE)
            r = []
            for obs_pos in self.obstacles.pos:
                r.append(np.sqrt(np.sum(np.square(obs_pos - self.gold.pos))))
            if min(r) > (self.gold.size + self.obstacles.size):
                break

    def _init_pgscreen(self, flag_display):
        # Initialise screen
        pg.init()
        if flag_display:
            self.screen = pg.display.set_mode(self.MAP_SIZE)
        else:
            self.screen = pg.Surface(self.MAP_SIZE)
        pg.display.set_caption("Swarm Control Algorithm")
        self.clock = pg.time.Clock()

    def update_screen(self):
        self.screen.fill(self.color_bg)
        self._show_env()
        self._show_robot()
    
    def _draw_uav(self, pose, size, degree, color = (0, 0, 255)):
        pg.draw.circle(self.screen, color, pose.astype(int), size, 1)
        dy = np.sin(degree)*size
        dx = np.cos(degree)*size
        pose = pose + np.array([dx, dy])
        pg.draw.circle(self.screen, color, pose.astype(int), int(size/3), 0)

    def _show_env(self):
        for pos in self.obstacles.pos:
            pg.draw.circle(self.screen, self.obstacles.color, pos.astype(int), self.obstacles.size, 0)
        #for pos in self.gold.pos:
        pg.draw.circle(self.screen, self.gold.color, self.gold.pos.astype(int), self.gold.size, 0)

    def _show_robot(self):
        #self.swarm..uav_group.draw(self.screen)
        for uav in self.swarm.uavs:
            #show communication radius
            if self.swarm.flag_show_comm:
                pg.draw.circle(self.screen, (255, 0, 0), uav.robot_pose.astype(int), self.r_comm, 1)
            #draw robot
            self._draw_uav(uav.robot_pose, uav.radius_size, np.arctan2(uav.robot_vel[1], uav.robot_vel[0]), uav.robot_color)

    def pg_update(self):
        pg.display.flip()
        #pg.display.update()
        self.clock.tick(self.screen_frequency)
        
    def pg_event(self):
        flag_running = True
        events =  pg.event.get()

        for event in events:
            if event.type == pg.QUIT:
                flag_running = False
            #mouse event
            # elif event.type == pg.MOUSEMOTION:
            #     print("[MOUSEMOTION]:", event.pos, event.rel, event.buttons)
            # elif event.type == pg.MOUSEBUTTONUP:
            #     print("[MOUSEBUTTONUP]:", event.pos, event.button)
            # elif event.type == pg.MOUSEBUTTONDOWN:
            #     print("[MOUSEBUTTONDOWN]:", event.pos, event.button)
            #keyboard event
            # elif event.type == pg.KEYDOWN:
            #     if event.key == pg.K_e:
            #         data = pg.image.tostring(self.screen, 'RGB')
            #         img = Image.frombytes('RGB', self.MAP_SIZE, data)
            #         img = img.convert('L')
            #         # img.save("111.png")
            #         pix = np.array(img)

        return flag_running

    def quit(self):
        pg.quit()