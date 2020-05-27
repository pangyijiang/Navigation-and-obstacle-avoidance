
from Env_UAV import ENV
import numpy as np
from DDPG_TF import DDPG

file_para = "mymodel"


"""
1. modify the env , a independent thread. hold the speed with 1 second, beyond 1 second without command, decrease the speed.
"""

def train(flag_train = True, flag_display = False):
    MAX_EPISODES = 1000
    MAX_EP_STEPS = 200
    env = ENV(flag_display)

    agent = DDPG(env.n_action, env.n_state_target)
    if(not flag_train):
        agent.Load_para(file_para)

    for episode in range(MAX_EPISODES):
        #reset env
        while True:
            env._init_env()
            state = env.swarm.init_swarm()
            env.swarm.uavs[0]._collision_detection()
            if True not in env.swarm.uavs[0].flag_collision.values():
                break
        ep_reward = 0.0
        step  = 0
        for j in range(MAX_EP_STEPS):
            act_values, action = agent.step(state[0], flag_train)
            n_state, reward, done = env.swarm.swarm_step([action])
            env.update_screen()
            if flag_display:
                env.pg_update()
                flag_running = env.pg_event()
                if not flag_running:
                    return
            if(flag_train):
                agent.remember(state[0], act_values, reward, n_state[0])
                if agent.pointer > agent.MEMORY_CAPACITY:
                    agent.learn()
            state = n_state
            ep_reward += reward
            step = j
            if done: 
                break

        if(episode%100 ==0 and episode!=0):
            agent.Save_para(file_para)
        print('Episode = %d, ep_Reward = %.2f, step = %d, explore_rate = %.2f'% (episode, ep_reward, step, agent.epsilon))

    if(flag_train):
        agent.Save_para(file_para)
        print("Training is completed...")


if __name__ == "__main__":
    train()