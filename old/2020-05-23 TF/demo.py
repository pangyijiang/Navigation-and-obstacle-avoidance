
from Env_UAV import ENV
import numpy as np
from DDPG_TF import DDPG

file_para = "mymodel"

def train(flag_train = True, flag_display = False):
# def train(flag_train = False, flag_display = True):
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
        done = False
        for j in range(MAX_EP_STEPS):
            act_values, action = agent.step(state, flag_train)
            n_state, reward, done = env.swarm.swarm_step([action])
            if flag_display:
                env.pg_update()
                flag_running = env.pg_event()
                if not flag_running:
                    return
            if(flag_train):
                agent.remember(state, act_values, reward, n_state)
                if len(agent.memory) >= agent.MEMORY_CAPACITY:
                    agent.learn()
            state = n_state
            ep_reward += reward
            step = j
            if done in ["loser", "winner"]:
                break

        if(episode%100 ==0 and episode!=0):
            agent.Save_para(file_para)
        print('Episode = %d, done = %s, ep_Reward = %.2f, step = %d, explore_rate = %.2f'% (episode, done, ep_reward, step, agent.epsilon))

    if(flag_train):
        agent.Save_para(file_para)
        print("Training is completed...")


if __name__ == "__main__":
    train()