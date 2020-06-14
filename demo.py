
from Env_UAV import ENV
import numpy as np
from ddpg_keras import DDPG

def train(flag_train = False, flag_display = True):
# def train(flag_train = True, flag_display = False):
    MAX_EPISODES = 4000
    MAX_EP_STEPS = 200
    env = ENV(15, flag_display)

    agent = DDPG(env.n_action, env.n_state)
    # if(not flag_train):
    agent.load_weights()

    for episode in range(MAX_EPISODES):
        #reset env
        while True:
            env._init_env()
            state = env.swarm.init_swarm()
            env.swarm.uavs[0]._collision_detection()
            if True not in env.swarm.uavs[0].flag_collision.values():
                break
        ep_reward = 0.0
        flag = "None"
        step  = 0
        for j in range(MAX_EP_STEPS):
            act_values, action = agent.step(state, flag_train)
            n_state, reward, done, flag = env.swarm.swarm_step([action])

            if flag_display:
                env.pg_update()
                flag_running = env.pg_event()
                if not flag_running:
                    return
            if(flag_train):
                agent.remember(state, act_values, reward, done, n_state)
                if agent.buffer.size() >= agent.MEMORY_CAPACITY:
                    agent.learn()
            state = n_state
            ep_reward += reward
            step = j
            if done: 
                break

        if(episode%100 ==0 and episode!=0):
            agent.save_weights()
        print('Episode = %d, flag = %s, ep_Reward = %.2f, step = %d, explore_rate = %.2f'% (episode, flag, ep_reward, step, agent.epsilon))

    if(flag_train):
        agent.save_weights()
        print("Training is completed...")


if __name__ == "__main__":
    train()