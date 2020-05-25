
from Env_UAV import ENV
import numpy as np
from ddpg_target.ddpg_keras import DDPG as DDPG_target


def train(flag_train = True, flag_display = False):
# def train(flag_train = False, flag_display = True):
    MAX_EPISODES = 1000
    MAX_EP_STEPS = 200
    env = ENV(0, flag_display) 
    model_target = DDPG_target(env.n_action, 8)
    if(not flag_train):
        model_target.load_weights()
    

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
            act_values, action = model_target.step(state[0], flag_train)
            n_state, reward, done = env.swarm.swarm_step([action])
            if flag_display:
                env.pg_update()
                flag_running = env.pg_event()
                if not flag_running:
                    return
            if(flag_train):
                model_target.remember(state[0], act_values, reward, done, n_state[0])
                if model_target.buffer.size() >= model_target.MEMORY_CAPACITY:
                    model_target.learn()
            state = n_state
            ep_reward += reward
            step = j
            if done in ["loser", "winner"]:
                break

        if(episode%100 ==0 and episode!=0):
            model_target.save_weights()
        print('Episode = %d, done = %s, ep_Reward = %.2f, step = %d, explore_rate = %.2f'% (episode, done, ep_reward, step, model_target.epsilon))

    if(flag_train):
        model_target.save_weights()
        print("Training is completed...")


if __name__ == "__main__":
    train()