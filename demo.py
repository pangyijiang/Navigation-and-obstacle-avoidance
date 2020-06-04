
from Env_UAV import ENV
import numpy as np
from ddpg_obstacle.ddpg_keras import DDPG as DDPG_obstacle


# def train(flag_train = False, flag_display = False):
def train(flag_train_obstacle = True, flag_display = False):
    MAX_EPISODES = 5000
    MAX_EP_STEPS = 100
    env = ENV(15, flag_display) 
    model_obstacle = DDPG_obstacle(3, (128,128,3))
    model_obstacle.load_weights("model_obstacle")
    
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
            act_values_obstacle, action = model_obstacle.step(state[1], flag_train_obstacle)
            n_state, reward, done, flag = env.swarm.swarm_step([action])

            if flag_display:
                env.pg_update()
                flag_running = env.pg_event()
                if not flag_running:
                    return

            model_obstacle.remember(state[1], act_values_obstacle, reward, done, n_state[1])
            if model_obstacle.buffer.size() >= model_obstacle.MEMORY_CAPACITY:
                model_obstacle.learn()
            state = n_state
            ep_reward += reward
            step = j
            if done:
                break

        if(episode%100 ==0 and episode!=0):
            model_obstacle.save_weights("model_obstacle")

        explore_rate = ("explore_rate_obs = %.2f" % (model_obstacle.epsilon))
        print('Episode = %d, done = %s, ep_Reward = %.2f, step = %d, %s'% (episode, flag, ep_reward, step, explore_rate))
    
    model_obstacle.save_weights("model_obstacle")
    print("Training is completed...")

if __name__ == "__main__":
    train()