
from Env_UAV import ENV
import numpy as np
from ddpg_target.ddpg_keras import DDPG as DDPG_target
from ddpg_obstacle.ddpg_keras import DDPG as DDPG_obstacle


# def train(flag_train = False, flag_display = False):
def train(flag_train_target = False, flag_train_obstacle = False, flag_display = True):
    flag_model = [True , True, False]
    MAX_EPISODES = 5000
    MAX_EP_STEPS = 200
    env = ENV(20, flag_display) 
    model_target = DDPG_target(env.n_action, 8)
    model_obstacle = DDPG_obstacle(8, (128,128,3))
    if flag_model[0]:
        model_target.load_weights("model_target")
    if flag_model[1]:
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
            action = []
            if flag_model[0]:
                act_values_target = model_target.step(state[0], flag_train_target)
                action.append(act_values_target)
            if flag_model[1]:
                act_values_obstacle = model_obstacle.step(state[1], flag_train_obstacle)
                action.append(act_values_obstacle)
            
            n_state, reward, done, flag = env.swarm.swarm_step([action])

            if flag_display:
                env.pg_update()
                flag_running = env.pg_event()
                if not flag_running:
                    return
            if(flag_train_target and flag_model[0]):
                model_target.remember(state[0], act_values_target, reward, done, n_state[0])
                if model_target.buffer.size() >= model_target.MEMORY_CAPACITY:
                    model_target.learn()
            if(flag_train_obstacle and flag_model[1]):
                model_obstacle.remember(state[1], act_values_obstacle, reward, done, n_state[1])
                if model_obstacle.buffer.size() >= model_obstacle.MEMORY_CAPACITY:
                    model_obstacle.learn()
            state = n_state
            ep_reward += reward
            step = j
            if done:
                break

        if(episode%100 ==0 and episode!=0):
            if flag_train_target and flag_model[0]:
                model_target.save_weights("model_target")
            if flag_train_obstacle and flag_model[1]:
                model_obstacle.save_weights("model_obstacle")

        if flag_train_target and flag_train_obstacle:
            explore_rate = ("explore_rate_target = %.2f, explore_rate_obs = %.2f" % (model_target.epsilon, model_obstacle.epsilon))
        elif flag_train_target:
            explore_rate = ("explore_rate_target = %.2f" % (model_target.epsilon))
        elif flag_train_obstacle:
            explore_rate = ("explore_rate_obs = %.2f" % (model_obstacle.epsilon))
        else:
            explore_rate = "None"
        print('Episode = %d, done = %s, ep_Reward = %.2f, step = %d, %s'% (episode, flag, ep_reward, step, explore_rate))
    
    if flag_train_target and flag_model[0]:
        model_target.save_weights("model_target")
    if flag_train_obstacle and flag_model[1]:
        model_obstacle.save_weights("model_obstacle")
    print("Training is completed...")

if __name__ == "__main__":
    train()