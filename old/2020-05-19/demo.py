
from Env_UAV import ENV
import numpy as np
from DQL_keras import DQN_keras

file_para = "uav_para.HDF5"
# def train(flag_display = True, flag_train = False):
def train(flag_display = False, flag_train = True):
    env = ENV(flag_display)
    state_size = [8, (env.swarm.uavs[0].radius_obs*2, env.swarm.uavs[0].radius_obs*2, 1)]
    action_size = env.n_actions
    agent = DQN_keras(state_size, action_size)
    # agent.load(file_para)

    for episode in range(5000):
        # initial observation
        #reset env
        while True:
            env._init_env()
            state = env.swarm.init_swarm()
            env.swarm.uavs[0]._collision_detection()
            if True not in env.swarm.uavs[0].flag_collision.values():
                break
        reward_eps = 0.0
        step_size = 0
        while step_size < 400:
            action = agent.act(state, flag_train)
            next_state, reward, done = env.swarm.swarm_step([action])
            env.update_screen()
            if flag_display:
                env.pg_update()
                flag_running = env.pg_event()
                if not flag_running:
                    return
            if done and step_size == 0: 
                break
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            reward_eps = reward_eps + reward
            # break while loop when end of this episode
            if done: 
                break
            step_size = step_size + 1
        if flag_train:
            loss = agent.replay_minibatch(episode = 1)
            agent.exploration_rate_decay()  #exploration_rate_decay
            agent.update_model_target() #update model_target para
            print("episode = %d, step = %d, reward_eps = %.2f, exploration_rate = %.2f, loss = %.4f" % (episode, step_size, reward_eps, agent.exploration_rate, loss))
        if(episode%100 ==0 and episode!=0):
            agent.save(file_para)
    agent.save(file_para)
    print("Training is completed...")

if __name__ == "__main__":
    train()
    #display()