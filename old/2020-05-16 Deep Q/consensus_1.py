from Env_UAV import ENV
import numpy as np

if __name__ == "__main__":

	env = ENV()
	flag_running = True
	while flag_running:
		flag_running = env.pg_event()
		state, reward, done = env.swarm.swarm_step([np.array([1,0,0,0])])
		env.update_screen()
		env.pg_update()
	env.quit()