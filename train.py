import os
import shutil
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from rl import ppo
from rl.networks import network_utils
from arguments import get_args
from rl.networks.envs import make_vec_envs
from rl.networks.model import Policy
from rl.networks.storage import RolloutStorage


from crowd_nav.configs.config import Config
from crowd_sim import *


def main():
	"""
	main function for training a robot policy network
	"""
	# read arguments
	algo_args = get_args()

	# create a directory for saving the logs and weights
	if not os.path.exists(algo_args.output_dir):
		os.makedirs(algo_args.output_dir)
	# if output_dir exists and overwrite = False
	elif not algo_args.overwrite:
		raise ValueError('output_dir already exists!')

	save_config_dir = os.path.join(algo_args.output_dir, 'configs')
	if not os.path.exists(save_config_dir):
		os.makedirs(save_config_dir)
	shutil.copy('crowd_nav/configs/config.py', save_config_dir)
	shutil.copy('crowd_nav/configs/__init__.py', save_config_dir)
	shutil.copy('arguments.py', algo_args.output_dir)


	env_config = config = Config()

	torch.manual_seed(algo_args.seed)
	torch.cuda.manual_seed_all(algo_args.seed)
	if algo_args.cuda:
		if algo_args.cuda_deterministic:
			# reproducible but slower
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			# not reproducible but faster
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False



	torch.set_num_threads(algo_args.num_threads)
	device = torch.device("cuda" if algo_args.cuda else "cpu")


	env_name = algo_args.env_name

	if config.sim.render:
		algo_args.num_processes = 1
		algo_args.num_mini_batch = 1

	# for visualization
	if config.sim.render:
		fig, ax = plt.subplots(figsize=(7, 7))
		ax.set_xlim(-10, 10)
		ax.set_ylim(-10, 10)
		ax.set_xlabel('x(m)', fontsize=16)
		ax.set_ylabel('y(m)', fontsize=16)
		plt.ion()
		plt.show()
	else:
		ax = None


	# Create a wrapped, monitored VecEnv
	envs = make_vec_envs(env_name, algo_args.seed, algo_args.num_processes,
						 algo_args.gamma, None, device, False, config=env_config, ax=ax, pretext_wrapper=config.env.use_wrapper)


	# create a policy network
	actor_critic = Policy(
		envs.observation_space.spaces, # pass the Dict into policy to parse
		envs.action_space,
		base_kwargs=algo_args,
		base=config.robot.policy)

	# storage buffer to store the agent's experience
	rollouts = RolloutStorage(algo_args.num_steps,
							  algo_args.num_processes,
							  envs.observation_space.spaces,
							  envs.action_space,
							  algo_args.human_node_rnn_size,
							  algo_args.human_human_edge_rnn_size)

	# continue training from an existing model if resume = True
	if algo_args.resume:
		load_path = config.training.load_path
		actor_critic.load_state_dict(torch.load(load_path))
		print("Loaded the following checkpoint:", load_path)


	# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
	nn.DataParallel(actor_critic).to(device)

	# create the ppo optimizer
	agent = ppo.PPO(
		actor_critic,
		algo_args.clip_param,
		algo_args.ppo_epoch,
		algo_args.num_mini_batch,
		algo_args.value_loss_coef,
		algo_args.entropy_coef,
		lr=algo_args.lr,
		eps=algo_args.eps,
		max_grad_norm=algo_args.max_grad_norm)



	obs = envs.reset()
	if isinstance(obs, dict):
		for key in obs:
			rollouts.obs[key][0].copy_(obs[key])
	else:
		rollouts.obs[0].copy_(obs)

	rollouts.to(device)

	episode_rewards = deque(maxlen=100)

	start = time.time()
	num_updates = int(
		algo_args.num_env_steps) // algo_args.num_steps // algo_args.num_processes

	# start the training loop
	for j in range(num_updates):
		# schedule learning rate if needed
		if algo_args.use_linear_lr_decay:
			network_utils.update_linear_schedule(
				agent.optimizer, j, num_updates,
				agent.optimizer.lr if algo_args.algo == "acktr" else algo_args.lr)

		# step the environment for a few times
		for step in range(algo_args.num_steps):
			# Sample actions
			with torch.no_grad():

				rollouts_obs = {}
				for key in rollouts.obs:
					rollouts_obs[key] = rollouts.obs[key][step]
				rollouts_hidden_s = {}
				for key in rollouts.recurrent_hidden_states:
					rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][step]
				value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
					rollouts_obs, rollouts_hidden_s,
					rollouts.masks[step])

			# if we use real prediction, send predictions to env for rendering
			if env_name == 'CrowdSimPredRealGST-v0' and env_config.env.use_wrapper:
				# [nenv, max_human_num, 2*(pred_steps+1)] -> [nenv, max_human_num, 2*pred_steps]
				out_pred = rollouts_obs['spatial_edges'][:, :, 2:].to('cpu').numpy()
				# send manager action to all processes
				ack = envs.talk2Env(out_pred)
				assert all(ack)

			if config.sim.render:
				envs.render()
			# Obser reward and next obs
			obs, reward, done, infos = envs.step(action)


			for info in infos:
				if 'episode' in info.keys():
					episode_rewards.append(info['episode']['r'])

			# If done then clean the history of observations.
			masks = torch.FloatTensor(
				[[0.0] if done_ else [1.0] for done_ in done])
			bad_masks = torch.FloatTensor(
				[[0.0] if 'bad_transition' in info.keys() else [1.0]
				 for info in infos])
			rollouts.insert(obs, recurrent_hidden_states, action,
							action_log_prob, value, reward, masks, bad_masks)
		# store the stepped experience to buffer
		with torch.no_grad():
			rollouts_obs = {}
			for key in rollouts.obs:
				rollouts_obs[key] = rollouts.obs[key][-1]
			rollouts_hidden_s = {}
			for key in rollouts.recurrent_hidden_states:
				rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][-1]
			next_value = actor_critic.get_value(
				rollouts_obs, rollouts_hidden_s,
				rollouts.masks[-1]).detach()

		# compute advantage and gradient, and update the network parameters
		rollouts.compute_returns(next_value, algo_args.use_gae, algo_args.gamma,
								 algo_args.gae_lambda, algo_args.use_proper_time_limits)

		value_loss, action_loss, dist_entropy = agent.update(rollouts)

		rollouts.after_update()

		# save the model for every interval-th episode or for the last epoch
		if (j % algo_args.save_interval == 0
			or j == num_updates - 1) :
			save_path = os.path.join(algo_args.output_dir, 'checkpoints')
			if not os.path.exists(save_path):
				os.mkdir(save_path)

			torch.save(actor_critic.state_dict(), os.path.join(save_path, '%.5i'%j + ".pt"))

		if j % algo_args.log_interval == 0 and len(episode_rewards) > 1:
			total_num_steps = (j + 1) * algo_args.num_processes * algo_args.num_steps
			end = time.time()
			print(
				"Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward "
				"{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
					.format(j, total_num_steps,
							int(total_num_steps / (end - start)),
							len(episode_rewards), np.mean(episode_rewards),
							np.median(episode_rewards), np.min(episode_rewards),
							np.max(episode_rewards), dist_entropy, value_loss,
							action_loss))

			df = pd.DataFrame({'misc/nupdates': [j], 'misc/total_timesteps': [total_num_steps],
							   'fps': int(total_num_steps / (end - start)), 'eprewmean': [np.mean(episode_rewards)],
							   'loss/policy_entropy': dist_entropy, 'loss/policy_loss': action_loss,
							   'loss/value_loss': value_loss})

			if os.path.exists(os.path.join(algo_args.output_dir, 'progress.csv')) and j > 20:
				df.to_csv(os.path.join(algo_args.output_dir, 'progress.csv'), mode='a', header=False, index=False)
			else:
				df.to_csv(os.path.join(algo_args.output_dir, 'progress.csv'), mode='w', header=True, index=False)

	envs.close()


if __name__ == '__main__':
	main()
