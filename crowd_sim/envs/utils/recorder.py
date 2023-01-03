import pandas as pd
import os
import numpy as np

class Recoder(object):
	def __init__(self):

		self.unsmoothed_actions = []
		self.actionList = []  # recorded in turtlebot_env.step. [trans,rot]
		self.wheelVelList = []  # recorded in apply_action(). [left,right]
		self.orientationList = []  # recorded in calc_state() [angle]
		self.positionList = []  # recorded in calc_state() [x,y]
		self.robot_goal = []

		self.saveTo='data/unicycle/selfAttn_truePred_noiseActt/record/real_recordings/'
		# self.loadFrom = 'data/unicycle/selfAttn_truePred_noiseActt/record/sim_recordings/ep0-1/action.csv'
		self.loadFrom = 'data/unicycle/selfAttn_truePred_noiseActt/record/sim_recordings/ep0-1point5/unsmoothed_action.csv'
		self.loadedAction=None
		self.episodeInitNum=0


	def saveEpisode(self,episodeCounter):

		savePath=os.path.join(self.saveTo,'ep'+str(self.episodeInitNum+episodeCounter))
		#savePath = os.path.join(self.saveTo, 'ep' + str(4))
		if not os.path.exists(savePath):
			#shutil.rmtree(savePath)
			os.makedirs(savePath)

		if len(self.actionList) > 0:
			action = pd.DataFrame({'trans': np.array(self.actionList)[:,0], 'rot': np.array(self.actionList)[:,1]})
			action.to_csv(os.path.join(savePath, 'action.csv'), index=False)
		if len(self.wheelVelList) > 0:
			wheelVel=pd.DataFrame({'left': np.array(self.wheelVelList)[:,0], 'right': np.array(self.wheelVelList)[:,1]})
			wheelVel.to_csv(os.path.join(savePath, 'wheelVel.csv'), index=False)
		if len(self.orientationList) > 0:
			orientation = pd.DataFrame({'ori': np.array(self.orientationList)})
			orientation.to_csv(os.path.join(savePath, 'orientation.csv'), index=False)
		if len(self.positionList) > 0:
			position = pd.DataFrame({'x': np.array(self.positionList)[:, 0], 'y': np.array(self.positionList)[:, 1]})
			position.to_csv(os.path.join(savePath, 'position.csv'), index=False)
		if len(self.unsmoothed_actions) > 0:
			unsmooth_action = pd.DataFrame({'trans': np.array(self.unsmoothed_actions)[:,0], 'rot': np.array(self.unsmoothed_actions)[:,1]})
			unsmooth_action.to_csv(os.path.join(savePath, 'unsmoothed_action.csv'), index=False)
		if len(self.robot_goal) > 0:
			robot_goal = pd.DataFrame({'x': np.array(self.robot_goal)[:, 0], 'y': np.array(self.robot_goal)[:, 1]})
			robot_goal.to_csv(os.path.join(savePath, 'robot_goal.csv'), index=False)

		print("csv written")
		self.clear()

	def loadActions(self):
		self.loadedAction = pd.read_csv(self.loadFrom)
		self.v_list = self.loadedAction['trans']
		self.delta_theta_list = self.loadedAction['rot']
		print("Reading actions from", self.loadFrom)

	def clear(self):
		self.actionList = []  # recorded in turtlebot_env.step. [trans,rot]
		self.wheelVelList = []  # recorded in apply_action(). [left,right]
		self.orientationList = []  # recorded in calc_state() [angle]
		self.positionList = []  # recorded in calc_state() [x,y]
		self.unsmoothed_actions = []
		self.robot_goal = []

class humanRecoder(object):
	def __init__(self, human_num):
		self.human_num = human_num
		# list of [px, py, radius, v_pref] with length = human_num
		self.initPosList = [[] for i in range(self.human_num)]
		# list of goal lists with length = human_num
		# i-th element is a list i-th human's goals in the episode
		self.goalPosList = [[] for i in range(self.human_num)]

	# update self.initPosList
	# human_id: 0~4
	# pos: list of [px, py, radius, v_pref]
	def addInitPos(self, human_id, px, py, radius, v_pref):
		self.initPosList[human_id] = [px, py, radius, v_pref]

	# append [px, py] at the end of i-th element of list
	def addGoalPos(self, human_id, px, py):
		self.goalPosList[human_id].append([px, py])

	def loadLists(self, initPos, goalPos):
		self.initPosList = initPos
		self.goalPosList = goalPos

	def getInitList(self):
		return self.initPosList

	def getGoalList(self):
		return self.goalPosList

	def getInitPos(self, human_id):
		return self.initPosList[human_id]

	# get next goal position for human i
	# if the last goal is already gotten before, return None
	def getNextGoalPos(self, human_id):
		# if the list is empty
		if not self.goalPosList[human_id]:
			return None
		return self.goalPosList[human_id].pop(0)

	# check whether human i has used up its goal
	def goalIsEmpty(self, human_id):
		if not self.goalPosList[human_id]:
			return True
		else:
			return False

	def clear(self):
		self.initPosList = [[] for i in range(self.human_num)]
		self.goalPosList = [[] for i in range(self.human_num)]

class jointStateRecoder(object):
	def __init__(self, human_num):
		self.human_num = human_num

		# px, py, r, gx, gy, v_pref, theta, vx, vy
		self.robot_s = []

		# nested list of [px1, ..., px5]
		self.humans_px = []
		# nested list of [py1, ..., py5]
		self.humans_py = []

		# record the random seed for this episode
		self.episode_num = []

		# data labels:
		self.episode_num_label = []
		self.traj_ratio = []
		self.time_ratio = []
		self.idle_time = []


	# append a joint state to the lists
	# ob should be the ob returned from env.step()
	def add_traj(self, seed, ob, srnn):
		# if ob is a dictionary from crowd_sim_dict.py
		if srnn:
			robot_s_noV = ob['robot_node'][0, 0].cpu().numpy()
			robot_v = ob['edges'][0, 0].cpu().numpy()
			self.robot_s.append(np.concatenate((robot_s_noV, robot_v)))
			self.humans_px.append(ob['edges'][0, 1:, 0].cpu().numpy())
			self.humans_py.append(ob['edges'][0, 1:, 1].cpu().numpy())
		# if ob is a list from crowd_sim.py
		else:
			ob = ob[0].cpu().numpy()
			# px, py, r, gx, gy, v_pref, theta, vx, vy
			self.robot_s.append(np.concatenate((ob[1:3], ob[5:10], ob[3:5])))
			px_list, py_list = [], []
			# human num = (ob.shape - 10) // 5
			for i in range((ob.shape[0] - 10) // 5):
				px_idx = 5 * i + 10
				py_idx = 5 * i + 11
				px_list.append(ob[px_idx])
				py_list.append(ob[py_idx])
			self.humans_px.append(px_list)
			self.humans_py.append(py_list)
		self.episode_num.append(seed)

	# seed: the random seed of this episode from env
	# respond_traj_len, respond_time: the total traj length and total timesteps when the user makes last choice
	# traj_len, tot_time: total traj length and timesteps of the whole episode
	def add_label(self, seed, respond_traj_len, traj_len, respond_time, tot_time, idle_time):
		self.traj_ratio.append(respond_traj_len/traj_len)
		self.time_ratio.append(respond_time/tot_time)
		self.idle_time.append(idle_time)
		self.episode_num_label.append(seed)



	def save_to_file(self, directory):
		if not os.path.exists(os.path.join('trajectories', directory)):
			os.makedirs(os.path.join('trajectories', directory))
		# 1. save all trajectories
		# add robot data
		robot_s = np.array(self.robot_s)
		data_dict = {'epi_num': np.array(self.episode_num), 'robot_px': robot_s[:, 0], 'robot_py': robot_s[:, 1],
					 'robot_r': robot_s[:, 2],
					 'robot_gx': robot_s[:, 3], 'robot_gy': robot_s[:, 4], 'robot_vpref': robot_s[:, 5],
					 'robot_theta': robot_s[:, 6], 'robot_vx': robot_s[:, 7], 'robot_vy': robot_s[:, 8]}
		# add humans data
		humans_px = np.array(self.humans_px)
		humans_py = np.array(self.humans_py)
		for i in range(self.human_num):
			data_dict['human'+str(i)+'px'] = humans_px[:, i]
			data_dict['human'+str(i)+'py'] = humans_py[:, i]

		all_data = pd.DataFrame(data_dict)
		all_data.to_csv(os.path.join('trajectories', directory, 'states.csv'), index=False)

		# 2. save all labels
		data_dict = {'epi_num': np.array(self.episode_num_label), 'traj_ratio': np.array(self.traj_ratio),
					 'time_ratio': np.array(self.time_ratio), 'idle_time': np.array(self.idle_time)}
		all_data = pd.DataFrame(data_dict)
		all_data.to_csv(os.path.join('trajectories', directory, 'labels.csv'), index=False)

		self.clear()

	def clear(self):
		# px, py, r, gx, gy, v_pref, theta, vx, vy
		self.robot_s = []

		# nested list of [px1, ..., px5]
		self.humans_px = []
		# nested list of [py1, ..., py5]
		self.humans_py = []

		# record the random seed for this episode
		self.episode_num = []

		# data labels:
		self.episode_num_label = []
		self.traj_ratio = []
		self.time_ratio = []
		self.idle_time = []

