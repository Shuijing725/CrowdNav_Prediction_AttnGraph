import gym
import numpy as np
from numpy.linalg import norm
import pandas as pd
import os

# prevent import error if other code is run in conda env
try:
	# import ROS related packages
	import rospy
	import tf2_ros
	from geometry_msgs.msg import Twist, TransformStamped, PoseArray
	import tf
	from sensor_msgs.msg import JointState
	from threading import Lock
	from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber

except:
	pass

import copy
import sys

from crowd_sim.envs.crowd_sim_pred_real_gst import CrowdSimPredRealGST


class rosTurtlebot2iEnv(CrowdSimPredRealGST):
	'''
	Environment for testing a simulated policy on a real Turtlebot2i
	To use it, change the env_name in arguments.py in the tested model folder to 'rosTurtlebot2iEnv-v0'
	'''
	metadata = {'render.modes': ['human']}

	def __init__(self):
		super(CrowdSimPredRealGST, self).__init__()

		# subscriber callback function will change these two variables
		self.robotMsg=None # robot state message
		self.humanMsg=None # human state message
		self.jointMsg=None # joint state message

		self.currentTime=0.0
		self.lastTime=0.0 # store time for calculating time interval

		self.human_visibility=None
		self.current_human_states = None  # (px,py)
		self.detectedHumanNum=0

		# goal positions will be set manually in self.reset()
		self.goal_x = 0.0
		self.goal_y = 0.0

		self.last_left = 0.
		self.last_right = 0.
		self.last_w = 0.0
		self.jointVel=None

		# to calculate vx, vy
		self.last_v = 0.0
		self.desiredVelocity=[0.0,0.0]

		self.mutex = Lock()



	def configure(self, config):
		super().configure(config)
		# kalman filter for human tracking

		if config.sim.predict_method == 'none':
			self.add_pred = False
		else:
			self.add_pred = True

		# define ob space and action space
		self.set_ob_act_space()
		# ROS
		rospy.init_node('ros_turtlebot2i_env_node', anonymous=True)

		self.actionPublisher = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=1)
		self.tfBuffer = tf2_ros.Buffer()
		self.transformListener = tf2_ros.TransformListener(self.tfBuffer)
		# ROS subscriber
		jointStateSub = Subscriber("/joint_states", JointState)
		humanStatesSub = Subscriber('/dr_spaam_detections', PoseArray)  # human px, py, visible
		if self.use_dummy_detect:
			subList = [jointStateSub]
		else:
			subList = [jointStateSub, humanStatesSub]

		# synchronize the robot base joint states and humnan detections with at most 1 seconds of difference
		self.ats = ApproximateTimeSynchronizer(subList, queue_size=1, slop=1)

		# if ignore sensor inputs and use fake human detections
		if self.use_dummy_detect:
			self.ats.registerCallback(self.state_cb_dummy)
		else:
			self.ats.registerCallback(self.state_cb)

		rospy.on_shutdown(self.shutdown)


	def set_robot(self, robot):
		self.robot = robot

	def set_ob_act_space(self):
		# set observation space and action space
		# we set the max and min of action/observation space as inf
		# clip the action and observation as you need

		d = {}
		# robot node: num_visible_humans, px, py, r, gx, gy, v_pref, theta
		d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7,), dtype=np.float32)
		# only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
		d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
		# add prediction
		if self.add_pred:
			self.spatial_edge_dim = int(2 * (self.predict_steps + 1))
			d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf,
												shape=(self.config.sim.human_num + self.config.sim.human_num_range,
													   self.spatial_edge_dim),
												dtype=np.float32)
		# no prediction
		else:
			d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf,
												shape=(self.config.sim.human_num + self.config.sim.human_num_range, 2),
												dtype=np.float32)
		# number of humans detected at each timestep

		# masks for gst pred model
		# whether each human is visible to robot
		d['visible_masks'] = gym.spaces.Box(low=-np.inf, high=np.inf,
											shape=(self.config.sim.human_num + self.config.sim.human_num_range,),
											dtype=np.bool)

		# number of humans detected at each timestep
		d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

		self.observation_space = gym.spaces.Dict(d)

		high = np.inf * np.ones([2, ])
		self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)


	# (used if self.use_dummy_detect is False)
	# callback function to store the realtime messages from the robot to this env
	def state_cb(self, jointStateMsg, humanArrayMsg):
		self.humanMsg=humanArrayMsg.poses
		self.jointMsg=jointStateMsg

	# (used if self.use_dummy_detect is True)
	# callback function to store the realtime messages from the robot to this env
	# no need to real human message
	def state_cb_dummy(self, jointStateMsg):
		self.jointMsg = jointStateMsg

	def readMsg(self):
		"""
		read messages passed through ROS & prepare for generating obervations
		this function should be called right before generate_ob() is called
		"""
		self.mutex.acquire()
		# get time
		# print(self.jointMsg.header.stamp.secs, self.jointMsg.header.stamp.nsecs)
		self.currentTime = self.jointMsg.header.stamp.secs + self.jointMsg.header.stamp.nsecs / 1e9

		# get robot pose from T265 SLAM camera
		try:
			self.robotMsg = self.tfBuffer.lookup_transform('t265_odom_frame', 't265_pose_frame', rospy.Time.now(), rospy.Duration(1.0))
		except:
			print("problem in getting transform")

		# get robot base velocity from the base
		try:
			self.jointVel=self.jointMsg.velocity
		except:
			print("problem in getting joint velocity")

		# print(self.robotMsg, "ROBOT mSG")
		# store the robot pose and robot base velocity in self variables
		self.robot.px = -self.robotMsg.transform.translation.y
		self.robot.py = self.robotMsg.transform.translation.x
		print('robot pos:', self.robot.px, self.robot.py)

		quaternion = (
			self.robotMsg.transform.rotation.x,
			self.robotMsg.transform.rotation.y,
			self.robotMsg.transform.rotation.z,
			self.robotMsg.transform.rotation.w
		)

		if self.use_dummy_detect:
			self.detectedHumanNum = 1
			self.human_visibility = np.zeros((self.max_human_num,), dtype=np.bool)
		else:
			# human states

			self.detectedHumanNum=min(len(self.humanMsg), self.max_human_num)
			self.current_human_states_raw = np.ones((self.detectedHumanNum, 2)) * 15
			self.human_visibility=np.zeros((self.max_human_num, ), dtype=np.bool)


			for i in range(self.detectedHumanNum):
				# use hard coded map to filter out obstacles
				global_x = self.robot.px + self.humanMsg[i].position.x
				global_y = self.robot.py + self.humanMsg[i].position.y
				# if -2.5 < global_x < 2.5 and -2.5 < global_y < 2.5:
				if True:
					self.current_human_states_raw[i,0]=self.humanMsg[i].position.x
					self.current_human_states_raw[i,1] = self.humanMsg[i].position.y

		self.mutex.release()

		# robot orientation
		self.robot.theta = tf.transformations.euler_from_quaternion(quaternion)[2] + np.pi / 2

		if self.robot.theta < 0:
			self.robot.theta = self.robot.theta + 2 * np.pi

		# add 180 degrees because of the transform from lidar frame to t265 camera frame
		hMatrix = np.array([[np.cos(self.robot.theta+np.pi), -np.sin(self.robot.theta+np.pi), 0, 0],
							  [np.sin(self.robot.theta+np.pi), np.cos(self.robot.theta+np.pi), 0, 0],
							 [0,0,1,0], [0,0,0,1]])

		# if we detected at least one person
		self.current_human_states = np.ones((self.max_human_num, 2)) * 15

		if not self.use_dummy_detect:
			for j in range(self.detectedHumanNum):
				xy=np.matmul(hMatrix,np.array([[self.current_human_states_raw[j,0],
												self.current_human_states_raw[j,1],
												0,
												1]]).T)

				self.current_human_states[j]=xy[:2,0]

		else:
			self.current_human_states[0] = np.array([0, 1]) - np.array([self.robot.px, self.robot.py])



		self.robot.vx = self.last_v * np.cos(self.robot.theta)
		self.robot.vy = self.last_v * np.sin(self.robot.theta)




	def reset(self):
		"""
		Reset function
		"""

		# stop the turtlebot
		self.smoothStop()
		self.step_counter=0
		self.currentTime=0.0
		self.lastTime=0.0
		self.global_time = 0.

		self.human_visibility = np.zeros((self.max_human_num,), dtype=np.bool)
		self.detectedHumanNum=0
		self.current_human_states = np.ones((self.max_human_num, 2)) * 15
		self.desiredVelocity = [0.0, 0.0]
		self.last_left  = 0.
		self.last_right = 0.
		self.last_w = 0.0

		self.last_v = 0.0

		while True:
			a = input("Press y for the next episode \t")
			if a == "y":
				self.robot.gx=float(input("Input goal location in x-axis\t"))
				self.robot.gy=float(input("Input goal location in y-axis\t"))
				break
			else:
				sys.exit()


		if self.record:
			self.episodeRecoder.robot_goal.append([self.robot.gx, self.robot.gy])


		self.readMsg()
		ob=self.generate_ob() # generate initial obs


		return ob

	# input: v, w
	# output: v, w
	def smooth(self, v, w):
		beta = 0.1 #TODO: you use 0.2 in the simulator
		left = (2. * v - 0.23 * w) / (2. * 0.035)
		right = (2. * v + 0.23 * w) / (2. * 0.035)
		# print('199:', left, right)
		left = np.clip(left, -17.5, 17.5)
		right = np.clip(right, -17.5, 17.5)
		# print('202:', left, right)
		left = (1.-beta) * self.last_left + beta * left
		right = (1.-beta) * self.last_right + beta * right
		# print('205:', left, right)
		
		self.last_left = copy.deepcopy(left)
		self.last_right = copy.deepcopy(right)

		v_smooth = 0.035 / 2 * (left + right)
		w_smooth = 0.035 / 0.23 * (right - left)

		return v_smooth, w_smooth

	def generate_ob(self):
		ob = {}

		ob['robot_node'] = np.array([[self.robot.px, self.robot.py, self.robot.radius,
								self.robot.gx, self.robot.gy, self.robot.v_pref, self.robot.theta]])
		ob['temporal_edges']=np.array([[self.robot.vx, self.robot.vy]])
		# print(self.current_human_states.shape)
		spatial_edges=self.current_human_states
		if self.add_pred:
			# predicted steps will be filled in the vec_pretext_normalize wrapper
			spatial_edges=np.concatenate([spatial_edges, np.zeros((self.max_human_num, 2 * self.predict_steps))], axis=1)

		else:
			# sort humans by distance to robot
			spatial_edges = np.array(sorted(spatial_edges, key=lambda x: np.linalg.norm(x)))
			print(spatial_edges)
		ob['spatial_edges'] = spatial_edges
		ob['visible_masks'] = self.human_visibility

		ob['detected_human_num'] = self.detectedHumanNum
		if ob['detected_human_num'] == 0:
			ob['detected_human_num'] = 1
		print(ob['detected_human_num'])
			
		return ob
		

	def step(self, action, update=True):
		""" Step function """
		print("Step", self.step_counter)
		# process action
		realAction = Twist()

		if self.load_act: # load action from file for robot dynamics checking
			v_unsmooth= self.episodeRecoder.v_list[self.step_counter]
			# in the simulator we use and recrod delta theta. We convert it to omega by dividing it by the time interval
			w_unsmooth = self.episodeRecoder.delta_theta_list[self.step_counter] / self.delta_t
			# v_smooth, w_smooth = self.desiredVelocity[0], self.desiredVelocity[1]
			v_smooth, w_smooth = self.smooth(v_unsmooth, w_unsmooth)
		else:
			action = self.robot.policy.clip_action(action, None)

			self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + action.v, -self.robot.v_pref, self.robot.v_pref)
			self.desiredVelocity[1] = action.r / self.fixed_time_interval # TODO: dynamic time step is not supported now


			v_smooth, w_smooth = self.smooth(self.desiredVelocity[0], self.desiredVelocity[1])

		
		self.last_v = v_smooth

		realAction.linear.x = v_smooth
		realAction.angular.z = w_smooth

		self.actionPublisher.publish(realAction)


		rospy.sleep(self.ROSStepInterval)  # act as frame skip

		# get the latest states

		self.readMsg()


		# update time
		if self.step_counter==0: # if it is the first step of the episode
			self.delta_t = np.inf
		else:
			# time interval between two steps
			if self.use_fixed_time_interval:
				self.delta_t=self.fixed_time_interval
			else: self.delta_t = self.currentTime - self.lastTime
			print('delta_t:', self.currentTime - self.lastTime)
			#print('actual delta t:', currentTime - self.baseEnv.lastTime)
			self.global_time = self.global_time + self.delta_t
		self.step_counter=self.step_counter+1
		self.lastTime = self.currentTime


		# generate new observation
		ob=self.generate_ob()


		# calculate reward
		reward = 0

		# determine if the episode ends
		done=False
		reaching_goal = norm(np.array([self.robot.gx, self.robot.gy]) - np.array([self.robot.px, self.robot.py]))  < 0.6
		if self.global_time >= self.time_limit:
			done = True
			print("Timeout")
		elif reaching_goal:
			done = True
			print("Goal Achieved")
		elif self.load_act and self.record:
			if self.step_counter >= len(self.episodeRecoder.v_list):
				done = True
		else:
			done = False


		info = {'info': None}

		if self.record:
			self.episodeRecoder.wheelVelList.append(self.jointVel) # it is the calculated wheel velocity, not the measured
			self.episodeRecoder.actionList.append([v_smooth, w_smooth])
			self.episodeRecoder.positionList.append([self.robot.px, self.robot.py])
			self.episodeRecoder.orientationList.append(self.robot.theta)

		if done:
			print('DOne!')
			if self.record:
				self.episodeRecoder.saveEpisode(self.case_counter['test'])


		return ob, reward, done, info

	def render(self, mode='human'):
		import matplotlib.pyplot as plt
		import matplotlib.lines as mlines
		from matplotlib import patches

		plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

		robot_color = 'yellow'
		goal_color = 'red'
		arrow_color = 'red'
		arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

		def calcFOVLineEndPoint(ang, point, extendFactor):
			# choose the extendFactor big enough
			# so that the endPoints of the FOVLine is out of xlim and ylim of the figure
			FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
								   [np.sin(ang), np.cos(ang), 0],
								   [0, 0, 1]])
			point.extend([1])
			# apply rotation matrix
			newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
			# increase the distance between the line start point and the end point
			newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
			return newPoint

		ax = self.render_axis
		artists = []

		# add goal
		goal = mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None',
							 markersize=15, label='Goal')
		ax.add_artist(goal)
		artists.append(goal)

		# add robot
		robotX, robotY = self.robot.px, self.robot.py

		robot = plt.Circle((robotX, robotY), self.robot.radius, fill=True, color=robot_color)
		ax.add_artist(robot)
		artists.append(robot)

		plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

		# compute orientation in each step and add arrow to show the direction
		radius = 0.1
		arrowStartEnd = []

		robot_theta = self.robot.theta

		arrowStartEnd.append(
			((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

		arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
				  for arrow in arrowStartEnd]
		for arrow in arrows:
			ax.add_artist(arrow)
			artists.append(arrow)


		# add humans and change the color of them based on visibility
		# print(self.current_human_states.shape)
		# print(self.track_num)
		# render tracked humans
		human_circles = [plt.Circle(self.current_human_states[i] + np.array([self.robot.px, self.robot.py]), 0.2, fill=False) \
						 for i in range(self.max_human_num)]
		# render untracked humans
		# human_circles = [
		# 	plt.Circle(self.current_human_states_raw[i] + np.array([self.robot.px, self.robot.py]), 0.2, fill=False) for i
		# 	in range(self.detectedHumanNum)]


		for i in range(self.max_human_num):
		# for i in range(self.detectedHumanNum):
			ax.add_artist(human_circles[i])
			artists.append(human_circles[i])

			# green: visible; red: invisible
			if self.human_visibility[i]:
				human_circles[i].set_color(c='g')
			else:
				human_circles[i].set_color(c='r')


		plt.pause(0.1)
		for item in artists:
			item.remove()  # there should be a better way to do this. For example,
		# initially use add_artist and draw_artist later on
		for t in ax.texts:
			t.set_visible(False)

	def shutdown(self):
		self.smoothStop()
		print("You are stopping the robot!")
		self.reset()
		

	def smoothStop(self):
		realAction = Twist()
		self.actionPublisher.publish(Twist())



