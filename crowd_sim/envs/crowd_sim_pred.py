import gym
import numpy as np
from numpy.linalg import norm
import copy

from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.crowd_sim_var_num import CrowdSimVarNum



class CrowdSimPred(CrowdSimVarNum):
    """
    The environment for our model with non-neural network trajectory predictors, including const vel predictor and ground truth predictor
    The number of humans at each timestep can change within a range
    """
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super(CrowdSimPred, self).__init__()
        self.pred_method = None
        self.cur_human_states=None

    def configure(self, config):
        """ read the config to the environment variables """
        super().configure(config)
        # 'const_vel', 'truth', or 'inferred'
        self.pred_method = config.sim.predict_method

    def reset(self, phase='train', test_case=None):
        ob = super().reset(phase=phase, test_case=test_case)
        return ob

    # set observation space and action space
    def set_robot(self, robot):
        self.robot = robot

        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need

        d = {}
        # robot node: num_visible_humans, px, py, r, gx, gy, v_pref, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7,), dtype=np.float32)
        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)

        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                            shape=(self.config.sim.human_num + self.config.sim.human_num_range, int(2*(self.predict_steps+1))),
                            dtype=np.float32)
        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)


    # reset = True: reset calls this function; reset = False: step calls this function
    def generate_ob(self, reset, sort=True):
        """Generate observation for reset and step functions"""
        ob = {}

        # nodes
        visible_humans, num_visibles, self.human_visibility = self.get_num_human_in_fov()

        ob['robot_node'] = self.robot.get_full_state_list_noV()

        self.prev_human_pos = copy.deepcopy(self.last_human_states)
        self.update_last_human_states(self.human_visibility, reset=reset)

        # edges
        ob['temporal_edges'] = np.array([self.robot.vx, self.robot.vy])

        # initialize storage space for max_human_num humans
        ob['spatial_edges'] = np.ones((self.config.sim.human_num + self.config.sim.human_num_range, int(2*(self.predict_steps+1)))) * np.inf

        # calculate the current and future positions of present humans (first self.human_num humans in ob['spatial_edges'])
        predicted_states = self.calc_human_future_traj(method=self.pred_method) # [pred_steps + 1, human_num, 4]
        # [human_num, pred_steps + 1, 2]
        predicted_pos = np.transpose(predicted_states[:, :, :2], (1, 0, 2)) - np.array([self.robot.px, self.robot.py])

        # [human_num, (pred_steps + 1)*2]
        ob['spatial_edges'][:self.human_num][self.human_visibility] = predicted_pos.reshape((self.human_num, -1))[self.human_visibility]
        # sort all humans by distance (invisible humans will be in the end automatically)
        if self.config.args.sort_humans:
            ob['spatial_edges'] = np.array(sorted(ob['spatial_edges'], key=lambda x: np.linalg.norm(x[:2])))
        ob['spatial_edges'][np.isinf(ob['spatial_edges'])] = 15

        ob['detected_human_num'] = num_visibles
        # if no human is detected, assume there is one dummy human at (15, 15) to make the pack_padded_sequence work
        if ob['detected_human_num'] == 0:
            ob['detected_human_num'] = 1

        return ob


    def step(self, action, update=True):
        """
        step function
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        if self.robot.policy.name == 'ORCA':
            # assemble observation for orca: px, py, vx, vy, r
            # include all observable humans from t to t+t_pred
            _, _, human_visibility = self.get_num_human_in_fov()
            # [self.predict_steps + 1, self.human_num, 4]
            human_states = copy.deepcopy(self.calc_human_future_traj(method='truth'))
            # append the radius, convert it to [human_num*(self.predict_steps+1), 5] by treating each predicted pos as a new human
            human_states = np.concatenate((human_states.reshape((-1, 4)),
                                           np.tile(self.last_human_states[:, -1], self.predict_steps+1).reshape((-1, 1))),
                                          axis=1)
            # get orca action
            action = self.robot.act(human_states.tolist())
        else:
            action = self.robot.policy.clip_action(action, self.robot.v_pref)

        if self.robot.kinematics == 'unicycle':
            self.desiredVelocity[0] = np.clip(self.desiredVelocity[0] + action.v, -self.robot.v_pref, self.robot.v_pref)
            action = ActionRot(self.desiredVelocity[0], action.r)

            # if action.r is delta theta
            action = ActionRot(self.desiredVelocity[0], action.r)
            if self.record:
                self.episodeRecoder.unsmoothed_actions.append(list(action))

            action = self.smooth_action(action)



        human_actions = self.get_human_actions()

        # need to update self.human_future_traj in testing to calculate number of intrusions
        if self.phase == 'test':
            # use ground truth future positions of humans
            self.calc_human_future_traj(method='truth')

        # compute reward and episode info
        reward, done, episode_info = self.calc_reward(action, danger_zone='future')


        if self.record:

            self.episodeRecoder.actionList.append(list(action))
            self.episodeRecoder.positionList.append([self.robot.px, self.robot.py])
            self.episodeRecoder.orientationList.append(self.robot.theta)

            if done:
                self.episodeRecoder.robot_goal.append([self.robot.gx, self.robot.gy])
                self.episodeRecoder.saveEpisode(self.case_counter['test'])

        # apply action and update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
            self.cur_human_states[i] = np.array([self.humans[i].px, self.humans[i].py, self.humans[i].radius])

        self.global_time += self.time_step # max episode length=time_limit/time_step
        self.step_counter = self.step_counter+1

        info={'info':episode_info}

        # Add or remove at most self.human_num_range humans
        # if self.human_num_range == 0 -> human_num is fixed at all times
        if self.human_num_range > 0 and self.global_time % 5 == 0:
            # remove humans
            if np.random.rand() < 0.5:
                # if no human is visible, anyone can be removed
                if len(self.observed_human_ids) == 0:
                    max_remove_num = self.human_num - 1
                else:
                    max_remove_num = (self.human_num - 1) - max(self.observed_human_ids)
                remove_num = np.random.randint(low=0, high=min(self.human_num_range, max_remove_num) + 1)
                for _ in range(remove_num):
                    self.humans.pop()
                self.human_num = self.human_num - remove_num
                self.last_human_states = self.last_human_states[:self.human_num]
            # add humans
            else:
                add_num = np.random.randint(low=0, high=self.human_num_range + 1)
                if add_num > 0:
                    # set human ids
                    true_add_num = 0
                    for i in range(self.human_num, self.human_num + add_num):
                        if i == self.config.sim.human_num + self.human_num_range:
                            break
                        self.generate_random_human_position(human_num=1)
                        self.humans[i].id = i
                        true_add_num = true_add_num + 1
                    self.human_num = self.human_num + true_add_num
                    if true_add_num > 0:
                        self.last_human_states = np.concatenate((self.last_human_states, np.array([[15, 15, 0, 0, 0.3]]*true_add_num)), axis=0)


        # compute the observation
        ob = self.generate_ob(reset=False)


        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()

        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing and not self.record:
            for i, human in enumerate(self.humans):
                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.humans[i] = self.generate_circle_crossing_human()
                    self.humans[i].id = i

        return ob, reward, done, info


    def calc_reward(self, action, danger_zone='future'):
        """Calculate the reward, which includes the functionality reward and social reward"""
        # compute the functionality reward, same as the reward function with no prediction
        reward, done, episode_info = super().calc_reward(action, danger_zone=danger_zone)
        # compute the social reward
        # if the robot collides with future states, give it a collision penalty
        relative_pos = self.human_future_traj[1:, :, :2] - np.array([self.robot.px, self.robot.py])
        # assumes constant radius of each personal zone circle
        collision_idx = np.linalg.norm(relative_pos, axis=-1) < self.robot.radius + self.config.humans.radius # [predict_steps, human_num]

        coefficients = 2. ** np.arange(2, self.predict_steps + 2).reshape((self.predict_steps, 1))  # 4, 8, 16, 32

        collision_penalties = self.collision_penalty / coefficients

        reward_future = collision_idx * collision_penalties # [predict_steps, human_num]
        reward_future = np.min(reward_future)

        return reward + reward_future, done, episode_info


    def render(self, mode='human'):
        """ Render the current status of the environment using matplotlib """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        robot_color = 'gold'
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



        ax=self.render_axis
        artists=[]

        # add goal
        goal=mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
        ax.add_artist(goal)
        artists.append(goal)

        # add robot
        robotX,robotY=self.robot.get_position()

        robot=plt.Circle((robotX,robotY), self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)

        # plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)


        # compute orientation in each step and add arrow to show the direction
        radius = self.robot.radius
        arrowStartEnd=[]

        robot_theta = self.robot.theta if self.robot.kinematics == 'unicycle' else np.arctan2(self.robot.vy, self.robot.vx)

        arrowStartEnd.append(((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

        for i, human in enumerate(self.humans):
            theta = np.arctan2(human.vy, human.vx)
            arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

        arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                  for arrow in arrowStartEnd]
        for arrow in arrows:
            ax.add_artist(arrow)
            artists.append(arrow)


        # draw FOV for the robot
        # add robot FOV
        if self.robot.FOV < 2 * np.pi:
            FOVAng = self.robot_fov / 2
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')


            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + radius * np.cos(robot_theta)
            endPointY = robotY + radius * np.sin(robot_theta)

            # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
            # the start point of the FOVLine is the center of the robot
            FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

            ax.add_artist(FOVLine1)
            ax.add_artist(FOVLine2)
            artists.append(FOVLine1)
            artists.append(FOVLine2)

        # add an arc of robot's sensor range
        sensor_range = plt.Circle(self.robot.get_position(), self.robot.sensor_range + self.robot.radius+self.config.humans.radius, fill=False, linestyle='--')
        ax.add_artist(sensor_range)
        artists.append(sensor_range)

        # add humans and change the color of them based on visibility
        human_circles = [plt.Circle(human.get_position(), human.radius, fill=False, linewidth=1.5) for human in self.humans]
        # human_circles = [plt.Circle(human.get_position(), 2, fill=False) for human in self.humans]

        # hardcoded for now
        actual_arena_size = self.arena_size + 0.5
        # plot current human states
        for i in range(len(self.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])

            # green: visible; red: invisible
            if self.detect_visible(self.robot, self.humans[i], robot1=True):
                human_circles[i].set_color(c='b')
            else:
                human_circles[i].set_color(c='r')


            if -actual_arena_size <= self.humans[i].px <= actual_arena_size and -actual_arena_size <= self.humans[
                i].py <= actual_arena_size:
                # label numbers on each human
                # plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(self.humans[i].id), color='black', fontsize=12)
                plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, i, color='black', fontsize=12)


        # save plot for slide show
        if self.config.save_slides:
            import os
            folder_path = os.path.join(self.config.save_path, str(self.rand_seed)+'pred')
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path, exist_ok = True)
            plt.savefig(os.path.join(folder_path, str(self.step_counter)+'.png'), dpi=300)

        plt.pause(0.1)
        for item in artists:
            item.remove() # there should be a better way to do this. For example,
            # initially use add_artist and draw_artist later on
        for t in ax.texts:
            t.set_visible(False)
