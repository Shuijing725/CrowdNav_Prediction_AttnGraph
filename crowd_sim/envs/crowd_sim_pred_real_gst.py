import gym
import numpy as np

from crowd_sim.envs.crowd_sim_pred import CrowdSimPred


class CrowdSimPredRealGST(CrowdSimPred):
    '''
    Same as CrowdSimPred, except that
    The future human traj in 'spatial_edges' are dummy placeholders
    and will be replaced by the outputs of a real GST pred model in the wrapper function in vec_pretext_normalize.py
    '''
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super(CrowdSimPredRealGST, self).__init__()
        self.pred_method = None

        # to receive data from gst pred model
        self.gst_out_traj = None


    def set_robot(self, robot):
        """set observation space and action space"""
        self.robot = robot

        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need

        d = {}
        # robot node: num_visible_humans, px, py, r, gx, gy, v_pref, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7,), dtype=np.float32)
        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        '''
        format of spatial_edges: [max_human_num, [state_t, state_(t+1), ..., state(t+self.pred_steps)]]
        '''

        # predictions only include mu_x, mu_y (or px, py)
        self.spatial_edge_dim = int(2*(self.predict_steps+1))

        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                            shape=(self.config.sim.human_num + self.config.sim.human_num_range, self.spatial_edge_dim),
                            dtype=np.float32)

        # masks for gst pred model
        # whether each human is visible to robot (ordered by human ID, should not be sorted)
        d['visible_masks'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.config.sim.human_num + self.config.sim.human_num_range,),
                                            dtype=np.bool)

        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def talk2Env(self, data):
        """
        Call this function when you want extra information to send to/recv from the env
        :param data: data that is sent from gst_predictor network to the env, it has 2 parts:
        output predicted traj and output masks
        :return: True means received
        """
        self.gst_out_traj=data
        return True


    # reset = True: reset calls this function; reset = False: step calls this function
    def generate_ob(self, reset, sort=False):
        """Generate observation for reset and step functions"""
        # since gst pred model needs ID tracking, don't sort all humans
        # inherit from crowd_sim_lstm, not crowd_sim_pred to avoid computation of true pred!
        # sort=False because we will sort in wrapper in vec_pretext_normalize.py later
        parent_ob = super(CrowdSimPred, self).generate_ob(reset=reset, sort=False)

        # add additional keys, removed unused keys
        ob = {}

        ob['visible_masks'] = parent_ob['visible_masks']
        ob['robot_node'] = parent_ob['robot_node']
        ob['temporal_edges'] = parent_ob['temporal_edges']

        ob['spatial_edges'] = np.tile(parent_ob['spatial_edges'], self.predict_steps+1)

        ob['detected_human_num'] = parent_ob['detected_human_num']

        return ob


    def calc_reward(self, action, danger_zone='future'):
        # inherit from crowd_sim_lstm, not crowd_sim_pred to prevent social reward calculation
        # since we don't know the GST predictions yet
        reward, done, episode_info = super(CrowdSimPred, self).calc_reward(action, danger_zone=danger_zone)
        return reward, done, episode_info


    def render(self, mode='human'):
        """
        render function
        use talk2env to plot the predicted future traj of humans
        """
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

        # hardcoded for now
        actual_arena_size = self.arena_size + 0.5

        # plot the current human states
        for i in range(len(self.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])

            # green: visible; red: invisible
            if self.human_visibility[i]:
                human_circles[i].set_color(c='b')
            else:
                human_circles[i].set_color(c='r')

            if -actual_arena_size <= self.humans[i].px <= actual_arena_size and -actual_arena_size <= self.humans[
                i].py <= actual_arena_size:
                # label numbers on each human
                # plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(self.humans[i].id), color='black', fontsize=12)
                plt.text(self.humans[i].px , self.humans[i].py , i, color='black', fontsize=12)

        # plot predicted human positions
        for i in range(len(self.humans)):
            # add future predicted positions of each human
            if self.gst_out_traj is not None:
                for j in range(self.predict_steps):
                    circle = plt.Circle(self.gst_out_traj[i, (2 * j):(2 * j + 2)] + np.array([robotX, robotY]),
                                        self.config.humans.radius, fill=False, color='tab:orange', linewidth=1.5)
                    # circle = plt.Circle(np.array([robotX, robotY]),
                    #                     self.humans[i].radius, fill=False)
                    ax.add_artist(circle)
                    artists.append(circle)

        plt.pause(0.1)
        for item in artists:
            item.remove() # there should be a better way to do this. For example,
            # initially use add_artist and draw_artist later on
        for t in ax.texts:
            t.set_visible(False)
