import gym
import numpy as np
from numpy.linalg import norm
import copy

from crowd_sim.envs import *
from crowd_sim.envs.utils.info import *



class CrowdSimVarNumCollect(CrowdSimVarNum):
    """
    An environment for collecting a dataset of simulated humans to train GST predictor (used in collect_data.py)
    The observation contains all detected humans
    A key in ob indicates how many humans are detected
    """
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super().__init__()


    def set_robot(self, robot):
        self.robot = robot

        # set observation space and action space
        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need

        d={}
        # frame id, human prediction id, px, py
        d['pred_info'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.sim.human_num + self.config.sim.human_num_range, 4), dtype=np.float32)
        self.observation_space=gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)


    def reset(self, phase='train', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case=self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case # test case is passed in to calculate specific seed to generate case
        self.global_time = 0
        self.id_counter = 0


        self.humans = []
        # self.human_num = self.config.sim.human_num
        # initialize a list to store observed humans' IDs
        self.observed_human_ids = []

        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        # here we use a counter to calculate seed. The seed=counter_offset + case_counter
        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)
        self.rand_seed = counter_offset[phase] + self.case_counter[phase] + self.thisSeed
        # print(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)
        # np.random.seed(1038)

        self.generate_robot_humans(phase)
        self.last_human_observability = np.zeros(self.human_num, dtype=bool)
        self.human_pred_id = np.arange(0, self.human_num)
        self.max_human_id = self.human_num

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]

        # get robot observation
        ob = self.generate_ob(reset=True)

        # initialize potential
        self.potential = -abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))

        return ob

    # reset = True: reset calls this function; reset = False: step calls this function
    def generate_ob(self, reset, sort=False):
        # we should keep human ID tracking for traj pred!
        #assert sort == False
        ob = {}

        # nodes
        visible_humans, num_visibles, human_visibility = self.get_num_human_in_fov()

        humans_out = np.logical_and(self.last_human_observability, np.logical_not(human_visibility))
        num_humans_out = np.sum(humans_out)

        self.human_pred_id[humans_out] = np.arange(self.max_human_id, self.max_human_id+num_humans_out)
        self.max_human_id = self.max_human_id + num_humans_out

        self.update_last_human_states(human_visibility, reset=reset)

        # ([relative px, relative py, disp_x, disp_y], human id)
        all_spatial_edges = np.ones((self.config.sim.human_num + self.config.sim.human_num_range, 2)) * np.inf

        for i in range(self.human_num):
            if human_visibility[i]:
                all_spatial_edges[self.humans[i].id, :2] = self.last_human_states[i, :2]

        frame_array = np.repeat(self.global_time/self.config.data.pred_timestep, self.human_num)
        ob['pred_info'] = np.concatenate((frame_array.reshape((self.human_num, 1)),
                          self.human_pred_id.reshape((self.human_num, 1)), all_spatial_edges), axis=1)

        # update self.observed_human_ids
        self.observed_human_ids = np.where(human_visibility)[0]
        self.ob = ob

        self.last_human_observability = copy.deepcopy(human_visibility)

        return ob

    # find R(s, a)
    # danger_zone: how to define the personal_zone (if the robot intrudes into this zone, the info will be Danger)
    # circle (traditional) or future (based on true future traj of humans)
    def calc_reward(self, action, danger_zone='circle'):
        # collision detection
        dmin = float('inf')

        danger_dists = []
        collision = False

        for i, human in enumerate(self.humans):
            dx = human.px - self.robot.px
            dy = human.py - self.robot.py
            closest_dist = (dx ** 2 + dy ** 2) ** (1 / 2) - human.radius - self.robot.radius

            if closest_dist < self.discomfort_dist:
                danger_dists.append(closest_dist)
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # check if reaching the goal
        reaching_goal = norm(
            np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())) < self.robot.radius


        if self.global_time >= 40000:
            done = True
            episode_info = Timeout()
        elif collision:
            done = False
            episode_info = Collision()
        elif reaching_goal:
            done = False
            episode_info = ReachGoal()
            # randomize human attributes

            # median of all humans
            if np.random.uniform(0, 1) < 0.5:
                human_pos = np.zeros((self.human_num, 2))
                for i, human in enumerate(self.humans):
                    human_pos[i] = np.array([human.px, human.py])
                self.robot.gx, self.robot.gy = np.median(human_pos, axis=0)
            # random goal
            else:
                self.robot.gx, self.robot.gy = np.random.uniform(-self.arena_size, self.arena_size, size=2)

        else:
            done = False
            episode_info = Nothing()



        return 0, done, episode_info