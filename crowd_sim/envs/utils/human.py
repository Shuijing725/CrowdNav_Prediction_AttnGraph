from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Human(Agent):
    # see Agent class in agent.py for details!!!
    def __init__(self, config, section):
        super().__init__(config, section)
        self.isObstacle = False # whether the human is a static obstacle (part of wall) or a moving agent
        self.id = None # the human's ID, used for collecting traj data
        self.observed_id = -1 # if it is observed, give it a tracking ID

    # ob: a list of observable states
    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    # ob: a joint state (ego agent's full state + other agents' observable states)
    def act_joint_state(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        action = self.policy.predict(ob)
        return action
