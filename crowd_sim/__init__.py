from gym.envs.registration import register

register(
    id='CrowdSim-v0',
    entry_point='crowd_sim.envs:CrowdSim',
)

register(
    id='CrowdSimPred-v0',
    entry_point='crowd_sim.envs:CrowdSimPred',
)

register(
    id='CrowdSimVarNum-v0',
    entry_point='crowd_sim.envs:CrowdSimVarNum',
)

register(
    id='CrowdSimVarNumCollect-v0',
    entry_point='crowd_sim.envs:CrowdSimVarNumCollect',
)

register(
    id='CrowdSimPredRealGST-v0',
    entry_point='crowd_sim.envs:CrowdSimPredRealGST',
)

register(
    id='rosTurtlebot2iEnv-v0',
    entry_point='crowd_sim.envs.ros_turtlebot2i_env:rosTurtlebot2iEnv',
)