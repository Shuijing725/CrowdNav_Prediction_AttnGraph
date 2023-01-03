import numpy as np
from crowd_nav.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY

class SOCIAL_FORCE(Policy):
    def __init__(self, config):
        super().__init__(config)
        self.name = 'social_force'


    def predict(self, state):
        """
        Produce action for agent with circular specification of social force model.
        """
        # Pull force to goal
        delta_x = state.self_state.gx - state.self_state.px
        delta_y = state.self_state.gy - state.self_state.py
        dist_to_goal = np.sqrt(delta_x**2 + delta_y**2)
        desired_vx = (delta_x / dist_to_goal) * state.self_state.v_pref
        desired_vy = (delta_y / dist_to_goal) * state.self_state.v_pref
        KI = self.config.sf.KI # Inverse of relocation time K_i
        curr_delta_vx = KI * (desired_vx - state.self_state.vx)
        curr_delta_vy = KI * (desired_vy - state.self_state.vy)
        
        # Push force(s) from other agents
        A = self.config.sf.A # Other observations' interaction strength: 1.5
        B = self.config.sf.B # Other observations' interaction range: 1.0
        interaction_vx = 0
        interaction_vy = 0
        for other_human_state in state.human_states:
            delta_x = state.self_state.px - other_human_state.px
            delta_y = state.self_state.py - other_human_state.py
            dist_to_human = np.sqrt(delta_x**2 + delta_y**2)
            interaction_vx += A * np.exp((state.self_state.radius + other_human_state.radius - dist_to_human) / B) * (delta_x / dist_to_human)
            interaction_vy += A * np.exp((state.self_state.radius + other_human_state.radius - dist_to_human) / B) * (delta_y / dist_to_human)

        # Sum of push & pull forces
        total_delta_vx = (curr_delta_vx + interaction_vx) * self.config.env.time_step
        total_delta_vy = (curr_delta_vy + interaction_vy) * self.config.env.time_step

        # clip the speed so that sqrt(vx^2 + vy^2) <= v_pref
        new_vx = state.self_state.vx + total_delta_vx
        new_vy = state.self_state.vy + total_delta_vy
        act_norm = np.linalg.norm([new_vx, new_vy])

        if act_norm > state.self_state.v_pref:
            return ActionXY(new_vx / act_norm * state.self_state.v_pref, new_vy / act_norm * state.self_state.v_pref)
        else:
            return ActionXY(new_vx, new_vy)
