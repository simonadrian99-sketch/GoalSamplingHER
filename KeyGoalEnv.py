from SimpleEnv import SimpleEnv
from gymnasium import spaces
import numpy as np
import gymnasium as gym
from minigrid.core.world_object import Key
from minigrid.manual_control import ManualControl


class KeyGoalEnv(SimpleEnv):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        image_space = self.observation_space.spaces['observation']
        self.observation_space = gym.spaces.Dict({
            "observation": image_space,
            "direction": gym.spaces.Box(low=0, high=3, shape=(1,), dtype=np.int64),
            "achieved_goal": gym.spaces.Box(low=0, high=12, shape=(3,), dtype=np.int64),
            "desired_goal":  gym.spaces.Box(low=0, high=12, shape=(3,), dtype=np.int64),
        })
        self.agent_has_key = False

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self.key_pos = (width//2, 2)
        self.key_obj = PassableKey('yellow')
        self.grid.set(int(self.key_pos[0]), int(self.key_pos[1]), self.key_obj)

    def reset(self, **kwargs):
        goal_obs, info = super().reset(**kwargs)
        self.agent_has_key = False

        goal_obs["achieved_goal"] = np.array(
            [self.agent_pos[0], self.agent_pos[1], 0], dtype=np.int64)
        goal_obs["desired_goal"] = np.array(
            [self.goal_pos[0], self.goal_pos[1], 1], dtype=np.int64)
        return goal_obs, info

    def step(self, action):

        old_pos = self.unwrapped.agent_pos
        obs, _, terminated, truncated, info = super(
            SimpleEnv, self).step(action)

        step_reward = 0
        if tuple(self.agent_pos) == self.key_pos and not self.agent_has_key:
            self.agent_has_key = True
            self.grid.set(self.key_pos[0], self.key_pos[1], None)
            step_reward = 0.5  # reward for picking up the key

        curr_pos = self.unwrapped.agent_pos
        achieved_goal = np.array(
            [curr_pos[0], curr_pos[1], 1 if self.agent_has_key else 0], dtype=np.int64)
        desired_goal = np.array(
            [self.goal_pos[0], self.goal_pos[1], 1], dtype=np.int64)

        is_success = np.array_equal(achieved_goal, desired_goal)
        info['is_success'] = is_success

        # check if the agent tried to move forward but stayed in the same position (indicating a collision with a wall)
        is_collision = (action == 2 and np.array_equal(old_pos, curr_pos))
        # add collision info to the info dict for logging in the callback
        info['is_collision'] = is_collision

        obs["achieved_goal"] = achieved_goal
        obs["desired_goal"] = desired_goal

        reward = self.compute_reward(achieved_goal, desired_goal, info)
        reward += step_reward

        goal_obs = {
            "observation": obs["image"],
            "direction": np.array([obs["direction"]], dtype=np.int64),
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }
        return goal_obs, reward, terminated, truncated, info


class PassableKey(Key):
    def can_overlap(self):
        return True


def main():
    env = KeyGoalEnv(render_mode="human")
    manual_control = ManualControl(env)  # enable manual control for testing
    manual_control.start()


if __name__ == "__main__":
    main()
