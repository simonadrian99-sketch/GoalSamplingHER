from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from random import randrange
import gymnasium as gym
import numpy as np
from minigrid.core.world_object import Goal
from minigrid.wrappers import ImgObsWrapper


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=12,
        agent_start_pos=(1, 1),
        agent_view_size=11,
        agent_start_dir=0,
        max_steps: int | None = None,
        GOAL_TYPE="random",  # "randLast" or "fixed" or "random" or "midWalls"
        START_POS_TYPE="random",  # "fixed" or "randFirst" or "random" or "midWalls"
        **kwargs,
    ):
        self.agent_view_size = agent_view_size
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.GOAL_TYPE = GOAL_TYPE
        self.START_POS_TYPE = START_POS_TYPE

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            **kwargs,
        )

        self.action_space = gym.spaces.Discrete(3)
        image_space = self.observation_space.spaces['image']

        self.observation_space = gym.spaces.Dict({
            "observation": image_space,
            "direction": gym.spaces.Box(low=0, high=3, shape=(1,), dtype=np.int64),
            "achieved_goal": gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.int64),
            "desired_goal":  gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.int64),
        })

    @staticmethod
    def _gen_mission():
        return "grand mission"

    # generate the custom grid for the environment

    def _gen_grid(self, width, height):

        self.grid = Grid(width, height)  # Create an empty grid
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall
        if self.START_POS_TYPE == "midWalls" and self.GOAL_TYPE == "midWalls":
            GAP_POS = self.np_random.integers(
                1, height - 2)  # Random gap position
            for i in range(0, height):
                if i != height//2 and i != height//2 - 1 and i != height//2 + 1:
                    self.grid.set((int(width/2)), i, Wall())

        # Place the door and key
      #  self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
      #  self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # Place the agent
        if self.START_POS_TYPE == "randFirst":
            a_pos = (1, self.np_random.integers(1, self.height - 2))
            self.agent_dir = self.agent_start_dir
        elif self.START_POS_TYPE == "fixed":
            a_pos = (1, 1)  # fixed start position
            self.agent_dir = self.agent_start_dir
        elif self.START_POS_TYPE == "random":
            a_pos = (self.np_random.integers(
                1, self.width - 2), self.np_random.integers(1, self.height - 2))
            while a_pos == (self.width//2, 2):
                a_pos = (self.np_random.integers(1, self.width - 2),
                         self.np_random.integers(1, self.height - 2))
            self.agent_dir = self.agent_start_dir
        elif self.START_POS_TYPE == "midWalls":
            a_pos = (self.np_random.integers(
                1, self.width//2 - 1), self.np_random.integers(1, self.height - 2))
            self.agent_dir = self.agent_start_dir

        self.agent_pos = a_pos

        # Place the goal
        if self.GOAL_TYPE == "randLast":
            # random goal position in the last column
            g_pos = (width - 2, self.np_random.integers(1, height - 2))
        elif self.GOAL_TYPE == "fixed":
            g_pos = (width - 2, height - 2)  # fixed goal position
        elif self.GOAL_TYPE == "random":
            g_pos = (self.np_random.integers(
                1, width - 2), self.np_random.integers(1, height - 2))
            while g_pos == (self.width//2, 2):
                g_pos = (self.np_random.integers(
                    1, width - 2), self.np_random.integers(1, height - 2))
        elif self.GOAL_TYPE == "midWalls":
            g_pos = (self.np_random.integers(
                self.width//2 + 1, self.width - 2), self.np_random.integers(1, self.height - 2))

        self.goal_pos = g_pos

        # place the goal object in the grid
        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

        self.mission = "grand mission"  # name of the mission

    def reset(self, **kwargs):

        # reset the environment
        obs, info = super().reset(**kwargs)

        # find the goal position and place the goal object there
        # self.goal_pos = self._find_goal_pos()
        # self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

        # get current agent position (achieved_goal)
        current_pos = self.unwrapped.agent_pos

        # construct the goal-conditioned dict (image, array, array)
        goal_obs = {
            "observation": obs["image"],
            "direction": np.array([obs["direction"]], dtype=np.int64),
            "achieved_goal": np.array(current_pos, dtype=np.int64),
            "desired_goal": np.array(self.goal_pos, dtype=np.int64),
        }

        return goal_obs, info

    def step(self, action):

        old_pos = self.unwrapped.agent_pos
        # step the environment
        obs, _, terminated, truncated, info = super().step(action)

        # get positions
        current_pos = self.unwrapped.agent_pos
        achieved_goal = np.array(current_pos, dtype=np.int64)
        desired_goal = np.array(self.goal_pos, dtype=np.int64)

        # check if the agent tried to move forward but stayed in the same position (indicating a collision with a wall)
        is_collision = (action == 2 and np.array_equal(old_pos, current_pos))
        # add collision info to the info dict for logging in the callback
        info['is_collision'] = is_collision

        # compute reward based on goal achievement
        reward = self.compute_reward(achieved_goal, desired_goal, info)

        # add success info for HER
        is_success = np.array_equal(achieved_goal, desired_goal)
        info['is_success'] = is_success

        # construct the goal-conditioned dict (image, array, array)
        goal_obs = {
            "observation": obs["image"],
            "direction": np.array([obs["direction"]], dtype=np.int64),
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

     #   print(f"Step: {self.step_count}, Action: {action}, Achieved Goal: {achieved_goal}, Desired Goal: {desired_goal}, Reward: {reward}, Terminated: {terminated}")

        return goal_obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):

        # If inputs are batched, compare each pair of achieved and desired goals
        if len(achieved_goal.shape) > 1:
            # Vectorized comparison
            is_success = np.all(achieved_goal == desired_goal, axis=1)
            reward = is_success.astype(np.float32) - 1.0
        else:
            # Single item comparison
            is_success = np.array_equal(achieved_goal, desired_goal)
            reward = 0.0 if is_success else -1.0
        """
        if info is not None:
            if isinstance(info, dict) and info.get("is_collision", False):
                reward -= 0.2  # Penalty for collision
            elif isinstance(info, (list, np.ndarray)):
                collision_mask = np.array(
                    [i.get("is_collision", False) for i in info])
                reward -= (collision_mask.astype(np.float32) * 0.2)
        """
        return reward

    def _find_goal_pos(self):
        # Iterate over the grid to find the object of type 'Goal'
        for i in range(self.unwrapped.grid.width):
            for j in range(self.unwrapped.grid.height):
                obj = self.unwrapped.grid.get(i, j)
                if obj is not None and isinstance(obj, Goal):
                    return (i, j)
        # Fallback if no goal found
        return (self.unwrapped.grid.width - 2, self.unwrapped.grid.height - 2)


def main():
    env = SimpleEnv(render_mode="human")
    manual_control = ManualControl(env)  # enable manual control for testing
    manual_control.start()


if __name__ == "__main__":
    main()
