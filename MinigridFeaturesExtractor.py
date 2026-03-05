import torch as torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512) -> None:

        image_space = observation_space.spaces['observation']
        print(f"DEBUG: Image Space Shape: {image_space.shape}")
        n_input_channels = image_space.shape[0]

        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_image = torch.as_tensor(image_space.sample()[None]).float()
            n_flatten = self.cnn(sample_image).shape[1]

            goal_dim = observation_space.spaces['achieved_goal'].shape[0]
            dir_dim = 4  # one-hot encoding for 4 directions (0-3)
            total_concat_size = n_flatten + dir_dim + (goal_dim * 2)

            print(
                f"DEBUG: CNN Output: {n_flatten}, Goals: {goal_dim*2}, Dir: {dir_dim}")
            print(f"DEBUG: Total Linear Input: {total_concat_size}")

        self.linear = nn.Sequential(
            nn.Linear(total_concat_size, features_dim), nn.ReLU())

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:

        img = observations["observation"]
        if img.shape[-1] == 3:
            # Convert to (batch, channels, height, width)
            img = img.permute(0, 3, 1, 2)

        # process the image through CNN
        image_features = self.cnn(img.float() / 255.0)

        # get the direction (ensure it's float for the linear layer)
       # direction = observations["direction"].float() / 3.0  # normalize direction to [0, 1]

        # one-hot encode the direction
        dir_indices = observations["direction"].long().view(-1)
        direction_one_hot = F.one_hot(dir_indices, num_classes=4).float()

        # get the goals (ensure they are float for the linear layer)
        achieved_goal = observations["achieved_goal"].float()
        desired_goal = observations["desired_goal"].float()

        # concatenate everything (Image_Features, direction, Achieved_Pos, Desired_Pos)
        combined = torch.cat(
            [image_features, direction_one_hot, achieved_goal, desired_goal], dim=1)

        if not hasattr(self, "_printed"):
            print(f"DEBUG: Image features shape: {image_features.shape}")
            print(f"DEBUG: One-hot direction shape: {direction_one_hot.shape}")
            print(f"DEBUG: Combined shape: {combined.shape}")
            self._printed = True

        return self.linear(combined)
