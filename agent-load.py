import minigrid
import pygame
from SimpleEnv import SimpleEnv
from MinigridFeaturesExtractor import MinigridFeaturesExtractor
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import DQN
import os
import numpy as np


# create environment
env = SimpleEnv(render_mode="human")

models_dir = "models/DQN+HER_10x10_future_randLast_20260327-134343"
model_path = f"{models_dir}/50000.zip"

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=512),
)


# load model from saved checkpoint
model = DQN.load(model_path, env=env, custom_objects={
                 "policy_kwargs": policy_kwargs}, device="cuda")
print(f"Model loaded from {model_path}")
episodes = 10

# visualize the agent's behavior for a few episodes
for ep in range(episodes):
    print(f"Episode: {ep+1}")
    obs, info = env.reset()
    done = False
    truncated = False
    step_count = 0

    print(f"Ziel gesetzt auf: {obs['desired_goal']}")

    while not done and not truncated:
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print(f"-> Episode {ep+1} übersprungen!")
                    done = True
                    break
            if event.type == pygame.QUIT:
                env.close()
                exit()
        if done:
            break
      #  print(obs["direction"], obs["direction"].dtype)
        obs['direction'] = np.array(obs['direction'], dtype=np.int64)
        obs['achieved_goal'] = np.array(obs['achieved_goal'], dtype=np.float32)
        obs['desired_goal'] = np.array(obs['desired_goal'], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)

        current = obs['achieved_goal']
        target = obs['desired_goal']
        direction = obs['direction'][0]
        dir_map = {0: "> (Ost)", 1: "v (Süd)", 2: "< (West)", 3: "^ (Nord)"}
        dir_text = dir_map.get(int(direction), "Unbekannt")
        print(f"Pos: {current} | Ziel: {target} | Blick: {dir_text}", end="\r")

        obs, reward, done, truncated, info = env.step(action)
        step_count += 1
        if step_count >= 200:  # only run for a limited number of steps to avoid long episodes
            print("Reached step limit, ending episode.")
            break

env.close()
