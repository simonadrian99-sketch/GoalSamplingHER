import itertools

from ActionLoggerCallback import ActionLoggerCallback
from CollisionLoggerCallback import CollisionLoggerCallback
from HeatmapLoggerCallback import HeatmapLoggerCallback
import minigrid
from SimpleEnv import SimpleEnv
from MinigridFeaturesExtractor import MinigridFeaturesExtractor
import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper, DictObservationSpaceWrapper
from stable_baselines3 import A2C, DQN, HerReplayBuffer
from her_novelty_buffer import her_novelty_buffer as HerNoveltyBuffer
from her_novelty_buffer import her_passive_logger_buffer as HerPassiveLoggerBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
import os
import time


LOG_ROOT = "logs"
MODEL_ROOT = "models"
ALGORITHM_NAME = "DQN+HER"
GOAL_SELECTION_STRATEGY = "episode"  # "final", "episode", "random", "future"
NOVELTY_ON = False
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")


policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=512),
)

if NOVELTY_ON:
    replay_buffer_class = HerNoveltyBuffer
    replay_buffer_kwargs = dict(
        grid_size=(10, 10),
        n_sampled_goal=8,
        goal_selection_strategy=GOAL_SELECTION_STRATEGY,
    )
else:
    replay_buffer_class = HerPassiveLoggerBuffer
    replay_buffer_kwargs = dict(
        n_sampled_goal=8,
        goal_selection_strategy=GOAL_SELECTION_STRATEGY,
    )


# create environment
env = SimpleEnv(render_mode="rgb_array")

# create model path based on environment size and goal selection strategy
if NOVELTY_ON == False:
    # models_dir = f"models/DQN+HER_{env.width-2}x{env.height-2}_{env.START_POS_TYPE}_{GOAL_SELECTION_STRATEGY}_{env.GOAL_TYPE}_{TIMESTAMP}"
    models_dir = os.path.join(MODEL_ROOT, ALGORITHM_NAME, f"{env.width-2}x{env.height-2}_{env.START_POS_TYPE}",
                              GOAL_SELECTION_STRATEGY, env.GOAL_TYPE, TIMESTAMP)
    LOG_DIR = os.path.join(
        LOG_ROOT, f"{env.width-2}x{env.height-2}_{env.START_POS_TYPE}", ALGORITHM_NAME, GOAL_SELECTION_STRATEGY
    )
else:
    LOG_DIR = os.path.join(
        LOG_ROOT, f"{env.width-2}x{env.height-2}_{env.START_POS_TYPE}", ALGORITHM_NAME, "novelty"
    )
    # models_dir = f"models/DQN+HER_{env.width-2}x{env.height-2}_{env.START_POS_TYPE}_novelty_{env.GOAL_TYPE}_{TIMESTAMP}"
    models_dir = os.path.join(MODEL_ROOT, ALGORITHM_NAME,
                              f"{env.width-2}x{env.height-2}_{env.START_POS_TYPE}", "novelty", env.GOAL_TYPE, TIMESTAMP)


# creating model and log directory
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(LOG_ROOT):
    os.makedirs(LOG_ROOT)


# initialize model
model = DQN(
    "MultiInputPolicy",
    env,
    replay_buffer_class=replay_buffer_class,
    replay_buffer_kwargs=replay_buffer_kwargs,
    policy_kwargs=policy_kwargs,
    batch_size=512,
    buffer_size=100000,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="cuda",
    learning_starts=2000,
    learning_rate=1e-4,
    target_update_interval=1000,
    max_grad_norm=1
)

TIMESTEPS = 30000
RUN_NAME = f"run_{TIMESTAMP}"

callbacks = []
action_cb = ActionLoggerCallback()
callbacks.append(action_cb)
collision_cb = CollisionLoggerCallback()
callbacks.append(collision_cb)
heatmap_cb = HeatmapLoggerCallback(model.replay_buffer)
callbacks.append(heatmap_cb)


# training loop
for i in range(1, 11):  # 10 Meilensteine à 30k Schritte
    try:
        # train model 30k steps and log to tensorboard
        """print(RUN_NAME)
        print("--- DIAGNOSE START ---")
        print(f"Learning Starts: {model.learning_starts}")
        print(f"Train Freq: {model.train_freq}")
        print(f"Total Timesteps: {TIMESTEPS}")
        print(f"Buffer Typ: {type(model.replay_buffer)}")
        print("--- DIAGNOSE ENDE ---")"""
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            callback=callbacks,
            tb_log_name=RUN_NAME
        )

        # save model checkpoint every 10k steps
        save_path = f"{models_dir}/{TIMESTEPS * i}"
        model.save(save_path)
        np.save(f"{LOG_DIR}/{RUN_NAME}_0/heatmap_{TIMESTEPS * i}.npy",
                model.replay_buffer.visit_counts)

        print(
            f"Meilenstein erreicht: {TIMESTEPS * i} Schritte. Modell gespeichert unter: {save_path}")

    except KeyboardInterrupt:
        # handle manual interruption (Ctrl+C) to stop training and save the final model
        print("Training wird manuell gestoppt. Speichere finalen Stand...")
        model.save(f"{models_dir}/final")
        break
