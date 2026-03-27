import numpy as np
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


class her_novelty_buffer(HerReplayBuffer):
    def __init__(self, *args, grid_size=(10, 10), **kwargs):
        print("!!! NOVELTY BUFFER INITIALISIERT !!!")
        super().__init__(*args, **kwargs)
        self.visit_counts = np.ones(grid_size)
        """self.visit_counts[0, :] = 1e6
        self.visit_counts[9, :] = 1e6
        self.visit_counts[:, 0] = 1e6
        self.visit_counts[:, 9] = 1e6"""
        self.novelty_hits = 0
        self.fallback_hits = 0
        self.debug_counter = 0

    def _sample_goals(self, batch_indices: np.ndarray, env_indices: np.ndarray) -> np.ndarray:

        batch_size = len(batch_indices)
        goal_dim = self.observations["achieved_goal"].shape[-1]
        new_goals = np.zeros((batch_size, goal_dim),
                             dtype=self.observations["achieved_goal"].dtype)

        current_novelty = 0
        current_fallback = 0

        for i in range(batch_size):
            batch_idx = batch_indices[i]
            env_idx = env_indices[i]

            candidates = []
            candidate_coords = []

            for _ in range(5):
                single_goal = super()._sample_goals(
                    np.array([batch_idx]), np.array([env_idx]))[0]
                coords = single_goal.astype(int)
                if 0 < coords[0] < 9 and 0 < coords[1] < 9:
                    candidates.append(single_goal)
                    candidate_coords.append(coords)

            if len(candidates) > 0:
                counts = [self.visit_counts[c[0], c[1]]
                          for c in candidate_coords]
                best_idx = np.argmin(counts)
                new_goals[i] = candidates[best_idx]
                current_novelty += 1
            else:
                new_goals[i] = super()._sample_goals(
                    np.array([batch_idx]), np.array([env_idx]))[0]
                current_fallback += 1

            self.debug_counter += 1
            self.novelty_hits += current_novelty
            self.fallback_hits += current_fallback

            if self.debug_counter % 1000 == 0:
                total = self.novelty_hits + self.fallback_hits
                percentage = (self.novelty_hits / total) * 100
                """print(f"\n[DEBUG BA] Goal Selection Stats:")
                print(f"  - Novelty (8x8): {self.novelty_hits}")
                print(f"  - Fallback: {self.fallback_hits}")
                print(f"  - Erfolgsrate Novelty-Logik: {percentage:.2f}%")
                print(
                    f"  - Aktueller Heatmap-Max-Wert: {np.max(self.visit_counts)}")
                print(
                    f"  - Heatmap Min/Max: {np.min(self.visit_counts)} / {np.max(self.visit_counts)}")
                print(f"  - Top-Links (1,1) Wert: {self.visit_counts[1, 1]}")"""

                # Reset für die nächste Phase
                self.novelty_hits = 0
                self.fallback_hits = 0

        return new_goals

    """def sample(self, batch_size, env):
        return self._sample_transitions(batch_size, maybe_vec_env=env)

    def _sample_transitions(self, batch_size, maybe_vec_env):

       


        obs = {key: np.copy(v) for key, v in obs.items()}
        next_obs = {key: np.copy(v) for key, v in next_obs.items()}

        her_indices = np.where(np.random.rand(batch_size) < self.her_ratio)[0]

        for idx in her_indices:
            ep_idx = episode_indices[idx]
            t_idx = trans_indices[idx]
            future_indices = np.arange(t_idx + 1, self.episode_lengths[ep_idx])

            if len(future_indices) > 0:
                candidate_goals = self.observations['achieved_goal'][ep_idx,
                                                                     future_indices]

                # debug
                print(f"DEBUG: Raw Goals Shape: {candidate_goals.shape}")
                print(f"DEBUG: Erste 3 Goals: {candidate_goals[:3]}")

                coords = candidate_goals.astype(int)

                mask = (coords[:, 0] > 0) & (coords[:, 0] < 9) & \
                       (coords[:, 1] > 0) & (coords[:, 1] < 9)

                if np.any(mask):
                    valid_coords = coords[mask]
                    valid_future_indices = future_indices[mask]

                    counts = self.visit_counts[valid_coords[:,
                                                            0], valid_coords[:, 1]]

                    best_local_idx = np.argmin(counts)
                    best_future_idx = valid_future_indices[best_local_idx]

                    new_goal = self.observations['achieved_goal'][ep_idx,
                                                                  best_future_idx]
                    obs['desired_goal'][idx] = new_goal
                    next_obs['desired_goal'][idx] = new_goal

                    if np.random.rand() < 0.05:
                        print(f"DEBUG Novelty: Aus {len(valid_coords)} Zielen gewählt. "
                              f"Min Visits: {np.min(counts)}, Max Visits: {np.max(counts)}")
                else:
                    if np.random.rand() < 0.05:
                        print(
                            f"DEBUG Novelty: default goal used, da kein gültiges Ziel gefunden wurde.")

                    new_goal = self.observations['achieved_goal'][ep_idx,
                                                                  future_indices[-1]]
                    obs['desired_goal'][idx] = new_goal
                    next_obs['desired_goal'][idx] = new_goal

        rewards = maybe_vec_env.env_method(
            "compute_reward",
            next_obs['achieved_goal'],
            next_obs['desired_goal'],
            [{} for _ in range(batch_size)],
            indices=range(batch_size)
        )
        rewards = np.array(rewards, dtype=np.float32)
        return self._preprocess_samples(obs, next_obs, rewards, episode_indices, trans_indices)"""
