import numpy as np
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


class her_novelty_buffer(HerReplayBuffer):
    def __init__(self, *args, grid_size=(12, 12), **kwargs):
        print("!!! NOVELTY BUFFER INITIALISIERT !!!")
        super().__init__(*args, **kwargs)

        goal_shape = self.observation_space.spaces['achieved_goal'].shape[0]
        if goal_shape == 3:
            # 3D Heatmap: [x, y, has_key]
            self.visit_counts = np.ones((*grid_size, 2))
        else:
            # 2D Heatmap: [x, y]
            self.visit_counts = np.ones(grid_size)
        self.novelty_hits = 0
        self.fallback_hits = 0
        self.debug_counter = 0

    def _sample_goals(self, batch_indices: np.ndarray, env_indices: np.ndarray) -> np.ndarray:

        batch_size = len(batch_indices)
        goal_dim = self.observations["achieved_goal"].shape[-1]

        n_candidates = 5
        rep_batch_indices = np.repeat(batch_indices, n_candidates)
        rep_env_indices = np.repeat(env_indices, n_candidates)

        all_candidates = super()._sample_goals(rep_batch_indices, rep_env_indices)

        candidate_grid = all_candidates.reshape(
            batch_size, n_candidates, goal_dim)

        """DEBUG
        if self.novelty_hits % 100 == 0:
            print(f"\n--- Debugging Novelty Sampling ---")
            print(f"Original Batch Index: {batch_indices[0]}")
            print(
                f"Die 5 gezogenen Kandidaten (Koordinaten):\n{candidate_grid[0]}")
            unique_candidates = np.unique(candidate_grid[0], axis=0)
            print(
                f"Anzahl unterschiedlicher Kandidaten: {len(unique_candidates)}/5")
        for j in range(n_candidates):
            c = candidate_grid[0][j].astype(int)
            count = self.visit_counts[c[0], c[1], c[2]
                                      ] if goal_dim == 3 else self.visit_counts[c[0], c[1]]
            print(f"Kandidat {j}: Pos {c} -> Visits: {count}")
        DEBUG ENDE"""

        new_goals = np.zeros((batch_size, goal_dim),
                             dtype=all_candidates.dtype)
        coords = candidate_grid.astype(int)

        mask = (coords[:, :, 0] >= 1) & (coords[:, :, 0] <= 10) & \
               (coords[:, :, 1] >= 1) & (coords[:, :, 1] <= 10)

        current_novelty = 0
        current_fallback = 0

        for i in range(batch_size):
            valid_mask = mask[i]

            if np.any(valid_mask):
                valid_candidates = candidate_grid[i][valid_mask]
                valid_coords = coords[i][valid_mask]

                if goal_dim == 3:
                    counts = self.visit_counts[valid_coords[:, 0],
                                               valid_coords[:, 1],
                                               valid_coords[:, 2]]
                else:
                    counts = self.visit_counts[valid_coords[:, 0],
                                               valid_coords[:, 1]]
                best_idx = np.argmin(counts)
                new_goals[i] = valid_candidates[best_idx]
                self.novelty_hits += 1
            else:
                new_goals[i] = candidate_grid[i][0]
                self.fallback_hits += 1

        return new_goals


class her_passive_logger_buffer(HerReplayBuffer):
    def __init__(self, *args, grid_size=(12, 12), **kwargs):
        super().__init__(*args, **kwargs)
        goal_shape = self.observation_space.spaces['achieved_goal'].shape[0]

        if goal_shape == 3:
            self.visit_counts = np.ones((*grid_size, 2), dtype=np.float32)
        else:
            self.visit_counts = np.ones(grid_size, dtype=np.float32)
