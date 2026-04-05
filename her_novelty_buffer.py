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


class her_passive_logger_buffer(HerReplayBuffer):
    def __init__(self, *args, grid_size=(10, 10), **kwargs):
        super().__init__(*args, **kwargs)
        self.visit_counts = np.ones(grid_size, dtype=np.float32)
