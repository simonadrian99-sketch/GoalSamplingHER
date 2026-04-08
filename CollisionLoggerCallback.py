from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class CollisionLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CollisionLoggerCallback, self).__init__(verbose)
        self.collision_counts = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("is_collision", False):
                self.collision_counts.append(1)
            else:
                self.collision_counts.append(0)

        if self.n_calls % 1000 == 0 and len(self.collision_counts) > 0:
            avg_collision = np.mean(self.collision_counts)
            self.logger.record("env/collision_rate", avg_collision)
            self.collision_counts = []

        return True
