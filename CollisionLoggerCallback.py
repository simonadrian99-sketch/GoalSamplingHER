from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CollisionLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CollisionLoggerCallback, self).__init__(verbose)
        self.collision_counts = []

    def _on_step(self) -> bool:
        # Wir greifen auf die 'infos' der aktuellen Schritte zu
        # Da wir oft mehrere Envs (Vectorized) haben, loopen wir kurz drüber
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("is_collision", False):
                self.collision_counts.append(1)
            else:
                self.collision_counts.append(0)
        
        # Alle 1000 Schritte loggen wir den Durchschnitt (Kollisionsrate)
        if self.n_calls % 1000 == 0 and len(self.collision_counts) > 0:
            avg_collision = np.mean(self.collision_counts)
            self.logger.record("env/collision_rate", avg_collision)
            # Liste leeren für das nächste Intervall
            self.collision_counts = []
            
        return True