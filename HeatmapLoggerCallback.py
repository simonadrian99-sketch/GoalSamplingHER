from stable_baselines3.common.callbacks import BaseCallback


class HeatmapLoggerCallback(BaseCallback):
    def __init__(self, buffer, verbose=0):
        super(HeatmapLoggerCallback, self).__init__(verbose)
        self.buffer = buffer

    def _on_step(self) -> bool:

        new_obs = self.locals['new_obs']
        pos = new_obs['achieved_goal'][0]

        if hasattr(pos, "cpu"):
            pos = pos.cpu().detach().numpy()

        try:
            x, y = int(round(float(pos[0]))), int(round(float(pos[1])))
            if len(pos) == 3:
                has_key = int(round(float(pos[2])))
                if 1 <= x <= 10 and 1 <= y <= 10:
                    self.buffer.visit_counts[x, y, has_key] += 1
            else:
                if 1 <= x <= 10 and 1 <= y <= 10:
                    self.buffer.visit_counts[x, y] += 1

        except (IndexError, ValueError, TypeError) as e:
            if self.n_calls % 1000 == 0:
                print(f"Error occurred while tracking position {pos}: {e}")

        return True
