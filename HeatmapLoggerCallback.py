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
            if 1 <= x <= 8 and 1 <= y <= 8:
                self.buffer.visit_counts[x, y] += 1

                """if self.n_calls % 1000 == 0:
                    print(
                        f"[Heatmap] Pos: {x},{y} | Count: {self.buffer.visit_counts[x, y]}")
            else:
                if self.n_calls % 5000 == 0:
                    print(f"[Heatmap] Ignoriere Randposition: {x}, {y}")"""
        except (IndexError, ValueError, TypeError) as e:
            print(f"Error occurred while tracking position {pos}: {e}")

        return True
