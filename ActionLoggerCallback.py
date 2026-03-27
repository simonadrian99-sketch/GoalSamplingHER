from stable_baselines3.common.callbacks import BaseCallback


class ActionLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ActionLoggerCallback, self).__init__(verbose)
        self.action_counts = {}

    def _on_step(self) -> bool:
        last_actions = self.locals['actions'].flatten()
        for action in last_actions:
            action = int(action)
            self.action_counts[action] = self.action_counts.get(action, 0) + 1

        # Alle 1000 Schritte loggen wir die Verteilung
        if self.n_calls % 1000 == 0:
            total = sum(self.action_counts.values())
            for action, count in self.action_counts.items():
                # Anteil der Aktion in Prozent loggen
                percentage = (count / total) * 100
                self.logger.record(
                    f"actions/action_{action}_percent", percentage)

            # Zähler zurücksetzen für das nächste Intervall
            self.action_counts = {}

        return True
