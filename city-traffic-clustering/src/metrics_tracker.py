import numpy as np


class MetricsTracker:
    def __init__(self):
        self.wait = []
        self.queue = []
        self.through = []
        self.reward = []

    def update(self, w, q, t, r=None):
        self.wait.append(w)
        self.queue.append(q)
        self.through.append(t)
        if r is not None:
            self.reward.append(r)

    def get_summary(self):
        return {
            "avg_wait": np.mean(self.wait),
            "avg_queue": np.mean(self.queue),
            "throughput": np.sum(self.through),
            "avg_reward": np.mean(self.reward) if self.reward else 0,
        }
