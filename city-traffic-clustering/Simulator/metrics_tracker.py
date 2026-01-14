"""
Performance metrics tracker
"""

import numpy as np


class MetricsTracker:
    def __init__(self):
        self.metrics = {
            "wait_times": [],
            "queue_lengths": [],
            "throughput": [],
            "rewards": [],
        }

    def update(self, wait_time, queue_length, throughput, reward=None):
        """Update metrics"""
        self.metrics["wait_times"].append(wait_time)
        self.metrics["queue_lengths"].append(queue_length)
        self.metrics["throughput"].append(throughput)
        if reward is not None:
            self.metrics["rewards"].append(reward)

    def get_summary(self):
        """Get summary statistics"""
        return {
            "avg_wait": np.mean(self.metrics["wait_times"]),
            "avg_queue": np.mean(self.metrics["queue_lengths"]),
            "total_throughput": np.sum(self.metrics["throughput"]),
            "avg_reward": np.mean(self.metrics["rewards"])
            if self.metrics["rewards"]
            else 0,
        }
