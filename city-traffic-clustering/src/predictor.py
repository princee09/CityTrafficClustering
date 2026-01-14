import numpy as np


class TrafficPredictor:
    def __init__(self, transition_matrix):
        self.tm = transition_matrix

    def predict(self, current_cluster, queue_ns, queue_ew, avg_speed):
        markov_pred = np.argmax(self.tm[current_cluster])

        total_queue = queue_ns + queue_ew

        if total_queue > 20 and avg_speed < 30:
            trend_pred = min(3, current_cluster + 1)
        elif total_queue < 5 and avg_speed > 50:
            trend_pred = max(0, current_cluster - 1)
        else:
            trend_pred = current_cluster

        if abs(markov_pred - trend_pred) <= 1:
            return markov_pred
        else:
            return int(0.7 * markov_pred + 0.3 * trend_pred)
