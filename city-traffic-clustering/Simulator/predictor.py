"""
Traffic State Predictor using Markov Model and Trend Analysis
"""

import numpy as np


class TrafficPredictor:
    def __init__(self, transition_matrix, traffic_data):
        self.transition_matrix = transition_matrix
        self.traffic_data = traffic_data
        self.history = []

    def predict(self, current_cluster, queue_ns, queue_ew, avg_speed):
        """
        Predict next traffic state using:
        1. Markov transition probabilities
        2. Current queue/speed trends
        """
        # Markov-based prediction
        markov_pred = np.argmax(self.transition_matrix[current_cluster])

        # Trend-based adjustment
        total_queue = queue_ns + queue_ew

        # High queue + low speed = likely moving toward congestion
        if total_queue > 20 and avg_speed < 30:
            trend_pred = min(3, current_cluster + 1)
        # Low queue + high speed = likely moving toward free flow
        elif total_queue < 5 and avg_speed > 50:
            trend_pred = max(0, current_cluster - 1)
        else:
            trend_pred = current_cluster

        # Weighted combination (70% Markov, 30% Trend)
        if abs(markov_pred - trend_pred) <= 1:
            prediction = markov_pred
        else:
            prediction = int(0.7 * markov_pred + 0.3 * trend_pred)

        # Store history
        self.history.append(
            {
                "current": current_cluster,
                "predicted": prediction,
                "markov": markov_pred,
                "trend": trend_pred,
            }
        )

        return prediction
