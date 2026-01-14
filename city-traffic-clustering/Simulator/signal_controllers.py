"""
Signal controllers: Fixed-Time and Agent-Based
"""


class FixedTimeController:
    def __init__(self, signal_timings_df):
        self.timings = signal_timings_df
        # Use average timings from Bangalore data
        self.standard_timing = 0  # Standard action

    def get_action(self, cluster):
        """Always return standard timing (action 0)"""
        return 0


class AgentController:
    def __init__(self, agent, predictor):
        self.agent = agent
        self.predictor = predictor

    def get_action(self, cluster, queue_ns, queue_ew, prediction):
        """Get action from Q-learning agent with prediction"""
        # Discretize queue lengths
        queue_state = min(9, (queue_ns + queue_ew) // 5)

        # Agent chooses action
        state_idx = cluster * 10 + queue_state
        action = self.agent.choose_action(state_idx, prediction)

        return action


# =============================================================================
# FILE 4: q_learning_agent.py
# =============================================================================
"""
Q-Learning Agent with Predictive Hybrid Approach
"""


class QLearningAgent:
    def __init__(
        self,
        n_states=4,
        n_queue_bins=10,
        n_actions=5,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.n_states = n_states
        self.n_queue_bins = n_queue_bins
        self.n_actions = n_actions
        self.state_size = n_states * n_queue_bins

        # Q-table
        self.q_table = np.zeros((self.state_size, n_actions))

        # Hyperparameters
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Prediction weights
        self.prediction_weight = 0.3

    def choose_action(self, state, prediction=None):
        """
        Choose action using hybrid approach:
        - Q-learning (exploitation/exploration)
        - Prediction-based bias
        """
        # Epsilon-greedy with prediction bias
        if np.random.random() < self.epsilon:
            # Exploration: random action
            action = np.random.randint(self.n_actions)
        else:
            # Exploitation: best Q-value with prediction adjustment
            q_values = self.q_table[state].copy()

            # Apply prediction bias if available
            if prediction is not None:
                # If predicting higher congestion, bias toward extending green
                if prediction > state // self.n_queue_bins:
                    q_values[1] += self.prediction_weight  # Extend NS
                    q_values[2] += self.prediction_weight  # Extend EW
                    q_values[4] += self.prediction_weight  # Increase cycle
                # If predicting lower congestion, bias toward reducing cycle
                elif prediction < state // self.n_queue_bins:
                    q_values[3] += self.prediction_weight  # Reduce cycle

            action = np.argmax(q_values)

        return action

    def learn(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule"""
        current_state = state["cluster"] * self.n_queue_bins + min(
            self.n_queue_bins - 1, (state["queue_ns"] + state["queue_ew"]) // 5
        )
        next_state_idx = next_state["cluster"] * self.n_queue_bins + min(
            self.n_queue_bins - 1,
            (next_state["queue_ns"] + next_state["queue_ew"]) // 5,
        )

        # Q-learning update
        current_q = self.q_table[current_state][action]
        max_next_q = np.max(self.q_table[next_state_idx])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)

        self.q_table[current_state][action] = new_q

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filename="q_table.npy"):
        """Save Q-table"""
        np.save(filename, self.q_table)
        print(f"✓ Q-table saved to {filename}")

    def load_model(self, filename="q_table.npy"):
        """Load Q-table"""
        self.q_table = np.load(filename)
        print(f"✓ Q-table loaded from {filename}")
