import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states=4,
        n_queue_bins=10,
        n_actions=5,
        lr=0.1,
        gamma=0.95,
        epsilon=1.0,
        decay=0.995,
        min_epsilon=0.01,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_queue_bins = n_queue_bins
        self.q = np.zeros((n_states * n_queue_bins, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.pred_weight = 0.3

    def state_to_index(self, state):
        """Convert a state dict or index to the Q-table row index.

        Expected state dict keys: 'cluster', 'queue_ns', 'queue_ew'.
        If an integer index is passed, it will be returned after basic bounds checks.
        """
        # allow passing an index directly
        if isinstance(state, int):
            idx = int(state)
            max_idx = self.q.shape[0] - 1
            if idx < 0:
                return 0
            return min(idx, max_idx)

        cluster = int(state.get("cluster", 0))
        # clamp cluster into valid range
        cluster = max(0, min(cluster, max(0, self.n_states - 1)))
        total_q = int(state.get("queue_ns", 0)) + int(state.get("queue_ew", 0))
        # simple fixed binning: adjust if your queues use a different scale
        bin_idx = max(0, min(self.n_queue_bins - 1, total_q // 5))
        return cluster * self.n_queue_bins + bin_idx

    def choose_action(self, state=None, prediction=None, action_prior=None):
        """Epsilon-greedy action selection.

        - state: dict or integer index. If None, random action is returned when exploring.
        - prediction: kept for backward compatibility (deprecated).
        - action_prior: optional iterable of length n_actions with priors to bias exploitation.
        """
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        # Resolve state to index
        idx = self.state_to_index(state) if state is not None else 0
        qvals = self.q[idx].copy()

        # Backward-compatible small bias for prediction (deprecated usage)
        if prediction is not None:
            # Keep previous behavior but safer: only apply if indices exist
            if self.n_actions >= 3:
                qvals[1] += self.pred_weight
                qvals[2] += self.pred_weight

        # Apply action_prior if provided (preferred over opaque prediction)
        if action_prior is not None:
            ap = np.asarray(action_prior)
            if ap.shape[0] == self.n_actions:
                qvals = qvals + ap * self.pred_weight

        return int(np.argmax(qvals))

    def learn(self, state, action, reward, next_state):
        # Map states to indices with safety checks
        s = self.state_to_index(state)
        ns = self.state_to_index(next_state)

        # Basic bounds checks for action
        if action < 0 or action >= self.n_actions:
            raise IndexError(
                f"Action {action} out of bounds for n_actions={self.n_actions}"
            )

        td_target = reward + self.gamma * np.max(self.q[ns])
        td_error = td_target - self.q[s, action]
        self.q[s, action] += self.lr * td_error

        # Epsilon decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def save_model(self, path):
        np.save(path, self.q)

    def load_model(self, path):
        self.q = np.load(path)
