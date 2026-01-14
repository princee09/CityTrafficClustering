import numpy as np
from src.q_learning_agent import QLearningAgent


def test_basic_learn_and_choose():
    agent = QLearningAgent(
        n_states=2,
        n_queue_bins=5,
        n_actions=4,
        lr=0.5,
        gamma=0.9,
        epsilon=0.5,
        decay=0.99,
    )

    s = {"cluster": 0, "queue_ns": 2, "queue_ew": 3}
    ns = {"cluster": 0, "queue_ns": 1, "queue_ew": 1}

    # copy initial q for comparison
    idx_s = agent.state_to_index(s)
    before_q = agent.q[idx_s].copy()

    # choose action
    a = agent.choose_action(s)
    assert 0 <= a < agent.n_actions

    # learn with a positive reward
    agent.learn(s, a, reward=5.0, next_state=ns)

    after_q = agent.q[idx_s]
    # q-value for chosen action should have changed
    assert not np.allclose(before_q, after_q)

    # epsilon should have decayed (or be at min)
    assert agent.epsilon <= 0.5


if __name__ == "__main__":
    test_basic_learn_and_choose()
    print("test passed")
