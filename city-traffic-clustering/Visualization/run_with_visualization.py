"""
Run simulation with live Pygame visualization
"""


def run_visual_simulation(mode="agent", max_steps=2000):
    """
    Run simulation with live visualization

    Args:
        mode: 'fixed' or 'agent'
        max_steps: Maximum simulation steps
    """
    print(f"\n{'=' * 60}")
    print(f"Starting VISUAL simulation in {mode.upper()} mode")
    print(f"{'=' * 60}\n")

    import os
    import sys

    # Ensure project root (parent of src) is on sys.path
    def find_project_root(start=os.getcwd()):
        cur = os.path.abspath(start)
        while True:
            if os.path.isdir(os.path.join(cur, "src")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                raise FileNotFoundError("Could not find project root containing 'src'")
            cur = parent

    proj_root = find_project_root()
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    from src.traffic_environment import TrafficEnvironment
    from src.signal_controllers import FixedTimeController, AgentController
    from src.q_learning_agent import QLearningAgent
    from src.predictor import TrafficPredictor
    # Visualization module lives in this folder (module file Visualization/Visualization.py)
    from Visualization.Visualization import TrafficVisualizer
    import pandas as pd
    import numpy as np

    # Load data
    # Use processed data paths inside project data/processed
    scenarios_path = os.path.join(
        proj_root, "data", "processed", "traffic_scenarios_from_clusters_k4.csv"
    )
    traffic_data_path = os.path.join(
        proj_root, "data", "processed", "bangalore_traffic_with_clusters_k4.csv"
    )
    signal_timings_path = os.path.join(
        proj_root, "data", "processed", "Bangalore_Signal_Timing_2025-11-30_19-13.csv"
    )

    # Verify files exist before reading
    for pth in (scenarios_path, traffic_data_path, signal_timings_path):
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Required data file not found: {pth}")

    scenarios = pd.read_csv(scenarios_path)
    traffic_data = pd.read_csv(traffic_data_path)
    signal_timings = pd.read_csv(signal_timings_path)

    # Calculate transitions
    clusters = traffic_data["cluster"].values
    n_clusters = 4
    transition_counts = np.zeros((n_clusters, n_clusters))
    for i in range(len(clusters) - 1):
        transition_counts[clusters[i]][clusters[i + 1]] += 1
    transition_matrix = np.zeros_like(transition_counts)
    for i in range(n_clusters):
        total = transition_counts[i].sum()
        if total > 0:
            transition_matrix[i] = transition_counts[i] / total
        else:
            transition_matrix[i] = 1.0 / n_clusters

    # Initialize components
    env = TrafficEnvironment(scenarios, traffic_data)
    predictor = TrafficPredictor(transition_matrix, traffic_data)
    visualizer = TrafficVisualizer()

    if mode == "fixed":
        controller = FixedTimeController(signal_timings)
    else:
        agent = QLearningAgent(n_states=4, n_queue_bins=10, n_actions=5)
        controller = AgentController(agent, predictor)

    # Simulation loop
    state = env.reset()
    running = True
    step = 0
    prediction = 0
    last_action = 0
    q_values = None

    print("Controls:")
    print("  ESC - Exit simulation")
    print("\nSimulation running...")

    while running and step < max_steps:
        # Handle events
        running = visualizer.handle_events()

        # Get current state
        cluster = env.current_cluster
        queue_ns = env.get_queue_length("ns")
        queue_ew = env.get_queue_length("ew")

        # Predict next state
        prediction = predictor.predict(cluster, queue_ns, queue_ew, env.avg_speed)

        # Controller decides action
        if mode == "fixed":
            action = controller.get_action(cluster)
        else:
            action = controller.get_action(cluster, queue_ns, queue_ew, prediction)
            # Get Q-values for visualization (agent stores table in .q)
            state_idx = cluster * 10 + min(9, (queue_ns + queue_ew) // 5)
            q_values = getattr(controller.agent, "q", None)
            if q_values is not None:
                q_values = q_values[state_idx]

        last_action = action

        # Environment step
        next_state, reward, done, info = env.step(action)

        # Agent learns (only in agent mode)
        if mode == "agent":
            controller.agent.learn(state, action, reward, next_state)

        # Render
        visualizer.render(env, mode, prediction, last_action, q_values)

        state = next_state
        step += 1

        # Print progress every 100 steps
        if step % 100 == 0:
            print(
                f"Step {step}: Wait={info['avg_wait_time']:.1f}s, "
                f"Queue={info['total_queue']}, Through={info['vehicles_passed']}"
            )

    visualizer.close()
    print(f"\n✓ Visual simulation completed ({step} steps)")

    # Save agent if trained
    if mode == "agent":
        controller.agent.save_model("visual_trained_q_agent.npy")


if __name__ == "__main__":
    # Default: run agent-based visual simulation for a short demo
    run_visual_simulation(mode="agent", max_steps=500)
