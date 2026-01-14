"""
Main simulation runner for traffic signal control comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os


class MainSimulation:
    def __init__(self, data_path="./"):
        """Initialize simulation with data files"""
        self.data_path = data_path
        self.load_data()

    def load_data(self):
        """Load traffic data and scenarios"""
        print("Loading data files...")

        # Load traffic scenarios from clusters
        self.scenarios = pd.read_csv(
            f"{self.data_path}traffic_scenarios_from_clusters_k4.csv"
        )
        print(f"✓ Loaded {len(self.scenarios)} traffic scenarios")

        # Load historical traffic data with clusters
        self.traffic_data = pd.read_csv(
            f"{self.data_path}bangalore_traffic_with_clusters_k4.csv"
        )
        print(f"✓ Loaded {len(self.traffic_data)} historical traffic records")

        # Load signal timing data
        self.signal_timings = pd.read_csv(
            f"{self.data_path}Bangalore_Signal_Timing_20251130_1913.csv"
        )
        print(f"✓ Loaded {len(self.signal_timings)} signal timing records")

        # Calculate cluster transition probabilities for prediction
        self.transition_matrix = self.calculate_transitions()
        print("✓ Calculated cluster transition matrix")

    def calculate_transitions(self):
        """Calculate transition probabilities between clusters"""
        clusters = self.traffic_data["cluster"].values
        n_clusters = 4
        transition_counts = np.zeros((n_clusters, n_clusters))

        for i in range(len(clusters) - 1):
            current = clusters[i]
            next_state = clusters[i + 1]
            transition_counts[current][next_state] += 1

        # Normalize to probabilities
        transition_probs = np.zeros_like(transition_counts)
        for i in range(n_clusters):
            total = transition_counts[i].sum()
            if total > 0:
                transition_probs[i] = transition_counts[i] / total
            else:
                transition_probs[i] = 1.0 / n_clusters  # Uniform if no data

        return transition_probs

    def run_simulation(self, mode="fixed", episodes=5, steps_per_episode=1000):
        """
        Run traffic simulation

        Args:
            mode: 'fixed' or 'agent'
            episodes: Number of episodes to run
            steps_per_episode: Steps per episode
        """
        print(f"\n{'=' * 60}")
        print(f"Running {mode.upper()} simulation...")
        print(f"Episodes: {episodes}, Steps per episode: {steps_per_episode}")
        print(f"{'=' * 60}\n")

        from traffic_environment import TrafficEnvironment
        from signal_controllers import FixedTimeController, AgentController
        from q_learning_agent import QLearningAgent
        from predictor import TrafficPredictor
        from metrics_tracker import MetricsTracker

        # Initialize components
        env = TrafficEnvironment(self.scenarios, self.traffic_data)
        predictor = TrafficPredictor(self.transition_matrix, self.traffic_data)
        metrics = MetricsTracker()

        if mode == "fixed":
            controller = FixedTimeController(self.signal_timings)
        else:
            agent = QLearningAgent(
                n_states=4,  # 4 clusters
                n_queue_bins=10,
                n_actions=5,
            )
            controller = AgentController(agent, predictor)

        # Training/Simulation loop
        all_metrics = []

        for episode in range(episodes):
            state = env.reset()
            episode_metrics = {
                "episode": episode,
                "total_wait": 0,
                "total_queue": 0,
                "throughput": 0,
                "steps": 0,
            }

            for step in range(steps_per_episode):
                # Get current state
                cluster = env.current_cluster
                queue_ns = env.get_queue_length("ns")
                queue_ew = env.get_queue_length("ew")

                # Predict next state
                prediction = predictor.predict(
                    cluster, queue_ns, queue_ew, env.avg_speed
                )

                # Controller decides action
                if mode == "fixed":
                    action = controller.get_action(cluster)
                else:
                    action = controller.get_action(
                        cluster, queue_ns, queue_ew, prediction
                    )

                # Environment step
                next_state, reward, done, info = env.step(action)

                # Agent learns (only in agent mode)
                if mode == "agent":
                    controller.agent.learn(state, action, reward, next_state)

                # Track metrics
                episode_metrics["total_wait"] += info["avg_wait_time"]
                episode_metrics["total_queue"] += info["total_queue"]
                episode_metrics["throughput"] += info["vehicles_passed"]
                episode_metrics["steps"] += 1

                state = next_state

                if done:
                    break

            # Average metrics
            episode_metrics["avg_wait"] = (
                episode_metrics["total_wait"] / episode_metrics["steps"]
            )
            episode_metrics["avg_queue"] = (
                episode_metrics["total_queue"] / episode_metrics["steps"]
            )
            all_metrics.append(episode_metrics)

            print(
                f"Episode {episode + 1}/{episodes} - "
                f"Avg Wait: {episode_metrics['avg_wait']:.2f}s, "
                f"Avg Queue: {episode_metrics['avg_queue']:.1f}, "
                f"Throughput: {episode_metrics['throughput']}"
            )

        # Save results
        results_df = pd.DataFrame(all_metrics)
        filename = f"results_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\n✓ Results saved to {filename}")

        return results_df, controller if mode == "agent" else None

    def compare_modes(self):
        """Run both modes and compare results"""
        print("\n" + "=" * 60)
        print("COMPARING FIXED-TIME vs AGENT-BASED CONTROL")
        print("=" * 60)

        # Run fixed-time
        fixed_results, _ = self.run_simulation(
            "fixed", episodes=3, steps_per_episode=500
        )

        # Run agent-based
        agent_results, trained_agent = self.run_simulation(
            "agent", episodes=3, steps_per_episode=500
        )

        # Compare
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)

        fixed_avg_wait = fixed_results["avg_wait"].mean()
        agent_avg_wait = agent_results["avg_wait"].mean()

        fixed_throughput = fixed_results["throughput"].sum()
        agent_throughput = agent_results["throughput"].sum()

        improvement_wait = ((fixed_avg_wait - agent_avg_wait) / fixed_avg_wait) * 100
        improvement_throughput = (
            (agent_throughput - fixed_throughput) / fixed_throughput
        ) * 100

        print(f"\nAverage Wait Time:")
        print(f"  Fixed-Time: {fixed_avg_wait:.2f}s")
        print(f"  Agent-Based: {agent_avg_wait:.2f}s")
        print(f"  Improvement: {improvement_wait:.1f}%")

        print(f"\nTotal Throughput:")
        print(f"  Fixed-Time: {fixed_throughput:.0f} vehicles")
        print(f"  Agent-Based: {agent_throughput:.0f} vehicles")
        print(f"  Improvement: {improvement_throughput:.1f}%")

        # Visualize comparison
        self.plot_comparison(fixed_results, agent_results)

        return fixed_results, agent_results, trained_agent

    def plot_comparison(self, fixed_results, agent_results):
        """Plot comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Fixed-Time vs Agent-Based Control Comparison", fontsize=16)

        # Average Wait Time
        axes[0, 0].plot(
            fixed_results["episode"],
            fixed_results["avg_wait"],
            marker="o",
            label="Fixed-Time",
            linewidth=2,
        )
        axes[0, 0].plot(
            agent_results["episode"],
            agent_results["avg_wait"],
            marker="s",
            label="Agent-Based",
            linewidth=2,
        )
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Average Wait Time (s)")
        axes[0, 0].set_title("Average Wait Time per Episode")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Average Queue Length
        axes[0, 1].plot(
            fixed_results["episode"],
            fixed_results["avg_queue"],
            marker="o",
            label="Fixed-Time",
            linewidth=2,
        )
        axes[0, 1].plot(
            agent_results["episode"],
            agent_results["avg_queue"],
            marker="s",
            label="Agent-Based",
            linewidth=2,
        )
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Average Queue Length")
        axes[0, 1].set_title("Average Queue Length per Episode")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Throughput
        axes[1, 0].bar(
            ["Fixed-Time", "Agent-Based"],
            [fixed_results["throughput"].sum(), agent_results["throughput"].sum()],
            color=["#ef4444", "#10b981"],
        )
        axes[1, 0].set_ylabel("Total Vehicles Passed")
        axes[1, 0].set_title("Total Throughput")
        axes[1, 0].grid(True, alpha=0.3, axis="y")

        # Improvement percentages
        improvement_wait = (
            (fixed_results["avg_wait"].mean() - agent_results["avg_wait"].mean())
            / fixed_results["avg_wait"].mean()
        ) * 100
        improvement_queue = (
            (fixed_results["avg_queue"].mean() - agent_results["avg_queue"].mean())
            / fixed_results["avg_queue"].mean()
        ) * 100

        metrics = ["Wait Time\nReduction", "Queue Length\nReduction"]
        improvements = [improvement_wait, improvement_queue]
        colors = ["#10b981" if x > 0 else "#ef4444" for x in improvements]

        axes[1, 1].bar(metrics, improvements, color=colors)
        axes[1, 1].set_ylabel("Improvement (%)")
        axes[1, 1].set_title("Performance Improvements")
        axes[1, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("comparison_results.png", dpi=300, bbox_inches="tight")
        print("\n✓ Comparison plot saved as 'comparison_results.png'")
        plt.show()
