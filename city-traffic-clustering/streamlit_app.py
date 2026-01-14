"""
Streamlit Traffic Simulation Dashboard
Real-time visualization of Q-Learning agent vs Fixed-time signal controller
"""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Wedge, Polygon
import pickle
import json


# Import project modules after setup
def setup_imports():
    """Setup path and import project modules"""

    def find_project_root():
        current = Path.cwd()
        while current != current.parent:
            if (current / "src").exists() and (current / "data").exists():
                return str(current)
            current = current.parent
        return str(Path(__file__).parent)

    proj_root = find_project_root()
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

    from src.traffic_environment import TrafficEnvironment
    from src.q_learning_agent import QLearningAgent

    return proj_root, TrafficEnvironment, QLearningAgent


proj_root, TrafficEnvironment, QLearningAgent = setup_imports()


# ==================== VISUALIZATION FUNCTIONS ====================
def draw_traffic_intersection(
    ax, state, signal_state, vehicles_ns, vehicles_ew, title="Intersection"
):
    """Draw an interactive traffic intersection visualization"""

    # Set up the plot
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # Draw roads
    # Vertical road (North-South)
    rect_ns = Rectangle(
        (30, 0), 40, 100, linewidth=2, edgecolor="gray", facecolor="#f0f0f0", zorder=1
    )
    ax.add_patch(rect_ns)
    # Road markings
    for y in range(10, 100, 15):
        ax.plot([50, 50], [y, y + 5], "w-", linewidth=2)

    # Horizontal road (East-West)
    rect_ew = Rectangle(
        (0, 30), 100, 40, linewidth=2, edgecolor="gray", facecolor="#f0f0f0", zorder=1
    )
    ax.add_patch(rect_ew)
    # Road markings
    for x in range(10, 100, 15):
        ax.plot([x, x + 5], [50, 50], "w-", linewidth=2)

    # Draw intersection
    intersection = Rectangle(
        (30, 30), 40, 40, linewidth=2, edgecolor="black", facecolor="#e8e8e8", zorder=2
    )
    ax.add_patch(intersection)

    # Get signal state
    signal_ns_color = "green" if signal_state == "green-ns" else "red"
    signal_ew_color = "green" if signal_state == "green-ew" else "red"

    # Draw traffic lights
    # NS light (left)
    light_ns = Circle((20, 50), 3, color=signal_ns_color, zorder=5)
    ax.add_patch(light_ns)
    ax.text(
        15,
        50,
        f"NS\n{signal_ns_color.upper()}",
        fontsize=8,
        ha="center",
        va="center",
        fontweight="bold",
    )

    # EW light (bottom)
    light_ew = Circle((50, 20), 3, color=signal_ew_color, zorder=5)
    ax.add_patch(light_ew)
    ax.text(
        50,
        12,
        f"EW {signal_ew_color.upper()}",
        fontsize=8,
        ha="center",
        va="center",
        fontweight="bold",
    )

    # Draw vehicles
    vehicle_width = 3
    vehicle_length = 5

    # NS vehicles (moving north)
    for i, vehicle in enumerate(vehicles_ns[:8]):  # Show max 8 vehicles
        y_pos = 15 + i * 7
        car = Rectangle(
            (45, y_pos),
            vehicle_width,
            vehicle_length,
            linewidth=1,
            edgecolor="black",
            facecolor="#FF6B6B",
            zorder=4,
        )
        ax.add_patch(car)
        # Add window
        ax.plot([45.5, 47.5], [y_pos + 2, y_pos + 2], "w-", linewidth=1)

    # EW vehicles (moving east)
    for i, vehicle in enumerate(vehicles_ew[:8]):  # Show max 8 vehicles
        x_pos = 15 + i * 7
        car = Rectangle(
            (x_pos, 47),
            vehicle_length,
            vehicle_width,
            linewidth=1,
            edgecolor="black",
            facecolor="#4ECDC4",
            zorder=4,
        )
        ax.add_patch(car)
        # Add window
        ax.plot([x_pos + 2, x_pos + 2], [47.5, 49.5], "w-", linewidth=1)

    # Add queue length text
    queue_ns = state.get("queue_ns", 0)
    queue_ew = state.get("queue_ew", 0)
    ax.text(
        8,
        50,
        f"Queue: {queue_ns}",
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="#FFE66D", alpha=0.7),
        zorder=6,
    )
    ax.text(
        50,
        8,
        f"Queue: {queue_ew}",
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="#FFE66D", alpha=0.7),
        zorder=6,
    )

    # Add direction labels
    ax.text(50, 97, "↑ North", fontsize=10, ha="center", fontweight="bold")
    ax.text(97, 50, "East →", fontsize=10, ha="center", fontweight="bold")


def draw_signal_comparison(
    ax, signal_history_agent, signal_history_fixed, title="Signal State Timeline"
):
    """Draw signal state comparison over time"""
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Convert signal states to numerical (0 = red, 1 = green)
    agent_states = [1 if s == "green-ns" else 0 for s in signal_history_agent]
    fixed_states = [1 if s == "green-ns" else 0 for s in signal_history_fixed]

    steps = range(len(agent_states))

    ax.fill_between(
        steps, agent_states, alpha=0.4, label="Q-Agent (NS Green)", color="green"
    )
    ax.fill_between(
        steps, fixed_states, -0.1, alpha=0.4, label="Fixed (NS Green)", color="orange"
    )

    ax.set_ylabel("NS Green State")
    ax.set_xlabel("Step")
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Red", "Green"])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


# Page configuration
st.set_page_config(
    page_title="Traffic Signal Optimization",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #0668a1;
    }
    .metric-label {
        font-size: 12px;
        color: #666;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Page title
st.title("🚦 Intelligent Traffic Signal Optimization System")
st.markdown("**Using Q-Learning for Automatic Signal Control in Bangalore**")
st.divider()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Configuration")

    # Load configuration
    config_path = os.path.join(proj_root, "results", "agent_config.json")
    with open(config_path, "r") as f:
        agent_config = json.load(f)

    # Load scenarios
    scenarios_path = os.path.join(
        proj_root, "data", "processed", "traffic_scenarios_from_clusters_k4.csv"
    )
    scenarios = pd.read_csv(scenarios_path)

    # Simulation parameters
    st.subheader("Simulation Parameters")

    mode = st.radio(
        "Select Control Strategy",
        ["Q-Learning Agent", "Fixed Time Controller", "Comparison"],
        help="Compare different signal control strategies",
    )

    cluster_name = st.selectbox(
        "Traffic Scenario",
        ["Free Flow", "Normal", "Peak Congestion", "Incident/Disrupted"],
        index=2,
        help="Choose traffic condition for simulation",
    )
    cluster_id = ["Free Flow", "Normal", "Peak Congestion", "Incident/Disrupted"].index(
        cluster_name
    )

    num_steps = st.slider(
        "Simulation Duration (steps)",
        min_value=50,
        max_value=500,
        value=300,
        step=50,
        help="1 step = ~2 seconds",
    )

    # Advanced parameters
    with st.expander("Advanced Settings"):
        signal_cycle = st.slider("Signal Cycle (sec)", 30, 120, 60, 10)
        ns_green = st.slider("NS Green Time (sec)", 10, 60, 30, 5)
        ew_green = st.slider("EW Green Time (sec)", 10, 60, 30, 5)

    st.divider()

    # Display scenario info
    st.subheader("📊 Scenario Details")
    scenario_data = scenarios[scenarios["cluster_id"] == cluster_id].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Traffic Volume", f"{scenario_data['Traffic Volume']:.0f} veh/hr")
        st.metric("Congestion Level", f"{scenario_data['Congestion Level']:.1f}%")
    with col2:
        st.metric("Avg Speed", f"{scenario_data['Average Speed']:.1f} km/h")
        st.metric("Incident Prob", f"{scenario_data['incident_prob']:.2%}")

# ==================== MAIN CONTENT ====================

# Load traffic data for environment
traffic_data_path = os.path.join(
    proj_root, "data", "processed", "bangalore_traffic_with_clusters_k4.csv"
)
traffic_data = pd.read_csv(traffic_data_path)

# Load trained agent
agent_path = os.path.join(proj_root, "results", "trained_q_agent.pkl")
try:
    with open(agent_path, "rb") as f:
        agent_data = pickle.load(f)
        if isinstance(agent_data, dict):
            # If saved as dict with q_table and config
            agent = QLearningAgent(
                n_states=agent_data["config"]["n_states"],
                n_queue_bins=agent_data["config"]["n_queue_bins"],
                n_actions=agent_data["config"]["n_actions"],
                lr=agent_data["config"]["lr"],
                gamma=agent_data["config"]["gamma"],
                epsilon=0.0,  # No exploration during simulation
            )
            agent.q = agent_data["q_table"]
        else:
            # If saved as agent object directly
            agent = agent_data
            agent.epsilon = 0.0  # No exploration during simulation
    agent_loaded = True
except Exception as e:
    agent_loaded = False
    st.warning(
        f"⚠️ Could not load trained agent: {e}. Running with newly initialized agent."
    )
    agent = QLearningAgent(
        n_states=4,
        n_queue_bins=10,
        n_actions=5,
        lr=0.1,
        gamma=0.95,
        epsilon=0.0,  # No exploration during simulation
        decay=1.0,
    )

# Create environments (only needs scenarios_df and traffic_data_df)
env_agent = TrafficEnvironment(scenarios_df=scenarios, traffic_data_df=traffic_data)
env_fixed = TrafficEnvironment(scenarios_df=scenarios, traffic_data_df=traffic_data)

# ==================== SIMULATION ====================

if st.button("▶️ Run Simulation", key="run_sim", use_container_width=True):
    # Initialize states
    state_agent = env_agent.reset()
    state_fixed = env_fixed.reset()

    # Metrics tracking
    metrics_agent = {
        "reward": [],
        "queue_ns": [],
        "queue_ew": [],
        "wait_time": [],
        "vehicles_passed": [],
        "action": [],
        "signal_state": [],
    }
    metrics_fixed = {
        "reward": [],
        "queue_ns": [],
        "queue_ew": [],
        "wait_time": [],
        "vehicles_passed": [],
        "action": [],
        "signal_state": [],
    }

    action_names = [
        "Default",
        "Extend NS",
        "Extend EW",
        "Reduce Cycle",
        "Increase Cycle",
    ]

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Run simulation
    for step in range(num_steps):
        # Agent control
        action_agent = agent.choose_action(state_agent)  # Greedy policy
        next_state_agent, reward_agent, _, info_agent = env_agent.step(action_agent)

        # Fixed-time control (simple fixed timing)
        action_fixed = 0  # No change (default)
        next_state_fixed, reward_fixed, _, info_fixed = env_fixed.step(action_fixed)

        # Store metrics - from state and info
        metrics_agent["reward"].append(reward_agent)
        metrics_agent["queue_ns"].append(state_agent.get("queue_ns", 0))
        metrics_agent["queue_ew"].append(state_agent.get("queue_ew", 0))
        metrics_agent["wait_time"].append(info_agent.get("avg_wait_time", 0))
        metrics_agent["vehicles_passed"].append(info_agent.get("vehicles_passed", 0))
        metrics_agent["action"].append(action_agent)
        metrics_agent["signal_state"].append(
            next_state_agent.get("signal_state", "green-ns")
        )

        metrics_fixed["reward"].append(reward_fixed)
        metrics_fixed["queue_ns"].append(state_fixed.get("queue_ns", 0))
        metrics_fixed["queue_ew"].append(state_fixed.get("queue_ew", 0))
        metrics_fixed["wait_time"].append(info_fixed.get("avg_wait_time", 0))
        metrics_fixed["vehicles_passed"].append(info_fixed.get("vehicles_passed", 0))
        metrics_fixed["action"].append(action_fixed)
        metrics_fixed["signal_state"].append(
            next_state_fixed.get("signal_state", "green-ns")
        )

        state_agent = next_state_agent
        state_fixed = next_state_fixed

        # Update progress
        progress = (step + 1) / num_steps
        progress_bar.progress(progress)
        status_text.text(
            f"Step {step + 1}/{num_steps} - {cluster_name} Traffic Scenario"
        )

    st.success("✓ Simulation completed!")
    st.divider()

    # ==================== VISUAL DEMONSTRATION ====================
    st.subheader("🎬 Live Traffic Intersection Visualization")

    # Create side-by-side intersection visualizations
    col_left, col_right = st.columns(2)

    with col_left:
        # Extract sample vehicle data for visualization
        vehicles_ns_agent = [
            {"pos": i} for i in range(min(int(metrics_agent["queue_ns"][-1]), 8))
        ]
        vehicles_ew_agent = [
            {"pos": i} for i in range(min(int(metrics_agent["queue_ew"][-1]), 8))
        ]

        fig_agent, ax_agent = plt.subplots(figsize=(8, 8))
        draw_traffic_intersection(
            ax_agent,
            state_agent,
            state_agent.get("signal_state", "green-ns"),
            vehicles_ns_agent,
            vehicles_ew_agent,
            title=f"Q-Learning Agent (Step {len(metrics_agent['reward']) - 1})",
        )
        st.pyplot(fig_agent, use_container_width=True)

    with col_right:
        # Extract sample vehicle data for visualization
        vehicles_ns_fixed = [
            {"pos": i} for i in range(min(int(metrics_fixed["queue_ns"][-1]), 8))
        ]
        vehicles_ew_fixed = [
            {"pos": i} for i in range(min(int(metrics_fixed["queue_ew"][-1]), 8))
        ]

        fig_fixed, ax_fixed = plt.subplots(figsize=(8, 8))
        draw_traffic_intersection(
            ax_fixed,
            state_fixed,
            state_fixed.get("signal_state", "green-ns"),
            vehicles_ns_fixed,
            vehicles_ew_fixed,
            title="Fixed Time Controller (Step {len(metrics_fixed['reward'])-1})",
        )
        st.pyplot(fig_fixed, use_container_width=True)

    st.divider()

    # ==================== RESULTS ====================

    # Metrics comparison
    st.subheader("📈 Performance Metrics")

    if mode in ["Q-Learning Agent", "Comparison"]:
        col1, col2, col3, col4 = st.columns(4)

        avg_queue_agent = (
            np.mean(metrics_agent["queue_ns"]) + np.mean(metrics_agent["queue_ew"])
        ) / 2
        avg_wait_agent = np.mean(metrics_agent["wait_time"])
        total_reward_agent = np.sum(metrics_agent["reward"])
        total_passed_agent = np.sum(metrics_agent["vehicles_passed"])

        with col1:
            st.metric("Q-Agent: Total Reward", f"{total_reward_agent:.1f}")
        with col2:
            st.metric("Q-Agent: Avg Queue", f"{avg_queue_agent:.1f}")
        with col3:
            st.metric("Q-Agent: Avg Wait", f"{avg_wait_agent:.1f}s")
        with col4:
            st.metric("Q-Agent: Vehicles Passed", f"{total_passed_agent:.0f}")

    if mode in ["Fixed Time Controller", "Comparison"]:
        col1, col2, col3, col4 = st.columns(4)

        avg_queue_fixed = (
            np.mean(metrics_fixed["queue_ns"]) + np.mean(metrics_fixed["queue_ew"])
        ) / 2
        avg_wait_fixed = np.mean(metrics_fixed["wait_time"])
        total_reward_fixed = np.sum(metrics_fixed["reward"])
        total_passed_fixed = np.sum(metrics_fixed["vehicles_passed"])

        with col1:
            st.metric("Fixed: Total Reward", f"{total_reward_fixed:.1f}")
        with col2:
            st.metric("Fixed: Avg Queue", f"{avg_queue_fixed:.1f}")
        with col3:
            st.metric("Fixed: Avg Wait", f"{avg_wait_fixed:.1f}s")
        with col4:
            st.metric("Fixed: Vehicles Passed", f"{total_passed_fixed:.0f}")

    st.divider()

    # Comparison charts
    if mode == "Comparison":
        st.subheader("🔄 Comparison: Q-Learning Agent vs Fixed Controller")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Reward", "Queue Length", "Wait Time", "Throughput", "Signal Control"]
        )

        with tab1:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(
                metrics_agent["reward"],
                label="Q-Learning Agent",
                linewidth=2,
                alpha=0.8,
            )
            ax.plot(
                metrics_fixed["reward"],
                label="Fixed Controller",
                linewidth=2,
                alpha=0.8,
            )
            ax.fill_between(
                range(len(metrics_agent["reward"])),
                metrics_agent["reward"],
                metrics_fixed["reward"],
                where=np.array(metrics_agent["reward"])
                >= np.array(metrics_fixed["reward"]),
                alpha=0.2,
                color="green",
                label="Q-Agent Advantage",
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.set_title("Reward per Step Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with tab2:
            fig, ax = plt.subplots(figsize=(12, 5))
            avg_queue_agent_ts = [
                (n + e) / 2
                for n, e in zip(metrics_agent["queue_ns"], metrics_agent["queue_ew"])
            ]
            avg_queue_fixed_ts = [
                (n + e) / 2
                for n, e in zip(metrics_fixed["queue_ns"], metrics_fixed["queue_ew"])
            ]
            ax.plot(
                avg_queue_agent_ts, label="Q-Learning Agent", linewidth=2, alpha=0.8
            )
            ax.plot(
                avg_queue_fixed_ts, label="Fixed Controller", linewidth=2, alpha=0.8
            )
            ax.fill_between(
                range(len(avg_queue_agent_ts)),
                avg_queue_agent_ts,
                avg_queue_fixed_ts,
                where=np.array(avg_queue_agent_ts) <= np.array(avg_queue_fixed_ts),
                alpha=0.2,
                color="green",
                label="Q-Agent Advantage",
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Average Queue Length")
            ax.set_title("Queue Length Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with tab3:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(
                metrics_agent["wait_time"],
                label="Q-Learning Agent",
                linewidth=2,
                alpha=0.8,
            )
            ax.plot(
                metrics_fixed["wait_time"],
                label="Fixed Controller",
                linewidth=2,
                alpha=0.8,
            )
            ax.fill_between(
                range(len(metrics_agent["wait_time"])),
                metrics_agent["wait_time"],
                metrics_fixed["wait_time"],
                where=np.array(metrics_agent["wait_time"])
                <= np.array(metrics_fixed["wait_time"]),
                alpha=0.2,
                color="green",
                label="Q-Agent Advantage",
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Average Wait Time (sec)")
            ax.set_title("Wait Time Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with tab4:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(
                np.arange(len(metrics_agent["vehicles_passed"])) - 0.2,
                metrics_agent["vehicles_passed"],
                width=0.4,
                label="Q-Learning Agent",
                alpha=0.8,
            )
            ax.bar(
                np.arange(len(metrics_fixed["vehicles_passed"])) + 0.2,
                metrics_fixed["vehicles_passed"],
                width=0.4,
                label="Fixed Controller",
                alpha=0.8,
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Vehicles Passed")
            ax.set_title("Throughput Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            st.pyplot(fig)

        with tab5:
            fig, ax = plt.subplots(figsize=(12, 5))
            draw_signal_comparison(
                ax,
                metrics_agent["signal_state"],
                metrics_fixed["signal_state"],
                "Signal Control Strategy (NS Green)",
            )
            st.pyplot(fig)

    elif mode == "Q-Learning Agent":
        st.subheader("Q-Learning Agent Performance")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Reward
        axes[0, 0].plot(metrics_agent["reward"], "b-", linewidth=2, alpha=0.8)
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].set_title("Reward per Step")
        axes[0, 0].grid(True, alpha=0.3)

        # Queue
        avg_queue = [
            (n + e) / 2
            for n, e in zip(metrics_agent["queue_ns"], metrics_agent["queue_ew"])
        ]
        axes[0, 1].plot(metrics_agent["queue_ns"], "g-", alpha=0.6, label="NS Queue")
        axes[0, 1].plot(
            metrics_agent["queue_ew"], "orange", alpha=0.6, label="EW Queue"
        )
        axes[0, 1].plot(avg_queue, "r-", linewidth=2, label="Average")
        axes[0, 1].set_ylabel("Queue Length")
        axes[0, 1].set_title("Queue Length Over Time")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Wait time
        axes[1, 0].plot(metrics_agent["wait_time"], "purple", linewidth=2, alpha=0.8)
        axes[1, 0].set_ylabel("Wait Time (sec)")
        axes[1, 0].set_title("Average Wait Time Over Time")
        axes[1, 0].grid(True, alpha=0.3)

        # Vehicles passed
        axes[1, 1].bar(
            range(len(metrics_agent["vehicles_passed"])),
            metrics_agent["vehicles_passed"],
            color="teal",
            alpha=0.6,
        )
        axes[1, 1].set_ylabel("Vehicles Passed")
        axes[1, 1].set_title("Throughput per Step")
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    # Summary statistics
    st.subheader("📊 Summary Statistics")

    if mode in ["Q-Learning Agent", "Comparison"]:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Q-Learning Agent**")
            st.write(f"- Total Reward: {total_reward_agent:.1f}")
            st.write(f"- Average Queue: {avg_queue_agent:.1f} vehicles")
            st.write(f"- Average Wait Time: {avg_wait_agent:.2f} seconds")
            st.write(f"- Total Vehicles Passed: {total_passed_agent:.0f}")
            st.write(
                f"- Most used action: {action_names[int(np.argmax(np.bincount(metrics_agent['action'])))]}"
            )

    if mode in ["Fixed Time Controller", "Comparison"]:
        col2, col3 = st.columns(2)
        with col2 if mode == "Q-Learning Agent" else col1:
            st.write("**Fixed Time Controller**")
            st.write(f"- Total Reward: {total_reward_fixed:.1f}")
            st.write(f"- Average Queue: {avg_queue_fixed:.1f} vehicles")
            st.write(f"- Average Wait Time: {avg_wait_fixed:.2f} seconds")
            st.write(f"- Total Vehicles Passed: {total_passed_fixed:.0f}")

        if mode == "Comparison":
            with col3:
                st.write("**Improvement (Q-Learning)**")
                reward_improvement = (
                    (total_reward_agent - total_reward_fixed) / abs(total_reward_fixed)
                ) * 100
                queue_improvement = (
                    (avg_queue_fixed - avg_queue_agent) / avg_queue_fixed
                ) * 100
                wait_improvement = (
                    (avg_wait_fixed - avg_wait_agent) / avg_wait_fixed
                ) * 100
                throughput_improvement = (
                    (total_passed_agent - total_passed_fixed) / total_passed_fixed
                ) * 100

                st.metric("Reward Improvement", f"{reward_improvement:+.1f}%")
                st.metric("Queue Reduction", f"{queue_improvement:+.1f}%")
                st.metric("Wait Time Reduction", f"{wait_improvement:+.1f}%")
                st.metric("Throughput Improvement", f"{throughput_improvement:+.1f}%")

# Footer
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 12px;">
    <p>🚦 Intelligent Traffic Signal Optimization | Q-Learning Agent for Bangalore Traffic Management</p>
    <p>Simulating real-time traffic control with adaptive signal timing</p>
    </div>
""",
    unsafe_allow_html=True,
)
