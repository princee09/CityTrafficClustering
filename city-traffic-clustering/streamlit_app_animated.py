"""
Streamlit Traffic Simulation Dashboard with Real-time Animation
Q-Learning Agent vs Fixed-time signal controller with animated intersection
"""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
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


# ==================== ANIMATION FUNCTIONS ====================
def render_intersection_frame(ax, frame_data, frame_idx, total_frames):
    """Render a single frame of the intersection with moving cars and changing signals"""
    ax.clear()
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)
    ax.set_aspect("equal")
    ax.axis("off")

    # Extract frame data
    signal_state = frame_data["signals"][frame_idx]
    vehicles_ns = frame_data["vehicles_ns"][frame_idx]
    vehicles_ew = frame_data["vehicles_ew"][frame_idx]
    queue_ns = frame_data["queues_ns"][frame_idx]
    queue_ew = frame_data["queues_ew"][frame_idx]
    reward = frame_data["rewards"][frame_idx]

    # Draw roads with gradient effect
    # Vertical road (North-South)
    ax.add_patch(
        Rectangle((30, 0), 40, 100, linewidth=2, edgecolor="#333", facecolor="#b0b0b0")
    )
    # Yellow lane markings
    for y in range(0, 100, 12):
        ax.plot([50, 50], [y, y + 6], "y-", linewidth=3)

    # Horizontal road (East-West)
    ax.add_patch(
        Rectangle((0, 30), 100, 40, linewidth=2, edgecolor="#333", facecolor="#b0b0b0")
    )
    # Yellow lane markings
    for x in range(0, 100, 12):
        ax.plot([x, x + 6], [50, 50], "y-", linewidth=3)

    # Draw intersection
    ax.add_patch(
        Rectangle((30, 30), 40, 40, linewidth=3, edgecolor="#222", facecolor="#a0a0a0")
    )

    # Determine light colors based on signal state
    # Colors will be used in the light rendering below

    # Draw traffic light posts and signals
    # NS traffic light (left side)
    ax.add_patch(
        Rectangle((5, 42), 10, 16, linewidth=2, edgecolor="black", facecolor="#222")
    )
    # Top light (red)
    ax.add_patch(
        Circle(
            (10, 54),
            2.5,
            color="#660000" if signal_state == "green-ns" else "#FF0000",
            zorder=10,
        )
    )
    # Middle light (yellow)
    ax.add_patch(Circle((10, 50), 2.5, color="#666600", zorder=10))
    # Bottom light (green)
    ax.add_patch(
        Circle(
            (10, 46),
            2.5,
            color="#00FF00" if signal_state == "green-ns" else "#003300",
            zorder=10,
        )
    )

    # EW traffic light (bottom side)
    ax.add_patch(
        Rectangle((42, 5), 16, 10, linewidth=2, edgecolor="black", facecolor="#222")
    )
    # Left light (red)
    ax.add_patch(
        Circle(
            (46, 10),
            2.5,
            color="#660000" if signal_state == "green-ew" else "#FF0000",
            zorder=10,
        )
    )
    # Middle light (yellow)
    ax.add_patch(Circle((50, 10), 2.5, color="#666600", zorder=10))
    # Right light (green)
    ax.add_patch(
        Circle(
            (54, 10),
            2.5,
            color="#00FF00" if signal_state == "green-ew" else "#003300",
            zorder=10,
        )
    )

    # Draw vehicles with smooth animation
    # NS vehicles (moving northward)
    for idx, vehicle in enumerate(vehicles_ns):
        # Animate position based on signal
        y_base = vehicle["y_base"] + vehicle["animation_offset"]
        if signal_state == "green-ns":
            y_base += (frame_idx % 5) * 2  # Move vehicles when green

        # Car body
        ax.add_patch(
            Rectangle(
                (vehicle["x"] - 2, y_base),
                4,
                6,
                linewidth=2,
                edgecolor="#8B0000",
                facecolor="#FF4444",
                zorder=15,
            )
        )
        # Windows
        ax.add_patch(
            Rectangle(
                (vehicle["x"] - 1.5, y_base + 1), 1.5, 2, facecolor="#87CEEB", zorder=16
            )
        )
        ax.add_patch(
            Rectangle(
                (vehicle["x"], y_base + 1), 1.5, 2, facecolor="#87CEEB", zorder=16
            )
        )
        # Headlights
        ax.add_patch(
            Circle((vehicle["x"] - 1.5, y_base), 0.5, color="yellow", zorder=16)
        )
        ax.add_patch(
            Circle((vehicle["x"] + 1.5, y_base), 0.5, color="yellow", zorder=16)
        )

    # EW vehicles (moving eastward)
    for idx, vehicle in enumerate(vehicles_ew):
        # Animate position based on signal
        x_base = vehicle["x_base"] + vehicle["animation_offset"]
        if signal_state == "green-ew":
            x_base += (frame_idx % 5) * 2  # Move vehicles when green

        # Car body
        ax.add_patch(
            Rectangle(
                (x_base, vehicle["y"] - 2),
                6,
                4,
                linewidth=2,
                edgecolor="#000080",
                facecolor="#00BFFF",
                zorder=15,
            )
        )
        # Windows
        ax.add_patch(
            Rectangle(
                (x_base + 1, vehicle["y"] - 1.5), 2, 1.5, facecolor="#87CEEB", zorder=16
            )
        )
        ax.add_patch(
            Rectangle(
                (x_base + 1, vehicle["y"]), 2, 1.5, facecolor="#87CEEB", zorder=16
            )
        )
        # Headlights
        ax.add_patch(
            Circle((x_base + 6, vehicle["y"] - 1), 0.5, color="yellow", zorder=16)
        )
        ax.add_patch(
            Circle((x_base + 6, vehicle["y"] + 1), 0.5, color="yellow", zorder=16)
        )

    # Info displays
    # NS Queue info
    ax.text(
        -5,
        50,
        f"NS Queue\n{int(queue_ns)}",
        fontsize=11,
        ha="right",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.7",
            facecolor="#FFFF99",
            edgecolor="#FF8C00",
            linewidth=2,
        ),
        fontweight="bold",
        zorder=20,
    )

    # EW Queue info
    ax.text(
        50,
        -5,
        f"EW Queue: {int(queue_ew)}",
        fontsize=11,
        ha="center",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.7",
            facecolor="#FFFF99",
            edgecolor="#FF8C00",
            linewidth=2,
        ),
        fontweight="bold",
        zorder=20,
    )

    # Reward display
    reward_color = "#90EE90" if reward > 0 else "#FFB6C6"
    ax.text(
        50,
        103,
        f"Reward: {reward:.2f}",
        fontsize=12,
        ha="center",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor=reward_color,
            edgecolor="black",
            linewidth=2,
        ),
        fontweight="bold",
        zorder=20,
    )

    # Step counter
    ax.text(
        50,
        -8,
        f"Step {frame_idx + 1}/{total_frames}",
        fontsize=10,
        ha="center",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="lightblue",
            edgecolor="blue",
            linewidth=1.5,
        ),
        fontweight="bold",
        zorder=20,
    )

    # Signal state indicator
    signal_text = "🟢 NS GREEN" if signal_state == "green-ns" else "🔴 EW GREEN"
    ax.text(
        50,
        110,
        signal_text,
        fontsize=13,
        ha="center",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="lightyellow",
            edgecolor="black",
            linewidth=2,
        ),
        fontweight="bold",
        zorder=20,
    )

    # Direction labels
    ax.text(
        50, 107, "↑ NORTH", fontsize=11, ha="center", fontweight="bold", color="#333"
    )
    ax.text(107, 50, "EAST →", fontsize=11, ha="right", fontweight="bold", color="#333")
    ax.text(
        50, -7.5, "SOUTH ↓", fontsize=11, ha="center", fontweight="bold", color="#333"
    )
    ax.text(-7.5, 50, "← WEST", fontsize=11, ha="left", fontweight="bold", color="#333")

    # Title
    ax.text(
        50,
        -1,
        "AI Traffic Control System - Real-time Simulation",
        fontsize=12,
        ha="center",
        fontweight="bold",
        zorder=20,
    )


def generate_animation_data(env, agent, num_steps=100):
    """Generate frame data for animation"""
    state = env.reset()

    frame_data = {
        "signals": [],
        "queues_ns": [],
        "queues_ew": [],
        "rewards": [],
        "vehicles_ns": [],
        "vehicles_ew": [],
    }

    for step in range(num_steps):
        action = agent.choose_action(state)
        next_state, reward, _, info = env.step(action)

        # Store signal and queue states
        frame_data["signals"].append(next_state.get("signal_state", "green-ns"))
        frame_data["queues_ns"].append(next_state.get("queue_ns", 0))
        frame_data["queues_ew"].append(next_state.get("queue_ew", 0))
        frame_data["rewards"].append(reward)

        # Generate vehicle positions with animation offsets
        queue_ns = int(next_state.get("queue_ns", 0))
        queue_ew = int(next_state.get("queue_ew", 0))

        vehicles_ns = []
        for i in range(min(queue_ns, 12)):
            vehicles_ns.append(
                {
                    "x": 45,
                    "y_base": 20 + i * 6,
                    "animation_offset": (step % 10) * 0.5,
                    "speed": 1.0,
                }
            )

        vehicles_ew = []
        for i in range(min(queue_ew, 12)):
            vehicles_ew.append(
                {
                    "x_base": 20 + i * 6,
                    "y": 47,
                    "animation_offset": (step % 10) * 0.5,
                    "speed": 1.0,
                }
            )

        frame_data["vehicles_ns"].append(vehicles_ns)
        frame_data["vehicles_ew"].append(vehicles_ew)

        state = next_state

    return frame_data


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Traffic Control",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main {
        background-color: #f5f5f5;
    }
    .metric-card {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Simulation Configuration")

    # Load configuration
    config_path = os.path.join(proj_root, "results", "agent_config.json")
    with open(config_path, "r") as f:
        agent_config = json.load(f)

    # Load scenarios
    scenarios_path = os.path.join(
        proj_root, "data", "processed", "traffic_scenarios_from_clusters_k4.csv"
    )
    scenarios = pd.read_csv(scenarios_path)

    # Load traffic data
    traffic_data_path = os.path.join(
        proj_root, "data", "processed", "bangalore_traffic_with_clusters_k4.csv"
    )
    traffic_data = pd.read_csv(traffic_data_path)

    mode = st.radio(
        "🎮 Control Strategy",
        ["Q-Learning Agent", "Fixed Time Controller", "Comparison"],
        help="Compare different signal control strategies",
    )

    cluster_name = st.selectbox(
        "🚗 Traffic Scenario",
        ["Free Flow", "Normal", "Peak Congestion", "Incident/Disrupted"],
        index=2,
    )
    cluster_id = ["Free Flow", "Normal", "Peak Congestion", "Incident/Disrupted"].index(
        cluster_name
    )

    num_steps = st.slider(
        "📊 Simulation Duration (steps)",
        min_value=50,
        max_value=300,
        value=150,
        step=25,
    )

    # Scenario details
    st.subheader("📋 Scenario Info")
    scenario_data = scenarios[scenarios["cluster_id"] == cluster_id].iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Traffic Volume", f"{scenario_data['Traffic Volume']:.0f} veh/hr")
        st.metric("Congestion", f"{scenario_data['Congestion Level']:.1f}%")
    with col2:
        st.metric("Avg Speed", f"{scenario_data['Average Speed']:.1f} km/h")
        st.metric("Incident Prob", f"{scenario_data['incident_prob']:.2%}")

# ==================== MAIN CONTENT ====================
st.title("🚦 AI Traffic Signal Optimization System")
st.markdown("**Real-time Animated Simulation with Q-Learning Agent**")
st.divider()

# Load trained agent
agent_path = os.path.join(proj_root, "results", "trained_q_agent.pkl")
try:
    with open(agent_path, "rb") as f:
        agent_data = pickle.load(f)
        if isinstance(agent_data, dict):
            agent = QLearningAgent(
                n_states=agent_data["config"]["n_states"],
                n_queue_bins=agent_data["config"]["n_queue_bins"],
                n_actions=agent_data["config"]["n_actions"],
                lr=agent_data["config"]["lr"],
                gamma=agent_data["config"]["gamma"],
                epsilon=0.0,
            )
            agent.q = agent_data["q_table"]
        else:
            agent = agent_data
            agent.epsilon = 0.0
    agent_loaded = True
except Exception as e:
    st.warning(f"⚠️ Could not load trained agent: {e}")
    agent = QLearningAgent(
        n_states=4, n_queue_bins=10, n_actions=5, lr=0.1, gamma=0.95, epsilon=0.0
    )
    agent_loaded = False

# ==================== SIMULATION ====================
if st.button("▶️ START ANIMATED SIMULATION", key="run_sim", use_container_width=True):
    st.info("🎬 Generating animated simulation... This may take a moment.")

    # Create environments
    env_agent = TrafficEnvironment(scenarios_df=scenarios, traffic_data_df=traffic_data)
    env_fixed = TrafficEnvironment(scenarios_df=scenarios, traffic_data_df=traffic_data)

    # Generate animation data
    with st.spinner("Rendering animation frames..."):
        frame_data_agent = generate_animation_data(env_agent, agent, num_steps)
        frame_data_fixed = generate_animation_data(env_fixed, agent, num_steps)

    # Create columns for side-by-side display
    col1, col2 = st.columns(2)

    # Select which frames to display (create slider for frame selection)
    st.subheader("🎬 Live Animation")

    frame_slider = st.slider(
        "Select Frame to Display",
        min_value=0,
        max_value=num_steps - 1,
        value=0,
        step=1,
    )

    with col1:
        st.subheader("Q-Learning Agent Control")
        fig_agent, ax_agent = plt.subplots(figsize=(9, 9))
        render_intersection_frame(ax_agent, frame_data_agent, frame_slider, num_steps)
        st.pyplot(fig_agent, use_container_width=True)

    with col2:
        st.subheader("Fixed Time Controller")
        fig_fixed, ax_fixed = plt.subplots(figsize=(9, 9))
        render_intersection_frame(ax_fixed, frame_data_fixed, frame_slider, num_steps)
        st.pyplot(fig_fixed, use_container_width=True)

    st.divider()

    # ==================== METRICS ====================
    st.subheader("📈 Performance Analysis")

    col1, col2, col3, col4 = st.columns(4)

    avg_queue_agent = (
        np.mean(frame_data_agent["queues_ns"]) + np.mean(frame_data_agent["queues_ew"])
    ) / 2
    total_reward_agent = np.sum(frame_data_agent["rewards"])

    avg_queue_fixed = (
        np.mean(frame_data_fixed["queues_ns"]) + np.mean(frame_data_fixed["queues_ew"])
    ) / 2
    total_reward_fixed = np.sum(frame_data_fixed["rewards"])

    with col1:
        st.metric("Q-Agent Reward", f"{total_reward_agent:.1f}")
    with col2:
        st.metric("Q-Agent Avg Queue", f"{avg_queue_agent:.1f}")
    with col3:
        st.metric("Fixed Reward", f"{total_reward_fixed:.1f}")
    with col4:
        st.metric("Fixed Avg Queue", f"{avg_queue_fixed:.1f}")

    st.divider()

    # Comparison charts
    st.subheader("📊 Detailed Comparison")

    tab1, tab2, tab3 = st.tabs(
        ["Rewards Over Time", "Queue Lengths", "Performance Summary"]
    )

    with tab1:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            frame_data_agent["rewards"],
            label="Q-Learning Agent",
            linewidth=2,
            color="green",
            alpha=0.7,
        )
        ax.plot(
            frame_data_fixed["rewards"],
            label="Fixed Controller",
            linewidth=2,
            color="orange",
            alpha=0.7,
        )
        ax.fill_between(
            range(len(frame_data_agent["rewards"])),
            frame_data_agent["rewards"],
            frame_data_fixed["rewards"],
            where=np.array(frame_data_agent["rewards"])
            >= np.array(frame_data_fixed["rewards"]),
            alpha=0.2,
            color="green",
            label="Agent Advantage",
        )
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Reward", fontsize=11)
        ax.set_title("Reward Comparison Over Time", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # NS queues
        axes[0].plot(frame_data_agent["queues_ns"], label="Q-Agent", linewidth=2)
        axes[0].plot(
            frame_data_fixed["queues_ns"], label="Fixed", linewidth=2, alpha=0.7
        )
        axes[0].set_title("North-South Queue", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Queue Length")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # EW queues
        axes[1].plot(frame_data_agent["queues_ew"], label="Q-Agent", linewidth=2)
        axes[1].plot(
            frame_data_fixed["queues_ew"], label="Fixed", linewidth=2, alpha=0.7
        )
        axes[1].set_title("East-West Queue", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Queue Length")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        st.pyplot(fig)

    with tab3:
        col1, col2, col3 = st.columns(3)

        improvement_reward = (
            (total_reward_agent - total_reward_fixed) / abs(total_reward_fixed)
        ) * 100
        improvement_queue = (
            (avg_queue_fixed - avg_queue_agent) / avg_queue_fixed
        ) * 100

        with col1:
            st.metric(
                "Reward Improvement",
                f"{improvement_reward:+.1f}%",
                delta=f"{improvement_reward:+.1f}",
            )
        with col2:
            st.metric(
                "Queue Reduction",
                f"{improvement_queue:+.1f}%",
                delta=f"{improvement_queue:+.1f}",
            )
        with col3:
            st.metric(
                "Efficiency Gain",
                f"{(improvement_reward + improvement_queue) / 2:+.1f}%",
            )

# ==================== FOOTER ====================
st.divider()
st.markdown(
    """
<div style="text-align: center; color: #666; padding: 20px;">
    <h3>🚦 Intelligent Traffic Signal Optimization</h3>
    <p>Using Q-Learning for Real-time Adaptive Signal Control</p>
    <p style="font-size: 12px;">Red cars (NS direction) • Cyan cars (EW direction) • Green signal = Active direction</p>
</div>
""",
    unsafe_allow_html=True,
)
