"""
Streamlit Traffic Simulation with Pygame
Real-time interactive traffic game with Q-Learning agent control
"""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import pickle
import json
import pygame
from pygame.locals import QUIT


# Import project modules
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


# ==================== PYGAME SIMULATION ====================
class TrafficIntersectionGame:
    """Pygame-based traffic intersection simulation"""

    def __init__(self, width=800, height=800):
        self.width = width
        self.height = height
        self.running = False
        self.clock = pygame.time.Clock()
        self.fps = 30

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.DARK_GRAY = (64, 64, 64)
        self.YELLOW = (255, 255, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.CYAN = (0, 255, 255)
        self.LIGHT_RED = (255, 100, 100)
        self.LIGHT_CYAN = (100, 255, 255)

        # Game state
        self.ns_vehicles = []
        self.ew_vehicles = []
        self.signal_state = "green-ns"
        self.frame_count = 0

    def init_pygame(self):
        """Initialize Pygame"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("🚦 AI Traffic Control - Q-Learning Agent")
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)

    def draw_road(self):
        """Draw the road layout"""
        self.screen.fill(self.DARK_GRAY)

        # Vertical road (NS)
        pygame.draw.rect(self.screen, self.GRAY, (250, 0, 300, 800))

        # Horizontal road (EW)
        pygame.draw.rect(self.screen, self.GRAY, (0, 250, 800, 300))

        # Intersection
        pygame.draw.rect(self.screen, (100, 100, 100), (250, 250, 300, 300))

        # Road markings - vertical
        for y in range(0, 800, 40):
            pygame.draw.line(self.screen, self.YELLOW, (400, y), (400, y + 20), 3)

        # Road markings - horizontal
        for x in range(0, 800, 40):
            pygame.draw.line(self.screen, self.YELLOW, (x, 400), (x + 20, 400), 3)

    def draw_traffic_lights(self, signal_state):
        """Draw traffic lights"""
        light_size = 20

        # NS traffic light (left side)
        ns_light_color = self.GREEN if signal_state == "green-ns" else self.RED

        # Traffic light box
        pygame.draw.rect(self.screen, self.BLACK, (100, 350, 50, 100), 5)

        # Red light
        pygame.draw.circle(
            self.screen,
            self.RED if signal_state != "green-ns" else (80, 0, 0),
            (125, 365),
            light_size,
        )
        # Green light
        pygame.draw.circle(self.screen, ns_light_color, (125, 415), light_size)

        # EW traffic light (bottom side)
        ew_light_color = self.GREEN if signal_state == "green-ew" else self.RED

        # Traffic light box
        pygame.draw.rect(self.screen, self.BLACK, (350, 650, 100, 50), 5)

        # Red light
        pygame.draw.circle(
            self.screen,
            self.RED if signal_state != "green-ew" else (80, 0, 0),
            (365, 675),
            light_size,
        )
        # Green light
        pygame.draw.circle(self.screen, ew_light_color, (415, 675), light_size)

        # Labels
        label_ns = self.font_small.render("NS", True, self.WHITE)
        self.screen.blit(label_ns, (115, 330))

        label_ew = self.font_small.render("EW", True, self.WHITE)
        self.screen.blit(label_ew, (360, 625))

    def draw_vehicle(self, x, y, width, height, color, direction="ns"):
        """Draw a vehicle"""
        # Body
        pygame.draw.rect(self.screen, color, (x, y, width, height))

        # Window
        pygame.draw.rect(
            self.screen, self.CYAN, (x + 5, y + 5, width - 10, height - 10)
        )

        # Direction indicator
        if direction == "ns":
            pygame.draw.polygon(
                self.screen,
                self.WHITE,
                [
                    (x + width // 2 - 3, y),
                    (x + width // 2 + 3, y),
                    (x + width // 2, y - 5),
                ],
            )
        else:
            pygame.draw.polygon(
                self.screen,
                self.WHITE,
                [
                    (x + width, y + height // 2 - 3),
                    (x + width, y + height // 2 + 3),
                    (x + width + 5, y + height // 2),
                ],
            )

    def update_vehicles(self, queue_ns, queue_ew, signal_state):
        """Update vehicle positions based on queue and signal"""
        self.ns_vehicles = []
        self.ew_vehicles = []
        self.signal_state = signal_state

        # NS vehicles (moving north)
        for i in range(min(int(queue_ns), 15)):
            y = 200 - (i * 25)  # Position from intersection
            if y < 0:
                continue

            # Animate movement when green
            if signal_state == "green-ns":
                y -= (self.frame_count % 20) * 2

            self.ns_vehicles.append(
                {"x": 370, "y": y, "width": 30, "height": 40, "color": self.LIGHT_RED}
            )

        # EW vehicles (moving east)
        for i in range(min(int(queue_ew), 15)):
            x = 200 - (i * 25)  # Position from intersection
            if x < 0:
                continue

            # Animate movement when green
            if signal_state == "green-ew":
                x -= (self.frame_count % 20) * 2

            self.ew_vehicles.append(
                {"x": x, "y": 370, "width": 40, "height": 30, "color": self.LIGHT_CYAN}
            )

    def draw_ui(self, reward, queue_ns, queue_ew, step, total_steps):
        """Draw UI elements"""
        # Step counter
        step_text = self.font_medium.render(
            f"Step: {step}/{total_steps}", True, self.WHITE
        )
        self.screen.blit(step_text, (20, 20))

        # Reward display
        reward_color = self.GREEN if reward > 0 else self.RED
        reward_text = self.font_medium.render(
            f"Reward: {reward:.2f}", True, reward_color
        )
        self.screen.blit(reward_text, (20, 70))

        # Queue information
        ns_queue_text = self.font_small.render(
            f"NS Queue: {int(queue_ns)}", True, self.YELLOW
        )
        self.screen.blit(ns_queue_text, (20, 150))

        ew_queue_text = self.font_small.render(
            f"EW Queue: {int(queue_ew)}", True, self.YELLOW
        )
        self.screen.blit(ew_queue_text, (20, 190))

        # Signal state
        signal_text = self.font_small.render(
            f"Signal: {self.signal_state.replace('-', ' ').upper()}",
            True,
            self.GREEN if "green" in self.signal_state else self.RED,
        )
        self.screen.blit(signal_text, (600, 20))

    def render_frame(self, state, reward, step, total_steps):
        """Render a single frame"""
        self.draw_road()
        self.draw_traffic_lights(state.get("signal_state", "green-ns"))

        # Update and draw vehicles
        self.update_vehicles(
            state.get("queue_ns", 0),
            state.get("queue_ew", 0),
            state.get("signal_state", "green-ns"),
        )

        for vehicle in self.ns_vehicles:
            self.draw_vehicle(
                vehicle["x"],
                vehicle["y"],
                vehicle["width"],
                vehicle["height"],
                vehicle["color"],
                "ns",
            )

        for vehicle in self.ew_vehicles:
            self.draw_vehicle(
                vehicle["x"],
                vehicle["y"],
                vehicle["width"],
                vehicle["height"],
                vehicle["color"],
                "ew",
            )

        # Draw UI
        self.draw_ui(
            reward,
            state.get("queue_ns", 0),
            state.get("queue_ew", 0),
            step,
            total_steps,
        )

        # Direction labels
        label_north = self.font_small.render("↑ NORTH", True, self.WHITE)
        self.screen.blit(label_north, (350, 10))

        label_east = self.font_small.render("EAST →", True, self.WHITE)
        self.screen.blit(label_east, (750, 380))

        self.frame_count += 1
        pygame.display.flip()
        self.clock.tick(self.fps)


def run_pygame_simulation(env, agent, num_steps=100):
    """Run the Pygame-based simulation"""
    game = TrafficIntersectionGame(width=800, height=800)
    game.init_pygame()

    state = env.reset()

    for step in range(num_steps):
        # Get action and step environment
        action = agent.choose_action(state)
        next_state, reward, _, info = env.step(action)

        # Render frame
        game.render_frame(next_state, reward, step + 1, num_steps)

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

        state = next_state

    pygame.quit()


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Traffic Control - Pygame",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 AI Traffic Signal Optimization - Pygame Edition")
st.markdown("**Interactive Game-like Simulation with Moving Vehicles**")
st.divider()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Configuration")

    # Load data
    config_path = os.path.join(proj_root, "results", "agent_config.json")
    with open(config_path, "r") as f:
        agent_config = json.load(f)

    scenarios_path = os.path.join(
        proj_root, "data", "processed", "traffic_scenarios_from_clusters_k4.csv"
    )
    scenarios = pd.read_csv(scenarios_path)

    traffic_data_path = os.path.join(
        proj_root, "data", "processed", "bangalore_traffic_with_clusters_k4.csv"
    )
    traffic_data = pd.read_csv(traffic_data_path)

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
        max_value=200,
        value=100,
        step=10,
    )

    # Scenario info
    st.subheader("📋 Scenario Details")
    scenario_data = scenarios[scenarios["cluster_id"] == cluster_id].iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Traffic Volume", f"{scenario_data['Traffic Volume']:.0f} veh/hr")
        st.metric("Congestion", f"{scenario_data['Congestion Level']:.1f}%")
    with col2:
        st.metric("Avg Speed", f"{scenario_data['Average Speed']:.1f} km/h")
        st.metric("Incident Prob", f"{scenario_data['incident_prob']:.2%}")

# ==================== MAIN CONTENT ====================
st.info("""
### 🎮 How to Use:
1. Configure the scenario in the sidebar
2. Click "**▶️ START PYGAME SIMULATION**" to watch the game
3. **Red vehicles** = North-South direction
4. **Cyan vehicles** = East-West direction
5. **Green light** = That direction can go
6. Watch how Q-Learning optimizes traffic flow!
""")

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
except Exception as e:
    st.warning(f"⚠️ Could not load trained agent: {e}")
    agent = QLearningAgent(
        n_states=4, n_queue_bins=10, n_actions=5, lr=0.1, gamma=0.95, epsilon=0.0
    )

# Run simulation
if st.button("▶️ START PYGAME SIMULATION", key="run_sim", use_container_width=True):
    st.info("🎮 Opening Pygame window... Watch your traffic intersection simulation!")

    # Create environment
    env = TrafficEnvironment(scenarios_df=scenarios, traffic_data_df=traffic_data)

    # Run pygame simulation
    try:
        run_pygame_simulation(env, agent, num_steps)
        st.success("✅ Simulation completed! The game window has closed.")
    except Exception as e:
        st.error(f"❌ Error running simulation: {e}")

st.divider()

# ==================== INFORMATION ====================
st.subheader("📚 About This Simulation")
st.markdown("""
**Features:**
- 🚗 Real-time animated vehicles moving through intersection
- 🚦 Smart traffic lights controlled by Q-Learning agent
- 📊 Live reward and queue display
- 🎮 Game-like interactive visualization
- 🔴 Red vehicles (North-South direction)
- 🔵 Cyan vehicles (East-West direction)
- ✨ Smooth animations and realistic traffic behavior

**Q-Learning Agent:**
- Learns optimal signal timing strategies
- Minimizes queue lengths and wait times
- Adapts to different traffic scenarios
- Trained on Bangalore traffic data

**Traffic Scenarios:**
- 🟢 Free Flow: Light traffic, smooth flow
- 🟡 Normal: Regular traffic conditions
- 🔴 Peak: Heavy congestion
- ⚠️ Incident: Disrupted traffic flow
""")

st.markdown(
    """
---
<div style="text-align: center; color: #666;">
    <p>🚦 AI-Powered Traffic Signal Control System</p>
    <p style="font-size: 12px;">Using Reinforcement Learning for Intelligent Adaptive Signals</p>
</div>
""",
    unsafe_allow_html=True,
)
