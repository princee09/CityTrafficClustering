"""
Intelligent Traffic Signal Control System
==========================================
A complete Python project comparing Fixed-Time and Q-Learning Agent-Based signal control
with predictive capabilities using Bangalore traffic clustering data.

Project Structure:
├── main.py                      # Main entry point (YOU RUN THIS)
├── config.py                    # Configuration settings
├── data_loader.py              # Data loading and preprocessing
├── traffic_environment.py      # Traffic simulation environment
├── signal_controllers.py       # Fixed and Agent-based controllers
├── q_learning_agent.py         # Q-Learning agent with prediction
├── predictor.py                # Traffic state predictor
├── metrics_tracker.py          # Performance metrics
├── visualization.py            # Pygame visualization
├── simulation_runner.py        # Simulation execution logic
└── utils.py                    # Utility functions

Requirements:
pip install numpy pandas matplotlib pygame scikit-learn
"""

# =============================================================================
# FILE 1: config.py
# =============================================================================
"""
Configuration settings for the traffic simulation
"""
import os


class Config:
    # Data paths
    DATA_DIR = "./"
    SCENARIOS_FILE = "traffic_scenarios_from_clusters_k4.csv"
    TRAFFIC_DATA_FILE = "bangalore_traffic_with_clusters_k4.csv"
    SIGNAL_TIMING_FILE = "Bangalore_Signal_Timing_20251130_1913.csv"

    # Simulation parameters
    N_CLUSTERS = 4
    CLUSTER_NAMES = ["Free Flow", "Normal", "Peak Congestion", "Incident/Disrupted"]

    # Q-Learning parameters
    Q_LEARNING_RATE = 0.1
    Q_DISCOUNT_FACTOR = 0.95
    Q_EPSILON_START = 1.0
    Q_EPSILON_DECAY = 0.995
    Q_EPSILON_MIN = 0.01
    Q_N_QUEUE_BINS = 10
    Q_N_ACTIONS = 5

    # Prediction parameters
    PREDICTION_WEIGHT = 0.3
    MARKOV_WEIGHT = 0.7
    TREND_WEIGHT = 0.3

    # Simulation settings
    DEFAULT_EPISODES = 5
    STEPS_PER_EPISODE = 1000
    VISUAL_STEPS = 2000

    # Visualization settings
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 900
    FPS = 30

    # Signal timing (default)
    BASE_GREEN_NS = 45
    BASE_GREEN_EW = 35
    YELLOW_TIME = 3

    # Output settings
    OUTPUT_DIR = "./results/"
    SAVE_PLOTS = True
    SAVE_MODELS = True

    @staticmethod
    def create_output_dir():
        """Create output directory if it doesn't exist"""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# =============================================================================
# FILE 2: data_loader.py
# =============================================================================
"""
Data loading and preprocessing
"""
import pandas as pd
import numpy as np
from config import Config


class DataLoader:
    def __init__(self, data_path=None):
        self.data_path = data_path or Config.DATA_DIR
        self.scenarios = None
        self.traffic_data = None
        self.signal_timings = None
        self.transition_matrix = None

    def load_all_data(self):
        """Load all required data files"""
        print("Loading data files...")

        # Load traffic scenarios
        self.scenarios = pd.read_csv(f"{self.data_path}{Config.SCENARIOS_FILE}")
        print(f"✓ Loaded {len(self.scenarios)} traffic scenarios")

        # Load historical traffic data
        self.traffic_data = pd.read_csv(f"{self.data_path}{Config.TRAFFIC_DATA_FILE}")
        print(f"✓ Loaded {len(self.traffic_data)} historical traffic records")

        # Load signal timing data
        self.signal_timings = pd.read_csv(
            f"{self.data_path}{Config.SIGNAL_TIMING_FILE}"
        )
        print(f"✓ Loaded {len(self.signal_timings)} signal timing records")

        # Calculate transition matrix
        self.transition_matrix = self.calculate_transition_matrix()
        print("✓ Calculated cluster transition matrix")

        return self

    def calculate_transition_matrix(self):
        """Calculate cluster transition probabilities"""
        clusters = self.traffic_data["cluster"].values
        n_clusters = Config.N_CLUSTERS
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
                transition_probs[i] = 1.0 / n_clusters

        return transition_probs

    def get_scenario(self, cluster_id):
        """Get traffic scenario for a specific cluster"""
        return self.scenarios.iloc[cluster_id]


# =============================================================================
# FILE 3: traffic_environment.py
# =============================================================================
"""
Traffic simulation environment
"""
import numpy as np


class TrafficEnvironment:
    def __init__(self, scenarios_df, traffic_data_df):
        self.scenarios = scenarios_df
        self.traffic_data = traffic_data_df
        self.current_cluster = 0
        self.vehicles = {"ns": [], "ew": []}
        self.signal_state = "green-ns"
        self.signal_timer = 0
        self.time_step = 0
        self.avg_speed = 50.0

    def reset(self):
        """Reset environment to initial state"""
        self.current_cluster = 0
        self.vehicles = {"ns": [], "ew": []}
        self.signal_state = "green-ns"
        self.signal_timer = 0
        self.time_step = 0
        self.avg_speed = 50.0
        return self.get_state()

    def get_state(self):
        """Get current state representation"""
        return {
            "cluster": self.current_cluster,
            "queue_ns": self.get_queue_length("ns"),
            "queue_ew": self.get_queue_length("ew"),
            "signal_state": self.signal_state,
            "avg_speed": self.avg_speed,
        }

    def step(self, action):
        """Execute one simulation step"""
        signal_timings = self.apply_action(action)
        self.spawn_vehicles()
        self.update_signal(signal_timings)
        vehicles_passed = self.move_vehicles()

        if self.time_step % 100 == 0 and self.time_step > 0:
            self.current_cluster = (self.current_cluster + 1) % 4

        self.time_step += 1
        reward = self.calculate_reward(vehicles_passed)
        next_state = self.get_state()

        info = {
            "avg_wait_time": self.get_avg_wait_time(),
            "total_queue": self.get_total_queue(),
            "vehicles_passed": vehicles_passed,
        }

        return next_state, reward, False, info

    def apply_action(self, action):
        """Apply action to determine signal timings"""
        from config import Config

        base_timings = {
            "green-ns": Config.BASE_GREEN_NS,
            "green-ew": Config.BASE_GREEN_EW,
            "yellow": Config.YELLOW_TIME,
        }

        if action == 1:  # Extend NS
            base_timings["green-ns"] = 55
            base_timings["green-ew"] = 30
        elif action == 2:  # Extend EW
            base_timings["green-ns"] = 35
            base_timings["green-ew"] = 45
        elif action == 3:  # Reduce cycle
            base_timings["green-ns"] = 30
            base_timings["green-ew"] = 25
        elif action == 4:  # Increase cycle
            base_timings["green-ns"] = 60
            base_timings["green-ew"] = 45

        return base_timings

    def spawn_vehicles(self):
        """Spawn vehicles based on current traffic state"""
        scenario = self.scenarios.iloc[self.current_cluster]
        spawn_rate = scenario["spawn_rate"]

        if np.random.random() < spawn_rate:
            direction = "ns" if np.random.random() < 0.5 else "ew"
            self.vehicles[direction].append(
                {
                    "position": 0,
                    "wait_time": 0,
                    "speed": scenario["Average Speed"] / 3.6,
                    "lane": 0 if np.random.random() < 0.5 else 1,
                }
            )

    def update_signal(self, timings):
        """Update traffic signal state"""
        self.signal_timer += 1

        if self.signal_state == "green-ns" and self.signal_timer >= timings["green-ns"]:
            self.signal_state = "yellow-ns"
            self.signal_timer = 0
        elif (
            self.signal_state == "yellow-ns" and self.signal_timer >= timings["yellow"]
        ):
            self.signal_state = "green-ew"
            self.signal_timer = 0
        elif (
            self.signal_state == "green-ew" and self.signal_timer >= timings["green-ew"]
        ):
            self.signal_state = "yellow-ew"
            self.signal_timer = 0
        elif (
            self.signal_state == "yellow-ew" and self.signal_timer >= timings["yellow"]
        ):
            self.signal_state = "green-ns"
            self.signal_timer = 0

    def move_vehicles(self):
        """Move vehicles and count passed"""
        passed = 0

        for direction in ["ns", "ew"]:
            can_move = (direction == "ns" and "green-ns" in self.signal_state) or (
                direction == "ew" and "green-ew" in self.signal_state
            )

            updated_vehicles = []
            for vehicle in self.vehicles[direction]:
                if can_move or vehicle["position"] < 50:
                    vehicle["position"] += vehicle["speed"]
                else:
                    vehicle["wait_time"] += 1

                if vehicle["position"] < 100:
                    updated_vehicles.append(vehicle)
                else:
                    passed += 1

            self.vehicles[direction] = updated_vehicles

        return passed

    def get_queue_length(self, direction):
        """Get queue length for a direction"""
        return len(
            [
                v
                for v in self.vehicles[direction]
                if v["position"] >= 50 and v["wait_time"] > 0
            ]
        )

    def get_total_queue(self):
        """Get total queue length"""
        return self.get_queue_length("ns") + self.get_queue_length("ew")

    def get_avg_wait_time(self):
        """Get average waiting time"""
        all_vehicles = self.vehicles["ns"] + self.vehicles["ew"]
        if not all_vehicles:
            return 0
        return np.mean([v["wait_time"] for v in all_vehicles])

    def calculate_reward(self, vehicles_passed):
        """Calculate reward for reinforcement learning"""
        wait_penalty = -self.get_avg_wait_time() * 0.1
        queue_penalty = -self.get_total_queue() * 0.5
        throughput_reward = vehicles_passed * 10
        return throughput_reward + wait_penalty + queue_penalty


# =============================================================================
# FILE 4: signal_controllers.py
# =============================================================================
"""
Signal controllers: Fixed-Time and Agent-Based
"""


class FixedTimeController:
    def __init__(self, signal_timings_df):
        self.timings = signal_timings_df
        self.standard_timing = 0

    def get_action(self, cluster):
        """Always return standard timing"""
        return 0


class AgentController:
    def __init__(self, agent, predictor):
        self.agent = agent
        self.predictor = predictor

    def get_action(self, cluster, queue_ns, queue_ew, prediction):
        """Get action from Q-learning agent with prediction"""
        queue_state = min(9, (queue_ns + queue_ew) // 5)
        state_idx = cluster * 10 + queue_state
        action = self.agent.choose_action(state_idx, prediction)
        return action


# =============================================================================
# FILE 5: q_learning_agent.py
# =============================================================================
"""
Q-Learning Agent with Predictive Hybrid Approach
"""
import numpy as np
from config import Config


class QLearningAgent:
    def __init__(
        self,
        n_states=None,
        n_queue_bins=None,
        n_actions=None,
        learning_rate=None,
        discount_factor=None,
        epsilon=None,
        epsilon_decay=None,
        epsilon_min=None,
    ):
        self.n_states = n_states or Config.N_CLUSTERS
        self.n_queue_bins = n_queue_bins or Config.Q_N_QUEUE_BINS
        self.n_actions = n_actions or Config.Q_N_ACTIONS
        self.state_size = self.n_states * self.n_queue_bins

        self.q_table = np.zeros((self.state_size, self.n_actions))

        self.lr = learning_rate or Config.Q_LEARNING_RATE
        self.gamma = discount_factor or Config.Q_DISCOUNT_FACTOR
        self.epsilon = epsilon or Config.Q_EPSILON_START
        self.epsilon_decay = epsilon_decay or Config.Q_EPSILON_DECAY
        self.epsilon_min = epsilon_min or Config.Q_EPSILON_MIN

        self.prediction_weight = Config.PREDICTION_WEIGHT

    def choose_action(self, state, prediction=None):
        """Choose action using hybrid Q-learning + prediction"""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            q_values = self.q_table[state].copy()

            if prediction is not None:
                if prediction > state // self.n_queue_bins:
                    q_values[1] += self.prediction_weight
                    q_values[2] += self.prediction_weight
                    q_values[4] += self.prediction_weight
                elif prediction < state // self.n_queue_bins:
                    q_values[3] += self.prediction_weight

            action = np.argmax(q_values)

        return action

    def learn(self, state, action, reward, next_state):
        """Update Q-table using Q-learning"""
        current_state = state["cluster"] * self.n_queue_bins + min(
            self.n_queue_bins - 1, (state["queue_ns"] + state["queue_ew"]) // 5
        )
        next_state_idx = next_state["cluster"] * self.n_queue_bins + min(
            self.n_queue_bins - 1,
            (next_state["queue_ns"] + next_state["queue_ew"]) // 5,
        )

        current_q = self.q_table[current_state][action]
        max_next_q = np.max(self.q_table[next_state_idx])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)

        self.q_table[current_state][action] = new_q
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filename="q_table.npy"):
        """Save Q-table"""
        np.save(filename, self.q_table)
        print(f"✓ Q-table saved to {filename}")

    def load_model(self, filename="q_table.npy"):
        """Load Q-table"""
        self.q_table = np.load(filename)
        print(f"✓ Q-table loaded from {filename}")


# =============================================================================
# FILE 6: predictor.py
# =============================================================================
"""
Traffic State Predictor using Markov Model and Trend Analysis
"""
import numpy as np
from config import Config


class TrafficPredictor:
    def __init__(self, transition_matrix, traffic_data):
        self.transition_matrix = transition_matrix
        self.traffic_data = traffic_data
        self.history = []
        self.markov_weight = Config.MARKOV_WEIGHT
        self.trend_weight = Config.TREND_WEIGHT

    def predict(self, current_cluster, queue_ns, queue_ew, avg_speed):
        """Predict next traffic state using Markov + Trend"""
        markov_pred = np.argmax(self.transition_matrix[current_cluster])

        total_queue = queue_ns + queue_ew

        if total_queue > 20 and avg_speed < 30:
            trend_pred = min(3, current_cluster + 1)
        elif total_queue < 5 and avg_speed > 50:
            trend_pred = max(0, current_cluster - 1)
        else:
            trend_pred = current_cluster

        if abs(markov_pred - trend_pred) <= 1:
            prediction = markov_pred
        else:
            prediction = int(
                self.markov_weight * markov_pred + self.trend_weight * trend_pred
            )

        self.history.append(
            {
                "current": current_cluster,
                "predicted": prediction,
                "markov": markov_pred,
                "trend": trend_pred,
            }
        )

        return prediction


# =============================================================================
# FILE 7: metrics_tracker.py
# =============================================================================
"""
Performance metrics tracker
"""
import numpy as np


class MetricsTracker:
    def __init__(self):
        self.metrics = {
            "wait_times": [],
            "queue_lengths": [],
            "throughput": [],
            "rewards": [],
        }

    def update(self, wait_time, queue_length, throughput, reward=None):
        """Update metrics"""
        self.metrics["wait_times"].append(wait_time)
        self.metrics["queue_lengths"].append(queue_length)
        self.metrics["throughput"].append(throughput)
        if reward is not None:
            self.metrics["rewards"].append(reward)

    def get_summary(self):
        """Get summary statistics"""
        return {
            "avg_wait": np.mean(self.metrics["wait_times"])
            if self.metrics["wait_times"]
            else 0,
            "avg_queue": np.mean(self.metrics["queue_lengths"])
            if self.metrics["queue_lengths"]
            else 0,
            "total_throughput": np.sum(self.metrics["throughput"]),
            "avg_reward": np.mean(self.metrics["rewards"])
            if self.metrics["rewards"]
            else 0,
        }


# =============================================================================
# FILE 8: visualization.py
# =============================================================================
"""
Real-time Pygame visualization for traffic simulation
"""
import pygame
import sys
from collections import deque


class TrafficVisualizer:
    def __init__(self, width=1400, height=900):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(
            "Intelligent Traffic Signal Control - Live Simulation"
        )

        # Colors
        self.colors = {
            "bg": (20, 20, 30),
            "road": (60, 60, 70),
            "line": (255, 200, 0),
            "green": (16, 185, 129),
            "yellow": (245, 158, 11),
            "red": (239, 68, 68),
            "blue": (59, 130, 246),
            "white": (255, 255, 255),
            "text": (200, 200, 210),
            "panel": (30, 30, 40),
            "cluster_colors": {
                0: (16, 185, 129),  # Free Flow - Green
                1: (59, 130, 246),  # Normal - Blue
                2: (245, 158, 11),  # Peak - Orange
                3: (239, 68, 68),  # Incident - Red
            },
        }

        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # Intersection position
        self.intersection_x = 400
        self.intersection_y = 400
        self.road_width = 120

        # Metrics history for graphs
        self.wait_history = deque(maxlen=100)
        self.queue_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)

        # Clock
        self.clock = pygame.time.Clock()
        self.fps = 30

    def draw_intersection(self):
        """Draw the road intersection"""
        # Vertical road (NS)
        pygame.draw.rect(
            self.screen,
            self.colors["road"],
            (
                self.intersection_x - self.road_width // 2,
                0,
                self.road_width,
                self.height,
            ),
        )

        # Horizontal road (EW)
        pygame.draw.rect(
            self.screen,
            self.colors["road"],
            (
                0,
                self.intersection_y - self.road_width // 2,
                self.width,
                self.road_width,
            ),
        )

        # Center intersection box
        pygame.draw.rect(
            self.screen,
            self.colors["panel"],
            (
                self.intersection_x - self.road_width // 2,
                self.intersection_y - self.road_width // 2,
                self.road_width,
                self.road_width,
            ),
        )

        # Lane markings
        for i in range(0, self.height, 40):
            pygame.draw.line(
                self.screen,
                self.colors["line"],
                (self.intersection_x, i),
                (self.intersection_x, i + 20),
                3,
            )

        for i in range(0, self.width, 40):
            pygame.draw.line(
                self.screen,
                self.colors["line"],
                (i, self.intersection_y),
                (i + 20, self.intersection_y),
                3,
            )

    def draw_traffic_lights(self, signal_state):
        """Draw traffic lights at intersection"""
        light_radius = 12

        # North light (top)
        ns_color = (
            self.colors["green"]
            if "green-ns" in signal_state
            else self.colors["yellow"]
            if "yellow-ns" in signal_state
            else self.colors["red"]
        )
        pygame.draw.circle(
            self.screen,
            ns_color,
            (self.intersection_x - 40, self.intersection_y - 80),
            light_radius,
        )

        # South light (bottom)
        pygame.draw.circle(
            self.screen,
            ns_color,
            (self.intersection_x + 40, self.intersection_y + 80),
            light_radius,
        )

        # East light (right)
        ew_color = (
            self.colors["green"]
            if "green-ew" in signal_state
            else self.colors["yellow"]
            if "yellow-ew" in signal_state
            else self.colors["red"]
        )
        pygame.draw.circle(
            self.screen,
            ew_color,
            (self.intersection_x + 80, self.intersection_y - 40),
            light_radius,
        )

        # West light (left)
        pygame.draw.circle(
            self.screen,
            ew_color,
            (self.intersection_x - 80, self.intersection_y + 40),
            light_radius,
        )

    def draw_vehicle(self, x, y, direction, stopped=False):
        """Draw a single vehicle"""
        color = self.colors["red"] if stopped else self.colors["blue"]

        if direction == "ns":
            pygame.draw.rect(
                self.screen, color, (x - 8, y - 12, 16, 24), border_radius=3
            )
        else:  # ew
            pygame.draw.rect(
                self.screen, color, (x - 12, y - 8, 24, 16), border_radius=3
            )

    def draw_vehicles(self, vehicles):
        """Draw all vehicles"""
        for direction in ["ns", "ew"]:
            for vehicle in vehicles[direction]:
                if direction == "ns":
                    # North-South: vehicles move downward
                    x = self.intersection_x + (
                        10 if vehicle.get("lane", 0) == 0 else -10
                    )
                    y = int(vehicle["position"] * 4)
                else:
                    # East-West: vehicles move rightward
                    x = int(vehicle["position"] * 4)
                    y = self.intersection_y + (
                        10 if vehicle.get("lane", 0) == 0 else -10
                    )

                self.draw_vehicle(x, y, direction, vehicle["wait_time"] > 0)

    def draw_info_panel(self, mode, cluster, state, time_step, scenario):
        """Draw information panel"""
        panel_x = 850
        panel_y = 50
        panel_width = 500

        # Title
        title = self.font_large.render(
            "Traffic Control System", True, self.colors["white"]
        )
        self.screen.blit(title, (panel_x, panel_y))

        y_offset = panel_y + 60

        # Mode
        mode_text = f"Mode: {'Fixed-Time' if mode == 'fixed' else 'Agent-Based (Q-Learning + Prediction)'}"
        mode_surface = self.font_medium.render(
            mode_text,
            True,
            self.colors["red"] if mode == "fixed" else self.colors["green"],
        )
        self.screen.blit(mode_surface, (panel_x, y_offset))
        y_offset += 40

        # Time
        time_text = f"Time: {time_step // 10}s"
        time_surface = self.font_small.render(time_text, True, self.colors["text"])
        self.screen.blit(time_surface, (panel_x, y_offset))
        y_offset += 35

        # Traffic State
        cluster_names = ["Free Flow", "Normal", "Peak Congestion", "Incident/Disrupted"]
        state_text = f"Traffic State: {cluster_names[cluster]}"
        state_surface = self.font_medium.render(
            state_text, True, self.colors["cluster_colors"][cluster]
        )
        self.screen.blit(state_surface, (panel_x, y_offset))
        y_offset += 40

        # Scenario details
        if scenario:
            details = [
                f"Volume: {scenario['Traffic Volume']:.0f} vehicles",
                f"Avg Speed: {scenario['Average Speed']:.1f} km/h",
                f"Congestion: {scenario['Congestion Level']:.2f}",
            ]
            for detail in details:
                detail_surface = self.font_small.render(
                    detail, True, self.colors["text"]
                )
                self.screen.blit(detail_surface, (panel_x + 20, y_offset))
                y_offset += 30

        y_offset += 20

        # Signal state
        signal_text = f"Signal: {state.replace('-', ' ').title()}"
        signal_surface = self.font_medium.render(
            signal_text, True, self.colors["white"]
        )
        self.screen.blit(signal_surface, (panel_x, y_offset))

    def draw_metrics_panel(self, queue_ns, queue_ew, avg_wait, throughput):
        """Draw real-time metrics"""
        panel_x = 850
        panel_y = 400

        # Panel background
        pygame.draw.rect(
            self.screen,
            self.colors["panel"],
            (panel_x - 10, panel_y - 10, 520, 250),
            border_radius=10,
        )

        # Title
        title = self.font_medium.render("Current Metrics", True, self.colors["white"])
        self.screen.blit(title, (panel_x, panel_y))

        y_offset = panel_y + 45

        metrics = [
            f"NS Queue: {queue_ns}",
            f"EW Queue: {queue_ew}",
            f"Total Queue: {queue_ns + queue_ew}",
            f"Avg Wait: {avg_wait:.1f}s",
            f"Throughput: {throughput}",
        ]

        for metric in metrics:
            metric_surface = self.font_small.render(metric, True, self.colors["text"])
            self.screen.blit(metric_surface, (panel_x + 10, y_offset))
            y_offset += 35

        # Update history
        self.wait_history.append(avg_wait)
        self.queue_history.append(queue_ns + queue_ew)
        self.throughput_history.append(throughput)

    def draw_agent_panel(self, prediction, last_action, q_values=None):
        """Draw agent decision panel"""
        panel_x = 850
        panel_y = 680

        # Panel background
        pygame.draw.rect(
            self.screen,
            self.colors["panel"],
            (panel_x - 10, panel_y - 10, 520, 180),
            border_radius=10,
        )

        # Title
        title = self.font_medium.render(
            "Agent Intelligence", True, self.colors["green"]
        )
        self.screen.blit(title, (panel_x, panel_y))

        y_offset = panel_y + 45

        # Prediction
        cluster_names = ["Free Flow", "Normal", "Peak Congestion", "Incident"]
        pred_text = f"Predicted State: {cluster_names[prediction]}"
        pred_surface = self.font_small.render(
            pred_text, True, self.colors["cluster_colors"][prediction]
        )
        self.screen.blit(pred_surface, (panel_x + 10, y_offset))
        y_offset += 35

        # Last action
        action_names = [
            "Standard",
            "Extend NS",
            "Extend EW",
            "Reduce Cycle",
            "Increase Cycle",
        ]
        action_text = f"Action Taken: {action_names[last_action]}"
        action_surface = self.font_small.render(
            action_text, True, self.colors["yellow"]
        )
        self.screen.blit(action_surface, (panel_x + 10, y_offset))
        y_offset += 35

        # Q-values visualization (if available)
        if q_values is not None:
            q_text = "Q-Values:"
            q_surface = self.font_small.render(q_text, True, self.colors["text"])
            self.screen.blit(q_surface, (panel_x + 10, y_offset))
            y_offset += 30

            # Draw small bar chart of Q-values
            max_q = max(q_values) if max(q_values) > 0 else 1
            for i, q_val in enumerate(q_values[:5]):
                bar_width = int((q_val / max_q) * 150) if max_q > 0 else 0
                pygame.draw.rect(
                    self.screen,
                    self.colors["blue"],
                    (panel_x + 100, y_offset + i * 15, max(bar_width, 1), 10),
                )
                val_text = f"{q_val:.1f}"
                val_surface = self.font_small.render(
                    val_text, True, self.colors["text"]
                )
                self.screen.blit(val_surface, (panel_x + 260, y_offset + i * 15 - 5))

    def draw_graph(self, data, x, y, width, height, title, color, max_val=None):
        """Draw a simple line graph"""
        if len(data) < 2:
            return

        # Background
        pygame.draw.rect(
            self.screen, self.colors["panel"], (x, y, width, height), border_radius=5
        )

        # Title
        title_surface = self.font_small.render(title, True, self.colors["white"])
        self.screen.blit(title_surface, (x + 10, y + 5))

        # Scale
        if max_val is None:
            max_val = max(data) if max(data) > 0 else 1

        # Draw line
        points = []
        for i, value in enumerate(data):
            px = x + 10 + (i / len(data)) * (width - 20)
            py = y + height - 10 - (value / max_val) * (height - 40)
            points.append((px, py))

        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 2)

    def render(self, env, mode, prediction=None, last_action=0, q_values=None):
        """Render complete frame"""
        self.screen.fill(self.colors["bg"])

        # Draw intersection and roads
        self.draw_intersection()

        # Draw traffic lights
        self.draw_traffic_lights(env.signal_state)

        # Draw vehicles
        self.draw_vehicles(env.vehicles)

        # Get current scenario
        scenario = env.scenarios.iloc[env.current_cluster]

        # Draw info panels
        self.draw_info_panel(
            mode, env.current_cluster, env.signal_state, env.time_step, scenario
        )

        # Draw metrics
        queue_ns = env.get_queue_length("ns")
        queue_ew = env.get_queue_length("ew")
        avg_wait = env.get_avg_wait_time()
        throughput = sum(
            [1 for d in ["ns", "ew"] for v in env.vehicles[d] if v["position"] >= 95]
        )

        self.draw_metrics_panel(queue_ns, queue_ew, avg_wait, throughput)

        # Draw agent panel (if in agent mode)
        if mode == "agent" and prediction is not None:
            self.draw_agent_panel(prediction, last_action, q_values)

        # Draw mini graphs
        if len(self.wait_history) > 1:
            self.draw_graph(
                list(self.wait_history),
                50,
                750,
                300,
                120,
                "Wait Time History",
                self.colors["red"],
                max_val=50,
            )

        if len(self.queue_history) > 1:
            self.draw_graph(
                list(self.queue_history),
                400,
                750,
                300,
                120,
                "Queue History",
                self.colors["yellow"],
                max_val=30,
            )

        pygame.display.flip()
        self.clock.tick(self.fps)

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def close(self):
        """Clean up pygame"""
        pygame.quit()
