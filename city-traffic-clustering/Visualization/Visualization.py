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
