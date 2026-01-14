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
        # Action determines signal timing adjustments
        signal_timings = self.apply_action(action)

        # Spawn vehicles based on current cluster
        self.spawn_vehicles()

        # Update signal
        self.update_signal(signal_timings)

        # Move vehicles
        vehicles_passed = self.move_vehicles()

        # Update cluster (change traffic state every 100 steps)
        if self.time_step % 100 == 0 and self.time_step > 0:
            self.current_cluster = (self.current_cluster + 1) % 4

        self.time_step += 1

        # Calculate reward
        reward = self.calculate_reward(vehicles_passed)

        # Get next state
        next_state = self.get_state()

        # Info
        info = {
            "avg_wait_time": self.get_avg_wait_time(),
            "total_queue": self.get_total_queue(),
            "vehicles_passed": vehicles_passed,
        }

        done = False  # Continuous simulation

        return next_state, reward, done, info

    def apply_action(self, action):
        """
        Apply action to determine signal timings
        Actions:
        0: Keep standard timing
        1: Extend NS green
        2: Extend EW green
        3: Reduce cycle time
        4: Increase cycle time
        """
        base_timings = {"green-ns": 45, "green-ew": 35, "yellow": 3}

        if action == 1:
            base_timings["green-ns"] = 55
            base_timings["green-ew"] = 30
        elif action == 2:
            base_timings["green-ns"] = 35
            base_timings["green-ew"] = 45
        elif action == 3:
            base_timings["green-ns"] = 30
            base_timings["green-ew"] = 25
        elif action == 4:
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
                    "speed": scenario["Average Speed"] / 3.6,  # km/h to m/s
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
        # Reward based on throughput and low wait times
        wait_penalty = -self.get_avg_wait_time() * 0.1
        queue_penalty = -self.get_total_queue() * 0.5
        throughput_reward = vehicles_passed * 10

        return throughput_reward + wait_penalty + queue_penalty
