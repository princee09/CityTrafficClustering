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
        self.current_cluster = 0
        self.vehicles = {"ns": [], "ew": []}
        self.signal_state = "green-ns"
        self.signal_timer = 0
        self.time_step = 0
        self.avg_speed = 50.0
        return self.get_state()

    def get_state(self):
        return {
            "cluster": self.current_cluster,
            "queue_ns": self.get_queue_length("ns"),
            "queue_ew": self.get_queue_length("ew"),
            "signal_state": self.signal_state,
            "avg_speed": self.avg_speed,
        }

    def step(self, action):
        timings = self.apply_action(action)
        self.spawn_vehicles()
        self.update_signal(timings)
        passed = self.move_vehicles()

        if self.time_step % 100 == 0 and self.time_step > 0:
            self.current_cluster = (self.current_cluster + 1) % 4

        self.time_step += 1
        reward = self.calculate_reward(passed)

        info = {
            "avg_wait_time": self.get_avg_wait_time(),
            "total_queue": self.get_total_queue(),
            "vehicles_passed": passed,
        }

        return self.get_state(), reward, False, info

    def apply_action(self, action):
        timings = {"green-ns": 45, "green-ew": 35, "yellow": 3}
        if action == 1:
            timings["green-ns"] = 55
            timings["green-ew"] = 30
        elif action == 2:
            timings["green-ns"] = 35
            timings["green-ew"] = 45
        elif action == 3:
            timings["green-ns"] = 30
            timings["green-ew"] = 25
        elif action == 4:
            timings["green-ns"] = 60
            timings["green-ew"] = 45
        return timings

    def spawn_vehicles(self):
        scenario = self.scenarios.iloc[self.current_cluster]
        if np.random.rand() < scenario["spawn_rate"]:
            direction = "ns" if np.random.rand() < 0.5 else "ew"
            self.vehicles[direction].append(
                {
                    "position": 0,
                    "wait_time": 0,
                    "speed": scenario["Average Speed"] / 3.6,
                }
            )

    def update_signal(self, timings):
        self.signal_timer += 1
        if self.signal_state == "green-ns" and self.signal_timer >= timings["green-ns"]:
            self.signal_state, self.signal_timer = "yellow-ns", 0
        elif (
            self.signal_state == "yellow-ns" and self.signal_timer >= timings["yellow"]
        ):
            self.signal_state, self.signal_timer = "green-ew", 0
        elif (
            self.signal_state == "green-ew" and self.signal_timer >= timings["green-ew"]
        ):
            self.signal_state, self.signal_timer = "yellow-ew", 0
        elif (
            self.signal_state == "yellow-ew" and self.signal_timer >= timings["yellow"]
        ):
            self.signal_state, self.signal_timer = "green-ns", 0

    def move_vehicles(self):
        passed = 0
        for d in ["ns", "ew"]:
            can_move = (d == "ns" and "green-ns" in self.signal_state) or (
                d == "ew" and "green-ew" in self.signal_state
            )
            updated = []
            for v in self.vehicles[d]:
                if can_move or v["position"] < 50:
                    v["position"] += v["speed"]
                else:
                    v["wait_time"] += 1
                if v["position"] < 100:
                    updated.append(v)
                else:
                    passed += 1
            self.vehicles[d] = updated
        return passed

    def get_queue_length(self, d):
        return len(
            [v for v in self.vehicles[d] if v["position"] >= 50 and v["wait_time"] > 0]
        )

    def get_total_queue(self):
        return self.get_queue_length("ns") + self.get_queue_length("ew")

    def get_avg_wait_time(self):
        all_v = self.vehicles["ns"] + self.vehicles["ew"]
        if not all_v:
            return 0
        return np.mean([v["wait_time"] for v in all_v])

    def calculate_reward(self, passed):
        return (
            passed * 10 - 0.1 * self.get_avg_wait_time() - 0.5 * self.get_total_queue()
        )
