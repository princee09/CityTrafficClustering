class FixedTimeController:
    def __init__(self, timings_df):
        self.timings_df = timings_df

    def get_action(self, cluster):
        return 0


class AgentController:
    def __init__(self, agent, predictor):
        self.agent = agent
        self.predictor = predictor

    def get_action(self, cluster, q_ns, q_ew, prediction):
        queue_state = min(9, (q_ns + q_ew) // 5)
        state_idx = cluster * 10 + queue_state
        return self.agent.choose_action(state_idx, prediction)
