# 🚦 Q-Learning Traffic Signal Optimization System

A comprehensive intelligent traffic management system using **Q-Learning reinforcement learning** to automatically optimize traffic signal timings in Bangalore. This project includes a Jupyter notebook for agent training and a Streamlit web app for real-time simulation and visualization.

## 📋 Project Overview

This system uses **Tabular Q-Learning** to train an agent that dynamically controls traffic signal timings to:
- ✅ Minimize vehicle queue lengths
- ✅ Reduce average wait times
- ✅ Maximize vehicle throughput
- ✅ Handle different traffic scenarios (Free Flow, Normal, Peak Congestion, Incidents)

**Key Technologies:**
- Python 3.8+
- Reinforcement Learning (Q-Learning)
- Jupyter Notebooks (training)
- Streamlit (interactive dashboard)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn (visualization)

## 🗂️ Project Structure

```
city-traffic-clustering/
├── notebooks/
│   └── 01_Q_Learning_Agent_Training.ipynb    # Agent training notebook (MAIN)
├── data/
│   └── processed/
│       ├── traffic_scenarios_from_clusters_k4.csv    # Cluster profiles
│       └── bangalore_traffic_with_clusters_k4.csv    # Historical data
├── src/
│   ├── traffic_environment.py     # Traffic simulation environment
│   ├── q_learning_agent.py        # Q-Learning agent implementation
│   ├── signal_controllers.py      # Signal control strategies
│   ├── metrics_tracker.py         # Performance metrics
│   └── predictor.py               # Traffic prediction
├── results/
│   ├── trained_q_agent.pkl        # Saved agent Q-table
│   ├── training_history.json      # Training metrics history
│   └── learning_curves.png        # Training visualization
├── streamlit_app.py               # Interactive Streamlit dashboard
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create a Python virtual environment (optional but recommended)
python -m venv traffic_env
source traffic_env/bin/activate  # On Windows: traffic_env\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn streamlit jupyter
```

### 2. Train the Q-Learning Agent

Run the training notebook to create and train the agent:

```bash
# Using Jupyter
jupyter notebook notebooks/01_Q_Learning_Agent_Training.ipynb

# Or using JupyterLab
jupyter lab notebooks/01_Q_Learning_Agent_Training.ipynb
```

**In the notebook:**
1. **Section 1-3**: Loads traffic data, configures environment
2. **Section 4**: Initializes Q-Learning agent
3. **Section 5**: Trains agent for 100 episodes (⏱️ ~2-3 minutes)
4. **Section 6**: Visualizes learning curves
5. **Section 7**: Saves trained agent to `results/trained_q_agent.pkl`

### 3. Run Streamlit Dashboard

After training, launch the interactive simulation:

```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`

## 📊 Q-Learning Agent Details

### State Space
- **4 Traffic Scenarios**: Free Flow, Normal, Peak, Incident
- **10 Queue Bins**: Discretized vehicle queue lengths
- **Total States**: 40 (4 clusters × 10 queue bins)

### Action Space
| Action | Description |
|--------|-------------|
| 0 | Default (no change) |
| 1 | Extend NS green |
| 2 | Extend EW green |
| 3 | Reduce cycle time |
| 4 | Increase cycle time |

### Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| **α** (Learning Rate) | 0.1 | How fast agent learns |
| **γ** (Discount Factor) | 0.95 | Importance of future rewards |
| **ε** (Initial Epsilon) | 1.0 | Exploration probability |
| **ε decay** | 0.995 | Exploration reduction per step |
| **ε min** | 0.01 | Minimum exploration rate |

### Learning Equation

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

Where:
- $s$ = current state
- $a$ = action taken
- $r$ = reward received
- $s'$ = next state
- $\alpha$ = learning rate
- $\gamma$ = discount factor

## 🎮 Streamlit Dashboard Features

### Left Sidebar Controls
- **Select Control Strategy**: Q-Learning Agent, Fixed Time, or Comparison
- **Traffic Scenario**: Choose from 4 traffic conditions
- **Simulation Duration**: 50-500 steps (1 step ≈ 2 seconds)
- **Advanced Settings**: Adjust signal timing parameters

### Main Dashboard
1. **Performance Metrics**: Real-time statistics (reward, queue, wait time, throughput)
2. **Comparison Charts**: Side-by-side comparison of strategies
3. **Learning Curves**: Reward, queue, wait time, throughput trends
4. **Summary Statistics**: Detailed performance breakdown

## 📈 Expected Results

After 100 episodes of training, the agent typically achieves:

| Metric | Improvement |
|--------|------------|
| Queue Length | ↓ 20-30% reduction |
| Average Wait Time | ↓ 15-25% reduction |
| Total Throughput | ↑ 10-20% increase |
| Reward | +50-100% improvement |

*Results vary based on traffic scenario and random seed*

## 🔄 Traffic Scenarios

### Cluster 0: Free Flow (9.15% of data)
- Traffic Volume: 32,187 vehicles/hr
- Average Speed: 38.4 km/h
- Congestion: 88.9%
- Spawn Rate: 25.5 vehicles/min

### Cluster 1: Normal (14.66% of data)
- Traffic Volume: 11,192 vehicles/hr
- Average Speed: 45.0 km/h
- Congestion: 37.2%
- Spawn Rate: 8.9 vehicles/min

### Cluster 2: Peak Congestion (24.12% of data)
- Traffic Volume: 20,425 vehicles/hr
- Average Speed: 45.1 km/h
- Congestion: 68.2%
- Spawn Rate: 16.2 vehicles/min

### Cluster 3: Incident/Disrupted (52.07% of data)
- Traffic Volume: 37,878 vehicles/hr
- Average Speed: 35.4 km/h
- Congestion: 97.6%
- Spawn Rate: 30.0 vehicles/min
- Incident Probability: 1.0 (high)

## 📁 Input Data Format

### traffic_scenarios_from_clusters_k4.csv
Cluster profiles with:
- `cluster_id`, `traffic_state_name`, `Traffic Volume`, `Average Speed`, `Congestion Level`, `spawn_rate`, `incident_prob`

### bangalore_traffic_with_clusters_k4.csv
Historical traffic data with:
- `Traffic Volume`, `Average Speed`, `Congestion Level`, `cluster` labels
- Weather conditions, incident reports, road capacity utilization

## 🎯 Training Tips

1. **Increase Episodes**: For better convergence, modify `n_episodes` in the notebook (e.g., 500)
2. **Adjust Learning Rate**: Increase `α` (0.1→0.2) for faster learning, decrease for stability
3. **Modify Epsilon Decay**: Faster decay (0.995→0.98) encourages earlier exploitation
4. **Try Different Scenarios**: Train on specific clusters for specialized control
5. **Monitor Metrics**: Watch for reward convergence and queue reduction trends

## 🐛 Troubleshooting

### "Could not load trained agent"
- Ensure you've run the training notebook first
- Check `results/trained_q_agent.pkl` exists
- Verify file permissions

### Slow simulation
- Reduce `num_steps` in Streamlit app
- Close other applications
- Use faster hardware

### Import errors
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check project root is correctly detected

## 📚 References

1. **Sutton & Barto** (2018). *Reinforcement Learning: An Introduction*
2. **Watkins & Dayan** (1992). Q-Learning paper
3. **Traffic Signal Control Literature**: Deep RL for traffic signal timing optimization

## 🤝 Contributing

Improvements welcome! Consider:
- Deep Q-Networks (DQN) for larger state spaces
- Multi-agent Q-Learning for multiple intersections
- Real-time data integration
- Policy gradient methods (A3C, PPO)

## 📄 License

MIT License - Feel free to use for research, education, or commercial projects.

## 👨‍💻 Author

Created as an intelligent traffic management solution for Bangalore urban traffic optimization.

---

**Questions?** Refer to the Jupyter notebook for detailed implementation, or check the Streamlit app code comments.

Happy traffic optimizing! 🚦✨
