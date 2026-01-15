# 🚦 Traffic Signal Optimization & Simulation

An intelligent traffic management system combining **K-Means Clustering** for real-time traffic state classification and **Q-Learning** for adaptive signal control research. This project features a comprehensive web-based simulation to visualize traffic flow optimization based on 14 different environmental and traffic features.

## ✨ Key Features

-   **14-Feature Traffic Analysis**: Classifies traffic states using a rich feature set including:
    -   Traffic Volume & Average Speed
    -   Congestion Levels & Travel Time Index
    -   Weather Conditions (Rain, Fog, Wind)
    -   Road Incidents & Construction Activity
    -   Temporal factors (Weekend vs Weekday)
-   **Real-Time Simulation**: Interactive web interface (`simulation/`) to visualize adaptive signal control.
    -   Dynamic traffic generation based on cluster profiles.
    -   Visual traffic signal switching.
    -   Live metrics: Queue length, Wait times, Throughput.
-   **Machine Learning Engines**:
    -   **K-Means Clustering**: Used in the simulation to drive traffic scenarios (Free Flow, Normal, Peak, Incident).
    -   **Q-Learning Agent**: Research notebooks for training reinforcement learning agents on traffic data.

## 📂 Project Structure

```text
ML_traffic/
├── simulation/                  # 🌐 Web-based Traffic Simulator
│   ├── index.html               # Main simulation interface
│   ├── simulation.js            # Core simulation logic
│   ├── clusterModel.js          # K-Means logic for frontend
│   └── data/clusterParams.json  # Extracted cluster centroids
│
├── city-traffic-clustering/     # 🧠 ML Notebooks & Data
│   ├── notebooks/               # Jupyter notebooks (Q-Learning, Clustering)
│   ├── data/                    # Raw and processed datasets
│   └── src/                     # Python source for RL agents
│
└── scripts/                     # 🛠️ Utility Scripts
    └── extractClusterParams.py  # Extracts K-Means parameters for the sim
```

## Getting Started

### 1. Running the Simulation (Frontend)
The simulation is a static web application. You can run it directly:

1.  Navigate to the `simulation` folder.
2.  Open `index.html` in any modern web browser.
3.  Use the control panel to:
    -   Start/Pause simulation.
    -   Toggle between **Adaptive (ML)** and **Fixed-Time** modes.
    -   Adjust traffic density and simulation speed.

### 2. Python Setup (For ML & Scripts)
To run the analysis notebooks or the parameter extraction script, you need Python installed.

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Run the parameter extraction script (if you retrained the model)
python scripts/extractClusterParams.py
```

### 3. Training the Q-Learning Agent
Navigate to `city-traffic-clustering/notebooks/` and run `01_Q_Learning_Agent_Training.ipynb` to train the reinforcement learning agent.

## 📊 Traffic States
The system continuously monitors traffic and classifies it into 4 primary states:

| State | Color | Characteristics |
|-------|-------|-----------------|
| **Free Flow** | 🟢 | High speed, low volume, minimal delay. |
| **Normal** | 🔵 | Balanced flow, moderate speed. |
| **Peak** | 🟠 | High volume, reduced speed, increased queues. |
| **Incident** | 🔴 | Disrupted flow, very low speed, maximum congestion. |

## 🛠️ Technologies Used
-   **Frontend**: HTML5, CSS3, Vanilla JavaScript (Canvas API).
-   **ML/Backend**: Python, Pandas, Scikit-Learn.
-   **Data**: Real-world inspired traffic datasets for Bangalore.

---
*Created for Advanced Traffic Signal Control Research.*
