/**
 * Main Application Controller
 * Orchestrates the entire traffic simulation system
 */

// Global state
let simulation = null;
let visualizer = null;
let signalController = null;
let clusterModel = null;
let metricsCollector = null;

let isRunning = false;
let lastFrameTime = 0;
let animationFrameId = null;
let currentClassification = null;
let simulationSpeed = 1.0;
let trafficDensity = 3;

// DOM Elements
const canvas = document.getElementById('trafficCanvas');
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');
const adaptiveModeBtn = document.getElementById('adaptiveMode');
const fixedModeBtn = document.getElementById('fixedMode');
const trafficDensitySlider = document.getElementById('trafficDensity');
const densityValueLabel = document.getElementById('densityValue');
const simulationSpeedSlider = document.getElementById('simulationSpeed');
const speedValueLabel = document.getElementById('speedValue');

/**
 * Initialize the application
 */
async function init() {
    console.log('🚀 Initializing Traffic Simulation System...');

    // Initialize cluster model
    clusterModel = new ClusterModel();
    await clusterModel.load();

    // Initialize components
    visualizer = new TrafficVisualizer(canvas);
    simulation = new TrafficSimulation(canvas.width, canvas.height);
    signalController = new TrafficSignalController('adaptive');
    metricsCollector = new MetricsCollector();

    // Setup event listeners
    setupEventListeners();

    // Initial render
    visualizer.draw(simulation, signalController);
    updateUI();

    console.log('✅ Simulation initialized successfully!');
}

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    startBtn.addEventListener('click', startSimulation);
    pauseBtn.addEventListener('click', pauseSimulation);
    resetBtn.addEventListener('click', resetSimulation);

    adaptiveModeBtn.addEventListener('click', () => setControlMode('adaptive'));
    fixedModeBtn.addEventListener('click', () => setControlMode('fixed'));

    trafficDensitySlider.addEventListener('input', (e) => {
        trafficDensity = parseInt(e.target.value);
        updateDensityLabel();
    });

    simulationSpeedSlider.addEventListener('input', (e) => {
        simulationSpeed = parseFloat(e.target.value);
        updateSpeedLabel();
    });

    window.addEventListener('resize', () => {
        visualizer.resize();
        simulation.resize(canvas.width, canvas.height);
    });
}

/**
 * Start the simulation
 */
function startSimulation() {
    if (isRunning) return;

    isRunning = true;
    lastFrameTime = performance.now();

    startBtn.disabled = true;
    pauseBtn.disabled = false;

    updateStatusBadge('Running', true);

    animationFrameId = requestAnimationFrame(gameLoop);

    console.log('▶️ Simulation started');
}

/**
 * Pause the simulation
 */
function pauseSimulation() {
    if (!isRunning) return;

    isRunning = false;

    startBtn.disabled = false;
    pauseBtn.disabled = true;

    updateStatusBadge('Paused', false);

    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }

    console.log('⏸️ Simulation paused');
}

/**
 * Reset the simulation
 */
function resetSimulation() {
    pauseSimulation();

    simulation.reset();
    signalController.reset();
    metricsCollector.reset();

    visualizer.draw(simulation, signalController);
    updateUI();

    updateStatusBadge('Ready', true);

    console.log('↻ Simulation reset');
}

/**
 * Set control mode (adaptive or fixed)
 */
function setControlMode(mode) {
    signalController.setMode(mode);

    if (mode === 'adaptive') {
        adaptiveModeBtn.classList.add('active');
        fixedModeBtn.classList.remove('active');
    } else {
        fixedModeBtn.classList.add('active');
        adaptiveModeBtn.classList.remove('active');
    }

    console.log(`🎛️ Control mode: ${mode}`);
}

/**
 * Main game loop
 */
function gameLoop(currentTime) {
    if (!isRunning) return;

    // Calculate delta time (with speed multiplier)
    const deltaTime = (currentTime - lastFrameTime) * simulationSpeed;
    lastFrameTime = currentTime;

    // Get current traffic metrics
    const trafficMetrics = simulation.getTrafficMetrics();

    // Classify traffic state using ML model
    if (clusterModel.isLoaded) {
        currentClassification = clusterModel.classify(trafficMetrics);
    }

    // Update signal controller (with classification if in adaptive mode)
    signalController.update(deltaTime, currentClassification);

    // Update simulation
    const densityMultiplier = trafficDensity / 3; // Normalize around medium
    simulation.update(deltaTime, signalController, densityMultiplier);

    // Update metrics
    metricsCollector.update(trafficMetrics);

    // Render
    visualizer.draw(simulation, signalController);

    // Update UI (throttled to every 10 frames for performance)
    if (Math.random() < 0.1) {
        updateUI();
    }

    // Continue loop
    animationFrameId = requestAnimationFrame(gameLoop);
}

/**
 * Update all UI elements
 */
function updateUI() {
    updateTrafficState();
    updateMetrics();
    updateLaneStatus();
    updateKMeansVisualization();  // New: K-Means educational display
    updateClusterCentroids();      // New: Cluster centroids display
    updateClusterTrafficLight();   // New: Cluster traffic light indicator
}

/**
 * Update cluster traffic light indicator
 */
function updateClusterTrafficLight() {
    if (!currentClassification) return;

    const { clusterId, trafficState } = currentClassification;

    // Update all lights - deactivate all first
    for (let i = 0; i < 4; i++) {
        const light = document.getElementById(`light${i}`);
        if (light) {
            light.classList.remove('active');
        }
    }

    // Activate current cluster light
    const activeLight = document.getElementById(`light${clusterId}`);
    if (activeLight) {
        activeLight.classList.add('active');
    }

    // Update label
    const labelState = document.getElementById('clusterLabelState');
    if (labelState) {
        labelState.textContent = trafficState.name;
        labelState.style.color = trafficState.color;
    }

    // Update badge
    const badgeCluster = document.getElementById('badgeCluster');
    if (badgeCluster) {
        badgeCluster.textContent = `Cluster ${clusterId}`;
        badgeCluster.style.background = `linear-gradient(135deg, ${trafficState.color}, ${adjustColor(trafficState.color, -20)})`;
    }
}

/**
 * Update K-Means classification visualization with all 14 features
 */
function updateKMeansVisualization() {
    if (!currentClassification) return;

    const { metrics, distances, clusterId, trafficState, confidence } = currentClassification;

    // Update section heading to show number of features
    const sectionTitle = document.querySelector('.metrics-panel .section-title:nth-of-type(3)');
    if (sectionTitle) {
        sectionTitle.innerHTML = `
            <span class="icon">🔬</span>
            K-Means Classification (14 Features)
        `;
    }

    // === Group 1: Traffic Flow Metrics (Core) ===
    const featureVehiclesElem = document.getElementById('featureVehicles');
    const featureSpeedElem = document.getElementById('featureSpeed');
    const featureCongestionElem = document.getElementById('featureCongestion');

    if (featureVehiclesElem) {
        featureVehiclesElem.textContent = Math.round(metrics.traffic_volume);
    }
    if (featureSpeedElem) {
        featureSpeedElem.textContent = Math.round(metrics.average_speed);
    }
    if (featureCongestionElem) {
        featureCongestionElem.textContent = Math.round(metrics.congestion_level);
    }

    // Update feature bars (normalize to 0-100%)
    const barVehicles = document.getElementById('barVehicles');
    const barSpeed = document.getElementById('barSpeed');
    const barCongestion = document.getElementById('barCongestion');

    if (barVehicles) {
        // Volume range: 4K-72K
        barVehicles.style.width = `${Math.min(100, ((metrics.traffic_volume - 4000) / (72000 - 4000)) * 100)}%`;
    }
    if (barSpeed) {
        // Speed range: 20-89 km/h
        barSpeed.style.width = `${Math.min(100, ((metrics.average_speed - 20) / (89 - 20)) * 100)}%`;
    }
    if (barCongestion) {
        barCongestion.style.width = `${metrics.congestion_level}%`;
    }

    // Update cluster distances
    const distanceList = document.getElementById('distanceList');
    if (distanceList) {
        distanceList.innerHTML = '';

        if (distances && distances.length > 0) {
            distances.forEach(dist => {
                const distItem = document.createElement('div');
                distItem.className = 'distance-item';
                if (dist.clusterId === clusterId) {
                    distItem.classList.add('selected');
                }

                distItem.innerHTML = `
                    <div class="distance-item-left">
                        <div class="cluster-dot" style="background: ${dist.trafficState.color}"></div>
                        <span class="distance-cluster-name">${dist.trafficState.name}</span>
                    </div>
                    <span class="distance-value">${dist.distance.toFixed(3)}</span>
                `;

                distanceList.appendChild(distItem);
            });
        }
    }

    // Update classification result
    const resultCluster = document.getElementById('resultCluster');
    if (resultCluster) {
        resultCluster.innerHTML = `
            <span class="cluster-badge">Cluster ${clusterId}</span>
            <span class="cluster-name">${trafficState.name}</span>
        `;
    }

    const confidenceValue = document.getElementById('confidenceValue');
    if (confidenceValue) {
        confidenceValue.textContent = `${Math.round(confidence)}%`;
    }

    // Add detailed feature information as tooltip or additional display
    // This helps users understand all 14 features without overwhelming the UI
    updateFeatureTooltip(metrics);
}

/**
 * Update feature tooltip showing all 14 features in detail
 */
function updateFeatureTooltip(metrics) {
    // Create or update tooltip element
    let tooltip = document.getElementById('featureDetailsTooltip');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.id = 'featureDetailsTooltip';
        tooltip.className = 'feature-details-tooltip';
        tooltip.style.display = 'none';

        // Find a good place to add it - in the K-Means section
        const kmeansSection = document.querySelector('.kmeans-explainer');
        if (kmeansSection) {
            kmeansSection.appendChild(tooltip);
        }
    }

    // Weather status
    let weatherStatus = 'Clear';
    if (metrics.weather_conditions_fog === 1) weatherStatus = 'Foggy';
    else if (metrics.weather_conditions_overcast === 1) weatherStatus = 'Overcast';
    else if (metrics.weather_conditions_rain === 1) weatherStatus = 'Rainy';
    else if (metrics.weather_conditions_windy === 1) weatherStatus = 'Windy';

    tooltip.innerHTML = `
        <div class="tooltip-header">All 14 Model Features</div>
        <div class="tooltip-grid">
            <div class="tooltip-group">
                <div class="tooltip-group-title">Traffic Flow</div>
                <div class="tooltip-item">Volume: ${Math.round(metrics.traffic_volume)}</div>
                <div class="tooltip-item">Avg Speed: ${Math.round(metrics.average_speed)} km/h</div>
                <div class="tooltip-item">Congestion: ${Math.round(metrics.congestion_level)}%</div>
                <div class="tooltip-item">Travel Time Index: ${metrics.travel_time_index.toFixed(2)}</div>
            </div>
            <div class="tooltip-group">
                <div class="tooltip-group-title">Road Conditions</div>
                <div class="tooltip-item">Capacity: ${Math.round(metrics.road_capacity_utilization)}%</div>
                <div class="tooltip-item">Incidents: ${metrics.incident_reports}</div>
                <div class="tooltip-item">Roadwork: ${metrics.roadwork_and_construction_activity_yes === 1 ? 'Yes' : 'No'}</div>
            </div>
            <div class="tooltip-group">
                <div class="tooltip-group-title">Environmental</div>
                <div class="tooltip-item">Impact: ${Math.round(metrics.environmental_impact)}</div>
                <div class="tooltip-item">Pedestrians: ${Math.round(metrics.pedestrian_and_cyclist_count)}</div>
            </div>
            <div class="tooltip-group">
                <div class="tooltip-group-title">Temporal & Weather</div>
                <div class="tooltip-item">Day Type: ${metrics.isweekend === 1 ? 'Weekend' : 'Weekday'}</div>
                <div class="tooltip-item">Weather: ${weatherStatus}</div>
            </div>
        </div>
    `;
}

/**
 * Update cluster centroids display
 */
function updateClusterCentroids() {
    if (!clusterModel || !clusterModel.isLoaded) return;

    const centroidsContainer = document.getElementById('clusterCentroids');

    // Only populate once
    if (centroidsContainer.children.length > 0 && !currentClassification) return;

    centroidsContainer.innerHTML = '';

    const clusters = clusterModel.clusters;
    const currentClusterId = currentClassification ? currentClassification.clusterId : null;

    Object.values(clusters).forEach(cluster => {
        const centroidCard = document.createElement('div');
        centroidCard.className = 'centroid-card';
        if (cluster.id === currentClusterId) {
            centroidCard.classList.add('active');
        }

        const features = cluster.features;

        centroidCard.innerHTML = `
            <div class="centroid-header">
                <div class="centroid-dot" style="background: ${cluster.traffic_state.color}"></div>
                <div class="centroid-label">
                    <div class="centroid-id">Cluster ${cluster.id}</div>
                    <div class="centroid-state">${cluster.traffic_state.name}</div>
                </div>
            </div>
            <div class="centroid-features">
                <div class="centroid-feature">
                    <span class="centroid-feature-label">Vehicles</span>
                    <span class="centroid-feature-value">${Math.round(features.vehicle_count.mean)}</span>
                </div>
                <div class="centroid-feature">
                    <span class="centroid-feature-label">Speed</span>
                    <span class="centroid-feature-value">${Math.round(features.average_speed.mean)}</span>
                </div>
                <div class="centroid-feature">
                    <span class="centroid-feature-label">Congest</span>
                    <span class="centroid-feature-value">${Math.round(features.congestion_level.mean * 100)}%</span>
                </div>
            </div>
        `;

        centroidsContainer.appendChild(centroidCard);
    });
}

/**
 * Update traffic state display
 */
function updateTrafficState() {
    if (!currentClassification) return;

    const stateDisplay = document.getElementById('currentState');
    const { trafficState, signalTiming } = currentClassification;

    const stateBadge = stateDisplay.querySelector('.state-badge');
    stateBadge.style.background = `linear-gradient(135deg, ${trafficState.color}, ${adjustColor(trafficState.color, -20)})`;

    stateDisplay.querySelector('.state-name').textContent = trafficState.name;
    stateDisplay.querySelector('.state-desc').textContent = trafficState.description;

    const params = stateDisplay.querySelectorAll('.param-value');
    params[0].textContent = `${signalTiming.green_time}s`;
    params[1].textContent = `${signalTiming.cycle_length}s`;
}

/**
 * Update performance metrics
 */
function updateMetrics() {
    const metrics = metricsCollector.getCurrentMetrics();

    document.getElementById('avgWaitTime').textContent = `${metrics.avgWaitTime.toFixed(1)}s`;
    document.getElementById('maxQueue').textContent = metrics.maxQueue;
    document.getElementById('throughput').textContent = Math.round(metrics.throughput);
    document.getElementById('efficiency').textContent = `${metrics.efficiency}%`;

    // Update metric change indicators
    const efficiencyChange = document.getElementById('efficiencyChange');
    if (metrics.efficiency > 70) {
        efficiencyChange.textContent = '↑ Excellent';
        efficiencyChange.className = 'metric-change positive';
    } else if (metrics.efficiency > 50) {
        efficiencyChange.textContent = '→ Good';
        efficiencyChange.className = 'metric-change';
    } else {
        efficiencyChange.textContent = '↓ Fair';
        efficiencyChange.className = 'metric-change negative';
    }
}

/**
 * Update lane status indicators
 */
function updateLaneStatus() {
    const metrics = metricsCollector.getCurrentMetrics();
    const queueLengths = metrics.queueLengths;

    ['north', 'south', 'east', 'west'].forEach(direction => {
        const queueElement = document.querySelector(`[data-lane="${direction}"]`);
        const signalElement = document.querySelector(`[data-signal="${direction}"]`);

        if (queueElement) {
            queueElement.textContent = queueLengths[direction] || 0;
        }

        if (signalElement) {
            const state = signalController.getSignalState(direction);
            signalElement.className = `signal-indicator ${state}`;
        }
    });
}

/**
 * Update model insights
 */
function updateInsights() {
    if (!currentClassification) return;

    const metrics = metricsCollector.getCurrentMetrics();

    document.getElementById('currentCluster').textContent = currentClassification.clusterId;
    document.getElementById('totalVehicles').textContent = metrics.totalVehicles;
}

/**
 * Update status badge
 */
function updateStatusBadge(status, isActive) {
    const badge = document.getElementById('simulationStatus');
    const dot = badge.querySelector('.status-dot');
    const text = badge.querySelector('span:last-child');

    text.textContent = status;
    dot.style.background = isActive ? '#10b981' : '#f59e0b';
}

/**
 * Update density label
 */
function updateDensityLabel() {
    const labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High'];
    densityValueLabel.textContent = labels[trafficDensity - 1];
}

/**
 * Update speed label
 */
function updateSpeedLabel() {
    speedValueLabel.textContent = `${simulationSpeed.toFixed(1)}x`;
}

/**
 * Adjust color brightness
 */
function adjustColor(color, amount) {
    const num = parseInt(color.replace('#', ''), 16);
    const r = Math.min(255, Math.max(0, (num >> 16) + amount));
    const g = Math.min(255, Math.max(0, ((num >> 8) & 0x00FF) + amount));
    const b = Math.min(255, Math.max(0, (num & 0x0000FF) + amount));
    return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')}`;
}

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
