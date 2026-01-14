/**
 * K-Means Cluster Model for Traffic Classification
 * Uses actual Bangalore traffic data with 14 features
 * Based on trained model from user's Final_cluster_model.ipynb
 */

class ClusterModel {
    constructor() {
        this.clusters = null;
        this.trafficStates = null;
        this.signalTimings = null;
        this.featureNames = null;
        this.scalerParams = null;
        this.isLoaded = false;
    }

    /**
     * Load cluster parameters from JSON file
     */
    async load() {
        try {
            const response = await fetch('data/clusterParams.json');
            const data = await response.json();

            this.clusters = data.clusters;
            this.signalTimings = data.signal_timings;
            this.featureNames = data.feature_names;
            this.scalerParams = data.scaler_params;
            this.metadata = data.metadata;
            this.isLoaded = true;

            console.log(`✓ Loaded K-Means model: ${this.metadata.num_clusters} clusters, ${this.metadata.num_features} features`);
            console.log(`✓ Training set: ${this.metadata.num_samples} samples, Silhouette: ${this.metadata.silhouette_score}`);

            return true;
        } catch (error) {
            console.error('Failed to load cluster parameters:', error);
            return false;
        }
    }

    /**
     * Standardize features using StandardScaler (same as sklearn)
     * scaled_value = (value - mean) / std
     */
    scaleFeatures(features) {
        const scaled = [];

        for (let i = 0; i < this.featureNames.length; i++) {
            const featureName = this.featureNames[i];
            const value = features[i];
            const params = this.scalerParams[featureName];

            if (params.std === 0) {
                // If std is 0, feature is constant, just subtract mean
                scaled.push(value - params.mean);
            } else {
                scaled.push((value - params.mean) / params.std);
            }
        }

        return scaled;
    }

    /**
     * Calculate Euclidean distance between two feature vectors
     */
    euclideanDistance(features1, features2) {
        let sumSquares = 0;
        for (let i = 0; i < features1.length; i++) {
            const diff = features1[i] - features2[i];
            sumSquares += diff * diff;
        }
        return Math.sqrt(sumSquares);
    }

    /**
     * Classify traffic state based on 14 features
     * @param {Object} trafficFeatures - Object with all 14 traffic features
     * @returns {Object} Classification result with cluster, state, timing, and distances
     */
    classify(trafficFeatures) {
        if (!this.isLoaded) {
            console.error('Model not loaded. Call load() first.');
            return null;
        }

        // Extract features in correct order (14 features)
        // Property names match the CSV dataset column names (snake_case)
        const rawFeatures = [
            trafficFeatures.traffic_volume,
            trafficFeatures.average_speed,
            trafficFeatures.travel_time_index,
            trafficFeatures.congestion_level,
            trafficFeatures.road_capacity_utilization,
            trafficFeatures.incident_reports,
            trafficFeatures.environmental_impact,
            trafficFeatures.pedestrian_and_cyclist_count,
            trafficFeatures.isweekend,
            trafficFeatures.weather_conditions_fog,
            trafficFeatures.weather_conditions_overcast,
            trafficFeatures.weather_conditions_rain,
            trafficFeatures.weather_conditions_windy,
            trafficFeatures.roadwork_and_construction_activity_yes
        ];

        // Scale features using StandardScaler
        const scaledFeatures = this.scaleFeatures(rawFeatures);

        // Calculate distance to each cluster centroid
        const distances = this.clusters.map(cluster => ({
            clusterId: cluster.id,
            trafficState: {
                name: cluster.traffic_state,
                description: cluster.description,
                color: cluster.color
            },
            distance: this.euclideanDistance(scaledFeatures, cluster.centroid_scaled)
        }));

        // Sort by distance (closest first)
        distances.sort((a, b) => a.distance - b.distance);

        // Best cluster is the one with smallest distance
        const bestCluster = distances[0];
        const bestClusterData = this.clusters[bestCluster.clusterId];

        // Get signal timing for this cluster
        const signalTiming = this.signalTimings[bestCluster.clusterId];

        // Calculate confidence (inverse of normalized distance)
        const totalDistance = distances.reduce((sum, d) => sum + d.distance, 0);
        const confidence = totalDistance > 0
            ? (1 - (bestCluster.distance / totalDistance)) * 100
            : 100;

        return {
            clusterId: bestCluster.clusterId,
            trafficState: bestCluster.trafficState,
            distance: bestCluster.distance,
            confidence: Math.min(100, Math.max(0, confidence)),
            signalTiming: signalTiming,
            distances: distances, // All distances for educational display
            scaledFeatures: scaledFeatures, // For debugging
            rawFeatures: rawFeatures, // For display
            clusterData: bestClusterData, // Full cluster information
            metrics: trafficFeatures // Original traffic features for UI
        };
    }

    /**
     * Get all cluster information for visualization
     */
    getAllClusters() {
        return this.clusters;
    }

    /**
     * Get feature names
     */
    getFeatureNames() {
        return this.featureNames;
    }

    /**
     * Get scaler parameters
     */
    getScalerParams() {
        return this.scalerParams;
    }
}

// Export for use in other modules
const clusterModel = new ClusterModel();
