/**
 * Metrics Collector
 * Tracks and calculates performance metrics for comparison
 */

class MetricsCollector {
    constructor() {
        this.reset();
    }

    update(trafficMetrics) {
        this.currentMetrics = trafficMetrics;

        // Calculate throughput (vehicles per minute)
        const currentTime = Date.now();
        const elapsedMinutes = (currentTime - this.startTime) / 60000;
        this.throughput = elapsedMinutes > 0 ? trafficMetrics.passedVehicles / elapsedMinutes : 0;

        // Calculate efficiency (inverse of average wait time, normalized)
        const maxWaitTime = 60; // seconds
        const normalizedWaitTime = Math.min(trafficMetrics.avgWaitTime, maxWaitTime) / maxWaitTime;
        this.efficiency = Math.round((1 - normalizedWaitTime) * 100);

        // Track history for comparison
        this.history.push({
            timestamp: currentTime,
            ...trafficMetrics,
            throughput: this.throughput,
            efficiency: this.efficiency
        });

        // Keep only last 100 data points
        if (this.history.length > 100) {
            this.history.shift();
        }
    }

    getCurrentMetrics() {
        return {
            avgWaitTime: this.currentMetrics.avgWaitTime,
            maxQueue: this.currentMetrics.maxQueueLength,
            throughput: this.throughput,
            efficiency: this.efficiency,
            totalVehicles: this.currentMetrics.vehicleCount,
            queueLengths: this.currentMetrics.queueLengths
        };
    }

    getComparison(baselineMetrics) {
        if (!baselineMetrics) return null;

        const current = this.getCurrentMetrics();

        return {
            waitTimeChange: this.calculateChange(baselineMetrics.avgWaitTime, current.avgWaitTime),
            queueChange: this.calculateChange(baselineMetrics.maxQueue, current.maxQueue),
            throughputChange: this.calculateChange(baselineMetrics.throughput, current.throughput, true),
            efficiencyChange: this.calculateChange(baselineMetrics.efficiency, current.efficiency, true)
        };
    }

    calculateChange(baseline, current, higherIsBetter = false) {
        if (baseline === 0) return { percent: 0, label: '--', isPositive: true };

        const percentChange = ((current - baseline) / baseline) * 100;
        const isPositive = higherIsBetter ? percentChange > 0 : percentChange < 0;

        const label = `${percentChange > 0 ? '+' : ''}${percentChange.toFixed(1)}%`;

        return {
            percent: percentChange,
            label: label,
            isPositive: isPositive
        };
    }

    getHistory() {
        return this.history;
    }

    reset() {
        this.startTime = Date.now();
        this.currentMetrics = {
            vehicleCount: 0,
            averageSpeed: 0,
            congestionLevel: 0,
            queueLengths: { north: 0, south: 0, east: 0, west: 0 },
            maxQueueLength: 0,
            passedVehicles: 0,
            avgWaitTime: 0
        };
        this.throughput = 0;
        this.efficiency = 0;
        this.history = [];
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MetricsCollector;
}
