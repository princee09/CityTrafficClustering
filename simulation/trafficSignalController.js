/**
 * Traffic Signal Controller
 * Manages both fixed-time and adaptive signal control strategies
 */

class TrafficSignalController {
    constructor(mode = 'adaptive') {
        this.mode = mode; // 'adaptive' or 'fixed'
        this.currentPhase = 0; // 0: NS green, 1: EW green
        this.phaseTimer = 0;
        this.phaseStates = ['green', 'yellow', 'red'];
        this.currentStateIndex = 0;

        // Signal timings (will be updated by adaptive mode)
        this.timings = {
            green: 30,
            yellow: 3,
            red: 30
        };

        // Fixed-time defaults
        this.fixedTimings = {
            green: 25,
            yellow: 3,
            red: 25
        };

        this.signals = {
            north: 'red',
            south: 'red',
            east: 'red',
            west: 'red'
        };

        this.cycleCount = 0;
        this.lastUpdate = Date.now();
    }

    /**
     * Update signal controller based on traffic classification
     */
    update(deltaTime, classification = null) {
        this.phaseTimer += deltaTime;

        // Adaptive mode: adjust timings based on classification
        if (this.mode === 'adaptive' && classification) {
            this.updateAdaptiveTimings(classification);
        } else {
            this.timings = { ...this.fixedTimings };
        }

        // Determine current state duration
        const currentState = this.phaseStates[this.currentStateIndex];
        let stateDuration = this.timings[currentState];

        // Check if it's time to transition
        if (this.phaseTimer >= stateDuration) {
            this.phaseTimer = 0;
            this.currentStateIndex++;

            // Move to next phase after completing all states
            if (this.currentStateIndex >= this.phaseStates.length) {
                this.currentStateIndex = 0;
                this.currentPhase = (this.currentPhase + 1) % 2;
                this.cycleCount++;
            }
        }

        // Update signal states
        this.updateSignalStates();
    }

    /**
     * Update timings based on adaptive ML classification
     */
    updateAdaptiveTimings(classification) {
        const { signalTiming } = classification;

        if (signalTiming) {
            this.timings.green = signalTiming.green_time;
            this.timings.yellow = signalTiming.yellow_time;
            this.timings.red = signalTiming.green_time; // Red time = green time for other direction
        }
    }

    /**
     * Update individual signal states based on current phase
     */
    updateSignalStates() {
        const currentState = this.phaseStates[this.currentStateIndex];

        if (this.currentPhase === 0) {
            // North-South green
            this.signals.north = currentState;
            this.signals.south = currentState;
            this.signals.east = currentState === 'green' || currentState === 'yellow' ? 'red' : 'green';
            this.signals.west = currentState === 'green' || currentState === 'yellow' ? 'red' : 'green';
        } else {
            // East-West green
            this.signals.east = currentState;
            this.signals.west = currentState;
            this.signals.north = currentState === 'green' || currentState === 'yellow' ? 'red' : 'green';
            this.signals.south = currentState === 'green' || currentState === 'yellow' ? 'red' : 'green';
        }
    }

    /**
     * Get signal state for a specific direction
     */
    getSignalState(direction) {
        return this.signals[direction] || 'red';
    }

    /**
     * Check if a direction has green light
     */
    isGreen(direction) {
        return this.signals[direction] === 'green';
    }

    /**
     * Set control mode
     */
    setMode(mode) {
        this.mode = mode;
        console.log(`Signal controller mode: ${mode}`);
    }

    /**
     * Reset controller
     */
    reset() {
        this.currentPhase = 0;
        this.phaseTimer = 0;
        this.currentStateIndex = 0;
        this.cycleCount = 0;
        this.updateSignalStates();
    }

    /**
     * Get current timings
     */
    getCurrentTimings() {
        return { ...this.timings };
    }

    /**
     * Get cycle information
     */
    getCycleInfo() {
        const currentState = this.phaseStates[this.currentStateIndex];
        const stateDuration = this.timings[currentState];
        const timeRemaining = stateDuration - this.phaseTimer;

        return {
            phase: this.currentPhase,
            state: currentState,
            timeRemaining: Math.max(0, timeRemaining),
            cycleCount: this.cycleCount,
            timings: this.timings
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TrafficSignalController;
}
