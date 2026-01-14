/**
 * Traffic Simulation Engine
 * Handles vehicle spawning, movement, and intersection management
 */

class Vehicle {
    constructor(x, y, direction, id) {
        this.id = id;
        this.x = x;
        this.y = y;
        this.direction = direction; // 'north', 'south', 'east', 'west'
        this.speed = 0;
        this.maxSpeed = 3;
        this.width = 8;
        this.height = 16;
        this.color = this.getRandomColor();
        this.waiting = false;
        this.waitTime = 0;
        this.passed = false;
        this.distanceToIntersection = 0;
    }

    getRandomColor() {
        const colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4'];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    update(deltaTime, signalState, vehicles, intersectionBounds) {
        // Calculate distance to intersection
        this.distanceToIntersection = this.calculateDistanceToIntersection(intersectionBounds);

        // Check for collision with vehicle ahead
        const vehicleAhead = this.getVehicleAhead(vehicles);
        const minDistance = 20;

        // Determine if we should stop
        let shouldStop = false;

        // Stop for red light
        if (signalState === 'red' && this.distanceToIntersection < 50 && this.distanceToIntersection > 0) {
            shouldStop = true;
            this.waiting = true;
        }
        // Slow down for yellow light
        else if (signalState === 'yellow' && this.distanceToIntersection < 30 && this.distanceToIntersection > 0) {
            shouldStop = true;
        }
        // Stop if too close to vehicle ahead
        else if (vehicleAhead && vehicleAhead.distanceTo < minDistance) {
            shouldStop = true;
        }
        else {
            this.waiting = false;
        }

        // Update speed
        if (shouldStop) {
            this.speed = Math.max(0, this.speed - 0.2);
            if (this.waiting) {
                this.waitTime += deltaTime;
            }
        } else {
            this.speed = Math.min(this.maxSpeed, this.speed + 0.1);
            this.waiting = false;
        }

        // Update position based on direction
        const movement = this.speed * deltaTime / 16.67; // Normalize to ~60fps

        switch (this.direction) {
            case 'north':
                this.y -= movement;
                break;
            case 'south':
                this.y += movement;
                break;
            case 'east':
                this.x += movement;
                break;
            case 'west':
                this.x -= movement;
                break;
        }
    }

    calculateDistanceToIntersection(intersectionBounds) {
        const { centerX, centerY, size } = intersectionBounds;

        switch (this.direction) {
            case 'north':
                return (centerY + size / 2) - this.y;
            case 'south':
                return this.y - (centerY - size / 2);
            case 'east':
                return this.x - (centerX - size / 2);
            case 'west':
                return (centerX + size / 2) - this.x;
            default:
                return 0;
        }
    }

    getVehicleAhead(vehicles) {
        const ahead = vehicles.filter(v => {
            if (v.id === this.id || v.direction !== this.direction) return false;

            const tolerance = 30; // Lane width tolerance

            switch (this.direction) {
                case 'north':
                    return v.y < this.y && Math.abs(v.x - this.x) < tolerance;
                case 'south':
                    return v.y > this.y && Math.abs(v.x - this.x) < tolerance;
                case 'east':
                    return v.x > this.x && Math.abs(v.y - this.y) < tolerance;
                case 'west':
                    return v.x < this.x && Math.abs(v.y - this.y) < tolerance;
                default:
                    return false;
            }
        });

        if (ahead.length === 0) return null;

        // Find closest vehicle ahead
        let closest = ahead[0];
        let minDist = this.getDistance(closest);

        ahead.forEach(v => {
            const dist = this.getDistance(v);
            if (dist < minDist) {
                minDist = dist;
                closest = v;
            }
        });

        return { vehicle: closest, distanceTo: minDist };
    }

    getDistance(other) {
        return Math.sqrt((this.x - other.x) ** 2 + (this.y - other.y) ** 2);
    }

    isOffScreen(canvasWidth, canvasHeight) {
        return this.x < -50 || this.x > canvasWidth + 50 ||
            this.y < -50 || this.y > canvasHeight + 50;
    }
}

class TrafficSimulation {
    constructor(canvasWidth, canvasHeight) {
        this.width = canvasWidth;
        this.height = canvasHeight;
        this.vehicles = [];
        this.vehicleIdCounter = 0;
        this.spawnRate = 0.5; // Base spawn rate
        this.spawnTimer = 0;
        this.passedVehicles = 0;
        this.totalWaitTime = 0;
        this.maxQueueLength = 0;

        // Intersection configuration
        this.intersection = {
            centerX: canvasWidth / 2,
            centerY: canvasHeight / 2,
            size: 100
        };

        // Spawn points for each direction
        this.spawnPoints = {
            north: { x: this.intersection.centerX - 15, y: canvasHeight + 20 },
            south: { x: this.intersection.centerX + 15, y: -20 },
            east: { x: -20, y: this.intersection.centerY - 15 },
            west: { x: canvasWidth + 20, y: this.intersection.centerY + 15 }
        };
    }

    update(deltaTime, signalController, densityMultiplier = 1) {
        // Update spawn rate based on density
        this.spawnRate = 0.3 * densityMultiplier;

        // Spawn new vehicles
        this.spawnTimer += deltaTime;
        const spawnInterval = 1000 / this.spawnRate;

        if (this.spawnTimer >= spawnInterval) {
            this.spawnVehicle();
            this.spawnTimer = 0;
        }

        // Update existing vehicles
        this.vehicles.forEach(vehicle => {
            const signalState = signalController.getSignalState(vehicle.direction);
            vehicle.update(deltaTime, signalState, this.vehicles, this.intersection);
        });

        // Remove off-screen vehicles
        const initialCount = this.vehicles.length;
        this.vehicles = this.vehicles.filter(v => {
            if (v.isOffScreen(this.width, this.height)) {
                if (!v.passed) {
                    this.passedVehicles++;
                    this.totalWaitTime += v.waitTime;
                    v.passed = true;
                }
                return false;
            }
            return true;
        });

        // Update max queue length
        const currentQueue = this.getMaxQueueLength();
        this.maxQueueLength = Math.max(this.maxQueueLength, currentQueue);
    }

    spawnVehicle() {
        const directions = ['north', 'south', 'east', 'west'];
        const direction = directions[Math.floor(Math.random() * directions.length)];
        const spawnPoint = this.spawnPoints[direction];

        const vehicle = new Vehicle(
            spawnPoint.x,
            spawnPoint.y,
            direction,
            this.vehicleIdCounter++
        );

        this.vehicles.push(vehicle);
    }

    getVehicleCount() {
        return this.vehicles.length;
    }

    getAverageSpeed() {
        if (this.vehicles.length === 0) return 0;
        const totalSpeed = this.vehicles.reduce((sum, v) => sum + v.speed, 0);
        return totalSpeed / this.vehicles.length;
    }

    getCongestionLevel() {
        const waitingCount = this.vehicles.filter(v => v.waiting).length;
        return this.vehicles.length > 0 ? waitingCount / this.vehicles.length : 0;
    }

    getQueueLengthByDirection(direction) {
        return this.vehicles.filter(v =>
            v.direction === direction &&
            v.waiting &&
            v.distanceToIntersection < 100
        ).length;
    }

    getMaxQueueLength() {
        const queues = ['north', 'south', 'east', 'west'].map(dir =>
            this.getQueueLengthByDirection(dir)
        );
        return Math.max(...queues, 0);
    }

    getTrafficMetrics() {
        // Basic metrics
        const vehicleCount = this.getVehicleCount();
        const averageSpeed = this.getAverageSpeed() * 15; // Scale to km/h (20-89 range)
        const congestionLevel = this.getCongestionLevel() * 100; // Convert to percentage

        // Calculate derived features based on traffic state
        // Traffic volume - scale vehicle count to Bangalore ranges (4K-72K)
        const trafficVolume = Math.max(4000, Math.min(72000,
            vehicleCount * 800 + Math.random() * 5000
        ));

        // Travel Time Index (1.0-1.5) - higher when congestion is higher
        const travelTimeIndex = 1.0 + (congestionLevel / 100) * 0.5;

        // Road Capacity Utilization (18-100%) - correlated with volume and congestion
        const roadCapacityUtilization = Math.max(18, Math.min(100,
            (trafficVolume / 720) + (congestionLevel * 0.3) + Math.random() * 10
        ));

        // Incident Reports (0-10) - more likely when congestion is high
        const incidentReports = congestionLevel > 70 ?
            Math.floor(Math.random() * 4) :
            Math.floor(Math.random() * 2);

        // Environmental Impact (58-194) - scales with volume and congestion
        const environmentalImpact = Math.max(58, Math.min(194,
            (trafficVolume / 500) + (100 - averageSpeed) * 1.5 + Math.random() * 20
        ));

        // Pedestrian and Cyclist Count (66-243) - varies independently
        const pedestrianCount = Math.floor(66 + Math.random() * 177);

        // Weekend flag (0 or 1) - randomly determined, weighted 28% weekend
        const isWeekend = Math.random() < 0.287 ? 1 : 0;

        // Weather conditions (one-hot encoded, mutually exclusive mostly)
        // Fog: 10.7%, Overcast: 14.5%, Rain: 9.3%, Windy: 4.8%
        const weatherRand = Math.random();
        let fog = 0, overcast = 0, rain = 0, windy = 0;

        if (weatherRand < 0.107) {
            fog = 1;
        } else if (weatherRand < 0.252) {
            overcast = 1;
        } else if (weatherRand < 0.345) {
            rain = 1;
        } else if (weatherRand < 0.393) {
            windy = 1;
        }
        // else: clear weather (all 0)

        // Roadwork activity (0 or 1) - 9.9% chance
        const roadwork = Math.random() < 0.099 ? 1 : 0;

        // Return all 14 features + additional metrics for display
        return {
            // === 14 K-Means Features (in exact order) ===
            traffic_volume: trafficVolume,
            average_speed: averageSpeed,
            travel_time_index: travelTimeIndex,
            congestion_level: congestionLevel,
            road_capacity_utilization: roadCapacityUtilization,
            incident_reports: incidentReports,
            environmental_impact: environmentalImpact,
            pedestrian_and_cyclist_count: pedestrianCount,
            isweekend: isWeekend,
            weather_conditions_fog: fog,
            weather_conditions_overcast: overcast,
            weather_conditions_rain: rain,
            weather_conditions_windy: windy,
            roadwork_and_construction_activity_yes: roadwork,

            // === Additional metrics for UI display ===
            vehicleCount: vehicleCount,
            queueLengths: {
                north: this.getQueueLengthByDirection('north'),
                south: this.getQueueLengthByDirection('south'),
                east: this.getQueueLengthByDirection('east'),
                west: this.getQueueLengthByDirection('west')
            },
            maxQueueLength: this.maxQueueLength,
            passedVehicles: this.passedVehicles,
            avgWaitTime: this.passedVehicles > 0 ? this.totalWaitTime / this.passedVehicles / 1000 : 0
        };
    }

    reset() {
        this.vehicles = [];
        this.vehicleIdCounter = 0;
        this.passedVehicles = 0;
        this.totalWaitTime = 0;
        this.maxQueueLength = 0;
        this.spawnTimer = 0;
    }

    resize(width, height) {
        this.width = width;
        this.height = height;
        this.intersection.centerX = width / 2;
        this.intersection.centerY = height / 2;

        this.spawnPoints = {
            north: { x: this.intersection.centerX - 15, y: height + 20 },
            south: { x: this.intersection.centerX + 15, y: -20 },
            east: { x: -20, y: this.intersection.centerY - 15 },
            west: { x: width + 20, y: this.intersection.centerY + 15 }
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Vehicle, TrafficSimulation };
}
