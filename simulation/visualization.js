/**
 * Visualization Renderer
 * Handles all canvas drawing for the traffic simulation
 */

class TrafficVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
    }

    clear() {
        this.ctx.fillStyle = '#0f172a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    draw(simulation, signalController) {
        this.clear();
        this.drawRoad(simulation.intersection);
        this.drawIntersection(simulation.intersection);
        this.drawTrafficLights(simulation.intersection, signalController);
        this.drawVehicles(simulation.vehicles);
        this.drawStopLines(simulation.intersection);
    }

    drawRoad(intersection) {
        const { centerX, centerY, size } = intersection;
        const roadWidth = 60;
        const laneWidth = roadWidth / 2;

        this.ctx.fillStyle = '#334155';

        // Vertical road
        this.ctx.fillRect(
            centerX - roadWidth / 2,
            0,
            roadWidth,
            this.canvas.height
        );

        // Horizontal road
        this.ctx.fillRect(
            0,
            centerY - roadWidth / 2,
            this.canvas.width,
            roadWidth
        );

        // Lane markings
        this.ctx.strokeStyle = '#475569';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([15, 10]);

        // Vertical lane marking
        this.ctx.beginPath();
        this.ctx.moveTo(centerX, 0);
        this.ctx.lineTo(centerX, centerY - size / 2);
        this.ctx.moveTo(centerX, centerY + size / 2);
        this.ctx.lineTo(centerX, this.canvas.height);
        this.ctx.stroke();

        // Horizontal lane marking
        this.ctx.beginPath();
        this.ctx.moveTo(0, centerY);
        this.ctx.lineTo(centerX - size / 2, centerY);
        this.ctx.moveTo(centerX + size / 2, centerY);
        this.ctx.lineTo(this.canvas.width, centerY);
        this.ctx.stroke();

        this.ctx.setLineDash([]);
    }

    drawIntersection(intersection) {
        const { centerX, centerY, size } = intersection;

        // Intersection fill
        this.ctx.fillStyle = '#1e293b';
        this.ctx.fillRect(
            centerX - size / 2,
            centerY - size / 2,
            size,
            size
        );

        // Intersection border
        this.ctx.strokeStyle = '#475569';
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(
            centerX - size / 2,
            centerY - size / 2,
            size,
            size
        );
    }

    drawStopLines(intersection) {
        const { centerX, centerY, size } = intersection;
        const roadWidth = 60;
        const offset = size / 2 + 2;

        this.ctx.strokeStyle = '#f8fafc';
        this.ctx.lineWidth = 4;

        // North stop line
        this.ctx.beginPath();
        this.ctx.moveTo(centerX - roadWidth / 4, centerY - offset);
        this.ctx.lineTo(centerX + roadWidth / 4, centerY - offset);
        this.ctx.stroke();

        // South stop line
        this.ctx.beginPath();
        this.ctx.moveTo(centerX - roadWidth / 4, centerY + offset);
        this.ctx.lineTo(centerX + roadWidth / 4, centerY + offset);
        this.ctx.stroke();

        // East stop line
        this.ctx.beginPath();
        this.ctx.moveTo(centerX + offset, centerY - roadWidth / 4);
        this.ctx.lineTo(centerX + offset, centerY + roadWidth / 4);
        this.ctx.stroke();

        // West stop line
        this.ctx.beginPath();
        this.ctx.moveTo(centerX - offset, centerY - roadWidth / 4);
        this.ctx.lineTo(centerX - offset, centerY + roadWidth / 4);
        this.ctx.stroke();
    }

    drawTrafficLights(intersection, signalController) {
        const { centerX, centerY, size } = intersection;
        const lightSize = 20;
        const offset = size / 2 + 15;

        const positions = {
            north: { x: centerX + 25, y: centerY - offset - 30 },
            south: { x: centerX - 25, y: centerY + offset + 30 },
            east: { x: centerX + offset + 30, y: centerY + 25 },
            west: { x: centerX - offset - 30, y: centerY - 25 }
        };

        Object.entries(positions).forEach(([direction, pos]) => {
            const state = signalController.getSignalState(direction);
            this.drawTrafficLight(pos.x, pos.y, state, lightSize);
        });
    }

    drawTrafficLight(x, y, state, size) {
        // Light housing
        this.ctx.fillStyle = '#1e293b';
        this.ctx.strokeStyle = '#475569';
        this.ctx.lineWidth = 2;

        const width = size;
        const height = size * 3 + 12;

        this.ctx.fillRect(x - width / 2, y - height / 2, width, height);
        this.ctx.strokeRect(x - width / 2, y - height / 2, width, height);

        // Lights
        const lightRadius = size / 3;
        const spacing = size;

        // Red light
        this.ctx.fillStyle = state === 'red' ? '#ef4444' : '#4c0000';
        this.ctx.beginPath();
        this.ctx.arc(x, y - spacing, lightRadius, 0, Math.PI * 2);
        this.ctx.fill();

        if (state === 'red') {
            this.ctx.shadowColor = '#ef4444';
            this.ctx.shadowBlur = 15;
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
        }

        // Yellow light
        this.ctx.fillStyle = state === 'yellow' ? '#f59e0b' : '#4c3700';
        this.ctx.beginPath();
        this.ctx.arc(x, y, lightRadius, 0, Math.PI * 2);
        this.ctx.fill();

        if (state === 'yellow') {
            this.ctx.shadowColor = '#f59e0b';
            this.ctx.shadowBlur = 15;
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
        }

        // Green light
        this.ctx.fillStyle = state === 'green' ? '#10b981' : '#003d1f';
        this.ctx.beginPath();
        this.ctx.arc(x, y + spacing, lightRadius, 0, Math.PI * 2);
        this.ctx.fill();

        if (state === 'green') {
            this.ctx.shadowColor = '#10b981';
            this.ctx.shadowBlur = 15;
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
        }
    }

    drawVehicles(vehicles) {
        vehicles.forEach(vehicle => {
            this.ctx.save();
            this.ctx.translate(vehicle.x, vehicle.y);

            // Rotate based on direction
            switch (vehicle.direction) {
                case 'north':
                    this.ctx.rotate(-Math.PI / 2);
                    break;
                case 'south':
                    this.ctx.rotate(Math.PI / 2);
                    break;
                case 'west':
                    this.ctx.rotate(Math.PI);
                    break;
            }

            // Vehicle body shadow
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            this.ctx.fillRect(-vehicle.width / 2 + 2, -vehicle.height / 2 + 2, vehicle.width, vehicle.height);

            // Vehicle body
            this.ctx.fillStyle = vehicle.color;
            this.ctx.fillRect(-vehicle.width / 2, -vehicle.height / 2, vehicle.width, vehicle.height);

            // Vehicle shine/highlight
            const gradient = this.ctx.createLinearGradient(0, -vehicle.height / 2, 0, vehicle.height / 2);
            gradient.addColorStop(0, 'rgba(255, 255, 255, 0.3)');
            gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0)');
            gradient.addColorStop(1, 'rgba(0, 0, 0, 0.2)');
            this.ctx.fillStyle = gradient;
            this.ctx.fillRect(-vehicle.width / 2, -vehicle.height / 2, vehicle.width, vehicle.height);

            // Windshield
            this.ctx.fillStyle = 'rgba(100, 150, 200, 0.5)';
            this.ctx.fillRect(-vehicle.width / 2 + 1, -vehicle.height / 2 + 2, vehicle.width - 2, 4);

            // Tail light (red)
            if (vehicle.waiting) {
                this.ctx.fillStyle = '#ef4444';
                this.ctx.fillRect(-vehicle.width / 2 + 1, vehicle.height / 2 - 3, 2, 2);
                this.ctx.fillRect(vehicle.width / 2 - 3, vehicle.height / 2 - 3, 2, 2);
            }

            this.ctx.restore();
        });
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TrafficVisualizer;
}
