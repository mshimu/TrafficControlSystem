// Socket.io connection
const socket = io();

// Simulation state
let simulationData = {
    vehicles: {},
    trafficLights: {},
    metrics: {}
};

// Performance tracking
let frameCount = 0;
let lastFpsUpdate = performance.now();
let fps = 0;

// Metrics history for charts
const metricHistory = {
    waitingTime: [],
    vehiclesCleared: [],
    averageSpeed: [],
    activeVehicles: [],
    timestamp: []
};

// Canvas setup
const canvas = document.getElementById('simulationCanvas');
const ctx = canvas.getContext('2d');

// Set canvas to full container size
function resizeCanvas() {
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    drawSimulation();
}

// Initialize on load
window.addEventListener('load', () => {
    resizeCanvas();
    initializeMetricsChart();
});

window.addEventListener('resize', resizeCanvas);

// Socket event handlers
socket.on('connect', () => {
    updateConnectionStatus(true, 'Connected to simulation server');
    showSuccess('Connected to simulation server');
});

socket.on('disconnect', (reason) => {
    updateConnectionStatus(false, `Disconnected: ${reason}`);
    showError(`Disconnected from server: ${reason}`);
});

socket.on('connect_error', (error) => {
    updateConnectionStatus(false, 'Connection failed');
    showError('Failed to connect to simulation server. Please check if the server is running.');
});

// Listen for simulation updates
socket.on('simulation_update', function(data) {
    simulationData = data.data;
    updateMetrics(data.metrics);
    drawSimulation();
    updateFrameRate();
    document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
});

// Update connection status UI
function updateConnectionStatus(connected, message) {
    const statusIndicator = document.getElementById('connectionStatus');
    const statusText = document.getElementById('connectionText');
    
    if (connected) {
        statusIndicator.className = 'status-indicator status-connected';
        statusText.textContent = message;
    } else {
        statusIndicator.className = 'status-indicator status-disconnected';
        statusText.textContent = message;
    }
}

// Show error message
function showError(message) {
    const alert = document.getElementById('errorAlert');
    document.getElementById('errorMessage').textContent = message;
    alert.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        alert.style.display = 'none';
    }, 5000);
}

// Show success message
function showSuccess(message) {
    const alert = document.getElementById('successAlert');
    document.getElementById('successMessage').textContent = message;
    alert.style.display = 'block';
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        alert.style.display = 'none';
    }, 3000);
}

// Set loading state for buttons
function setLoadingState(loading) {
    const buttons = document.querySelectorAll('.btn');
    const controlPanel = document.querySelector('.control-panel');
    
    buttons.forEach(btn => {
        btn.disabled = loading;
    });
    
    if (loading) {
        controlPanel.classList.add('loading');
    } else {
        controlPanel.classList.remove('loading');
    }
}

// Simulation control functions
async function startSimulation() {
    setLoadingState(true);
    try {
        const useAI = document.getElementById('useAI').checked;
        const emergencyVehicles = document.getElementById('emergencyVehicles').checked;
        const vehicleDensity = document.getElementById('vehicleDensity').value;
        const simSpeed = document.getElementById('simSpeed').value;
        
        const response = await fetch('/api/start_simulation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                use_ai: useAI,
                emergency_vehicles: emergencyVehicles,
                vehicle_density: parseInt(vehicleDensity),
                simulation_speed: parseInt(simSpeed)
            })
        });
        
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }
        
        const result = await response.json();
        showSuccess('Simulation started successfully');
        
    } catch (error) {
        console.error('Error starting simulation:', error);
        showError(`Failed to start simulation: ${error.message}`);
    } finally {
        setLoadingState(false);
    }
}

async function stopSimulation() {
    setLoadingState(true);
    try {
        const response = await fetch('/api/stop_simulation', { 
            method: 'POST' 
        });
        
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }
        
        showSuccess('Simulation stopped');
        
    } catch (error) {
        console.error('Error stopping simulation:', error);
        showError(`Failed to stop simulation: ${error.message}`);
    } finally {
        setLoadingState(false);
    }
}

async function resetSimulation() {
    setLoadingState(true);
    try {
        // First stop the simulation
        await stopSimulation();
        
        // Clear the simulation data
        simulationData = {
            vehicles: {},
            trafficLights: {},
            metrics: {}
        };
        
        // Reset metrics
        updateMetrics({});
        
        // Clear the chart data
        Object.keys(metricHistory).forEach(key => {
            metricHistory[key] = [];
        });
        
        // Redraw empty simulation
        drawSimulation();
        
        // Reinitialize chart
        initializeMetricsChart();
        
        // Start a new simulation after a brief delay
        setTimeout(() => {
            startSimulation();
        }, 1000);
        
    } catch (error) {
        console.error('Error resetting simulation:', error);
        showError(`Failed to reset simulation: ${error.message}`);
        setLoadingState(false);
    }
}

// Calculate and display frame rate
function updateFrameRate() {
    frameCount++;
    const now = performance.now();
    const elapsed = now - lastFpsUpdate;
    
    if (elapsed > 1000) {
        fps = Math.round((frameCount * 1000) / elapsed);
        frameCount = 0;
        lastFpsUpdate = now;
        
        document.getElementById('frameRate').textContent = fps;
    }
}

// Drawing functions
function drawSimulation() {
    // Clear canvas
    ctx.fillStyle = '#2c3e50';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw roads
    drawRoads();
    
    // Draw traffic lights
    drawTrafficLights();
    
    // Draw vehicles
    drawVehicles();
}

function drawRoads() {
    ctx.strokeStyle = '#7f8c8d';
    ctx.lineWidth = Math.min(canvas.width, canvas.height) / 20;
    
    // Calculate road positions based on canvas size
    const horizontalRoad1 = canvas.height * 0.25;
    const horizontalRoad2 = canvas.height * 0.75;
    const verticalRoad1 = canvas.width * 0.25;
    const verticalRoad2 = canvas.width * 0.75;
    
    // Horizontal roads
    ctx.beginPath();
    ctx.moveTo(50, horizontalRoad1);
    ctx.lineTo(canvas.width - 50, horizontalRoad1);
    ctx.moveTo(50, horizontalRoad2);
    ctx.lineTo(canvas.width - 50, horizontalRoad2);
    
    // Vertical roads
    ctx.moveTo(verticalRoad1, 50);
    ctx.lineTo(verticalRoad1, canvas.height - 50);
    ctx.moveTo(verticalRoad2, 50);
    ctx.lineTo(verticalRoad2, canvas.height - 50);
    
    ctx.stroke();
    
    // Draw road markings
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 15]);
    
    // Horizontal road markings
    for (let x = 70; x < canvas.width - 50; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, horizontalRoad1 - 5);
        ctx.lineTo(x + 20, horizontalRoad1 - 5);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(x, horizontalRoad2 - 5);
        ctx.lineTo(x + 20, horizontalRoad2 - 5);
        ctx.stroke();
    }
    
    // Vertical road markings
    for (let y = 70; y < canvas.height - 50; y += 40) {
        ctx.beginPath();
        ctx.moveTo(verticalRoad1 - 5, y);
        ctx.lineTo(verticalRoad1 - 5, y + 20);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(verticalRoad2 - 5, y);
        ctx.lineTo(verticalRoad2 - 5, y + 20);
        ctx.stroke();
    }
    
    ctx.setLineDash([]);
}

function drawTrafficLights() {
    const lightSize = Math.min(canvas.width, canvas.height) / 40;
    
    for (const [id, light] of Object.entries(simulationData.trafficLights || {})) {
        // Calculate position based on canvas size
        let x, y;
        switch(id) {
            case 'J1': x = canvas.width * 0.25; y = canvas.height * 0.25; break;
            case 'J2': x = canvas.width * 0.75; y = canvas.height * 0.25; break;
            case 'J3': x = canvas.width * 0.25; y = canvas.height * 0.75; break;
            case 'J4': x = canvas.width * 0.75; y = canvas.height * 0.75; break;
            default: x = light.x || canvas.width/2; y = light.y || canvas.height/2;
        }
        
        // Draw traffic light base
        ctx.fillStyle = '#34495e';
        ctx.fillRect(x - lightSize, y - lightSize, lightSize * 2, lightSize * 2);
        
        // Draw light state
        const state = light.state;
        if (state.includes('G')) {
            ctx.fillStyle = '#2ecc71'; // Green
        } else if (state.includes('Y')) {
            ctx.fillStyle = '#f39c12'; // Yellow
        } else {
            ctx.fillStyle = '#e74c3c'; // Red
        }
        
        ctx.beginPath();
        ctx.arc(x, y, lightSize * 0.6, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw junction label
        ctx.fillStyle = 'white';
        ctx.font = `${lightSize * 0.8}px Arial`;
        ctx.textAlign = 'center';
        ctx.fillText(id, x, y + lightSize * 1.8);
    }
}

function drawVehicles() {
    const vehicleWidth = Math.min(canvas.width, canvas.height) / 60;
    const vehicleHeight = vehicleWidth * 0.6;
    
    for (const [id, vehicle] of Object.entries(simulationData.vehicles || {})) {
        // Calculate position based on canvas size
        const x = (vehicle.x / 800) * canvas.width;
        const y = (vehicle.y / 500) * canvas.height;
        
        // Set vehicle color based on type
        if (vehicle.type === 'emergency') {
            ctx.fillStyle = '#e74c3c'; // Red for emergency vehicles
        } else {
            ctx.fillStyle = vehicle.color || '#3498db'; // Blue for regular vehicles
        }
        
        // Draw vehicle as rectangle
        ctx.fillRect(x - vehicleWidth/2, y - vehicleHeight/2, vehicleWidth, vehicleHeight);
        
        // Draw direction indicator
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y);
        
        // Calculate direction based on vehicle orientation
        let endX = x, endY = y;
        if (vehicle.direction === 'right') endX = x + vehicleWidth/2;
        else if (vehicle.direction === 'left') endX = x - vehicleWidth/2;
        else if (vehicle.direction === 'up') endY = y - vehicleHeight/2;
        else if (vehicle.direction === 'down') endY = y + vehicleHeight/2;
        
        ctx.lineTo(endX, endY);
        ctx.stroke();
        
        // Draw vehicle ID for debugging
        if (Object.keys(simulationData.vehicles).length < 20) {
            ctx.fillStyle = 'white';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(id.substring(0, 4), x, y - vehicleHeight);
        }
    }
}

// Metrics functions
function updateMetrics(metrics) {
    document.getElementById('waitingTime').textContent = 
        Math.round(metrics.total_waiting_time || 0);
    document.getElementById('vehiclesCleared').textContent = 
        metrics.vehicles_cleared || 0;
    document.getElementById('emergencyCleared').textContent = 
        metrics.emergency_cleared || 0;
    document.getElementById('averageSpeed').textContent = 
        Math.round((metrics.average_speed || 0) * 10) / 10;
    document.getElementById('activeVehicles').textContent = 
        metrics.active_vehicles || Object.keys(simulationData.vehicles || {}).length;
        
    // Update metrics history for chart
    updateMetricsChart(metrics);
}

function initializeMetricsChart() {
    const layout = {
        title: 'Real-time Performance Metrics',
        xaxis: { title: 'Time' },
        yaxis: { title: 'Waiting Time (s)' },
        yaxis2: {
            title: 'Average Speed (m/s)',
            overlaying: 'y',
            side: 'right'
        },
        legend: { x: 0, y: 1.1, orientation: 'h' },
        margin: { t: 50, r: 50, b: 50, l: 50 }
    };
    
    Plotly.newPlot('metricsChart', [], layout, { displayModeBar: false });
}

function updateMetricsChart(metrics) {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    
    // Add new data points
    metricHistory.waitingTime.push(metrics.total_waiting_time || 0);
    metricHistory.vehiclesCleared.push(metrics.vehicles_cleared || 0);
    metricHistory.averageSpeed.push(metrics.average_speed || 0);
    metricHistory.activeVehicles.push(metrics.active_vehicles || Object.keys(simulationData.vehicles || {}).length);
    metricHistory.timestamp.push(timeString);
    
    // Keep only last 50 data points
    if (metricHistory.timestamp.length > 50) {
        metricHistory.waitingTime.shift();
        metricHistory.vehiclesCleared.shift();
        metricHistory.averageSpeed.shift();
        metricHistory.activeVehicles.shift();
        metricHistory.timestamp.shift();
    }
    
    // Create traces for the chart
    const trace1 = {
        x: metricHistory.timestamp,
        y: metricHistory.waitingTime,
        type: 'scatter',
        mode: 'lines',
        name: 'Waiting Time',
        line: { color: '#e74c3c', width: 2 }
    };
    
    const trace2 = {
        x: metricHistory.timestamp,
        y: metricHistory.averageSpeed,
        type: 'scatter',
        mode: 'lines',
        name: 'Avg Speed',
        yaxis: 'y2',
        line: { color: '#3498db', width: 2 }
    };
    
    const trace3 = {
        x: metricHistory.timestamp,
        y: metricHistory.activeVehicles,
        type: 'scatter',
        mode: 'lines',
        name: 'Active Vehicles',
        line: { color: '#2ecc71', width: 2 }
    };
    
    const layout = {
        title: 'Real-time Performance Metrics',
        xaxis: { 
            title: 'Time',
            tickangle: -45
        },
        yaxis: { 
            title: 'Waiting Time (s) / Vehicle Count',
            rangemode: 'tozero'
        },
        yaxis2: {
            title: 'Average Speed (m/s)',
            overlaying: 'y',
            side: 'right',
            rangemode: 'tozero'
        },
        legend: { 
            x: 0, 
            y: 1.2, 
            orientation: 'h',
            bgcolor: 'rgba(255,255,255,0.8)'
        },
        margin: { t: 50, r: 50, b: 80, l: 50 }
    };
    
    Plotly.react('metricsChart', [trace1, trace2, trace3], layout, { displayModeBar: true });
}

// Initial draw
drawSimulation();
initializeMetricsChart();