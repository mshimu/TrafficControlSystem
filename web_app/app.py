from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import traci.constants as tc
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import NETWORK_DIR, SUMO_BINARY, SIMULATION_CONFIG
except ImportError:
    # Fallback values
    NETWORK_DIR = "environments/networks"
    SUMO_BINARY = "sumo-gui"
    SIMULATION_CONFIG = {
        'max_steps': 3600,
        'emergency_spawn_prob': 0.001,
        'num_intersections': 4,
    }

from environments.traffic_sim import TrafficSimulation
from agents.mappo_agent import MAPPOAgent

app = Flask(__name__)
app.config['SECRET_KEY'] = 'traffic_ai_secret'
socketio = SocketIO(app, async_mode='eventlet')

# Global variables for simulation
simulation_thread = None
simulation_running = False
current_simulation = None

class WebSimulation:
    def __init__(self):
        self.env = None
        self.agent = None
        self.metrics_history = []
        self.max_history = 100
    
    def initialize(self, use_ai: bool = True, emergency_vehicles: bool = True):
        """Initialize simulation environment"""
        # USE CONFIG PATHS 
        network_file = "simple_grid.net.xml"  
        #config_file = "environments/networks/simple_grid.sumocfg"
        
        self.env = TrafficSimulation(
            network_file=network_file,  # ← traffic_sim.py handles NETWORK_DIR
            emergency_vehicles=emergency_vehicles
        )
        
        if use_ai:
            # Load trained agent
            self.agent = MAPPOAgent(
                num_agents=self.env.num_intersections,
                state_dim=self.env.observation_space.shape[0],
                action_dim=self.env.action_space.nvec[0]
            )
            # TODO: Load pre-trained weights here
    
    def step(self):
        """Execute one simulation step"""
        if self.env is None:
            return None
        
        if self.agent:
            # Use AI agent
            states = self.env._get_observation()
            actions, _ = self.agent.select_actions([states] * self.env.num_intersections)
            next_states, reward, done, info = self.env.step(np.array(actions))
        else:
            # Use fixed-time control
            actions = np.random.randint(0, 4, size=self.env.num_intersections)
            next_states, reward, done, info = self.env.step(actions)
        
        # Update metrics history
        self.metrics_history.append(info['metrics'])
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        return info
    
    def get_simulation_data(self):
        """Get current simulation state for visualization"""
        # ADD TRACI IMPORT 
        try:
            import traci
        except ImportError:
            print("❌ traci not available for visualization")
            return None

        if self.env is None or not traci.isLoaded():
            return None
        
        data = {
            'vehicles': {},
            'traffic_lights': {},
            'metrics': self.metrics_history[-1] if self.metrics_history else {}
        }
        
        try:
            # Get vehicle positions and states
            for vehicle_id in traci.vehicle.getIDList():
                pos = traci.vehicle.getPosition(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                vehicle_type = traci.vehicle.getTypeID(vehicle_id)
                
                data['vehicles'][vehicle_id] = {
                    'x': pos[0],
                    'y': pos[1],
                    'speed': speed,
                    'type': vehicle_type,
                    'color': 'red' if 'emergency' in vehicle_id else 'blue'
                }
            
            # Get traffic light states
            for tl_id in ['J0', 'J1', 'J2', 'J3']:
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                data['traffic_lights'][tl_id] = {
                    'state': state,
                    'position': self._get_junction_position(tl_id)
                }
                
        except Exception as e:
            print(f"Error getting simulation data: {e}")
        
        return data
    
    def _get_junction_position(self, junction_id: str):
        """Get junction position for visualization"""
        positions = {
            'J0': [100, 100],
            'J1': [300, 100],
            'J2': [100, 300],
            'J3': [300, 300]
        }
        return positions.get(junction_id, [0, 0])
    
    def close(self):
        """Close simulation"""
        if self.env:
            self.env.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_simulation', methods=['POST'])
def start_simulation():
    global simulation_thread, simulation_running, current_simulation
    
    if simulation_running:
        return jsonify({'status': 'error', 'message': 'Simulation already running'})
    
    data = request.json
    use_ai = data.get('use_ai', True)
    emergency_vehicles = data.get('emergency_vehicles', True)
    
    current_simulation = WebSimulation()
    current_simulation.initialize(use_ai=use_ai, emergency_vehicles=emergency_vehicles)
    
    simulation_running = True
    simulation_thread = threading.Thread(target=run_simulation)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    return jsonify({'status': 'success', 'message': 'Simulation started'})

@app.route('/api/stop_simulation', methods=['POST'])
def stop_simulation():
    global simulation_running, current_simulation
    
    simulation_running = False
    if current_simulation:
        current_simulation.close()
        current_simulation = None
    
    return jsonify({'status': 'success', 'message': 'Simulation stopped'})

@app.route('/api/status')
def get_status():
    global simulation_running, current_simulation
    
    status = {
        'running': simulation_running,
        'metrics': current_simulation.metrics_history[-1] if current_simulation and current_simulation.metrics_history else {}
    }
    
    return jsonify(status)

def run_simulation():
    """Background simulation thread"""
    global simulation_running, current_simulation
    
    while simulation_running and current_simulation:
        try:
            # Run simulation step
            info = current_simulation.step()
            
            if info:
                # Get simulation data for visualization
                sim_data = current_simulation.get_simulation_data()
                
                # Emit update to all connected clients
                socketio.emit('simulation_update', {
                    'data': sim_data,
                    'metrics': info['metrics']
                })
            else:
                # Simulation might not be properly initialized
                print("⚠️ Simulation step returned None")
                break
            
            # Control simulation speed
            time.sleep(0.1)  # 10 FPS
            
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    simulation_running = False
    print("🛑 Simulation thread stopped")

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_response', {'data': 'Connected to traffic simulation'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)