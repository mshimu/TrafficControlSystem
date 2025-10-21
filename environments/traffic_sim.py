import traci
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import random
import sys  
from typing import Dict, List, Tuple

# ← ADD THESE LINES ↓
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from config import SUMO_BINARY, NETWORK_DIR, SIMULATION_CONFIG
except ImportError as e:
    print(f"Config import failed: {e}")
    # Fallback values
    SUMO_BINARY = "sumo-gui"
    NETWORK_DIR = "environments/networks"
    SIMULATION_CONFIG = {
        'max_steps': 3600,
        'step_length': 1, 
        'emergency_spawn_prob': 0.001,
        'num_intersections': 16,
    }

class TrafficSimulation(gym.Env):
    def __init__(self, network_file: str = "simple_grid.sumocfg", emergency_vehicles: bool = True):  # ← REMOVED config_file parameter
        super().__init__()
        
        # SUMO configuration - UPDATE THESE LINES ↓
        self.network_file = network_file
        config_file = f"{NETWORK_DIR}/{network_file}"  # ← USE CONFIG PATH
        self.sumo_cmd = [SUMO_BINARY, "-c", config_file, "--start", "--quit-on-end"]  # ← USE SUMO_BINARY
        
        # Simulation parameters - UPDATE THESE LINES ↓
        self.simulation_step = 0
        self.max_steps = SIMULATION_CONFIG['max_steps']  # ← USE CONFIG
        self.emergency_vehicles = emergency_vehicles
        
        # Define action and observation spaces
        self.num_intersections = SIMULATION_CONFIG['num_intersections']  # ← USE CONFIG
        self.num_phases = 4  # NS green, EW green, NS left, EW left
        
        # Action space: phase selection for each intersection
        self.action_space = spaces.MultiDiscrete([self.num_phases] * self.num_intersections)
        
        # Observation space: queue lengths, waiting times, vehicle counts, emergency vehicle info
        obs_dim = self.num_intersections * 12  # 4 approaches * 3 metrics per intersection
        if emergency_vehicles:
            obs_dim += self.num_intersections * 4  # Emergency vehicle info
        self.observation_space = spaces.Box(low=0, high=1000, shape=(obs_dim,), dtype=np.float32)
        
        # Emergency vehicles tracking
        self.emergency_vehicles_active = []
        self.emergency_routes = {}
        
        # Performance metrics
        self.metrics = {
            'total_waiting_time': 0,
            'vehicles_cleared': 0,
            'emergency_cleared': 0,
            'average_speed': 0
        }
    
    def reset(self):
        """Reset the simulation environment"""
        if traci.isLoaded():
            traci.close()
        
        # Start SUMO simulation
        traci.start(self.sumo_cmd)
        self.simulation_step = 0
        
        # Initialize emergency vehicles if enabled
        if self.emergency_vehicles:
            self._initialize_emergency_vehicles()
        
        return self._get_observation()
    
    def step(self, actions: np.ndarray):
        """Execute one simulation step"""
        # Set traffic light phases based on actions
        self._set_traffic_lights(actions)
        
        # Advance simulation
        traci.simulationStep()
        self.simulation_step += 1
        
        # Spawn emergency vehicles randomly - UPDATE THIS LINE ↓
        if self.emergency_vehicles and random.random() < SIMULATION_CONFIG['emergency_spawn_prob']:  # ← USE CONFIG
            self._spawn_emergency_vehicle()
        
        # Update emergency vehicles tracking
        self._update_emergency_vehicles()
        
        # Calculate reward
        reward = self._calculate_reward(actions)
        
        # Get next observation
        observation = self._get_observation()
        
        # Check if episode is done - UPDATE THIS LINE ↓
        done = self.simulation_step >= SIMULATION_CONFIG['max_steps']  # ← USE CONFIG
        
        # Update metrics
        self._update_metrics()
        
        info = {
            'metrics': self.metrics.copy(),
            'emergency_vehicles': len(self.emergency_vehicles_active)
        }
        
        return observation, reward, done, info
    
    def _set_traffic_lights(self, actions: np.ndarray):
        """Set traffic light phases based on agent actions"""
        intersection_ids = ['J0', 'J1', 'J2', 'J3']  # Junction IDs in SUMO network
        
        for i, junction_id in enumerate(intersection_ids):
            phase = actions[i]
            phase_duration = 10  # Fixed phase duration for simplicity
            
            # Define phase mappings (simplified)
            phase_programs = {
                0: "GGGrrrrrrrrrr",  # NS straight
                1: "rrrrGGGrrrrrr",  # EW straight  
                2: "yyyrrrrrrrrrr",  # NS transition
                3: "rrrryyyrrrrrr"   # EW transition
            }
            
            if phase in phase_programs:
                traci.trafficlight.setRedYellowGreenState(junction_id, phase_programs[phase])
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from simulation"""
        observation = []
        intersection_ids = ['J0', 'J1', 'J2', 'J3']
        
        for junction_id in intersection_ids:
            # Get queue lengths for each approach
            queue_lengths = self._get_queue_lengths(junction_id)
            
            # Get waiting times
            waiting_times = self._get_waiting_times(junction_id)
            
            # Get vehicle counts
            vehicle_counts = self._get_vehicle_counts(junction_id)
            
            junction_obs = np.concatenate([queue_lengths, waiting_times, vehicle_counts])
            observation.extend(junction_obs)
        
        # Add emergency vehicle information
        if self.emergency_vehicles:
            emergency_info = self._get_emergency_vehicle_info()
            observation.extend(emergency_info)
        
        return np.array(observation, dtype=np.float32)
    
    def _get_queue_lengths(self, junction_id: str) -> np.ndarray:
        """Get queue lengths for all approaches to a junction"""
        # Simplified implementation - in practice, use traci.lane.getLastStepHaltingNumber()
        queue_lengths = []
        for lane in traci.lane.getIDList():
            if junction_id in lane:
                queue = traci.lane.getLastStepHaltingNumber(lane)
                queue_lengths.append(queue)
        
        # Pad to 4 approaches if needed
        while len(queue_lengths) < 4:
            queue_lengths.append(0.0)
            
        return np.array(queue_lengths[:4])
    
    def _get_waiting_times(self, junction_id: str) -> np.ndarray:
        """Get average waiting times for approaches"""
        waiting_times = []
        for lane in traci.lane.getIDList():
            if junction_id in lane:
                waiting_time = traci.lane.getWaitingTime(lane)
                waiting_times.append(waiting_time)
        
        while len(waiting_times) < 4:
            waiting_times.append(0.0)
            
        return np.array(waiting_times[:4])
    
    def _get_vehicle_counts(self, junction_id: str) -> np.ndarray:
        """Get vehicle counts for approaches"""
        vehicle_counts = []
        for lane in traci.lane.getIDList():
            if junction_id in lane:
                count = traci.lane.getLastStepVehicleNumber(lane)
                vehicle_counts.append(count)
        
        while len(vehicle_counts) < 4:
            vehicle_counts.append(0.0)
            
        return np.array(vehicle_counts[:4])
    
    def _calculate_reward(self, actions: np.ndarray) -> float:
        """Calculate reward for current state"""
        reward = 0.0
        
        # Base reward: negative of total waiting time
        total_waiting = 0
        for lane in traci.lane.getIDList():
            total_waiting += traci.lane.getWaitingTime(lane)
        
        reward -= total_waiting * 0.01
        
        # Bonus for clearing emergency vehicles
        for vehicle_id in self.emergency_vehicles_active:
            if traci.vehicle.getSpeed(vehicle_id) > 5:  # Moving well
                reward += 2.0
        
        # Penalty for frequent phase changes
        if hasattr(self, 'last_actions'):
            phase_changes = np.sum(actions != self.last_actions)
            reward -= phase_changes * 0.1
        
        self.last_actions = actions.copy()
        
        return reward
    
    def _initialize_emergency_vehicles(self):
        """Initialize emergency vehicle routes and tracking"""
        self.emergency_vehicles_active = []
        self.emergency_routes = {
            'route1': ['edge1', 'edge2', 'edge3'],
            'route2': ['edge4', 'edge5', 'edge6'],
            # Add more routes as needed
        }
    
    def _spawn_emergency_vehicle(self):
        """Spawn a new emergency vehicle"""
        try:
            route_id = random.choice(list(self.emergency_routes.keys()))
            vehicle_id = f"emergency_{len(self.emergency_vehicles_active)}"
            
            traci.vehicle.add(
                vehID=vehicle_id,
                routeID=route_id,
                typeID="emergency",
                depart="now",
                departLane="best",
                departPos="0",
                departSpeed="0"
            )
            
            # Set emergency vehicle properties
            traci.vehicle.setColor(vehicle_id, (255, 0, 0))  # Red color
            traci.vehicle.setSpeed(vehicle_id, 50)  # Higher speed
            
            self.emergency_vehicles_active.append(vehicle_id)
            
        except Exception as e:
            print(f"Error spawning emergency vehicle: {e}")
    
    def _update_emergency_vehicles(self):
        """Update emergency vehicles tracking"""
        completed_vehicles = []
        for vehicle_id in self.emergency_vehicles_active:
            if not traci.vehicle.isStopped(vehicle_id) and traci.vehicle.getSpeed(vehicle_id) > 0:
                # Vehicle is still active and moving
                pass
            else:
                # Vehicle completed its route or stopped
                completed_vehicles.append(vehicle_id)
                self.metrics['emergency_cleared'] += 1
        
        # Remove completed vehicles
        for vehicle_id in completed_vehicles:
            self.emergency_vehicles_active.remove(vehicle_id)
    
    def _update_metrics(self):
        """Update performance metrics"""
        total_waiting = 0
        total_vehicles = 0
        
        for lane in traci.lane.getIDList():
            total_waiting += traci.lane.getWaitingTime(lane)
            total_vehicles += traci.lane.getLastStepVehicleNumber(lane)
        
        self.metrics['total_waiting_time'] = total_waiting
        self.metrics['vehicles_cleared'] = traci.simulation.getArrivedNumber()
        
        if total_vehicles > 0:
            self.metrics['average_speed'] = traci.vehicle.getTotalCO2Emission() / total_vehicles  # Simplified metric
    
    def close(self):
        """Close the simulation"""
        if traci.isLoaded():
            traci.close()