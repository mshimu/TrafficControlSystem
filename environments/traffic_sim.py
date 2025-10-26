# environments/traffic_sim.py
import traci
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import random
import sys  
from typing import Dict, List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

class TrafficSimulation(gym.Env):
    def __init__(self, use_ai: bool = True, baseline_strategy: str = "adaptive"):
        super().__init__()
        
        # Get config values
        try:
            from utils.config_loader import get_config
            config = get_config()
            self.sumo_binary = config.get('environment.sumo.binary', 'sumo')
            self.network_dir = config.get('environment.sumo.network_dir', 'environments/networks')
            self.max_steps = config.get('environment.simulation.max_steps', 3600)
            self.emergency_spawn_prob = config.get('environment.traffic.emergency_vehicle_probability', 0.001)
            self.num_intersections = config.get('environment.intersections.num_junctions', 4)
        except ImportError:
            # Fallback values
            self.sumo_binary = "sumo"
            self.network_dir = "environments/networks"
            self.max_steps = 3600
            self.emergency_spawn_prob = 0.001
            self.num_intersections = 4
        
        # SUMO configuration
        self.network_file = "simple_grid.sumocfg"
        config_file = f"{self.network_dir}/{self.network_file}"
        self.sumo_cmd = [
            self.sumo_binary, 
            "-c", config_file, 
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "300",
            "--collision.action", "none",
            "--random"
]
        
        # Simulation parameters
        self.simulation_step = 0
        self.use_ai = use_ai
        self.baseline_strategy = baseline_strategy
        
        # Define action and observation spaces
        self.num_phases = 4  # NS green, EW green, NS left, EW left
        
        # Action space: phase selection for each intersection
        self.action_space = spaces.MultiDiscrete([self.num_phases] * self.num_intersections)
        
        # Observation space: queue lengths, waiting times, vehicle counts
        obs_dim = self.num_intersections * 12  # 4 approaches * 3 metrics per intersection
        self.observation_space = spaces.Box(low=0, high=1000, shape=(obs_dim,), dtype=np.float32)
        
        # Emergency vehicles tracking
        self.emergency_vehicles_active = []
        self.emergency_routes = {}
        
        # Performance metrics
        self.metrics = {
            'total_waiting_time': 0,
            'vehicles_cleared': 0,
            'emergency_cleared': 0,
            'average_speed': 0,
            'active_vehicles': 0
        }
        
        # Controller for baseline strategies
        self.controller = None
        if not use_ai:
            self._setup_baseline_controller()
        
        print(f"✅ TrafficSimulation initialized (AI: {use_ai}, Strategy: {baseline_strategy})")
    
    def _setup_baseline_controller(self):
        """Setup baseline traffic controller"""
        try:
            from agents.baseline_controllers import BaselineController
            self.controller = BaselineController(strategy=self.baseline_strategy)
        except ImportError:
            print("⚠️ Baseline controller not available, using simple fallback")
            self.controller = SimpleController() 

    def _get_intersection_ids(self):
        """Get actual junction IDs from the network"""
        try:
            junction_ids = traci.trafficlight.getIDList()
            if junction_ids:
                print(f"🔍 Found junctions: {junction_ids}")
                # RETURN ACTUAL JUNCTION NAMES - no mapping!
                return junction_ids[:self.num_intersections]
            else:
                return ['J0', 'J1', 'J2', 'J3'][:self.num_intersections]  # Use actual names
        except Exception as e:
            print(f"⚠️ Could not get junction IDs: {e}")
            return ['J0', 'J1', 'J2', 'J3'][:self.num_intersections]  # Use actual names      
    
    def reset(self):
        """Reset the simulation environment"""
        if traci.isLoaded():
            traci.close()
        
        # DEBUG: Verify SUMO paths
        #config_file = f"{self.network_dir}/{self.network_file}"
        #print(f"🔍 SUMO binary: {self.sumo_binary}")
        #print(f"🔍 Network dir: {os.path.abspath(self.network_dir)}")
        #print(f"🔍 Config file: {os.path.abspath(config_file)}")
        #print(f"🔍 Config exists: {os.path.exists(config_file)}")
        
        # Check if we can find SUMO
        #import subprocess
        #try:
            #result = subprocess.run([self.sumo_binary, "--version"], capture_output=True, text=True)
            #print(f"🔍 SUMO version: {result.stdout.strip() if result.stdout else 'Not found'}")
        #except:
            #print("❌ SUMO binary not found!")

        # Start SUMO simulation
        try:
            traci.start(self.sumo_cmd)
            print("✅ SUMO started successfully!")

            traffic_lights = traci.trafficlight.getIDList()
            print(f"🔍 SUMO Traffic Lights: {traffic_lights}")
        
            junctions = traci.junction.getIDList()
            print(f"🔍 SUMO Junctions: {junctions}")

            self.simulation_step = 0
            
            # Initialize emergency vehicles
            self._initialize_emergency_vehicles()
            
            # Reset metrics
            self.metrics = {
                'total_waiting_time': 0,
                'vehicles_cleared': 0,
                'emergency_cleared': 0,
                'average_speed': 0,
                'active_vehicles': 0
            }
            
            return self._get_observation()
            
        except Exception as e:
            print(f"❌ Error starting SUMO: {e}")
            # Return dummy observation
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def step(self, actions: np.ndarray):
        """Execute one simulation step"""
        try:
            # Set traffic light phases based on actions
            self._set_traffic_lights(actions)
            
            # Advance simulation
            traci.simulationStep()
            self.simulation_step += 1
            
            # Spawn emergency vehicles randomly
            if random.random() < self.emergency_spawn_prob:
                self._spawn_emergency_vehicle()
            
            # Update emergency vehicles tracking
            self._update_emergency_vehicles()
            
            # Calculate reward
            reward = self._calculate_reward(actions)
            
            # Get next observation
            observation = self._get_observation()
            
            # Check if episode is done
            done = self.simulation_step >= self.max_steps
            
            # Update metrics
            self._update_metrics()
            
            info = {
                'metrics': self.metrics.copy(),
                'emergency_vehicles': len(self.emergency_vehicles_active),
                'simulation_step': self.simulation_step
            }
            
            return observation, reward, done, info
            
        except Exception as e:
            print(f"❌ Error in simulation step: {e}")
            # Return safe fallback values
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            return observation, 0.0, True, {'error': str(e)}
    
    def _set_traffic_lights(self, actions: np.ndarray):
        """Set traffic light phases based on agent actions"""
        intersection_ids = self._get_intersection_ids()  # ← Use auto-detection
        
        for i, junction_id in enumerate(intersection_ids):
            if i < len(actions):
                phase = actions[i]
                #phase_duration = 10  # Fixed phase duration
                
                # Define phase mappings (simplified)
                phase_programs = {
                    0: "GGGrrrrrrrrrr",  # NS straight
                    1: "rrrrGGGrrrrrr",  # EW straight  
                    2: "yyyrrrrrrrrrr",  # NS transition
                    3: "rrrryyyrrrrrr"   # EW transition
                }
                
                if phase in phase_programs:
                    try:
                        traci.trafficlight.setRedYellowGreenState(junction_id, phase_programs[phase])
                    except:
                        pass  # Junction might not exist
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from simulation"""
        observation = []
        intersection_ids = self._get_intersection_ids()
        
        for junction_id in intersection_ids:
            try:
                # Get queue lengths for each approach
                queue_lengths = self._get_queue_lengths(junction_id)
                
                # Get waiting times
                waiting_times = self._get_waiting_times(junction_id)
                
                # Get vehicle counts
                vehicle_counts = self._get_vehicle_counts(junction_id)
                
                junction_obs = np.concatenate([queue_lengths, waiting_times, vehicle_counts])
                observation.extend(junction_obs)
            except:
                # If junction doesn't exist, use zeros
                observation.extend([0.0] * 12)
        
        # Ensure correct observation size
        if len(observation) < self.observation_space.shape[0]:
            observation.extend([0.0] * (self.observation_space.shape[0] - len(observation)))
        
        return np.array(observation[:self.observation_space.shape[0]], dtype=np.float32)
    
    def _get_queue_lengths(self, junction_id: str) -> np.ndarray:
        """Get queue lengths for all approaches to a junction"""
        queue_lengths = []
        try:
            for lane in traci.lane.getIDList():
                if junction_id in lane:
                    queue = traci.lane.getLastStepHaltingNumber(lane)
                    queue_lengths.append(queue)
        except:
            pass
        
        # Pad to 4 approaches if needed
        while len(queue_lengths) < 4:
            queue_lengths.append(0.0)
            
        return np.array(queue_lengths[:4])
    
    def _get_waiting_times(self, junction_id: str) -> np.ndarray:
        """Get average waiting times for approaches"""
        waiting_times = []
        try:
            for lane in traci.lane.getIDList():
                if junction_id in lane:
                    waiting_time = traci.lane.getWaitingTime(lane)
                    waiting_times.append(waiting_time)
        except:
            pass
        
        while len(waiting_times) < 4:
            waiting_times.append(0.0)
            
        return np.array(waiting_times[:4])
    
    def _get_vehicle_counts(self, junction_id: str) -> np.ndarray:
        """Get vehicle counts for approaches"""
        vehicle_counts = []
        try:
            for lane in traci.lane.getIDList():
                if junction_id in lane:
                    count = traci.lane.getLastStepVehicleNumber(lane)
                    vehicle_counts.append(count)
        except:
            pass
        
        while len(vehicle_counts) < 4:
            vehicle_counts.append(0.0)
            
        return np.array(vehicle_counts[:4])
    
    def _calculate_reward(self, actions: np.ndarray) -> float:
        """Calculate reward for current state"""
        reward = 0.0
        
        try:
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
            
        except:
            reward = 0.0
        
        return reward
    
    def _initialize_emergency_vehicles(self):
        """Initialize emergency vehicle routes and tracking"""
        self.emergency_vehicles_active = []
        self.emergency_counter = 0
        self.emergency_routes = {
            'emergency_h': ['E0'],       # single-edge horizontal route
            'emergency_v': ['E2'],       # single-edge vertical route
            'emergency_loop': ['E0','E3','-E1','-E2'],
        }
    
    def _spawn_emergency_vehicle(self):
        """Spawn a new emergency vehicle"""
        try:
            self.emergency_vehicles_active = [
            v for v in self.emergency_vehicles_active if v in traci.vehicle.getIDList()
        ]
            route_id = random.choice(list(self.emergency_routes.keys()))
            #vehicle_id = f"emergency_{len(self.emergency_vehicles_active)}"
            vehicle_id = f"emergency_{self.emergency_counter}"
            self.emergency_counter += 1  # increment to keep IDs unique
            
            traci.vehicle.add(
                vehID=vehicle_id,
                routeID=route_id,
                typeID="emergency",
                depart="now"
            )
            
            # Set emergency vehicle properties
            traci.vehicle.setColor(vehicle_id, (255, 0, 0))
            traci.vehicle.setSpeed(vehicle_id, 50)
            
            self.emergency_vehicles_active.append(vehicle_id)
            
        except Exception as e:
            pass  # Silent fail for now
    
    def _update_emergency_vehicles(self):
        """Update emergency vehicles tracking"""
        completed_vehicles = []
        for vehicle_id in self.emergency_vehicles_active:
            try:
                if not traci.vehicle.isStopped(vehicle_id) and traci.vehicle.getSpeed(vehicle_id) > 0:
                    pass
                else:
                    completed_vehicles.append(vehicle_id)
                    self.metrics['emergency_cleared'] += 1
            except:
                completed_vehicles.append(vehicle_id)
        
        # Remove completed vehicles
        for vehicle_id in completed_vehicles:
            if vehicle_id in self.emergency_vehicles_active:
                self.emergency_vehicles_active.remove(vehicle_id)
    
    def _update_metrics(self):
        """Update performance metrics"""
        try:
            total_waiting = 0
            total_vehicles = 0
            
            for lane in traci.lane.getIDList():
                total_waiting += traci.lane.getWaitingTime(lane)
                total_vehicles += traci.lane.getLastStepVehicleNumber(lane)
            
            self.metrics['total_waiting_time'] = total_waiting
            self.metrics['vehicles_cleared'] = traci.simulation.getArrivedNumber()
            self.metrics['active_vehicles'] = total_vehicles
            
            if total_vehicles > 0:
                self.metrics['average_speed'] = traci.vehicle.getTotalCO2Emission() / total_vehicles
        except:
            pass
    
    def get_metrics(self) -> Dict:
        """Get current metrics (required by train_model.py)"""
        return self.metrics.copy()
    
    def close(self):
        """Close the simulation"""
        try:
            if traci.isLoaded():
                traci.close()
        except:
            pass

# Fallback controller class
class SimpleController:
    def __init__(self, strategy="fixed"):
        self.strategy = strategy
    
    def decide_actions(self, vehicles, traffic_lights, metrics):
        """Simple fixed-time controller"""
        return {
            'J0': {'state': 'G', 'duration': 30.0, 'reason': 'Fixed'},
            'J1': {'state': 'G', 'duration': 30.0, 'reason': 'Fixed'},
            'J2': {'state': 'G', 'duration': 30.0, 'reason': 'Fixed'}, 
            'J3': {'state': 'G', 'duration': 30.0, 'reason': 'Fixed'}
        }

# Test the environment
if __name__ == "__main__":
    env = TrafficSimulation()
    print("✅ TrafficSimulation test successful")