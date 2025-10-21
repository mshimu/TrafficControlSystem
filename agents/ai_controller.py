import numpy as np
import torch
from typing import Dict, List, Any
import time
import random
from enum import Enum

class AIAction(Enum):
    """Available actions for AI controller"""
    MAINTAIN_CURRENT = 0
    SWITCH_TO_GREEN_NS = 1  # North-South green
    SWITCH_TO_GREEN_EW = 2  # East-West green
    SWITCH_TO_GREEN_ALL = 3  # All directions green (emergency)
    EXTEND_GREEN = 4
    SHORTEN_GREEN = 5

class TrafficStateEncoder:
    """Encodes traffic environment state for AI processing"""
    
    def __init__(self):
        self.junction_positions = {
            'J1': [100, 100], 'J2': [300, 100],
            'J3': [100, 300], 'J4': [300, 300]
        }
        self.state_dim = 32  # Dimension of encoded state
    
    def encode_state(self, junction_id: str, vehicles: Dict, traffic_lights: Dict, 
                    metrics: Dict, emergency_vehicles: Dict) -> np.ndarray:
        """Encode the state for a specific junction"""
        state = []
        
        # 1. Current traffic light state (one-hot encoded)
        current_light = traffic_lights.get(junction_id, {}).get('state', 'R')
        state.extend(self._one_hot_light_state(current_light))
        
        # 2. Traffic density in different approach directions
        state.extend(self._get_traffic_density(junction_id, vehicles))
        
        # 3. Queue lengths in each direction
        state.extend(self._get_queue_lengths(junction_id, vehicles))
        
        # 4. Vehicle speeds approaching junction
        state.extend(self._get_approach_speeds(junction_id, vehicles))
        
        # 5. Emergency vehicle presence and proximity
        state.extend(self._get_emergency_info(junction_id, emergency_vehicles))
        
        # 6. Historical metrics
        state.append(min(metrics.get('total_waiting_time', 0) / 1000.0, 1.0))
        state.append(min(metrics.get('vehicles_cleared', 0) / 50.0, 1.0))
        state.append(min(metrics.get('average_speed', 0) / 20.0, 1.0))
        
        # 7. Time since last phase change
        state.append(self._get_phase_duration(junction_id, traffic_lights))
        
        # 8. Neighbor junction states
        state.extend(self._get_neighbor_states(junction_id, traffic_lights))
        
        # 9. Time of day factor (simulated)
        state.append(self._get_time_factor())
        
        return np.array(state, dtype=np.float32)
    
    def _one_hot_light_state(self, state: str) -> List[float]:
        """One-hot encode traffic light state"""
        states = ['G', 'Y', 'R']
        return [1.0 if s == state else 0.0 for s in states]
    
    def _get_traffic_density(self, junction_id: str, vehicles: Dict) -> List[float]:
        """Get traffic density in each approach direction"""
        junction_pos = self.junction_positions.get(junction_id, [0, 0])
        directions = ['north', 'south', 'east', 'west']
        densities = []
        
        for direction in directions:
            count = 0
            for vehicle in vehicles.values():
                if self._is_in_approach_lane(vehicle, junction_pos, direction):
                    count += 1
            densities.append(min(count / 8.0, 1.0))  # Normalize
        
        return densities
    
    def _get_queue_lengths(self, junction_id: str, vehicles: Dict) -> List[float]:
        """Get queue lengths in each direction"""
        junction_pos = self.junction_positions.get(junction_id, [0, 0])
        directions = ['north', 'south', 'east', 'west']
        queues = []
        
        for direction in directions:
            queue_vehicles = 0
            for vehicle in vehicles.values():
                if (self._is_in_approach_lane(vehicle, junction_pos, direction) and
                    self._is_stopped_or_slow(vehicle)):
                    queue_vehicles += 1
            queues.append(min(queue_vehicles / 5.0, 1.0))
        
        return queues
    
    def _get_approach_speeds(self, junction_id: str, vehicles: Dict) -> List[float]:
        """Get average approach speeds in each direction"""
        junction_pos = self.junction_positions.get(junction_id, [0, 0])
        directions = ['north', 'south', 'east', 'west']
        speeds = []
        
        for direction in directions:
            direction_speeds = []
            for vehicle in vehicles.values():
                if self._is_in_approach_lane(vehicle, junction_pos, direction):
                    speed = vehicle.get('speed', 0)
                    direction_speeds.append(speed)
            
            avg_speed = np.mean(direction_speeds) if direction_speeds else 0
            speeds.append(min(avg_speed / 15.0, 1.0))  # Normalize
        
        return speeds
    
    def _get_emergency_info(self, junction_id: str, emergency_vehicles: Dict) -> List[float]:
        """Get emergency vehicle information"""
        junction_pos = self.junction_positions.get(junction_id, [0, 0])
        
        if not emergency_vehicles:
            return [0.0, 0.0, 0.0]  # [presence, proximity, priority]
        
        # Find closest emergency vehicle
        min_distance = float('inf')
        max_priority = 0
        for emergency in emergency_vehicles.values():
            vehicle_pos = getattr(emergency, 'position', [0, 0])
            distance = np.linalg.norm(np.array(vehicle_pos) - np.array(junction_pos))
            min_distance = min(min_distance, distance)
            priority = getattr(emergency, 'priority', 1)
            max_priority = max(max_priority, priority)
        
        normalized_distance = min(min_distance / 150.0, 1.0)
        normalized_priority = min(max_priority / 10.0, 1.0)
        
        return [1.0, 1.0 - normalized_distance, normalized_priority]
    
    def _get_phase_duration(self, junction_id: str, traffic_lights: Dict) -> float:
        """Get normalized time since last phase change"""
        junction_data = traffic_lights.get(junction_id, {})
        phase_start = junction_data.get('phase_start_time', time.time())
        duration = time.time() - phase_start
        return min(duration / 60.0, 1.0)  # Normalize to 1 minute
    
    def _get_neighbor_states(self, junction_id: str, traffic_lights: Dict) -> List[float]:
        """Get states of neighboring junctions"""
        neighbor_states = []
        junctions = ['J1', 'J2', 'J3', 'J4']
        
        for neighbor_id in junctions:
            if neighbor_id != junction_id:
                state = traffic_lights.get(neighbor_id, {}).get('state', 'R')
                neighbor_states.extend(self._one_hot_light_state(state))
        
        return neighbor_states
    
    def _get_time_factor(self) -> float:
        """Simulate time of day factor (peak vs off-peak)"""
        # Simple simulation - could be based on actual time
        current_minute = time.localtime().tm_min
        # Simulate peak hours (0.8-1.0) and off-peak (0.3-0.6)
        return 0.7 + 0.3 * np.sin(2 * np.pi * current_minute / 60)
    
    def _is_in_approach_lane(self, vehicle: Dict, junction_pos: List, direction: str) -> bool:
        """Check if vehicle is in approach lane for given direction"""
        vehicle_pos = [vehicle.get('x', 0), vehicle.get('y', 0)]
        vehicle_dir = vehicle.get('direction', 'right')
        distance = np.linalg.norm(np.array(vehicle_pos) - np.array(junction_pos))
        
        if distance > 80:  # Too far
            return False
        
        # Simple directional check
        if direction == 'north' and vehicle_dir == 'down' and vehicle_pos[1] < junction_pos[1]:
            return True
        elif direction == 'south' and vehicle_dir == 'up' and vehicle_pos[1] > junction_pos[1]:
            return True
        elif direction == 'east' and vehicle_dir == 'right' and vehicle_pos[0] < junction_pos[0]:
            return True
        elif direction == 'west' and vehicle_dir == 'left' and vehicle_pos[0] > junction_pos[0]:
            return True
        
        return False
    
    def _is_stopped_or_slow(self, vehicle: Dict) -> bool:
        """Check if vehicle is stopped or moving slowly (in queue)"""
        speed = vehicle.get('speed', 0)
        return speed < 2.0  # Considered stopped/slow if < 2 m/s


class MAPPOIntegratedController:
    """
    AI Traffic Controller that integrates with your MAPPO agent
    """
    
    def __init__(self, use_mappo: bool = True):
        self.state_encoder = TrafficStateEncoder()
        self.use_mappo = use_mappo
        self.mappo_agent = None
        self.decision_history = []
        self.emergency_manager = None
        
        # Initialize MAPPO agent for 4 junctions
        if use_mappo:
            try:
                from agents.mappo_agent import MAPPOAgent
                # 4 agents (junctions), 32 state dimensions, 6 actions
                self.mappo_agent = MAPPOAgent(
                    num_agents=4, 
                    state_dim=self.state_encoder.state_dim, 
                    action_dim=len(AIAction),
                    learning_rate=0.001,
                    gamma=0.99
                )
                print("🤖 AI Controller: MAPPO agent initialized successfully")
            except ImportError as e:
                print(f"❌ AI Controller: Could not load MAPPO agent - {e}")
                print("🔄 AI Controller: Falling back to rule-based AI")
                self.use_mappo = False
        
        # Initialize emergency manager
        try:
            from agents.emergency_module import emergency_manager
            self.emergency_manager = emergency_manager
            print("🚑 AI Controller: Emergency module loaded")
        except ImportError:
            print("⚠️ AI Controller: Emergency module not available")
    
    def decide_actions(self, vehicles: Dict, traffic_lights: Dict, metrics: Dict) -> Dict:
        """
        Main decision method for all junctions using MAPPO
        """
        actions = {}
        
        # First, check for emergency vehicles and get priority actions
        emergency_actions = self._handle_emergencies(vehicles, traffic_lights)
        if emergency_actions:
            return emergency_actions
        
        # Get states for all junctions
        junction_states = self._get_all_junction_states(vehicles, traffic_lights, metrics)
        
        # Get actions from MAPPO agent
        if self.use_mappo and self.mappo_agent:
            mappo_actions = self._get_mappo_actions(junction_states)
        else:
            mappo_actions = self._get_rule_based_actions(junction_states, vehicles, traffic_lights)
        
        # Convert MAPPO actions to traffic light commands
        for junction_id, action_idx in mappo_actions.items():
            command = self._action_to_command(action_idx, junction_id, traffic_lights)
            if command:
                actions[junction_id] = command
        
        return actions
    
    def _get_all_junction_states(self, vehicles: Dict, traffic_lights: Dict, metrics: Dict) -> Dict[str, np.ndarray]:
        """Get encoded states for all junctions"""
        states = {}
        emergency_vehicles = {}
        
        if self.emergency_manager:
            emergency_vehicles = self.emergency_manager.detector.emergency_vehicles
        
        for junction_id in traffic_lights.keys():
            state = self.state_encoder.encode_state(
                junction_id, vehicles, traffic_lights, metrics, emergency_vehicles
            )
            states[junction_id] = state
        
        return states
    
    def _get_mappo_actions(self, junction_states: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Get actions from MAPPO agent for all junctions"""
        try:
            # Convert junction states to list in consistent order
            junction_order = ['J1', 'J2', 'J3', 'J4']
            states_list = [junction_states.get(jid, np.zeros(self.state_encoder.state_dim)) 
                          for jid in junction_order]
            
            # Get actions from MAPPO
            actions, log_probs = self.mappo_agent.select_actions(states_list)
            
            # Map back to junction IDs
            mappo_actions = {}
            for i, junction_id in enumerate(junction_order):
                if i < len(actions):
                    mappo_actions[junction_id] = actions[i]
            
            return mappo_actions
            
        except Exception as e:
            print(f"❌ MAPPO action selection failed: {e}")
            # Fallback to rule-based
            return self._get_rule_based_actions(junction_states, {}, {})
    
    def _get_rule_based_actions(self, junction_states: Dict[str, np.ndarray], 
                               vehicles: Dict, traffic_lights: Dict) -> Dict[str, int]:
        """Rule-based fallback when MAPPO is unavailable"""
        actions = {}
        
        for junction_id, state in junction_states.items():
            # Extract features from state vector
            traffic_density = state[3:7]  # densities for 4 directions
            queue_lengths = state[7:11]   # queues for 4 directions
            emergency_presence = state[11] > 0.5
            phase_duration = state[24]
            
            # Rule 1: Emergency vehicle - give priority
            if emergency_presence:
                actions[junction_id] = AIAction.SWITCH_TO_GREEN_ALL.value
                continue
            
            # Rule 2: High queue in one direction - switch to serve that direction
            max_queue_idx = np.argmax(queue_lengths)
            if queue_lengths[max_queue_idx] > 0.7:
                if max_queue_idx in [0, 1]:  # North-South
                    actions[junction_id] = AIAction.SWITCH_TO_GREEN_NS.value
                else:  # East-West
                    actions[junction_id] = AIAction.SWITCH_TO_GREEN_EW.value
                continue
            
            # Rule 3: Phase been on too long - consider switching
            if phase_duration > 0.8:  # Been green for too long
                current_light = traffic_lights.get(junction_id, {}).get('state', 'G')
                if current_light == 'G':
                    # Switch to the other direction
                    if random.random() > 0.5:
                        actions[junction_id] = AIAction.SWITCH_TO_GREEN_EW.value
                    else:
                        actions[junction_id] = AIAction.SWITCH_TO_GREEN_NS.value
                continue
            
            # Rule 4: Default - maintain current
            actions[junction_id] = AIAction.MAINTAIN_CURRENT.value
        
        return actions
    
    def _handle_emergencies(self, vehicles: Dict, traffic_lights: Dict) -> Dict:
        """Handle emergency vehicle priority"""
        if not self.emergency_manager:
            return {}
        
        try:
            emergency_actions = self.emergency_manager.update(vehicles, traffic_lights)
            if emergency_actions:
                print("🚨 AI Controller: Emergency priority activated")
                return emergency_actions
        except Exception as e:
            print(f"❌ AI Controller: Error in emergency handling - {e}")
        
        return {}
    
    def _action_to_command(self, action_idx: int, junction_id: str, traffic_lights: Dict) -> Dict:
        """Convert AI action to traffic light command"""
        try:
            action = AIAction(action_idx)
        except:
            action = AIAction.MAINTAIN_CURRENT
        
        current_state = traffic_lights.get(junction_id, {}).get('state', 'G')
        
        if action == AIAction.MAINTAIN_CURRENT:
            state = current_state
            duration = 10.0
            reason = "AI: Maintain current phase"
        
        elif action == AIAction.SWITCH_TO_GREEN_NS:
            state = 'G'
            duration = 15.0
            reason = "AI: Priority to North-South traffic"
        
        elif action == AIAction.SWITCH_TO_GREEN_EW:
            state = 'G'
            duration = 15.0
            reason = "AI: Priority to East-West traffic"
        
        elif action == AIAction.SWITCH_TO_GREEN_ALL:
            state = 'G'
            duration = 20.0
            reason = "AI: Emergency priority - all directions green"
        
        elif action == AIAction.EXTEND_GREEN:
            state = current_state
            duration = 20.0 if current_state == 'G' else 3.0
            reason = "AI: Extended phase duration"
        
        elif action == AIAction.SHORTEN_GREEN:
            state = 'Y' if current_state == 'G' else current_state
            duration = 2.0
            reason = "AI: Shortened phase duration"
        
        else:
            state = current_state
            duration = 10.0
            reason = "AI: Default action"
        
        return {
            'state': state,
            'duration': duration,
            'strategy': 'mappo_ai' if self.use_mappo else 'rule_based_ai',
            'reason': reason,
            'ai_action': action.name,
            'action_confidence': 0.95 if self.use_mappo else 0.7,
            'junction_id': junction_id
        }
    
    def update_learning(self, rewards: Dict[str, float]):
        """Update MAPPO learning based on rewards from environment"""
        if self.use_mappo and self.mappo_agent:
            try:
                # This would need to be integrated with your training loop
                # For now, we'll just record the rewards
                print(f"📊 AI Learning: Received rewards {rewards}")
                # In a full implementation, you would call self.mappo_agent.update()
                # with the proper experience data
            except Exception as e:
                print(f"❌ AI Learning update failed: {e}")
    
    def get_ai_stats(self) -> Dict:
        """Get AI controller statistics"""
        total_decisions = len(self.decision_history)
        mappo_decisions = sum(1 for d in self.decision_history if d.get('using_mappo', False))
        
        return {
            'total_decisions': total_decisions,
            'mappo_decisions': mappo_decisions,
            'rule_based_decisions': total_decisions - mappo_decisions,
            'mappo_usage_rate': mappo_decisions / max(total_decisions, 1),
            'state_dimension': self.state_encoder.state_dim,
            'emergency_module_available': self.emergency_manager is not None,
            'mappo_agent_available': self.mappo_agent is not None
        }


# Global instance for easy access
ai_controller = MAPPOIntegratedController()

# Compatibility function
def get_ai_controller(use_mappo: bool = True) -> MAPPOIntegratedController:
    """Get AI controller instance"""
    return MAPPOIntegratedController(use_mappo=use_mappo)


if __name__ == "__main__":
    # Test the AI controller
    controller = MAPPOIntegratedController()
    
    # Test data
    test_vehicles = {
        'v1': {'x': 80, 'y': 100, 'direction': 'right', 'speed': 5.0},
        'v2': {'x': 320, 'y': 100, 'direction': 'left', 'speed': 0.5},
        'v3': {'x': 100, 'y': 80, 'direction': 'down', 'speed': 8.0},
    }
    
    test_lights = {
        'J1': {'state': 'G', 'phase_start_time': time.time() - 30},
        'J2': {'state': 'R', 'phase_start_time': time.time() - 10},
        'J3': {'state': 'G', 'phase_start_time': time.time() - 5},
        'J4': {'state': 'Y', 'phase_start_time': time.time() - 1}
    }
    
    test_metrics = {
        'total_waiting_time': 150,
        'vehicles_cleared': 25,
        'average_speed': 12.5
    }
    
    print("🧪 Testing AI Controller with MAPPO:")
    print("=" * 50)
    
    actions = controller.decide_actions(test_vehicles, test_lights, test_metrics)
    
    for junction, action in actions.items():
        print(f"📍 {junction}: {action['state']} for {action['duration']:.1f}s")
        print(f"   Reason: {action['reason']}")
        print(f"   AI Action: {action.get('ai_action', 'N/A')}")
        print(f"   Strategy: {action.get('strategy', 'N/A')}")
    
    # Show stats
    stats = controller.get_ai_stats()
    print(f"\n📈 AI Controller Stats:")
    print(f"   Total decisions: {stats['total_decisions']}")
    print(f"   MAPPO usage: {stats['mappo_usage_rate']:.1%}")
    print(f"   MAPPO agent available: {stats['mappo_agent_available']}")
    print(f"   Emergency module: {stats['emergency_module_available']}")