import numpy as np
import time
from typing import Dict, List, Any
import random
from enum import Enum

class ControlStrategy(Enum):
    FIXED_TIMER = "fixed_timer"
    ACTUATED = "actuated"
    ADAPTIVE = "adaptive"
    MANUAL = "manual"
    EMERGENCY_PRIORITY = "emergency_priority"

class FixedTimerController:
    """
    Baseline controller using fixed-time signal timing
    Commonly used in traditional traffic systems
    """
    
    def __init__(self, cycle_time: float = 60.0, green_split: float = 0.5):
        self.cycle_time = cycle_time  # Total cycle time in seconds
        self.green_split = green_split  # Proportion of cycle for green
        self.green_time = cycle_time * green_split
        self.yellow_time = 3.0  # Fixed yellow time
        self.red_time = cycle_time - self.green_time - self.yellow_time
        self.junction_states = {}  # Track state per junction
        
    def update(self, junction_id: str, vehicles: Dict, current_state: str) -> Dict:
        """
        Update traffic light state based on fixed timing
        """
        current_time = time.time()
        
        # Initialize or get junction state
        if junction_id not in self.junction_states:
            self.junction_states[junction_id] = {
                'current_state': current_state,
                'last_switch_time': current_time
            }
        
        junction_state = self.junction_states[junction_id]
        elapsed = current_time - junction_state['last_switch_time']
        current_state = junction_state['current_state']
        
        # Phase transition logic
        if current_state == 'G' and elapsed >= self.green_time:
            # Switch to yellow
            junction_state['current_state'] = 'Y'
            junction_state['last_switch_time'] = current_time
        elif current_state == 'Y' and elapsed >= self.yellow_time:
            # Switch to red
            junction_state['current_state'] = 'R'
            junction_state['last_switch_time'] = current_time
        elif current_state == 'R' and elapsed >= self.red_time:
            # Switch back to green
            junction_state['current_state'] = 'G'
            junction_state['last_switch_time'] = current_time
        
        # Calculate remaining time in current phase
        remaining = self._get_remaining_duration(junction_state, current_time)
        
        return {
            'state': junction_state['current_state'],
            'duration': remaining,
            'strategy': ControlStrategy.FIXED_TIMER.value,
            'reason': f'Fixed timer: {elapsed:.1f}s in {current_state} phase'
        }
    
    def _get_remaining_duration(self, junction_state: Dict, current_time: float) -> float:
        """Calculate remaining time in current phase"""
        elapsed = current_time - junction_state['last_switch_time']
        current_state = junction_state['current_state']
        
        if current_state == 'G':
            return max(0, self.green_time - elapsed)
        elif current_state == 'Y':
            return max(0, self.yellow_time - elapsed)
        else:  # Red
            return max(0, self.red_time - elapsed)


class ActuatedController:
    """
    Actuated controller that responds to vehicle presence
    Uses induction loops or sensors to detect vehicles
    """
    
    def __init__(self, min_green: float = 10.0, max_green: float = 30.0, extension: float = 5.0):
        self.min_green = min_green
        self.max_green = max_green
        self.extension = extension
        self.junction_states = {}  # Track state per junction
        
    def update(self, junction_id: str, vehicles: Dict, current_state: str) -> Dict:
        """
        Update based on vehicle presence detection
        """
        current_time = time.time()
        
        # Initialize or get junction state
        if junction_id not in self.junction_states:
            self.junction_states[junction_id] = {
                'current_state': current_state,
                'green_start_time': current_time,
                'last_vehicle_time': current_time
            }
        
        junction_state = self.junction_states[junction_id]
        current_state = junction_state['current_state']
        
        # Detect vehicles approaching this junction
        vehicles_approaching = self._detect_approaching_vehicles(junction_id, vehicles)
        
        if current_state == 'G':
            green_elapsed = current_time - junction_state['green_start_time']
            
            # Update last vehicle detection time
            if vehicles_approaching:
                junction_state['last_vehicle_time'] = current_time
            
            # Check if we should extend green or switch
            time_since_last_vehicle = current_time - junction_state['last_vehicle_time']
            
            if vehicles_approaching and green_elapsed < self.max_green:
                # Extend green due to approaching vehicles
                remaining = self.extension
                reason = f'Vehicles detected: {len(vehicles_approaching)} approaching'
            elif green_elapsed >= self.min_green and time_since_last_vehicle >= self.extension:
                # Switch to yellow - minimum green met and gap detected
                junction_state['current_state'] = 'Y'
                junction_state['green_start_time'] = current_time
                remaining = 3.0  # Yellow duration
                reason = f'Gap detected, switching to yellow after {green_elapsed:.1f}s'
            else:
                # Stay green
                remaining = min(self.extension, self.max_green - green_elapsed)
                reason = f'Maintaining green, {len(vehicles_approaching)} vehicles'
                
        elif current_state == 'Y':
            yellow_elapsed = current_time - junction_state['green_start_time']
            if yellow_elapsed >= 3.0:
                junction_state['current_state'] = 'R'
                junction_state['green_start_time'] = current_time
                remaining = 2.0  # All-red duration
                reason = 'Yellow phase completed'
            else:
                remaining = 3.0 - yellow_elapsed
                reason = 'Yellow phase'
                
        else:  # Red state
            red_elapsed = current_time - junction_state['green_start_time']
            
            # Switch to green if vehicles waiting or max red time reached
            if vehicles_approaching or red_elapsed >= 15.0:
                junction_state['current_state'] = 'G'
                junction_state['green_start_time'] = current_time
                junction_state['last_vehicle_time'] = current_time
                remaining = self.min_green
                reason = f'Vehicles waiting: {len(vehicles_approaching)}'
            else:
                remaining = 15.0 - red_elapsed
                reason = 'Red phase, no vehicles waiting'
        
        return {
            'state': junction_state['current_state'],
            'duration': remaining,
            'strategy': ControlStrategy.ACTUATED.value,
            'reason': reason,
            'vehicles_detected': len(vehicles_approaching)
        }
    
    def _detect_approaching_vehicles(self, junction_id: str, vehicles: Dict) -> List[str]:
        """Detect vehicles approaching the junction"""
        approaching = []
        junction_positions = {
            'J1': [100, 100],
            'J2': [300, 100],
            'J3': [100, 300],
            'J4': [300, 300]
        }
        
        junction_pos = junction_positions.get(junction_id, [0, 0])
        detection_range = 50.0
        
        for vehicle_id, vehicle in vehicles.items():
            vehicle_pos = [vehicle.get('x', 0), vehicle.get('y', 0)]
            distance = np.sqrt((vehicle_pos[0]-junction_pos[0])**2 + 
                             (vehicle_pos[1]-junction_pos[1])**2)
            
            if distance <= detection_range:
                # Check if vehicle is moving toward junction
                if self._is_approaching_junction(vehicle_pos, vehicle.get('direction', 'right'), junction_pos):
                    approaching.append(vehicle_id)
        
        return approaching
    
    def _is_approaching_junction(self, vehicle_pos: List, direction: str, junction_pos: List) -> bool:
        """Check if vehicle is moving toward junction"""
        pos_x, pos_y = vehicle_pos
        j_x, j_y = junction_pos
        
        # Calculate distance first
        distance = np.sqrt((pos_x - j_x)**2 + (pos_y - j_y)**2)
        if distance > 50:  # Too far away
            return False
        
        # Check direction
        if direction == 'right' and pos_x < j_x and abs(pos_y - j_y) < 25:
            return True
        elif direction == 'left' and pos_x > j_x and abs(pos_y - j_y) < 25:
            return True
        elif direction == 'down' and pos_y < j_y and abs(pos_x - j_x) < 25:
            return True
        elif direction == 'up' and pos_y > j_y and abs(pos_x - j_x) < 25:
            return True
        
        return False


class AdaptiveController:
    """
    Adaptive controller that adjusts timing based on traffic conditions
    More sophisticated than actuated control
    """
    
    def __init__(self):
        self.junction_states = {}  # Track state per junction
        self.traffic_density_history = {}
        self.learning_rate = 0.1
        
    def update(self, junction_id: str, vehicles: Dict, current_state: str) -> Dict:
        """
        Adaptive control based on traffic patterns and learning
        """
        current_time = time.time()
        
        # Initialize junction state
        if junction_id not in self.junction_states:
            self.junction_states[junction_id] = {
                'current_state': current_state,
                'state_start_time': current_time,
                'planned_duration': 20.0 if current_state == 'G' else 3.0
            }
        
        junction_state = self.junction_states[junction_id]
        
        # Calculate current traffic density
        current_density = self._calculate_traffic_density(junction_id, vehicles)
        
        # Update density history
        if junction_id not in self.traffic_density_history:
            self.traffic_density_history[junction_id] = []
        
        self.traffic_density_history[junction_id].append(current_density)
        if len(self.traffic_density_history[junction_id]) > 100:
            self.traffic_density_history[junction_id].pop(0)
        
        # Get optimal timing based on density
        optimal_timing = self._calculate_optimal_timing(junction_id, current_density)
        
        # Check if phase should change
        state_elapsed = current_time - junction_state['state_start_time']
        
        if state_elapsed >= junction_state['planned_duration']:
            # Transition to next phase
            if junction_state['current_state'] == 'G':
                new_state = 'Y'
                duration = 3.0
                reason = f'Green phase completed ({state_elapsed:.1f}s), density: {current_density:.2f}'
            elif junction_state['current_state'] == 'Y':
                new_state = 'R'
                duration = 2.0  # All-red time
                reason = 'Yellow phase completed'
            else:  # Red
                new_state = 'G'
                duration = optimal_timing['green_time']
                reason = f'Switching to green, optimal time: {duration:.1f}s for density: {current_density:.2f}'
            
            # Update junction state
            junction_state['current_state'] = new_state
            junction_state['state_start_time'] = current_time
            junction_state['planned_duration'] = duration
        else:
            # Maintain current state
            new_state = junction_state['current_state']
            duration = junction_state['planned_duration'] - state_elapsed
            
            # Adaptive adjustment: extend green if high density
            if (new_state == 'G' and current_density > 0.7 and 
                duration < 5.0 and state_elapsed < optimal_timing['max_green']):
                extension = 5.0
                junction_state['planned_duration'] += extension
                duration += extension
                reason = f'Extended green due to high density: {current_density:.2f}'
            else:
                reason = f'Maintaining {new_state}, density: {current_density:.2f}'
        
        return {
            'state': new_state,
            'duration': duration,
            'strategy': ControlStrategy.ADAPTIVE.value,
            'reason': reason,
            'traffic_density': current_density,
            'optimal_green_time': optimal_timing['green_time']
        }
    
    def _calculate_traffic_density(self, junction_id: str, vehicles: Dict) -> float:
        """Calculate traffic density around junction (0-1 scale)"""
        junction_positions = {
            'J1': [100, 100],
            'J2': [300, 100],
            'J3': [100, 300],
            'J4': [300, 300]
        }
        
        junction_pos = junction_positions.get(junction_id, [0, 0])
        detection_range = 80.0
        vehicles_in_range = 0
        
        for vehicle_id, vehicle in vehicles.items():
            vehicle_pos = [vehicle.get('x', 0), vehicle.get('y', 0)]
            distance = np.sqrt((vehicle_pos[0]-junction_pos[0])**2 + 
                             (vehicle_pos[1]-junction_pos[1])**2)
            
            if distance <= detection_range:
                vehicles_in_range += 1
        
        # Normalize to 0-1 scale (assuming max 12 vehicles in range is high density)
        density = min(1.0, vehicles_in_range / 12.0)
        return density
    
    def _calculate_optimal_timing(self, junction_id: str, density: float) -> Dict:
        """Calculate optimal signal timing based on traffic density"""
        # Base timing parameters
        min_green = 10.0
        max_green = 45.0
        
        # Adjust green time based on density (longer for higher density)
        green_time = min_green + (max_green - min_green) * density
        
        # Adjust cycle time based on density (shorter cycles for higher density)
        base_cycle = 60.0
        cycle_time = base_cycle * (1.0 - 0.3 * density)  # Reduce cycle time by up to 30%
        
        return {
            'green_time': green_time,
            'cycle_time': cycle_time,
            'yellow_time': 3.0,
            'all_red_time': 2.0,
            'max_green': max_green
        }


class ManualController:
    """
    Manual controller for baseline comparison
    Simulates human traffic operator decisions
    """
    
    def __init__(self):
        self.junction_states = {}  # Track state per junction
        self.manual_mode_probability = 0.02  # 2% chance of manual intervention
        
    def update(self, junction_id: str, vehicles: Dict, current_state: str) -> Dict:
        """
        Simple manual control with periodic changes
        """
        current_time = time.time()
        
        # Initialize junction state
        if junction_id not in self.junction_states:
            self.junction_states[junction_id] = {
                'current_state': current_state,
                'state_start_time': current_time,
                'planned_duration': 20.0 if current_state == 'G' else (3.0 if current_state == 'Y' else 10.0),
                'manual_mode': False,
                'last_manual_change': current_time
            }
        
        junction_state = self.junction_states[junction_id]
        
        # Random chance to toggle manual mode
        if random.random() < self.manual_mode_probability:
            junction_state['manual_mode'] = not junction_state['manual_mode']
            junction_state['last_manual_change'] = current_time
        
        state_elapsed = current_time - junction_state['state_start_time']
        
        if junction_state['manual_mode']:
            # Manual control mode
            if state_elapsed >= junction_state['planned_duration']:
                # Manual phase change
                if junction_state['current_state'] == 'G':
                    new_state = 'Y'
                    duration = 3.0
                elif junction_state['current_state'] == 'Y':
                    new_state = 'R'
                    duration = random.uniform(5.0, 15.0)  # Variable red time
                else:
                    new_state = 'G'
                    duration = random.uniform(15.0, 30.0)  # Variable green time
                
                junction_state['current_state'] = new_state
                junction_state['state_start_time'] = current_time
                junction_state['planned_duration'] = duration
                reason = "Manual operator intervention"
            else:
                new_state = junction_state['current_state']
                duration = junction_state['planned_duration'] - state_elapsed
                reason = "Manual control active"
        else:
            # Automatic mode - simple fixed timing
            if state_elapsed >= junction_state['planned_duration']:
                if junction_state['current_state'] == 'G':
                    new_state = 'Y'
                    duration = 3.0
                elif junction_state['current_state'] == 'Y':
                    new_state = 'R'
                    duration = 10.0
                else:
                    new_state = 'G'
                    duration = 20.0
                
                junction_state['current_state'] = new_state
                junction_state['state_start_time'] = current_time
                junction_state['planned_duration'] = duration
                reason = "Automatic phase transition"
            else:
                new_state = junction_state['current_state']
                duration = junction_state['planned_duration'] - state_elapsed
                reason = "Automatic timing"
        
        return {
            'state': new_state,
            'duration': duration,
            'strategy': ControlStrategy.MANUAL.value,
            'reason': reason,
            'manual_mode': junction_state['manual_mode']
        }


class EmergencyPriorityController:
    """
    Special controller that prioritizes emergency vehicles
    Can be used as a standalone or integrated with other controllers
    """
    
    def __init__(self):
        self.junction_states = {}
        self.emergency_active = False
        self.emergency_vehicle_id = None
        
    def update(self, junction_id: str, vehicles: Dict, current_state: str) -> Dict:
        """
        Priority control for emergency vehicles
        """
        current_time = time.time()
        
        # Initialize junction state
        if junction_id not in self.junction_states:
            self.junction_states[junction_id] = {
                'current_state': current_state,
                'state_start_time': current_time,
                'planned_duration': 20.0
            }
        
        junction_state = self.junction_states[junction_id]
        
        # Detect emergency vehicles
        emergency_vehicles = self._detect_emergency_vehicles(vehicles)
        
        if emergency_vehicles:
            # Emergency vehicle detected - give priority
            self.emergency_active = True
            self.emergency_vehicle_id = list(emergency_vehicles.keys())[0]
            
            # Set all directions to green for emergency passage
            if junction_state['current_state'] != 'G':
                junction_state['current_state'] = 'G'
                junction_state['state_start_time'] = current_time
                junction_state['planned_duration'] = 15.0  # Extended green for emergency
            
            duration = junction_state['planned_duration'] - (current_time - junction_state['state_start_time'])
            reason = f"EMERGENCY: {self.emergency_vehicle_id} approaching - Priority green"
            
        else:
            # No emergency - use normal fixed timing
            self.emergency_active = False
            self.emergency_vehicle_id = None
            
            state_elapsed = current_time - junction_state['state_start_time']
            if state_elapsed >= junction_state['planned_duration']:
                # Normal phase transition
                if junction_state['current_state'] == 'G':
                    new_state = 'Y'
                    duration = 3.0
                elif junction_state['current_state'] == 'Y':
                    new_state = 'R'
                    duration = 10.0
                else:
                    new_state = 'G'
                    duration = 20.0
                
                junction_state['current_state'] = new_state
                junction_state['state_start_time'] = current_time
                junction_state['planned_duration'] = duration
                reason = "Normal operation - phase transition"
            else:
                duration = junction_state['planned_duration'] - state_elapsed
                reason = "Normal operation"
        
        return {
            'state': junction_state['current_state'],
            'duration': duration,
            'strategy': ControlStrategy.EMERGENCY_PRIORITY.value,
            'reason': reason,
            'emergency_active': self.emergency_active,
            'emergency_vehicle': self.emergency_vehicle_id
        }
    
    def _detect_emergency_vehicles(self, vehicles: Dict) -> Dict:
        """Detect emergency vehicles in the simulation"""
        emergency_vehicles = {}
        for vehicle_id, vehicle in vehicles.items():
            if (vehicle.get('is_emergency', False) or 
                vehicle.get('type') == 'emergency' or
                vehicle.get('emergency_priority', 0) > 0):
                emergency_vehicles[vehicle_id] = vehicle
        return emergency_vehicles


class BaselineControllerManager:
    """
    Manager for all baseline controllers
    Allows easy switching between different control strategies
    """
    
    def __init__(self):
        self.controllers = {}
        self.current_strategy = ControlStrategy.FIXED_TIMER
        self.performance_metrics = {}
        
        # Initialize all controllers
        self.controllers[ControlStrategy.FIXED_TIMER] = FixedTimerController()
        self.controllers[ControlStrategy.ACTUATED] = ActuatedController()
        self.controllers[ControlStrategy.ADAPTIVE] = AdaptiveController()
        self.controllers[ControlStrategy.MANUAL] = ManualController()
        self.controllers[ControlStrategy.EMERGENCY_PRIORITY] = EmergencyPriorityController()
    
    def set_strategy(self, strategy: ControlStrategy):
        """Set the current control strategy"""
        self.current_strategy = strategy
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available control strategies"""
        return [strategy.value for strategy in ControlStrategy]
    
    def update_junction(self, junction_id: str, vehicles: Dict, current_state: str) -> Dict:
        """Update a single junction using current strategy"""
        controller = self.controllers.get(self.current_strategy)
        if not controller:
            # Fallback to fixed timer
            controller = self.controllers[ControlStrategy.FIXED_TIMER]
        
        result = controller.update(junction_id, vehicles, current_state)
        
        # Record performance metrics
        self._record_metrics(junction_id, vehicles, result)
        
        return result
    
    def _record_metrics(self, junction_id: str, vehicles: Dict, control_result: Dict):
        """Record performance metrics for analysis"""
        if junction_id not in self.performance_metrics:
            self.performance_metrics[junction_id] = {
                'total_vehicles': 0,
                'waiting_time': 0,
                'state_changes': 0,
                'control_decisions': [],
                'total_vehicles_processed': 0
            }
        
        metrics = self.performance_metrics[junction_id]
        
        # Count vehicles in detection range
        vehicles_in_range = self._count_vehicles_in_range(junction_id, vehicles)
        metrics['total_vehicles_processed'] += vehicles_in_range
        
        # Record control decision
        metrics['control_decisions'].append({
            'timestamp': time.time(),
            'strategy': control_result['strategy'],
            'state': control_result['state'],
            'duration': control_result['duration'],
            'reason': control_result.get('reason', ''),
            'vehicles_count': vehicles_in_range,
            'emergency_active': control_result.get('emergency_active', False)
        })
        
        # Keep only last 500 decisions
        if len(metrics['control_decisions']) > 500:
            metrics['control_decisions'].pop(0)
    
    def _count_vehicles_in_range(self, junction_id: str, vehicles: Dict) -> int:
        """Count vehicles within detection range of junction"""
        junction_positions = {
            'J1': [100, 100],
            'J2': [300, 100],
            'J3': [100, 300],
            'J4': [300, 300]
        }
        
        junction_pos = junction_positions.get(junction_id, [0, 0])
        detection_range = 60.0
        count = 0
        
        for vehicle in vehicles.values():
            vehicle_pos = [vehicle.get('x', 0), vehicle.get('y', 0)]
            distance = np.sqrt((vehicle_pos[0]-junction_pos[0])**2 + 
                             (vehicle_pos[1]-junction_pos[1])**2)
            if distance <= detection_range:
                count += 1
        
        return count
    
    def get_performance_report(self) -> Dict:
        """Generate performance report for all baseline controllers"""
        report = {
            'current_strategy': self.current_strategy.value,
            'junction_metrics': self.performance_metrics,
            'strategy_comparison': self._compare_strategies(),
            'total_controllers': len(self.controllers)
        }
        
        # Calculate some basic statistics
        total_vehicles = 0
        total_decisions = 0
        
        for junction_metrics in self.performance_metrics.values():
            total_vehicles += junction_metrics['total_vehicles_processed']
            total_decisions += len(junction_metrics['control_decisions'])
        
        report['summary'] = {
            'total_vehicles_processed': total_vehicles,
            'total_control_decisions': total_decisions,
            'junctions_monitored': len(self.performance_metrics)
        }
        
        return report
    
    def _compare_strategies(self) -> Dict:
        """Compare performance of different strategies (placeholder)"""
        # In a real implementation, this would analyze historical data
        # For now, return theoretical performance data
        return {
            'fixed_timer': {
                'avg_waiting_time': 25.3, 
                'throughput': 45,
                'emergency_response': 'poor',
                'adaptability': 'low'
            },
            'actuated': {
                'avg_waiting_time': 18.7, 
                'throughput': 52,
                'emergency_response': 'medium', 
                'adaptability': 'medium'
            },
            'adaptive': {
                'avg_waiting_time': 15.2, 
                'throughput': 58,
                'emergency_response': 'good',
                'adaptability': 'high'
            },
            'manual': {
                'avg_waiting_time': 22.1, 
                'throughput': 48,
                'emergency_response': 'variable',
                'adaptability': 'variable'
            },
            'emergency_priority': {
                'avg_waiting_time': 28.5, 
                'throughput': 42,
                'emergency_response': 'excellent',
                'adaptability': 'low'
            }
        }
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        self.performance_metrics = {}


# Global instance for easy access
baseline_manager = BaselineControllerManager()


def get_controller(use_ai: bool = True, baseline_strategy: str = "fixed_timer"):
    """
    Get the appropriate controller based on configuration
    
    Args:
        use_ai: Whether to use AI controller (True) or baseline (False)
        baseline_strategy: Which baseline strategy to use if not AI
    
    Returns:
        Controller instance
    """
    if use_ai:
        try:
            # Try to import AI controller
            from agents.mappo_agent import MAPPOAgent
            return MAPPOAgent()
        except ImportError:
            print("Warning: AI controller not available. Falling back to adaptive baseline.")
            baseline_manager.set_strategy(ControlStrategy.ADAPTIVE)
            return baseline_manager
    else:
        try:
            strategy = ControlStrategy(baseline_strategy)
            baseline_manager.set_strategy(strategy)
            return baseline_manager
        except ValueError:
            print(f"Warning: Unknown strategy {baseline_strategy}. Using fixed timer.")
            baseline_manager.set_strategy(ControlStrategy.FIXED_TIMER)
            return baseline_manager


# Example usage and testing
if __name__ == "__main__":
    # Test the baseline controllers
    manager = BaselineControllerManager()
    
    # Test each strategy
    test_vehicles = {
        'v1': {'x': 80, 'y': 100, 'direction': 'right', 'speed': 5.0},
        'v2': {'x': 320, 'y': 100, 'direction': 'left', 'speed': 4.0},
        'emergency_1': {'x': 150, 'y': 300, 'direction': 'up', 'speed': 8.0, 
                       'is_emergency': True, 'type': 'ambulance'}
    }
    
    strategies = list(ControlStrategy)
    
    print("Testing Baseline Controllers:")
    print("=" * 50)
    
    for strategy in strategies:
        manager.set_strategy(strategy)
        print(f"\nStrategy: {strategy.value}")
        print("-" * 30)
        
        for junction in ['J1', 'J2', 'J3', 'J4']:
            result = manager.update_junction(junction, test_vehicles, 'G')
            print(f"  {junction}: {result['state']} for {result['duration']:.1f}s - {result['reason']}")
    
    # Show performance report
    report = manager.get_performance_report()
    print(f"\nPerformance Summary:")
    print(f"Total vehicles processed: {report['summary']['total_vehicles_processed']}")
    print(f"Junctions monitored: {report['summary']['junctions_monitored']}")