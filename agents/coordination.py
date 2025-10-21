import numpy as np
import time
from typing import Dict, List, Any, Tuple
from enum import Enum
import networkx as nx

class CoordinationStrategy(Enum):
    """Coordination strategies for multiple junctions"""
    INDEPENDENT = "independent"           # Each junction acts independently
    GREEN_WAVE = "green_wave"             # Coordinated green waves
    PRESSURE_BASED = "pressure_based"     # Pressure-based coordination
    CENTRALIZED = "centralized"           # Centralized control
    EMERGENCY_PRIORITY = "emergency_priority"  # Emergency vehicle coordination

class JunctionRelation:
    """Represents relationship between two junctions"""
    
    def __init__(self, junction_a: str, junction_b: str, distance: float, 
                 connection_type: str, traffic_flow: float = 0.0):
        self.junction_a = junction_a
        self.junction_b = junction_b
        self.distance = distance
        self.connection_type = connection_type  # 'straight', 'turn', 'merge'
        self.traffic_flow = traffic_flow
        self.coordination_weight = 1.0
        
    def update_traffic_flow(self, new_flow: float):
        """Update traffic flow with exponential smoothing"""
        self.traffic_flow = 0.8 * self.traffic_flow + 0.2 * new_flow

class GreenWaveCoordinator:
    """Coordinates green waves along arterial roads"""
    
    def __init__(self, wave_speed: float = 10.0):  # meters per second
        self.wave_speed = wave_speed
        self.active_waves = {}
        self.junction_relations = {}
        self.road_network = nx.Graph()
        
        # Initialize road network
        self._initialize_road_network()
    
    def _initialize_road_network(self):
        """Initialize the road network graph"""
        # Add junctions
        junctions = ['J1', 'J2', 'J3', 'J4']
        for junction in junctions:
            self.road_network.add_node(junction)
        
        # Add roads with distances and properties
        roads = [
            ('J1', 'J2', 200, 'east_west'),    # Horizontal road
            ('J3', 'J4', 200, 'east_west'),    # Horizontal road  
            ('J1', 'J3', 200, 'north_south'),  # Vertical road
            ('J2', 'J4', 200, 'north_south')   # Vertical road
        ]
        
        for road in roads:
            self.road_network.add_edge(road[0], road[1], 
                                     distance=road[2], 
                                     direction=road[3],
                                     traffic_flow=0.0)
    
    def plan_green_wave(self, start_junction: str, direction: str, 
                       vehicles: Dict, emergency_vehicle: str = None) -> Dict:
        """Plan a green wave for a specific direction"""
        wave_plan = {}
        
        if direction == 'east_west':
            if start_junction == 'J1':
                wave_path = ['J1', 'J2']
            else:  # J3
                wave_path = ['J3', 'J4']
        else:  # north_south
            if start_junction == 'J1':
                wave_path = ['J1', 'J3']
            else:  # J2
                wave_path = ['J2', 'J4']
        
        # Calculate timing for green wave
        current_time = time.time()
        for i, junction in enumerate(wave_path):
            # Calculate offset based on distance and wave speed
            if i == 0:
                offset = 0
            else:
                prev_junction = wave_path[i-1]
                distance = self.road_network[prev_junction][junction]['distance']
                offset = distance / self.wave_speed
            
            wave_plan[junction] = {
                'green_start_time': current_time + offset,
                'duration': 15.0,  # Green duration
                'wave_id': f"wave_{start_junction}_{direction}_{int(current_time)}",
                'emergency_priority': emergency_vehicle is not None
            }
        
        self.active_waves[wave_plan[wave_path[0]]['wave_id']] = wave_plan
        return wave_plan
    
    def update_traffic_flows(self, vehicles: Dict):
        """Update traffic flow measurements between junctions"""
        for u, v in self.road_network.edges():
            flow = self._calculate_flow_between_junctions(u, v, vehicles)
            self.road_network[u][v]['traffic_flow'] = flow
    
    def _calculate_flow_between_junctions(self, junction_a: str, junction_b: str, 
                                        vehicles: Dict) -> float:
        """Calculate traffic flow between two junctions"""
        flow_count = 0
        junction_a_pos = self._get_junction_position(junction_a)
        junction_b_pos = self._get_junction_position(junction_b)
        
        for vehicle in vehicles.values():
            vehicle_pos = [vehicle.get('x', 0), vehicle.get('y', 0)]
            
            # Check if vehicle is moving between these junctions
            if (self._is_between_junctions(vehicle_pos, junction_a_pos, junction_b_pos) and
                self._is_moving_toward(vehicle, junction_b_pos)):
                flow_count += 1
        
        return min(flow_count / 10.0, 1.0)  # Normalize
    
    def _get_junction_position(self, junction_id: str) -> List[float]:
        """Get position of a junction"""
        positions = {
            'J1': [100, 100],
            'J2': [300, 100], 
            'J3': [100, 300],
            'J4': [300, 300]
        }
        return positions.get(junction_id, [0, 0])
    
    def _is_between_junctions(self, point: List, pos_a: List, pos_b: List) -> bool:
        """Check if point is between two junctions"""
        # Simple bounding box check
        min_x, max_x = sorted([pos_a[0], pos_b[0]])
        min_y, max_y = sorted([pos_a[1], pos_b[1]])
        
        return (min_x <= point[0] <= max_x and 
                min_y <= point[1] <= max_y)
    
    def _is_moving_toward(self, vehicle: Dict, target_pos: List) -> bool:
        """Check if vehicle is moving toward target position"""
        vehicle_pos = [vehicle.get('x', 0), vehicle.get('y', 0)]
        direction = vehicle.get('direction', 'right')
        
        if direction == 'right' and vehicle_pos[0] < target_pos[0]:
            return True
        elif direction == 'left' and vehicle_pos[0] > target_pos[0]:
            return True
        elif direction == 'down' and vehicle_pos[1] < target_pos[1]:
            return True
        elif direction == 'up' and vehicle_pos[1] > target_pos[1]:
            return True
        
        return False

class PressureBasedCoordinator:
    """Pressure-based coordination using network pressure calculations"""
    
    def __init__(self):
        self.junction_pressures = {}
        self.pressure_history = {}
        self.learning_rate = 0.1
    
    def calculate_pressures(self, vehicles: Dict, traffic_lights: Dict) -> Dict[str, float]:
        """Calculate pressure for each junction"""
        pressures = {}
        
        for junction_id in traffic_lights.keys():
            pressure = self._calculate_junction_pressure(junction_id, vehicles, traffic_lights)
            pressures[junction_id] = pressure
            
            # Update history
            if junction_id not in self.pressure_history:
                self.pressure_history[junction_id] = []
            self.pressure_history[junction_id].append(pressure)
            if len(self.pressure_history[junction_id]) > 100:
                self.pressure_history[junction_id].pop(0)
        
        self.junction_pressures = pressures
        return pressures
    
    def _calculate_junction_pressure(self, junction_id: str, vehicles: Dict, 
                                   traffic_lights: Dict) -> float:
        """Calculate pressure for a specific junction"""
        junction_pos = self._get_junction_position(junction_id)
        
        # Calculate incoming and outgoing pressures
        incoming_pressure = self._calculate_incoming_pressure(junction_id, vehicles)
        outgoing_pressure = self._calculate_outgoing_pressure(junction_id, vehicles, traffic_lights)
        
        # Total pressure is difference between incoming and outgoing
        total_pressure = incoming_pressure - outgoing_pressure
        
        return max(0, total_pressure)  # Only positive pressure
    
    def _calculate_incoming_pressure(self, junction_id: str, vehicles: Dict) -> float:
        """Calculate pressure from vehicles approaching the junction"""
        junction_pos = self._get_junction_position(junction_id)
        pressure = 0.0
        
        for vehicle in vehicles.values():
            vehicle_pos = [vehicle.get('x', 0), vehicle.get('y', 0)]
            distance = np.linalg.norm(np.array(vehicle_pos) - np.array(junction_pos))
            
            if distance < 80:  # Within detection range
                # Pressure increases with closer distance and lower speed
                speed = vehicle.get('speed', 0)
                distance_factor = 1.0 - (distance / 80.0)
                speed_factor = 1.0 - (speed / 15.0)  # Lower speed = higher pressure
                
                pressure += distance_factor * speed_factor
        
        return min(pressure / 5.0, 1.0)  # Normalize
    
    def _calculate_outgoing_pressure(self, junction_id: str, vehicles: Dict, 
                                   traffic_lights: Dict) -> float:
        """Calculate pressure from vehicles leaving the junction"""
        junction_pos = self._get_junction_position(junction_id)
        current_light = traffic_lights.get(junction_id, {}).get('state', 'R')
        
        # Higher outgoing pressure when light is green
        if current_light == 'G':
            return 0.8  # High capacity to release pressure
        elif current_light == 'Y':
            return 0.3  # Reduced capacity
        else:
            return 0.1  # Low capacity
    
    def _get_junction_position(self, junction_id: str) -> List[float]:
        """Get position of a junction"""
        positions = {
            'J1': [100, 100],
            'J2': [300, 100],
            'J3': [100, 300],
            'J4': [300, 300]
        }
        return positions.get(junction_id, [0, 0])
    
    def get_coordination_advice(self, pressures: Dict) -> Dict[str, Any]:
        """Get coordination advice based on pressure distribution"""
        advice = {}
        
        # Find junction with highest pressure
        max_pressure_junction = max(pressures.keys(), key=lambda x: pressures[x])
        max_pressure = pressures[max_pressure_junction]
        
        # Give priority to high-pressure junctions
        for junction_id, pressure in pressures.items():
            if pressure > 0.7:  # High pressure threshold
                advice[junction_id] = {
                    'priority': 'high',
                    'suggested_action': 'extend_green',
                    'pressure_level': pressure,
                    'reason': f'High pressure ({pressure:.2f}) requires priority'
                }
            elif pressure < 0.2:  # Low pressure
                advice[junction_id] = {
                    'priority': 'low', 
                    'suggested_action': 'shorten_green',
                    'pressure_level': pressure,
                    'reason': f'Low pressure ({pressure:.2f}) allows phase change'
                }
            else:
                advice[junction_id] = {
                    'priority': 'medium',
                    'suggested_action': 'maintain',
                    'pressure_level': pressure,
                    'reason': f'Moderate pressure ({pressure:.2f})'
                }
        
        return advice

class CentralizedCoordinator:
    """Centralized coordination with global optimization"""
    
    def __init__(self):
        self.global_state = {}
        self.optimization_history = []
        self.coordination_matrix = self._initialize_coordination_matrix()
    
    def _initialize_coordination_matrix(self) -> np.ndarray:
        """Initialize coordination matrix between junctions"""
        # 4x4 matrix representing coordination strength between junctions
        matrix = np.array([
            [1.0, 0.8, 0.8, 0.6],  # J1 coordination with J1, J2, J3, J4
            [0.8, 1.0, 0.6, 0.8],  # J2 coordination
            [0.8, 0.6, 1.0, 0.8],  # J3 coordination  
            [0.6, 0.8, 0.8, 1.0]   # J4 coordination
        ])
        return matrix
    
    def optimize_globally(self, vehicles: Dict, traffic_lights: Dict, 
                         metrics: Dict) -> Dict[str, Any]:
        """Perform global optimization of traffic light timing"""
        optimization_result = {}
        
        # Calculate global metrics
        total_waiting_time = metrics.get('total_waiting_time', 0)
        total_vehicles = len(vehicles)
        avg_speed = metrics.get('average_speed', 0)
        
        # Simple global optimization logic
        for junction_id in traffic_lights.keys():
            # Consider neighbor states in optimization
            neighbor_influence = self._calculate_neighbor_influence(junction_id, traffic_lights)
            
            optimization_result[junction_id] = {
                'suggested_state': self._suggest_optimal_state(junction_id, vehicles, traffic_lights),
                'suggested_duration': 15.0,  # Base duration
                'neighbor_influence': neighbor_influence,
                'global_priority': self._calculate_global_priority(junction_id, vehicles),
                'optimization_score': self._calculate_optimization_score(junction_id, vehicles, metrics)
            }
        
        # Record optimization
        self.optimization_history.append({
            'timestamp': time.time(),
            'result': optimization_result,
            'global_metrics': {
                'total_waiting_time': total_waiting_time,
                'total_vehicles': total_vehicles,
                'average_speed': avg_speed
            }
        })
        
        # Keep only recent history
        if len(self.optimization_history) > 100:
            self.optimization_history.pop(0)
        
        return optimization_result
    
    def _calculate_neighbor_influence(self, junction_id: str, traffic_lights: Dict) -> float:
        """Calculate influence from neighboring junctions"""
        junction_index = {'J1': 0, 'J2': 1, 'J3': 2, 'J4': 3}[junction_id]
        influence = 0.0
        
        for neighbor_id, idx in {'J1': 0, 'J2': 1, 'J3': 2, 'J4': 3}.items():
            if neighbor_id != junction_id:
                neighbor_light = traffic_lights.get(neighbor_id, {}).get('state', 'R')
                coordination_strength = self.coordination_matrix[junction_index][idx]
                
                # Higher influence if neighbor has green light
                if neighbor_light == 'G':
                    influence += coordination_strength * 0.8
                elif neighbor_light == 'Y':
                    influence += coordination_strength * 0.3
                else:
                    influence += coordination_strength * 0.1
        
        return min(influence, 1.0)
    
    def _suggest_optimal_state(self, junction_id: str, vehicles: Dict, 
                             traffic_lights: Dict) -> str:
        """Suggest optimal traffic light state based on global optimization"""
        current_state = traffic_lights.get(junction_id, {}).get('state', 'G')
        
        # Simple rule-based optimization
        traffic_density = self._calculate_local_density(junction_id, vehicles)
        
        if traffic_density > 0.7:
            return 'G'  # Keep green for high density
        elif traffic_density < 0.3:
            return 'R'  # Can switch to red for low density
        else:
            return current_state  # Maintain current state
    
    def _calculate_local_density(self, junction_id: str, vehicles: Dict) -> float:
        """Calculate local traffic density around junction"""
        junction_pos = self._get_junction_position(junction_id)
        vehicles_near = 0
        
        for vehicle in vehicles.values():
            vehicle_pos = [vehicle.get('x', 0), vehicle.get('y', 0)]
            distance = np.linalg.norm(np.array(vehicle_pos) - np.array(junction_pos))
            if distance < 60:
                vehicles_near += 1
        
        return min(vehicles_near / 8.0, 1.0)
    
    def _calculate_global_priority(self, junction_id: str, vehicles: Dict) -> float:
        """Calculate global priority for junction"""
        # Junctions with more vehicles get higher priority
        local_density = self._calculate_local_density(junction_id, vehicles)
        
        # Adjust based on junction role (corners might have different priorities)
        if junction_id in ['J1', 'J4']:  # Corner junctions
            priority = local_density * 0.9
        else:  # Edge junctions
            priority = local_density * 1.1
        
        return min(priority, 1.0)
    
    def _calculate_optimization_score(self, junction_id: str, vehicles: Dict, 
                                   metrics: Dict) -> float:
        """Calculate optimization score for evaluation"""
        density = self._calculate_local_density(junction_id, vehicles)
        avg_speed = metrics.get('average_speed', 0) / 20.0  # Normalize
        waiting_time = min(metrics.get('total_waiting_time', 0) / 1000.0, 1.0)
        
        # Higher score is better (lower density, higher speed, lower waiting time)
        score = (1.0 - density) * 0.4 + avg_speed * 0.4 + (1.0 - waiting_time) * 0.2
        return score
    
    def _get_junction_position(self, junction_id: str) -> List[float]:
        """Get position of a junction"""
        positions = {
            'J1': [100, 100],
            'J2': [300, 100],
            'J3': [100, 300],
            'J4': [300, 300]
        }
        return positions.get(junction_id, [0, 0])

class MultiAgentCoordinator:
    """Main coordinator that combines all coordination strategies"""
    
    def __init__(self):
        self.green_wave_coordinator = GreenWaveCoordinator()
        self.pressure_coordinator = PressureBasedCoordinator()
        self.centralized_coordinator = CentralizedCoordinator()
        self.current_strategy = CoordinationStrategy.PRESSURE_BASED
        self.coordination_history = []
    
    def coordinate_actions(self, vehicles: Dict, traffic_lights: Dict, 
                          metrics: Dict, emergency_vehicles: Dict) -> Dict[str, Any]:
        """Main coordination method"""
        coordination_result = {
            'strategy': self.current_strategy.value,
            'timestamp': time.time(),
            'junction_advice': {},
            'global_optimization': {},
            'green_waves': {}
        }
        
        # Apply selected coordination strategy
        if self.current_strategy == CoordinationStrategy.GREEN_WAVE:
            coordination_result.update(self._apply_green_wave_coordination(vehicles, emergency_vehicles))
        
        elif self.current_strategy == CoordinationStrategy.PRESSURE_BASED:
            coordination_result.update(self._apply_pressure_coordination(vehicles, traffic_lights))
        
        elif self.current_strategy == CoordinationStrategy.CENTRALIZED:
            coordination_result.update(self._apply_centralized_coordination(vehicles, traffic_lights, metrics))
        
        elif self.current_strategy == CoordinationStrategy.EMERGENCY_PRIORITY:
            coordination_result.update(self._apply_emergency_coordination(emergency_vehicles))
        
        # Update traffic flows for green wave planning
        self.green_wave_coordinator.update_traffic_flows(vehicles)
        
        # Record coordination decision
        self.coordination_history.append(coordination_result)
        if len(self.coordination_history) > 200:
            self.coordination_history.pop(0)
        
        return coordination_result
    
    def _apply_green_wave_coordination(self, vehicles: Dict, emergency_vehicles: Dict) -> Dict:
        """Apply green wave coordination"""
        result = {}
        
        # Check for emergency vehicles that need green waves
        emergency_waves = {}
        for emergency_id, emergency_vehicle in emergency_vehicles.items():
            wave_plan = self.green_wave_coordinator.plan_green_wave(
                start_junction='J1',  # Would determine from emergency position
                direction='east_west',
                vehicles=vehicles,
                emergency_vehicle=emergency_id
            )
            emergency_waves.update(wave_plan)
        
        result['green_waves'] = emergency_waves or self.green_wave_coordinator.active_waves
        return result
    
    def _apply_pressure_coordination(self, vehicles: Dict, traffic_lights: Dict) -> Dict:
        """Apply pressure-based coordination"""
        pressures = self.pressure_coordinator.calculate_pressures(vehicles, traffic_lights)
        advice = self.pressure_coordinator.get_coordination_advice(pressures)
        
        return {
            'junction_advice': advice,
            'pressure_levels': pressures
        }
    
    def _apply_centralized_coordination(self, vehicles: Dict, traffic_lights: Dict, 
                                      metrics: Dict) -> Dict:
        """Apply centralized coordination"""
        optimization = self.centralized_coordinator.optimize_globally(vehicles, traffic_lights, metrics)
        
        return {
            'global_optimization': optimization
        }
    
    def _apply_emergency_coordination(self, emergency_vehicles: Dict) -> Dict:
        """Apply emergency priority coordination"""
        emergency_plan = {}
        
        for emergency_id, emergency_vehicle in emergency_vehicles.items():
            # Create emergency corridor
            emergency_plan[emergency_id] = {
                'type': 'emergency_corridor',
                'affected_junctions': ['J1', 'J2', 'J3', 'J4'],  # All junctions
                'priority': 'highest',
                'action': 'set_all_green'
            }
        
        return {
            'emergency_plan': emergency_plan
        }
    
    def set_coordination_strategy(self, strategy: CoordinationStrategy):
        """Set the current coordination strategy"""
        self.current_strategy = strategy
        print(f"🔄 Coordination strategy changed to: {strategy.value}")
    
    def get_coordination_stats(self) -> Dict:
        """Get coordination statistics"""
        if not self.coordination_history:
            return {}
        
        recent_coordinations = self.coordination_history[-50:]
        strategy_counts = {}
        
        for coord in recent_coordinations:
            strategy = coord.get('strategy', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'current_strategy': self.current_strategy.value,
            'recent_strategy_distribution': strategy_counts,
            'total_coordinations': len(self.coordination_history),
            'green_waves_active': len(self.green_wave_coordinator.active_waves),
            'average_pressure': np.mean(list(self.pressure_coordinator.junction_pressures.values())) 
                               if self.pressure_coordinator.junction_pressures else 0
        }

# Global coordinator instance
global_coordinator = MultiAgentCoordinator()

def get_coordinator() -> MultiAgentCoordinator:
    """Get the global coordinator instance"""
    return global_coordinator

if __name__ == "__main__":
    # Test the coordination system
    coordinator = MultiAgentCoordinator()
    
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
    
    test_emergencies = {}
    
    print("🔄 Testing Multi-Agent Coordination:")
    print("=" * 50)
    
    # Test each strategy
    strategies = list(CoordinationStrategy)
    
    for strategy in strategies:
        coordinator.set_coordination_strategy(strategy)
        result = coordinator.coordinate_actions(test_vehicles, test_lights, test_metrics, test_emergencies)
        
        print(f"\n📋 Strategy: {strategy.value}")
        print(f"   Junction Advice: {len(result.get('junction_advice', {}))} recommendations")
        print(f"   Green Waves: {len(result.get('green_waves', {}))} active")
        print(f"   Global Optimization: {len(result.get('global_optimization', {}))} results")
    
    # Show statistics
    stats = coordinator.get_coordination_stats()
    print(f"\n📊 Coordination Statistics:")
    print(f"   Current Strategy: {stats['current_strategy']}")
    print(f"   Total Coordinations: {stats['total_coordinations']}")
    print(f"   Strategy Distribution: {stats['recent_strategy_distribution']}")
    print(f"   Average Pressure: {stats['average_pressure']:.3f}")