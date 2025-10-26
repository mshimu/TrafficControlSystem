import numpy as np
import time
from typing import Dict, List, Any, Tuple
from collections import deque, defaultdict
import pandas as pd

class TrafficMetrics:
    """Computes and tracks traffic simulation metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        # Core metrics
        self.total_waiting_time = 0.0
        self.vehicles_cleared = 0
        self.emergency_vehicles_cleared = 0
        self.total_travel_time = 0.0
        self.total_distance = 0.0
        
        # Time series data
        self.waiting_times = deque(maxlen=self.window_size)
        self.throughput = deque(maxlen=self.window_size)
        self.speeds = deque(maxlen=self.window_size)
        self.vehicle_counts = deque(maxlen=self.window_size)
        self.emergency_response_times = deque(maxlen=self.window_size)
        
        # Vehicle tracking
        self.vehicle_start_times = {}
        self.vehicle_waiting_times = {}
        self.vehicle_positions = {}
        
        # Junction-specific metrics
        self.junction_metrics = defaultdict(lambda: {
            'waiting_time': 0.0,
            'throughput': 0,
            'queue_length': 0,
            'utilization': 0.0
        })
        
        # Episode statistics
        self.episode_start_time = time.time()
        self.step_count = 0
    
    def update_vehicle_metrics(self, vehicles: Dict, traffic_lights: Dict):
        """Update metrics based on current vehicle states"""
        current_time = time.time()
        self.step_count += 1
        
        # Track vehicle states
        for vehicle_id, vehicle in vehicles.items():
            position = (vehicle.get('x', 0), vehicle.get('y', 0))
            speed = vehicle.get('speed', 0)
            is_emergency = vehicle.get('is_emergency', False)
            
            # Initialize vehicle tracking
            if vehicle_id not in self.vehicle_start_times:
                self.vehicle_start_times[vehicle_id] = current_time
                self.vehicle_waiting_times[vehicle_id] = 0.0
                self.vehicle_positions[vehicle_id] = position
            
            # Calculate waiting time (when speed is very low)
            if speed < 0.5:  # Considered waiting
                self.vehicle_waiting_times[vehicle_id] += 1.0  # 1 second per step
                self.total_waiting_time += 1.0
            
            # Update position and distance
            old_position = self.vehicle_positions[vehicle_id]
            distance = np.sqrt((position[0]-old_position[0])**2 + (position[1]-old_position[1])**2)
            self.total_distance += distance
            self.vehicle_positions[vehicle_id] = position
            
            # Add to speed history
            self.speeds.append(speed)
        
        # Update vehicle count
        self.vehicle_counts.append(len(vehicles))
        
        # Update junction metrics
        self._update_junction_metrics(vehicles, traffic_lights)
    
    def _update_junction_metrics(self, vehicles: Dict, traffic_lights: Dict):
        """Update metrics for each junction"""
        junction_positions = {
            'J1': (100, 100),
            'J2': (300, 100),
            'J3': (100, 300),
            'J4': (300, 300)
        }
        
        for junction_id, position in junction_positions.items():
            jx, jy = position
            waiting_time = 0.0
            queue_length = 0
            
            # Calculate vehicles near junction
            for vehicle in vehicles.values():
                vx, vy = vehicle.get('x', 0), vehicle.get('y', 0)
                distance = np.sqrt((vx-jx)**2 + (vy-jy)**2)
                
                if distance < 30:  # Within junction influence
                    if vehicle.get('speed', 0) < 1.0:  # Stopped or very slow
                        waiting_time += 1.0
                        queue_length += 1
            
            self.junction_metrics[junction_id]['waiting_time'] = waiting_time
            self.junction_metrics[junction_id]['queue_length'] = queue_length
            
            # Calculate utilization (proportion of time with vehicles)
            if len(self.vehicle_counts) > 0:
                avg_vehicles_near = queue_length / max(len(vehicles), 1)
                self.junction_metrics[junction_id]['utilization'] = min(avg_vehicles_near, 1.0)
    
    def vehicle_cleared(self, vehicle_id: str, is_emergency: bool = False):
        """Record when a vehicle clears the simulation"""
        if vehicle_id in self.vehicle_start_times:
            travel_time = time.time() - self.vehicle_start_times[vehicle_id]
            self.total_travel_time += travel_time
            
            # Record waiting time for this vehicle
            if vehicle_id in self.vehicle_waiting_times:
                self.waiting_times.append(self.vehicle_waiting_times[vehicle_id])
            
            # Clean up
            del self.vehicle_start_times[vehicle_id]
            del self.vehicle_waiting_times[vehicle_id]
            if vehicle_id in self.vehicle_positions:
                del self.vehicle_positions[vehicle_id]
        
        self.vehicles_cleared += 1
        self.throughput.append(1)  # Record this clearance
        
        if is_emergency:
            self.emergency_vehicles_cleared += 1
    
    def record_emergency_response(self, emergency_id: str, response_time: float):
        """Record emergency vehicle response time"""
        self.emergency_response_times.append(response_time)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values"""
        avg_waiting_time = np.mean(self.waiting_times) if self.waiting_times else 0.0
        avg_speed = np.mean(self.speeds) if self.speeds else 0.0
        current_throughput = sum(self.throughput) if self.throughput else 0
        
        # Calculate efficiency metrics
        total_operation_time = time.time() - self.episode_start_time
        efficiency = (self.total_distance / max(total_operation_time, 1)) if total_operation_time > 0 else 0
        
        return {
            # Core metrics
            'total_waiting_time': self.total_waiting_time,
            'vehicles_cleared': self.vehicles_cleared,
            'emergency_cleared': self.emergency_vehicles_cleared,
            'average_speed': avg_speed,
            'active_vehicles': len(self.vehicle_start_times),
            
            # Averages
            'average_waiting_time': avg_waiting_time,
            'throughput_rate': current_throughput / max(len(self.throughput), 1),
            'efficiency': efficiency,
            
            # Emergency response
            'average_emergency_response': np.mean(self.emergency_response_times) if self.emergency_response_times else 0,
            'emergencies_handled': len(self.emergency_response_times),
            
            # System performance
            'step_count': self.step_count,
            'operation_time': total_operation_time,
            
            # Junction metrics
            'junction_metrics': dict(self.junction_metrics)
        }
    
    def get_performance_score(self) -> float:
        """Calculate overall performance score (0-1)"""
        metrics = self.get_current_metrics()
        
        # Normalize individual metrics to 0-1 scale
        waiting_score = 1.0 - min(metrics['total_waiting_time'] / 1000.0, 1.0)
        throughput_score = min(metrics['vehicles_cleared'] / 50.0, 1.0)
        speed_score = min(metrics['average_speed'] / 20.0, 1.0)
        emergency_score = 1.0 - min(metrics.get('average_emergency_response', 30) / 30.0, 1.0)
        
        # Weighted combination
        performance_score = (
            waiting_score * 0.3 +
            throughput_score * 0.3 + 
            speed_score * 0.2 +
            emergency_score * 0.2
        )
        
        return max(0.0, min(1.0, performance_score))

class AITrainingMetrics:
    """Tracks metrics specific to AI training"""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_values = []
        
        self.episode_metrics = []
        self.best_reward = -float('inf')
    
    def record_episode(self, episode_reward: float, episode_length: int, metrics: Dict):
        """Record episode results"""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_metrics.append(metrics)
        
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
    
    def record_training_step(self, total_loss: float, value_loss: float, 
                           policy_loss: float, entropy: float):
        """Record training step metrics"""
        self.training_losses.append(total_loss)
        self.value_losses.append(value_loss)
        self.policy_losses.append(policy_loss)
        self.entropy_values.append(entropy)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
        recent_lengths = self.episode_lengths[-100:]
        
        return {
            'episodes_completed': len(self.episode_rewards),
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'best_reward': self.best_reward,
            'mean_episode_length': np.mean(recent_lengths),
            'training_steps': len(self.training_losses),
            'mean_loss': np.mean(self.training_losses[-1000:]) if self.training_losses else 0,
            'current_entropy': self.entropy_values[-1] if self.entropy_values else 0
        }
    
    def get_success_rate(self, reward_threshold: float = 0.0) -> float:
        """Calculate success rate based on reward threshold"""
        if not self.episode_rewards:
            return 0.0
        
        successful_episodes = sum(1 for reward in self.episode_rewards if reward >= reward_threshold)
        return successful_episodes / len(self.episode_rewards)

class ComparativeAnalyzer:
    """Compares performance between different controllers"""
    
    def __init__(self):
        self.controller_metrics = {}
        self.comparison_results = {}
    
    def add_controller_metrics(self, controller_name: str, metrics: TrafficMetrics):
        """Add metrics for a controller"""
        self.controller_metrics[controller_name] = metrics.get_current_metrics()
    
    def compare_controllers(self) -> Dict[str, Any]:
        """Compare all controllers and rank them"""
        if len(self.controller_metrics) < 2:
            return {}
        
        comparison = {}
        
        for metric_name in ['total_waiting_time', 'vehicles_cleared', 'average_speed']:
            best_controller = None
            best_value = float('inf') if 'waiting' in metric_name else -float('inf')
            
            for controller, metrics in self.controller_metrics.items():
                value = metrics.get(metric_name, 0)
                
                if 'waiting' in metric_name:  # Lower is better
                    if value < best_value:
                        best_value = value
                        best_controller = controller
                else:  # Higher is better
                    if value > best_value:
                        best_value = value
                        best_controller = controller
            
            comparison[metric_name] = {
                'best_controller': best_controller,
                'best_value': best_value
            }
        
        # Calculate overall ranking
        rankings = self._calculate_overall_ranking()
        comparison['overall_ranking'] = rankings
        
        self.comparison_results = comparison
        return comparison
    
    def _calculate_overall_ranking(self) -> List[Tuple[str, float]]:
        """Calculate overall ranking based on multiple metrics"""
        scores = {}
        
        for controller, metrics in self.controller_metrics.items():
            # Normalize and combine multiple metrics
            waiting_score = 1.0 - min(metrics.get('total_waiting_time', 0) / 1000.0, 1.0)
            throughput_score = min(metrics.get('vehicles_cleared', 0) / 50.0, 1.0)
            speed_score = min(metrics.get('average_speed', 0) / 20.0, 1.0)
            
            overall_score = (waiting_score * 0.4 + throughput_score * 0.4 + speed_score * 0.2)
            scores[controller] = overall_score
        
        # Sort by score (descending)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def generate_comparison_report(self) -> str:
        """Generate a text report of the comparison"""
        if not self.comparison_results:
            return "No comparison data available."
        
        report = "🚦 Controller Performance Comparison Report\n"
        report += "=" * 50 + "\n\n"
        
        # Overall ranking
        report += "Overall Ranking:\n"
        for i, (controller, score) in enumerate(self.comparison_results.get('overall_ranking', [])):
            report += f"{i+1}. {controller}: {score:.3f}\n"
        
        report += "\nBest Performance by Metric:\n"
        for metric, data in self.comparison_results.items():
            if metric != 'overall_ranking':
                report += f"- {metric}: {data['best_controller']} ({data['best_value']:.2f})\n"
        
        return report

# Global metrics instances
traffic_metrics = TrafficMetrics()
training_metrics = AITrainingMetrics()
comparative_analyzer = ComparativeAnalyzer()

def get_traffic_metrics() -> TrafficMetrics:
    """Get the global traffic metrics instance"""
    return traffic_metrics

def get_training_metrics() -> AITrainingMetrics:
    """Get the global training metrics instance"""
    return training_metrics

def get_comparative_analyzer() -> ComparativeAnalyzer:
    """Get the global comparative analyzer instance"""
    return comparative_analyzer