import numpy as np
import time
from typing import Dict, List, Any, Tuple
from enum import Enum
import random

class ControlMode(Enum):
    """Control modes for mixed control system"""
    AI_ONLY = "ai_only"
    BASELINE_ONLY = "baseline_only"
    HYBRID = "hybrid"
    ADAPTIVE_SWITCHING = "adaptive_switching"
    ENSEMBLE = "ensemble"
    EMERGENCY_OVERRIDE = "emergency_override"

class JunctionAssignment(Enum):
    """How to assign controllers to junctions"""
    ALL_AI = "all_ai"
    ALL_BASELINE = "all_baseline"
    ALTERNATING = "alternating"
    TRAFFIC_AWARE = "traffic_aware"
    PERFORMANCE_BASED = "performance_based"

class MixedControlManager:
    """
    Manages hybrid control strategies combining AI and baseline controllers
    """
    
    def __init__(self):
        self.control_mode = ControlMode.HYBRID
        self.junction_assignment = JunctionAssignment.TRAFFIC_AWARE
        self.controllers = {}
        self.junction_assignments = {}
        self.performance_metrics = {}
        self.switching_history = []
        
        # Performance thresholds for adaptive switching
        self.performance_thresholds = {
            'waiting_time_threshold': 30.0,  # seconds
            'throughput_threshold': 40,      # vehicles cleared per minute
            'emergency_response_threshold': 10.0,  # seconds
        }
        
        # Initialize controllers
        self._initialize_controllers()
        
        # Initialize junction assignments
        self._initialize_junction_assignments()
    
    def _initialize_controllers(self):
        """Initialize both AI and baseline controllers"""
        try:
            from agents.ai_controller import get_ai_controller
            self.controllers['ai'] = get_ai_controller(use_mappo=True)
            print("🤖 Mixed Control: AI controller initialized")
        except ImportError as e:
            print(f"❌ Mixed Control: AI controller failed - {e}")
            self.controllers['ai'] = None
        
        try:
            from agents.baseline_controllers import baseline_manager
            self.controllers['baseline'] = baseline_manager
            print("🔄 Mixed Control: Baseline controller initialized")
        except ImportError as e:
            print(f"❌ Mixed Control: Baseline controller failed - {e}")
            self.controllers['baseline'] = None
        
        try:
            from agents.coordination import global_coordinator
            self.controllers['coordinator'] = global_coordinator
            print("🔄 Mixed Control: Coordinator initialized")
        except ImportError as e:
            print(f"❌ Mixed Control: Coordinator failed - {e}")
            self.controllers['coordinator'] = None
    
    def _initialize_junction_assignments(self):
        """Initialize default junction assignments"""
        junctions = ['J1', 'J2', 'J3', 'J4']
        
        if self.junction_assignment == JunctionAssignment.ALL_AI:
            self.junction_assignments = {jid: 'ai' for jid in junctions}
        
        elif self.junction_assignment == JunctionAssignment.ALL_BASELINE:
            self.junction_assignments = {jid: 'baseline' for jid in junctions}
        
        elif self.junction_assignment == JunctionAssignment.ALTERNATING:
            self.junction_assignments = {
                'J1': 'ai', 'J2': 'baseline', 
                'J3': 'ai', 'J4': 'baseline'
            }
        
        elif self.junction_assignment == JunctionAssignment.TRAFFIC_AWARE:
            # Will be updated based on traffic conditions
            self.junction_assignments = {jid: 'ai' for jid in junctions}
        
        print(f"📍 Mixed Control: Junction assignments initialized: {self.junction_assignments}")
    
    def decide_actions(self, vehicles: Dict, traffic_lights: Dict, metrics: Dict) -> Dict:
        """
        Main decision method using mixed control strategies
        """
        actions = {}
        
        # Handle emergency override first
        emergency_actions = self._check_emergency_override(vehicles, traffic_lights)
        if emergency_actions:
            return emergency_actions
        
        # Apply selected control mode
        if self.control_mode == ControlMode.AI_ONLY:
            actions = self._ai_only_control(vehicles, traffic_lights, metrics)
        
        elif self.control_mode == ControlMode.BASELINE_ONLY:
            actions = self._baseline_only_control(vehicles, traffic_lights, metrics)
        
        elif self.control_mode == ControlMode.HYBRID:
            actions = self._hybrid_control(vehicles, traffic_lights, metrics)
        
        elif self.control_mode == ControlMode.ADAPTIVE_SWITCHING:
            actions = self._adaptive_switching_control(vehicles, traffic_lights, metrics)
        
        elif self.control_mode == ControlMode.ENSEMBLE:
            actions = self._ensemble_control(vehicles, traffic_lights, metrics)
        
        # Apply coordination if available
        coordinated_actions = self._apply_coordination(vehicles, traffic_lights, metrics, actions)
        if coordinated_actions:
            actions.update(coordinated_actions)
        
        # Update performance metrics
        self._update_performance_metrics(actions, metrics)
        
        return actions
    
    def _ai_only_control(self, vehicles: Dict, traffic_lights: Dict, metrics: Dict) -> Dict:
        """Use only AI controller for all junctions"""
        if self.controllers['ai']:
            return self.controllers['ai'].decide_actions(vehicles, traffic_lights, metrics)
        else:
            print("⚠️ Mixed Control: AI controller not available, falling back to baseline")
            return self._baseline_only_control(vehicles, traffic_lights, metrics)
    
    def _baseline_only_control(self, vehicles: Dict, traffic_lights: Dict, metrics: Dict) -> Dict:
        """Use only baseline controller for all junctions"""
        if self.controllers['baseline']:
            actions = {}
            for junction_id, light_data in traffic_lights.items():
                action = self.controllers['baseline'].update_junction(
                    junction_id, vehicles, light_data.get('state', 'G')
                )
                actions[junction_id] = action
            return actions
        else:
            print("❌ Mixed Control: No controllers available")
            return {}
    
    def _hybrid_control(self, vehicles: Dict, traffic_lights: Dict, metrics: Dict) -> Dict:
        """Use hybrid control based on junction assignments"""
        actions = {}
        
        for junction_id in traffic_lights.keys():
            controller_type = self.junction_assignments.get(junction_id, 'ai')
            
            if controller_type == 'ai' and self.controllers['ai']:
                # Get AI action for this junction
                ai_actions = self.controllers['ai'].decide_actions(vehicles, traffic_lights, metrics)
                if junction_id in ai_actions:
                    actions[junction_id] = ai_actions[junction_id]
            
            elif controller_type == 'baseline' and self.controllers['baseline']:
                # Get baseline action for this junction
                current_state = traffic_lights.get(junction_id, {}).get('state', 'G')
                action = self.controllers['baseline'].update_junction(junction_id, vehicles, current_state)
                actions[junction_id] = action
        
        return actions
    
    def _adaptive_switching_control(self, vehicles: Dict, traffic_lights: Dict, metrics: Dict) -> Dict:
        """Adaptively switch between AI and baseline based on performance"""
        # Analyze current performance
        performance_score = self._calculate_performance_score(metrics)
        
        # Switch strategy if performance is poor
        if performance_score < 0.6:  # Poor performance threshold
            self._switch_control_strategy(vehicles, metrics)
        
        # Use hybrid control with current assignments
        return self._hybrid_control(vehicles, traffic_lights, metrics)
    
    def _ensemble_control(self, vehicles: Dict, traffic_lights: Dict, metrics: Dict) -> Dict:
        """Use ensemble of both controllers and vote for best action"""
        actions = {}
        
        for junction_id in traffic_lights.keys():
            ai_action = None
            baseline_action = None
            
            # Get AI action
            if self.controllers['ai']:
                ai_actions = self.controllers['ai'].decide_actions(vehicles, traffic_lights, metrics)
                ai_action = ai_actions.get(junction_id)
            
            # Get baseline action
            if self.controllers['baseline']:
                current_state = traffic_lights.get(junction_id, {}).get('state', 'G')
                baseline_action = self.controllers['baseline'].update_junction(
                    junction_id, vehicles, current_state
                )
            
            # Vote for best action
            best_action = self._vote_best_action(junction_id, ai_action, baseline_action, vehicles, metrics)
            if best_action:
                actions[junction_id] = best_action
        
        return actions
    
    def _vote_best_action(self, junction_id: str, ai_action: Dict, baseline_action: Dict, 
                         vehicles: Dict, metrics: Dict) -> Dict:
        """Vote for the best action between AI and baseline"""
        if not ai_action and not baseline_action:
            return None
        
        if not ai_action:
            return baseline_action
        if not baseline_action:
            return ai_action
        
        # Score each action
        ai_score = self._score_action(ai_action, junction_id, vehicles, metrics)
        baseline_score = self._score_action(baseline_action, junction_id, vehicles, metrics)
        
        # Choose action with higher score
        if ai_score >= baseline_score:
            ai_action['strategy'] = f"ensemble_ai_{ai_score:.2f}"
            return ai_action
        else:
            baseline_action['strategy'] = f"ensemble_baseline_{baseline_score:.2f}"
            return baseline_action
    
    def _score_action(self, action: Dict, junction_id: str, vehicles: Dict, metrics: Dict) -> float:
        """Score an action based on expected performance"""
        score = 0.0
        
        # Score based on action type and current conditions
        action_state = action.get('state', 'G')
        
        # Get traffic conditions around junction
        traffic_density = self._calculate_junction_density(junction_id, vehicles)
        emergency_presence = self._check_emergency_presence(junction_id, vehicles)
        
        # Scoring rules
        if emergency_presence and action_state == 'G':
            score += 0.8  # High score for green during emergency
        
        if traffic_density > 0.7 and action_state == 'G':
            score += 0.6  # Good for high density
        
        if traffic_density < 0.3 and action_state == 'R':
            score += 0.4  # Can switch to red for low density
        
        # Consider phase duration
        phase_duration = action.get('duration', 10.0)
        if 10.0 <= phase_duration <= 20.0:
            score += 0.2  # Reasonable duration
        
        return min(score, 1.0)
    
    def _check_emergency_override(self, vehicles: Dict, traffic_lights: Dict) -> Dict:
        """Check for emergency vehicles and override control"""
        try:
            from agents.emergency_module import emergency_manager
            emergency_actions = emergency_manager.update(vehicles, traffic_lights)
            if emergency_actions:
                print("🚨 Mixed Control: Emergency override activated")
                # Update strategy to emergency mode
                self.control_mode = ControlMode.EMERGENCY_OVERRIDE
                return emergency_actions
        except ImportError:
            pass
        
        # Reset emergency mode if no emergencies
        if self.control_mode == ControlMode.EMERGENCY_OVERRIDE:
            self.control_mode = ControlMode.HYBRID
        
        return {}
    
    def _apply_coordination(self, vehicles: Dict, traffic_lights: Dict, 
                          metrics: Dict, current_actions: Dict) -> Dict:
        """Apply coordination to current actions"""
        if not self.controllers['coordinator']:
            return {}
        
        try:
            emergency_vehicles = {}
            if hasattr(self.controllers.get('ai'), 'emergency_manager'):
                emergency_vehicles = self.controllers['ai'].emergency_manager.detector.emergency_vehicles
            
            coordination_advice = self.controllers['coordinator'].coordinate_actions(
                vehicles, traffic_lights, metrics, emergency_vehicles
            )
            
            # Apply coordination advice (simplified - would be more sophisticated)
            coordinated_actions = {}
            for junction_id, action in current_actions.items():
                coordinated_action = action.copy()
                
                # Modify action based on coordination advice
                junction_advice = coordination_advice.get('junction_advice', {}).get(junction_id)
                if junction_advice and junction_advice.get('priority') == 'high':
                    coordinated_action['duration'] = min(coordinated_action.get('duration', 10.0) * 1.5, 30.0)
                    coordinated_action['reason'] += " (Coordinated priority)"
                
                coordinated_actions[junction_id] = coordinated_action
            
            return coordinated_actions
        
        except Exception as e:
            print(f"❌ Mixed Control: Coordination failed - {e}")
            return current_actions
    
    def _switch_control_strategy(self, vehicles: Dict, metrics: Dict):
        """Switch control strategy based on performance"""
        current_strategy = self.control_mode
        current_assignments = self.junction_assignment
        
        # Analyze which junctions need different control
        junction_performance = self._analyze_junction_performance(vehicles, metrics)
        
        # Switch to better performing controller for problematic junctions
        for junction_id, performance in junction_performance.items():
            if performance < 0.5:  # Poor performance
                current_controller = self.junction_assignments.get(junction_id, 'ai')
                new_controller = 'baseline' if current_controller == 'ai' else 'ai'
                self.junction_assignments[junction_id] = new_controller
        
        # Record switch
        self.switching_history.append({
            'timestamp': time.time(),
            'from_strategy': current_strategy.value,
            'from_assignments': current_assignments.value,
            'to_assignments': self.junction_assignment.value,
            'reason': 'Poor performance detected'
        })
        
        print(f"🔄 Mixed Control: Strategy switched due to poor performance")
    
    def _calculate_performance_score(self, metrics: Dict) -> float:
        """Calculate overall performance score"""
        waiting_time = metrics.get('total_waiting_time', 0)
        vehicles_cleared = metrics.get('vehicles_cleared', 0)
        average_speed = metrics.get('average_speed', 0)
        
        # Normalize metrics to 0-1 scale (higher is better)
        waiting_score = 1.0 - min(waiting_time / 1000.0, 1.0)
        throughput_score = min(vehicles_cleared / 50.0, 1.0)
        speed_score = min(average_speed / 20.0, 1.0)
        
        # Weighted average
        performance_score = (waiting_score * 0.4 + throughput_score * 0.4 + speed_score * 0.2)
        return performance_score
    
    def _analyze_junction_performance(self, vehicles: Dict, metrics: Dict) -> Dict[str, float]:
        """Analyze performance at each junction"""
        junction_performance = {}
        
        for junction_id in ['J1', 'J2', 'J3', 'J4']:
            # Simplified performance calculation
            density = self._calculate_junction_density(junction_id, vehicles)
            congestion_level = min(density * 1.5, 1.0)  # Higher density = worse performance
            
            # Invert for performance score (higher is better)
            performance = 1.0 - congestion_level
            junction_performance[junction_id] = performance
        
        return junction_performance
    
    def _calculate_junction_density(self, junction_id: str, vehicles: Dict) -> float:
        """Calculate traffic density around a junction"""
        junction_positions = {
            'J1': [100, 100], 'J2': [300, 100],
            'J3': [100, 300], 'J4': [300, 300]
        }
        
        junction_pos = junction_positions.get(junction_id, [0, 0])
        vehicles_near = 0
        
        for vehicle in vehicles.values():
            vehicle_pos = [vehicle.get('x', 0), vehicle.get('y', 0)]
            distance = np.linalg.norm(np.array(vehicle_pos) - np.array(junction_pos))
            if distance < 60:
                vehicles_near += 1
        
        return min(vehicles_near / 8.0, 1.0)
    
    def _check_emergency_presence(self, junction_id: str, vehicles: Dict) -> bool:
        """Check if emergency vehicles are present near junction"""
        try:
            if hasattr(self.controllers.get('ai'), 'emergency_manager'):
                emergencies = self.controllers['ai'].emergency_manager.detector.emergency_vehicles
                for emergency in emergencies.values():
                    emergency_pos = getattr(emergency, 'position', [0, 0])
                    junction_pos = {
                        'J1': [100, 100], 'J2': [300, 100],
                        'J3': [100, 300], 'J4': [300, 300]
                    }.get(junction_id, [0, 0])
                    
                    distance = np.linalg.norm(np.array(emergency_pos) - np.array(junction_pos))
                    if distance < 100:
                        return True
        except:
            pass
        
        return False
    
    def _update_performance_metrics(self, actions: Dict, metrics: Dict):
        """Update performance metrics for analysis"""
        for junction_id, action in actions.items():
            if junction_id not in self.performance_metrics:
                self.performance_metrics[junction_id] = {
                    'total_actions': 0,
                    'ai_actions': 0,
                    'baseline_actions': 0,
                    'average_duration': 0,
                    'last_update': time.time()
                }
            
            junction_metrics = self.performance_metrics[junction_id]
            junction_metrics['total_actions'] += 1
            
            strategy = action.get('strategy', '')
            if 'ai' in strategy.lower():
                junction_metrics['ai_actions'] += 1
            elif 'baseline' in strategy.lower():
                junction_metrics['baseline_actions'] += 1
            
            # Update average duration
            current_duration = action.get('duration', 10.0)
            junction_metrics['average_duration'] = (
                0.8 * junction_metrics['average_duration'] + 0.2 * current_duration
            )
    
    def set_control_mode(self, mode: ControlMode):
        """Set the control mode"""
        self.control_mode = mode
        print(f"🔄 Mixed Control: Control mode set to {mode.value}")
    
    def set_junction_assignment(self, assignment: JunctionAssignment):
        """Set how controllers are assigned to junctions"""
        self.junction_assignment = assignment
        self._initialize_junction_assignments()
    
    def get_control_stats(self) -> Dict:
        """Get mixed control statistics"""
        total_actions = 0
        ai_actions = 0
        baseline_actions = 0
        
        for metrics in self.performance_metrics.values():
            total_actions += metrics['total_actions']
            ai_actions += metrics['ai_actions']
            baseline_actions += metrics['baseline_actions']
        
        return {
            'control_mode': self.control_mode.value,
            'junction_assignment': self.junction_assignment.value,
            'total_actions': total_actions,
            'ai_actions': ai_actions,
            'baseline_actions': baseline_actions,
            'ai_usage_rate': ai_actions / max(total_actions, 1),
            'baseline_usage_rate': baseline_actions / max(total_actions, 1),
            'junction_assignments': self.junction_assignments,
            'switching_history_count': len(self.switching_history),
            'controllers_available': {
                'ai': self.controllers['ai'] is not None,
                'baseline': self.controllers['baseline'] is not None,
                'coordinator': self.controllers['coordinator'] is not None
            }
        }


# Global mixed control instance
mixed_control_manager = MixedControlManager()

def get_mixed_controller() -> MixedControlManager:
    """Get the global mixed control manager"""
    return mixed_control_manager

if __name__ == "__main__":
    # Test the mixed control system
    controller = MixedControlManager()
    
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
    
    print("🔄 Testing Mixed Control System:")
    print("=" * 50)
    
    # Test different control modes
    modes = list(ControlMode)
    
    for mode in modes:
        controller.set_control_mode(mode)
        actions = controller.decide_actions(test_vehicles, test_lights, test_metrics)
        
        print(f"\n📋 Control Mode: {mode.value}")
        print(f"   Actions Generated: {len(actions)}")
        
        for junction, action in actions.items():
            print(f"   {junction}: {action['state']} - {action.get('strategy', 'unknown')}")
    
    # Show statistics
    stats = controller.get_control_stats()
    print(f"\n📊 Mixed Control Statistics:")
    print(f"   Current Mode: {stats['control_mode']}")
    print(f"   AI Usage: {stats['ai_usage_rate']:.1%}")
    print(f"   Baseline Usage: {stats['baseline_usage_rate']:.1%}")
    print(f"   Controllers Available: {stats['controllers_available']}")