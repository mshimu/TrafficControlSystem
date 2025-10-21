import numpy as np
from typing import Dict, List, Tuple, Any
import time
import logging

class EmergencyVehicleDetector:
    """
    Detects and tracks emergency vehicles in the simulation
    using multiple detection methods
    """
    
    def __init__(self, detection_range: float = 100.0):
        self.detection_range = detection_range
        self.emergency_vehicles = {}
        self.detection_history = {}
        self.logger = logging.getLogger(__name__)
        
        # Detection method weights (simulating different sensor reliabilities)
        self.detection_weights = {
            'acoustic': 0.3,    # Siren detection
            'optical': 0.4,     # Flashing lights
            'v2i': 0.8,         # Vehicle-to-Infrastructure communication
            'predefined': 1.0   # Known emergency vehicles
        }
        
    def detect_emergency_vehicles(self, vehicles: Dict, traffic_lights: Dict) -> Dict:
        """
        Main detection method that uses multiple approaches to identify emergency vehicles
        """
        detected_emergencies = {}
        
        for vehicle_id, vehicle in vehicles.items():
            detection_score = 0
            detection_methods = []
            
            # Method 1: Pre-defined emergency vehicles
            if self._check_predefined_emergency(vehicle):
                detection_score += self.detection_weights['predefined']
                detection_methods.append('predefined')
            
            # Method 2: Acoustic detection (siren)
            acoustic_score = self._acoustic_detection(vehicle, traffic_lights)
            if acoustic_score > 0:
                detection_score += acoustic_score * self.detection_weights['acoustic']
                detection_methods.append('acoustic')
            
            # Method 3: Optical detection (flashing lights)
            optical_score = self._optical_detection(vehicle)
            if optical_score > 0:
                detection_score += optical_score * self.detection_weights['optical']
                detection_methods.append('optical')
            
            # Method 4: V2I communication (most reliable)
            v2i_score = self._v2i_communication(vehicle)
            if v2i_score > 0:
                detection_score += v2i_score * self.detection_weights['v2i']
                detection_methods.append('v2i')
            
            # If detection score exceeds threshold, mark as emergency
            if detection_score >= 0.6:  # 60% confidence threshold
                detected_emergencies[vehicle_id] = {
                    'vehicle': vehicle,
                    'confidence': detection_score,
                    'methods': detection_methods,
                    'first_detected': self.detection_history.get(vehicle_id, time.time()),
                    'priority_level': self._calculate_priority_level(vehicle, detection_score)
                }
                
                # Update detection history
                self.detection_history[vehicle_id] = time.time()
        
        self.emergency_vehicles = detected_emergencies
        return detected_emergencies
    
    def _check_predefined_emergency(self, vehicle) -> bool:
        """Check if vehicle is predefined as emergency"""
        return getattr(vehicle, 'is_emergency', False) or \
               getattr(vehicle, 'type', '') == 'emergency' or \
               getattr(vehicle, 'emergency_priority', 0) > 0
    
    def _acoustic_detection(self, vehicle, traffic_lights: Dict) -> float:
        """
        Simulate acoustic detection of sirens
        Returns confidence score between 0-1
        """
        # Check if vehicle has siren active
        siren_active = getattr(vehicle, 'siren_active', False)
        if not siren_active:
            return 0.0
        
        # Calculate distance to nearest traffic light for signal strength
        vehicle_pos = getattr(vehicle, 'position', [0, 0])
        min_distance = float('inf')
        
        for tl_id, tl_data in traffic_lights.items():
            tl_pos = tl_data.get('position', [0, 0])
            distance = np.sqrt((vehicle_pos[0]-tl_pos[0])**2 + (vehicle_pos[1]-tl_pos[1])**2)
            min_distance = min(min_distance, distance)
        
        # Signal strength decreases with distance
        if min_distance <= self.detection_range:
            signal_strength = 1.0 - (min_distance / self.detection_range)
            return max(0.0, signal_strength)
        
        return 0.0
    
    def _optical_detection(self, vehicle) -> float:
        """
        Simulate optical detection of flashing lights
        Returns confidence score between 0-1
        """
        # Check for emergency vehicle characteristics
        has_flashing_lights = getattr(vehicle, 'flashing_lights', False)
        emergency_color = getattr(vehicle, 'color', '') in ['red', '#e74c3c', '#ff0000']
        special_marking = getattr(vehicle, 'special_marking', False)
        
        score = 0.0
        if has_flashing_lights:
            score += 0.6
        if emergency_color:
            score += 0.3
        if special_marking:
            score += 0.2
        
        return min(1.0, score)
    
    def _v2i_communication(self, vehicle) -> float:
        """
        Simulate Vehicle-to-Infrastructure communication
        Most reliable method - returns high confidence if available
        """
        v2i_equipped = getattr(vehicle, 'v2i_equipped', False)
        emergency_broadcast = getattr(vehicle, 'emergency_broadcast', False)
        
        if v2i_equipped and emergency_broadcast:
            # V2I provides high-confidence detection
            emergency_type = getattr(vehicle, 'emergency_type', 'unknown')
            priority_map = {
                'ambulance': 1.0,
                'fire_truck': 1.0,
                'police': 0.9,
                'other': 0.7
            }
            return priority_map.get(emergency_type, 0.8)
        
        return 0.0
    
    def _calculate_priority_level(self, vehicle, detection_score: float) -> int:
        """Calculate priority level for emergency vehicle (1-10)"""
        base_priority = int(detection_score * 5)  # 1-5 based on confidence
        
        # Adjust based on emergency type
        emergency_type = getattr(vehicle, 'emergency_type', 'unknown')
        type_priority = {
            'ambulance': 3,
            'fire_truck': 3,
            'police': 2,
            'other': 1
        }
        
        total_priority = base_priority + type_priority.get(emergency_type, 1)
        return min(10, total_priority)
    
    def get_emergency_vehicle_approaching_junctions(self, traffic_lights: Dict) -> Dict:
        """
        Identify which junctions have emergency vehicles approaching
        """
        approaching_junctions = {}
        
        for emergency_id, emergency_data in self.emergency_vehicles.items():
            vehicle = emergency_data['vehicle']
            vehicle_pos = getattr(vehicle, 'position', [0, 0])
            vehicle_direction = getattr(vehicle, 'direction', 'right')
            
            for junction_id, junction_data in traffic_lights.items():
                junction_pos = junction_data.get('position', [0, 0])
                
                # Calculate if vehicle is approaching this junction
                if self._is_approaching_junction(vehicle_pos, vehicle_direction, junction_pos):
                    distance = np.sqrt((vehicle_pos[0]-junction_pos[0])**2 + 
                                     (vehicle_pos[1]-junction_pos[1])**2)
                    
                    if junction_id not in approaching_junctions:
                        approaching_junctions[junction_id] = []
                    
                    approaching_junctions[junction_id].append({
                        'vehicle_id': emergency_id,
                        'distance': distance,
                        'priority': emergency_data['priority_level'],
                        'confidence': emergency_data['confidence'],
                        'eta': self._calculate_eta(distance, vehicle)
                    })
        
        # Sort by distance and priority for each junction
        for junction_id, emergencies in approaching_junctions.items():
            emergencies.sort(key=lambda x: (x['distance'], -x['priority']))
        
        return approaching_junctions
    
    def _is_approaching_junction(self, vehicle_pos: List, direction: str, junction_pos: List) -> bool:
        """Check if vehicle is moving toward the junction"""
        pos_x, pos_y = vehicle_pos
        j_x, j_y = junction_pos
        
        # Simple directional check
        if direction == 'right' and pos_x < j_x and abs(pos_y - j_y) < 50:
            return True
        elif direction == 'left' and pos_x > j_x and abs(pos_y - j_y) < 50:
            return True
        elif direction == 'down' and pos_y < j_y and abs(pos_x - j_x) < 50:
            return True
        elif direction == 'up' and pos_y > j_y and abs(pos_x - j_x) < 50:
            return True
        
        return False
    
    def _calculate_eta(self, distance: float, vehicle) -> float:
        """Calculate estimated time of arrival at junction"""
        speed = getattr(vehicle, 'speed', 5.0)  # m/s
        if speed > 0:
            return distance / speed
        return float('inf')
    
    def clear_expired_detections(self, expiry_time: float = 30.0):
        """Clear old detections that are no longer relevant"""
        current_time = time.time()
        expired_vehicles = []
        
        for vehicle_id, first_detected in self.detection_history.items():
            if current_time - first_detected > expiry_time:
                expired_vehicles.append(vehicle_id)
        
        for vehicle_id in expired_vehicles:
            if vehicle_id in self.emergency_vehicles:
                del self.emergency_vehicles[vehicle_id]
            if vehicle_id in self.detection_history:
                del self.detection_history[vehicle_id]


class EmergencyTrafficManager:
    """
    Manages traffic light priorities for emergency vehicles
    """
    
    def __init__(self):
        self.detector = EmergencyVehicleDetector()
        self.green_wave_routes = {}
        self.emergency_log = []
        
    def update(self, vehicles: Dict, traffic_lights: Dict) -> Dict:
        """
        Main update method - detects emergencies and returns traffic light commands
        """
        # Detect emergency vehicles
        emergencies = self.detector.detect_emergency_vehicles(vehicles, traffic_lights)
        
        # Get approaching junctions
        approaching = self.detector.get_emergency_vehicle_approaching_junctions(traffic_lights)
        
        # Generate traffic light commands
        commands = self._generate_emergency_commands(approaching, traffic_lights)
        
        # Log activity
        if emergencies:
            self._log_emergency_activity(emergencies, commands)
        
        return commands
    
    def _generate_emergency_commands(self, approaching_junctions: Dict, traffic_lights: Dict) -> Dict:
        """
        Generate traffic light commands to prioritize emergency vehicles
        """
        commands = {}
        
        for junction_id, emergencies in approaching_junctions.items():
            if not emergencies:
                continue
                
            # Get the highest priority emergency approaching this junction
            highest_priority = emergencies[0]
            
            # Determine which direction should get green light
            optimal_state = self._calculate_optimal_light_state(
                junction_id, highest_priority, traffic_lights
            )
            
            commands[junction_id] = {
                'state': optimal_state,
                'duration': 15,  # Extended green time for emergency
                'priority': 'emergency',
                'emergency_id': highest_priority['vehicle_id'],
                'reason': f'Emergency vehicle approaching - ETA: {highest_priority["eta"]:.1f}s'
            }
        
        return commands
    
    def _calculate_optimal_light_state(self, junction_id: str, emergency_data: Dict, traffic_lights: Dict) -> str:
        """
        Calculate the optimal traffic light state for emergency vehicle passage
        """
        # This is a simplified implementation
        # In a real system, this would consider the emergency vehicle's approach direction
        
        junction_data = traffic_lights.get(junction_id, {})
        current_state = junction_data.get('state', 'G')
        
        # For now, set all directions to green for emergency passage
        # In a more sophisticated system, we would only set the relevant direction
        return 'G'  # Green in all directions for emergency
        
    def _log_emergency_activity(self, emergencies: Dict, commands: Dict):
        """Log emergency vehicle detection and response"""
        log_entry = {
            'timestamp': time.time(),
            'emergencies_detected': len(emergencies),
            'commands_issued': len(commands),
            'vehicle_ids': list(emergencies.keys()),
            'confidence_scores': [e['confidence'] for e in emergencies.values()]
        }
        
        self.emergency_log.append(log_entry)
        
        # Keep only last 100 entries
        if len(self.emergency_log) > 100:
            self.emergency_log.pop(0)
    
    def get_emergency_stats(self) -> Dict:
        """Get statistics about emergency vehicle handling"""
        if not self.emergency_log:
            return {}
        
        recent_logs = self.emergency_log[-10:]  # Last 10 entries
        
        total_emergencies = sum(entry['emergencies_detected'] for entry in recent_logs)
        avg_confidence = np.mean([np.mean(entry['confidence_scores']) 
                                for entry in recent_logs if entry['confidence_scores']])
        
        return {
            'total_emergencies_handled': total_emergencies,
            'average_confidence': avg_confidence,
            'active_emergencies': len(self.detector.emergency_vehicles),
            'detection_methods_used': self._get_detection_methods_summary()
        }
    
    def _get_detection_methods_summary(self) -> Dict:
        """Get summary of detection methods used"""
        method_count = {}
        for emergency_data in self.detector.emergency_vehicles.values():
            for method in emergency_data['methods']:
                method_count[method] = method_count.get(method, 0) + 1
        
        return method_count


# Singleton instance for easy access
emergency_manager = EmergencyTrafficManager()