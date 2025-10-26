import numpy as np
from typing import Dict, List, Tuple, Any
import time
import logging
from enum import Enum

class EmergencyType(Enum):
    AMBULANCE = "ambulance"
    FIRE_TRUCK = "fire_truck"
    POLICE = "police"
    OTHER = "other"

class EmergencyVehicle:
    """Represents an emergency vehicle with priority characteristics"""
    
    def __init__(self, vehicle_id: str, emergency_type: EmergencyType, priority: int = 10):
        self.id = vehicle_id
        self.type = emergency_type
        self.priority = priority
        self.position = [0, 0]
        self.speed = 0.0
        self.direction = "right"
        self.siren_active = True
        self.flashing_lights = True
        self.v2i_equipped = True
        self.detection_confidence = 1.0
        self.route = []
        self.eta_to_junctions = {}
        
    def update_position(self, x: float, y: float, speed: float, direction: str):
        self.position = [x, y]
        self.speed = speed
        self.direction = direction
    
    def get_priority_level(self) -> int:
        """Get priority level (1-10) based on emergency type"""
        priority_map = {
            EmergencyType.AMBULANCE: 10,
            EmergencyType.FIRE_TRUCK: 9,
            EmergencyType.POLICE: 8,
            EmergencyType.OTHER: 7
        }
        return priority_map.get(self.type, 7)

class EmergencyDetector:
    """Detects and tracks emergency vehicles using multiple methods"""
    
    def __init__(self, detection_range: float = 100.0):
        self.detection_range = detection_range
        self.emergency_vehicles: Dict[str, EmergencyVehicle] = {}
        self.detection_history = {}
        
        # Detection method reliability weights
        self.detection_weights = {
            'v2i': 0.9,      # Vehicle-to-Infrastructure
            'acoustic': 0.7,  # Siren detection
            'optical': 0.6,   # Visual detection
            'predefined': 1.0 # Known emergency vehicles
        }
    
    def detect_emergencies(self, vehicles: Dict, traffic_lights: Dict) -> Dict[str, EmergencyVehicle]:
        """Detect emergency vehicles from regular vehicle pool"""
        detected_emergencies = {}
        
        for vehicle_id, vehicle_data in vehicles.items():
            detection_score = 0.0
            detection_methods = []
            
            # Method 1: Check predefined emergency attributes
            if self._is_predefined_emergency(vehicle_data):
                detection_score += self.detection_weights['predefined']
                detection_methods.append('predefined')
                
                # Create emergency vehicle object
                emergency_type = EmergencyType(vehicle_data.get('emergency_type', 'other'))
                emergency_vehicle = EmergencyVehicle(vehicle_id, emergency_type)
                emergency_vehicle.update_position(
                    vehicle_data.get('x', 0),
                    vehicle_data.get('y', 0),
                    vehicle_data.get('speed', 0),
                    vehicle_data.get('direction', 'right')
                )
                emergency_vehicle.detection_confidence = detection_score
                
                detected_emergencies[vehicle_id] = emergency_vehicle
            
            # Method 2: V2I Communication detection
            elif self._detect_v2i_emergency(vehicle_data):
                detection_score += self.detection_weights['v2i']
                detection_methods.append('v2i')
            
            # Method 3: Acoustic detection
            acoustic_score = self._acoustic_detection(vehicle_data, traffic_lights)
            if acoustic_score > 0:
                detection_score += acoustic_score * self.detection_weights['acoustic']
                detection_methods.append('acoustic')
            
            # Method 4: Optical detection
            optical_score = self._optical_detection(vehicle_data)
            if optical_score > 0:
                detection_score += optical_score * self.detection_weights['optical']
                detection_methods.append('optical')
            
            # Create emergency vehicle if confidence is high enough
            if detection_score >= 0.7 and 'v2i' in detection_methods:
                emergency_vehicle = EmergencyVehicle(vehicle_id, EmergencyType.OTHER)
                emergency_vehicle.detection_confidence = detection_score
                detected_emergencies[vehicle_id] = emergency_vehicle
        
        self.emergency_vehicles.update(detected_emergencies)
        return detected_emergencies
    
    def _is_predefined_emergency(self, vehicle_data: Dict) -> bool:
        """Check if vehicle is predefined as emergency"""
        return (vehicle_data.get('is_emergency', False) or 
                vehicle_data.get('type') == 'emergency' or 
                vehicle_data.get('emergency_priority', 0) > 0)
    
    def _detect_v2i_emergency(self, vehicle_data: Dict) -> bool:
        """Detect emergency via V2I communication"""
        return (vehicle_data.get('v2i_equipped', False) and 
                vehicle_data.get('emergency_broadcast', False))
    
    def _acoustic_detection(self, vehicle_data: Dict, traffic_lights: Dict) -> float:
        """Simulate acoustic siren detection"""
        if not vehicle_data.get('siren_active', False):
            return 0.0
        
        # Calculate signal strength based on distance to traffic lights
        vehicle_pos = [vehicle_data.get('x', 0), vehicle_data.get('y', 0)]
        min_distance = float('inf')
        
        for tl_data in traffic_lights.values():
            tl_pos = tl_data.get('position', [0, 0])
            distance = np.linalg.norm(np.array(vehicle_pos) - np.array(tl_pos))
            min_distance = min(min_distance, distance)
        
        if min_distance <= self.detection_range:
            return 1.0 - (min_distance / self.detection_range)
        
        return 0.0
    
    def _optical_detection(self, vehicle_data: Dict) -> float:
        """Simulate optical detection of emergency features"""
        score = 0.0
        if vehicle_data.get('flashing_lights', False):
            score += 0.6
        if vehicle_data.get('color') in ['red', '#e74c3c', '#ff0000']:
            score += 0.3
        if vehicle_data.get('special_marking', False):
            score += 0.2
        
        return min(1.0, score)
    
    def get_approaching_emergencies(self, junctions: Dict) -> Dict[str, List[EmergencyVehicle]]:
        """Get emergencies approaching each junction"""
        approaching = {}
        
        for junction_id, junction_data in junctions.items():
            junction_pos = junction_data.get('position', [0, 0])
            approaching[junction_id] = []
            
            for emergency in self.emergency_vehicles.values():
                if self._is_approaching_junction(emergency, junction_pos):
                    distance = np.linalg.norm(
                        np.array(emergency.position) - np.array(junction_pos)
                    )
                    approaching[junction_id].append((emergency, distance))
            
            # Sort by distance
            approaching[junction_id].sort(key=lambda x: x[1])
        
        return approaching
    
    def _is_approaching_junction(self, emergency: EmergencyVehicle, junction_pos: List) -> bool:
        """Check if emergency vehicle is approaching the junction"""
        pos_x, pos_y = emergency.position
        j_x, j_y = junction_pos
        
        if emergency.direction == 'right' and pos_x < j_x and abs(pos_y - j_y) < 30:
            return True
        elif emergency.direction == 'left' and pos_x > j_x and abs(pos_y - j_y) < 30:
            return True
        elif emergency.direction == 'down' and pos_y < j_y and abs(pos_x - j_x) < 30:
            return True
        elif emergency.direction == 'up' and pos_y > j_y and abs(pos_x - j_x) < 30:
            return True
        
        return False

class EmergencyModule:
    """Manages traffic light priorities for emergency vehicles"""
    
    def __init__(self):
        self.detector = EmergencyDetector()
        self.green_wave_routes = {}
        self.emergency_log = []
    
    def update(self, vehicles: Dict, traffic_lights: Dict) -> Dict:
        """Update emergency vehicle handling and return control commands"""
        # Detect emergency vehicles
        emergencies = self.detector.detect_emergencies(vehicles, traffic_lights)
        
        if not emergencies:
            return {}
        
        # Get approaching emergencies for each junction
        approaching = self.detector.get_approaching_emergencies(traffic_lights)
        
        # Generate priority commands
        commands = self._generate_priority_commands(approaching, traffic_lights)
        
        # Log emergency activity
        self._log_emergency_activity(emergencies, commands)
        
        return commands
    
    def _generate_priority_commands(self, approaching: Dict, traffic_lights: Dict) -> Dict:
        """Generate traffic light commands for emergency priority"""
        commands = {}
        
        for junction_id, emergencies in approaching.items():
            if not emergencies:
                continue
            
            # Get highest priority emergency
            emergency, distance = emergencies[0]
            
            # Calculate optimal light state for emergency passage
            optimal_state = self._calculate_emergency_light_state(
                junction_id, emergency, traffic_lights
            )
            
            commands[junction_id] = {
                'state': optimal_state,
                'duration': 20,  # Extended green for emergency
                'priority': 'emergency',
                'emergency_id': emergency.id,
                'emergency_type': emergency.type.value,
                'confidence': emergency.detection_confidence,
                'reason': f'Emergency {emergency.type.value} approaching - ETA: {distance/emergency.speed:.1f}s'
            }
        
        return commands
    
    def _calculate_emergency_light_state(self, junction_id: str, emergency: EmergencyVehicle, 
                                       traffic_lights: Dict) -> str:
        """Calculate optimal light state for emergency vehicle passage"""
        # Simple implementation: set all directions to green for emergency
        # In a real system, this would be more sophisticated
        return 'G'  # Green in all directions
    
    def _log_emergency_activity(self, emergencies: Dict, commands: Dict):
        """Log emergency detection and response"""
        log_entry = {
            'timestamp': time.time(),
            'emergencies_detected': len(emergencies),
            'commands_issued': len(commands),
            'vehicle_ids': list(emergencies.keys())
        }
        self.emergency_log.append(log_entry)
        
        # Keep only recent logs
        if len(self.emergency_log) > 100:
            self.emergency_log.pop(0)
    
    def get_emergency_stats(self) -> Dict:
        """Get emergency handling statistics"""
        if not self.emergency_log:
            return {}
        
        return {
            'active_emergencies': len(self.detector.emergency_vehicles),
            'total_handled': len(self.emergency_log),
            'recent_detections': len(self.emergency_log[-10:]) if self.emergency_log else 0
        }

# Global instance
emergency_manager = EmergencyModule()