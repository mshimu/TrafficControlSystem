import socketio
import eventlet
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
from collections import deque

# Create SocketIO server
sio = socketio.Server(cors_allowed_origins="*", async_mode='eventlet')
app = socketio.WSGIApp(sio)

class TrafficSocketHandler:
    """Handles real-time WebSocket communication for traffic simulation"""
    
    def __init__(self):
        self.connected_clients = set()
        self.simulation_data = {}
        self.metrics_history = deque(maxlen=100)  # Keep last 100 data points
        self.emergency_alerts = deque(maxlen=20)   # Keep last 20 alerts
        self.client_info = {}  # Store client-specific information
        
        # Rate limiting
        self.last_broadcast_time = 0
        self.broadcast_interval = 0.1  # 10 FPS max for simulation updates
        
        # Initialize default simulation data
        self._initialize_default_data()
    
    def _initialize_default_data(self):
        """Initialize default simulation data structure"""
        self.simulation_data = {
            'vehicles': {},
            'traffic_lights': {},
            'metrics': {
                'total_waiting_time': 0,
                'vehicles_cleared': 0,
                'emergency_cleared': 0,
                'average_speed': 0,
                'active_vehicles': 0
            },
            'emergency_vehicles': {},
            'controller_info': {
                'type': 'ai',
                'strategy': 'mappo',
                'performance': 0.0
            },
            'timestamp': time.time()
        }
    
    def update_simulation_data(self, vehicles: Dict, traffic_lights: Dict, 
                             metrics: Dict, emergency_vehicles: Dict = None,
                             controller_info: Dict = None):
        """Update simulation data and broadcast to clients"""
        current_time = time.time()
        
        # Rate limiting - don't broadcast too frequently
        if current_time - self.last_broadcast_time < self.broadcast_interval:
            return
        
        self.last_broadcast_time = current_time
        
        # Update simulation data
        self.simulation_data.update({
            'vehicles': vehicles,
            'traffic_lights': traffic_lights,
            'metrics': metrics,
            'emergency_vehicles': emergency_vehicles or {},
            'controller_info': controller_info or self.simulation_data['controller_info'],
            'timestamp': current_time
        })
        
        # Store metrics for history
        self.metrics_history.append({
            'timestamp': current_time,
            'waiting_time': metrics.get('total_waiting_time', 0),
            'vehicles_cleared': metrics.get('vehicles_cleared', 0),
            'average_speed': metrics.get('average_speed', 0),
            'active_vehicles': metrics.get('active_vehicles', 0)
        })
        
        # Broadcast to all connected clients
        self.broadcast_simulation_update()
    
    def broadcast_simulation_update(self):
        """Broadcast simulation update to all connected clients"""
        if not self.connected_clients:
            return
        
        try:
            sio.emit('simulation_update', {
                'data': self.simulation_data,
                'metrics_history': list(self.metrics_history),
                'timestamp': time.time()
            })
        except Exception as e:
            print(f"❌ Error broadcasting simulation update: {e}")
    
    def send_emergency_alert(self, emergency_vehicle: Dict, action: str, reason: str):
        """Send emergency vehicle alert to clients"""
        alert_data = {
            'vehicle_id': emergency_vehicle.get('id', 'unknown'),
            'vehicle_type': emergency_vehicle.get('type', 'emergency'),
            'position': emergency_vehicle.get('position', [0, 0]),
            'action': action,
            'reason': reason,
            'timestamp': time.time(),
            'priority': emergency_vehicle.get('priority', 1)
        }
        
        self.emergency_alerts.append(alert_data)
        
        # Broadcast emergency alert
        sio.emit('emergency_alert', alert_data)
        print(f"🚨 Emergency alert sent: {alert_data}")
    
    def send_controller_change(self, controller_type: str, strategy: str, reason: str):
        """Notify clients about controller change"""
        change_data = {
            'controller_type': controller_type,
            'strategy': strategy,
            'reason': reason,
            'timestamp': time.time()
        }
        
        sio.emit('controller_changed', change_data)
        print(f"🔄 Controller change: {controller_type} - {strategy}")
    
    def send_performance_metrics(self, detailed_metrics: Dict):
        """Send detailed performance metrics"""
        sio.emit('performance_metrics', {
            'metrics': detailed_metrics,
            'timestamp': time.time()
        })
    
    def get_client_count(self) -> int:
        """Get number of connected clients"""
        return len(self.connected_clients)

# Global socket handler instance
socket_handler = TrafficSocketHandler()

# SocketIO event handlers
@sio.event
def connect(sid, environ):
    """Handle client connection"""
    socket_handler.connected_clients.add(sid)
    socket_handler.client_info[sid] = {
        'connect_time': time.time(),
        'user_agent': environ.get('HTTP_USER_AGENT', 'Unknown'),
        'ip_address': environ.get('REMOTE_ADDR', 'Unknown')
    }
    
    print(f"✅ Client connected: {sid}")
    print(f"   Total clients: {len(socket_handler.connected_clients)}")
    
    # Send current simulation state to new client
    sio.emit('simulation_update', {
        'data': socket_handler.simulation_data,
        'metrics_history': list(socket_handler.metrics_history),
        'timestamp': time.time()
    }, room=sid)
    
    # Send connection acknowledgement
    sio.emit('connection_established', {
        'message': 'Connected to AI Traffic Control System',
        'client_id': sid,
        'server_time': time.time()
    }, room=sid)

@sio.event
def disconnect(sid):
    """Handle client disconnection"""
    socket_handler.connected_clients.discard(sid)
    if sid in socket_handler.client_info:
        del socket_handler.client_info[sid]
    
    print(f"❌ Client disconnected: {sid}")
    print(f"   Remaining clients: {len(socket_handler.connected_clients)}")

@sio.event
def start_simulation(sid, data):
    """Handle simulation start request from client"""
    print(f"🎬 Simulation start requested by {sid}")
    
    try:
        # Validate input data
        use_ai = data.get('use_ai', True)
        emergency_vehicles = data.get('emergency_vehicles', True)
        vehicle_density = data.get('vehicle_density', 5)
        simulation_speed = data.get('simulation_speed', 5)
        
        # Import simulation manager (avoid circular imports)
        from web_app.app import get_simulation_manager
        
        simulation_manager = get_simulation_manager()
        
        # Start simulation
        success = simulation_manager.start_simulation(
            use_ai=use_ai,
            emergency_vehicles=emergency_vehicles,
            vehicle_density=vehicle_density,
            simulation_speed=simulation_speed
        )
        
        if success:
            # Notify all clients
            sio.emit('simulation_started', {
                'message': 'Simulation started successfully',
                'config': data,
                'timestamp': time.time()
            })
            
            # Send success to requesting client
            sio.emit('command_success', {
                'command': 'start_simulation',
                'message': 'Simulation started successfully',
                'timestamp': time.time()
            }, room=sid)
            
        else:
            # Send error to requesting client
            sio.emit('command_error', {
                'command': 'start_simulation',
                'message': 'Failed to start simulation',
                'timestamp': time.time()
            }, room=sid)
            
    except Exception as e:
        print(f"❌ Error starting simulation: {e}")
        sio.emit('command_error', {
            'command': 'start_simulation',
            'message': f'Error: {str(e)}',
            'timestamp': time.time()
        }, room=sid)

@sio.event
def stop_simulation(sid, data=None):
    """Handle simulation stop request from client"""
    print(f"⏹️ Simulation stop requested by {sid}")
    
    try:
        from web_app.app import get_simulation_manager
        
        simulation_manager = get_simulation_manager()
        simulation_manager.stop_simulation()
        
        # Notify all clients
        sio.emit('simulation_stopped', {
            'message': 'Simulation stopped',
            'timestamp': time.time()
        })
        
        # Send success to requesting client
        sio.emit('command_success', {
            'command': 'stop_simulation',
            'message': 'Simulation stopped successfully',
            'timestamp': time.time()
        }, room=sid)
        
    except Exception as e:
        print(f"❌ Error stopping simulation: {e}")
        sio.emit('command_error', {
            'command': 'stop_simulation',
            'message': f'Error: {str(e)}',
            'timestamp': time.time()
        }, room=sid)

@sio.event
def reset_simulation(sid, data=None):
    """Handle simulation reset request from client"""
    print(f"🔄 Simulation reset requested by {sid}")
    
    try:
        from web_app.app import get_simulation_manager
        
        simulation_manager = get_simulation_manager()
        simulation_manager.reset_simulation()
        
        # Reset socket handler data
        socket_handler._initialize_default_data()
        
        # Notify all clients
        sio.emit('simulation_reset', {
            'message': 'Simulation reset',
            'timestamp': time.time()
        })
        
        # Send success to requesting client
        sio.emit('command_success', {
            'command': 'reset_simulation',
            'message': 'Simulation reset successfully',
            'timestamp': time.time()
        }, room=sid)
        
    except Exception as e:
        print(f"❌ Error resetting simulation: {e}")
        sio.emit('command_error', {
            'command': 'reset_simulation',
            'message': f'Error: {str(e)}',
            'timestamp': time.time()
        }, room=sid)

@sio.event
def change_controller(sid, data):
    """Handle controller change request from client"""
    print(f"🎮 Controller change requested by {sid}: {data}")
    
    try:
        controller_type = data.get('controller_type', 'ai')
        strategy = data.get('strategy', 'mappo')
        
        from web_app.app import get_simulation_manager
        
        simulation_manager = get_simulation_manager()
        
        if controller_type == 'ai':
            success = simulation_manager.set_ai_controller(strategy)
        else:
            success = simulation_manager.set_baseline_controller(strategy)
        
        if success:
            # Update socket handler
            socket_handler.simulation_data['controller_info'] = {
                'type': controller_type,
                'strategy': strategy,
                'performance': 0.0
            }
            
            # Notify all clients
            socket_handler.send_controller_change(controller_type, strategy, "User request")
            
            # Send success to requesting client
            sio.emit('command_success', {
                'command': 'change_controller',
                'message': f'Controller changed to {controller_type} ({strategy})',
                'timestamp': time.time()
            }, room=sid)
            
        else:
            sio.emit('command_error', {
                'command': 'change_controller',
                'message': 'Failed to change controller',
                'timestamp': time.time()
            }, room=sid)
            
    except Exception as e:
        print(f"❌ Error changing controller: {e}")
        sio.emit('command_error', {
            'command': 'change_controller',
            'message': f'Error: {str(e)}',
            'timestamp': time.time()
        }, room=sid)

@sio.event
def request_simulation_status(sid, data=None):
    """Handle simulation status request from client"""
    print(f"📊 Simulation status requested by {sid}")
    
    try:
        from web_app.app import get_simulation_manager
        
        simulation_manager = get_simulation_manager()
        status = simulation_manager.get_simulation_status()
        
        # Send status to requesting client
        sio.emit('simulation_status', {
            'status': status,
            'timestamp': time.time()
        }, room=sid)
        
    except Exception as e:
        print(f"❌ Error getting simulation status: {e}")
        sio.emit('command_error', {
            'command': 'request_simulation_status',
            'message': f'Error: {str(e)}',
            'timestamp': time.time()
        }, room=sid)

@sio.event
def request_performance_data(sid, data=None):
    """Handle performance data request from client"""
    print(f"📈 Performance data requested by {sid}")
    
    try:
        from web_app.app import get_simulation_manager
        
        simulation_manager = get_simulation_manager()
        performance_data = simulation_manager.get_performance_data()
        
        # Send performance data to requesting client
        sio.emit('performance_data', {
            'performance': performance_data,
            'metrics_history': list(socket_handler.metrics_history),
            'emergency_alerts': list(socket_handler.emergency_alerts),
            'timestamp': time.time()
        }, room=sid)
        
    except Exception as e:
        print(f"❌ Error getting performance data: {e}")
        sio.emit('command_error', {
            'command': 'request_performance_data',
            'message': f'Error: {str(e)}',
            'timestamp': time.time()
        }, room=sid)

@sio.event
def client_ready(sid, data):
    """Handle client ready notification"""
    print(f"✅ Client ready: {sid}")
    
    # Send current simulation state
    sio.emit('simulation_update', {
        'data': socket_handler.simulation_data,
        'metrics_history': list(socket_handler.metrics_history),
        'timestamp': time.time()
    }, room=sid)

@sio.event
def ping(sid, data):
    """Handle ping from client (for connection testing)"""
    sio.emit('pong', {
        'server_time': time.time(),
        'message': 'pong'
    }, room=sid)

@sio.event
def emergency_test(sid, data):
    """Handle emergency test request from client"""
    print(f"🚨 Emergency test requested by {sid}")
    
    # Create test emergency vehicle
    test_emergency = {
        'id': 'test_emergency_001',
        'type': 'ambulance',
        'position': [100, 100],
        'priority': 10,
        'direction': 'right'
    }
    
    # Send test emergency alert
    socket_handler.send_emergency_alert(
        test_emergency, 
        'test_activation', 
        'Emergency vehicle test'
    )
    
    sio.emit('command_success', {
        'command': 'emergency_test',
        'message': 'Emergency test activated',
        'timestamp': time.time()
    }, room=sid)

# Background task for periodic updates
def background_updates():
    """Background task for periodic updates"""
    while True:
        try:
            # Send periodic health check to all clients
            sio.emit('health_check', {
                'server_time': time.time(),
                'connected_clients': len(socket_handler.connected_clients),
                'system_status': 'healthy'
            })
            
            # If simulation is running, broadcast metrics periodically
            if socket_handler.connected_clients:
                from web_app.app import get_simulation_manager
                simulation_manager = get_simulation_manager()
                
                if simulation_manager.is_running():
                    # Get current metrics and broadcast
                    metrics = simulation_manager.get_current_metrics()
                    if metrics:
                        socket_handler.send_performance_metrics(metrics)
            
        except Exception as e:
            print(f"❌ Error in background updates: {e}")
        
        # Sleep for 5 seconds
        eventlet.sleep(5)

# Start background task
def start_background_tasks():
    """Start background tasks for SocketIO"""
    sio.start_background_task(background_updates)
    print("✅ Background tasks started")

def get_socket_handler() -> TrafficSocketHandler:
    """Get the global socket handler instance"""
    return socket_handler

def get_socketio_app():
    """Get the SocketIO app for Flask integration"""
    return app

if __name__ == "__main__":
    # Test the socket handler
    print("🧪 Testing Socket Handler:")
    print("=" * 50)
    
    # Test initialization
    handler = TrafficSocketHandler()
    print(f"✅ Socket handler initialized")
    print(f"✅ Connected clients: {handler.get_client_count()}")
    
    # Test simulation data update
    test_vehicles = {
        'v1': {'x': 100, 'y': 100, 'direction': 'right', 'speed': 5.0},
        'v2': {'x': 300, 'y': 100, 'direction': 'left', 'speed': 3.0}
    }
    
    test_lights = {
        'J1': {'state': 'G', 'position': [100, 100]},
        'J2': {'state': 'R', 'position': [300, 100]}
    }
    
    test_metrics = {
        'total_waiting_time': 45.2,
        'vehicles_cleared': 12,
        'average_speed': 8.5,
        'active_vehicles': 2
    }
    
    handler.update_simulation_data(test_vehicles, test_lights, test_metrics)
    print(f"✅ Simulation data updated")
    print(f"✅ Metrics history: {len(handler.metrics_history)} entries")
    
    # Test emergency alert
    test_emergency = {
        'id': 'test_001',
        'type': 'ambulance',
        'position': [150, 100],
        'priority': 10
    }
    
    handler.send_emergency_alert(test_emergency, 'detected', 'Test emergency')
    print(f"✅ Emergency alert sent")
    print(f"✅ Emergency alerts: {len(handler.emergency_alerts)}")
    
    print("✅ Socket handler test completed!")