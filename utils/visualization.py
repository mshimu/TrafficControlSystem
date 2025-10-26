import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io
import base64

class TrafficVisualizer:
    """Visualizes traffic simulation and AI performance"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.colors = {
            'roads': '#7f8c8d',
            'vehicles': {
                'normal': '#3498db',
                'emergency': '#e74c3c', 
                'truck': '#f39c12',
                'motorcycle': '#2ecc71'
            },
            'traffic_lights': {
                'green': '#2ecc71',
                'yellow': '#f39c12',
                'red': '#e74c3c'
            },
            'background': '#2c3e50'
        }
        
        # Initialize matplotlib style
        plt.style.use('dark_background')
        sns.set_palette("husl")
    
    def create_simulation_plot(self, width: int = 800, height: int = 600):
        """Create the main simulation plot"""
        self.fig, self.ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        self.ax.set_facecolor(self.colors['background'])
        self.ax.set_xlim(0, 400)
        self.ax.set_ylim(0, 400)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Draw initial roads
        self._draw_roads()
        
        plt.tight_layout()
        return self.fig, self.ax
    
    def _draw_roads(self):
        """Draw the road network"""
        # Horizontal roads
        self.ax.plot([50, 350], [100, 100], color=self.colors['roads'], linewidth=40, alpha=0.7)
        self.ax.plot([50, 350], [300, 300], color=self.colors['roads'], linewidth=40, alpha=0.7)
        
        # Vertical roads  
        self.ax.plot([100, 100], [50, 350], color=self.colors['roads'], linewidth=40, alpha=0.7)
        self.ax.plot([300, 300], [50, 350], color=self.colors['roads'], linewidth=40, alpha=0.7)
        
        # Road markings (dashed lines)
        for y in [100, 300]:
            for x in range(60, 350, 40):
                self.ax.plot([x, x+20], [y-2, y-2], 'w-', linewidth=2, alpha=0.8)
        
        for x in [100, 300]:
            for y in range(60, 350, 40):
                self.ax.plot([x-2, x-2], [y, y+20], 'w-', linewidth=2, alpha=0.8)
    
    def update_simulation(self, vehicles: Dict, traffic_lights: Dict, metrics: Dict):
        """Update the simulation visualization"""
        if not self.ax:
            return
        
        # Clear previous vehicles and lights
        for artist in self.ax.collections + self.ax.patches + self.ax.texts:
            if hasattr(artist, 'traffic_element'):
                artist.remove()
        
        # Draw traffic lights
        self._draw_traffic_lights(traffic_lights)
        
        # Draw vehicles
        self._draw_vehicles(vehicles)
        
        # Update metrics display
        self._update_metrics_display(metrics)
        
        self.fig.canvas.draw_idle()
    
    def _draw_traffic_lights(self, traffic_lights: Dict):
        """Draw traffic lights at junctions"""
        junction_positions = {
            'J1': (100, 100),
            'J2': (300, 100),
            'J3': (100, 300),
            'J4': (300, 300)
        }
        
        for junction_id, light_data in traffic_lights.items():
            if junction_id in junction_positions:
                x, y = junction_positions[junction_id]
                state = light_data.get('state', 'R')
                color = self.colors['traffic_lights'].get(state, '#95a5a6')
                
                # Draw traffic light
                circle = plt.Circle((x, y), 8, color=color, alpha=0.9, zorder=10)
                circle.traffic_element = True
                self.ax.add_patch(circle)
                
                # Draw junction label
                text = self.ax.text(x, y-15, junction_id, 
                                  color='white', ha='center', va='center', 
                                  fontsize=8, fontweight='bold')
                text.traffic_element = True
    
    def _draw_vehicles(self, vehicles: Dict):
        """Draw vehicles on the roads"""
        for vehicle_id, vehicle in vehicles.items():
            x = vehicle.get('x', 0)
            y = vehicle.get('y', 0)
            vehicle_type = vehicle.get('type', 'normal')
            is_emergency = vehicle.get('is_emergency', False)
            
            if is_emergency:
                color = self.colors['vehicles']['emergency']
                # Add flashing effect for emergency vehicles
                alpha = 0.9 if int(time.time() * 2) % 2 == 0 else 0.6
            else:
                color = self.colors['vehicles'].get(vehicle_type, self.colors['vehicles']['normal'])
                alpha = 0.9
            
            # Draw vehicle as rectangle
            rect = plt.Rectangle((x-5, y-3), 10, 6, color=color, alpha=alpha, zorder=5)
            rect.traffic_element = True
            self.ax.add_patch(rect)
            
            # Draw direction indicator
            direction = vehicle.get('direction', 'right')
            if direction == 'right':
                self.ax.arrow(x+2, y, 3, 0, head_width=2, head_length=2, 
                            fc='white', ec='white', alpha=0.8)
            elif direction == 'left':
                self.ax.arrow(x-2, y, -3, 0, head_width=2, head_length=2, 
                            fc='white', ec='white', alpha=0.8)
            elif direction == 'up':
                self.ax.arrow(x, y+2, 0, 3, head_width=2, head_length=2, 
                            fc='white', ec='white', alpha=0.8)
            elif direction == 'down':
                self.ax.arrow(x, y-2, 0, -3, head_width=2, head_length=2, 
                            fc='white', ec='white', alpha=0.8)
    
    def _update_metrics_display(self, metrics: Dict):
        """Update metrics display on the plot"""
        # Clear previous metrics text
        for text in self.ax.texts:
            if hasattr(text, 'is_metric') and text.is_metric:
                text.remove()
        
        # Display key metrics
        metrics_text = [
            f"Waiting Time: {metrics.get('total_waiting_time', 0):.0f}s",
            f"Vehicles Cleared: {metrics.get('vehicles_cleared', 0)}",
            f"Avg Speed: {metrics.get('average_speed', 0):.1f}m/s",
            f"Active Vehicles: {len(metrics.get('active_vehicles', {}))}"
        ]
        
        for i, text in enumerate(metrics_text):
            metric_text = self.ax.text(10, 380 - i*20, text, color='white', 
                                     fontsize=10, fontfamily='monospace')
            metric_text.is_metric = True
    
    def create_performance_dashboard(self, metrics_history: Dict) -> go.Figure:
        """Create an interactive performance dashboard using Plotly"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Waiting Time Over Time', 'Throughput', 
                          'Average Speed', 'Emergency Response'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Waiting Time
        if 'waiting_time' in metrics_history:
            fig.add_trace(
                go.Scatter(y=metrics_history['waiting_time'], name='Waiting Time',
                          line=dict(color='#e74c3c')),
                row=1, col=1
            )
        
        # Throughput (Vehicles Cleared)
        if 'vehicles_cleared' in metrics_history:
            fig.add_trace(
                go.Scatter(y=metrics_history['vehicles_cleared'], name='Throughput',
                          line=dict(color='#2ecc71')),
                row=1, col=2
            )
        
        # Average Speed
        if 'average_speed' in metrics_history:
            fig.add_trace(
                go.Scatter(y=metrics_history['average_speed'], name='Avg Speed',
                          line=dict(color='#3498db')),
                row=2, col=1
            )
        
        # Emergency Response Time
        if 'emergency_response' in metrics_history:
            fig.add_trace(
                go.Scatter(y=metrics_history['emergency_response'], name='Emergency Response',
                          line=dict(color='#f39c12')),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="AI Traffic Control Performance Dashboard",
            showlegend=True,
            template="plotly_dark",
            height=600
        )
        
        return fig
    
    def plot_learning_curves(self, training_history: Dict) -> plt.Figure:
        """Plot learning curves from training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Performance Metrics', fontsize=16, fontweight='bold')
        
        # Episode Rewards
        if 'episode_rewards' in training_history:
            axes[0,0].plot(training_history['episode_rewards'])
            axes[0,0].set_title('Episode Rewards')
            axes[0,0].set_ylabel('Total Reward')
            axes[0,0].grid(True, alpha=0.3)
        
        # Average Waiting Time
        if 'waiting_times' in training_history:
            axes[0,1].plot(training_history['waiting_times'])
            axes[0,1].set_title('Average Waiting Time')
            axes[0,1].set_ylabel('Seconds')
            axes[0,1].grid(True, alpha=0.3)
        
        # Success Rate
        if 'success_rates' in training_history:
            axes[1,0].plot(training_history['success_rates'])
            axes[1,0].set_title('Success Rate')
            axes[1,0].set_ylabel('Rate')
            axes[1,0].set_ylim(0, 1)
            axes[1,0].grid(True, alpha=0.3)
        
        # Loss
        if 'losses' in training_history:
            axes[1,1].plot(training_history['losses'])
            axes[1,1].set_title('Training Loss')
            axes[1,1].set_ylabel('Loss')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_heatmap(self, traffic_data: np.ndarray, title: str = "Traffic Density Heatmap") -> plt.Figure:
        """Create a heatmap of traffic density"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(traffic_data, cmap='RdYlGn_r', interpolation='nearest', 
                      extent=[0, 400, 0, 400], alpha=0.7)
        
        # Add roads
        self._draw_roads_on_heatmap(ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        plt.colorbar(im, ax=ax, label='Traffic Density')
        plt.tight_layout()
        
        return fig
    
    def _draw_roads_on_heatmap(self, ax):
        """Draw roads on heatmap"""
        # Horizontal roads
        ax.axhline(y=100, xmin=50/400, xmax=350/400, color='white', linewidth=3, alpha=0.8)
        ax.axhline(y=300, xmin=50/400, xmax=350/400, color='white', linewidth=3, alpha=0.8)
        
        # Vertical roads
        ax.axvline(x=100, ymin=50/400, ymax=350/400, color='white', linewidth=3, alpha=0.8)
        ax.axvline(x=300, ymin=50/400, ymax=350/400, color='white', linewidth=3, alpha=0.8)
    
    def save_animation(self, frames: List, filename: str = "traffic_simulation.gif", fps: int = 10):
        """Save simulation as animated GIF"""
        def animate(frame):
            self.update_simulation(frame['vehicles'], frame['traffic_lights'], frame['metrics'])
            return self.ax.artists
        
        anim = animation.FuncAnimation(self.fig, animate, frames=frames, 
                                     interval=1000/fps, blit=False)
        
        anim.save(filename, writer='pillow', fps=fps)
        print(f"✅ Animation saved as {filename}")
    
    def fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string for web display"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return f"data:image/png;base64,{img_str}"

# Global visualizer instance
visualizer = TrafficVisualizer()

def get_visualizer() -> TrafficVisualizer:
    """Get the global visualizer instance"""
    return visualizer