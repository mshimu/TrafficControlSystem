import logging
import sys
import os
from typing import Dict, Any
import json
from datetime import datetime
import colorama
from colorama import Fore, Style

# Initialize colorama for colored console output
colorama.init()

class ColorFormatter(logging.Formatter):
    """Custom formatter for colored console output"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        
        # Format the message
        formatted = super().format(record)
        return formatted

class TrafficLogger:
    """Custom logger for AI Traffic Control System"""
    
    def __init__(self, name: str = "traffic_ai", log_dir: str = "data/logs"):
        self.name = name
        self.log_dir = log_dir
        self.setup_logging()
        
        # Specialized loggers
        self.traffic_logger = logging.getLogger(f"{name}.traffic")
        self.training_logger = logging.getLogger(f"{name}.training")
        self.emergency_logger = logging.getLogger(f"{name}.emergency")
        self.coordination_logger = logging.getLogger(f"{name}.coordination")
        
    def setup_logging(self):
        """Setup logging configuration"""
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"traffic_ai_{timestamp}.log")
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Apply color formatting to console handler
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(ColorFormatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
        
        print(f"📝 Logging initialized. Log file: {log_file}")
    
    def log_traffic_update(self, vehicles: int, waiting_time: float, cleared: int):
        """Log traffic state update"""
        self.traffic_logger.info(
            f"Traffic Update - Vehicles: {vehicles}, "
            f"Waiting: {waiting_time:.1f}s, Cleared: {cleared}"
        )
    
    def log_emergency_event(self, emergency_id: str, event_type: str, details: Dict = None):
        """Log emergency vehicle events"""
        details_str = f" - {json.dumps(details)}" if details else ""
        self.emergency_logger.warning(
            f"EMERGENCY - {emergency_id} - {event_type}{details_str}"
        )
    
    def log_training_progress(self, episode: int, reward: float, metrics: Dict):
        """Log training progress"""
        self.training_logger.info(
            f"Training - Episode {episode}, Reward: {reward:.2f}, "
            f"Metrics: {json.dumps(metrics, indent=None)}"
        )
    
    def log_coordination_event(self, strategy: str, junctions: list, result: str):
        """Log coordination events"""
        self.coordination_logger.info(
            f"Coordination - Strategy: {strategy}, "
            f"Junctions: {junctions}, Result: {result}"
        )
    
    def log_ai_decision(self, junction: str, action: str, reason: str, confidence: float):
        """Log AI decision making"""
        self.traffic_logger.debug(
            f"AI Decision - {junction}: {action} "
            f"(confidence: {confidence:.2f}) - {reason}"
        )
    
    def log_system_metrics(self, metrics: Dict[str, Any]):
        """Log system performance metrics"""
        metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        logging.info(f"System Metrics - {metrics_str}")
    
    def log_experiment_start(self, experiment_name: str, config: Dict):
        """Log experiment start with configuration"""
        logging.info(f"🚀 Starting Experiment: {experiment_name}")
        logging.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    def log_experiment_end(self, experiment_name: str, results: Dict):
        """Log experiment end with results"""
        logging.info(f"🏁 Experiment Completed: {experiment_name}")
        logging.info(f"Results: {json.dumps(results, indent=2)}")

class PerformanceMonitor:
    """Monitors and logs system performance"""
    
    def __init__(self, logger: TrafficLogger):
        self.logger = logger
        self.metrics_history = []
        self.alert_thresholds = {
            'high_waiting_time': 300.0,  # seconds
            'low_throughput': 5,         # vehicles per minute
            'high_emergency_response': 30.0,  # seconds
            'system_overload': 0.9       # 90% capacity
        }
    
    def check_performance_alerts(self, metrics: Dict[str, Any]):
        """Check for performance issues and log alerts"""
        alerts = []
        
        # Check waiting time
        if metrics.get('total_waiting_time', 0) > self.alert_thresholds['high_waiting_time']:
            alerts.append(f"High waiting time: {metrics['total_waiting_time']:.1f}s")
        
        # Check throughput
        throughput = metrics.get('vehicles_cleared', 0)
        if throughput < self.alert_thresholds['low_throughput']:
            alerts.append(f"Low throughput: {throughput} vehicles")
        
        # Check emergency response
        emergency_time = metrics.get('average_emergency_response', 0)
        if emergency_time > self.alert_thresholds['high_emergency_response']:
            alerts.append(f"Slow emergency response: {emergency_time:.1f}s")
        
        # Log alerts
        for alert in alerts:
            self.logger.emergency_logger.error(f"PERFORMANCE ALERT: {alert}")
        
        return alerts
    
    def log_performance_summary(self, metrics: Dict[str, Any], interval: int = 60):
        """Log periodic performance summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'interval_seconds': interval,
            'vehicles_cleared': metrics.get('vehicles_cleared', 0),
            'average_waiting_time': metrics.get('average_waiting_time', 0),
            'average_speed': metrics.get('average_speed', 0),
            'efficiency': metrics.get('efficiency', 0),
            'performance_score': metrics.get('performance_score', 0)
        }
        
        self.metrics_history.append(summary)
        self.logger.log_system_metrics(summary)
        
        # Keep history manageable
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)

class ExperimentLogger:
    """Manages logging for training experiments"""
    
    def __init__(self, experiment_name: str, log_dir: str = "data/logs/experiments"):
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.config_file = os.path.join(self.log_dir, "config.json")
        self.results_file = os.path.join(self.log_dir, "results.json")
        self.metrics_file = os.path.join(self.log_dir, "metrics.csv")
        
        self.metrics_data = []
    
    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Experiment config saved to {self.config_file}")
    
    def log_training_step(self, step: int, metrics: Dict[str, Any]):
        """Log training step metrics"""
        metrics['step'] = step
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_data.append(metrics)
        
        # Save periodically to avoid memory issues
        if step % 100 == 0:
            self._save_metrics()
    
    def save_results(self, results: Dict[str, Any]):
        """Save final experiment results"""
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save final metrics
        self._save_metrics()
        
        print(f"✅ Experiment results saved to {self.results_file}")
    
    def _save_metrics(self):
        """Save metrics to CSV file"""
        if self.metrics_data:
            import pandas as pd
            df = pd.DataFrame(self.metrics_data)
            df.to_csv(self.metrics_file, index=False)

# Global logger instances
traffic_logger = TrafficLogger()
performance_monitor = PerformanceMonitor(traffic_logger)

def get_logger() -> TrafficLogger:
    """Get the global logger instance"""
    return traffic_logger

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor"""
    return performance_monitor

def create_experiment_logger(experiment_name: str) -> ExperimentLogger:
    """Create a new experiment logger"""
    return ExperimentLogger(experiment_name)

if __name__ == "__main__":
    # Test the logging system
    logger = get_logger()
    
    print("🧪 Testing Logging System:")
    print("=" * 50)
    
    # Test different log levels and types
    logger.log_traffic_update(vehicles=15, waiting_time=45.2, cleared=8)
    logger.log_emergency_event("AMB-001", "vehicle_detected", {"priority": 10})
    logger.log_training_progress(episode=100, reward=25.5, metrics={"waiting_time": 30.2})
    logger.log_coordination_event("green_wave", ["J1", "J2"], "success")
    logger.log_ai_decision("J1", "switch_green", "high_traffic", 0.85)
    
    # Test performance monitoring
    test_metrics = {
        'total_waiting_time': 350.0,  # Should trigger alert
        'vehicles_cleared': 3,        # Should trigger alert
        'average_emergency_response': 35.0,  # Should trigger alert
        'average_speed': 12.5,
        'efficiency': 0.7
    }
    
    alerts = performance_monitor.check_performance_alerts(test_metrics)
    print(f"Performance Alerts: {alerts}")
    
    print("✅ Logging system test completed!")