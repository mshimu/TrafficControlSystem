import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    """Loads and manages configuration from YAML files"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), 
                                                      "..", "training", "config.yaml")
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            print(f"✅ Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            print(f"❌ Config file not found: {self.config_path}")
            self.config = self._get_default_config()
        except yaml.YAMLError as e:
            print(f"❌ Error parsing config file: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file is missing"""
        return {
            'system': {
                'name': 'AI Traffic Control System',
                'mode': 'training',
                'log_level': 'INFO'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None):
        """Save configuration to YAML file"""
        save_path = path or self.config_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
        
        print(f"💾 Configuration saved to {save_path}")
    
    def validate(self) -> bool:
        """Validate configuration"""
        required_sections = ['system', 'environment', 'mappo', 'training']
        
        for section in required_sections:
            if section not in self.config:
                print(f"❌ Missing required configuration section: {section}")
                return False
        
        # Validate specific values
        if self.get('training.total_timesteps', 0) <= 0:
            print("❌ total_timesteps must be positive")
            return False
        
        if self.get('mappo.training.learning_rate', 0) <= 0:
            print("❌ learning_rate must be positive")
            return False
        
        print("✅ Configuration validation passed")
        return True
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n📋 Configuration Summary:")
        print("=" * 50)
        
        # System
        print(f"System: {self.get('system.name')} (v{self.get('system.version', '1.0.0')})")
        print(f"Mode: {self.get('system.mode')}")
        print(f"Log Level: {self.get('system.log_level')}")
        
        # Training
        print(f"\nTraining:")
        print(f"  Total Timesteps: {self.get('training.total_timesteps'):,}")
        print(f"  Learning Rate: {self.get('mappo.training.learning_rate')}")
        print(f"  Batch Size: {self.get('mappo.replay_buffer.batch_size')}")
        
        # Environment
        print(f"\nEnvironment:")
        print(f"  Junctions: {len(self.get('environment.junctions', {}))}")
        print(f"  Max Vehicles: {self.get('environment.traffic.max_vehicles')}")
        print(f"  Emergency Probability: {self.get('environment.traffic.emergency_vehicle_probability')}")
        
        # Curriculum
        stages = self.get('curriculum.stages', [])
        print(f"\nCurriculum:")
        print(f"  Stages: {len(stages)}")
        for stage in stages[:3]:  # Show first 3 stages
            print(f"    - {stage['name']} ({stage['difficulty']})")
        if len(stages) > 3:
            print(f"    - ... and {len(stages) - 3} more")
        
        print("=" * 50)

# Global config instance
config_loader = ConfigLoader()

def get_config() -> ConfigLoader:
    """Get the global configuration loader"""
    return config_loader

if __name__ == "__main__":
    # Test the config loader
    config = get_config()
    config.print_summary()
    
    # Test getting values
    print(f"\n🧪 Testing config access:")
    print(f"System name: {config.get('system.name')}")
    print(f"Learning rate: {config.get('mappo.training.learning_rate')}")
    print(f"Non-existent key: {config.get('non.existent.key', 'default_value')}")