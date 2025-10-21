import os

# ===== PATH CONFIGURATION =====
PYTHON_PATH = "C:/Python313"
SUMO_PATH = "D:/MixSoftware/Installed/Sumo" 
PROJECT_ROOT = "D:/WorkSpace/VisualStudio/TrafficControlSystem"

# ===== VIRTUAL ENVIRONMENT =====
VENV_PATH = f"{PROJECT_ROOT}/traffic_env"

# ===== SUMO PATHS =====
SUMO_BINARY = f"{SUMO_PATH}/bin/sumo-gui.exe"
NETCONVERT_BINARY = f"{SUMO_PATH}/bin/netconvert.exe"

# ===== PROJECT PATHS =====
NETWORK_DIR = f"{PROJECT_ROOT}/environments/networks"
MODEL_DIR = f"{PROJECT_ROOT}/data/models"
LOG_DIR = f"{PROJECT_ROOT}/data/logs"
RESULTS_DIR = f"{PROJECT_ROOT}/data/results"

# ===== SIMULATION CONFIG =====
SIMULATION_CONFIG = {
    'max_steps': 3600,           # 1 hour simulation
    'step_length': 1,            # 1 second per step
    'emergency_spawn_prob': 0.001,  # 0.1% chance per step
    'num_intersections': 16,
}

# ===== AI TRAINING CONFIG =====
TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'gamma': 0.99,
    'batch_size': 64,
    'epochs': 1000,
}

# ===== WEB APP CONFIG =====
WEB_CONFIG = {
    'host': 'localhost',
    'port': 5000,
    'debug': True,
}

# ===== CREATE DIRECTORIES =====
def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [NETWORK_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Run setup when config is imported
setup_directories()