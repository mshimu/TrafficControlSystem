Step-by-Step Setup for Windows:

1. Create the Project Structure:
:: Run in Command Prompt as Administrator
mkdir TRAFFIC_AI_SYSTEM
cd TRAFFIC_AI_SYSTEM

:: Create all folders
mkdir environments agents training web_app data utils docs
mkdir environments\networks
mkdir web_app\templates web_app\static
mkdir data\models data\logs data\results data\datasets

2. Install Python & Dependencies:
:: 1. Install Python 3.8+ from python.org
:: 2. Create virtual environment
python -m venv traffic_env
traffic_env\Scripts\activate

:: 3. Install requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install sumo traci flask flask-socketio numpy pandas matplotlib seaborn
pip install stable-baselines3 gym tqdm pyyaml plotly

3. Install SUMO on Windows:
:: Download SUMO from: https://github.com/eclipse/sumo/releases
:: Install to: C:\Program Files\sumo\
:: Add to PATH: C:\Program Files\sumo\bin

:: Verify installation
sumo-gui --version

