#!/usr/bin/env python3
import os
import sys
import subprocess
import webbrowser
import time
from threading import Thread

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import traci
        import torch
        import flask
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def start_web_app():
    """Start the Flask web application"""
    print("🚀 Starting web application...")
    os.chdir('web_app')
    subprocess.run([sys.executable, 'app.py'])

def main():
    print("🤖 AI Traffic Control System - Starting...")
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies and try again")
        return
    
    # Create necessary directories
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/logs', exist_ok=True)
    
    print("📋 System ready!")
    print("\nAvailable options:")
    print("1. Train new AI model")
    print("2. Start web application")
    print("3. Run demo simulation")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("🧠 Starting training...")
        from training.trainer import main as train_main
        train_main()
    
    elif choice == "2":
        # Start web app in background thread
        web_thread = Thread(target=start_web_app)
        web_thread.daemon = True
        web_thread.start()
        
        # Wait for app to start
        time.sleep(3)
        
        # Open browser
        print("🌐 Opening web interface...")
        webbrowser.open('http://localhost:5000')
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
    
    elif choice == "3":
        print("🎮 Starting demo...")
        # Run a quick demo
        from environments.traffic_sim import TrafficSimulation
        env = TrafficSimulation(
            "environments/networks/simple_grid.net.xml",
            "environments/networks/simple_grid.sumocfg"
        )
        
        state = env.reset()
        for step in range(100):
            action = [0, 0, 0, 0]  # Simple fixed action
            state, reward, done, info = env.step(action)
            print(f"Step {step}: Reward {reward:.2f}, Vehicles cleared: {info['metrics']['vehicles_cleared']}")
            
            if done:
                break
        
        env.close()
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()