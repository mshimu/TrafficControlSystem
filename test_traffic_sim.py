import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from environments.traffic_sim import TrafficSimulation
    print("✅ Traffic simulation imports successfully!")
    
    # Test creating instance
    sim = TrafficSimulation()
    print(f"✅ SUMO command: {sim.sumo_cmd}")
    print(f"✅ Max steps: {sim.max_steps}")
    
except Exception as e:
    print(f"❌ Error: {e}")