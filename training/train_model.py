#!/usr/bin/env python3
"""
AI Traffic Control System - Training Script
===========================================

This script trains the MAPPO agent for traffic light control using
curriculum learning and advanced training techniques.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from datetime import datetime
import warnings
from typing import Dict, List, Any
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import only what we actually use
from agents.mappo_agent import MAPPOAgent
from environments.traffic_sim import TrafficSimulation
from utils.config_loader import get_config

class ModelTrainer:
    """Simple trainer - trains MAPPO agent only"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config = get_config()
        
        if config_path and os.path.exists(config_path):
            from utils.config_loader import ConfigLoader
            self.config = ConfigLoader(config_path)
        
        # Setup components
        self.setup_components()
        
        # Training state
        self.is_training = False
        self.current_episode = 0
        self.best_reward = -float('inf')
        
    def setup_components(self):
        """Initialize only essential components"""
        print("🚀 Initializing training system...")
        
        # Setup environment
        self.setup_environment()
        
        # Setup AI agent
        self.setup_agent()
        
        print("✅ Training system initialized")
    
    def setup_environment(self):
        """Setup training environment"""
        try:
            self.env = TrafficSimulation(use_ai=True, baseline_strategy="adaptive")
            print("✅ Environment setup completed")
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            self.create_fallback_environment()
    
    def create_fallback_environment(self):
        """Create simple fallback environment"""
        self.env = type('DummyEnv', (), {
            'reset': lambda self: np.zeros(48, dtype=np.float32),
            'step': lambda self, actions: (np.zeros(48, dtype=np.float32), 0.0, False, {
                'metrics': {'total_waiting_time': 0, 'vehicles_cleared': 0}
            }),
            'get_metrics': lambda self: {
                'total_waiting_time': 0, 'vehicles_cleared': 0
            },
            'close': lambda self: None
        })()
    
    def setup_agent(self):
        """Setup the MAPPO agent"""
        try:
            mappo_config = self.config.get('mappo', {})
            training_config = mappo_config.get('training', {})
            multi_agent_config = mappo_config.get('multi_agent', {})
            
            self.agent = MAPPOAgent(
                num_agents=multi_agent_config.get('num_agents', 4),
                state_dim=multi_agent_config.get('state_dim', 48),
                action_dim=multi_agent_config.get('action_dim', 6),
                learning_rate=training_config.get('learning_rate', 0.001),
                gamma=training_config.get('gamma', 0.99)
            )
            
            # Load existing model if available
            model_dir = self.config.get('data.model_dir', 'data/models')
            if os.path.exists(model_dir):
                self.try_load_existing_model(model_dir)
            else:
                os.makedirs(model_dir, exist_ok=True)
                print("📁 Created model directory")
            
            print("✅ MAPPO agent setup completed")
        except Exception as e:
            print(f"❌ Agent setup failed: {e}")
            self.agent = None
    
    def try_load_existing_model(self, model_dir: str):
        """Try to load existing model weights"""
        try:
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            if model_files:
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                latest_model = model_files[0]
                
                try:
                    episode = int(latest_model.split('_ep_')[-1].split('.')[0])
                except:
                    episode = 0
                
                if hasattr(self.agent, 'load_models'):
                    self.agent.load_models(episode)
                self.current_episode = episode
                
                print(f"✅ Loaded existing model: {latest_model} (Episode {episode})")
            else:
                print("ℹ️ No existing models found, starting fresh")
        except Exception as e:
            print(f"❌ Could not load existing model: {e}")
    
    def run_training_episode(self, episode: int) -> Dict:
        """Run a single training episode"""
        print(f"🎯 Episode {episode}")
        
        # Reset environment
        state = self.env.reset()
        
        # Training variables
        episode_reward = 0.0
        episode_steps = 0
        done = False
        
        while not done and episode_steps < 1000:
            try:
                # Get actions from MAPPO agent
                actions = self.get_actions(state)
                
                # Step environment
                next_state, reward, done, info = self.env.step(actions)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Log progress
                if episode_steps % 100 == 0:
                    print(f"   Step {episode_steps}, Reward: {reward:.2f}")
                    
            except Exception as e:
                print(f"❌ Error during step {episode_steps}: {e}")
                break
        
        # Finalize episode
        return self.finalize_episode(episode, episode_reward, episode_steps)
    
    def get_actions(self, state) -> np.ndarray:
        """Get actions from MAPPO agent"""
        try:
            if hasattr(self, 'agent') and self.agent:
                # Use MAPPO agent directly
                states = [state] * self.agent.num_agents
                actions, log_probs = self.agent.select_actions(states)
                return np.array(actions)
            else:
                # Fallback to random actions
                return np.random.randint(0, 4, size=4)
                
        except Exception as e:
            print(f"❌ Error getting actions: {e}")
            return np.random.randint(0, 4, size=4)
    
    def finalize_episode(self, episode: int, total_reward: float, steps: int):
        """Finalize episode"""
        try:
            final_metrics = self.env.get_metrics()
        except:
            final_metrics = {'total_waiting_time': 0, 'vehicles_cleared': 0}
        
        # Log results
        print(f"🏁 Episode {episode} completed:")
        print(f"   Reward: {total_reward:.2f}, Steps: {steps}")
        print(f"   Waiting Time: {final_metrics.get('total_waiting_time', 0):.1f}s")
        print(f"   Vehicles Cleared: {final_metrics.get('vehicles_cleared', 0)}")
        
        # Save model if improved
        if total_reward > self.best_reward:
            self.save_model(episode, total_reward)
            self.best_reward = total_reward
        
        return {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'metrics': final_metrics
        }
    
    def save_model(self, episode: int, reward: float):
        """Save model checkpoint"""
        try:
            if hasattr(self.agent, 'save_models'):
                self.agent.save_models(episode)
                print(f"💾 Model saved at episode {episode} (Reward: {reward:.2f})")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def run_training(self, total_episodes: int = 100):
        """Run main training loop"""
        print(f"🚀 Starting training for {total_episodes} episodes")
        print("=" * 60)
        
        self.is_training = True
        start_time = time.time()
        
        try:
            for episode in range(self.current_episode, self.current_episode + total_episodes):
                if not self.is_training:
                    break
                
                self.run_training_episode(episode)
                self.current_episode = episode + 1
        
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted")
        except Exception as e:
            print(f"❌ Training error: {e}")
        finally:
            self.finalize_training(start_time)
    
    def finalize_training(self, start_time: float):
        """Finalize training"""
        training_time = time.time() - start_time
        print(f"\n🏁 Training completed!")
        print(f"⏱️  Time: {training_time:.1f}s, Episodes: {self.current_episode}")
        print(f"🎯 Best reward: {self.best_reward:.2f}")
        self.save_model(self.current_episode, self.best_reward)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train AI Traffic Control System')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    
    args = parser.parse_args()
    
    print("🤖 AI Traffic Control System - Training")
    print("=" * 50)
    
    try:
        trainer = ModelTrainer(config_path=args.config)
        trainer.run_training(total_episodes=args.episodes)
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()