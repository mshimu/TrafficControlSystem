import numpy as np
import torch
import time
from tqdm import tqdm
from environments.traffic_sim import TrafficSimulation
from agents.mappo_agent import MAPPOAgent
import json
import os

class TrafficTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.setup_environment()
        self.setup_agents()
        self.setup_logging()
    
    def setup_environment(self):
        """Initialize traffic simulation environment"""
        network_file = "environments/networks/simple_grid.net.xml"
        config_file = "environments/networks/simple_grid.sumocfg"
        
        self.env = TrafficSimulation(
            network_file=network_file,
            config_file=config_file,
            emergency_vehicles=self.config.get('emergency_vehicles', True)
        )
        
        self.num_agents = self.env.num_intersections
    
    def setup_agents(self):
        """Initialize RL agents"""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.nvec[0]  # Same for all agents
        
        self.agent = MAPPOAgent(
            num_agents=self.num_agents,
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=self.config.get('learning_rate', 0.001),
            gamma=self.config.get('gamma', 0.99)
        )
    
    def setup_logging(self):
        """Setup training logging"""
        self.log_dir = "data/logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_metrics = []
    
    def train(self, num_episodes: int = 1000):
        """Main training loop"""
        print(f"Starting training for {num_episodes} episodes...")
        
        for episode in tqdm(range(num_episodes)):
            states = self.env.reset()
            episode_reward = 0
            episode_metrics = []
            
            done = False
            step_count = 0
            
            while not done and step_count < self.env.max_steps:
                # Get actions from agents
                actions, log_probs = self.agent.select_actions([states] * self.num_agents)
                
                # Take step in environment
                next_states, rewards, done, info = self.env.step(np.array(actions))
                
                # Store experience and update
                self.agent.update(
                    states=[states] * self.num_agents,
                    actions=actions,
                    rewards=[rewards] * self.num_agents,  # Same reward for all agents
                    next_states=[next_states] * self.num_agents,
                    dones=[done] * self.num_agents,
                    log_probs=log_probs
                )
                
                states = next_states
                episode_reward += rewards
                episode_metrics.append(info['metrics'])
                step_count += 1
            
            # Log episode results
            self.episode_rewards.append(episode_reward)
            self.episode_metrics.append(episode_metrics)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
            
            # Save model periodically
            if (episode + 1) % 100 == 0:
                self.save_model(episode + 1)
        
        self.env.close()
        self.save_training_data()
    
    def save_model(self, episode: int):
        """Save trained models"""
        model_dir = "data/models"
        os.makedirs(model_dir, exist_ok=True)
        
        for i, actor in enumerate(self.agent.actors):
            torch.save(actor.state_dict(), f"{model_dir}/actor_agent_{i}_ep_{episode}.pth")
        
        print(f"Models saved at episode {episode}")
    
    def save_training_data(self):
        """Save training logs and metrics"""
        training_data = {
            'episode_rewards': self.episode_rewards,
            'episode_metrics': self.episode_metrics,
            'config': self.config
        }
        
        with open(f"{self.log_dir}/training_data.json", 'w') as f:
            json.dump(training_data, f, indent=2)

def main():
    config = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'emergency_vehicles': True,
        'mixed_control': False  # Start without mixed control
    }
    
    trainer = TrafficTrainer(config)
    trainer.train(num_episodes=500)

if __name__ == "__main__":
    main()