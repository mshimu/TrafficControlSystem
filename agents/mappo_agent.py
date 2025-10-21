import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
import random
from collections import deque

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from config import MODEL_DIR, TRAINING_CONFIG
except ImportError:
    # Fallback values
    MODEL_DIR = "data/models"
    TRAINING_CONFIG = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'batch_size': 64,
        'epochs': 1000,
    }

class ActorNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def save_models(self, episode: int):
    """Save all agent models using config paths"""
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for i, actor in enumerate(self.actors):
        torch.save(actor.state_dict(), f"{MODEL_DIR}/actor_agent_{i}_ep_{episode}.pth")
    
    for i, critic in enumerate(self.critics):
        torch.save(critic.state_dict(), f"{MODEL_DIR}/critic_agent_{i}_ep_{episode}.pth")
    
    print(f"💾 Models saved to {MODEL_DIR} at episode {episode}")

def load_models(self, episode: int):
    """Load agent models"""
    for i, actor in enumerate(self.actors):
        model_path = f"{MODEL_DIR}/actor_agent_{i}_ep_{episode}.pth"
        if os.path.exists(model_path):
            actor.load_state_dict(torch.load(model_path))
    
    for i, critic in enumerate(self.critics):
        model_path = f"{MODEL_DIR}/critic_agent_{i}_ep_{episode}.pth"
        if os.path.exists(model_path):
            critic.load_state_dict(torch.load(model_path))
    
    print(f"📂 Models loaded from {MODEL_DIR} episode {episode}")

class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class MAPPOAgent:
    def __init__(self, num_agents: int, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.001, gamma: float = 0.99):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Create networks for each agent
        self.actors = [ActorNetwork(state_dim, action_dim) for _ in range(num_agents)]
        self.critics = [CriticNetwork(state_dim) for _ in range(num_agents)]
        
        # Optimizers
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=learning_rate) 
                               for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=learning_rate) 
                                for critic in self.critics]
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        
    def select_actions(self, states: List[np.ndarray]) -> List[int]:
        """Select actions for all agents"""
        actions = []
        log_probs = []
        
        for i, state in enumerate(states):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.actors[i](state_tensor)
            
            # Sample action from distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            actions.append(action.item())
            log_probs.append(log_prob)
        
        return actions, log_probs
    
    def update(self, states: List[np.ndarray], actions: List[int], 
               rewards: List[float], next_states: List[np.ndarray], 
               dones: List[bool], log_probs: List[torch.Tensor]):
        """Update agent policies"""
        for i in range(self.num_agents):
            # Convert to tensors
            state = torch.FloatTensor(states[i]).unsqueeze(0)
            next_state = torch.FloatTensor(next_states[i]).unsqueeze(0)
            action = torch.LongTensor([actions[i]])
            reward = torch.FloatTensor([rewards[i]])
            done = torch.FloatTensor([dones[i]])
            old_log_prob = log_probs[i].unsqueeze(0)
            
            # Calculate advantages
            with torch.no_grad():
                current_value = self.critics[i](state)
                next_value = self.critics[i](next_state)
                target_value = reward + self.gamma * next_value * (1 - done)
                advantage = target_value - current_value
            
            # Update critic
            critic_loss = nn.MSELoss()(current_value, target_value)
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()
            
            # Update actor
            action_probs = self.actors[i](state)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_prob = action_dist.log_prob(action)
            
            # PPO ratio
            ratio = torch.exp(new_log_prob - old_log_prob)
            surrogate1 = ratio * advantage
            surrogate2 = torch.clamp(ratio, 0.8, 1.2) * advantage
            
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()