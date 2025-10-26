import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
import random
from collections import deque
import time

class Experience:
    """Represents a single experience for reinforcement learning"""
    
    def __init__(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool, info: Dict = None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info or {}
        self.timestamp = time.time()
        self.priority = 1.0  # Initial priority
    
    def to_tensor(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Convert experience to tensors"""
        return {
            'state': torch.FloatTensor(self.state).to(device),
            'action': torch.LongTensor([self.action]).to(device),
            'reward': torch.FloatTensor([self.reward]).to(device),
            'next_state': torch.FloatTensor(self.next_state).to(device),
            'done': torch.FloatTensor([float(self.done)]).to(device)
        }

class MultiAgentExperience:
    """Represents experiences for multiple agents"""
    
    def __init__(self, agent_experiences: Dict[str, Experience], global_reward: float = 0.0):
        self.agent_experiences = agent_experiences
        self.global_reward = global_reward
        self.timestamp = time.time()
        self.priority = 1.0
    
    def get_agent_ids(self) -> List[str]:
        """Get list of agent IDs in this experience"""
        return list(self.agent_experiences.keys())

class PriorityReplayBuffer:
    """Prioritized Experience Replay Buffer for single agent"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.beta_increment = 0.001
        
    def add(self, experience: Experience, priority: float = None):
        """Add an experience to the buffer"""
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample a batch of experiences with priority"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for specific experiences"""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if len(self.buffer) == 0:
            return {'size': 0, 'avg_priority': 0.0, 'fullness': 0.0}
        
        return {
            'size': len(self.buffer),
            'avg_priority': np.mean(self.priorities),
            'max_priority': np.max(self.priorities),
            'min_priority': np.min(self.priorities),
            'fullness': len(self.buffer) / self.capacity,
            'beta': self.beta
        }

class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent reinforcement learning"""
    
    def __init__(self, capacity: int = 50000, num_agents: int = 4):
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffers = {f'agent_{i}': PriorityReplayBuffer(capacity // num_agents) 
                       for i in range(num_agents)}
        self.global_buffer = deque(maxlen=1000)  # Store global experiences
        self.episode_buffer = []  # Temporary storage for current episode
    
    def add_agent_experience(self, agent_id: str, experience: Experience, priority: float = None):
        """Add experience for a specific agent"""
        if agent_id in self.buffers:
            self.buffers[agent_id].add(experience, priority)
    
    def add_multi_agent_experience(self, multi_agent_experience: MultiAgentExperience):
        """Add multi-agent experience"""
        # Add individual agent experiences
        for agent_id, experience in multi_agent_experience.agent_experiences.items():
            if agent_id in self.buffers:
                self.buffers[agent_id].add(experience, multi_agent_experience.priority)
        
        # Store global experience
        self.global_buffer.append(multi_agent_experience)
    
    def add_episode_experience(self, multi_agent_experience: MultiAgentExperience):
        """Add experience to current episode buffer"""
        self.episode_buffer.append(multi_agent_experience)
    
    def finalize_episode(self, episode_reward: float):
        """Finalize current episode and add to replay buffers"""
        if not self.episode_buffer:
            return
        
        # Calculate priorities based on episode reward
        base_priority = 1.0 + abs(episode_reward) * 0.1
        
        for experience in self.episode_buffer:
            experience.priority = base_priority
            self.add_multi_agent_experience(experience)
        
        # Clear episode buffer
        self.episode_buffer.clear()
    
    def sample_agent_batch(self, agent_id: str, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch for a specific agent"""
        if agent_id in self.buffers:
            return self.buffers[agent_id].sample(batch_size)
        else:
            return [], np.array([]), np.array([])
    
    def sample_multi_agent_batch(self, batch_size: int) -> Dict[str, Tuple[List[Experience], np.ndarray, np.ndarray]]:
        """Sample batch for all agents"""
        batches = {}
        for agent_id, buffer in self.buffers.items():
            experiences, indices, weights = buffer.sample(batch_size)
            batches[agent_id] = (experiences, indices, weights)
        
        return batches
    
    def sample_global_batch(self, batch_size: int) -> List[MultiAgentExperience]:
        """Sample batch of global experiences"""
        if len(self.global_buffer) < batch_size:
            return random.sample(self.global_buffer, len(self.global_buffer))
        else:
            return random.sample(self.global_buffer, batch_size)
    
    def update_agent_priorities(self, agent_id: str, indices: List[int], priorities: List[float]):
        """Update priorities for a specific agent"""
        if agent_id in self.buffers:
            self.buffers[agent_id].update_priorities(indices, priorities)
    
    def __len__(self) -> int:
        return sum(len(buffer) for buffer in self.buffers.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive buffer statistics"""
        stats = {
            'total_experiences': self.__len__(),
            'global_experiences': len(self.global_buffer),
            'current_episode_length': len(self.episode_buffer),
            'capacity': self.capacity,
            'fullness': self.__len__() / self.capacity,
            'agent_stats': {}
        }
        
        for agent_id, buffer in self.buffers.items():
            stats['agent_stats'][agent_id] = buffer.get_stats()
        
        return stats

class TrafficExperienceProcessor:
    """Processes and enhances traffic control experiences"""
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_normalizer = RewardNormalizer()
        self.state_normalizer = StateNormalizer(state_dim)
    
    def process_experience(self, raw_experience: Dict) -> Experience:
        """Process raw experience into standardized format"""
        state = self.state_normalizer.normalize(raw_experience['state'])
        action = raw_experience['action']
        reward = self.reward_normalizer.normalize(raw_experience['reward'])
        next_state = self.state_normalizer.normalize(raw_experience['next_state'])
        done = raw_experience.get('done', False)
        info = raw_experience.get('info', {})
        
        return Experience(state, action, reward, next_state, done, info)
    
    def calculate_traffic_reward(self, metrics: Dict, action: int, 
                               previous_metrics: Dict) -> float:
        """Calculate reward for traffic control action"""
        reward = 0.0
        
        # Waiting time improvement
        current_waiting = metrics.get('total_waiting_time', 0)
        previous_waiting = previous_metrics.get('total_waiting_time', 0)
        waiting_improvement = previous_waiting - current_waiting
        reward += waiting_improvement * 0.01
        
        # Throughput reward
        vehicles_cleared = metrics.get('vehicles_cleared', 0)
        reward += vehicles_cleared * 0.1
        
        # Average speed reward
        avg_speed = metrics.get('average_speed', 0)
        reward += avg_speed * 0.05
        
        # Emergency response reward
        emergency_time = metrics.get('emergency_response_time', 30)
        reward += (30 - emergency_time) * 0.2  # Lower time = higher reward
        
        # Action efficiency penalty (avoid frequent switching)
        if action in [1, 2]:  # Switching actions
            reward -= 0.1
        
        # Congestion penalty
        if metrics.get('congestion_level', 0) > 0.8:
            reward -= 0.5
        
        return reward
    
    def enhance_experience(self, experience: Experience, 
                          additional_info: Dict) -> Experience:
        """Enhance experience with additional information"""
        enhanced_info = experience.info.copy()
        enhanced_info.update(additional_info)
        
        # Add temporal information
        enhanced_info['timestamp'] = time.time()
        enhanced_info['training_step'] = additional_info.get('training_step', 0)
        
        return Experience(
            experience.state,
            experience.action,
            experience.reward,
            experience.next_state,
            experience.done,
            enhanced_info
        )

class RewardNormalizer:
    """Normalizes rewards for stable learning"""
    
    def __init__(self, clip_value: float = 10.0):
        self.clip_value = clip_value
        self.reward_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}
    
    def normalize(self, reward: float) -> float:
        """Normalize reward using running statistics"""
        # Update statistics
        self.reward_stats['count'] += 1
        delta = reward - self.reward_stats['mean']
        self.reward_stats['mean'] += delta / self.reward_stats['count']
        delta2 = reward - self.reward_stats['mean']
        self.reward_stats['std'] += delta * delta2
        
        # Calculate normalized reward
        if self.reward_stats['std'] > 0 and self.reward_stats['count'] > 1:
            std = np.sqrt(self.reward_stats['std'] / (self.reward_stats['count'] - 1))
            normalized = (reward - self.reward_stats['mean']) / (std + 1e-8)
        else:
            normalized = reward
        
        # Clip extreme values
        return np.clip(normalized, -self.clip_value, self.clip_value)
    
    def reset(self):
        """Reset normalization statistics"""
        self.reward_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}

class StateNormalizer:
    """Normalizes state vectors"""
    
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.state_stats = {
            'mean': np.zeros(state_dim),
            'std': np.ones(state_dim),
            'count': 0
        }
    
    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize state vector"""
        if self.state_stats['count'] == 0:
            return state
        
        normalized = (state - self.state_stats['mean']) / (self.state_stats['std'] + 1e-8)
        return np.clip(normalized, -5, 5)  # Clip extreme values
    
    def update(self, state: np.ndarray):
        """Update normalization statistics with new state"""
        self.state_stats['count'] += 1
        delta = state - self.state_stats['mean']
        self.state_stats['mean'] += delta / self.state_stats['count']
        delta2 = state - self.state_stats['mean']
        self.state_stats['std'] += delta * delta2
    
    def reset(self):
        """Reset normalization statistics"""
        self.state_stats = {
            'mean': np.zeros(self.state_dim),
            'std': np.ones(self.state_dim),
            'count': 0
        }

# Global replay buffer instance
replay_buffer = MultiAgentReplayBuffer(capacity=100000, num_agents=4)

def get_replay_buffer() -> MultiAgentReplayBuffer:
    """Get the global replay buffer instance"""
    return replay_buffer

if __name__ == "__main__":
    # Test the replay buffer system
    buffer = MultiAgentReplayBuffer(capacity=1000, num_agents=4)
    processor = TrafficExperienceProcessor(state_dim=32, action_dim=6)
    
    print("🧪 Testing Replay Buffer System:")
    print("=" * 50)
    
    # Generate test experiences
    for i in range(100):
        for agent_id in [f'agent_{j}' for j in range(4)]:
            state = np.random.randn(32)
            action = random.randint(0, 5)
            reward = random.uniform(-1, 1)
            next_state = np.random.randn(32)
            done = random.random() < 0.1
            
            raw_experience = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'info': {'step': i}
            }
            
            experience = processor.process_experience(raw_experience)
            buffer.add_agent_experience(agent_id, experience)
    
    # Test sampling
    batch = buffer.sample_multi_agent_batch(32)
    
    print(f"📊 Buffer Statistics:")
    stats = buffer.get_stats()
    print(f"   Total Experiences: {stats['total_experiences']}")
    print(f"   Global Experiences: {stats['global_experiences']}")
    print(f"   Buffer Fullness: {stats['fullness']:.1%}")
    
    for agent_id, agent_stats in stats['agent_stats'].items():
        print(f"   {agent_id}: {agent_stats['size']} experiences, "
              f"avg priority: {agent_stats['avg_priority']:.3f}")