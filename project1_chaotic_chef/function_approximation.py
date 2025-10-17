"""
Function Approximation Methods: Linear and Neural Network Q-Learning
For The Chaotic Chef's Quest for the Perfect Meal
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Tuple
import pickle
from environment import ChaoticChefEnvironment


class LinearQNetwork:
    """
    Linear Function Approximation for Q-Learning.
    Q(s,a) = w^T * φ(s,a)
    """
    
    def __init__(self,
                 env: ChaoticChefEnvironment,
                 learning_rate: float = 0.01,
                 discount_factor: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Linear Q-Network.
        
        Args:
            env: Environment instance
            learning_rate: Learning rate α
            discount_factor: Discount factor γ
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
        """
        self.env = env
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Feature size (state features * actions)
        self.state_size = env.get_state_size()
        self.action_size = env.get_action_size()
        self.feature_size = self.state_size + self.action_size
        
        # Weight vector
        self.weights = np.zeros(self.feature_size)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        self.loss_history = []
    
    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Extract features φ(s,a) for linear approximation.
        
        Args:
            state: State vector
            action: Action index
            
        Returns:
            Feature vector
        """
        # One-hot encode action
        action_onehot = np.zeros(self.action_size)
        action_onehot[action] = 1
        
        # Concatenate state and action
        features = np.concatenate([state, action_onehot])
        
        return features
    
    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """
        Compute Q(s,a) = w^T * φ(s,a).
        
        Args:
            state: State vector
            action: Action index
            
        Returns:
            Q-value
        """
        features = self.get_features(state, action)
        return np.dot(self.weights, features)
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.action_size)]
            return np.argmax(q_values)
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> float:
        """
        Update weights using gradient descent.
        
        w ← w + α[R + γ max_a' Q(s',a') - Q(s,a)] * ∇_w Q(s,a)
        """
        # Current Q-value
        current_q = self.get_q_value(state, action)
        
        # Target Q-value
        if done:
            target = reward
        else:
            max_next_q = max([self.get_q_value(next_state, a) 
                            for a in range(self.action_size)])
            target = reward + self.gamma * max_next_q
        
        # TD error
        td_error = target - current_q
        
        # Gradient (for linear: ∇_w Q(s,a) = φ(s,a))
        features = self.get_features(state, action)
        
        # Weight update
        self.weights += self.alpha * td_error * features
        
        return td_error ** 2  # Return squared error for logging
    
    def train(self, num_episodes: int, verbose: bool = True) -> Dict:
        """
        Train the linear Q-network.
        
        Args:
            num_episodes: Number of training episodes
            verbose: Whether to print progress
            
        Returns:
            Training statistics
        """
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0
            
            while True:
                # Select action
                action = self.get_action(state, training=True)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Update weights
                loss = self.update(state, action, reward, next_state, done)
                episode_loss += loss
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            self.epsilon_history.append(self.epsilon)
            self.loss_history.append(episode_loss / steps)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_loss = np.mean(self.loss_history[-100:])
                print(f"Episode {episode+1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Loss: {avg_loss:.4f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'loss_history': self.loss_history
        }
    
    def evaluate(self, num_episodes: int = 100) -> Dict:
        """Evaluate the trained policy."""
        eval_rewards = []
        eval_lengths = []
        dishes_created = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                action = self.get_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(steps)
            dishes_created.append(sum(self.env.dishes_created.values()))
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'mean_dishes': np.mean(dishes_created),
            'all_rewards': eval_rewards
        }


class DQNetwork(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DeepQLearning:
    """
    Deep Q-Learning with Experience Replay.
    """
    
    def __init__(self,
                 env: ChaoticChefEnvironment,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 batch_size: int = 32,
                 memory_size: int = 10000,
                 hidden_size: int = 128):
        """
        Initialize Deep Q-Learning agent.
        
        Args:
            env: Environment instance
            learning_rate: Learning rate
            discount_factor: Discount factor γ
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            batch_size: Batch size for training
            memory_size: Size of replay buffer
            hidden_size: Hidden layer size
        """
        self.env = env
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.state_size = env.get_state_size()
        self.action_size = env.get_action_size()
        self.q_network = DQNetwork(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_network = DQNetwork(self.state_size, self.action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        self.loss_history = []
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using ε-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def replay(self) -> float:
        """Train on a batch from replay buffer."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q-values
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss
        loss = self.criterion(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self, num_episodes: int, target_update_freq: int = 10, 
              verbose: bool = True) -> Dict:
        """
        Train the DQN agent.
        
        Args:
            num_episodes: Number of training episodes
            target_update_freq: Frequency to update target network
            verbose: Whether to print progress
            
        Returns:
            Training statistics
        """
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0
            loss_count = 0
            
            while True:
                # Select action
                action = self.get_action(state, training=True)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.remember(state, action, reward, next_state, done)
                
                # Train on batch
                loss = self.replay()
                if loss > 0:
                    episode_loss += loss
                    loss_count += 1
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # Update target network
            if (episode + 1) % target_update_freq == 0:
                self.update_target_network()
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            self.epsilon_history.append(self.epsilon)
            avg_loss = episode_loss / loss_count if loss_count > 0 else 0
            self.loss_history.append(avg_loss)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_loss = np.mean(self.loss_history[-100:])
                print(f"Episode {episode+1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Loss: {avg_loss:.4f}, "
                      f"Epsilon: {self.epsilon:.3f}, "
                      f"Memory: {len(self.memory)}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'loss_history': self.loss_history
        }
    
    def evaluate(self, num_episodes: int = 100) -> Dict:
        """Evaluate the trained policy."""
        eval_rewards = []
        eval_lengths = []
        dishes_created = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                action = self.get_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(steps)
            dishes_created.append(sum(self.env.dishes_created.values()))
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'mean_dishes': np.mean(dishes_created),
            'all_rewards': eval_rewards
        }


if __name__ == "__main__":
    print("Testing Function Approximation Methods\n")
    
    # Create environment
    env = ChaoticChefEnvironment(variant='basic')
    
    # Test Linear Q-Learning
    print("Training Linear Q-Learning...")
    linear_agent = LinearQNetwork(env, learning_rate=0.01)
    linear_stats = linear_agent.train(num_episodes=500, verbose=True)
    
    print("\nEvaluating Linear Q-Learning...")
    linear_eval = linear_agent.evaluate(num_episodes=100)
    print(f"Mean Reward: {linear_eval['mean_reward']:.2f} ± {linear_eval['std_reward']:.2f}")
    print(f"Mean Dishes Created: {linear_eval['mean_dishes']:.2f}")
    
    # Test Deep Q-Learning
    print("\n" + "="*50)
    print("Training Deep Q-Learning...")
    dqn_agent = DeepQLearning(env, learning_rate=0.001, batch_size=32)
    dqn_stats = dqn_agent.train(num_episodes=500, verbose=True)
    
    print("\nEvaluating Deep Q-Learning...")
    dqn_eval = dqn_agent.evaluate(num_episodes=100)
    print(f"Mean Reward: {dqn_eval['mean_reward']:.2f} ± {dqn_eval['std_reward']:.2f}")
    print(f"Mean Dishes Created: {dqn_eval['mean_dishes']:.2f}")
