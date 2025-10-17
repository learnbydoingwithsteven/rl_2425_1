"""
Tabular RL Methods: Q-Learning and SARSA
For The Chaotic Chef's Quest for the Perfect Meal
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import pickle
from environment import ChaoticChefEnvironment


class TabularQLearning:
    """
    Q-Learning: Off-policy TD control algorithm.
    Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
    """
    
    def __init__(self, 
                 env: ChaoticChefEnvironment,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent.
        
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
        
        # Q-table: Q[state][action] = value
        self.Q = defaultdict(lambda: np.zeros(env.get_action_size()))
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        self.q_value_history = []
        
    def get_action(self, state: Tuple, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state tuple
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.env.get_action_size())
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state: Tuple, action: int, reward: float, 
               next_state: Tuple, done: bool):
        """
        Update Q-value using Q-learning update rule.
        
        Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
        """
        current_q = self.Q[state][action]
        
        if done:
            target = reward
        else:
            max_next_q = np.max(self.Q[next_state])
            target = reward + self.gamma * max_next_q
        
        # Q-learning update
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
    
    def train(self, num_episodes: int, verbose: bool = True) -> Dict:
        """
        Train the Q-learning agent.
        
        Args:
            num_episodes: Number of training episodes
            verbose: Whether to print progress
            
        Returns:
            Training statistics
        """
        for episode in range(num_episodes):
            state = self.env.reset()
            state_tuple = self.env._get_state_tuple()
            episode_reward = 0
            steps = 0
            
            while True:
                # Select action
                action = self.get_action(state_tuple, training=True)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                next_state_tuple = self.env._get_state_tuple()
                
                # Update Q-value
                self.update(state_tuple, action, reward, next_state_tuple, done)
                
                episode_reward += reward
                steps += 1
                
                state_tuple = next_state_tuple
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            self.epsilon_history.append(self.epsilon)
            
            # Record average Q-value
            avg_q = np.mean([np.max(q_vals) for q_vals in self.Q.values()])
            self.q_value_history.append(avg_q)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode+1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}, "
                      f"Avg Q: {avg_q:.2f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'q_value_history': self.q_value_history,
            'final_epsilon': self.epsilon,
            'q_table_size': len(self.Q)
        }
    
    def evaluate(self, num_episodes: int = 100) -> Dict:
        """
        Evaluate the trained policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation statistics
        """
        eval_rewards = []
        eval_lengths = []
        dishes_created = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            state_tuple = self.env._get_state_tuple()
            episode_reward = 0
            steps = 0
            
            while True:
                action = self.get_action(state_tuple, training=False)
                next_state, reward, done, info = self.env.step(action)
                next_state_tuple = self.env._get_state_tuple()
                
                episode_reward += reward
                steps += 1
                state_tuple = next_state_tuple
                
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
    
    def save(self, filepath: str):
        """Save the Q-table and training history."""
        data = {
            'Q': dict(self.Q),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'q_value_history': self.q_value_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load the Q-table and training history."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.Q = defaultdict(lambda: np.zeros(self.env.get_action_size()), data['Q'])
        self.episode_rewards = data['episode_rewards']
        self.episode_lengths = data['episode_lengths']
        self.epsilon_history = data['epsilon_history']
        self.q_value_history = data['q_value_history']


class TabularSARSA:
    """
    SARSA: On-policy TD control algorithm.
    Q(S,A) ← Q(S,A) + α[R + γ Q(S',A') - Q(S,A)]
    """
    
    def __init__(self, 
                 env: ChaoticChefEnvironment,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize SARSA agent.
        
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
        
        # Q-table: Q[state][action] = value
        self.Q = defaultdict(lambda: np.zeros(env.get_action_size()))
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        self.q_value_history = []
        
    def get_action(self, state: Tuple, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state tuple
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.env.get_action_size())
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state: Tuple, action: int, reward: float, 
               next_state: Tuple, next_action: int, done: bool):
        """
        Update Q-value using SARSA update rule.
        
        Q(S,A) ← Q(S,A) + α[R + γ Q(S',A') - Q(S,A)]
        """
        current_q = self.Q[state][action]
        
        if done:
            target = reward
        else:
            next_q = self.Q[next_state][next_action]
            target = reward + self.gamma * next_q
        
        # SARSA update
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
    
    def train(self, num_episodes: int, verbose: bool = True) -> Dict:
        """
        Train the SARSA agent.
        
        Args:
            num_episodes: Number of training episodes
            verbose: Whether to print progress
            
        Returns:
            Training statistics
        """
        for episode in range(num_episodes):
            state = self.env.reset()
            state_tuple = self.env._get_state_tuple()
            action = self.get_action(state_tuple, training=True)
            
            episode_reward = 0
            steps = 0
            
            while True:
                # Take action
                next_state, reward, done, info = self.env.step(action)
                next_state_tuple = self.env._get_state_tuple()
                
                # Select next action
                next_action = self.get_action(next_state_tuple, training=True)
                
                # Update Q-value
                self.update(state_tuple, action, reward, next_state_tuple, 
                          next_action, done)
                
                episode_reward += reward
                steps += 1
                
                state_tuple = next_state_tuple
                action = next_action
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            self.epsilon_history.append(self.epsilon)
            
            # Record average Q-value
            avg_q = np.mean([np.max(q_vals) for q_vals in self.Q.values()])
            self.q_value_history.append(avg_q)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode+1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}, "
                      f"Avg Q: {avg_q:.2f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'q_value_history': self.q_value_history,
            'final_epsilon': self.epsilon,
            'q_table_size': len(self.Q)
        }
    
    def evaluate(self, num_episodes: int = 100) -> Dict:
        """
        Evaluate the trained policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation statistics
        """
        eval_rewards = []
        eval_lengths = []
        dishes_created = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            state_tuple = self.env._get_state_tuple()
            episode_reward = 0
            steps = 0
            
            while True:
                action = self.get_action(state_tuple, training=False)
                next_state, reward, done, info = self.env.step(action)
                next_state_tuple = self.env._get_state_tuple()
                
                episode_reward += reward
                steps += 1
                state_tuple = next_state_tuple
                
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
    
    def save(self, filepath: str):
        """Save the Q-table and training history."""
        data = {
            'Q': dict(self.Q),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'q_value_history': self.q_value_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load the Q-table and training history."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.Q = defaultdict(lambda: np.zeros(self.env.get_action_size()), data['Q'])
        self.episode_rewards = data['episode_rewards']
        self.episode_lengths = data['episode_lengths']
        self.epsilon_history = data['epsilon_history']
        self.q_value_history = data['q_value_history']


if __name__ == "__main__":
    # Test tabular methods
    print("Testing Tabular RL Methods\n")
    
    # Create environment
    env = ChaoticChefEnvironment(variant='basic')
    
    # Test Q-Learning
    print("Training Q-Learning...")
    q_agent = TabularQLearning(env, learning_rate=0.1, epsilon_decay=0.995)
    q_stats = q_agent.train(num_episodes=500, verbose=True)
    
    print("\nEvaluating Q-Learning...")
    q_eval = q_agent.evaluate(num_episodes=100)
    print(f"Mean Reward: {q_eval['mean_reward']:.2f} ± {q_eval['std_reward']:.2f}")
    print(f"Mean Dishes Created: {q_eval['mean_dishes']:.2f}")
    
    # Test SARSA
    print("\n" + "="*50)
    print("Training SARSA...")
    sarsa_agent = TabularSARSA(env, learning_rate=0.1, epsilon_decay=0.995)
    sarsa_stats = sarsa_agent.train(num_episodes=500, verbose=True)
    
    print("\nEvaluating SARSA...")
    sarsa_eval = sarsa_agent.evaluate(num_episodes=100)
    print(f"Mean Reward: {sarsa_eval['mean_reward']:.2f} ± {sarsa_eval['std_reward']:.2f}")
    print(f"Mean Dishes Created: {sarsa_eval['mean_dishes']:.2f}")
