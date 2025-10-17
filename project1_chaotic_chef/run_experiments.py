"""
Comprehensive Experiment Runner for Project 1
Trains all algorithms and generates comparison results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from environment import ChaoticChefEnvironment
from tabular_methods import TabularQLearning, TabularSARSA
from function_approximation import LinearQNetwork, DeepQLearning
import pickle
import json
import time
from datetime import datetime
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class ExperimentRunner:
    """Run comprehensive experiments comparing all RL methods."""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}
        
    def run_all_experiments(self, 
                           num_episodes=1000,
                           num_eval_episodes=100,
                           num_seeds=5,
                           variant='basic'):
        """
        Run all experiments with multiple seeds.
        
        Args:
            num_episodes: Training episodes per algorithm
            num_eval_episodes: Evaluation episodes
            num_seeds: Number of random seeds
            variant: Environment variant ('basic' or 'monetary')
        """
        print("="*80)
        print("CHAOTIC CHEF RL EXPERIMENTS")
        print("="*80)
        print(f"Configuration:")
        print(f"  - Training Episodes: {num_episodes}")
        print(f"  - Evaluation Episodes: {num_eval_episodes}")
        print(f"  - Random Seeds: {num_seeds}")
        print(f"  - Variant: {variant}")
        print("="*80)
        
        algorithms = [
            ('Q-Learning', TabularQLearning, {'learning_rate': 0.1}),
            ('SARSA', TabularSARSA, {'learning_rate': 0.1}),
            ('Linear-Q', LinearQNetwork, {'learning_rate': 0.01}),
            ('Deep-Q', DeepQLearning, {'learning_rate': 0.001, 'batch_size': 32})
        ]
        
        for algo_name, AgentClass, params in algorithms:
            print(f"\n{'='*80}")
            print(f"Training {algo_name}")
            print(f"{'='*80}")
            
            algo_results = {
                'training_rewards': [],
                'eval_rewards': [],
                'eval_stds': [],
                'eval_dishes': [],
                'training_times': [],
                'convergence_episodes': []
            }
            
            for seed in range(num_seeds):
                print(f"\n--- Seed {seed+1}/{num_seeds} ---")
                
                # Set random seeds
                np.random.seed(seed)
                
                # Create environment
                env = ChaoticChefEnvironment(variant=variant)
                
                # Create agent
                agent = AgentClass(env, **params)
                
                # Training
                start_time = time.time()
                
                episode_rewards = []
                for episode in range(num_episodes):
                    state = env.reset()
                    if algo_name in ['Q-Learning', 'SARSA']:
                        state = env._get_state_tuple()
                    
                    episode_reward = 0
                    
                    if algo_name == 'SARSA':
                        action = agent.get_action(state, training=True)
                    
                    while True:
                        if algo_name == 'SARSA':
                            next_state_raw, reward, done, info = env.step(action)
                            next_state = env._get_state_tuple()
                            next_action = agent.get_action(next_state, training=True)
                            agent.update(state, action, reward, next_state, next_action, done)
                            action = next_action
                        else:
                            if algo_name == 'Q-Learning':
                                action = agent.get_action(state, training=True)
                                next_state_raw, reward, done, info = env.step(action)
                                next_state = env._get_state_tuple()
                                agent.update(state, action, reward, next_state, done)
                            else:
                                action = agent.get_action(state, training=True)
                                next_state, reward, done, info = env.step(action)
                                if algo_name == 'Deep-Q':
                                    agent.remember(state, action, reward, next_state, done)
                                    agent.replay()
                                else:
                                    agent.update(state, action, reward, next_state, done)
                        
                        episode_reward += reward
                        state = next_state
                        
                        if done:
                            break
                    
                    episode_rewards.append(episode_reward)
                    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
                    
                    if algo_name == 'Deep-Q' and (episode + 1) % 10 == 0:
                        agent.update_target_network()
                    
                    if (episode + 1) % 200 == 0:
                        avg_reward = np.mean(episode_rewards[-100:])
                        print(f"  Episode {episode+1}: Avg Reward = {avg_reward:.2f}")
                
                training_time = time.time() - start_time
                
                # Find convergence episode
                moving_avg = pd.Series(episode_rewards).rolling(window=100).mean()
                final_avg = moving_avg.iloc[-100:].mean()
                threshold = final_avg * 0.9
                convergence_ep = num_episodes
                for i, val in enumerate(moving_avg):
                    if val >= threshold:
                        convergence_ep = i
                        break
                
                # Evaluation
                print("  Evaluating...")
                eval_rewards = []
                eval_dishes = []
                
                for _ in range(num_eval_episodes):
                    state = env.reset()
                    if algo_name in ['Q-Learning', 'SARSA']:
                        state = env._get_state_tuple()
                    
                    episode_reward = 0
                    
                    while True:
                        action = agent.get_action(state, training=False)
                        
                        if algo_name in ['Q-Learning', 'SARSA']:
                            next_state_raw, reward, done, info = env.step(action)
                            next_state = env._get_state_tuple()
                        else:
                            next_state, reward, done, info = env.step(action)
                        
                        episode_reward += reward
                        state = next_state
                        
                        if done:
                            break
                    
                    eval_rewards.append(episode_reward)
                    eval_dishes.append(sum(env.dishes_created.values()))
                
                # Store results
                algo_results['training_rewards'].append(episode_rewards)
                algo_results['eval_rewards'].append(np.mean(eval_rewards))
                algo_results['eval_stds'].append(np.std(eval_rewards))
                algo_results['eval_dishes'].append(np.mean(eval_dishes))
                algo_results['training_times'].append(training_time)
                algo_results['convergence_episodes'].append(convergence_ep)
                
                print(f"  Training Time: {training_time:.2f}s")
                print(f"  Eval Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
                print(f"  Eval Dishes: {np.mean(eval_dishes):.2f}")
                print(f"  Convergence: Episode {convergence_ep}")
            
            self.results[algo_name] = algo_results
            
            # Print summary
            print(f"\n{algo_name} Summary (across {num_seeds} seeds):")
            print(f"  Mean Eval Reward: {np.mean(algo_results['eval_rewards']):.2f} ± {np.std(algo_results['eval_rewards']):.2f}")
            print(f"  Mean Dishes: {np.mean(algo_results['eval_dishes']):.2f}")
            print(f"  Mean Training Time: {np.mean(algo_results['training_times']):.2f}s")
            print(f"  Mean Convergence: {np.mean(algo_results['convergence_episodes']):.0f} episodes")
        
        # Save results
        self.save_results()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "="*80)
        print("EXPERIMENTS COMPLETE!")
        print(f"Results saved to: {self.output_dir}/")
        print("="*80)
    
    def save_results(self):
        """Save results to file."""
        filepath = os.path.join(self.output_dir, 'experiment_results.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"\nResults saved to: {filepath}")
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        print("\nGenerating visualizations...")
        
        # 1. Learning Curves
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for algo_name, data in self.results.items():
            # Average across seeds
            all_rewards = np.array(data['training_rewards'])
            mean_rewards = np.mean(all_rewards, axis=0)
            std_rewards = np.std(all_rewards, axis=0)
            
            # Moving average
            window = 50
            moving_avg = pd.Series(mean_rewards).rolling(window=window).mean()
            moving_std = pd.Series(std_rewards).rolling(window=window).mean()
            
            # Plot
            episodes = np.arange(len(mean_rewards))
            axes[0, 0].plot(episodes, moving_avg, label=algo_name, linewidth=2)
            axes[0, 0].fill_between(episodes, 
                                     moving_avg - moving_std,
                                     moving_avg + moving_std,
                                     alpha=0.2)
        
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].set_title('Learning Curves (Moving Average)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Evaluation Performance
        algo_names = list(self.results.keys())
        eval_means = [np.mean(self.results[name]['eval_rewards']) for name in algo_names]
        eval_stds = [np.std(self.results[name]['eval_rewards']) for name in algo_names]
        
        axes[0, 1].bar(algo_names, eval_means, yerr=eval_stds, capsize=5, alpha=0.7)
        axes[0, 1].set_ylabel('Mean Evaluation Reward')
        axes[0, 1].set_title('Evaluation Performance Comparison')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Dishes Created
        dishes_means = [np.mean(self.results[name]['eval_dishes']) for name in algo_names]
        dishes_stds = [np.std(self.results[name]['eval_dishes']) for name in algo_names]
        
        axes[1, 0].bar(algo_names, dishes_means, yerr=dishes_stds, capsize=5, 
                      alpha=0.7, color='orange')
        axes[1, 0].set_ylabel('Mean Dishes Created')
        axes[1, 0].set_title('Dishes Created per Episode')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Training Time
        time_means = [np.mean(self.results[name]['training_times']) for name in algo_names]
        time_stds = [np.std(self.results[name]['training_times']) for name in algo_names]
        
        axes[1, 1].bar(algo_names, time_means, yerr=time_stds, capsize=5, 
                      alpha=0.7, color='green')
        axes[1, 1].set_ylabel('Training Time (seconds)')
        axes[1, 1].set_title('Computational Efficiency')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'comparison_plots.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
        
        # 5. Convergence Analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        conv_means = [np.mean(self.results[name]['convergence_episodes']) for name in algo_names]
        conv_stds = [np.std(self.results[name]['convergence_episodes']) for name in algo_names]
        
        ax.bar(algo_names, conv_means, yerr=conv_stds, capsize=5, alpha=0.7, color='purple')
        ax.set_ylabel('Episodes to Convergence')
        ax.set_title('Convergence Speed Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'convergence_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    
    def generate_report(self):
        """Generate text report."""
        print("\nGenerating report...")
        
        report_path = os.path.join(self.output_dir, 'experiment_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CHAOTIC CHEF RL EXPERIMENTS - COMPREHENSIVE REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Algorithm':<15} {'Eval Reward':<20} {'Dishes':<15} {'Time (s)':<15} {'Convergence':<15}\n")
            f.write("-"*80 + "\n")
            
            for algo_name in self.results.keys():
                data = self.results[algo_name]
                reward_mean = np.mean(data['eval_rewards'])
                reward_std = np.std(data['eval_rewards'])
                dishes_mean = np.mean(data['eval_dishes'])
                time_mean = np.mean(data['training_times'])
                conv_mean = np.mean(data['convergence_episodes'])
                
                f.write(f"{algo_name:<15} {reward_mean:>7.2f} ± {reward_std:<7.2f} "
                       f"{dishes_mean:>7.2f}       {time_mean:>7.2f}       {conv_mean:>7.0f}\n")
            
            f.write("-"*80 + "\n\n")
            
            # Best performers
            f.write("BEST PERFORMERS\n")
            f.write("-"*80 + "\n")
            
            best_reward = max(self.results.items(), 
                            key=lambda x: np.mean(x[1]['eval_rewards']))
            f.write(f"Highest Reward: {best_reward[0]} ({np.mean(best_reward[1]['eval_rewards']):.2f})\n")
            
            best_dishes = max(self.results.items(),
                            key=lambda x: np.mean(x[1]['eval_dishes']))
            f.write(f"Most Dishes: {best_dishes[0]} ({np.mean(best_dishes[1]['eval_dishes']):.2f})\n")
            
            fastest = min(self.results.items(),
                         key=lambda x: np.mean(x[1]['training_times']))
            f.write(f"Fastest Training: {fastest[0]} ({np.mean(fastest[1]['training_times']):.2f}s)\n")
            
            fastest_conv = min(self.results.items(),
                              key=lambda x: np.mean(x[1]['convergence_episodes']))
            f.write(f"Fastest Convergence: {fastest_conv[0]} ({np.mean(fastest_conv[1]['convergence_episodes']):.0f} episodes)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Saved: {report_path}")
        
        # Also create CSV
        csv_path = os.path.join(self.output_dir, 'results_summary.csv')
        
        summary_data = []
        for algo_name in self.results.keys():
            data = self.results[algo_name]
            summary_data.append({
                'Algorithm': algo_name,
                'Mean_Reward': np.mean(data['eval_rewards']),
                'Std_Reward': np.std(data['eval_rewards']),
                'Mean_Dishes': np.mean(data['eval_dishes']),
                'Mean_Time': np.mean(data['training_times']),
                'Mean_Convergence': np.mean(data['convergence_episodes'])
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    # Run experiments
    runner = ExperimentRunner(output_dir='results')
    
    print("\nStarting comprehensive experiments...")
    print("This will take several minutes...\n")
    
    runner.run_all_experiments(
        num_episodes=1000,
        num_eval_episodes=100,
        num_seeds=5,
        variant='basic'
    )
    
    print("\n✅ All experiments completed successfully!")
    print("\nGenerated files:")
    print("  - results/experiment_results.pkl")
    print("  - results/comparison_plots.png")
    print("  - results/convergence_analysis.png")
    print("  - results/experiment_report.txt")
    print("  - results/results_summary.csv")
