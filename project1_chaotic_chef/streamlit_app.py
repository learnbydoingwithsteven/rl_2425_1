"""
Streamlit Dashboard for The Chaotic Chef's Quest for the Perfect Meal
Interactive visualization of training progress and results comparison
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from environment import ChaoticChefEnvironment
from tabular_methods import TabularQLearning, TabularSARSA
from function_approximation import LinearQNetwork, DeepQLearning
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Chaotic Chef RL Dashboard",
    page_icon="👨‍🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">👨‍🍳 The Chaotic Chef\'s Quest for the Perfect Meal</h1>', 
            unsafe_allow_html=True)
st.markdown("### Comparing Tabular vs Function Approximation RL Methods")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Environment settings
st.sidebar.subheader("Environment Settings")
variant = st.sidebar.selectbox("Environment Variant", ["basic", "monetary"])
grid_size = st.sidebar.slider("Grid Size", 3, 7, 5)
max_steps = st.sidebar.slider("Max Steps per Episode", 50, 200, 100)

# Training settings
st.sidebar.subheader("Training Settings")
num_episodes = st.sidebar.slider("Number of Episodes", 100, 2000, 1000, step=100)
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.5, 0.1, step=0.001, format="%.3f")
discount_factor = st.sidebar.slider("Discount Factor (γ)", 0.9, 0.999, 0.99, step=0.001)
epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.99, 0.999, 0.995, step=0.001)

# Algorithm selection
st.sidebar.subheader("Algorithms to Train")
train_qlearning = st.sidebar.checkbox("Q-Learning (Tabular)", value=True)
train_sarsa = st.sidebar.checkbox("SARSA (Tabular)", value=True)
train_linear = st.sidebar.checkbox("Linear Q-Learning", value=True)
train_dqn = st.sidebar.checkbox("Deep Q-Learning", value=True)

# Initialize session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Training", "📊 Learning Curves", "📈 Performance Comparison", 
    "🎮 Live Demo", "📝 Analysis"
])

# Tab 1: Training
with tab1:
    st.header("Training Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            st.session_state.trained = False
            st.session_state.results = {}
            
            # Create environment
            env = ChaoticChefEnvironment(
                grid_size=grid_size, 
                max_steps=max_steps, 
                variant=variant
            )
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            algorithms_to_train = []
            if train_qlearning:
                algorithms_to_train.append(('Q-Learning', TabularQLearning))
            if train_sarsa:
                algorithms_to_train.append(('SARSA', TabularSARSA))
            if train_linear:
                algorithms_to_train.append(('Linear Q', LinearQNetwork))
            if train_dqn:
                algorithms_to_train.append(('Deep Q', DeepQLearning))
            
            total_algorithms = len(algorithms_to_train)
            
            for idx, (name, AgentClass) in enumerate(algorithms_to_train):
                status_text.markdown(f"### Training {name}... ({idx+1}/{total_algorithms})")
                
                # Create agent
                if name in ['Q-Learning', 'SARSA']:
                    agent = AgentClass(
                        env,
                        learning_rate=learning_rate,
                        discount_factor=discount_factor,
                        epsilon_decay=epsilon_decay
                    )
                elif name == 'Linear Q':
                    agent = AgentClass(
                        env,
                        learning_rate=learning_rate * 0.1,  # Lower LR for linear
                        discount_factor=discount_factor,
                        epsilon_decay=epsilon_decay
                    )
                else:  # Deep Q
                    agent = AgentClass(
                        env,
                        learning_rate=learning_rate * 0.01,  # Lower LR for DQN
                        discount_factor=discount_factor,
                        epsilon_decay=epsilon_decay,
                        batch_size=32
                    )
                
                # Training progress container
                progress_container = st.container()
                with progress_container:
                    chart_placeholder = st.empty()
                    metrics_placeholder = st.empty()
                
                # Train with live updates
                episode_rewards = []
                update_freq = max(1, num_episodes // 20)  # Update 20 times
                
                for episode in range(num_episodes):
                    state = env.reset()
                    if name in ['Q-Learning', 'SARSA']:
                        state = env._get_state_tuple()
                    
                    episode_reward = 0
                    
                    if name == 'SARSA':
                        action = agent.get_action(state, training=True)
                    
                    while True:
                        if name == 'SARSA':
                            next_state_raw, reward, done, info = env.step(action)
                            next_state = env._get_state_tuple()
                            next_action = agent.get_action(next_state, training=True)
                            agent.update(state, action, reward, next_state, next_action, done)
                            action = next_action
                        else:
                            if name in ['Q-Learning']:
                                action = agent.get_action(state, training=True)
                                next_state_raw, reward, done, info = env.step(action)
                                next_state = env._get_state_tuple()
                                agent.update(state, action, reward, next_state, done)
                            else:
                                action = agent.get_action(state, training=True)
                                next_state, reward, done, info = env.step(action)
                                if name == 'Deep Q':
                                    agent.remember(state, action, reward, next_state, done)
                                    agent.replay()
                                else:
                                    agent.update(state, action, reward, next_state, done)
                        
                        episode_reward += reward
                        state = next_state
                        
                        if done:
                            break
                    
                    episode_rewards.append(episode_reward)
                    
                    # Update epsilon
                    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
                    
                    # Update target network for DQN
                    if name == 'Deep Q' and (episode + 1) % 10 == 0:
                        agent.update_target_network()
                    
                    # Live visualization update
                    if (episode + 1) % update_freq == 0 or episode == num_episodes - 1:
                        # Create live chart
                        window = min(100, len(episode_rewards))
                        moving_avg = pd.Series(episode_rewards).rolling(window=window).mean()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=episode_rewards,
                            mode='lines',
                            name='Episode Reward',
                            line=dict(color='lightblue', width=1),
                            opacity=0.3
                        ))
                        fig.add_trace(go.Scatter(
                            y=moving_avg,
                            mode='lines',
                            name=f'Moving Avg ({window})',
                            line=dict(color='blue', width=3)
                        ))
                        fig.update_layout(
                            title=f"{name} Training Progress",
                            xaxis_title="Episode",
                            yaxis_title="Reward",
                            height=300,
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                        
                        # Show metrics
                        recent_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                        metrics_placeholder.metric(
                            f"Average Reward (last 100)",
                            f"{recent_avg:.2f}",
                            delta=f"Episode {episode+1}/{num_episodes}"
                        )
                
                # Evaluation
                status_text.markdown(f"### Evaluating {name}...")
                eval_results = agent.evaluate(num_episodes=100)
                
                # Store results
                st.session_state.results[name] = {
                    'agent': agent,
                    'training_rewards': episode_rewards,
                    'eval_results': eval_results
                }
                
                # Update progress
                progress_bar.progress((idx + 1) / total_algorithms)
            
            status_text.markdown("### ✅ Training Complete!")
            st.session_state.trained = True
            st.balloons()
    
    with col2:
        st.subheader("Environment Preview")
        
        # Show environment grid
        env_preview = ChaoticChefEnvironment(grid_size=grid_size, variant=variant)
        env_preview.reset()
        
        st.text(env_preview.render())
        
        st.subheader("Market Locations")
        for ingredient, pos in env_preview.markets.items():
            st.text(f"🏪 {ingredient.capitalize()}: {pos}")
        
        st.subheader("Dishes")
        for dish, ingredients in env_preview.dishes.items():
            ing_str = ", ".join([f"{v}x {k}" for k, v in ingredients.items()])
            reward = env_preview.dish_rewards[dish]
            st.text(f"🍽️ {dish.capitalize()}: {ing_str} → {reward} pts")

# Tab 2: Learning Curves
with tab2:
    st.header("Learning Curves Comparison")
    
    if st.session_state.trained:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Episode Rewards', 'Moving Average (100 episodes)', 
                          'Epsilon Decay', 'Q-Value Evolution'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        colors = {
            'Q-Learning': '#FF6B6B',
            'SARSA': '#4ECDC4',
            'Linear Q': '#45B7D1',
            'Deep Q': '#FFA07A'
        }
        
        for name, data in st.session_state.results.items():
            rewards = data['training_rewards']
            agent = data['agent']
            color = colors.get(name, '#888888')
            
            # Episode rewards (transparent)
            fig.add_trace(
                go.Scatter(y=rewards, mode='lines', name=f'{name}',
                          line=dict(color=color, width=1), opacity=0.3,
                          showlegend=True),
                row=1, col=1
            )
            
            # Moving average
            moving_avg = pd.Series(rewards).rolling(window=100).mean()
            fig.add_trace(
                go.Scatter(y=moving_avg, mode='lines', name=f'{name} (MA)',
                          line=dict(color=color, width=3),
                          showlegend=False),
                row=1, col=2
            )
            
            # Epsilon decay
            if hasattr(agent, 'epsilon_history'):
                fig.add_trace(
                    go.Scatter(y=agent.epsilon_history, mode='lines', name=name,
                              line=dict(color=color, width=2),
                              showlegend=False),
                    row=2, col=1
                )
            
            # Q-value evolution
            if hasattr(agent, 'q_value_history'):
                fig.add_trace(
                    go.Scatter(y=agent.q_value_history, mode='lines', name=name,
                              line=dict(color=color, width=2),
                              showlegend=False),
                    row=2, col=2
                )
        
        fig.update_xaxes(title_text="Episode", row=1, col=1)
        fig.update_xaxes(title_text="Episode", row=1, col=2)
        fig.update_xaxes(title_text="Episode", row=2, col=1)
        fig.update_xaxes(title_text="Episode", row=2, col=2)
        
        fig.update_yaxes(title_text="Reward", row=1, col=1)
        fig.update_yaxes(title_text="Reward", row=1, col=2)
        fig.update_yaxes(title_text="Epsilon", row=2, col=1)
        fig.update_yaxes(title_text="Avg Q-Value", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("👆 Train the algorithms first to see learning curves!")

# Tab 3: Performance Comparison
with tab3:
    st.header("Performance Comparison")
    
    if st.session_state.trained:
        # Create comparison dataframe
        comparison_data = []
        for name, data in st.session_state.results.items():
            eval_results = data['eval_results']
            comparison_data.append({
                'Algorithm': name,
                'Mean Reward': eval_results['mean_reward'],
                'Std Reward': eval_results['std_reward'],
                'Mean Episode Length': eval_results['mean_length'],
                'Mean Dishes Created': eval_results['mean_dishes']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Display table
        st.subheader("📊 Evaluation Results (100 episodes)")
        st.dataframe(df_comparison.style.highlight_max(axis=0, subset=['Mean Reward', 'Mean Dishes Created'])
                                       .highlight_min(axis=0, subset=['Std Reward', 'Mean Episode Length'])
                                       .format({
                                           'Mean Reward': '{:.2f}',
                                           'Std Reward': '{:.2f}',
                                           'Mean Episode Length': '{:.1f}',
                                           'Mean Dishes Created': '{:.2f}'
                                       }), use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart: Mean Reward
            fig1 = px.bar(df_comparison, x='Algorithm', y='Mean Reward',
                         error_y='Std Reward',
                         title='Mean Reward Comparison',
                         color='Algorithm',
                         color_discrete_map=colors)
            fig1.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Bar chart: Mean Dishes Created
            fig2 = px.bar(df_comparison, x='Algorithm', y='Mean Dishes Created',
                         title='Mean Dishes Created',
                         color='Algorithm',
                         color_discrete_map=colors)
            fig2.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Reward distribution
        st.subheader("📈 Reward Distribution")
        fig3 = go.Figure()
        for name, data in st.session_state.results.items():
            eval_rewards = data['eval_results']['all_rewards']
            fig3.add_trace(go.Box(y=eval_rewards, name=name,
                                 marker_color=colors.get(name, '#888888')))
        fig3.update_layout(title='Reward Distribution (100 evaluation episodes)',
                          yaxis_title='Reward', height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Statistical analysis
        st.subheader("📉 Statistical Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        best_mean = df_comparison.loc[df_comparison['Mean Reward'].idxmax()]
        most_stable = df_comparison.loc[df_comparison['Std Reward'].idxmin()]
        most_efficient = df_comparison.loc[df_comparison['Mean Episode Length'].idxmin()]
        
        with col1:
            st.metric("🏆 Best Mean Reward", 
                     best_mean['Algorithm'],
                     f"{best_mean['Mean Reward']:.2f}")
        
        with col2:
            st.metric("🎯 Most Stable", 
                     most_stable['Algorithm'],
                     f"σ = {most_stable['Std Reward']:.2f}")
        
        with col3:
            st.metric("⚡ Most Efficient", 
                     most_efficient['Algorithm'],
                     f"{most_efficient['Mean Episode Length']:.1f} steps")
        
    else:
        st.info("👆 Train the algorithms first to see performance comparison!")

# Tab 4: Live Demo
with tab4:
    st.header("Live Agent Demo")
    
    if st.session_state.trained:
        # Select algorithm
        algorithm_name = st.selectbox("Select Algorithm", list(st.session_state.results.keys()))
        
        if st.button("▶️ Run Episode", use_container_width=True):
            agent = st.session_state.results[algorithm_name]['agent']
            env = ChaoticChefEnvironment(grid_size=grid_size, max_steps=max_steps, variant=variant)
            
            state = env.reset()
            if algorithm_name in ['Q-Learning', 'SARSA']:
                state = env._get_state_tuple()
            
            # Create placeholders
            grid_placeholder = st.empty()
            metrics_placeholder = st.empty()
            trajectory_data = []
            
            episode_reward = 0
            step = 0
            
            while True:
                # Get action
                action = agent.get_action(state, training=False)
                
                # Take step
                if algorithm_name in ['Q-Learning', 'SARSA']:
                    next_state_raw, reward, done, info = env.step(action)
                    next_state = env._get_state_tuple()
                else:
                    next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                step += 1
                
                # Record trajectory
                trajectory_data.append({
                    'step': step,
                    'action': env.ACTION_NAMES[action],
                    'reward': reward,
                    'total_reward': episode_reward
                })
                
                # Display current state
                with grid_placeholder.container():
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.text(env.render())
                    
                    with col2:
                        st.markdown(f"**Step:** {step}")
                        st.markdown(f"**Action:** {env.ACTION_NAMES[action]}")
                        st.markdown(f"**Reward:** {reward:.2f}")
                        st.markdown(f"**Total Reward:** {episode_reward:.2f}")
                        
                        if 'collected' in info:
                            st.success(f"✅ Collected: {info['collected']}")
                        if 'dish_created' in info:
                            st.success(f"🍽️ Created: {info['dish_created']}")
                
                state = next_state
                time.sleep(0.3)  # Slow down for visualization
                
                if done:
                    break
            
            # Show trajectory
            st.subheader("Episode Trajectory")
            df_trajectory = pd.DataFrame(trajectory_data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_trajectory['step'], 
                                    y=df_trajectory['total_reward'],
                                    mode='lines+markers',
                                    name='Cumulative Reward',
                                    line=dict(color='blue', width=3)))
            fig.update_layout(title='Cumulative Reward Over Time',
                            xaxis_title='Step',
                            yaxis_title='Cumulative Reward',
                            height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_trajectory, use_container_width=True)
    
    else:
        st.info("👆 Train the algorithms first to run live demos!")

# Tab 5: Analysis
with tab5:
    st.header("Analysis & Insights")
    
    if st.session_state.trained:
        st.subheader("🔍 Key Findings")
        
        # Compare tabular vs function approximation
        tabular_methods = [name for name in st.session_state.results.keys() 
                          if name in ['Q-Learning', 'SARSA']]
        approx_methods = [name for name in st.session_state.results.keys() 
                         if name in ['Linear Q', 'Deep Q']]
        
        if tabular_methods and approx_methods:
            tabular_avg = np.mean([st.session_state.results[name]['eval_results']['mean_reward'] 
                                  for name in tabular_methods])
            approx_avg = np.mean([st.session_state.results[name]['eval_results']['mean_reward'] 
                                 for name in approx_methods])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Tabular Methods Avg", f"{tabular_avg:.2f}")
            with col2:
                st.metric("🧮 Function Approximation Avg", f"{approx_avg:.2f}")
            
            if tabular_avg > approx_avg:
                st.success("✅ Tabular methods performed better on average!")
                st.write("""
                **Possible reasons:**
                - State space is relatively small and discrete
                - Tabular methods can represent exact Q-values
                - Function approximation may need more training
                """)
            else:
                st.success("✅ Function approximation performed better on average!")
                st.write("""
                **Possible reasons:**
                - Better generalization across similar states
                - More efficient learning with limited experience
                - Better handling of state space complexity
                """)
        
        # Q-Learning vs SARSA
        if 'Q-Learning' in st.session_state.results and 'SARSA' in st.session_state.results:
            st.subheader("🆚 Q-Learning vs SARSA")
            
            q_reward = st.session_state.results['Q-Learning']['eval_results']['mean_reward']
            sarsa_reward = st.session_state.results['SARSA']['eval_results']['mean_reward']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Q-Learning (Off-Policy)", f"{q_reward:.2f}")
            with col2:
                st.metric("SARSA (On-Policy)", f"{sarsa_reward:.2f}")
            
            st.write("""
            **Key Differences:**
            - **Q-Learning**: Off-policy, learns optimal policy regardless of behavior
            - **SARSA**: On-policy, learns value of policy being followed
            - Q-Learning typically more aggressive, SARSA more conservative
            """)
        
        # Learning efficiency
        st.subheader("📈 Learning Efficiency")
        
        for name, data in st.session_state.results.items():
            rewards = data['training_rewards']
            
            # Calculate convergence point (when moving average stabilizes)
            window = 100
            moving_avg = pd.Series(rewards).rolling(window=window).mean()
            final_avg = moving_avg.iloc[-100:].mean()
            
            # Find when it reached 90% of final performance
            threshold = final_avg * 0.9
            convergence_episode = None
            for i, val in enumerate(moving_avg):
                if val >= threshold:
                    convergence_episode = i
                    break
            
            if convergence_episode:
                st.write(f"**{name}**: Converged at episode ~{convergence_episode} "
                        f"(reached 90% of final performance)")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        best_algorithm = max(st.session_state.results.items(), 
                           key=lambda x: x[1]['eval_results']['mean_reward'])[0]
        
        st.success(f"**Best Algorithm for this task: {best_algorithm}**")
        
        st.write("""
        **General Recommendations:**
        1. **For small state spaces**: Use tabular methods (Q-Learning/SARSA)
        2. **For large/continuous spaces**: Use function approximation (Linear/Deep Q)
        3. **For safety-critical applications**: Consider SARSA (on-policy)
        4. **For optimal performance**: Consider Q-Learning (off-policy)
        5. **For fast learning**: Deep Q-Learning with experience replay
        
        **Hyperparameter Tuning:**
        - Increase learning rate for faster convergence (but less stability)
        - Increase discount factor for long-term planning
        - Adjust epsilon decay for exploration-exploitation balance
        """)
        
    else:
        st.info("👆 Train the algorithms first to see analysis!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Project 1: The Chaotic Chef's Quest for the Perfect Meal</p>
    <p>Reinforcement Learning Course 2024-2025</p>
</div>
""", unsafe_allow_html=True)
