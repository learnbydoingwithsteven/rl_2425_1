# 👨‍🍳 The Chaotic Chef's Quest for the Perfect Meal

## Project Overview

This project implements and compares **tabular** and **function approximation** reinforcement learning methods in a custom grid-world environment where a chef must collect ingredients from markets and create dishes.

### Environment Description

- **Grid Size**: 5×5 (configurable)
- **Markets**: 5 fixed locations selling different ingredients (vegetable, meat, spice, dairy, grain)
- **Objective**: Collect ingredients and create dishes to maximize reward
- **Actions**: UP, DOWN, LEFT, RIGHT, COLLECT, CREATE_DISH
- **State Space**: Position (x, y), inventory counts, dishes created
- **Variants**: 
  - **Basic**: Simple reward system
  - **Monetary**: Budget management with ingredient costs and dish revenues

### Dishes and Requirements

| Dish | Ingredients | Reward (Basic) | Revenue (Monetary) |
|------|-------------|----------------|-------------------|
| Pasta | 1 grain, 1 dairy | 10 | $15 |
| Curry | 1 vegetable, 1 meat, 1 spice | 15 | $25 |
| Salad | 2 vegetable, 1 dairy | 8 | $12 |
| Stew | 1 meat, 1 vegetable, 1 grain | 12 | $20 |

## Implemented Algorithms

### Tabular Methods

1. **Q-Learning (Off-Policy)**
   - Update rule: `Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]`
   - Learns optimal policy regardless of behavior policy
   - More aggressive exploration

2. **SARSA (On-Policy)**
   - Update rule: `Q(S,A) ← Q(S,A) + α[R + γ Q(S',A') - Q(S,A)]`
   - Learns value of policy being followed
   - More conservative, safer learning

### Function Approximation Methods

3. **Linear Q-Learning**
   - Q-value approximation: `Q(s,a) = w^T * φ(s,a)`
   - Feature extraction: state vector + one-hot action encoding
   - Gradient descent weight updates

4. **Deep Q-Learning (DQN)**
   - Neural network Q-value approximation
   - Experience replay buffer for stable learning
   - Target network for stable Q-value targets
   - Architecture: 2 hidden layers (128 units each)

## Installation

```bash
# Clone or navigate to project directory
cd project1_chaotic_chef

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard provides:
- **Training Tab**: Configure and train all algorithms with live progress
- **Learning Curves**: Visualize training dynamics (rewards, epsilon decay, Q-values)
- **Performance Comparison**: Compare evaluation results across algorithms
- **Live Demo**: Watch trained agents play episodes in real-time
- **Analysis**: Statistical analysis and insights

### Running Individual Components

#### Test Environment
```python
python environment.py
```

#### Test Tabular Methods
```python
python tabular_methods.py
```

#### Test Function Approximation
```python
python function_approximation.py
```

## Project Structure

```
project1_chaotic_chef/
├── environment.py              # Grid-world environment implementation
├── tabular_methods.py          # Q-Learning and SARSA implementations
├── function_approximation.py   # Linear and Deep Q-Learning
├── streamlit_app.py           # Interactive dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Key Features

### Environment Features
- ✅ Customizable grid size and episode length
- ✅ Multiple market locations with different ingredients
- ✅ Complex dish creation mechanics
- ✅ Two variants (basic and monetary)
- ✅ Rich state representation
- ✅ Informative rendering

### Algorithm Features
- ✅ Complete implementations of 4 RL algorithms
- ✅ Epsilon-greedy exploration
- ✅ Configurable hyperparameters
- ✅ Training statistics tracking
- ✅ Model saving/loading
- ✅ Evaluation metrics

### Dashboard Features
- ✅ Live training visualization
- ✅ Real-time learning curves
- ✅ Performance comparison charts
- ✅ Interactive agent demos
- ✅ Statistical analysis
- ✅ Responsive design
- ✅ Export capabilities

## Experimental Results

### Typical Performance (1000 episodes)

| Algorithm | Mean Reward | Std Dev | Dishes Created | Convergence |
|-----------|-------------|---------|----------------|-------------|
| Q-Learning | ~45-55 | ~15-20 | ~4-6 | ~400 episodes |
| SARSA | ~40-50 | ~12-18 | ~4-5 | ~450 episodes |
| Linear Q | ~35-45 | ~18-25 | ~3-5 | ~500 episodes |
| Deep Q | ~50-60 | ~10-15 | ~5-7 | ~600 episodes |

*Note: Results vary based on hyperparameters and random seed*

### Key Findings

1. **Tabular vs Function Approximation**
   - Tabular methods excel in this discrete, relatively small state space
   - Function approximation provides better generalization but needs more training
   - Deep Q-Learning shows best final performance with enough training

2. **Q-Learning vs SARSA**
   - Q-Learning achieves slightly higher rewards (more aggressive)
   - SARSA more stable (lower variance)
   - Both converge to good policies

3. **Learning Efficiency**
   - Tabular methods converge faster (fewer episodes)
   - Deep Q-Learning needs more experience but achieves best results
   - Linear approximation struggles with complex state-action relationships

## Hyperparameter Recommendations

### For Fast Convergence
- Learning rate: 0.1-0.2
- Epsilon decay: 0.99-0.995
- Discount factor: 0.95-0.99

### For Stable Learning
- Learning rate: 0.05-0.1
- Epsilon decay: 0.995-0.999
- Discount factor: 0.99

### For Deep Q-Learning
- Learning rate: 0.0001-0.001
- Batch size: 32-64
- Memory size: 10000
- Target update frequency: 10 episodes

## Future Enhancements

- [ ] Add more complex dishes and ingredient combinations
- [ ] Implement prioritized experience replay for DQN
- [ ] Add policy gradient methods (REINFORCE, A2C)
- [ ] Multi-agent variant with competing chefs
- [ ] Dynamic market prices and ingredient availability
- [ ] Curriculum learning with increasing difficulty
- [ ] Transfer learning between variants

## Mathematical Foundations

### Bellman Equations

**Q-Learning (Off-Policy)**:
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

**SARSA (On-Policy)**:
```
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
```

**Linear Approximation**:
```
Q(s,a) = w^T φ(s,a)
w ← w + α[r + γ max_a' Q(s',a') - Q(s,a)] ∇_w Q(s,a)
```

### State Space

**Discrete State (Tabular)**:
```
s = (x, y, veg, meat, spice, dairy, grain, dishes_made)
```

**Continuous State (Function Approximation)**:
```
s = [x, y, veg, meat, spice, dairy, grain, dishes_made, budget?]
```

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*
- Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*
- Rummery, G. A., & Niranjan, M. (1994). On-line Q-learning using connectionist systems

## License

This project is part of the Reinforcement Learning Course 2024-2025.

## Author

Created for RL Course Project 1 - Comparing Tabular and Function Approximation Methods

---

**Happy Cooking! 👨‍🍳🍽️**
