# 🚀 Quick Start Guide - Chaotic Chef Project

## Installation

```bash
cd project1_chaotic_chef
pip install -r requirements.txt
```

## Option 1: Interactive Dashboard (Recommended)

Launch the Streamlit dashboard for full interactive experience:

```bash
streamlit run streamlit_app.py
```

This will open a browser with:
- ✅ **Training Tab**: Configure and train all 4 algorithms with live progress
- ✅ **Learning Curves**: Real-time visualization of training dynamics
- ✅ **Performance Comparison**: Statistical analysis and charts
- ✅ **Live Demo**: Watch trained agents play episodes
- ✅ **Analysis**: Insights and recommendations

### Dashboard Features:
1. **Configure Settings** (left sidebar):
   - Environment variant (basic/monetary)
   - Grid size and max steps
   - Training episodes
   - Learning rate, discount factor, epsilon decay
   - Select which algorithms to train

2. **Start Training**:
   - Click "🚀 Start Training" button
   - Watch live progress with updating charts
   - See real-time metrics and statistics

3. **Explore Results**:
   - Navigate through tabs to see different visualizations
   - Compare algorithm performance
   - Run live demos with trained agents

## Option 2: Run Comprehensive Experiments

Run all experiments with multiple seeds and generate publication-ready results:

```bash
python run_experiments.py
```

This will:
- Train all 4 algorithms with 5 different random seeds
- Generate comprehensive visualizations
- Create detailed performance reports
- Save results to `results/` directory

**Output Files:**
- `results/experiment_results.pkl` - Raw results data
- `results/comparison_plots.png` - Learning curves and performance comparison
- `results/convergence_analysis.png` - Convergence speed analysis
- `results/experiment_report.txt` - Detailed text report
- `results/results_summary.csv` - Summary statistics in CSV format

## Option 3: Test Individual Components

### Test Environment
```bash
python environment.py
```
See the grid world in action with random actions.

### Test Tabular Methods
```bash
python tabular_methods.py
```
Train Q-Learning and SARSA for 500 episodes.

### Test Function Approximation
```bash
python function_approximation.py
```
Train Linear Q-Learning and Deep Q-Learning.

## Quick Tips

### For Fast Testing (5 minutes)
- Set training episodes to 200-500
- Use only 1-2 algorithms
- Basic variant is faster than monetary

### For Best Results (20-30 minutes)
- Set training episodes to 1000-2000
- Train all 4 algorithms
- Use multiple random seeds in `run_experiments.py`

### For Publication Quality (1-2 hours)
- Run `run_experiments.py` with 5+ seeds
- Increase episodes to 2000+
- Try both basic and monetary variants

## Expected Performance

### Typical Results (1000 episodes, basic variant):

| Algorithm | Mean Reward | Dishes Created | Convergence |
|-----------|-------------|----------------|-------------|
| Q-Learning | 45-55 | 4-6 | ~400 episodes |
| SARSA | 40-50 | 4-5 | ~450 episodes |
| Linear Q | 35-45 | 3-5 | ~500 episodes |
| Deep Q | 50-60 | 5-7 | ~600 episodes |

## Troubleshooting

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### Streamlit Not Opening
- Check if port 8501 is available
- Try: `streamlit run streamlit_app.py --server.port 8502`

### CUDA/GPU Issues (Deep Q-Learning)
- The code automatically falls back to CPU
- For GPU: Ensure PyTorch with CUDA is installed

### Slow Training
- Reduce number of episodes
- Use fewer algorithms
- For Deep Q: Reduce batch size or hidden layer size

## Next Steps

1. **Experiment with Hyperparameters**:
   - Try different learning rates
   - Adjust epsilon decay
   - Change discount factors

2. **Try Monetary Variant**:
   - More challenging with budget constraints
   - Different optimal strategies

3. **Analyze Results**:
   - Compare tabular vs function approximation
   - Study Q-Learning vs SARSA differences
   - Examine convergence patterns

4. **Extend the Project**:
   - Add more dishes and ingredients
   - Implement dynamic markets
   - Create multi-agent variant

## File Structure

```
project1_chaotic_chef/
├── environment.py              # Grid world environment
├── tabular_methods.py          # Q-Learning & SARSA
├── function_approximation.py   # Linear & Deep Q-Learning
├── streamlit_app.py           # Interactive dashboard
├── run_experiments.py         # Batch experiment runner
├── requirements.txt           # Dependencies
├── README.md                  # Full documentation
├── QUICKSTART.md             # This file
└── results/                  # Generated results (created on first run)
```

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review code comments for implementation details
3. Examine example outputs in results/ directory

---

**Ready to start? Run:**
```bash
streamlit run streamlit_app.py
```

**Happy Cooking! 👨‍🍳🍽️**
