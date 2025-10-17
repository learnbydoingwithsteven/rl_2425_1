"""
The Chaotic Chef's Quest for the Perfect Meal - Environment Implementation
5x5 Grid City with Markets, Ingredient Collection, and Dish Creation
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Optional

class ChaoticChefEnvironment:
    """
    5x5 Grid environment where the chef collects ingredients and creates dishes.
    
    State Space: (x, y, ingredients_collected, current_dish_progress)
    Action Space: {UP, DOWN, LEFT, RIGHT, COLLECT, CREATE_DISH}
    """
    
    # Actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    COLLECT = 4
    CREATE_DISH = 5
    
    ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'COLLECT', 'CREATE_DISH']
    
    def __init__(self, grid_size: int = 5, max_steps: int = 100, variant: str = 'basic'):
        """
        Initialize the Chaotic Chef environment.
        
        Args:
            grid_size: Size of the grid (default 5x5)
            max_steps: Maximum steps per episode
            variant: 'basic' or 'monetary' (with budget system)
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.variant = variant
        
        # Market locations (fixed)
        self.markets = {
            'vegetable': (0, 0),
            'meat': (0, 4),
            'spice': (4, 0),
            'dairy': (4, 4),
            'grain': (2, 2)
        }
        
        # Ingredient requirements for dishes
        self.dishes = {
            'pasta': {'grain': 1, 'dairy': 1},
            'curry': {'vegetable': 1, 'meat': 1, 'spice': 1},
            'salad': {'vegetable': 2, 'dairy': 1},
            'stew': {'meat': 1, 'vegetable': 1, 'grain': 1}
        }
        
        # Rewards
        self.dish_rewards = {
            'pasta': 10,
            'curry': 15,
            'salad': 8,
            'stew': 12
        }
        
        # Monetary variant
        if self.variant == 'monetary':
            self.ingredient_costs = {
                'vegetable': 2,
                'meat': 5,
                'spice': 3,
                'dairy': 3,
                'grain': 2
            }
            self.dish_revenues = {
                'pasta': 15,
                'curry': 25,
                'salad': 12,
                'stew': 20
            }
            self.initial_budget = 50
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        # Random starting position (not on markets)
        while True:
            self.chef_pos = (random.randint(0, self.grid_size-1), 
                           random.randint(0, self.grid_size-1))
            if self.chef_pos not in self.markets.values():
                break
        
        # Inventory
        self.inventory = {
            'vegetable': 0,
            'meat': 0,
            'spice': 0,
            'dairy': 0,
            'grain': 0
        }
        
        # Dishes created
        self.dishes_created = {dish: 0 for dish in self.dishes}
        
        # Episode tracking
        self.steps = 0
        self.total_reward = 0
        self.done = False
        
        # Monetary variant
        if self.variant == 'monetary':
            self.budget = self.initial_budget
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation.
        
        State: [x, y, veg, meat, spice, dairy, grain, dishes_made, budget (if monetary)]
        """
        state = [
            self.chef_pos[0],
            self.chef_pos[1],
            self.inventory['vegetable'],
            self.inventory['meat'],
            self.inventory['spice'],
            self.inventory['dairy'],
            self.inventory['grain'],
            sum(self.dishes_created.values())
        ]
        
        if self.variant == 'monetary':
            state.append(self.budget)
        
        return np.array(state, dtype=np.float32)
    
    def _get_state_tuple(self) -> Tuple:
        """Get state as tuple for tabular methods."""
        state = (
            self.chef_pos[0],
            self.chef_pos[1],
            min(self.inventory['vegetable'], 3),  # Cap for state space
            min(self.inventory['meat'], 3),
            min(self.inventory['spice'], 3),
            min(self.inventory['dairy'], 3),
            min(self.inventory['grain'], 3),
            min(sum(self.dishes_created.values()), 5)
        )
        
        if self.variant == 'monetary':
            budget_bin = min(self.budget // 10, 10)  # Discretize budget
            state = state + (budget_bin,)
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info.
        
        Args:
            action: Action to take
            
        Returns:
            next_state, reward, done, info
        """
        if self.done:
            return self._get_state(), 0, True, {}
        
        self.steps += 1
        reward = -0.1  # Small step penalty
        info = {'action': self.ACTION_NAMES[action]}
        
        # Movement actions
        if action == self.UP:
            new_pos = (max(0, self.chef_pos[0] - 1), self.chef_pos[1])
            self.chef_pos = new_pos
            
        elif action == self.DOWN:
            new_pos = (min(self.grid_size - 1, self.chef_pos[0] + 1), self.chef_pos[1])
            self.chef_pos = new_pos
            
        elif action == self.LEFT:
            new_pos = (self.chef_pos[0], max(0, self.chef_pos[1] - 1))
            self.chef_pos = new_pos
            
        elif action == self.RIGHT:
            new_pos = (self.chef_pos[0], min(self.grid_size - 1, self.chef_pos[1] + 1))
            self.chef_pos = new_pos
        
        # Collect ingredient
        elif action == self.COLLECT:
            collected = False
            for ingredient, pos in self.markets.items():
                if self.chef_pos == pos:
                    if self.variant == 'monetary':
                        cost = self.ingredient_costs[ingredient]
                        if self.budget >= cost:
                            self.inventory[ingredient] += 1
                            self.budget -= cost
                            reward = 1
                            collected = True
                            info['collected'] = ingredient
                            info['cost'] = cost
                        else:
                            reward = -1  # Tried to collect without budget
                            info['failed'] = 'insufficient_budget'
                    else:
                        self.inventory[ingredient] += 1
                        reward = 1
                        collected = True
                        info['collected'] = ingredient
                    break
            
            if not collected and self.variant != 'monetary':
                reward = -0.5  # Penalty for collecting at wrong location
        
        # Create dish
        elif action == self.CREATE_DISH:
            dish_created = False
            for dish_name, requirements in self.dishes.items():
                can_make = all(self.inventory[ing] >= qty 
                             for ing, qty in requirements.items())
                
                if can_make:
                    # Deduct ingredients
                    for ing, qty in requirements.items():
                        self.inventory[ing] -= qty
                    
                    # Award reward
                    if self.variant == 'monetary':
                        revenue = self.dish_revenues[dish_name]
                        self.budget += revenue
                        reward = revenue
                        info['revenue'] = revenue
                    else:
                        reward = self.dish_rewards[dish_name]
                    
                    self.dishes_created[dish_name] += 1
                    dish_created = True
                    info['dish_created'] = dish_name
                    break
            
            if not dish_created:
                reward = -1  # Penalty for trying to create dish without ingredients
        
        # Check termination
        if self.steps >= self.max_steps:
            self.done = True
            info['termination'] = 'max_steps'
        
        if self.variant == 'monetary' and self.budget <= 0:
            self.done = True
            reward = -10  # Penalty for bankruptcy
            info['termination'] = 'bankruptcy'
        
        # Bonus for multiple dishes
        total_dishes = sum(self.dishes_created.values())
        if total_dishes >= 5:
            reward += 5
            if total_dishes >= 10:
                reward += 10
                self.done = True
                info['termination'] = 'success'
        
        self.total_reward += reward
        
        return self._get_state(), reward, self.done, info
    
    def get_state_size(self) -> int:
        """Get size of state vector."""
        return 9 if self.variant == 'monetary' else 8
    
    def get_action_size(self) -> int:
        """Get number of actions."""
        return 6
    
    def render(self) -> str:
        """Render the current state as text."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place markets
        for ingredient, pos in self.markets.items():
            grid[pos[0]][pos[1]] = ingredient[0].upper()
        
        # Place chef
        grid[self.chef_pos[0]][self.chef_pos[1]] = 'C'
        
        # Create display
        display = '\n'.join([' '.join(row) for row in grid])
        display += f"\n\nInventory: {self.inventory}"
        display += f"\nDishes Created: {self.dishes_created}"
        display += f"\nSteps: {self.steps}/{self.max_steps}"
        display += f"\nTotal Reward: {self.total_reward:.2f}"
        
        if self.variant == 'monetary':
            display += f"\nBudget: ${self.budget}"
        
        return display
    
    def get_optimal_policy_value(self) -> float:
        """Estimate optimal policy value (for comparison)."""
        # Simple heuristic: optimal agent creates ~10 dishes
        if self.variant == 'monetary':
            return 100  # Approximate optimal profit
        else:
            return 120  # Approximate optimal reward


if __name__ == "__main__":
    # Test the environment
    print("Testing Chaotic Chef Environment\n")
    
    # Basic variant
    env = ChaoticChefEnvironment(variant='basic')
    state = env.reset()
    print("Initial State:")
    print(env.render())
    print("\n" + "="*50 + "\n")
    
    # Random episode
    for i in range(20):
        action = random.randint(0, 5)
        next_state, reward, done, info = env.step(action)
        
        if reward > 0.5 or 'dish_created' in info:
            print(f"Step {i+1}: Action={env.ACTION_NAMES[action]}, Reward={reward:.2f}")
            if 'collected' in info:
                print(f"  Collected: {info['collected']}")
            if 'dish_created' in info:
                print(f"  Created: {info['dish_created']}")
            print(env.render())
            print("\n" + "="*50 + "\n")
        
        if done:
            break
    
    print("\nFinal State:")
    print(env.render())
