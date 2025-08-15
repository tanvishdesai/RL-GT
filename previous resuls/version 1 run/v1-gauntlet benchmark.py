import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import logging
import wandb
from typing import Dict, List, Callable, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gymnasium as gym
from gymnasium.spaces import Space, Discrete, Box
import hydra
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import time
from pathlib import Path
import copy

# Enhanced imports for generalization and metrics
try:
    import nashpy as nash
    NASH_AVAILABLE = True
except ImportError:
    NASH_AVAILABLE = False
    print("Warning: nashpy not available. Nash convergence metrics will be simplified.")

try:
    from pettingzoo.utils import AECEnv
    from pettingzoo import ParallelEnv
    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False
    print("Warning: pettingzoo not available. PettingZoo environments will not be supported.")

# Visualization imports
import matplotlib.patches as patches
from matplotlib.patches import Circle, RegularPolygon
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

# ============================================================================
# Core Data Structures and Configuration
# ============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for Gauntlet evaluation."""
    num_episodes: int = 1000
    max_episode_steps: int = 200
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    parallel_workers: int = 0
    save_trajectories: bool = False
    compute_exploitability: bool = True
    enable_continual_eval: bool = True
    population_size: int = 5
    tournament_rounds: int = 2
    
    # New configuration options for generalization
    support_continuous_actions: bool = True
    support_multi_agent: bool = True
    max_agents: int = 1000  # For large-scale evaluations
    vectorized_evaluation: bool = False
    
    # Visualization configuration
    save_visualizations: bool = True
    visualization_format: str = "png"  # png, pdf, svg
    dpi: int = 300
    style: str = "seaborn-v0_8"  # matplotlib style
    
    # Metrics configuration
    use_nashpy_metrics: bool = NASH_AVAILABLE
    compute_transfer_metrics: bool = True
    compute_population_diversity: bool = True
    regret_bound: float = 1.0  # Upper bound for regret computation

@dataclass
class ContinualConfig:
    """Configuration for continual learning evaluation."""
    num_tasks: int = 100
    task_transition_episodes: int = 50
    forgetting_threshold: float = 0.1
    plasticity_threshold: float = 0.05
    memory_replay: bool = True
    replay_buffer_size: int = 10000

@dataclass
class RobustnessMetrics:
    """Comprehensive robustness metrics."""
    overall_win_rate: float = 0.0
    min_win_rate: float = 0.0
    max_win_rate: float = 0.0
    win_rate_std: float = 0.0
    avg_reward: float = 0.0
    worst_case_reward: float = 0.0
    exploitability: float = 0.0
    regret: float = 0.0
    adaptation_rate: float = 0.0
    forgetting_rate: float = 0.0
    plasticity_score: float = 0.0
    population_diversity: float = 0.0
    nash_conv: float = 0.0
    
    # New metrics for transfer learning and population analysis
    forward_transfer: float = 0.0
    backward_transfer: float = 0.0
    population_entropy: float = 0.0
    jensen_shannon_divergence: float = 0.0
    nash_equilibrium_distance: float = 0.0
    regret_bound_achieved: bool = False
    
    @property
    def robustness_score(self) -> float:
        """Comprehensive robustness score combining multiple metrics."""
        return (
            0.25 * self.overall_win_rate +
            0.15 * self.min_win_rate +
            0.12 * (1.0 - self.exploitability) +
            0.12 * (1.0 - self.regret) +
            0.10 * self.adaptation_rate +
            0.08 * self.plasticity_score +
            0.08 * self.forward_transfer +
            0.05 * (1.0 - self.forgetting_rate) +
            0.05 * self.population_diversity
        )

# ============================================================================
# Base Classes and Interfaces
# ============================================================================

class ChallengerAgent(ABC):
    """Abstract base class for challenger agents."""
    
    def __init__(self, name: str, difficulty: str = "medium"):
        self.name = name
        self.difficulty = difficulty
        self.history = deque(maxlen=1000)
        self.adaptation_rate = 0.1
    
    @abstractmethod
    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        """Select an action given observation and opponent history."""
        pass
    
    # --- NEW REQUIRED PROPERTY ---
    @property
    @abstractmethod
    def compatible_action_space(self) -> Space:
        """The Gymnasium action space this challenger is compatible with."""
        pass
    
    @abstractmethod
    def update(self, reward: float, observation: torch.Tensor, action: int):
        """Update internal state based on outcome."""
        pass
    
    def reset(self):
        """Reset agent state for new episode."""
        self.history.clear()

class Environment(ABC):
    """Abstract environment interface."""
    
    @abstractmethod
    def reset(self) -> torch.Tensor:
        pass
    
    @abstractmethod
    def step(self, actions: List[int]) -> Tuple[torch.Tensor, List[float], bool, Dict]:
        pass
    
    @property
    @abstractmethod
    def observation_space(self) -> Space:
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> Space:
        pass
    
    @property
    def num_actions(self) -> int:
        """Get number of actions (for discrete) or action dimension (for continuous)."""
        if isinstance(self.action_space, Discrete):
            return self.action_space.n
        elif isinstance(self.action_space, Box):
            return self.action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space type: {type(self.action_space)}")
    
    @property
    def is_continuous_action(self) -> bool:
        """Check if environment uses continuous actions."""
        return isinstance(self.action_space, Box)
    
    @property
    def is_discrete_action(self) -> bool:
        """Check if environment uses discrete actions."""
        return isinstance(self.action_space, Discrete)

class GeneralizedEnvironment(Environment):
    """Generalized environment wrapper that supports both discrete and continuous actions."""
    
    def __init__(self, base_env: Environment):
        self.base_env = base_env
        self._validate_action_space()
    
    def _validate_action_space(self):
        """Validate that the action space is supported."""
        if not (isinstance(self.action_space, (Discrete, Box))):
            raise ValueError(f"Unsupported action space: {type(self.action_space)}")
    
    def reset(self) -> torch.Tensor:
        return self.base_env.reset()
    
    def step(self, actions: List[Union[int, float]]) -> Tuple[torch.Tensor, List[float], bool, Dict]:
        # Validate actions based on action space
        if self.is_discrete_action:
            actions = [int(action) for action in actions]
            for action in actions:
                if not (0 <= action < self.num_actions):
                    raise ValueError(f"Discrete action {action} out of range [0, {self.num_actions})")
        elif self.is_continuous_action:
            actions = [float(action) for action in actions]
            for action in actions:
                if not (self.action_space.low[0] <= action <= self.action_space.high[0]):
                    raise ValueError(f"Continuous action {action} out of bounds")
        
        return self.base_env.step(actions)
    
    @property
    def observation_space(self) -> Space:
        return self.base_env.observation_space
    
    @property
    def action_space(self) -> Space:
        return self.base_env.action_space

# ============================================================================
# Advanced Challenger Implementations
# ============================================================================

class AdaptiveCounterAgent(ChallengerAgent):
    """Counter-exploiter that adapts to opponent patterns."""
    
    def __init__(self, name: str = "AdaptiveCounter"):
        super().__init__(name, "hard")
        self.pattern_detector = PatternDetector()
        self.counter_strategy = CounterStrategy()
        self.meta_learner = MetaLearner()
        self._action_space = Discrete(3) # This agent is hard-coded for RPS
 
    # --- NEW ---
    @property
    def compatible_action_space(self) -> Space:
        return self._action_space

    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        """
        Select an action given observation and opponent history.
        
        CORRECTION: The modulo operation now correctly uses `3` (the number of actions in RPS)
        instead of `observation.shape[-1]` (the observation dimension), preventing the IndexError.
        """
        num_actions = 3  # For Rock-Paper-Scissors

        if opponent_history and len(opponent_history) > 10:
            pattern = self.pattern_detector.detect(opponent_history)
            counter_action = self.counter_strategy.counter(pattern)
            meta_adjustment = self.meta_learner.adjust(self.history, opponent_history)
            
            # Ensure the final action is within the valid range [0, 2]
            return (counter_action + meta_adjustment) % num_actions
        
        return random.randint(0, num_actions - 1)
    
    def update(self, reward: float, observation: torch.Tensor, action: int):
        self.history.append((action, reward))
        self.meta_learner.update(reward)

class PopulationBasedAgent(ChallengerAgent):
    """Agent that maintains a population of diverse strategies."""
    # --- NEW ---
    def __init__(self, name: str = "PopulationBased", population_size: int = 10):
        super().__init__(name, "expert")
        self.population = [self._create_diverse_strategy(i) for i in range(population_size)]
        self.selection_probs = np.ones(population_size) / population_size
        self.performance_history = defaultdict(list)
        self._action_space = Discrete(3) # This agent is hard-coded for RPS
    # --- NEW ---
    @property
    def compatible_action_space(self) -> Space:
        return self._action_space

    def _create_diverse_strategy(self, seed: int) -> Callable:
        """Create a diverse strategy based on seed."""
        np.random.seed(seed)
        # Added a default case to prevent returning None
        strategy_type = np.random.choice(['cyclic', 'frequency', 'pattern', 'mixed'])
        
        if strategy_type == 'cyclic':
            cycle = np.random.permutation(3).tolist()
            return lambda h: cycle[len(h) % len(cycle)] if h else 0
        elif strategy_type == 'frequency':
            freqs = np.random.dirichlet([1, 1, 1])
            return lambda h: np.random.choice(3, p=freqs)
        # Add more strategy types...
        else: # Default case for 'pattern', 'mixed', or any other type
            return lambda h: random.randint(0, 2)
        
    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        selected_strategy = np.random.choice(self.population, p=self.selection_probs)
        return selected_strategy(opponent_history or [])
    
    def update(self, reward: float, observation: torch.Tensor, action: int):
        # Update selection probabilities based on performance
        self.history.append((action, reward))
        # Implement evolutionary selection logic here

# ============================================================================
# Corrected NeuralAdversaryAgent with Lazy Initialization
# ============================================================================

# ============================================================================
# Corrected NeuralAdversaryAgent with Lazy Initialization
# ============================================================================

class NeuralAdversaryAgent(ChallengerAgent):
    """
    Neural network-based adversary that dynamically adapts its input size
    to the environment it is playing in.
    """
    
    # --- START: CORRECTED __init__ METHOD ---
    def __init__(self, name: str = "NeuralAdversary", hidden_dim: int = 64, 
                 action_space: Space = Discrete(3)):
        super().__init__(name, "expert")
        # Store the provided action space
        self._action_space = action_space
        
        # Derive properties directly from the action space, making the agent general
        self.is_continuous = isinstance(self._action_space, Box)
        if self.is_continuous:
            self.action_dim = self._action_space.shape[0]
        else:
            self.action_dim = self._action_space.n
            
        self.hidden_dim = hidden_dim # Store hidden_dim for later use

        # Lazy Initialization for the network and optimizer
        self.network = None
        self.optimizer = None
        self.memory = deque(maxlen=10000)
    # --- END: CORRECTED __init__ METHOD ---

    # --- NEW REQUIRED PROPERTY ---
    @property
    def compatible_action_space(self) -> Space:
        """The Gymnasium action space this challenger is compatible with."""
        return self._action_space

    def _initialize_network(self, observation: torch.Tensor):
        """
        Builds the neural network and optimizer based on the shape of the
        first observation tensor received from the environment.
        """
        input_dim = observation.shape[-1]
        device = observation.device
        
        print(f"Initializing NeuralAdversary for input_dim={input_dim} on device='{device}'...")

        if self.is_continuous:
            self.network = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.action_dim * 2),
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.action_dim),
                nn.Softmax(dim=-1)
            )
        
        self.network.to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)

    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> Union[int, List[float]]:
        if self.network is None:
            self._initialize_network(observation)

        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)
        
        with torch.no_grad():
            if self.is_continuous:
                output = self.network(observation)
                mean = output[:, :self.action_dim]
                log_std = output[:, self.action_dim:]
                std = torch.exp(log_std)
                action = torch.normal(mean, std)
                action = torch.clamp(action, -1.0, 1.0)
                return action.squeeze().tolist()
            else:
                action_probs = self.network(observation)
                action = torch.multinomial(action_probs, 1).item()
                return action
    
    def update(self, reward: float, observation: torch.Tensor, action: Union[int, List[float]]):
        if self.network is None:
            return

        self.memory.append((observation.cpu(), action, reward))
        if len(self.memory) > 32:
            self._train_batch()
    
    def _train_batch(self):
        if self.optimizer is None:
            return

        batch = random.sample(self.memory, min(32, len(self.memory)))
        observations, actions, rewards = zip(*batch)
        
        device = next(self.network.parameters()).device
        observations = torch.stack(observations).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        
        if self.is_continuous:
            actions = torch.tensor(actions, dtype=torch.float32).to(device)
            output = self.network(observations)
            mean = output[:, :self.action_dim]
            log_std = output[:, self.action_dim:]
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            loss = -(log_probs * rewards).mean()
        else:
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            action_probs = self.network(observations)
            selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
            loss = -(torch.log(selected_probs + 1e-9) * rewards).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def reset(self):
        super().reset()

# ============================================================================
# Utility Classes
# ============================================================================

class PatternDetector:
    """Detects patterns in opponent behavior."""
    
    def __init__(self):
        self.min_pattern_length = 2
        self.max_pattern_length = 10
    
    def detect(self, history: List[int]) -> Optional[List[int]]:
        """Detect repeating patterns in history."""
        if len(history) < self.min_pattern_length * 2:
            return None
            
        for pattern_length in range(self.min_pattern_length, min(self.max_pattern_length, len(history) // 2)):
            pattern = history[-pattern_length:]
            if self._is_repeating_pattern(history, pattern):
                return pattern
        return None
    
    def _is_repeating_pattern(self, history: List[int], pattern: List[int]) -> bool:
        """Check if pattern repeats in recent history."""
        pattern_len = len(pattern)
        if len(history) < pattern_len * 2:
            return False
        
        for i in range(pattern_len):
            if history[-(pattern_len * 2) + i] != pattern[i]:
                return False
        return True

class CounterStrategy:
    """Implements counter-strategies against detected patterns."""
    
    def counter(self, pattern: Optional[List[int]]) -> int:
        """Generate counter-action for detected pattern."""
        if pattern is None:
            return random.randint(0, 2)
        
        # Predict next action in pattern
        predicted_action = pattern[0]  # Simplified prediction
        # Return counter-action (Rock->Paper, Paper->Scissors, Scissors->Rock)
        return (predicted_action + 1) % 3

class MetaLearner:
    """Meta-learning component for strategy adaptation."""
    
    def __init__(self):
        self.strategy_performance = defaultdict(float)
        self.current_strategy = "default"
        self.exploration_rate = 0.1
    
    def adjust(self, self_history: deque, opponent_history: List[int]) -> int:
        """Meta-adjustment based on performance history."""
        if len(self_history) < 10:
            return 0
        
        recent_performance = np.mean([reward for _, reward in list(self_history)[-10:]])
        if recent_performance < 0:
            return random.randint(-1, 1)  # Add randomness if performing poorly
        return 0
    
    def update(self, reward: float):
        """Update meta-learning based on reward."""
        self.strategy_performance[self.current_strategy] += reward

# ============================================================================
# Enhanced Gauntlet Framework
# ============================================================================

class EnhancedGauntletBenchmark:
    """Next-generation MARL evaluation framework with comprehensive robustness testing."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.challengers = {} # --- MODIFIED: This is now ONLY for custom-added challengers ---
        self.environments = {}
        self.results_history = []
        self.logger = self._setup_logging()
        
        # --- NEW ---
        self.master_challenger_list = self._build_master_challenger_list()

        # Initialize components
        # self._build_challenger_suite() # --- REMOVED ---
        self._setup_metrics_tracking()
        self._setup_continual_learning()
        
        # GPU acceleration setup
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("Gauntlet")
        
        # WandB integration
        # if wandb.run is None:
        #     wandb.init(project="gauntlet-marl-benchmark", config=self.config.__dict__)
        
        return logger
# Add or replace these methods inside the EnhancedGauntletBenchmark class

    def _analyze_performance_trends(self) -> Dict:
        """
        Analyzes the policy's performance trend over multiple evaluation runs.
        Uses linear regression on robustness scores.
        """
        if len(self.results_history) < 2:
            return {
                "trend": "N/A",
                "details": "Insufficient data (only 1 evaluation run). Run more evaluations to see a trend."
            }
    
        # Extract robustness scores from history
        scores = [res['metrics'].robustness_score for res in self.results_history]
        eval_indices = np.arange(len(scores))
    
        # Perform linear regression to find the trend
        try:
            # Fit a line (degree 1 polynomial) to the data
            slope, intercept = np.polyfit(eval_indices, scores, 1)
        except np.linalg.LinAlgError:
            # This can happen in rare cases with ill-conditioned matrices
            return {
                "trend": "undetermined",
                "details": "Could not determine trend due to a numerical error."
            }
    
        # Determine the trend based on the slope of the regression line
        if slope > 0.05:
            trend = "Improving"
        elif slope < -0.05:
            trend = "Declining"
        else:
            trend = "Stable"
            
        return {
            "trend": trend,
            "details": f"Trend calculated over {len(scores)} evaluations with a slope of {slope:.4f}."
        }
    
    def _analyze_challenger_performance(self) -> Dict:
        """
        Identifies the easiest and hardest challengers from the most recent evaluation.
        """
        if not self.results_history:
            return {
                "easiest_challenger": "N/A",
                "hardest_challenger": "N/A",
                "details": "No evaluation results found."
            }
    
        latest_results = self.results_history[-1]
        challenger_scores = defaultdict(list)
    
        # Aggregate win rates for each challenger across all tested environments
        for env_name, env_results in latest_results['detailed_results'].items():
            # --- START: CORRECTED CODE ---
            for challenger_name, results in env_results.items():
                # Add the check to process only challenger result dictionaries
                if isinstance(results, dict):
                    challenger_scores[challenger_name].append(results['win_rate'])
            # --- END: CORRECTED CODE ---
    
        if not challenger_scores:
            return {
                "easiest_challenger": "N/A",
                "hardest_challenger": "N/A",
                "details": "No challenger results available in the latest evaluation."
            }
            
        # Calculate the average win rate for each challenger
        avg_scores = {name: np.mean(scores) for name, scores in challenger_scores.items()}
    
        # Find the challenger with the highest and lowest average win rate
        easiest_challenger = max(avg_scores, key=avg_scores.get)
        hardest_challenger = min(avg_scores, key=avg_scores.get)
    
        return {
            "easiest_challenger": f"{easiest_challenger} (Win Rate: {avg_scores[easiest_challenger]:.3f})",
            "hardest_challenger": f"{hardest_challenger} (Win Rate: {avg_scores[hardest_challenger]:.3f})",
        }


    def _identify_weakness_patterns(self) -> List[str]:
        """
        Identifies patterns of weakness against categories of challengers.
        """
        if not self.results_history:
            return ["No evaluation results found to analyze patterns."]
    
        # Define challenger categories
        challenger_categories = {
            'fixed_strategy': ['AlwaysRock', 'AlwaysPaper', 'AlwaysScissors'],
            'biased_strategy': ['BiasedRock', 'BiasedPaper', 'BiasedScissors'],
            'pattern_based': ['CyclicRPS', 'CyclicRSP', 'TitForTat', 'Copycat'],
            'adaptive_learning': ['AdaptiveCounter', 'PopulationBased', 'NeuralAdversary'],
            'noise_robustness': ['NoisyUniform', 'AdversarialNoise']
        }
        
        # Invert the dictionary for easy lookup
        challenger_map = {challenger: category for category, challengers in challenger_categories.items() for challenger in challengers}
    
        latest_results = self.results_history[-1]
        category_scores = defaultdict(list)
    
        # Aggregate scores by category
        for env_results in latest_results['detailed_results'].values():
            # --- START: CORRECTED CODE ---
            for challenger_name, results in env_results.items():
                # Add the check to process only challenger result dictionaries
                if isinstance(results, dict):
                    category = challenger_map.get(challenger_name)
                    if category:
                        category_scores[category].append(results['win_rate'])
            # --- END: CORRECTED CODE ---
    
        if not category_scores:
            return ["Could not categorize challengers to identify weakness patterns."]
    
        # Analyze performance against each category
        weaknesses = []
        avg_category_scores = {cat: np.mean(scores) for cat, scores in category_scores.items()}
        
        # Define thresholds for what constitutes a weakness
        WEAKNESS_THRESHOLD = 0.4  # Win rate below which we consider it a weakness
        
        for category, avg_score in avg_category_scores.items():
            if avg_score < WEAKNESS_THRESHOLD:
                weaknesses.append(f"Struggles against '{category}' opponents (Avg Win Rate: {avg_score:.3f})")
    
        # Check for inconsistent performance
        metrics = latest_results['metrics']
        if metrics.win_rate_std > 0.2:
            weaknesses.append(f"Shows high performance variance (Std Dev: {metrics.win_rate_std:.3f}), indicating inconsistency.")
    
        if not weaknesses:
            return ["No significant weakness patterns identified. The policy is well-rounded."]
            
        return weaknesses
    
    def _generate_improvement_suggestions(self) -> List[str]:
        """
        Generates actionable improvement suggestions based on identified weaknesses.
        """
        if not self.results_history:
            return ["Run an evaluation to generate suggestions."]
        
        latest_results = self.results_history[-1]
        metrics = latest_results['metrics']
        weakness_patterns = self._identify_weakness_patterns()
        suggestions = []
        
        # Suggestions based on top-level metrics
        if metrics.exploitability > 0.4:
            suggestions.append("High Exploitability: Consider adversarial training or add more diverse, adaptive agents (like NeuralAdversary) to the training opponents.")
        
        if metrics.regret > 0.5:
            suggestions.append("High Regret: The policy is far from optimal. This could indicate a need for a more complex model architecture, longer training, or hyperparameter tuning.")
        
        if metrics.nash_conv < 0.6:
            suggestions.append("Low Nash Convergence: The policy's strategy is not close to a game-theoretic equilibrium. Improve this by training against a wider variety of strong opponents or using self-play schemes like Fictitious Play.")
    
        if metrics.win_rate_std > 0.2:
            suggestions.append("Inconsistent Performance: To stabilize performance, try using regularization techniques (e.g., entropy regularization) or policy ensemble methods.")
            
        # Suggestions based on weakness patterns
        for pattern in weakness_patterns:
            if 'adaptive_learning' in pattern:
                suggestions.append("Weak against Adaptive Agents: The policy is being out-learned. Enhance its adaptability by incorporating memory (e.g., LSTMs) into the policy network or using meta-learning techniques.")
            if 'pattern_based' in pattern:
                suggestions.append("Weak against Pattern-Based Agents: The policy is predictable. Introduce mechanisms to detect and break patterns, such as adding memory (LSTMs) or increasing stochasticity in its actions.")
            if 'noise_robustness' in pattern:
                suggestions.append("Weak against Noise: Improve robustness by injecting noise into observations or actions during the training process.")
    
        if not suggestions:
            return ["The policy appears robust. Continue monitoring for any emerging weaknesses."]
    
        # Return a unique set of suggestions
        return list(dict.fromkeys(suggestions))    

# In benchmark.py -> class EnhancedGauntletBenchmark

    # --- REPLACED METHOD ---
    def _build_master_challenger_list(self) -> Dict[str, ChallengerAgent]:
        """Builds a comprehensive list of ALL challengers across ALL supported games."""
        all_challengers = {}

        # --- Game Action Spaces ---
        rps_space = Discrete(3)   # 0:Rock, 1:Paper, 2:Scissors
        ipd_space = Discrete(2)   # 0:Cooperate, 1:Defect
        kuhn_poker_space = Discrete(2) # 0:Pass/Check, 1:Bet/Call
        matching_pennies_space = Discrete(2) # 0:Heads, 1:Tails

        # ==========================================================
        # 1. Rock-Paper-Scissors Challengers (Action Space: Discrete(3))
        # ==========================================================
        all_challengers["RPS_AlwaysRock"] = self._create_fixed_action_bot(0, rps_space)
        all_challengers["RPS_AlwaysPaper"] = self._create_fixed_action_bot(1, rps_space)
        all_challengers["RPS_AlwaysScissors"] = self._create_fixed_action_bot(2, rps_space)
        all_challengers["RPS_Uniform"] = self._create_random_bot(rps_space)
        all_challengers["RPS_BiasedRock"] = self._create_biased_bot([0.7, 0.2, 0.1], rps_space)
        all_challengers["RPS_CyclicRPS"] = self._create_cyclic_bot([0, 1, 2], rps_space)
        all_challengers["RPS_Copycat"] = self._create_copycat_bot(rps_space)
        all_challengers["RPS_AdaptiveCounter"] = AdaptiveCounterAgent()
        all_challengers["RPS_PopulationBased"] = PopulationBasedAgent()
        all_challengers["RPS_NeuralAdversary"] = NeuralAdversaryAgent(action_space=rps_space)

        # ==========================================================
        # 2. Iterated Prisoner's Dilemma (IPD) Challengers (Action Space: Discrete(2))
        # ==========================================================
        all_challengers["IPD_AlwaysCooperate"] = self._create_fixed_action_bot(0, ipd_space)
        all_challengers["IPD_AlwaysDefect"] = self._create_fixed_action_bot(1, ipd_space)
        all_challengers["IPD_Uniform"] = self._create_random_bot(ipd_space)
        all_challengers["IPD_TitForTat"] = self._create_tit_for_tat_bot(ipd_space)
        all_challengers["IPD_Copycat"] = self._create_copycat_bot(ipd_space)
        # Grudger: Cooperates until the opponent defects once, then defects forever.
        class GrudgerBot(ChallengerAgent):
            def __init__(self):
                super().__init__("IPD_Grudger", "medium")
                self.has_grudge = False
                self._action_space = ipd_space

            # Corrected signature to match the ChallengerAgent interface
            def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
                # If the opponent has a history and their last move was Defect (1)
                if opponent_history and opponent_history[-1] == 1:
                    self.has_grudge = True
                
                # If a grudge is held, always defect. Otherwise, cooperate.
                return 1 if self.has_grudge else 0

            @property
            def compatible_action_space(self) -> Space:
                return self._action_space

            def update(self, reward: float, observation: torch.Tensor, action: int):
                pass # This agent's logic is stateless within an episode

            def reset(self):
                super().reset()
                self.has_grudge = False # Reset the grudge for each new episode
        # --- END: CORRECTED GrudgerBot DEFINITION ---
        all_challengers["IPD_Grudger"] = GrudgerBot()

        # ==========================================================
        # 3. Kuhn Poker Challengers (Action Space: Discrete(2))
        # ==========================================================
        all_challengers["Kuhn_AlwaysPass"] = self._create_fixed_action_bot(0, kuhn_poker_space)
        all_challengers["Kuhn_AlwaysBet"] = self._create_fixed_action_bot(1, kuhn_poker_space)
        all_challengers["Kuhn_Uniform"] = self._create_random_bot(kuhn_poker_space)

        # ==========================================================
        # 4. Matching Pennies Challengers (Action Space: Discrete(2))
        # ==========================================================
        all_challengers["Pennies_AlwaysHeads"] = self._create_fixed_action_bot(0, matching_pennies_space)
        all_challengers["Pennies_AlwaysTails"] = self._create_fixed_action_bot(1, matching_pennies_space)
        all_challengers["Pennies_Uniform"] = self._create_random_bot(matching_pennies_space)
        all_challengers["Pennies_Copycat"] = self._create_copycat_bot(matching_pennies_space)

        print(f"Built a master list of {len(all_challengers)} challengers for various games.")
        return all_challengers


    def _setup_metrics_tracking(self):
        """Setup comprehensive metrics tracking."""
        self.metrics_tracker = {
            'win_rates': defaultdict(list),
            'rewards': defaultdict(list),
            'exploitability': defaultdict(list),
            'regret': defaultdict(list),
            'adaptation_rates': defaultdict(list),
            'population_diversity': defaultdict(list)
        }
    
    def _setup_continual_learning(self):
        """Setup continual learning evaluation components."""
        self.continual_config = ContinualConfig()
        self.task_generator = TaskGenerator()
        self.forgetting_detector = ForgettingDetector()
        self.plasticity_evaluator = PlasticityEvaluator()
    

    # --- MODIFIED METHOD SIGNATURE AND LOGIC ---
    def register_environment(self, name: str, env_factory: Callable, 
                           payoff_matrix: Optional[np.ndarray] = None, 
                           game_prefix: Optional[str] = None) -> None:
        """
        Register a new environment for evaluation.

        Args:
            name (str): The unique name for the environment (e.g., "MatchingPennies").
            env_factory (Callable): A function that returns a new instance of the environment.
            payoff_matrix (Optional[np.ndarray]): The game's payoff matrix for Nash calculations.
            game_prefix (Optional[str]): A prefix to link this environment to challengers
                                         (e.g., "Pennies"). If None, 'name' is used.
        """
        self.environments[name] = {
            "factory": env_factory,
            "payoff_matrix": payoff_matrix,
            "game_prefix": game_prefix 
        }
        if game_prefix:
            self.logger.info(f"Registered environment '{name}' with game prefix '{game_prefix}'.")
        else:
            self.logger.info(f"Registered environment '{name}'.")    
    def register_pettingzoo_env(self, name: str, env: Union[AECEnv, ParallelEnv]) -> None:
        """Register a PettingZoo environment for evaluation."""
        if not PETTINGZOO_AVAILABLE:
            print("PettingZoo not available. Environment registration skipped.")
            return
        
        # Create wrapper for PettingZoo environment
        wrapped_env = self._create_pettingzoo_wrapper(env)
        self.environments[name] = lambda: wrapped_env
        print(f"Registered PettingZoo environment: {name}")
    
    def _create_pettingzoo_wrapper(self, env: Union[AECEnv, ParallelEnv]) -> Environment:
        """Create a wrapper for PettingZoo environments."""
        class PettingZooWrapper(Environment):
            def __init__(self, pettingzoo_env):
                self.env = pettingzoo_env
                self.env.reset()
                self.agents = list(self.env.agents)
                self.current_agent = self.agents[0] if self.agents else None
                self.state = None
                
                # --- START: CORRECTED CODE ---
                # Use private attributes to store the spaces, avoiding name clash with properties.
                agent_key = self.agents[0] if self.agents else None

                # Determine action space using the official PettingZoo API: .action_space(agent)
                if agent_key and hasattr(self.env, 'action_space') and callable(getattr(self.env, 'action_space', None)):
                    self._action_space = self.env.action_space(agent_key)
                else:
                    # Fallback for older or non-standard environments
                    self._action_space = Discrete(3)
                
                # Determine observation space
                if agent_key and hasattr(self.env, 'observation_space') and callable(getattr(self.env, 'observation_space', None)):
                    self._observation_space = self.env.observation_space(agent_key)
                else:
                    # Fallback
                    self._observation_space = Box(low=0, high=1, shape=(6,))
            
            def reset(self) -> torch.Tensor:
                self.env.reset()
                self.current_agent = self.agents[0] if self.agents else None
                
                # For AECEnv, use last() to get the first observation
                obs, _, _, _, _ = self.env.last()
                
                # Convert to torch tensor
                if isinstance(obs, np.ndarray):
                    self.state = torch.from_numpy(obs).float()
                else:
                    self.state = torch.tensor(obs, dtype=torch.float32)
                
                return self.state
            
            def step(self, actions: List[Union[int, float]]) -> Tuple[torch.Tensor, List[float], bool, Dict]:
                # This wrapper assumes a two-player, alternating game like RPS.
                # It takes both actions and completes one full "turn".
                policy_action, challenger_action = actions[0], actions[1]
                
                # Step for the first agent (our policy)
                self.env.step(policy_action)
                
                # Step for the second agent (challenger) and get the resulting state
                obs, _, terminated, truncated, info = self.env.last() # Get state for challenger
                self.env.step(challenger_action) # Challenger acts
                obs, _, terminated, truncated, info = self.env.last() # Get state for our policy again

                done = terminated or truncated
                
                # Collect cumulative rewards for both agents
                rewards_list = [self.env.rewards[self.agents[0]], self.env.rewards[self.agents[1]]]
                
                # Convert observation to torch tensor
                if isinstance(obs, np.ndarray):
                    self.state = torch.from_numpy(obs).float()
                else:
                    self.state = torch.tensor(obs, dtype=torch.float32)
                
                return self.state, rewards_list, done, info
            
            @property
            def observation_space(self) -> Space:
                # Return the stored private attribute
                return self._observation_space
            
            @property
            def action_space(self) -> Space:
                # Return the stored private attribute
                return self._action_space
        
        return PettingZooWrapper(env)
    
    def register_gymnasium_env(self, name: str, env_id: str, **kwargs) -> None:
        """Register a Gymnasium environment for evaluation."""
        def env_factory():
            env = gym.make(env_id, **kwargs)
            return self._create_gymnasium_wrapper(env)
        
        self.environments[name] = env_factory
        print(f"Registered Gymnasium environment: {name} ({env_id})")
    
    def _create_gymnasium_wrapper(self, env: gym.Env) -> Environment:
        """Create a wrapper for Gymnasium environments."""
        class GymnasiumWrapper(Environment):
            def __init__(self, gym_env):
                self.env = gym_env
                self.state = None
            
            def reset(self) -> torch.Tensor:
                obs, _ = self.env.reset()
                if isinstance(obs, np.ndarray):
                    self.state = torch.from_numpy(obs).float()
                else:
                    self.state = torch.tensor(obs, dtype=torch.float32)
                return self.state
            
            def step(self, actions: List[Union[int, float]]) -> Tuple[torch.Tensor, List[float], bool, Dict]:
                if isinstance(actions, (int, float)):
                    actions = [actions]
                
                # For single-agent environments, use the first action
                action = actions[0] if actions else 0
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Convert observation to torch tensor
                if isinstance(obs, np.ndarray):
                    self.state = torch.from_numpy(obs).float()
                else:
                    self.state = torch.tensor(obs, dtype=torch.float32)
                
                return self.state, [reward], done, info
            
            @property
            def observation_space(self) -> Space:
                return self.env.observation_space
            
            @property
            def action_space(self) -> Space:
                return self.env.action_space
        
        return GymnasiumWrapper(env)
    
    def add_custom_challenger(self, name: str, challenger: ChallengerAgent) -> None:
        """Add a custom challenger agent."""
        self.challengers[name] = challenger
        print(f"Added custom challenger: {name}")
    
    def create_specialist_exploiters(self, policies: Dict[str, nn.Module], 
                                   training_episodes: int = 5000) -> Dict[str, ChallengerAgent]:
        """Create specialist exploiter agents trained against specific policies."""
        exploiters = {}
        
        for policy_name, policy in policies.items():
            print(f"Training exploiter against {policy_name}...")
            
            exploiter = NeuralAdversaryAgent(f"{policy_name}-Buster")
            
            # Train exploiter in parallel
            with ProcessPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                future = executor.submit(
                    self._train_exploiter, exploiter, policy, training_episodes
                )
                trained_exploiter = future.result()
            
            exploiters[f"{policy_name}-Buster"] = trained_exploiter
            self.challengers[f"{policy_name}-Buster"] = trained_exploiter
        
        return exploiters
    
    def _train_exploiter(self, exploiter: NeuralAdversaryAgent, 
                        target_policy: nn.Module, episodes: int) -> NeuralAdversaryAgent:
        """Train an exploiter against a target policy."""
        env = self._create_default_environment()
        target_policy.eval()
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(self.config.max_episode_steps):
                # Get target policy action
                with torch.no_grad():
                    target_action = self._get_policy_action(target_policy, state)
                
                # Get exploiter action
                exploiter_action = exploiter.act(state)
                
                # Step environment
                next_state, rewards, done, _ = env.step([exploiter_action, target_action])
                exploiter_reward = rewards[0]
                
                # Update exploiter
                exploiter.update(exploiter_reward, state, exploiter_action)
                
                episode_reward += exploiter_reward
                state = next_state
                
                if done:
                    break
            
            if episode % 1000 == 0:
                print(f"Exploiter training episode {episode}, reward: {episode_reward:.3f}")
        
        return exploiter
    
    def evaluate_policy(self, policy: nn.Module, policy_name: str = "Policy", 
                       environments: Optional[List[str]] = None) -> RobustnessMetrics:
        """Comprehensive policy evaluation across all challengers and environments."""
        print(f"Starting comprehensive evaluation of {policy_name}")
        
        if environments is None:
            environments = list(self.environments.keys()) or ["default"]
        
        all_results = {}
        
        for env_name in environments:
            env_results = self._evaluate_in_environment(policy, policy_name, env_name)
            all_results[env_name] = env_results
        
        # Compute comprehensive metrics
        metrics = self._compute_robustness_metrics(all_results)
        
        # Log results
        self._log_evaluation_results(policy_name, metrics, all_results)
        
        # Store results
        self.results_history.append({
            'policy_name': policy_name,
            'timestamp': time.time(),
            'metrics': metrics,
            'detailed_results': all_results
        })
        
        return metrics
    
    def _evaluate_in_environment(self, policy: nn.Module, policy_name: str, 
                               env_name: str) -> Dict:
        """
        Evaluate policy in a specific environment by automatically discovering and
        filtering for compatible challengers using action space and game prefix.
        """
        env_data = self.environments.get(env_name)
        
        if not env_data:
            # This part for default environment remains the same
            self.logger.error(f"Environment '{env_name}' not found. Using default.")
            env_factory = self._create_default_environment
            payoff_matrix_for_metrics = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
            # The filter key for the default 'RPS' environment
            filter_key = "RPS"
        else:
            env_factory = env_data["factory"]
            payoff_matrix_for_metrics = env_data.get("payoff_matrix")
            # --- START: IMPROVED FILTERING KEY LOGIC ---
            # Use the specific game_prefix if provided, otherwise fall back to the env_name.
            game_prefix = env_data.get("game_prefix")
            filter_key = game_prefix if game_prefix else env_name
            # --- END: IMPROVED FILTERING KEY LOGIC ---

        temp_env = env_factory()
        env_action_space = temp_env.action_space
        
        active_challengers = {
            name: challenger for name, challenger in self.master_challenger_list.items()
            # Use the new filter_key for the startswith check
            if challenger.compatible_action_space == env_action_space and name.startswith(filter_key)
        }
        
        for name, challenger in self.challengers.items():
            # Apply the same logic to custom challengers
            if challenger.compatible_action_space == env_action_space and name.startswith(filter_key):
                active_challengers[name] = challenger
                self.logger.info(f"Including custom challenger '{name}' for this evaluation.")

        self.logger.info(f"Environment '{env_name}' (Filter Key: '{filter_key}') is compatible with {len(active_challengers)} challengers. Starting evaluation.")
        if not active_challengers:
            self.logger.warning(f"No compatible challengers found for environment '{env_name}'. Skipping.")
            return {}

        # The rest of the function remains unchanged...
        results = {}
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            future_to_challenger = {
                executor.submit(
                    self._evaluate_against_challenger, policy, challenger_name, 
                    challenger, env_factory
                ): challenger_name
                for challenger_name, challenger in active_challengers.items()
            }
            for future in future_to_challenger:
                challenger_name = future_to_challenger[future]
                try:
                    challenger_results = future.result()
                    results[challenger_name] = challenger_results
                except Exception as exc:
                    self.logger.error(f"Evaluation against {challenger_name} in env '{env_name}' failed: {exc}")

        results['_payoff_matrix'] = payoff_matrix_for_metrics
        return results
    
    def _evaluate_against_challenger(self, policy: nn.Module, challenger_name: str,
                                   challenger: ChallengerAgent, env_factory: Callable) -> Dict:
        """Evaluate policy against a specific challenger."""
        policy.eval()
        
        total_reward = 0.0
        wins = losses = draws = 0
        episode_rewards = []
        trajectories = [] if self.config.save_trajectories else None
        
        for episode in range(self.config.num_episodes):
            env = env_factory()
            state = env.reset()
            challenger.reset()
            
            episode_reward = 0
            trajectory = [] if self.config.save_trajectories else None
            
            # --- START: CORRECTED CODE FOR STATEFUL CHALLENGERS ---
            # This history tracks the actions taken by the policy being evaluated.
            policy_action_history = []
            
            for step in range(self.config.max_episode_steps):
                # Get policy action
                with torch.no_grad():
                    policy_action = self._get_policy_action(policy, state)
                
                # Get challenger action, providing it with the policy's history
                if hasattr(challenger, 'act'):
                    # Pass the history of the opponent's (the policy's) actions
                    challenger_action = challenger.act(state, opponent_history=policy_action_history)
                else:
                    challenger_action = challenger(state)
                
                # Append the policy's current action to its history for the next step
                policy_action_history.append(policy_action)
                # --- END: CORRECTED CODE ---
                
                # Step environment
                next_state, rewards, done, info = env.step([policy_action, challenger_action])
                policy_reward = rewards[0]
                
                # Update challenger
                if hasattr(challenger, 'update'):
                    challenger.update(-policy_reward, state, challenger_action)
                
                episode_reward += policy_reward
                
                if self.config.save_trajectories:
                    trajectory.append({
                        'state': state.clone(),
                        'policy_action': policy_action,
                        'challenger_action': challenger_action,
                        'reward': policy_reward
                    })
                
                state = next_state
                if done:
                    break
            
            total_reward += episode_reward
            episode_rewards.append(episode_reward)
            
            if episode_reward > 0:
                wins += 1
            elif episode_reward < 0:
                losses += 1
            else:
                draws += 1
            
            if self.config.save_trajectories:
                trajectories.append(trajectory)
        
        results = {
            'avg_reward': total_reward / self.config.num_episodes,
            'win_rate': wins / self.config.num_episodes,
            'loss_rate': losses / self.config.num_episodes,
            'draw_rate': draws / self.config.num_episodes,
            'reward_std': np.std(episode_rewards),
            'min_reward': min(episode_rewards),
            'max_reward': max(episode_rewards)
        }
        
        if self.config.save_trajectories:
            results['trajectories'] = trajectories
        
        # Compute exploitability if enabled
        if self.config.compute_exploitability and challenger_name.endswith('-Buster'):
            exploitability = self._compute_exploitability(episode_rewards)
            results['exploitability'] = exploitability
        
        return results


    def _compute_robustness_metrics(self, all_results: Dict) -> RobustnessMetrics:
        """Compute comprehensive robustness metrics with enhanced rigor."""
        all_win_rates = []
        all_rewards = []
        
        # --- START: CORRECTED CODE ---
        # Iterate over each environment's results
        for env_results in all_results.values():
            # Iterate over the items (key-value pairs) in the environment's results
            for key, challenger_results in env_results.items():
                # Check if the value is a dictionary (i.e., actual challenger results)
                # This skips metadata like '_payoff_matrix' which is a numpy array.
                if isinstance(challenger_results, dict):
                    all_win_rates.append(challenger_results['win_rate'])
                    all_rewards.append(challenger_results['avg_reward'])
        # --- END: CORRECTED CODE ---
        
        metrics = RobustnessMetrics(
            overall_win_rate=np.mean(all_win_rates) if all_win_rates else 0.0,
            min_win_rate=np.min(all_win_rates) if all_win_rates else 0.0,
            max_win_rate=np.max(all_win_rates) if all_win_rates else 0.0,
            win_rate_std=np.std(all_win_rates) if all_win_rates else 0.0,
            avg_reward=np.mean(all_rewards) if all_rewards else 0.0,
            worst_case_reward=np.min(all_rewards) if all_rewards else 0.0
        )
        
        # Compute advanced metrics
        if self.config.compute_exploitability:
            metrics.exploitability = self._compute_overall_exploitability(all_results)
        
        metrics.regret = self._compute_regret(all_results)
        metrics.nash_conv = self._compute_nash_convergence(all_results)
        
        # Compute transfer learning metrics if enabled
        if self.config.compute_transfer_metrics:
            metrics.forward_transfer = self._compute_forward_transfer(all_results)
            metrics.backward_transfer = self._compute_backward_transfer(all_results)
        
        # Compute population diversity metrics if enabled
        if self.config.compute_population_diversity:
            metrics.population_diversity = self._compute_population_diversity(all_results)
            metrics.population_entropy = self._compute_population_entropy(all_results)
            metrics.jensen_shannon_divergence = self._compute_jensen_shannon_divergence(all_results)
        
        # Compute Nash equilibrium distance if nashpy is available
        if self.config.use_nashpy_metrics and NASH_AVAILABLE:
            metrics.nash_equilibrium_distance = self._compute_nash_equilibrium_distance(all_results)
        
        # Check if regret bound is achieved
        metrics.regret_bound_achieved = metrics.regret <= self.config.regret_bound
        
        return metrics



    def _compute_nash_convergence(self, all_results: Dict) -> float:
        """Compute Nash convergence using formal game theory metrics."""
        if NASH_AVAILABLE:
            return self._compute_nashpy_convergence(all_results)
        else:
            return self._compute_simplified_nash_convergence(all_results)
    
    def _compute_nashpy_convergence(self, all_results: Dict) -> float:
        """Compute Nash convergence using nashpy library."""
        try:
            # Build payoff matrix from results
            payoff_matrix = self._build_payoff_matrix(all_results)
            
            # Create game using nashpy
            game = nash.Game(payoff_matrix)
            
            # Find Nash equilibria
            equilibria = list(game.support_enumeration())
            
            if not equilibria:
                return 0.0
            
            # Compute distance to Nash equilibrium
            current_strategy = self._extract_current_strategy(all_results)
            min_distance = float('inf')
            
            for equilibrium in equilibria:
                distance = self._compute_strategy_distance(current_strategy, equilibrium)
                min_distance = min(min_distance, distance)
            
            # Normalize to [0, 1] where 1 is perfect Nash convergence
            nash_conv = max(0, 1.0 - min_distance)
            return nash_conv
            
        except Exception as e:
            print(f"Nash convergence computation failed: {e}")
            return self._compute_simplified_nash_convergence(all_results)
    
    def _compute_simplified_nash_convergence(self, all_results: Dict) -> float:
        """Simplified Nash convergence computation when nashpy is not available."""
        win_rates = []
        
        # --- START: CORRECTED CODE ---
        for env_results in all_results.values():
            for results in env_results.values():
                # Check if the item is a dictionary to skip metadata
                if isinstance(results, dict):
                    win_rates.append(results['win_rate'])
        # --- END: CORRECTED CODE ---
        
        if not win_rates:
            return 0.0
        
        # Nash equilibrium in RPS should have win_rate ≈ 1/3
        nash_target = 1/3
        nash_deviation = np.mean([abs(wr - nash_target) for wr in win_rates])
        return max(0, 1.0 - nash_deviation * 3)  # Normalize to [0, 1]

    def _compute_forward_transfer(self, all_results: Dict) -> float:
        """Compute forward transfer - ability to perform well on new tasks."""
        # ... (rest of the docstring)
        
        adaptive_challengers = ['AdaptiveCounter', 'NeuralAdversary', 'PopulationBased']
        basic_challengers = ['AlwaysRock', 'AlwaysPaper', 'AlwaysScissors', 'Uniform']
        
        adaptive_performance = []
        basic_performance = []
        
        for env_results in all_results.values():
            # --- START: CORRECTED CODE ---
            for challenger_name, results in env_results.items():
                if not isinstance(results, dict):
                    continue # Skip non-dictionary items like _payoff_matrix
                # --- END: CORRECTED CODE ---
                if challenger_name in adaptive_challengers:
                    adaptive_performance.append(results['win_rate'])
                elif challenger_name in basic_challengers:
                    basic_performance.append(results['win_rate'])
        
        if not adaptive_performance or not basic_performance:
            return 0.0
        
        # Forward transfer is the improvement on adaptive challengers
        baseline_performance = np.mean(basic_performance)
        adaptive_performance_avg = np.mean(adaptive_performance)
        
        forward_transfer = max(0, adaptive_performance_avg - baseline_performance)
        return min(forward_transfer, 1.0)  # Normalize to [0, 1]
    
    def _compute_backward_transfer(self, all_results: Dict) -> float:
        """Compute backward transfer - ability to retain performance on old tasks."""
        # This is a simplified implementation
        # In practice, this would compare performance on old tasks before/after learning
        
        # For now, we'll use consistency across different environments as a proxy
        env_performances = []
        
        for env_name, env_results in all_results.items():
            # --- START: CORRECTED CODE ---
            # Add a check to ensure we only process result dictionaries
            win_rates = [
                results['win_rate'] 
                for results in env_results.values() 
                if isinstance(results, dict)
            ]
            if win_rates:
                env_avg = np.mean(win_rates)
                env_performances.append(env_avg)
            # --- END: CORRECTED CODE ---
        
        if len(env_performances) < 2:
            return 1.0 # If only one environment, performance is perfectly consistent
        
        # Backward transfer is the consistency across environments
        performance_std = np.std(env_performances)
        backward_transfer = max(0, 1.0 - performance_std * 2) # Penalize std more
        
        return backward_transfer

    def _compute_population_diversity(self, all_results: Dict) -> float:
        """Compute population diversity based on strategy variation."""
        strategies = []
        
        for env_results in all_results.values():
            # --- START: CORRECTED CODE ---
            for challenger_name, results in env_results.items():
                if not isinstance(results, dict):
                    continue # Skip non-dictionary items
                # --- END: CORRECTED CODE ---
                strategy_vector = [
                    results['win_rate'],
                    results['avg_reward'],
                    results.get('reward_std', 0.0)
                ]
                strategies.append(strategy_vector)
        
        if len(strategies) < 2:
            return 0.0
        # ... (rest of the function is the same)
        strategies_array = np.array(strategies)
        diversity = 0.0
        count = 0
        
        for i in range(len(strategies_array)):
            for j in range(i + 1, len(strategies_array)):
                distance = np.linalg.norm(strategies_array[i] - strategies_array[j])
                diversity += distance
                count += 1
        
        if count > 0:
            diversity /= count
        
        return min(diversity, 1.0)

    def _compute_population_entropy(self, all_results: Dict) -> float:
        """Compute population entropy as a measure of diversity."""
        performance_levels = []
        
        for env_results in all_results.values():
            # --- START: CORRECTED CODE ---
            for results in env_results.values():
                if isinstance(results, dict):
                    performance_levels.append(results['win_rate'])
            # --- END: CORRECTED CODE ---
        
        if not performance_levels:
            return 0.0
        # ... (rest of the function is the same)
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(performance_levels, bins=bins)
        
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        
        prob = hist / hist.sum()
        entropy = -np.sum(prob * np.log2(prob))
        
        max_entropy = np.log2(len(prob)) if len(prob) > 1 else 1.0
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0
        
        return normalized_entropy

    def _compute_jensen_shannon_divergence(self, all_results: Dict) -> float:
        """Compute Jensen-Shannon divergence between different challenger groups."""
        adaptive_group = []
        basic_group = []
        
        for env_results in all_results.values():
            # --- START: CORRECTED CODE ---
            for challenger_name, results in env_results.items():
                if not isinstance(results, dict):
                    continue
                # --- END: CORRECTED CODE ---
                if challenger_name in ['AdaptiveCounter', 'NeuralAdversary']:
                    adaptive_group.append(results['win_rate'])
                elif challenger_name in ['AlwaysRock', 'AlwaysPaper', 'AlwaysScissors']:
                    basic_group.append(results['win_rate'])
        
        if not adaptive_group or not basic_group:
            return 0.0
        # ... (rest of the function is the same)
        bins = np.linspace(0, 1, 11)
        hist1, _ = np.histogram(adaptive_group, bins=bins)
        hist2, _ = np.histogram(basic_group, bins=bins)
        
        hist1 = hist1 / hist1.sum() if hist1.sum() > 0 else np.zeros_like(hist1)
        hist2 = hist2 / hist2.sum() if hist2.sum() > 0 else np.zeros_like(hist2)
        
        m = 0.5 * (hist1 + hist2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            js_divergence = 0.5 * (
                np.nansum(hist1 * np.log2(hist1 / m)) +
                np.nansum(hist2 * np.log2(hist2 / m))
            )
        
        return min(js_divergence, 1.0) if not np.isnan(js_divergence) else 0.0
    
    def _compute_nash_equilibrium_distance(self, all_results: Dict) -> float:
        """Compute distance to Nash equilibrium using nashpy."""
        try:
            payoff_matrix = self._build_payoff_matrix(all_results)
            game = nash.Game(payoff_matrix)
            equilibria = list(game.support_enumeration())
            
            if not equilibria:
                return 1.0  # Maximum distance if no equilibrium found
            
            # Find the closest equilibrium
            current_strategy = self._extract_current_strategy(all_results)
            min_distance = float('inf')
            
            for equilibrium in equilibria:
                distance = self._compute_strategy_distance(current_strategy, equilibrium)
                min_distance = min(min_distance, distance)
            
            return min_distance
            
        except Exception as e:
            print(f"Nash equilibrium distance computation failed: {e}")
            return 0.5  # Default value
    
    def _build_payoff_matrix(self, all_results: Dict) -> np.ndarray:
        """
        Builds or retrieves the payoff matrix for Nash calculations.
        It prioritizes a formally provided matrix and falls back to a
        heuristic based on win rates if none is available.
        """
        # --- START: CORRECTED CODE ---
        # Check if a formal payoff matrix was passed along with the results.
        # This assumes the evaluation is run on one type of game at a time.
        formal_payoff_matrix = None
        for env_results in all_results.values():
            if '_payoff_matrix' in env_results and env_results['_payoff_matrix'] is not None:
                formal_payoff_matrix = env_results['_payoff_matrix']
                break

        if formal_payoff_matrix is not None:
            self.logger.info("Using provided formal payoff matrix for Nash calculation.")
            return formal_payoff_matrix
        
        self.logger.warning(
            "No formal payoff matrix provided. Falling back to simplified Nash "
            "calculation based on win rates. This is a heuristic and not game-theoretically rigorous."
        )
        
        # --- Fallback logic based on win rates (original simplified code) ---
        default_results = all_results.get('default', {})
        challengers = [k for k in default_results.keys() if k != '_payoff_matrix']
        
        if len(challengers) < 2:
            # Default 3x3 matrix for RPS-like games as a last resort
            return np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        
        matrix = np.zeros((2, 2))
        challengers_to_use = challengers[:2]
        for i, challenger1_name in enumerate(challengers_to_use):
            for j, challenger2_name in enumerate(challengers_to_use):
                if i == j:
                    matrix[i, j] = 0
                else:
                    results1 = default_results[challenger1_name]
                    results2 = default_results[challenger2_name]
                    matrix[i, j] = results1['win_rate'] - results2['win_rate']
        
        return matrix
        # --- END: CORRECTED CODE ---
    
    def _extract_current_strategy(self, all_results: Dict) -> np.ndarray:
        """Extract current strategy from evaluation results."""
        # Simplified: use average win rate as strategy
        win_rates = []
        
        # --- START: CORRECTED CODE ---
        for env_results in all_results.values():
            for results in env_results.values():
                # Check if the item is a dictionary to skip metadata
                if isinstance(results, dict):
                    win_rates.append(results['win_rate'])
        # --- END: CORRECTED CODE ---
        
        if not win_rates:
            return np.array([1/3, 1/3, 1/3])  # Uniform strategy
        
        avg_win_rate = np.mean(win_rates)
        # Convert to strategy probabilities (simplified)
        strategy = np.array([avg_win_rate, (1 - avg_win_rate) / 2, (1 - avg_win_rate) / 2])
        return strategy / strategy.sum()  # Normalize

    def _compute_strategy_distance(self, strategy1: np.ndarray, strategy2: Tuple) -> float:
        """Compute distance between two strategies."""
        # strategy2 is a tuple from nashpy equilibrium
        if len(strategy2) == 2:  # Two-player game
            equilibrium_strategy = strategy2[0]  # First player's strategy
        else:
            equilibrium_strategy = strategy2
        
        # Convert to numpy array if needed
        if not isinstance(equilibrium_strategy, np.ndarray):
            equilibrium_strategy = np.array(equilibrium_strategy)
        
        # Ensure same length
        min_len = min(len(strategy1), len(equilibrium_strategy))
        strategy1 = strategy1[:min_len]
        equilibrium_strategy = equilibrium_strategy[:min_len]
        
        # Compute Euclidean distance
        distance = np.linalg.norm(strategy1 - equilibrium_strategy)
        return distance
    
    def continual_evaluation(self, policy: nn.Module, policy_name: str) -> Dict:
        """Evaluate policy in continual learning setting with task sequences."""
        if not self.config.enable_continual_eval:
            return {}
        
        print(f"Starting continual evaluation of {policy_name}")
        
        continual_results = {
            'task_performance': [],
            'forgetting_scores': [],
            'plasticity_scores': [],
            'adaptation_rates': []
        }
        
        # Generate task sequence
        tasks = self.task_generator.generate_sequence(self.continual_config.num_tasks)
        
        for task_idx, task in enumerate(tasks):
            print(f"Evaluating on task {task_idx + 1}/{len(tasks)}")
            
            # Evaluate on current task
            task_performance = self._evaluate_on_task(policy, task)
            continual_results['task_performance'].append(task_performance)
            
            # Compute forgetting (if not first task)
            if task_idx > 0:
                forgetting_score = self.forgetting_detector.compute_forgetting(
                    continual_results['task_performance'], task_idx
                )
                continual_results['forgetting_scores'].append(forgetting_score)
            
            # Compute plasticity
            plasticity_score = self.plasticity_evaluator.compute_plasticity(
                task_performance, task_idx
            )
            continual_results['plasticity_scores'].append(plasticity_score)
            
            # Compute adaptation rate
            if task_idx > 0:
                adaptation_rate = self._compute_adaptation_rate(
                    continual_results['task_performance'][-2:], task
                )
                continual_results['adaptation_rates'].append(adaptation_rate)
        
        return continual_results
    
    def tournament_evaluation(self, policies: Dict[str, nn.Module]) -> Dict:
        """Run tournament-style evaluation between multiple policies."""
        print("Starting tournament evaluation")
        
        tournament_results = {}
        policy_names = list(policies.keys())
        
        # All vs All tournament
        for i, policy1_name in enumerate(policy_names):
            for j, policy2_name in enumerate(policy_names):
                if i != j:
                    match_result = self._run_tournament_match(
                        policies[policy1_name], policy1_name,
                        policies[policy2_name], policy2_name
                    )
                    tournament_results[f"{policy1_name}_vs_{policy2_name}"] = match_result
        
        # Compute ELO ratings
        elo_ratings = self._compute_elo_ratings(tournament_results, policy_names)
        
        return {
            'match_results': tournament_results,
            'elo_ratings': elo_ratings,
            'champion': max(elo_ratings.items(), key=lambda x: x[1])
        }
    
    def generate_report(self, save_path: Optional[str] = None) -> Dict:
        """Generate comprehensive evaluation report."""
        if not self.results_history:
            print("No evaluation results to report")
            return {}
        
        report = {
            'summary': self._generate_summary(),
            'detailed_analysis': self._generate_detailed_analysis(),
            'visualizations': self._generate_visualizations(),
            'recommendations': self._generate_recommendations()
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved to {save_path}")
        
        return report
    
    # ============================================================================
    # Helper Methods
    # ============================================================================
    
# In benchmark.py -> class EnhancedGauntletBenchmark

    # --- MODIFIED ---
    def _create_fixed_action_bot(self, action: int, action_space: Space) -> ChallengerAgent:
        """Create a bot that always plays a fixed action."""
        class FixedActionBot(ChallengerAgent):
            def __init__(self, fixed_action, space):
                super().__init__(f"FixedAction{fixed_action}")
                self.fixed_action = fixed_action
                self._action_space = space
            
            def act(self, observation, opponent_history=None):
                return self.fixed_action
            
            @property
            def compatible_action_space(self) -> Space:
                return self._action_space

            def update(self, reward, observation, action):
                pass
        
        return FixedActionBot(action, action_space)

    # --- MODIFIED ---
    def _create_random_bot(self, action_space: Space) -> ChallengerAgent:
        """Create a random action bot for a given action space."""
        class RandomBot(ChallengerAgent):
            def __init__(self, space):
                super().__init__("Random")
                self._action_space = space
            
            def act(self, observation, opponent_history=None):
                return self._action_space.sample()

            @property
            def compatible_action_space(self) -> Space:
                return self._action_space
            
            def update(self, reward, observation, action):
                pass
        
        return RandomBot(action_space)

    # --- MODIFIED ---
    def _create_biased_bot(self, probs: List[float], action_space: Space) -> ChallengerAgent:
        """Create a bot with biased action probabilities."""
        class BiasedBot(ChallengerAgent):
            def __init__(self, probabilities, space):
                super().__init__("Biased")
                self.probs = np.array(probabilities)
                self.probs = self.probs / self.probs.sum()
                self._action_space = space
            
            def act(self, observation, opponent_history=None):
                return np.random.choice(len(self.probs), p=self.probs)

            @property
            def compatible_action_space(self) -> Space:
                return self._action_space

            def update(self, reward, observation, action):
                pass
        
        return BiasedBot(probs, action_space)

    # --- MODIFIED ---
    def _create_cyclic_bot(self, cycle: List[int], action_space: Space) -> ChallengerAgent:
        """Create a bot that cycles through actions."""
        class CyclicBot(ChallengerAgent):
            def __init__(self, action_cycle, space):
                super().__init__("Cyclic")
                self.cycle = action_cycle
                self.step = 0
                self._action_space = space
            
            def act(self, observation, opponent_history=None):
                action = self.cycle[self.step % len(self.cycle)]
                self.step += 1
                return action
            
            @property
            def compatible_action_space(self) -> Space:
                return self._action_space

            def update(self, reward, observation, action):
                pass
            
            def reset(self):
                super().reset()
                self.step = 0
        
        return CyclicBot(cycle, action_space)

    # --- MODIFIED ---
    def _create_tit_for_tat_bot(self, action_space: Space) -> ChallengerAgent:
        """Create a Tit-for-Tat bot that copies opponent's last action."""
        class TitForTatBot(ChallengerAgent):
            def __init__(self, space):
                super().__init__("TitForTat")
                self._action_space = space

            def act(self, observation, opponent_history=None):
                if opponent_history:
                    return opponent_history[-1]
                return 0 # Cooperate (or default action) on the first move

            @property
            def compatible_action_space(self) -> Space:
                return self._action_space

            def update(self, reward, observation, action):
                pass

        return TitForTatBot(action_space)

    # --- MODIFIED ---
    def _create_copycat_bot(self, action_space: Space) -> ChallengerAgent:
        """Create a Copycat bot with delayed copying."""
        class CopycatBot(ChallengerAgent):
            def __init__(self, space):
                super().__init__("Copycat")
                self.delay = 1
                self._action_space = space

            def act(self, observation, opponent_history=None):
                if opponent_history and len(opponent_history) >= self.delay:
                    return opponent_history[-self.delay]
                return self._action_space.sample()

            @property
            def compatible_action_space(self) -> Space:
                return self._action_space

            def update(self, reward, observation, action):
                pass
            
            def reset(self):
                super().reset()

        return CopycatBot(action_space)

    # Note: Noisy and Adversarial bots are more complex. For simplicity, we will tie them to a specific action space
    # in the master list builder. A more advanced version could make them configurable.
    
    def _create_noisy_bot(self, noise_level: float = 0.1) -> ChallengerAgent:
        """Create a bot that adds noise to optimal strategy."""
        class NoisyBot(ChallengerAgent):
            def __init__(self, noise):
                super().__init__("Noisy")
                self.noise_level = noise
                self.action_dim = 3  # Default
                self.base_strategy = np.array([1/3, 1/3, 1/3])
            
            def act(self, observation, opponent_history=None):
                # Add noise to base strategy
                noisy_probs = self.base_strategy + np.random.normal(0, self.noise_level, self.action_dim)
                noisy_probs = np.clip(noisy_probs, 0, 1)

                # --- START: CORRECTED CODE ---
                # Add a small epsilon to the denominator to prevent division by zero (NaN)
                # if all probabilities are clipped to zero.
                noisy_sum = noisy_probs.sum()
                if noisy_sum > 0:
                    noisy_probs /= noisy_sum
                else:
                    # Fallback to uniform if sum is zero
                    return np.random.choice(self.action_dim)
                # --- END: CORRECTED CODE ---
                
                return np.random.choice(self.action_dim, p=noisy_probs)
            
            def update(self, reward, observation, action):
                pass
        
        return NoisyBot(noise_level)
    
    def _create_adversarial_noise_bot(self) -> ChallengerAgent:
        """Create a bot that uses adversarial perturbations."""
        class AdversarialNoiseBot(ChallengerAgent):
            def __init__(self):
                super().__init__("AdversarialNoise")
                self.perturbation_strength = 0.1
                self.action_dim = 3  # Default
            
            def act(self, observation, opponent_history=None):
                # Generate adversarial action based on observation
                perturbed_obs = observation + torch.randn_like(observation) * self.perturbation_strength
                # Use perturbed observation to make decision
                return torch.argmax(perturbed_obs).item() % self.action_dim
            
            def update(self, reward, observation, action):
                pass
        
        return AdversarialNoiseBot()
    
    def _create_default_environment(self):
        """Create default Rock-Paper-Scissors environment."""
        class RPSEnvironment(Environment):
            def __init__(self):
                self._observation_space = Box(low=0, high=1, shape=(6,))  # One-hot encoded
                self._action_space = Discrete(3)
                self.state = None

            @property
            def observation_space(self):
                return self._observation_space

            @property
            def action_space(self):
                return self._action_space

            def reset(self):
                self.state = torch.zeros(6)  # Empty state initially
                return self.state

            def step(self, actions):
                action1, action2 = actions[0], actions[1]

                # Update state with one-hot encoding of actions
                self.state = torch.zeros(6)
                self.state[action1] = 1.0
                self.state[3 + action2] = 1.0

                # Compute rewards (Rock-Paper-Scissors logic)
                if action1 == action2:
                    rewards = [0.0, 0.0]  # Draw
                elif (action1 - action2) % 3 == 1:
                    rewards = [1.0, -1.0]  # Player 1 wins
                else:
                    rewards = [-1.0, 1.0]  # Player 2 wins

                return self.state, rewards, True, {}

        return RPSEnvironment()    

    def _get_policy_action(self, policy: nn.Module, state: torch.Tensor) -> Union[int, List[float]]:
        """Get action from policy given state, supporting both discrete and continuous actions."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            if hasattr(policy, 'act'):
                # --- START: CORRECTED CODE ---
                # This is the key change. We now just call the .act() method and use
                # its return value directly, without trying to unpack it. This makes
                # it compatible with your DQNAgent and PPOAgent.
                action = policy.act(state.to(self.device))
                # --- END: CORRECTED CODE ---
                
                if torch.is_tensor(action):
                    if action.dim() == 0:
                        return action.item()
                    else:
                        return action.squeeze().tolist()
                else:
                    return action
            else:
                # This part remains the same for policies without a .act() method.
                output = policy(state.to(self.device))
                
                if output.shape[-1] == 1:
                    return output.squeeze().tolist()
                else:
                    if hasattr(policy, 'is_continuous') and policy.is_continuous:
                        return output.squeeze().tolist()
                    else:
                        action_probs = torch.softmax(output, dim=-1)
                        action = torch.multinomial(action_probs, 1)
                        return action.item()

    def _evaluate_on_task(self, policy: nn.Module, task: Dict) -> float:
        """Evaluate policy on a specific task."""
        # Task-specific evaluation logic
        task_challengers = task.get('challengers', ['Uniform'])
        total_performance = 0.0
        
        for challenger_name in task_challengers:
            if challenger_name in self.challengers:
                challenger = self.challengers[challenger_name]
                # Run evaluation
                env = self._create_default_environment()
                episode_rewards = []
                
                for _ in range(100):  # Shorter evaluation per task
                    state = env.reset()
                    policy_action = self._get_policy_action(policy, state)
                    challenger_action = challenger.act(state) if hasattr(challenger, 'act') else challenger(state)
                    _, rewards, _, _ = env.step([policy_action, challenger_action])
                    episode_rewards.append(rewards[0])
                
                total_performance += np.mean(episode_rewards)
        
        return total_performance / len(task_challengers)
    
    def _compute_exploitability(self, episode_rewards: List[float]) -> float:
        """Compute exploitability based on episode rewards."""
        if not episode_rewards:
            return 1.0
        
        # Exploitability is how much an exploiter can gain
        avg_reward = np.mean(episode_rewards)
        exploitability = max(0, -avg_reward)  # How much exploiter gained
        return min(exploitability, 1.0)
    
    def _compute_overall_exploitability(self, all_results: Dict) -> float:
        """Compute overall exploitability across all exploiter challenges."""
        exploiter_results = []
        
        for env_results in all_results.values():
            for challenger_name, results in env_results.items():
                if challenger_name.endswith('-Buster') and 'exploitability' in results:
                    exploiter_results.append(results['exploitability'])
        
        return np.mean(exploiter_results) if exploiter_results else 0.0
    
    def _compute_regret(self, all_results: Dict) -> float:
        """Compute regret against best possible strategy."""
        all_rewards = []
        
        # --- START: CORRECTED CODE ---
        # Iterate over each environment's results
        for env_results in all_results.values():
            # Iterate over the values (challenger results or metadata)
            for results in env_results.values():
                # Check if the item is a dictionary (i.e., actual challenger results)
                # This ensures we skip metadata like '_payoff_matrix'.
                if isinstance(results, dict):
                    all_rewards.append(results['avg_reward'])
        # --- END: CORRECTED CODE ---
        
        if not all_rewards:
            return 0.0
        
        max_possible_reward = 1.0  # Best case in RPS
        actual_reward = np.mean(all_rewards)
        regret = max_possible_reward - actual_reward
        return max(0, regret)
    
    def _compute_adaptation_rate(self, recent_performance: List[float], task: Dict) -> float:
        """Compute how quickly policy adapts to new task."""
        if len(recent_performance) < 2:
            return 0.0
        
        improvement = recent_performance[-1] - recent_performance[-2]
        return max(0, improvement)  # Only positive adaptation
    
    def _run_tournament_match(self, policy1: nn.Module, name1: str,
                            policy2: nn.Module, name2: str) -> Dict:
        """Run a tournament match between two policies."""
        env = self._create_default_environment()
        
        wins1 = wins2 = draws = 0
        total_episodes = self.config.tournament_rounds * 10
        
        for _ in range(total_episodes):
            state = env.reset()
            
            action1 = self._get_policy_action(policy1, state)
            action2 = self._get_policy_action(policy2, state)
            
            _, rewards, _, _ = env.step([action1, action2])
            
            if rewards[0] > rewards[1]:
                wins1 += 1
            elif rewards[1] > rewards[0]:
                wins2 += 1
            else:
                draws += 1
        
        return {
            f'{name1}_wins': wins1,
            f'{name2}_wins': wins2,
            'draws': draws,
            'win_rate_1': wins1 / total_episodes,
            'win_rate_2': wins2 / total_episodes
        }
    
    def _compute_elo_ratings(self, tournament_results: Dict, policy_names: List[str]) -> Dict[str, float]:
        """Compute ELO ratings from tournament results."""
        elo_ratings = {name: 1500.0 for name in policy_names}  # Initial rating
        K = 32  # ELO K-factor
        
        for match_name, results in tournament_results.items():
            if '_vs_' in match_name:
                name1, name2 = match_name.split('_vs_')
                
                # Expected scores
                expected1 = 1 / (1 + 10**((elo_ratings[name2] - elo_ratings[name1]) / 400))
                expected2 = 1 - expected1
                
                # Actual scores
                total_games = results[f'{name1}_wins'] + results[f'{name2}_wins'] + results['draws']
                actual1 = (results[f'{name1}_wins'] + 0.5 * results['draws']) / total_games
                actual2 = 1 - actual1
                
                # Update ratings
                elo_ratings[name1] += K * (actual1 - expected1)
                elo_ratings[name2] += K * (actual2 - expected2)
        
        return elo_ratings
    
    def _log_evaluation_results(self, policy_name: str, metrics: RobustnessMetrics, 
                              detailed_results: Dict):
        """Log comprehensive evaluation results."""
        print(f"\n{'='*100}")
        print(f"🏆 ENHANCED GAUNTLET EVALUATION: {policy_name}")
        print(f"{'='*100}")
        
        print(f"\n🎯 ROBUSTNESS METRICS:")
        print(f"  Overall Win Rate:     {metrics.overall_win_rate:.3f}")
        print(f"  Minimum Win Rate:     {metrics.min_win_rate:.3f}")
        print(f"  Win Rate Std:         {metrics.win_rate_std:.3f}")
        print(f"  Average Reward:       {metrics.avg_reward:.3f}")
        print(f"  Worst Case Reward:    {metrics.worst_case_reward:.3f}")
        print(f"  Exploitability:       {metrics.exploitability:.3f}")
        print(f"  Regret:              {metrics.regret:.3f}")
        print(f"  Nash Convergence:     {metrics.nash_conv:.3f}")
        print(f"  🏅 ROBUSTNESS SCORE:  {metrics.robustness_score:.3f}")
        
        print(f"\n📊 DETAILED CHALLENGER RESULTS:")
        for env_name, env_results in detailed_results.items():
            print(f"\n  Environment: {env_name}")
            for challenger_name, results in sorted(env_results.items()):
                # --- START: CORRECTED CODE ---
                # Add a check to ensure 'results' is a dictionary before accessing keys.
                if isinstance(results, dict):
                    print(f"    {challenger_name.ljust(20)}: WR={results['win_rate']:.3f}, "
                          f"AR={results['avg_reward']:.3f}, STD={results.get('reward_std', 0):.3f}")
                # --- END: CORRECTED CODE --- 

    def _generate_summary(self) -> Dict:
        """Generate evaluation summary."""
        if not self.results_history:
            return {}
        
        latest_results = self.results_history[-1]
        metrics = latest_results['metrics']
        
        return {
            'policy_name': latest_results['policy_name'],
            'evaluation_timestamp': latest_results['timestamp'],
            'robustness_score': metrics.robustness_score,
            'key_strengths': self._identify_strengths(latest_results),
            'key_weaknesses': self._identify_weaknesses(latest_results),
            'overall_grade': self._compute_overall_grade(metrics)
        }
    
    def _identify_strengths(self, results: Dict) -> List[str]:
        """Identify policy strengths from results."""
        strengths = []
        metrics = results['metrics']
        
        if metrics.overall_win_rate > 0.6:
            strengths.append("High overall win rate")
        if metrics.min_win_rate > 0.3:
            strengths.append("Consistent performance across challengers")
        if metrics.exploitability < 0.2:
            strengths.append("Low exploitability")
        if metrics.win_rate_std < 0.1:
            strengths.append("Stable performance")
        
        return strengths
    
    def _identify_weaknesses(self, results: Dict) -> List[str]:
        """Identify policy weaknesses from results."""
        weaknesses = []
        metrics = results['metrics']
        
        if metrics.min_win_rate < 0.2:
            weaknesses.append("Vulnerable to specific challengers")
        if metrics.exploitability > 0.5:
            weaknesses.append("Highly exploitable")
        if metrics.regret > 0.3:
            weaknesses.append("High regret compared to optimal")
        if metrics.win_rate_std > 0.2:
            weaknesses.append("Inconsistent performance")
        
        return weaknesses
    
    def _compute_overall_grade(self, metrics: RobustnessMetrics) -> str:
        """Compute letter grade based on robustness score."""
        score = metrics.robustness_score
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C+"
        elif score >= 0.4:
            return "C"
        else:
            return "F"
    
    def _generate_detailed_analysis(self) -> Dict:
        """Generate detailed analysis of evaluation results."""
        return {
            'performance_trends': self._analyze_performance_trends(),
            'challenger_analysis': self._analyze_challenger_performance(),
            'weakness_patterns': self._identify_weakness_patterns(),
            'improvement_suggestions': self._generate_improvement_suggestions()
        }
    
    def _generate_visualizations(self) -> Dict:
        """Generate comprehensive visualization data and plots for results."""
        if not self.results_history:
            return {}
        
        # Set matplotlib style
        plt.style.use(self.config.style)
        
        latest_results = self.results_history[-1]
        policy_name = latest_results['policy_name']
        
        # Prepare data for visualization
        challenger_names = []
        win_rates = []
        avg_rewards = []
        exploitability_scores = []
        
        for env_results in latest_results['detailed_results'].values():
            for challenger_name, results in env_results.items():
                # --- START: CORRECTED CODE ---
                # Add the check to filter out non-dictionary metadata.
                if isinstance(results, dict):
                    challenger_names.append(challenger_name)
                    win_rates.append(results['win_rate'])
                    avg_rewards.append(results['avg_reward'])
                    exploitability_scores.append(results.get('exploitability', 0.0))
                # --- END: CORRECTED CODE ---
        
        # Generate all visualizations
        viz_data = {
            'challenger_performance': self._generate_challenger_performance_plot(
                challenger_names, win_rates, avg_rewards, policy_name
            ),
            'robustness_radar': self._generate_robustness_radar_chart(
                latest_results['metrics'], policy_name
            ),
            'performance_heatmap': self._generate_performance_heatmap(
                latest_results, policy_name
            ),
            'metrics_comparison': self._generate_metrics_comparison_chart(
                latest_results, policy_name
            )
        }
        
        return viz_data

    def _generate_challenger_performance_plot(self, challenger_names: List[str], 
                                           win_rates: List[float], avg_rewards: List[float],
                                           policy_name: str) -> Dict:
        """Generate challenger performance comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Win rates plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(challenger_names)))
        bars1 = ax1.bar(range(len(challenger_names)), win_rates, color=colors)
        ax1.set_title(f'{policy_name} - Win Rates vs Challengers', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Challenger')
        ax1.set_ylabel('Win Rate')
        ax1.set_xticks(range(len(challenger_names)))
        ax1.set_xticklabels(challenger_names, rotation=45, ha='right')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Baseline')
        ax1.legend()
        
        # Add value labels on bars
        for bar, rate in zip(bars1, win_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Average rewards plot
        bars2 = ax2.bar(range(len(challenger_names)), avg_rewards, color=colors)
        ax2.set_title(f'{policy_name} - Average Rewards vs Challengers', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Challenger')
        ax2.set_ylabel('Average Reward')
        ax2.set_xticks(range(len(challenger_names)))
        ax2.set_xticklabels(challenger_names, rotation=45, ha='right')
        ax2.axhline(y=0.0, color='red', linestyle='--', alpha=0.7, label='Zero Baseline')
        ax2.legend()
        
        # Add value labels on bars
        for bar, reward in zip(bars2, avg_rewards):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{reward:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        if self.config.save_visualizations:
            filename = f"{policy_name}_challenger_performance.{self.config.visualization_format}"
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            return {'plot_path': filename, 'data': {'names': challenger_names, 'win_rates': win_rates, 'avg_rewards': avg_rewards}}
        
        return {'data': {'names': challenger_names, 'win_rates': win_rates, 'avg_rewards': avg_rewards}}
    
    def _generate_robustness_radar_chart(self, metrics: RobustnessMetrics, policy_name: str) -> Dict:
        """Generate comprehensive radar chart for robustness metrics."""
        # Categories and values for radar chart
        categories = [
            'Overall Win Rate', 'Min Win Rate', 'Low Exploitability', 
            'Low Regret', 'Nash Convergence', 'Forward Transfer',
            'Population Diversity', 'Low Forgetting'
        ]
        
        values = [
            metrics.overall_win_rate,
            metrics.min_win_rate,
            1.0 - metrics.exploitability,
            1.0 - metrics.regret,
            metrics.nash_conv,
            metrics.forward_transfer,
            metrics.population_diversity,
            1.0 - metrics.forgetting_rate
        ]
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], categories, size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], 
                   color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot data
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=policy_name)
        ax.fill(angles, values, alpha=0.25)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        plt.title(f'{policy_name} - Robustness Radar Chart', size=16, fontweight='bold', pad=20)
        
        # Save plot
        if self.config.save_visualizations:
            filename = f"{policy_name}_robustness_radar.{self.config.visualization_format}"
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            return {'plot_path': filename, 'data': {'categories': categories, 'values': values}}
        
        return {'data': {'categories': categories, 'values': values}}
    

    def _generate_performance_heatmap(self, results: Dict, policy_name: str) -> Dict:
        """Generate comprehensive performance heatmap."""
        # Extract data for heatmap
        env_names = list(results['detailed_results'].keys())
        challenger_names = []
        
        # Check if there are any results to process
        if not env_names or not results['detailed_results'][env_names[0]]:
            return {'data': {'matrix': [], 'envs': [], 'challengers': []}}
            
        # Dynamically get the list of all challengers from the first environment's results
        # --- START: CORRECTED CODE ---
        # Dynamically and safely get the list of all challengers by filtering
        first_env_results = results['detailed_results'][env_names[0]]
        challenger_names = sorted([
            name for name, res in first_env_results.items() if isinstance(res, dict)
        ])
        # --- END: CORRECTED CODE ---
        performance_matrix = []

        # Build performance matrix
        for env_name in env_names:
            env_results = results['detailed_results'][env_name]
            row = []
            for challenger_name in challenger_names:
                challenger_results = env_results.get(challenger_name, {})
                
                # Default values if a challenger result is missing
                win_rate = challenger_results.get('win_rate', 0.0)
                avg_reward = challenger_results.get('avg_reward', 0.0)

                # --- CORRECTION START ---
                # Clip the average reward to the range of single-step rewards [-1, 1].
                # This prevents extreme cumulative rewards from skewing the visualization
                # and fixes the matplotlib warning.
                clipped_avg_reward = np.clip(avg_reward, -1.0, 1.0)
                
                # Combine win rate and a normalized reward for a more stable performance score.
                performance_score = (win_rate + (clipped_avg_reward + 1) / 2) / 2
                # --- CORRECTION END ---

                row.append(performance_score)
            performance_matrix.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create heatmap using seaborn
        heatmap_data = np.array(performance_matrix)
        sns.heatmap(heatmap_data, 
                   xticklabels=challenger_names,
                   yticklabels=env_names,
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlGn', # Red-Yellow-Green colormap is great for performance
                   center=0.5,   # Center the colormap at 0.5 (neutral performance)
                   cbar_kws={'label': 'Performance Score (0=Bad, 1=Good)'},
                   ax=ax)
        
        plt.title(f'{policy_name} - Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Challenger', fontsize=12)
        plt.ylabel('Environment', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        if self.config.save_visualizations:
            filename = f"{policy_name}_performance_heatmap.{self.config.visualization_format}"
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            return {'plot_path': filename, 'data': {'matrix': performance_matrix, 'envs': env_names, 'challengers': challenger_names}}
        
        return {'data': {'matrix': performance_matrix, 'envs': env_names, 'challengers': challenger_names}}
    
    def _generate_metrics_comparison_chart(self, results: Dict, policy_name: str) -> Dict:
        """Generate metrics comparison chart."""
        metrics = results['metrics']
        
        # Define metrics to compare
        metric_names = [
            'Overall Win Rate', 'Min Win Rate', 'Avg Reward', 
            'Low Exploitability', 'Low Regret', 'Nash Convergence',
            'Forward Transfer', 'Population Diversity'
        ]
        
        metric_values = [
            metrics.overall_win_rate,
            metrics.min_win_rate,
            (metrics.avg_reward + 1) / 2,  # Normalize to [0, 1]
            1.0 - metrics.exploitability,
            1.0 - metrics.regret,
            metrics.nash_conv,
            metrics.forward_transfer,
            metrics.population_diversity
        ]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(metric_names)))
        bars = ax.bar(range(len(metric_names)), metric_values, color=colors)
        
        ax.set_title(f'{policy_name} - Metrics Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score (Normalized)')
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add horizontal line for baseline
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Baseline')
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        if self.config.save_visualizations:
            filename = f"{policy_name}_metrics_comparison.{self.config.visualization_format}"
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            return {'plot_path': filename, 'data': {'names': metric_names, 'values': metric_values}}
        
        return {'data': {'names': metric_names, 'values': metric_values}}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on evaluation."""
        if not self.results_history:
            return []
        
        latest_results = self.results_history[-1]
        metrics = latest_results['metrics']
        recommendations = []
        
        if metrics.min_win_rate < 0.3:
            recommendations.append("Consider regularization to improve worst-case performance")
        
        if metrics.exploitability > 0.4:
            recommendations.append("Implement adversarial training to reduce exploitability")
        
        if metrics.win_rate_std > 0.15:
            recommendations.append("Add ensemble methods to improve consistency")
        
        if metrics.nash_conv < 0.7:
            recommendations.append("Fine-tune strategy to better approximate Nash equilibrium")
        
        return recommendations
    
    def save_checkpoint(self, filepath: str):
        """Save benchmark state for reproducibility."""
        checkpoint = {
            'config': self.config,
            'results_history': self.results_history,
            'challengers_state': self._serialize_challengers()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load benchmark state from checkpoint."""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.config = checkpoint['config']
        self.results_history = checkpoint['results_history']
        self._deserialize_challengers(checkpoint['challengers_state'])
        
        print(f"Checkpoint loaded from {filepath}")
    
    def _serialize_challengers(self) -> Dict:
        """Serialize challenger states for checkpointing."""
        # Implementation for serializing challenger states
        return {}
    
    def _deserialize_challengers(self, challenger_data: Dict):
        """Deserialize challenger states from checkpoint."""
        # Implementation for deserializing challenger states
        pass


# ============================================================================
# Utility Classes for Advanced Features
# ============================================================================

class TaskGenerator:
    """Generates diverse tasks for continual learning evaluation."""
    
    def __init__(self):
        self.task_types = ['adversarial', 'cooperative', 'mixed', 'noisy', 'distribution_shift']
    
    def generate_sequence(self, num_tasks: int) -> List[Dict]:
        """Generate a sequence of diverse tasks."""
        tasks = []
        
        for i in range(num_tasks):
            task_type = random.choice(self.task_types)
            task = self._generate_task(task_type, i)
            tasks.append(task)
        
        return tasks
    
    def _generate_task(self, task_type: str, task_id: int) -> Dict:
        """Generate a specific task based on type."""
        if task_type == 'adversarial':
            return {
                'id': task_id,
                'type': task_type,
                'challengers': ['AdaptiveCounter', 'NeuralAdversary'],
                'difficulty': 'hard'
            }
        elif task_type == 'cooperative':
            return {
                'id': task_id,
                'type': task_type,
                'challengers': ['TitForTat', 'Copycat'],
                'difficulty': 'medium'
            }
        elif task_type == 'noisy':
            return {
                'id': task_id,
                'type': task_type,
                'challengers': ['NoisyUniform', 'AdversarialNoise'],
                'difficulty': 'medium'
            }
        else:
            return {
                'id': task_id,
                'type': 'mixed',
                'challengers': random.sample(list(self.challengers.keys()), 3),
                'difficulty': 'varied'
            }

class ForgettingDetector:
    """Detects catastrophic forgetting in continual learning."""
    
    def compute_forgetting(self, task_performance: List[float], current_task: int) -> float:
        """Compute forgetting score based on performance degradation."""
        if current_task < 1:
            return 0.0
        
        # Compare current performance on old tasks vs original performance
        original_performance = task_performance[0]
        current_performance = task_performance[-1]
        
        forgetting = max(0, original_performance - current_performance)
        return forgetting

class PlasticityEvaluator:
    """Evaluates plasticity (ability to learn new tasks)."""
    
    def compute_plasticity(self, task_performance: float, task_id: int) -> float:
        """Compute plasticity score based on learning speed."""
        # Simplified plasticity computation
        baseline_performance = 0.33  # Random performance in RPS
        improvement = max(0, task_performance - baseline_performance)
        
        # Normalize by task difficulty (later tasks assumed harder)
        difficulty_factor = 1.0 + (task_id * 0.01)
        plasticity = improvement / difficulty_factor
        
        return min(plasticity, 1.0)


# ============================================================================
# Integration and Factory Functions
# ============================================================================

def create_gauntlet_benchmark(config_path: Optional[str] = None) -> EnhancedGauntletBenchmark:
    """Factory function to create configured Gauntlet benchmark."""
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = EvaluationConfig(**config_dict)
    else:
        config = EvaluationConfig()
    
    return EnhancedGauntletBenchmark(config)

def load_policies_from_checkpoint(checkpoint_dir: str) -> Dict[str, nn.Module]:
    """Load multiple policies from checkpoint directory."""
    policies = {}
    checkpoint_path = Path(checkpoint_dir)
    
    for policy_file in checkpoint_path.glob("*.pt"):
        policy_name = policy_file.stem
        policy = torch.load(policy_file, map_location='cpu')
        policies[policy_name] = policy
    
    return policies

def run_comprehensive_evaluation(policies: Dict[str, nn.Module], 
                                config: Optional[EvaluationConfig] = None) -> Dict:
    """Run comprehensive evaluation suite on multiple policies."""
    if config is None:
        config = EvaluationConfig()
    
    gauntlet = EnhancedGauntletBenchmark(config)
    
    # Add specialist exploiters
    gauntlet.create_specialist_exploiters(policies)
    
    # Evaluate each policy
    evaluation_results = {}
    
    for policy_name, policy in policies.items():
        print(f"\n🚀 Evaluating {policy_name}...")
        
        # Standard evaluation
        metrics = gauntlet.evaluate_policy(policy, policy_name)
        evaluation_results[policy_name] = {'standard': metrics}
        
        # Continual learning evaluation
        if config.enable_continual_eval:
            continual_results = gauntlet.continual_evaluation(policy, policy_name)
            evaluation_results[policy_name]['continual'] = continual_results
    
    # Tournament evaluation
    tournament_results = gauntlet.tournament_evaluation(policies)
    evaluation_results['tournament'] = tournament_results
    
    # Generate comprehensive report
    report = gauntlet.generate_report()
    evaluation_results['report'] = report
    
    return evaluation_results


# ============================================================================
# Example Usage and Demo
# ============================================================================


if __name__ == "__main__":
    print("Initializing Enhanced Gauntlet Benchmark...")

    config = EvaluationConfig(
        num_episodes=10,
        parallel_workers=1,
        enable_continual_eval=False,
        compute_exploitability=True,
        support_continuous_actions=True,
        support_multi_agent=True,
        save_visualizations=True,
        use_nashpy_metrics=NASH_AVAILABLE,
        compute_transfer_metrics=True,
        compute_population_diversity=True
    )

    # Create benchmark
    gauntlet = EnhancedGauntletBenchmark(config)

    # --- Create a simple RPS policy for the default environment ---
    class SimpleRPSPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_continuous = False
            # For RPS, input_dim=6 (one-hot for both players), output_dim=3 (rock, paper, scissors)
            self.network = nn.Sequential(
                nn.Linear(6, 32),
                nn.ReLU(),
                nn.Linear(32, 3)
            )

        def forward(self, x):
            return self.network(x)

    # Create the default RPS environment and policy
    env_factory = gauntlet._create_default_environment
    temp_env = env_factory()
    obs_space_dim = temp_env.observation_space.shape[0]
    action_space_dim = temp_env.action_space.n

    test_policy = SimpleRPSPolicy()
    device = torch.device(config.device)
    test_policy.to(device)

    print(f"\n🚀 Enhanced Gauntlet Benchmark initialized successfully!")
    print(f"📊 Loaded {len(gauntlet.challengers)} challenger agents.")
    print(f"🌍 Registered {len(gauntlet.environments)} environments (default RPS only).")

    # --- Run evaluation on the RPS environment ---
    print(f"\n🔬 Running comprehensive evaluation on 'RPS' (default environment)...")
    try:
        metrics = gauntlet.evaluate_policy(
            test_policy,
            "TestRPSPolicy",
            environments=None  # This will use the default RPS environment
        )

        print("\n📊 Evaluation Complete. Metrics:")
        print(f"   Robustness Score: {metrics.robustness_score:.3f}")
        print(f"   Overall Win Rate: {metrics.overall_win_rate:.3f}")

        # Generate and save the final report
        report = gauntlet.generate_report("enhanced_gauntlet_report.json")
        print(f"\n📄 Report generated: enhanced_gauntlet_report.json")

    except Exception as e:
        import traceback
        print(f"❌ Evaluation failed: {e}")
        traceback.print_exc()

    print(f"\n✅ Enhanced Gauntlet Benchmark process finished!")