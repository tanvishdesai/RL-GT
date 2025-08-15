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

# Optional OpenSpiel dependency for extensive-game exploitability
try:
    import pyspiel as openspiel
    OPENSPIEL_AVAILABLE = True
except Exception:
    OPENSPIEL_AVAILABLE = False
    openspiel = None
    # Silent: only used if available

# Optional mapping from our env names to OpenSpiel game strings
OPENSPIEL_ENV_MAP = {
    "KuhnPoker": "kuhn_poker",
    "LeducPoker": "leduc_poker",
}

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
import warnings

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
    # May be unavailable if no formal (A,B) is provided
    nash_conv: Optional[float] = None
    
    # New metrics for transfer learning and population analysis
    forward_transfer: float = 0.0
    backward_transfer: float = 0.0
    population_entropy: float = 0.0
    jensen_shannon_divergence: float = 0.0
    nash_equilibrium_distance: float = 0.0
    regret_bound_achieved: bool = False
    # --- NEW (general-sum support) ---
    general_sum: bool = False
    cooperation_rate: float = 0.0
    social_welfare: float = 0.0
    cc_rate_tft: float = 0.0
    # Domain-specific extras (not folded into robustness score)
    ipd_exploitability_proxy: float = 0.0
    cc_rate_grudger: float = 0.0
    
    @property
    def robustness_score(self) -> float:
        """Comprehensive robustness score combining multiple metrics.
        Ensures each component is on [0,1] and the weighted sum also stays in [0,1].
        """
        def clip01(x: float) -> float:
            try:
                return float(min(1.0, max(0.0, x)))
            except Exception:
                return 0.0

        # Normalize reward-like signals to [0,1] using dynamic ranges when available.
        # Fallback to legacy [-1,1] and [-2,2] assumptions if ranges are not provided.
        if hasattr(self, 'reward_min') and hasattr(self, 'reward_max') and isinstance(getattr(self, 'reward_min'), (int, float)) and isinstance(getattr(self, 'reward_max'), (int, float)) and getattr(self, 'reward_max') > getattr(self, 'reward_min'):
            avg_reward_n = clip01((self.avg_reward - getattr(self, 'reward_min')) / (getattr(self, 'reward_max') - getattr(self, 'reward_min')))
        else:
            avg_reward_n = clip01((self.avg_reward + 1.0) / 2.0)

        if hasattr(self, 'social_welfare'):
            if hasattr(self, 'social_welfare_min') and hasattr(self, 'social_welfare_max') and isinstance(getattr(self, 'social_welfare_min'), (int, float)) and isinstance(getattr(self, 'social_welfare_max'), (int, float)) and getattr(self, 'social_welfare_max') > getattr(self, 'social_welfare_min'):
                social_welfare_n = clip01((self.social_welfare - getattr(self, 'social_welfare_min')) / (getattr(self, 'social_welfare_max') - getattr(self, 'social_welfare_min')))
            else:
                social_welfare_n = clip01((self.social_welfare + 2.0) / 4.0)
        else:
            social_welfare_n = 0.0
        low_exploitability = clip01(1.0 - self.exploitability)
        low_regret = clip01(1.0 - self.regret)
        low_variance = clip01(1.0 - self.win_rate_std)
        fwd_transfer = clip01(self.forward_transfer)
        population_div = clip01(self.population_diversity)
        min_wr = clip01(self.min_win_rate)
        overall_wr = clip01(self.overall_win_rate)

        if self.general_sum:
            # General-sum emphasis
            weights = {
                'avg_reward': 0.30,
                'social_welfare': 0.20,
                'cooperation': 0.15,
                'low_exploitability': 0.12,
                'low_regret': 0.10,
                'population_diversity': 0.08,
                'low_variance': 0.05,
            }
            score = (
                weights['avg_reward'] * avg_reward_n +
                weights['social_welfare'] * social_welfare_n +
                weights['cooperation'] * clip01(self.cooperation_rate) +
                weights['low_exploitability'] * low_exploitability +
                weights['low_regret'] * low_regret +
                weights['population_diversity'] * population_div +
                weights['low_variance'] * low_variance
            )
        else:
            # Zero-sum emphasis
            weights = {
                'overall_wr': 0.25,
                'min_wr': 0.15,
                'low_exploitability': 0.12,
                'low_regret': 0.12,
                'adaptation_rate': 0.10,
                'plasticity': 0.08,
                'forward_transfer': 0.08,
                'low_forgetting': 0.05,
                'population_diversity': 0.05,
            }
            score = (
                weights['overall_wr'] * overall_wr +
                weights['min_wr'] * min_wr +
                weights['low_exploitability'] * low_exploitability +
                weights['low_regret'] * low_regret +
                weights['adaptation_rate'] * clip01(self.adaptation_rate) +
                weights['plasticity'] * clip01(self.plasticity_score) +
                weights['forward_transfer'] * fwd_transfer +
                weights['low_forgetting'] * clip01(1.0 - self.forgetting_rate) +
                weights['population_diversity'] * population_div
            )

        return clip01(score)

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
                 action_space: Space = Discrete(3), noise_level: float = 0.0):
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
        self.noise_level = noise_level

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

class MetaLearnerAgent(ChallengerAgent):
    def __init__(self, action_space: Space = Discrete(3)):
        super().__init__("MetaLearner", "medium")
        self.meta = MetaLearner()
        self._action_space = action_space
        self._action_history = deque(maxlen=100)
    
    @property
    def compatible_action_space(self) -> Space:
        return self._action_space
    
    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        adj = self.meta.adjust(self.history, opponent_history or [])
        # Map adjustment {-1,0,1} into valid discrete action space
        if isinstance(self._action_space, Discrete):
            n = int(self._action_space.n)
            base = 0
            return (base + adj) % n
        return 0
    
    def update(self, reward: float, observation: torch.Tensor, action: int):
        self.meta.update(float(reward))
        self._action_history.append(int(action))
    
    def reset(self):
        super().reset()
        self._action_history.clear()

class ForgivingTFTBot(ChallengerAgent):
    def __init__(self, action_space: Space = Discrete(2)):
        super().__init__("ForgivingTFT", "medium")
        self._action_space = action_space
        self.defect_streak = 0
    
    @property
    def compatible_action_space(self) -> Space:
        return self._action_space
    
    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        if not opponent_history:
            return 0
        last = int(opponent_history[-1])
        if last == 1:
            self.defect_streak += 1
        else:
            self.defect_streak = max(0, self.defect_streak - 1)
        if self.defect_streak >= 2:
            return 1
        return last
    
    def update(self, reward: float, observation: torch.Tensor, action: int):
        pass
    
    def reset(self):
        super().reset()
        self.defect_streak = 0

class KuhnBluffer(ChallengerAgent):
    def __init__(self):
        super().__init__("Kuhn_Bluffer", "medium")
        self._action_space = Discrete(2)
    
    @property
    def compatible_action_space(self) -> Space:
        return self._action_space
    
    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        if observation.dim() > 1:
            observation = observation[0]
        card = int(torch.argmax(observation).item())
        if card == 0:
            return 1  # bluff with J
        if card == 2:
            return 0  # slow-play K
        return random.randint(0, 1)
    
    def update(self, reward: float, observation: torch.Tensor, action: int):
        pass

class KuhnConservative(ChallengerAgent):
    def __init__(self):
        super().__init__("Kuhn_Conservative", "medium")
        self._action_space = Discrete(2)
    
    @property
    def compatible_action_space(self) -> Space:
        return self._action_space
    
    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        if observation.dim() > 1:
            observation = observation[0]
        card = int(torch.argmax(observation).item())
        if card < 2:
            return 0
        return 1
    
    def update(self, reward: float, observation: torch.Tensor, action: int):
        pass

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
        
        # Load persisted results history for cross-run trend analysis (minimal records)
        try:
            hist_path = Path('gauntlet_results_history.json')
            if hist_path.exists():
                with open(hist_path, 'r') as f:
                    data = json.load(f)
                for rec in data.get('history', []):
                    score = float(rec.get('robustness_score', 0.0))
                    # Minimal metrics object with robustness_score attribute
                    metrics_obj = type('MetricsStub', (), {'robustness_score': score})()
                    self.results_history.append({
                        'policy_name': rec.get('policy_name', ''),
                        'timestamp': float(rec.get('timestamp', 0.0)),
                        'metrics': metrics_obj,
                        'detailed_results': {}
                    })
        except Exception:
            pass
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("Gauntlet")
        
        # WandB integration
        # if wandb.run is None:
        #     wandb.init(project="gauntlet-marl-benchmark", config=self.config.__dict__)
        
        return logger

    def _analyze_performance_trends(self) -> Dict:
        """
        Analyzes the policy's performance trend over multiple evaluation runs.
        Uses linear regression on robustness scores.
        """
        if len(self.results_history) < 3:
            return {
                "trend": "N/A",
                "details": f"Insufficient data ({len(self.results_history)} runs). Need at least 3 runs to estimate a trend."
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
            'fixed_strategy': [
                # RPS
                'AlwaysRock', 'AlwaysPaper', 'AlwaysScissors',
                # Matching Pennies
                'AlwaysHeads', 'AlwaysTails',
                # IPD
                'AlwaysCooperate', 'AlwaysDefect',
                # Kuhn (placeholder common names)
                'AlwaysBet', 'AlwaysPass',
                # Stag Hunt
                'AlwaysStag', 'AlwaysHare'
            ],
            'biased_strategy': ['BiasedRock', 'BiasedPaper', 'BiasedScissors', 'BiasedStag'],
            'pattern_based': ['CyclicRPS', 'CyclicRSP', 'CycleReverse', 'TitForTat', 'Copycat'],
            'adaptive_learning': ['AdaptiveCounter', 'PopulationBased', 'NeuralAdversary'],
            'noise_robustness': ['NoisyUniform', 'AdversarialNoise', 'Uniform']
        }
        
        # Invert the dictionary for easy lookup
        challenger_map = {challenger: category for category, challengers in challenger_categories.items() for challenger in challengers}
    
        latest_results = self.results_history[-1]
        category_scores = defaultdict(list)
    
        # Aggregate scores by category
        for env_results in latest_results['detailed_results'].values():
            for challenger_name, results in env_results.items():
                if isinstance(results, dict):
                    # NEW: strip game prefix "RPS_", "IPD_", "Kuhn_", etc.
                    base_name = challenger_name.split('_', 1)[-1] if '_' in challenger_name else challenger_name
                    category = challenger_map.get(base_name)
                    if category:
                        category_scores[category].append(results['win_rate'])
    
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
        
        if getattr(metrics, 'nash_conv', None) is not None and metrics.nash_conv < 0.6:
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

    @staticmethod
    def compute_mean_ci(xs, alpha=0.05):
        """Compute mean and 95% confidence interval across seeds for scalar metrics."""
        import numpy as np
        try:
            import scipy.stats as st
        except ImportError:
            st = None
            
        xs = np.array(xs, dtype=float)
        m = xs.mean()
        se = xs.std(ddof=1) / max(1, np.sqrt(len(xs)))
        
        if st is not None and len(xs) > 1:
            h = st.t.ppf(1 - alpha/2, len(xs)-1) * se
        else:
            h = 0.0
            
        return {'mean': float(m), 'ci95': [float(m - h), float(m + h)]}

    def _create_matrix_game_environment(self, A: np.ndarray, B: np.ndarray) -> Environment:
        """Create a simple matrix game environment with given payoff matrices."""
        class MatrixGameEnv(Environment):
            def __init__(self, A, B):
                self.A, self.B = np.asarray(A, float), np.asarray(B, float)
                n = self.A.shape[0]
                self._observation_space = Box(low=0, high=1, shape=(2,), dtype=np.float32)
                self._action_space = Discrete(n)
                self.state = torch.zeros(2, dtype=torch.float32)
            @property
            def observation_space(self): return self._observation_space
            @property
            def action_space(self): return self._action_space
            def reset(self): self.state.zero_(); return self.state
            def step(self, actions):
                i, j = int(actions[0]), int(actions[1])
                r1, r2 = float(self.A[i, j]), float(self.B[i, j])
                return self.state, [r1, r2], True, {'general_sum': True, 'action0_is_cooperate': True}
            def get_legal_actions(self) -> List[int]:
                try:
                    return list(range(int(self._action_space.n)))
                except Exception:
                    return [0, 1]
        return MatrixGameEnv(A, B)

    def _create_openspiel_wrapper(self, game_string: str) -> Environment:
        """Create a minimal 2p wrapper for OpenSpiel turn-based games."""
        if not OPENSPIEL_AVAILABLE:
            raise ImportError("OpenSpiel not available")
        
        game = openspiel.load_game(game_string)
        
        class OpenSpielEnv(Environment):
            def __init__(self, game):
                self.game = game
                self._action_space = Discrete(self.game.num_distinct_actions())
                # Observation uses information state tensor length if available; fallback to a fixed size
                try:
                    info_state_len = self.game.information_state_tensor_size()
                    self._observation_space = Box(low=-1, high=1, shape=(info_state_len,), dtype=np.float32)
                except Exception:
                    self._observation_space = Box(low=0, high=1, shape=(64,), dtype=np.float32)
                self.state = None
            @property
            def observation_space(self): return self._observation_space
            @property
            def action_space(self): return self._action_space

            def _obs(self, state, player_id):
                try:
                    vec = np.array(state.information_state_tensor(player_id), dtype=np.float32)
                except Exception:
                    vec = np.zeros(self._observation_space.shape[0], dtype=np.float32)
                return torch.from_numpy(vec)

            def reset(self) -> torch.Tensor:
                self.state = self.game.new_initial_state()
                # Player 0 to act first in OpenSpiel; return that player's obs
                cur = self.state.current_player()
                if cur < 0:  # chance/terminal
                    while self.state.is_chance_node():
                        outcomes, probs = zip(*self.state.chance_outcomes())
                        self.state.apply_action(np.random.choice(outcomes, p=probs))
                    if self.state.is_terminal():
                        return torch.zeros(self._observation_space.shape[0])
                    cur = self.state.current_player()
                return self._obs(self.state, cur)

            def step(self, actions):
                # actions = [a_policy, a_challenger], but OpenSpiel is turn-based:
                # apply current player's action; then advance until next decision/terminal
                rewards = [0.0, 0.0]
                for _ in range(2):  # apply two moves max (policy, then opponent), if both move this turn cycle
                    cur = self.state.current_player()
                    if cur < 0:  # chance/terminal
                        while self.state.is_chance_node():
                            outcomes, probs = zip(*self.state.chance_outcomes())
                            self.state.apply_action(np.random.choice(outcomes, p=probs))
                        if self.state.is_terminal():
                            player_returns = self.state.returns()
                            return torch.zeros(self._observation_space.shape[0]), player_returns, True, {}
                        cur = self.state.current_player()
                    # choose which action to apply based on cur (0=policy,1=challenger)
                    idx = 0 if cur == 0 else 1
                    proposed = int(actions[idx])
                    # Ensure proposed action is legal for current OpenSpiel state
                    try:
                        legal = self.state.legal_actions()
                        if proposed not in legal:
                            # Map to a random legal action to avoid illegal-action crashes
                            proposed = int(np.random.choice(legal)) if len(legal) > 0 else proposed
                    except Exception:
                        pass
                    self.state.apply_action(proposed)
                # Next obs is for next player to act
                cur = self.state.current_player()
                if self.state.is_terminal():
                    player_returns = self.state.returns()
                    return torch.zeros(self._observation_space.shape[0]), player_returns, True, {}
                return self._obs(self.state, cur), [0.0, 0.0], False, {}
            def get_legal_actions(self) -> List[int]:
                try:
                    if self.state is not None and not self.state.is_terminal():
                        return list(self.state.legal_actions())
                except Exception:
                    pass
                try:
                    return list(range(int(self._action_space.n)))
                except Exception:
                    return []
        return OpenSpielEnv(game)

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
        stag_hunt_space = Discrete(2) # 0:Stag, 1:Hare
        # Note: Leduc action space will be dynamically determined by OpenSpiel game
        if OPENSPIEL_AVAILABLE:
            try:
                leduc_game = openspiel.load_game("leduc_poker")
                leduc_space = Discrete(leduc_game.num_distinct_actions())
            except Exception:
                leduc_space = Discrete(4)  # fallback: fold, call, raise, check
        else:
            leduc_space = Discrete(4)  # fallback

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
        # Additional RPS challengers
        all_challengers["RPS_BiasedPaper"] = self._create_biased_bot([0.1, 0.7, 0.2], rps_space)
        all_challengers["RPS_BiasedScissors"] = self._create_biased_bot([0.2, 0.1, 0.7], rps_space)
        all_challengers["RPS_NoisyUniform"] = self._create_biased_bot([1/3, 1/3, 1/3], rps_space)
        all_challengers["RPS_CycleReverse"] = self._create_cyclic_bot([0, 2, 1], rps_space)

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
        # Additional IPD bots
        all_challengers["IPD_AlwaysAlternate"] = self._create_cyclic_bot([0, 1], ipd_space)
        all_challengers["IPD_Pavlov"] = self._create_cyclic_bot([0, 0, 1, 1], ipd_space)
        all_challengers["IPD_ForgivingTFT"] = ForgivingTFTBot(ipd_space)

        # ==========================================================
        # 3. Kuhn Poker Challengers (Action Space: Discrete(2))
        # ==========================================================
        all_challengers["Kuhn_AlwaysPass"] = self._create_fixed_action_bot(0, kuhn_poker_space)
        all_challengers["Kuhn_AlwaysBet"] = self._create_fixed_action_bot(1, kuhn_poker_space)
        all_challengers["Kuhn_Uniform"] = self._create_random_bot(kuhn_poker_space)
        # Additional simple Kuhn poker heuristics
        all_challengers["Kuhn_BetOnHigh"] = self._create_fixed_action_bot(1, kuhn_poker_space)
        all_challengers["Kuhn_CheckOnLow"] = self._create_fixed_action_bot(0, kuhn_poker_space)
        all_challengers["Kuhn_Bluffer"] = KuhnBluffer()
        all_challengers["Kuhn_Conservative"] = KuhnConservative()
        all_challengers["Kuhn_NeuralAdversary"] = NeuralAdversaryAgent(action_space=kuhn_poker_space)

        # ==========================================================
        # 4. Matching Pennies Challengers (Action Space: Discrete(2))
        # ==========================================================
        all_challengers["Pennies_AlwaysHeads"] = self._create_fixed_action_bot(0, matching_pennies_space)
        all_challengers["Pennies_AlwaysTails"] = self._create_fixed_action_bot(1, matching_pennies_space)
        all_challengers["Pennies_Uniform"] = self._create_random_bot(matching_pennies_space)
        all_challengers["Pennies_Copycat"] = self._create_copycat_bot(matching_pennies_space)
        # Additional pennies bots
        all_challengers["Pennies_BiasedHeads"] = self._create_biased_bot([0.7, 0.3], matching_pennies_space)
        all_challengers["Pennies_BiasedTails"] = self._create_biased_bot([0.3, 0.7], matching_pennies_space)

        # ==========================================================
        # 5. Stag Hunt Challengers (Action Space: Discrete(2))
        # ==========================================================
        all_challengers["StagHunt_AlwaysStag"] = self._create_fixed_action_bot(0, stag_hunt_space)
        all_challengers["StagHunt_AlwaysHare"] = self._create_fixed_action_bot(1, stag_hunt_space)
        all_challengers["StagHunt_TitForTat"] = self._create_copycat_bot(stag_hunt_space)
        all_challengers["StagHunt_BiasedStag"] = self._create_biased_bot([0.7, 0.3], stag_hunt_space)
        all_challengers["StagHunt_Uniform"] = self._create_random_bot(stag_hunt_space)
        all_challengers["StagHunt_ForgivingTFT"] = ForgivingTFTBot(stag_hunt_space)

        # ==========================================================
        # 6. Leduc Poker Challengers (Action Space: Variable)
        # ==========================================================
        all_challengers["Leduc_Uniform"] = self._create_random_bot(leduc_space)
        # Simple heuristic baseline (if we can determine valid actions)
        if OPENSPIEL_AVAILABLE:
            try:
                # Add a simple heuristic that always calls/checks (action 1) when possible
                all_challengers["Leduc_AlwaysCall"] = self._create_fixed_action_bot(1, leduc_space)
                # Add a conservative player that folds often (action 0)  
                all_challengers["Leduc_AlwaysFold"] = self._create_fixed_action_bot(0, leduc_space)
            except Exception:
                pass
        all_challengers["Leduc_Bluffer"] = self._create_biased_bot([0.1, 0.2, 0.7], leduc_space)  # bias to raise (assume 2=raise)
        all_challengers["Leduc_Conservative"] = self._create_biased_bot([0.7, 0.2, 0.1], leduc_space)  # bias to fold
        all_challengers["Leduc_NeuralAdversary"] = NeuralAdversaryAgent(action_space=leduc_space)

        print(f"Built a master list of {len(all_challengers)} challengers (including new additions) for various games.")
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
                           payoff_matrices: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                           game_prefix: Optional[str] = None,
                           zero_sum: bool = True,
                           symmetric_identical_payoffs: bool = False) -> None:
        """
        Register a new environment for evaluation.

        Args:
            name (str): The unique name for the environment (e.g., "MatchingPennies").
            env_factory (Callable): A function that returns a new instance of the environment.
            payoff_matrix (Optional[np.ndarray]): Row player's per-step payoff matrix A.
            payoff_matrices (Optional[Tuple[np.ndarray, np.ndarray]]): Tuple (A, B) for row/col payoffs.
            game_prefix (Optional[str]): A prefix to link this environment to challengers
                                         (e.g., "Pennies"). If None, 'name' is used.
            zero_sum (bool): If True, and only A is provided, derive B = -A.
            symmetric_identical_payoffs (bool): If True and not zero-sum, derive B = A.T.
        """
        self.environments[name] = {
            "factory": env_factory,
            "payoff_matrix": payoff_matrix,
            "payoff_matrices": payoff_matrices,
            "game_prefix": game_prefix,
            "zero_sum": zero_sum,
            "symmetric_identical_payoffs": symmetric_identical_payoffs,
        }
        if game_prefix:
            self.logger.info(f"Registered environment '{name}' with game prefix '{game_prefix}'.")
        else:
            self.logger.info(f"Registered environment '{name}'.")

    def register_stag_hunt(self) -> None:
        """Register Stag Hunt as a general-sum matrix game with standard payoffs."""
        # Example matrices (scaled to a simple range)
        A = np.array([[4, 0],
                      [3, 3]], dtype=float)
        B = A.T.copy()  # symmetric-identical
        
        def env_factory():
            return self._create_matrix_game_environment(A, B)
        
        self.register_environment(
            name="StagHunt",
            env_factory=env_factory,
            payoff_matrices=(A, B),
            game_prefix="StagHunt",
            zero_sum=False,
            symmetric_identical_payoffs=True
        )

    def register_openspiel_game(self, name: str, game_string: str, **kwargs) -> None:
        """Register an OpenSpiel game for evaluation."""
        if not OPENSPIEL_AVAILABLE:
            self.logger.warning("OpenSpiel not available.")
            return
        def env_factory():
            return self._create_openspiel_wrapper(game_string)
        self.register_environment(name, env_factory, game_prefix=kwargs.get('game_prefix', name),
                                  payoff_matrices=None, zero_sum=kwargs.get('zero_sum', True))

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
        # print suppressed for cleanliness; use logger if needed
        
        # Stash reference for downstream EF (OpenSpiel) metrics like NashConv
        try:
            self._last_eval_policy = policy
        except Exception:
            pass

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
        # Persist history for trend analysis across runs
        try:
            history_rec = []
            for rec in self.results_history:
                history_rec.append({
                    'policy_name': rec.get('policy_name', ''),
                    'timestamp': float(rec.get('timestamp', 0.0)),
                    'robustness_score': float(getattr(rec.get('metrics'), 'robustness_score', 0.0)),
                })
            with open('gauntlet_results_history.json', 'w') as f:
                json.dump({'history': history_rec}, f, indent=2)
        except Exception:
            pass
        
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
            # Ensure filter_prefix is defined for the default branch
            filter_prefix = f"{filter_key}_"
        else:
            env_factory = env_data["factory"]
            payoff_matrix_for_metrics = env_data.get("payoff_matrix")
            # --- START: IMPROVED FILTERING KEY LOGIC ---
            # Use the specific game_prefix if provided, otherwise fall back to the env_name.
            game_prefix = env_data.get("game_prefix")
            filter_key = game_prefix if game_prefix else env_name
            # Use an explicit underscore-delimited prefix for challenger name matching
            filter_prefix = f"{filter_key}_"
            # --- END: IMPROVED FILTERING KEY LOGIC ---

        temp_env = env_factory()
        env_action_space = temp_env.action_space
        
        # Helper: structural action-space compatibility check (duck-typed)
        def _spaces_compatible(space_a: Space, space_b: Space) -> bool:
            """Return True when spaces are structurally compatible.
            Be permissive for Discrete spaces because we enforce legality at runtime.
            """
            # Treat any objects with attribute 'n' as Discrete-like
            try:
                if hasattr(space_a, 'n') and hasattr(space_b, 'n'):
                    # Permissive: any Discrete-vs-Discrete pair is acceptable.
                    # We later remap illegal actions to legal ones using env.get_legal_actions().
                    return True
            except Exception:
                pass

            # Treat any objects with attributes 'shape', 'low', 'high' as Box-like
            try:
                is_box_like_a = all(hasattr(space_a, attr) for attr in ('shape', 'low', 'high'))
                is_box_like_b = all(hasattr(space_b, attr) for attr in ('shape', 'low', 'high'))
                if is_box_like_a and is_box_like_b:
                    same_shape = tuple(getattr(space_a, 'shape')) == tuple(getattr(space_b, 'shape'))
                    try:
                        a_low = np.broadcast_to(getattr(space_a, 'low'), getattr(space_a, 'shape'))
                        b_low = np.broadcast_to(getattr(space_b, 'low'), getattr(space_b, 'shape'))
                        a_high = np.broadcast_to(getattr(space_a, 'high'), getattr(space_a, 'shape'))
                        b_high = np.broadcast_to(getattr(space_b, 'high'), getattr(space_b, 'shape'))
                        same_low = np.allclose(a_low, b_low, equal_nan=True)
                        same_high = np.allclose(a_high, b_high, equal_nan=True)
                        return bool(same_shape and same_low and same_high)
                    except Exception:
                        return bool(same_shape)
            except Exception:
                pass

            # Fallback: gymnasium type match
            if isinstance(space_a, Discrete) and isinstance(space_b, Discrete):
                return True
            if isinstance(space_a, Box) and isinstance(space_b, Box):
                try:
                    same_shape = tuple(space_a.shape) == tuple(space_b.shape)
                    same_low = np.allclose(np.broadcast_to(space_a.low, space_a.shape),
                                           np.broadcast_to(space_b.low, space_b.shape), equal_nan=True)
                    same_high = np.allclose(np.broadcast_to(space_a.high, space_a.shape),
                                            np.broadcast_to(space_b.high, space_b.shape), equal_nan=True)
                    return bool(same_shape and same_low and same_high)
                except Exception:
                    return tuple(space_a.shape) == tuple(space_b.shape)
            return False

        active_challengers = {
            name: challenger for name, challenger in self.master_challenger_list.items()
            if name.startswith(filter_prefix) and _spaces_compatible(challenger.compatible_action_space, env_action_space)
        }
        
        for name, challenger in self.challengers.items():
            # Apply the same logic to custom challengers
            if name.startswith(filter_prefix) and _spaces_compatible(challenger.compatible_action_space, env_action_space):
                active_challengers[name] = challenger
                self.logger.info(f"Including custom challenger '{name}' for this evaluation.")

        self.logger.info(f"Environment '{env_name}' (Filter Key: '{filter_key}') is compatible with {len(active_challengers)} challengers. Starting evaluation.")
        try:
            # Debug print suppressed; enable logging if needed
            # print(f"[DEBUG][Gauntlet] env={env_name} episodes={self.config.num_episodes} active={list(active_challengers.keys())}")
            pass
        except Exception:
            pass
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
                    try:
                        wr = challenger_results.get('win_rate', 0.0)
                        ar = challenger_results.get('avg_reward', 0.0)
                        # Debug print suppressed
                        # print(f"[DEBUG][Gauntlet] Completed {env_name} vs {challenger_name}: WR={wr:.3f}, AR={ar:.3f}")
                    except Exception:
                        pass
                except Exception as exc:
                    self.logger.error(f"Evaluation against {challenger_name} in env '{env_name}' failed: {exc}")

        # Attach payoff matrices information for downstream Nash/metrics
        # Prefer explicit (A,B); else derive according to zero-sum/symmetry flags
        env_zero_sum = None
        env_symmetric = None
        A = None
        B = None
        if env_data:
            env_zero_sum = env_data.get("zero_sum", True)
            env_symmetric = env_data.get("symmetric_identical_payoffs", False)
            A = env_data.get("payoff_matrix", None)
            AB = env_data.get("payoff_matrices", None)
            if AB is not None:
                try:
                    A_tuple, B_tuple = AB
                    A = np.array(A_tuple)
                    B = np.array(B_tuple)
                except Exception:
                    A = None
                    B = None
        else:
            # Handle default environment case (RPS)
            # Set proper metadata for default RPS environment to enable regret calculation
            env_zero_sum = True  # RPS is a zero-sum game
            env_symmetric = False
            A = payoff_matrix_for_metrics  # Use the RPS payoff matrix set earlier
            if A is not None:
                A = np.array(A)
        if A is not None and B is None:
            try:
                A = np.array(A)
                if env_zero_sum:
                    B = -A
                elif env_symmetric:
                    B = A.T.copy()
                else:
                    self.logger.warning("No column-player payoff provided and zero_sum/symmetric flags not set. Defaulting to zero-sum assumption (B=-A).")
                    B = -A
            except Exception:
                A, B = None, None
        # Backward compatibility: also keep _payoff_matrix key
        results['_payoff_matrix'] = payoff_matrix_for_metrics
        if A is not None and B is not None:
            results['_payoff_matrices'] = (A, B)
            # Ensure zero_sum flag is properly set - default to True for RPS if not specified
            if env_zero_sum is None:
                env_zero_sum = True  # Default assumption for games like RPS
            results['_zero_sum'] = bool(env_zero_sum)

        # Check for placeholder/proxy matrices and flag to skip Nash metrics
        if env_name == "KuhnPoker" and "_payoff_matrices" in results and results["_payoff_matrices"] is not None:
            A_check, B_check = results["_payoff_matrices"]
            # Check if this is the 2x2 proxy matrix [[0,-1],[1,0]]
            if (np.array_equal(A_check, np.array([[0, -1], [1, 0]])) and 
                np.array_equal(B_check, np.array([[0, 1], [-1, 0]]))):
                results['_skip_nash_metrics'] = True

        # Aggregate policy action distribution and cooperate rate across challengers when available
        try:
            dists = []
            opp_dists = []
            coop_rates = []
            for v in results.values():
                if isinstance(v, dict):
                    if '_policy_action_dist' in v:
                        dists.append(np.array(v['_policy_action_dist'], dtype=float))
                    if '_opponent_action_dist' in v:
                        opp_dists.append(np.array(v['_opponent_action_dist'], dtype=float))
                    if 'policy_action0_rate' in v:
                        coop_rates.append(float(v['policy_action0_rate']))
            if dists:
                avg_dist = np.mean(np.stack(dists, axis=0), axis=0)
                s = float(np.sum(avg_dist))
                if s > 0:
                    avg_dist = (avg_dist / s).tolist()
                results['_policy_action_dist'] = avg_dist
            if opp_dists:
                avg_opp = np.mean(np.stack(opp_dists, axis=0), axis=0)
                s = float(np.sum(avg_opp))
                if s > 0:
                    avg_opp = (avg_opp / s).tolist()
                results['_opponent_action_dist'] = avg_opp
            if coop_rates:
                results['_policy_coop_rate'] = float(np.mean(coop_rates))
        except Exception:
            pass
        
        return results
    
    def _evaluate_against_challenger(self, policy: nn.Module, challenger_name: str,
                                   challenger: ChallengerAgent, env_factory: Callable) -> Dict:
        """Evaluate policy against a specific challenger."""
        # Some policies are plain wrappers (not nn.Module). Guard the eval() call.
        if hasattr(policy, 'eval'):
            policy.eval()
        
        total_reward = 0.0
        wins = losses = draws = 0
        episode_rewards = []
        episode_payoff_diffs = []
        episode_steps_list = []
        trajectories = [] if self.config.save_trajectories else None
        # --- NEW: cooperation/social stats ---
        policy_action0_total = 0
        joint_action00_total = 0
        total_action_count = 0
        # Action distributions for discrete, single-step games
        policy_action_counts: Optional[Dict[int, int]] = None
        opponent_action_counts: Optional[Dict[int, int]] = None
        opp_per_step_rewards: List[float] = []
        
        for episode in range(self.config.num_episodes):
            env = env_factory()
            state = env.reset()
            challenger.reset()
            
            episode_reward = 0
            opp_episode_reward = 0.0
            episode_steps = 0
            trajectory = [] if self.config.save_trajectories else None
            prev_challenger_action = None
            
            # --- START: CORRECTED CODE FOR STATEFUL CHALLENGERS ---
            # This history tracks the actions taken by the policy being evaluated.
            policy_action_history = []
            
            for step in range(self.config.max_episode_steps):
                # Query legal actions from the environment when available
                try:
                    legal_actions = state.new_empty(0)  # placeholder to keep torch in scope
                except Exception:
                    pass
                try:
                    env_legal = env.get_legal_actions()
                except Exception:
                    env_legal = None
                # Get policy action
                with torch.no_grad():
                    policy_action = self._get_policy_action(policy, state)
                    # If env provides legal actions and policy supports select_action(state, legal)
                    if env_legal is not None and hasattr(policy, 'select_action'):
                        try:
                            policy_action = int(policy.select_action(state, env_legal))
                        except TypeError:
                            # Fallback to already computed action
                            pass
                    # Ensure action legality if we have legal actions
                    if env_legal is not None and len(env_legal) > 0 and int(policy_action) not in env_legal:
                        policy_action = int(random.choice(env_legal))
                # Initialize action counter lazily for discrete spaces
                try:
                    if hasattr(env, 'action_space') and isinstance(env.action_space, Discrete):
                        if policy_action_counts is None:
                            policy_action_counts = {i: 0 for i in range(int(env.action_space.n))}
                        policy_action_counts[int(policy_action)] += 1
                except Exception:
                    pass
                
                # Get challenger action, providing it with the policy's history
                if hasattr(challenger, 'act'):
                    # Pass the history of the opponent's (the policy's) actions
                    if env_legal is not None and hasattr(challenger, 'select_action'):
                        try:
                            challenger_action = int(challenger.select_action(state, env_legal))
                        except Exception:
                            challenger_action = challenger.act(state, opponent_history=policy_action_history)
                    else:
                        challenger_action = challenger.act(state, opponent_history=policy_action_history)
                else:
                    challenger_action = challenger(state)
                # Ensure challenger action legality if we have legal actions
                if env_legal is not None and len(env_legal) > 0 and int(challenger_action) not in env_legal:
                    challenger_action = int(random.choice(env_legal))
                # Track opponent action distribution for discrete envs
                try:
                    if hasattr(env, 'action_space') and isinstance(env.action_space, Discrete):
                        if opponent_action_counts is None:
                            opponent_action_counts = {i: 0 for i in range(int(env.action_space.n))}
                        opponent_action_counts[int(challenger_action)] += 1
                except Exception:
                    pass
                
                # Append the policy's current action to its history for the next step
                policy_action_history.append(policy_action)
                # --- END: CORRECTED CODE ---
                
                next_state, rewards, done, info = env.step([policy_action, challenger_action])
                policy_reward = rewards[0]
                opp_reward = rewards[1]

                # Update challenger
                if hasattr(challenger, 'update'):
                    challenger.update(-policy_reward, state, challenger_action)

                episode_reward += policy_reward
                opp_episode_reward += opp_reward
                episode_steps += 1
                # --- NEW: accumulate cooperation-like stats (action==0) ---
                # Only count cooperation-style metrics if the env signals it
                is_general_sum = isinstance(info, dict) and info.get('general_sum', False)
                if is_general_sum and info.get('action0_is_cooperate', False):
                    policy_action0_total += int(policy_action == 0)
                    joint_action00_total += int(policy_action == 0 and challenger_action == 0)
                total_action_count += 1
                prev_challenger_action = challenger_action
                
                if self.config.save_trajectories:
                    trajectory.append({
                        'state': state.clone(),
                        'policy_action': policy_action,
                        'challenger_action': challenger_action,
                        'reward': policy_reward
                    })
                
                state = next_state
                if done:
                    # Debug print suppressed
                    # try:
                    #     print(f"[DEBUG][Gauntlet] episode_end env_step={step+1} policy_r={episode_reward:.2f} opp_r={opp_episode_reward:.2f}")
                    # except Exception:
                    #     pass
                    break
            
            total_reward += episode_reward
            episode_rewards.append(episode_reward)
            episode_payoff_diffs.append(episode_reward - opp_episode_reward)
            episode_steps_list.append(episode_steps if episode_steps > 0 else 1)
            if episode_steps > 0:
                opp_per_step_rewards.append(opp_episode_reward / float(episode_steps))
            
            # Determine outcome by comparing against the opponent's total reward
            if episode_reward > opp_episode_reward:
                wins += 1
            elif episode_reward < opp_episode_reward:
                losses += 1
            else:
                draws += 1
            
            if self.config.save_trajectories:
                trajectories.append(trajectory)
        
        # Compute per-step average reward for better cross-game comparability
        avg_reward_per_step = float(np.mean([
            (r / s) if s > 0 else 0.0 for r, s in zip(episode_rewards, episode_steps_list)
        ])) if episode_rewards else 0.0
        opp_avg_reward_per_step = float(np.mean(opp_per_step_rewards)) if opp_per_step_rewards else 0.0

        results = {
            'avg_reward': total_reward / self.config.num_episodes,
            'avg_reward_per_step': avg_reward_per_step,
            'opp_avg_reward_per_step': opp_avg_reward_per_step,
            'avg_step_count': float(np.mean(episode_steps_list)) if episode_steps_list else 0.0,
            'win_rate': wins / self.config.num_episodes,
            'loss_rate': losses / self.config.num_episodes,
            'draw_rate': draws / self.config.num_episodes,
            'avg_payoff_diff': float(np.mean(episode_payoff_diffs)),
            'payoff_diff_std': float(np.std(episode_payoff_diffs)),
            'reward_std': np.std(episode_rewards),
            'min_reward': min(episode_rewards),
            'max_reward': max(episode_rewards)
        }
        if total_action_count > 0:
            results['policy_action0_rate'] = float(policy_action0_total) / float(total_action_count)
            results['joint_action00_rate'] = float(joint_action00_total) / float(total_action_count)
        if policy_action_counts is not None:
            total = float(sum(policy_action_counts.values()))
            if total > 0:
                results['_policy_action_dist'] = [policy_action_counts[i] / total for i in range(len(policy_action_counts))]
        if opponent_action_counts is not None:
            total_opp = float(sum(opponent_action_counts.values()))
            if total_opp > 0:
                results['_opponent_action_dist'] = [opponent_action_counts[i] / total_opp for i in range(len(opponent_action_counts))]
        
        if self.config.save_trajectories:
            results['trajectories'] = trajectories
        
        # Compute exploitability per-challenger if enabled (fallback path)
        if self.config.compute_exploitability:
            results['exploitability'] = self._compute_exploitability(episode_payoff_diffs)
        
        return results


    def _compute_robustness_metrics(self, all_results: Dict) -> RobustnessMetrics:
        """Compute comprehensive robustness metrics with enhanced rigor."""
        all_win_rates = []
        all_rewards = []
        all_rewards_per_step = []
        all_opp_rewards_per_step = []
        all_policy_a0_rates = []
        all_joint_a00_rates = []
        any_general_sum = False
        
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
                    all_rewards_per_step.append(challenger_results.get('avg_reward_per_step', 0.0))
                    if 'opp_avg_reward_per_step' in challenger_results:
                        all_opp_rewards_per_step.append(challenger_results['opp_avg_reward_per_step'])
                    if 'policy_action0_rate' in challenger_results:
                        all_policy_a0_rates.append(challenger_results['policy_action0_rate'])
                    if 'joint_action00_rate' in challenger_results:
                        all_joint_a00_rates.append(challenger_results['joint_action00_rate'])
        # --- END: CORRECTED CODE ---
        
        metrics = RobustnessMetrics(
            overall_win_rate=np.mean(all_win_rates) if all_win_rates else 0.0,
            min_win_rate=np.min(all_win_rates) if all_win_rates else 0.0,
            max_win_rate=np.max(all_win_rates) if all_win_rates else 0.0,
            win_rate_std=np.std(all_win_rates) if all_win_rates else 0.0,
            avg_reward=np.mean(all_rewards_per_step) if all_rewards_per_step else (np.mean(all_rewards) if all_rewards else 0.0),
            worst_case_reward=np.min(all_rewards_per_step) if all_rewards_per_step else (np.min(all_rewards) if all_rewards else 0.0)
        )

        # Infer dynamic reward ranges from provided payoff matrices (if available)
        reward_min = None
        reward_max = None
        sw_min = None
        sw_max = None
        for env_results in all_results.values():
            AB = env_results.get('_payoff_matrices')
            if AB is not None:
                try:
                    A, B = AB
                    r_min = float(np.min(A))
                    r_max = float(np.max(A))
                    # Update reward range
                    reward_min = r_min if reward_min is None else min(reward_min, r_min)
                    reward_max = r_max if reward_max is None else max(reward_max, r_max)
                    # Social welfare min/max from A+B elementwise
                    sw = A + B
                    sw_min_i = float(np.min(sw))
                    sw_max_i = float(np.max(sw))
                    sw_min = sw_min_i if sw_min is None else min(sw_min, sw_min_i)
                    sw_max = sw_max_i if sw_max is None else max(sw_max, sw_max_i)
                except Exception:
                    pass

        # Attach discovered ranges to metrics for normalization downstream
        if reward_min is not None and reward_max is not None and reward_max > reward_min:
            setattr(metrics, 'reward_min', reward_min)
            setattr(metrics, 'reward_max', reward_max)
        if sw_min is not None and sw_max is not None and sw_max > sw_min:
            setattr(metrics, 'social_welfare_min', sw_min)
            setattr(metrics, 'social_welfare_max', sw_max)

        # Detect general-sum via provided payoff matrices flags if present
        for env_results in all_results.values():
            if env_results.get('_payoff_matrices') is not None and env_results.get('_zero_sum') is not None:
                if env_results.get('_zero_sum') is False:
                    any_general_sum = True
                    break

        # Populate general-sum extras
        if any_general_sum:
            metrics.general_sum = True
            metrics.social_welfare = float(np.mean(all_rewards_per_step) + np.mean(all_opp_rewards_per_step)) if all_rewards_per_step and all_opp_rewards_per_step else float(np.mean(all_rewards))
            metrics.cooperation_rate = float(np.mean(all_policy_a0_rates)) if all_policy_a0_rates else 0.0
            # Conditional cooperation proxies vs TFT/Grudger: use joint a00 rate when available
            metrics.cc_rate_tft = float(np.mean(all_joint_a00_rates)) if all_joint_a00_rates else 0.0
            metrics.cc_rate_grudger = metrics.cc_rate_tft
        
        # Compute advanced metrics
        if self.config.compute_exploitability:
            metrics.exploitability = self._compute_overall_exploitability(all_results)
            # Capture IPD proxy if produced during exploitability computation
            ipd_proxy = getattr(self, '_last_ipd_exploitability_proxy', None)
            if isinstance(ipd_proxy, (int, float)):
                metrics.ipd_exploitability_proxy = float(ipd_proxy)
                # Do not fold proxy into robustness score; keep separate diagnostic
        
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



    def _compute_nash_convergence(self, all_results: Dict) -> Optional[float]:
        """Compute Nash convergence only when formal (A,B) or valid A exists.
        Returns None if not computable or for general-sum games like IPD.
        """
        # Skip for general-sum environments
        for env_results in all_results.values():
            if env_results.get('_zero_sum') is False:
                return None
        
        # Skip if any environment is flagged to skip Nash metrics (e.g., proxy matrices)
        for env_results in all_results.values():
            if env_results.get('_skip_nash_metrics'):
                return None

        # If OpenSpiel is available and environment is known EF game (e.g., Kuhn/Leduc), compute NashConv using best responses
        if OPENSPIEL_AVAILABLE and hasattr(self, '_last_eval_policy'):
            try:
                for env_name in all_results.keys():
                    if env_name in OPENSPIEL_ENV_MAP:
                        game = openspiel.load_game(OPENSPIEL_ENV_MAP[env_name])
                        return self._compute_openspiel_nashconv(game, self._last_eval_policy)
            except Exception as e:
                print(f"OpenSpiel NashConv failed: {e}")

        # Require formal matrices
        payoff_tuple = self._build_payoff_matrix(all_results)
        if payoff_tuple is None:
            return None

        if NASH_AVAILABLE:
            try:
                return self._compute_nashpy_convergence_with_tuple(payoff_tuple, all_results)
            except Exception as e:
                print(f"Nash convergence computation failed: {e}")
                return None
        return None

    def _compute_openspiel_nashconv(self, game: Any, policy: nn.Module) -> Optional[float]:
        """Compute NashConv in OpenSpiel by evaluating best responses against the policy.
        Assumes a two-player zero-sum game. Uses logit policy adapter.
        """
        try:
            # Adapter: convert our torch policy to an OpenSpiel policy
            class TorchPolicyAdapter(openspiel.python.policy.Policy):
                def __init__(self, game, torch_policy: nn.Module, device: torch.device):
                    super().__init__(game, list(range(game.num_players())))
                    self.torch_policy = torch_policy
                    self.device = device
                def action_probabilities(self, state, player_id=None):
                    obs = state.information_state_tensor() if hasattr(state, 'information_state_tensor') else state.observation_tensor()
                    x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        logits = self.torch_policy.actor(x)
                        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    legal = state.legal_actions()
                    # Restrict to legal actions; renormalize
                    masked = np.zeros_like(probs)
                    for a in legal:
                        if a < probs.shape[-1]:
                            masked[a] = probs[a]
                    s = masked.sum()
                    if s <= 0:
                        # uniform over legal
                        masked = np.array([1.0/len(legal) if i in legal else 0.0 for i in range(probs.shape[-1])])
                    else:
                        masked = masked / s
                    return {a: masked[a] for a in legal}

            from open_spiel.python import policy as _osp_policy  # type: ignore
            from open_spiel.python import value  as _osp_value   # type: ignore
            from open_spiel.python.algorithms import best_response as _osp_br  # type: ignore

            adapter = TorchPolicyAdapter(game, policy, self.device)
            # NashConv = sum_i (u_i(BR_i(pi_-i), pi_-i) - u_i(pi))
            # Compute each player's BR value against fixed opponent policy
            total_regret = 0.0
            for player_id in range(game.num_players()):
                br = _osp_br.BestResponsePolicy(game, player_id, adapter)
                # Evaluate values of (br, adapter) profile
                vals_profile = _osp_value.policy_value(game.new_initial_state(), [br, adapter])
                vals_base = _osp_value.policy_value(game.new_initial_state(), [adapter, adapter])
                total_regret += max(0.0, vals_profile[player_id] - vals_base[player_id])
            return float(total_regret)
        except Exception:
            return None
    
    def _compute_nashpy_convergence_with_tuple(self, payoff_tuple: Tuple[np.ndarray, np.ndarray], all_results: Dict) -> Optional[float]:
        """Compute Nash convergence using nashpy when (A,B) is provided or derived.
        Returns None if no equilibrium found.
        """
        if not NASH_AVAILABLE:
            return None
        A, B = payoff_tuple
        game = nash.Game(A, B)
        equilibria = list(game.support_enumeration())
        if not equilibria:
            return None
        current_strategy = self._extract_current_strategy(all_results)
        min_distance = float('inf')
        for equilibrium in equilibria:
            distance = self._compute_strategy_distance(current_strategy, equilibrium)
            min_distance = min(min_distance, distance)
        return max(0.0, 1.0 - min_distance)
    
    def _compute_simplified_nash_convergence(self, all_results: Dict) -> Optional[float]:
        """Deprecated: no longer use heuristic RPS 1/3 fallback. Return None."""
        return None

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
            payoff_tuple = self._build_payoff_matrix(all_results)
            if payoff_tuple is None:
                return 1.0
            game = nash.Game(*payoff_tuple)
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
    
    def _build_payoff_matrix(self, all_results: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieve formal payoff matrices for Nash/exploitability.
        Priority: (A,B) -> A with flags -> None (no heuristic fallback).
        """
        # --- START: UPDATED CODE FOR ZERO-SUM SUPPORT ---
        # Prefer explicit (A,B) if available
        formal_AB = None
        formal_A = None
        for env_results in all_results.values():
            if '_payoff_matrices' in env_results and env_results['_payoff_matrices'] is not None:
                formal_AB = env_results['_payoff_matrices']
                break
            if '_payoff_matrix' in env_results and env_results['_payoff_matrix'] is not None:
                formal_A = env_results['_payoff_matrix']

        if formal_AB is not None:
            self.logger.info("Using provided (A,B) payoff matrices for calculations.")
            A, B = formal_AB
            A = np.array(A)
            B = np.array(B)
            if A.ndim != 2 or B.ndim != 2:
                raise ValueError("Provided payoff matrices must be 2D for calculations.")
            return (A, B)

        if formal_A is not None:
            self.logger.info("Only row-player payoff matrix A provided; deriving B based on flags if possible.")
            A = np.array(formal_A)
            if A.ndim != 2:
                raise ValueError("Provided payoff matrix must be 2D for calculations.")
            # Attempt to infer flags from env metadata
            zero_sum_flag = None
            for env_results in all_results.values():
                if '_zero_sum' in env_results:
                    zero_sum_flag = env_results.get('_zero_sum')
                    break
            if zero_sum_flag is True:
                B = -A
                return (A, B)
            elif zero_sum_flag is False:
                # Try symmetric-identical as a conservative guess when square
                if A.shape[0] == A.shape[1]:
                    B = A.T.copy()
                    return (A, B)
                return None
            else:
                return None
        
        # No formal matrices available
        return None
        # --- END: UPDATED CODE ---
    
    def _extract_current_strategy(self, all_results: Dict) -> np.ndarray:
        """Extract the evaluated policy's action distribution for matrix games.
        Prefers aggregated action distribution collected during evaluation.
        Falls back to uniform distribution with size inferred from payoff matrix or default 3.
        """
        # 1) Prefer env-level aggregated policy action distribution
        for env_results in all_results.values():
            p_dist = env_results.get('_policy_action_dist') if isinstance(env_results, dict) else None
            if p_dist is not None:
                p = np.array(p_dist, dtype=float)
                s = float(p.sum())
                if s > 0 and np.all(np.isfinite(p)):
                    return (p / s).astype(float)
        # 2) Fallback: infer action size from formal payoff matrix if present
        action_count = 3
        for env_results in all_results.values():
            pm = env_results.get('_payoff_matrix') if isinstance(env_results, dict) else None
            if pm is not None:
                A = pm[0] if isinstance(pm, tuple) else np.array(pm)
                if hasattr(A, 'shape') and getattr(A, 'ndim', 0) == 2 and A.shape[0] > 0:
                    action_count = int(A.shape[0])
                    break
        # 3) Uniform
        return np.ones(action_count, dtype=float) / float(action_count)

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
            'recommendations': self._generate_recommendations(),
            'composites': self._generate_composite_scores()
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
                # Handle policies that return (action, log_prob, value) or similar structures
                if isinstance(action, (tuple, list)) and len(action) > 0:
                    action = action[0]
                # --- END: CORRECTED CODE ---
                
                if torch.is_tensor(action):
                    if action.dim() == 0:
                        return action.item()
                    else:
                        return action.squeeze().tolist()
                else:
                    return action
            elif hasattr(policy, 'select_action'):
                # Support agents that expose a select_action API (e.g., OpenSpiel Leduc agents)
                try:
                    action = policy.select_action(state.to(self.device))
                except TypeError:
                    # Some implementations may not accept batched state
                    action = policy.select_action(state.squeeze(0).to(self.device))
                if torch.is_tensor(action):
                    return int(action.item()) if action.dim() == 0 else int(action.squeeze()[0].item())
                if isinstance(action, (tuple, list)):
                    return int(action[0])
                return int(action)
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
    
    def _compute_exploitability(self, episode_payoff_diffs: List[float]) -> float:
        """
        Compute exploitability from per-episode payoff differences:
        positive if policy beats opponent, negative if policy is exploited.
        Returns a normalized score in [0, 1], where higher means more exploitable.
        """
        if not episode_payoff_diffs:
            return 0.0
        mean_diff = float(np.mean(episode_payoff_diffs))  # <0 means exploited
        observed_max = float(np.max(np.abs(episode_payoff_diffs)))
        denom = observed_max if observed_max > 0 else 1.0
        score = max(0.0, -mean_diff) / denom
        return float(min(1.0, score))
    
    def _compute_overall_exploitability(self, all_results: Dict) -> float:
        """Standardize exploitability across domains.
        - Matrix zero-sum (RPS/MP): exact normal-form exploitability via (A, -A) and policy action dist
        - General-sum matrix (e.g., Stag Hunt): compute one-sided best-response gaps for both players using (A,B) and (p,q)
        - IPD: do NOT fold proxy into exploitability; store under ipd_exploitability_proxy
        - Extensive-form (e.g., Kuhn/Leduc): not supported here; return 0.0 unless provided separately
        Fallback: use episode payoff-diff exploitability if available, else 0.0
        """
        per_env_vals: List[float] = []
        ipd_proxy_vals: List[float] = []
        for env_name, env_results in all_results.items():
            if not isinstance(env_results, dict):
                continue
            AB = env_results.get('_payoff_matrices')
            A_only = env_results.get('_payoff_matrix')
            p_dist = env_results.get('_policy_action_dist')
            is_zero_sum = env_results.get('_zero_sum') if '_zero_sum' in env_results else None

            # General-sum matrix games: compute BR gaps for both players if (A,B) and policy/opponent dists exist
            if is_zero_sum is False and AB is not None:
                try:
                    A, B = AB
                    A = np.array(A, dtype=float)
                    B = np.array(B, dtype=float)
                    p = env_results.get('_policy_action_dist')
                    q = env_results.get('_opponent_action_dist')
                    if p is None:
                        # Try to infer from challenger-aggregated per-challenger counts
                        p = env_results.get('_policy_action_dist')
                    if p is not None:
                        p = np.array(p, dtype=float)
                    else:
                        p = np.ones(A.shape[0], dtype=float) / float(A.shape[0])
                    if q is not None:
                        q = np.array(q, dtype=float)
                    else:
                        q = np.ones(A.shape[1], dtype=float) / float(A.shape[1])
                    if p.ndim == 1 and q.ndim == 1 and p.size == A.shape[0] and q.size == A.shape[1]:
                        per_env_vals.append(self._compute_exploitability_general_sum(A, B, p, q))
                except Exception:
                    pass
                # Also capture IPD-like proxy for diagnostics when available
                try:
                    A_only, _ = AB
                    coop_rate = env_results.get('_policy_coop_rate')
                    if coop_rate is None:
                        for v in env_results.values():
                            if isinstance(v, dict) and 'policy_action0_rate' in v:
                                coop_rate = float(v['policy_action0_rate'])
                                break
                    if coop_rate is not None:
                        ipd_proxy_vals.append(self._compute_ipd_exploitability_proxy(np.array(A_only, dtype=float), float(coop_rate)))
                except Exception:
                    pass
                continue

            # Zero-sum normal-form path
            if AB is not None and p_dist is not None and (is_zero_sum is True or is_zero_sum is None):
                try:
                    A, _ = AB
                    A = np.array(A, dtype=float)
                    p = np.array(p_dist, dtype=float)
                    if A.ndim == 2 and p.ndim == 1 and A.shape[0] == p.size:
                        per_env_vals.append(self._compute_exploitability_normal_form(A, p))
                except Exception:
                    pass
            elif A_only is not None and p_dist is not None:
                try:
                    A = np.array(A_only, dtype=float)
                    p = np.array(p_dist, dtype=float)
                    if A.ndim == 2 and p.ndim == 1 and A.shape[0] == p.size:
                        per_env_vals.append(self._compute_exploitability_normal_form(A, p))
                except Exception:
                    pass

        if ipd_proxy_vals:
            # Attach to last metrics if available via side-channel: stored later in _compute_robustness_metrics
            try:
                self._last_ipd_exploitability_proxy = float(np.mean(ipd_proxy_vals))
            except Exception:
                self._last_ipd_exploitability_proxy = 0.0

        if per_env_vals:
            return float(np.mean(per_env_vals))

        # Fallback: use max of per-challenger payoff-diff exploitability if present (only for zero-sum envs)
        fallback_vals: List[float] = []
        for env_results in all_results.values():
            if not isinstance(env_results, dict):
                continue
            if env_results.get('_zero_sum') is False:
                # Skip general-sum environments in exploitability aggregation
                continue
            for res in env_results.values():
                if isinstance(res, dict) and 'exploitability' in res:
                    fallback_vals.append(float(res['exploitability']))
        return float(max(fallback_vals)) if fallback_vals else 0.0

    def _compute_exploitability_normal_form(self, A: np.ndarray, p: np.ndarray) -> float:
        """
        One-sided exploitability for the row player's mixed strategy p in a zero-sum normal-form game A.
        Interpreted as the opponent's (column) best-response value against p; row wants to minimize it.
        Returns value normalized to [0,1] using payoff range.
        """
        A = np.asarray(A, dtype=float)
        p = np.asarray(p, dtype=float)
        p = p / p.sum() if p.sum() > 0 else p

        # Opponent chooses column j that minimizes row payoff; opponent's value = - (row payoff)
        row_payoffs_per_col = p @ A             # shape: (num_cols,)
        opp_best_value = -float(np.min(row_payoffs_per_col))

        a_min = float(np.min(A))
        a_max = float(np.max(A))
        denom = max(1e-8, a_max - a_min)

        # Normalize to [0,1] so larger = more exploitable
        return float(np.clip((opp_best_value - a_min) / denom, 0.0, 1.0))

    def _compute_exploitability_general_sum(self, A: np.ndarray, B: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
        """General-sum exploitability: average unilateral BR improvement for both players.
        Row regret: max_i (A[i]·q) - p^T A q; Col regret: max_j (p^T B[:,j]) - p^T B q.
        Normalize each by respective payoff range; return mean in [0,1].
        """
        # Current payoffs
        v_row = float(p @ A @ q)
        v_col = float(p @ B @ q)
        # Best responses
        row_br = float(np.max(A @ q))
        col_br = float(np.max(p @ B))
        # Regrets (non-negative)
        row_regret = max(0.0, row_br - v_row)
        col_regret = max(0.0, col_br - v_col)
        # Normalize by ranges
        a_min, a_max = float(np.min(A)), float(np.max(A))
        b_min, b_max = float(np.min(B)), float(np.max(B))
        a_den = max(1e-8, a_max - a_min)
        b_den = max(1e-8, b_max - b_min)
        row_n = float(np.clip(row_regret / a_den, 0.0, 1.0))
        col_n = float(np.clip(col_regret / b_den, 0.0, 1.0))
        return float((row_n + col_n) / 2.0)

    def _compute_ipd_exploitability_proxy(self, A: np.ndarray, coop_rate: float) -> float:
        """Stage-game BR exploitability proxy for IPD.
        A = [[R,S],[T,P]] for the evaluated player; coop_rate = P(Cooperate).
        Returns normalized BR payoff in [0,1].
        """
        try:
            A = np.array(A, dtype=float)
            if A.shape != (2, 2):
                return 0.0
            R, S = A[0, 0], A[0, 1]
            T, P = A[1, 0], A[1, 1]
            c = float(np.clip(coop_rate, 0.0, 1.0))
            br_c = c * R + (1.0 - c) * S
            br_d = c * T + (1.0 - c) * P
            br = max(br_c, br_d)
            a_min = float(np.min(A)); a_max = float(np.max(A))
            denom = max(1e-8, a_max - a_min)
            return float(np.clip((br - a_min) / denom, 0.0, 1.0))
        except Exception:
            return 0.0
    
    def _compute_regret(self, all_results: Dict) -> float:
        """Compute regret as the difference between best response value and achieved value.
        
        For zero-sum games: regret = best_response_value - achieved_value
        For general-sum matrix games: use unilateral BR improvement for both players
        
        This correctly measures how much better the agent could have performed if it
        played optimally against the opponents it actually faced.
        """
        general_sum_regrets: List[float] = []
        for env_results in all_results.values():
            if not isinstance(env_results, dict):
                continue
            AB = env_results.get('_payoff_matrices')
            is_zero_sum = env_results.get('_zero_sum') if '_zero_sum' in env_results else None
            if AB is None:
                continue
            if is_zero_sum is False:
                # General-sum: compute unilateral BR improvement averaged across players
                try:
                    A, B = AB
                    A = np.array(A, dtype=float)
                    B = np.array(B, dtype=float)
                    p = env_results.get('_policy_action_dist')
                    q = env_results.get('_opponent_action_dist')
                    if p is None:
                        p = np.ones(A.shape[0], dtype=float) / float(A.shape[0])
                    else:
                        p = np.array(p, dtype=float)
                    if q is None:
                        q = np.ones(A.shape[1], dtype=float) / float(A.shape[1])
                    else:
                        q = np.array(q, dtype=float)
                    # Current payoffs
                    v_row = float(p @ A @ q)
                    v_col = float(p @ B @ q)
                    # Best responses
                    row_br = float(np.max(A @ q))
                    col_br = float(np.max(p @ B))
                    # Regrets
                    row_regret = max(0.0, row_br - v_row)
                    col_regret = max(0.0, col_br - v_col)
                    # Normalize
                    a_den = max(1e-8, float(np.max(A)) - float(np.min(A)))
                    b_den = max(1e-8, float(np.max(B)) - float(np.min(B)))
                    row_n = float(np.clip(row_regret / a_den, 0.0, 1.0))
                    col_n = float(np.clip(col_regret / b_den, 0.0, 1.0))
                    general_sum_regrets.append((row_n + col_n) / 2.0)
                except Exception:
                    pass
                continue
            try:
                A, _ = AB
                A = np.array(A, dtype=float)
                
                # Estimate achieved value from challenger results
                achieved_values: List[float] = []
                for v in env_results.values():
                    if isinstance(v, dict) and 'avg_reward' in v:
                        achieved_values.append(float(v['avg_reward']))
                achieved_value = float(np.mean(achieved_values)) if achieved_values else 0.0
                
                # For zero-sum games, approximate the agent's strategy from its performance
                # against different challengers, then compute best response value
                if A.shape[0] == A.shape[1]:  # Square payoff matrix
                    n_actions = A.shape[0]
                    
                    # Estimate agent's mixed strategy from its performance patterns
                    # This is a heuristic: assume uniform if we can't estimate better
                    estimated_strategy = np.ones(n_actions) / n_actions
                    
                    # For games like RPS where we have specific challenger types,
                    # try to infer strategy from performance against deterministic opponents
                    deterministic_results = {}
                    for challenger_name, results in env_results.items():
                        if isinstance(results, dict) and 'avg_reward' in results:
                            # Try to map challenger names to strategies they represent
                            if 'AlwaysRock' in challenger_name or 'Rock' in challenger_name:
                                deterministic_results[0] = results['avg_reward']
                            elif 'AlwaysPaper' in challenger_name or 'Paper' in challenger_name:
                                deterministic_results[1] = results['avg_reward']
                            elif 'AlwaysScissors' in challenger_name or 'Scissors' in challenger_name:
                                deterministic_results[2] = results['avg_reward']
                    
                    # If we have enough deterministic results, estimate strategy
                    if len(deterministic_results) >= 2 and n_actions <= 3:
                        # For RPS: if agent gets reward r against AlwaysRock,
                        # this tells us about agent's Paper vs (Rock+Scissors) ratio
                        # This is a simplified heuristic estimation
                        try:
                            # Use the deterministic results to estimate mixed strategy
                            # This is approximate but better than uniform assumption
                            if n_actions == 3 and len(deterministic_results) == 3:
                                # Convert rewards to implied frequencies (heuristic)
                                # High reward against AlwaysRock -> agent plays Paper often
                                rewards = [deterministic_results.get(i, 0.0) for i in range(3)]
                                # Transform rewards to probabilities (with safety bounds)
                                probs = [(r + 1.0) / 2.0 for r in rewards]  # Map [-1,1] to [0,1]
                                prob_sum = sum(probs)
                                if prob_sum > 0:
                                    estimated_strategy = np.array(probs) / prob_sum
                        except Exception:
                            pass  # Fall back to uniform
                    
                    # Compute best response value against estimated strategy
                    # Best response chooses the action that maximizes expected payoff
                    expected_payoffs = A @ estimated_strategy
                    best_response_value = float(np.max(expected_payoffs))
                    
                    # Regret is the difference between best response and actual performance
                    regret = max(0.0, best_response_value - achieved_value)
                    return regret
                
            except Exception as e:
                # Fallback: use a simple heuristic based on performance variance
                try:
                    achieved_values: List[float] = []
                    for v in env_results.values():
                        if isinstance(v, dict) and 'avg_reward' in v:
                            achieved_values.append(float(v['avg_reward']))
                    if achieved_values:
                        achieved_value = float(np.mean(achieved_values))
                        worst_performance = float(np.min(achieved_values))
                        best_performance = float(np.max(achieved_values))
                        # Heuristic: regret is related to the gap between best and achieved
                        # In zero-sum games, if there's high variance in performance,
                        # it suggests the agent is exploitable
                        regret = max(0.0, best_performance - achieved_value)
                        return min(regret, 1.0)  # Cap at 1.0 for numerical stability
                except Exception:
                    pass
                continue
        if general_sum_regrets:
            return float(np.mean(general_sum_regrets))
        return 0.0
    
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
        if metrics.nash_conv is not None:
            print(f"  Nash Convergence:     {metrics.nash_conv:.3f}")
        print(f"  🏅 ROBUSTNESS SCORE:  {metrics.robustness_score:.3f}")
        
        print(f"\n📊 DETAILED CHALLENGER RESULTS:")
        for env_name, env_results in detailed_results.items():
            print(f"\n  Environment: {env_name}")
            for challenger_name, results in sorted(env_results.items()):
                # --- START: CORRECTED CODE ---
                # Add a check to ensure 'results' is a dictionary before accessing keys.
                if isinstance(results, dict):
                    ar_display = results.get('avg_reward_per_step', results.get('avg_reward', 0.0))
                    ar_label = 'AR/step' if 'avg_reward_per_step' in results else 'AR'
                    print(f"    {challenger_name.ljust(20)}: WR={results['win_rate']:.3f}, "
                          f"{ar_label}={ar_display:.3f}, STD={results.get('reward_std', 0):.3f}")
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

    def _generate_composite_scores(self) -> Dict:
        """Produce separate composites for zero-sum and general-sum contexts."""
        latest = self.results_history[-1]
        metrics: RobustnessMetrics = latest['metrics']
        zero_sum_score = float(metrics.robustness_score) if not metrics.general_sum else None
        general_sum_score = float(metrics.robustness_score) if metrics.general_sum else None
        return {
            'zero_sum_composite': zero_sum_score,
            'general_sum_composite': general_sum_score,
            'ipd_exploitability_proxy': float(getattr(metrics, 'ipd_exploitability_proxy', 0.0))
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
        
        # Sanitize arrays helper
        def _clean(arr):
            return np.nan_to_num(np.array(arr, dtype=float), nan=0.5, posinf=1.0, neginf=0.0).tolist()

        # Clean values before plotting
        challenger_names = challenger_names
        win_rates = _clean(win_rates)
        avg_rewards = _clean(avg_rewards)
        exploitability_scores = _clean(exploitability_scores)

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
        # Build categories/values dynamically; include Nash only if available
        categories = [
            'Overall Win Rate', 'Min Win Rate', 'Low Exploitability',
            'Low Regret', 'Forward Transfer', 'Population Diversity', 'Low Forgetting'
        ]
        values = [
            float(metrics.overall_win_rate),
            float(metrics.min_win_rate),
            float(max(0.0, 1.0 - metrics.exploitability)),
            float(max(0.0, 1.0 - metrics.regret)),
            float(metrics.forward_transfer),
            float(metrics.population_diversity),
            float(max(0.0, 1.0 - metrics.forgetting_rate)),
        ]
        if getattr(metrics, 'nash_conv', None) is not None:
            categories.insert(4, 'Nash Convergence')
            values.insert(4, float(metrics.nash_conv))
        values = np.nan_to_num(np.array(values, dtype=float), nan=0.5, posinf=1.0, neginf=0.0).tolist()
        
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
        """Generate comprehensive performance heatmap.
        Robust to missing/empty result matrices by emitting a placeholder instead of erroring.
        """
        # Extract data for heatmap
        env_names = list(results.get('detailed_results', {}).keys())
        
        # Early exit if no environments
        if not env_names:
            return {'data': {'matrix': [], 'envs': [], 'challengers': []}}

        # Dynamically get the list of all challengers from the first environment's results
        first_env_results = results['detailed_results'].get(env_names[0], {})
        if not isinstance(first_env_results, dict) or len(first_env_results) == 0:
            # Optional: create a placeholder image
            if self.config.save_visualizations:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.axis('off')
                ax.text(0.5, 0.5, 'No challenger data available for heatmap', ha='center', va='center', fontsize=12)
                filename = f"{policy_name}_performance_heatmap.{self.config.visualization_format}"
                plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                return {'plot_path': filename, 'data': {'matrix': [], 'envs': [], 'challengers': []}}
            return {'data': {'matrix': [], 'envs': [], 'challengers': []}}

        # Dynamically and safely get the list of all challengers by filtering
        challenger_names = sorted([
            name for name, res in first_env_results.items() if isinstance(res, dict)
        ])

        # If we still have no challengers, emit a placeholder
        if not challenger_names:
            if self.config.save_visualizations:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.axis('off')
                ax.text(0.5, 0.5, 'No challenger data available for heatmap', ha='center', va='center', fontsize=12)
                filename = f"{policy_name}_performance_heatmap.{self.config.visualization_format}"
                plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                return {'plot_path': filename, 'data': {'matrix': [], 'envs': env_names, 'challengers': []}}
            return {'data': {'matrix': [], 'envs': env_names, 'challengers': []}}

        # Build performance matrix
        performance_matrix = []
        for env_name in env_names:
            env_results = results['detailed_results'].get(env_name, {})
            row = []
            for challenger_name in challenger_names:
                challenger_results = env_results.get(challenger_name, {})
                # Default values if a challenger result is missing
                win_rate = challenger_results.get('win_rate', 0.0)
                avg_reward = challenger_results.get('avg_reward', 0.0)
                # Clip the average reward to [-1, 1] and combine with win rate
                clipped_avg_reward = np.clip(avg_reward, -1.0, 1.0)
                performance_score = (win_rate + (clipped_avg_reward + 1) / 2) / 2
                row.append(performance_score)
            performance_matrix.append(row)

        # If matrix is empty or degenerate, emit placeholder
        if not performance_matrix or (len(performance_matrix) > 0 and len(performance_matrix[0]) == 0):
            if self.config.save_visualizations:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.axis('off')
                ax.text(0.5, 0.5, 'No data available for performance heatmap', ha='center', va='center', fontsize=12)
                filename = f"{policy_name}_performance_heatmap.{self.config.visualization_format}"
                plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                return {'plot_path': filename, 'data': {'matrix': [], 'envs': env_names, 'challengers': challenger_names}}
            return {'data': {'matrix': [], 'envs': env_names, 'challengers': challenger_names}}

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        heatmap_data = np.nan_to_num(np.array(performance_matrix, dtype=float), nan=0.5, posinf=1.0, neginf=0.0)
        # Ensure bounds and avoid warnings
        vmin, vmax = 0.0, 1.0
        heatmap_data = np.clip(heatmap_data, vmin, vmax)
        sns.heatmap(
            heatmap_data,
            xticklabels=challenger_names,
            yticklabels=env_names,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',  # Red-Yellow-Green colormap is great for performance
            center=0.5,     # Center the colormap at 0.5 (neutral performance)
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': 'Performance Score (0=Bad, 1=Good)'},
            ax=ax
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            plt.title(f'{policy_name} - Performance Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Challenger', fontsize=12)
        plt.ylabel('Environment', fontsize=12)
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

        # Build metrics dynamically; skip missing metrics like Nash when None
        names: List[str] = []
        values: List[float] = []

        def add_metric(name: str, value: Optional[float]):
            if value is None:
                return
            try:
                v = float(value)
                if not np.isnan(v):
                    names.append(name)
                    values.append(max(0.0, min(1.0, v)))
            except Exception:
                pass

        if metrics.general_sum:
            add_metric('Avg Reward', (metrics.avg_reward + 1) / 2)
            add_metric('Social Welfare', (metrics.social_welfare + 2) / 4)
            add_metric('Cooperation Rate', metrics.cooperation_rate)
            add_metric('Low Exploitability', 1.0 - metrics.exploitability)
            add_metric('Low Regret', 1.0 - metrics.regret)
            add_metric('Population Diversity', metrics.population_diversity)
        else:
            add_metric('Overall Win Rate', metrics.overall_win_rate)
            add_metric('Min Win Rate', metrics.min_win_rate)
            add_metric('Avg Reward', (metrics.avg_reward + 1) / 2)
            add_metric('Low Exploitability', 1.0 - metrics.exploitability)
            add_metric('Low Regret', 1.0 - metrics.regret)
            add_metric('Forward Transfer', metrics.forward_transfer)
            add_metric('Population Diversity', metrics.population_diversity)
            if getattr(metrics, 'nash_conv', None) is not None:
                add_metric('Nash Convergence', metrics.nash_conv)
        # Clean metric values to avoid NaNs
        metric_values = np.nan_to_num(np.array(values, dtype=float), nan=0.5, posinf=1.0, neginf=0.0).tolist()
        metric_names = names
        
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
        
        if getattr(metrics, 'nash_conv', None) is not None and metrics.nash_conv < 0.7:
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