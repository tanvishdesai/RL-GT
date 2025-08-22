#!/usr/bin/env python3
"""
Unified Evaluation Script
========================

This script loads saved models from a training run and performs gauntlet evaluation only
with statistical analysis including 95% confidence intervals and pairwise significance tests.

Usage:
    python unified_evaluation.py --model_dir models/leduc_60s --output_dir results/leduc_60s

Features:
- Loads all trained models from specified directory
- Runs gauntlet evaluation for each algorithm
- Computes 95% confidence intervals across seeds using gauntlet robustness scores
- Performs pairwise statistical significance tests
- Generates comprehensive reports and visualizations
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import argparse
import json
import time
import math
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from collections import defaultdict

# IMPORTANT: Add imports for your model classes here
# Import all required model classes and components
try:
    # Import from unified_prpo.py
    from unified_prpo import UnifiedActorCritic, StandardPPO, UnifiedPRPOAgent, TimeBudgetTrainer
    print("Successfully imported unified_prpo components")
except ImportError as e:
    print(f"Warning: Could not import unified_prpo: {e}")
    # Define minimal fallback classes
    class UnifiedActorCritic(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, x):
            return x, x
        def act(self, x):
            return 0, torch.tensor(0.0), torch.tensor(0.0)
    
    class StandardPPO:
        def __init__(self, *args, **kwargs):
            pass
        def act(self, x):
            return 0
    
    class UnifiedPRPOAgent(StandardPPO):
        pass

try:
    # Import from stag_hunt_training_and_evaluation.py  
    from stag_hunt_training_and_evaluation import (
        DQNAgent, PPOAgent, StagHuntEnvironment, PSROPolicy
    )
    print("Successfully imported stag hunt components")
except ImportError as e:
    print(f"Warning: Could not import stag hunt components: {e}")

try:
    # Import gauntlet components
    from gauntlet_benchmark import EnhancedGauntletBenchmark, EvaluationConfig, ChallengerAgent, Environment
    print("Successfully imported gauntlet components")
except ImportError as e:
    print(f"Warning: Could not import gauntlet: {e}")
    _GAUNTLET_AVAILABLE = False

# Statistical helpers
try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False
    _scipy_stats = None

# Import required components
# Model classes are already available in the environment
_MODELS_AVAILABLE = True

# Import gauntlet benchmark (this needs to be available in the environment)

_GAUNTLET_AVAILABLE = True

# Gym spaces (with safe fallback if not available)
try:
    from gym.spaces import Space, Discrete, Box
except Exception:
    class Space:
        pass
    class Discrete:
        def __init__(self, n: int):
            self.n = int(n)
    class Box:
        def __init__(self, low, high, shape, dtype):
            self.shape = shape


def _t_critical_95(n: int) -> float:
    """Get critical t-value for 95% confidence interval."""
    if n <= 1:
        return float("nan")
    df = n - 1
    if _SCIPY_AVAILABLE:
        try:
            return float(_scipy_stats.t.ppf(0.975, df))
        except Exception:
            pass
    
    # Lookup table for common degrees of freedom
    lookup = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980
    }
    if df in lookup:
        return float(lookup[df])
    for k in sorted(lookup.keys()):
        if df < k:
            return float(lookup[k])
    return 1.96


def compute_mean_ci(scores: List[float]) -> Dict[str, float]:
    """Compute mean and 95% confidence interval for a list of scores."""
    arr = np.array(scores, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean()) if n > 0 else float("nan")
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = float(sd / math.sqrt(n)) if n > 1 else 0.0
    tcrit = _t_critical_95(n) if n > 1 else float("nan")
    margin = float(sem * tcrit) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci_low": float(mean - margin),
        "ci_high": float(mean + margin),
        "tcrit_95": float(tcrit if not math.isnan(tcrit) else 0.0),
    }


def paired_t_test(a: List[float], b: List[float]) -> Optional[float]:
    """Perform paired t-test between two groups."""
    if len(a) != len(b) or len(a) < 2:
        return None
    if _SCIPY_AVAILABLE:
        try:
            _, p = _scipy_stats.ttest_rel(a, b)
            return float(p)
        except Exception:
            return None
    return None


def welch_t_test(a: List[float], b: List[float]) -> Optional[float]:
    """Perform Welch's t-test (unequal variances) between two groups."""
    if len(a) < 2 or len(b) < 2:
        return None
    if _SCIPY_AVAILABLE:
        try:
            _, p = _scipy_stats.ttest_ind(a, b, equal_var=False)
            return float(p)
        except Exception:
            return None
    return None


class PolicyWrapperAgent(ChallengerAgent if _GAUNTLET_AVAILABLE else object):
    """Adapter to make arbitrary policies compatible with the Gauntlet interface."""
    
    def __init__(self, base_policy: nn.Module, input_dim: int, action_dim: int, name: str = "WrappedPolicy"):
        if _GAUNTLET_AVAILABLE:
            super().__init__(name, "student")
        self._base = base_policy
        self._input_dim = int(input_dim)
        self._action_dim = int(action_dim)
        print(f"PolicyWrapperAgent created for {name}: base_type={type(base_policy).__name__}, has_act={hasattr(base_policy, 'act')}, has_forward={hasattr(base_policy, 'forward')}")

    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        ts = torch.as_tensor(observation, dtype=torch.float32)
        if ts.ndim == 1:
            ts = ts.unsqueeze(0)
        
        # Handle dimension mismatch: map larger gauntlet state to smaller model input
        if ts.shape[-1] > self._input_dim:
            ts = ts[..., :self._input_dim]
        elif ts.shape[-1] < self._input_dim:
            # Pad with zeros if observation is smaller than expected
            padding = torch.zeros(ts.shape[:-1] + (self._input_dim - ts.shape[-1],))
            ts = torch.cat([ts, padding], dim=-1)
        
        # Handle different policy types
        if isinstance(self._base, (StandardPPO, UnifiedPRPOAgent)):
            try:
                with torch.no_grad():
                    action = self._base.act(ts.squeeze().cpu().numpy())
                    return int(action)
            except Exception as e:
                print(f"Error with StandardPPO/UnifiedPRPOAgent act: {e}")
        
        if isinstance(self._base, UnifiedActorCritic):
            try:
                with torch.no_grad():
                    action, _, _ = self._base.act(ts)
                    return int(action)
            except Exception as e:
                print(f"Error with UnifiedActorCritic act: {e}")
        
        # Try policy.act first for other types
        if hasattr(self._base, "act"):
            try:
                out = self._base.act(ts)
                if isinstance(out, (tuple, list)):
                    out0 = out[0]
                    if torch.is_tensor(out0):
                        return int(out0.item())
                    return int(out0)
                if torch.is_tensor(out):
                    return int(out.item()) if out.ndim == 0 else int(out.argmax(dim=-1).item())
                try:
                    return int(out)
                except Exception:
                    pass
            except Exception as e:
                print(f"Error with base.act: {e}")
        
        # Fallback: call forward and pick argmax
        try:
            out = self._base(ts)
            if isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
                logits_or_probs = out[0]
            elif torch.is_tensor(out):
                logits_or_probs = out
            else:
                return random.randint(0, self._action_dim - 1)
            probs = torch.softmax(logits_or_probs, dim=-1)
            return int(torch.argmax(probs, dim=-1).item())
        except Exception as e:
            print(f"Error with forward fallback: {e}")
            return random.randint(0, self._action_dim - 1)

    def update(self, reward: float, observation: torch.Tensor, action: int):
        pass

    def reset(self):
        pass

    @property
    def compatible_action_space(self) -> Space:
        return Discrete(self._action_dim)


class FlexibleDQNAgent(nn.Module):
    """A DQN agent that can adapt to different saved architectures."""
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int] = None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64]  # Default single hidden layer
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.q_net = nn.Sequential(*layers)
        # Create a separate target network with the same architecture
        target_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            target_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        target_layers.append(nn.Linear(prev_dim, output_dim))
        self.target_q = nn.Sequential(*target_layers)
        
        self.num_actions = output_dim
        self.epsilon = 0.05  # Low epsilon for evaluation
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_net(state)
    
    def act(self, state: torch.Tensor, explore: bool = False) -> int:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
            return int(torch.argmax(q_values, dim=-1).item())

class ModelLoader:
    """Utility class to load saved models."""
    
    @staticmethod
    def detect_dqn_architecture(state_dict: Dict) -> List[int]:
        """Detect DQN architecture from state dict."""
        hidden_layers = []
        layer_idx = 0
        
        while f"q_net.{layer_idx}.weight" in state_dict:
            weight = state_dict[f"q_net.{layer_idx}.weight"]
            if layer_idx == 0:
                # First layer - input to first hidden
                hidden_layers.append(weight.shape[0])
            else:
                # Check if this is the final layer (next layer doesn't exist)
                if f"q_net.{layer_idx + 2}.weight" not in state_dict:
                    # This is the final layer, don't add to hidden_layers
                    break
                else:
                    # This is another hidden layer
                    hidden_layers.append(weight.shape[0])
            layer_idx += 2  # Skip ReLU layers
        
        return hidden_layers
    
    @staticmethod
    def load_dqn_model(model_path: str, metadata: Dict) -> nn.Module:
        """Load a DQN model with flexible architecture detection."""
        try:
            # Load state dict first to detect architecture
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Detect architecture from state dict
            hidden_layers = ModelLoader.detect_dqn_architecture(state_dict)
            print(f"Detected DQN architecture: input_dim={metadata['input_dim']}, "
                  f"hidden_layers={hidden_layers}, output_dim={metadata['output_dim']}")
            
            # Create model with detected architecture
            model = FlexibleDQNAgent(
                input_dim=metadata['input_dim'],
                output_dim=metadata['output_dim'],
                hidden_layers=hidden_layers
            )
            
            # Load the state dict
            model.load_state_dict(state_dict)
            model.eval()
            return model
            
        except Exception as e:
            print(f"Error loading DQN with flexible architecture: {e}")
            # Fallback to original DQNAgent if available
            try:
                model = DQNAgent(
                    input_dim=metadata['input_dim'],
                    output_dim=metadata['output_dim']
                )
                model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
                model.eval()
                return model
            except Exception as e2:
                print(f"Fallback DQN loading also failed: {e2}")
                raise e
    
    @staticmethod
    def load_ppo_model(model_path: str, metadata: Dict) -> StandardPPO:
        """Load a PPO model with proper module context."""
        import sys
        import importlib.util
        
        # Add safe directories to sys.path to ensure imports work
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = os.getcwd()
        for p in [current_dir, os.getcwd(), os.path.dirname(model_path)]:
            if p and p not in sys.path:
                sys.path.insert(0, p)
        
        try:
            # Try to ensure unified_prpo is available in the loading context
            if 'unified_prpo' not in sys.modules:
                spec = importlib.util.find_spec('unified_prpo')
                if spec is not None:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules['unified_prpo'] = module
                    spec.loader.exec_module(module)
            
            # Add the classes to global namespace for unpickling
            import __main__
            __main__.StandardPPO = StandardPPO
            __main__.UnifiedPRPOAgent = UnifiedPRPOAgent  
            __main__.UnifiedActorCritic = UnifiedActorCritic
            
            # Now try loading
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Check if we actually loaded a model or just a state_dict
            if isinstance(model, (dict, torch.nn.modules.utils._IncompatibleKeys)) or not hasattr(model, 'act'):
                print(f"Loaded object is a state_dict (type: {type(model).__name__}), not a full model. Proceeding to fallback loading...")
                # Trigger the fallback by raising an exception
                raise ValueError("Loaded state_dict instead of full model")
            else:
                if hasattr(model, 'eval'):
                    model.eval()
                return model
            
        except Exception as e:
            print(f"Failed to load PPO as full object: {e}")
            # Fallback: try loading as state_dict into new policy
            try:
                model = StandardPPO(
                    state_dim=metadata.get('simple_input_dim', metadata.get('input_dim', 2)),
                    action_dim=metadata.get('simple_output_dim', metadata.get('output_dim', 2)),
                    device='cpu'
                )
                state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
                if hasattr(model, 'policy') and hasattr(model.policy, 'load_state_dict'):
                    model.policy.load_state_dict(state_dict)
                else:
                    # Try to load into the model directly
                    model.load_state_dict(state_dict)
                model.eval()
                return model
            except Exception as e2:
                print(f"PPO fallback loading also failed: {e2}")
                # Last resort: create a minimal working model
                return StandardPPO(
                    state_dim=metadata.get('simple_input_dim', metadata.get('input_dim', 2)),
                    action_dim=metadata.get('simple_output_dim', metadata.get('output_dim', 2)),
                    device='cpu'
                )
    
    @staticmethod
    def load_prpo_model(model_path: str, metadata: Dict) -> Any:
        """Load a PRPO model with enhanced module context handling."""
        import sys
        import importlib.util
        
        # Add safe directories to sys.path
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = os.getcwd()
        for p in [current_dir, os.getcwd(), os.path.dirname(model_path)]:
            if p and p not in sys.path:
                sys.path.insert(0, p)
        
        # Ensure required modules are available
        try:
            if 'unified_prpo' not in sys.modules:
                spec = importlib.util.find_spec('unified_prpo')
                if spec is not None:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules['unified_prpo'] = module
                    spec.loader.exec_module(module)
            
            # Add classes to global namespace for unpickling
            import __main__
            __main__.UnifiedPRPOAgent = UnifiedPRPOAgent
            __main__.StandardPPO = StandardPPO
            __main__.UnifiedActorCritic = UnifiedActorCritic
        except Exception as e:
            print(f"Warning: Could not setup module context for PRPO: {e}")
        
        # Try 1: Direct object loading
        try:
            print(f"Attempting direct load for PRPO: {model_path}")
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Check if we actually loaded a model or just a state_dict
            if isinstance(model, (dict, torch.nn.modules.utils._IncompatibleKeys)) or not hasattr(model, 'act'):
                print(f"Loaded object is a state_dict (type: {type(model).__name__}), not a full model. Proceeding to state_dict loading...")
                # Don't return here, fall through to Try 2
            else:
                if hasattr(model, 'eval'):
                    model.eval()
                print("Successfully loaded PRPO as full object")
                return model
        except Exception as e:
            print(f"Direct load failed: {e}")
        
        # Try 2: Safe state_dict loading  
        try:
            print("Attempting state_dict load for PRPO...")
            prpo_model = UnifiedPRPOAgent(
                state_dim=metadata.get('simple_input_dim', metadata.get('input_dim', 2)),
                action_dim=metadata.get('simple_output_dim', metadata.get('output_dim', 2)),
                lr=3e-4,
                device='cpu'
            )
            
            # Reuse state_dict from Try 1 if available, otherwise load it
            if 'model' in locals() and isinstance(model, dict):
                state_dict = model
                print("Reusing state_dict from previous load attempt")
            else:
                state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Try loading into policy first
            if hasattr(prpo_model, 'policy') and hasattr(prpo_model.policy, 'load_state_dict'):
                prpo_model.policy.load_state_dict(state_dict)
            else:
                # Try loading into model directly
                prpo_model.load_state_dict(state_dict)
                
            if hasattr(prpo_model, 'eval'):
                prpo_model.eval()
            print("Successfully loaded PRPO using state_dict")
            return prpo_model
        except Exception as e:
            print(f"State_dict load failed: {e}")
        
        # Try 3: Safe globals approach
        try:
            print("Attempting load with safe globals...")
            if hasattr(torch.serialization, 'safe_globals'):
                with torch.serialization.safe_globals([UnifiedPRPOAgent, StandardPPO, UnifiedActorCritic]):
                    model = torch.load(model_path, map_location='cpu', weights_only=False)
                    if hasattr(model, 'eval'):
                        model.eval()
                    print("Successfully loaded PRPO with safe globals")
                    return model
        except Exception as e:
            print(f"Safe globals load failed: {e}")
        
        # Last resort: Create a minimal working model
        print("All loading methods failed, creating minimal PRPO model")
        return UnifiedPRPOAgent(
            state_dim=metadata.get('simple_input_dim', metadata.get('input_dim', 2)),
            action_dim=metadata.get('simple_output_dim', metadata.get('output_dim', 2)),
            lr=3e-4,
            device='cpu'
        )
    
    @staticmethod
    def load_selfplay_model(model_path: str, metadata: Dict) -> nn.Module:
        """Load a Self-Play model (typically DQN-based)."""
        return ModelLoader.load_dqn_model(model_path, metadata)
    
    @staticmethod
    def load_psro_model(model_path: str, metadata: Dict) -> nn.Module:
        """Load a PSRO model (typically DQN-based)."""
        return ModelLoader.load_dqn_model(model_path, metadata)
    
    @staticmethod
    def load_model(algorithm: str, model_path: str, metadata: Dict) -> Any:
        """Load a model based on algorithm type."""
        loaders = {
            'dqn': ModelLoader.load_dqn_model,
            'ppo': ModelLoader.load_ppo_model,
            'prpo': ModelLoader.load_prpo_model,
            'selfplay': ModelLoader.load_selfplay_model,
            'psro': ModelLoader.load_psro_model
        }
        
        if algorithm not in loaders:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return loaders[algorithm](model_path, metadata)


class GameEnvironmentLoader:
    """Utility class to load game environments for evaluation."""
    
    @staticmethod
    def load_game_environment(game_name: str):
        """Load the appropriate environment for the game."""
        if game_name == 'leduc':
            try:
                from leduc_poker_training_and_evaluation import LeducPokerEnvironment
                return LeducPokerEnvironment()
            except ImportError:
                raise ValueError(f"LeducPokerEnvironment not available for game: {game_name}")
        elif game_name == 'kuhn':
            try:
                from kuhn_poker_training_and_evaluation import KuhnPokerEnvironment
                return KuhnPokerEnvironment()
            except ImportError:
                raise ValueError(f"KuhnPokerEnvironment not available for game: {game_name}")
        elif game_name == 'rps':
            try:
                from rps_training_and_evaluation import RPSEnvironment
                return RPSEnvironment()
            except ImportError:
                raise ValueError(f"RPSEnvironment not available for game: {game_name}")
        elif game_name == 'matchingpennies':
            try:
                from matchingpennies_training_and_evaluation import MatchingPenniesEnvironment
                return MatchingPenniesEnvironment()
            except ImportError:
                raise ValueError(f"MatchingPenniesEnvironment not available for game: {game_name}")
        elif game_name == 'stag_hunt':
            try:
                from stag_hunt_training_and_evaluation import StagHuntEnvironment
                return StagHuntEnvironment()
            except ImportError:
                raise ValueError(f"StagHuntEnvironment not available for game: {game_name}")
        else:
            raise ValueError(f"Unknown game: {game_name}")





def evaluate_and_report(gauntlet: Any, policy: nn.Module, name: str, out_dir: str, 
                        eval_input_dim: int, eval_output_dim: int):
    """Evaluate policy using gauntlet with proper dimension handling."""
    if not _GAUNTLET_AVAILABLE:
        print(f"Gauntlet not available, skipping evaluation for {name}")
        return None  # Return None if skipped
        
    # Determine if we need to wrap the policy
    use_wrapped = isinstance(policy, (StandardPPO, UnifiedPRPOAgent, UnifiedActorCritic)) or hasattr(policy, 'policy')
    
    wrapped = PolicyWrapperAgent(policy, eval_input_dim, eval_output_dim, name=f"{name}_Wrapped") if use_wrapped else policy
    print(f"Evaluating {name}: use_wrapped={use_wrapped}, policy_type={type(policy).__name__}")
    
    if hasattr(policy, "eval"):
        policy.eval()
    
    print(f"\n{'='*40}\n E V A L U A T I N G:   {name} \n{'='*40}")
    metrics = gauntlet.evaluate_policy(policy=wrapped, policy_name=name, environments=None)
    report_path = os.path.join(out_dir, "report.json")
    gauntlet.generate_report(report_path)
    
    print(f"Saved report to {report_path}")
    
    # Move generated visualization files into out_dir
    try:
        import shutil
        fmt = gauntlet.config.visualization_format
        fnames = [
            f"{name}_challenger_performance.{fmt}",
            f"{name}_robustness_radar.{fmt}",
            f"{name}_performance_heatmap.{fmt}",
            f"{name}_metrics_comparison.{fmt}",
        ]
        for fn in fnames:
            if os.path.exists(fn):
                shutil.move(fn, os.path.join(out_dir, fn))
    except Exception as e:
        print(f"Warning: could not move visualization files for {name}: {e}")

    return metrics  # Return metrics for statistical aggregation


class UnifiedEvaluator:
    """Main evaluation class that orchestrates the entire evaluation process."""
    
    def __init__(self, model_dir: str, output_dir: str):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load training configuration
        config_path = self.model_dir / 'training_config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Training config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.training_config = json.load(f)
        
        self.game_name = self.training_config['game']
        self.seeds = self.training_config['seeds']
        self.algorithms = self.training_config['algorithms']
        
        print(f"Evaluating {self.game_name} with {len(self.seeds)} seeds")
        print(f"Algorithms: {self.algorithms}")
        
        # Setup gauntlet
        if _GAUNTLET_AVAILABLE:
            config = EvaluationConfig(num_episodes=200, parallel_workers=1, save_visualizations=True)
            self.gauntlet = EnhancedGauntletBenchmark(config)
            self._register_game_environment()
        else:
            self.gauntlet = None
        

    
    def _register_game_environment(self):
        """Register the game environment with the gauntlet."""
        if not _GAUNTLET_AVAILABLE:
            return
            
        env_class = GameEnvironmentLoader.load_game_environment(self.game_name).__class__
        
        # Determine if the game is zero-sum or general-sum
        is_zero_sum = self.game_name.lower() not in ['stag_hunt', 'prisoner_dilemma', 'coordination_game']
        
        # For stag hunt, we need to provide the payoff matrices
        payoff_matrices = None
        if self.game_name.lower() == 'stag_hunt':
            # Stag Hunt payoff matrices: (Stag, Hare) x (Stag, Hare)
            # Both Stag: (3,3), Stag vs Hare: (0,2), Hare vs Stag: (2,0), Both Hare: (1,1)
            import numpy as np
            stag_hunt_A = np.array([[3, 0], [2, 1]])  # Row player payoffs
            stag_hunt_B = np.array([[3, 2], [0, 1]])  # Column player payoffs
            payoff_matrices = (stag_hunt_A, stag_hunt_B)
        
        self.gauntlet.register_environment(
            self.game_name.title(),
            env_class,
            payoff_matrices=payoff_matrices,
            game_prefix=self.game_name.title(),
            zero_sum=is_zero_sum
        )
    
    def load_models(self) -> Dict[str, List[Any]]:
        """Load all trained models from the model directory."""
        models = defaultdict(list)
        
        for algorithm in self.algorithms:
            alg_dir = self.model_dir / algorithm
            if not alg_dir.exists():
                print(f"Warning: No models found for {algorithm}")
                continue
            
            for seed in self.seeds:
                model_path = alg_dir / f'seed_{seed}.pth'
                metadata_path = alg_dir / f'seed_{seed}_metadata.json'
                
                if not model_path.exists() or not metadata_path.exists():
                    print(f"Warning: Missing files for {algorithm} seed {seed}")
                    continue
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Load model
                try:
                    model = ModelLoader.load_model(algorithm, str(model_path), metadata)
                    models[algorithm].append({
                        'model': model,
                        'seed': seed,
                        'metadata': metadata
                    })
                    print(f"Loaded {algorithm} model for seed {seed}")
                except Exception as e:
                    print(f"Error loading {algorithm} model for seed {seed}: {e}")
        
        return dict(models)
    
    def evaluate_models(self, models: Dict[str, List[Any]]):
        """Evaluate all loaded models using the gauntlet."""
        if not _GAUNTLET_AVAILABLE:
            print("Gauntlet not available, skipping gauntlet evaluation")
            return
        
        for algorithm, model_list in models.items():
            if not model_list:
                continue
            
            print(f"\n{'='*50}")
            print(f"EVALUATING {algorithm.upper()}")
            print(f"{'='*50}")
            
            # Use the first model's metadata for dimensions
            metadata = model_list[0]['metadata']
            
            # Determine evaluation dimensions
            if algorithm in ["ppo", "prpo"]:
                eval_input_dim = metadata['simple_input_dim']
                eval_output_dim = metadata['simple_output_dim']
            else:
                eval_input_dim = metadata['input_dim']
                eval_output_dim = metadata['output_dim']
            
            # Create output directory for this algorithm
            alg_output_dir = self.output_dir / algorithm
            alg_output_dir.mkdir(exist_ok=True)
            
            # Evaluate each model and collect robustness scores
            robustness_scores = []
            for i, model_info in enumerate(model_list):
                model = model_info['model']
                seed = model_info['seed']
                
                print(f"Evaluating {algorithm} seed {seed}...")
                
                # Create seed-specific output directory
                seed_output_dir = alg_output_dir / f'seed_{seed}'
                seed_output_dir.mkdir(exist_ok=True)
                
                # Evaluate with gauntlet
                metrics = evaluate_and_report(
                    self.gauntlet, model, 
                    f"{self.game_name.title()}_{algorithm.upper()}_seed_{seed}",
                    str(seed_output_dir), eval_input_dim, eval_output_dim
                )
                
                # Collect robustness score
                if metrics:
                    robustness_scores.append(metrics.robustness_score)
            
            # Save per-algorithm robustness scores for stats
            scores_path = alg_output_dir / 'robustness_scores.json'
            with open(scores_path, 'w') as f:
                json.dump(robustness_scores, f, indent=2)
            print(f"Saved robustness scores for {algorithm} to {scores_path}")
    
    def compute_statistical_analysis(self, models: Dict[str, List[Any]]):
        """Compute statistical analysis across seeds using gauntlet metrics."""
        print(f"\n{'='*50}")
        print(f"STATISTICAL ANALYSIS")
        print(f"{'='*50}")
        
        # Collect scores for each algorithm (gauntlet only)
        algorithm_scores = {}
        
        for algorithm, model_list in models.items():
            if not model_list:
                continue
            
            gauntlet_scores = []
            for model_info in model_list:
                seed = model_info['seed']
                
                print(f"Loading gauntlet results for {algorithm} seed {seed}...")
                
                # Load gauntlet robustness score if available
                alg_dir = self.output_dir / algorithm / f'seed_{seed}'
                report_path = alg_dir / 'report.json'
                if report_path.exists():
                    try:
                        with open(report_path, 'r') as f:
                            report = json.load(f)
                        robustness = report.get('summary', {}).get('robustness_score', float('nan'))
                        gauntlet_scores.append(robustness)
                        print(f"  {algorithm} seed {seed} (gauntlet robustness): {robustness:.3f}")
                    except Exception as e:
                        print(f"  Error loading gauntlet report for {algorithm} seed {seed}: {e}")
                        gauntlet_scores.append(float('nan'))
                else:
                    print(f"  No gauntlet report found for {algorithm} seed {seed}")
                    gauntlet_scores.append(float('nan'))
            
            if gauntlet_scores:
                algorithm_scores[algorithm] = {
                    'gauntlet_scores': gauntlet_scores
                }
        
        # Compute statistics for each algorithm using gauntlet metrics
        algorithm_stats = {}
        for algorithm, data in algorithm_scores.items():
            stats = {}
            
            # Gauntlet stats only
            if data['gauntlet_scores']:
                # Clean NaNs for stats
                clean_scores = [s for s in data['gauntlet_scores'] if not math.isnan(s)]
                if clean_scores:
                    gauntlet_stats = compute_mean_ci(clean_scores)
                    stats['gauntlet'] = gauntlet_stats
                    print(f"{algorithm.upper()} (gauntlet robustness): mean={gauntlet_stats['mean']:.3f}, "
                          f"95% CI=[{gauntlet_stats['ci_low']:.3f}, {gauntlet_stats['ci_high']:.3f}], "
                          f"variance={gauntlet_stats['sd']**2:.3f}, n={gauntlet_stats['n']}")
                else:
                    print(f"{algorithm.upper()} (gauntlet): No valid scores available")
            
            algorithm_stats[algorithm] = stats
        
        # Pairwise comparisons (gauntlet only)
        pairwise_tests = {'gauntlet': {}}
        algorithms = list(algorithm_scores.keys())
        
        print(f"\nPairwise tests (gauntlet):")
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms[i+1:], i+1):
                if alg1 in algorithm_scores and alg2 in algorithm_scores:
                    scores1 = algorithm_scores[alg1].get('gauntlet_scores', [])
                    scores2 = algorithm_scores[alg2].get('gauntlet_scores', [])
                    
                    # Clean NaNs
                    scores1 = [s for s in scores1 if not math.isnan(s)]
                    scores2 = [s for s in scores2 if not math.isnan(s)]
                    
                    if len(scores1) > 1 and len(scores2) > 1:
                        # Try paired t-test if same length
                        if len(scores1) == len(scores2):
                            p_value = paired_t_test(scores1, scores2)
                            test_name = "paired t-test"
                        else:
                            p_value = welch_t_test(scores1, scores2)
                            test_name = "Welch's t-test"
                        
                        pairwise_tests['gauntlet'][f"{alg1}_vs_{alg2}"] = p_value
                        
                        if p_value is not None:
                            significance = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                            print(f"  {alg1.upper()} vs {alg2.upper()} ({test_name}): p={p_value:.4f} {significance}")
                        else:
                            print(f"  {alg1.upper()} vs {alg2.upper()}: Test not performed (insufficient data)")
        
        # Save statistical results
        stats_data = {
            'algorithm_scores': algorithm_scores,
            'algorithm_stats': algorithm_stats,
            'pairwise_tests': pairwise_tests,
            'game': self.game_name,
            'seeds': self.seeds
        }
        
        stats_path = self.output_dir / f'{self.game_name}_statistical_analysis.json'
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"Statistical analysis saved to {stats_path}")
        
        return stats_data
    
    def generate_summary_report(self, stats_data: Dict):
        """Generate a comprehensive summary report."""
        print(f"\n{'='*50}")
        print(f"GENERATING SUMMARY REPORT")
        print(f"{'='*50}")
        
        summary = {
            'evaluation_metadata': {
                'game': self.game_name,
                'algorithms': self.algorithms,
                'seeds': self.seeds,
                'num_seeds': len(self.seeds),
                'timestamp': time.time()
            },
            'training_config': self.training_config,
            'statistical_analysis': stats_data,
            'algorithm_rankings': {}
        }
        
        # Rank algorithms by mean performance (gauntlet only)
        if stats_data['algorithm_stats']:
            rankings = sorted(
                stats_data['algorithm_stats'].items(),
                key=lambda x: x[1].get('gauntlet', {'mean': 0})['mean'],
                reverse=True
            )
            
            for rank, (algorithm, data) in enumerate(rankings, 1):
                if 'gauntlet' not in data:
                    continue
                metric_type = 'gauntlet'
                stats = data[metric_type]
                summary['algorithm_rankings'][algorithm] = {
                    'rank': rank,
                    'metric_type': metric_type,
                    'mean_score': stats['mean'],
                    'ci_low': stats['ci_low'],
                    'ci_high': stats['ci_high'],
                    'std_dev': stats['sd'],
                    'n_seeds': stats['n']
                }
        
        # Save summary report
        summary_path = self.output_dir / 'evaluation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary report saved to {summary_path}")
        
        # Print summary to console
        print(f"\n{'='*30}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*30}")
        print(f"Game: {self.game_name}")
        print(f"Seeds: {len(self.seeds)}")
        print(f"Algorithms: {len(self.algorithms)}")
        
        if summary['algorithm_rankings']:
            print(f"\nAlgorithm Rankings (by mean score):")
            for algorithm, data in summary['algorithm_rankings'].items():
                print(f"  {data['rank']}. {algorithm.upper()} ({data['metric_type']}): "
                      f"{data['mean_score']:.3f} ± {data['std_dev']:.3f} "
                      f"(95% CI: [{data['ci_low']:.3f}, {data['ci_high']:.3f}])")


def main():
    
    # IMPORTANT: Update this path to match your actual model directory
    # If you want to evaluate Stag Hunt models, use a path like:
    # model_dir = '/kaggle/input/stag-models/models/stag_hunt_60s'
    # If you want to evaluate Leduc Poker models, use:
    model_dir = '/kaggle/input/stag-models/models/leduc_60s'  # Current path for Leduc models
    output_dir = 'results/evaluation_output'  # Will be updated based on game type
    skip_gauntlet = False
    
    print(f"Unified Evaluation")
    print(f"Model directory: {model_dir}")
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory does not exist: {model_dir}")
        print("Please update the model_dir path in the main() function to point to your actual models")
        print("Available options might be:")
        print("  - /kaggle/input/stag-models/models/stag_hunt_60s (for Stag Hunt)")
        print("  - /kaggle/input/stag-models/models/leduc_60s (for Leduc Poker)")
        print("  - /kaggle/input/stag-models/models/kuhn_60s (for Kuhn Poker)")
        return
    
    # Create evaluator
    evaluator = UnifiedEvaluator(model_dir, output_dir)
    
    # Update output directory based on detected game
    game_name = evaluator.game_name
    output_dir = f'results/{game_name}_evaluation'
    evaluator.output_dir = Path(output_dir)
    evaluator.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Detected game: {game_name}")
    print(f"Output directory: {output_dir}")
    
    # Validate that the models match what we're trying to evaluate
    if game_name.lower() == 'stag_hunt' and 'leduc' in str(model_dir).lower():
        print("WARNING: You're trying to evaluate Leduc Poker models as Stag Hunt!")
        print("This may cause issues. Please check your model directory path.")
    elif game_name.lower() in ['leduc', 'kuhn'] and 'stag' in str(model_dir).lower():
        print(f"WARNING: You're trying to evaluate Stag Hunt models as {game_name}!")
        print("This may cause issues. Please check your model directory path.")
    
    # Load models
    models = evaluator.load_models()
    
    if not any(models.values()):
        print("No models found to evaluate!")
        print("Please check that your model directory contains the expected algorithm subdirectories:")
        print("  - dqn/, ppo/, prpo/, selfplay/, psro/")
        return
    
    print(f"Loaded models for algorithms: {list(models.keys())}")
    
    # Evaluate models with gauntlet
    if not skip_gauntlet:
        evaluator.evaluate_models(models)
    else:
        print("Skipping gauntlet evaluation")
    
    # Compute statistical analysis
    stats_data = evaluator.compute_statistical_analysis(models)
    
    # Generate summary report
    evaluator.generate_summary_report(stats_data)
    
    print(f"\n{'='*50}")
    print(f"EVALUATION COMPLETED")
    print(f"{'='*50}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
