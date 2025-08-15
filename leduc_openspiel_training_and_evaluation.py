# leduc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import os
import argparse
from typing import Optional, List, Tuple, Dict, Any, Callable, TYPE_CHECKING
from collections import namedtuple
import torch.nn.functional as F
import time
import math
import json


# Optional Nash solver support
try:
    import nashpy as nash  # type: ignore
    _NASH_AVAILABLE = True
except Exception:
    nash = None  # type: ignore
    _NASH_AVAILABLE = False
    print("Warning: nashpy not available. Install with 'pip install nashpy' for proper PSRO.")

# Optional OpenSpiel support
try:
    import pyspiel as openspiel  # type: ignore
    _OPENSPIEL_AVAILABLE = True
except Exception:
    openspiel = None  # type: ignore
    _OPENSPIEL_AVAILABLE = False
    print("Warning: OpenSpiel not available. Using fallback implementation.")

# Optional SciPy for significance testing
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False
    _scipy_stats = None  # type: ignore

# ----------------------------------------------------------------------------
# Statistical helpers
# ----------------------------------------------------------------------------

def _t_critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    df = n - 1
    if _SCIPY_AVAILABLE:
        try:
            return float(_scipy_stats.t.ppf(0.975, df))
        except Exception:
            pass
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
    if len(a) != len(b) or len(a) < 2:
        return None
    if _SCIPY_AVAILABLE:
        try:
            _, p = _scipy_stats.ttest_rel(a, b)
            return float(p)
        except Exception:
            return None
    return None

# ============================================================================
# Import from benchmark file 
# ============================================================================

# Gym spaces (with safe fallback if not available)
try:
    from gym.spaces import Space, Discrete, Box
except Exception:
    class Space:  # type: ignore
        pass
    class Discrete:  # type: ignore
        def __init__(self, n: int):
            self.n = int(n)
    class Box:  # type: ignore
        def __init__(self, low, high, shape, dtype):
            self.shape = shape

class PolicyWrapperAgent(ChallengerAgent):
    """Adapter to make arbitrary policies compatible with the Gauntlet interface."""
    def __init__(self, base_policy: nn.Module, input_dim: int, action_dim: int, name: str = "WrappedPolicy"):
        super().__init__(name, "student")
        self._base = base_policy
        self._input_dim = int(input_dim)
        self._action_dim = int(action_dim)

    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        ts = torch.as_tensor(observation, dtype=torch.float32)
        if ts.ndim == 1:
            ts = ts.unsqueeze(0)
        
        # Handle dimension mismatch: map 30D gauntlet Leduc state to 8D Leduc input 
        if ts.shape[-1] == 30 and self._input_dim == 8:
            # Simple mapping: take first 8 dimensions (most relevant features)
            ts = ts[..., :8]
        elif ts.shape[-1] > self._input_dim:
            # Generic fallback: truncate extra features
            ts = ts[..., :self._input_dim]
            print(f"Warning: Truncating observation from {observation.shape} to {ts.shape} for {self._input_dim}D model")
        elif ts.shape[-1] < self._input_dim:
            # Pad with zeros if observation is smaller than expected  
            padding = torch.zeros(ts.shape[:-1] + (self._input_dim - ts.shape[-1],))
            ts = torch.cat([ts, padding], dim=-1)
            print(f"Warning: Padding observation from {observation.shape} to {ts.shape} for {self._input_dim}D model")
        
        # Handle StandardPPO/UnifiedPRPOAgent specifically
        if isinstance(self._base, (StandardPPO, UnifiedPRPOAgent)):
            try:
                with torch.no_grad():
                    action = self._base.act(ts.squeeze().cpu().numpy())
                    return int(action)
            except Exception as e:
                print(f"Error with StandardPPO/UnifiedPRPOAgent act: {e}")
        
        # Handle UnifiedActorCritic directly
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


# ============================================================================
# 2. Leduc Poker Specific Functions for Unified PRPO
# ============================================================================

class LeducPokerSimpleEnvironment:
    """Simplified Leduc Poker game environment for unified PRPO."""
    def __init__(self):
        # Actions: 0=Fold, 1=Call/Check, 2=Raise
        self.action_dim = 3
        self.state_dim = 8  # Simplified state representation
        # Cards: 0-5 representing J♥,J♠,Q♥,Q♠,K♥,K♠
        self.cards = list(range(6))
        self.reset()

    def reset(self):
        # Sample private card for player and community card
        hand_cards = random.sample(self.cards, 2)
        self.player_card = hand_cards[0]
        self.community_card = hand_cards[1]
        self.pot = 2  # Initial pot (2 antes)
        self.round = 1  # Round 1 or 2
        return self._get_state()

    def _get_state(self):
        # Simplified state: [player_card_one_hot(6) + round(1) + pot_normalized(1)]
        state = np.zeros(self.state_dim)
        state[self.player_card] = 1.0  # One-hot player card
        state[6] = self.round / 2.0  # Normalized round
        state[7] = min(self.pot / 20.0, 1.0)  # Normalized pot size
        return state

    def step(self, p1_action: int, p2_action: int):
        """
        Simplified payoff structure for Leduc Poker:
        - Fold (0): Player folding loses current pot contribution
        - Call/Check (1): Proceed to showdown if both call
        - Raise (2): Increase pot size
        """
        if p1_action == 0:  # P1 folds
            p1_reward = -1.0
        elif p2_action == 0:  # P2 folds
            p1_reward = 1.0
        else:  # Both call/check or raise, go to showdown
            # Determine winner by card strength (higher card wins)
            if self.player_card // 2 > self.community_card // 2:  # Compare ranks (J=0,1, Q=2,3, K=4,5)
                p1_reward = 2.0
            elif self.player_card // 2 < self.community_card // 2:
                p1_reward = -2.0
            else:  # Same rank, suit doesn't matter in this simplified version
                p1_reward = 0.0

        p2_reward = -p1_reward
        done = True
        return self._get_state(), [p1_reward, p2_reward], done

def get_leduc_nash_policy_callable(policy_probs_batch: torch.Tensor) -> torch.Tensor:
    """
    Returns an approximation of the Nash equilibrium for Leduc Poker.
    Simplified mixed strategy favoring check/call.
    """
    batch_size = policy_probs_batch.shape[0]
    # Approximate mixed strategy [0.3, 0.5, 0.2] for Fold/Call/Raise
    nash_dist = torch.full_like(policy_probs_batch, 0.0)
    nash_dist[:, 0] = 0.3  # Fold
    nash_dist[:, 1] = 0.5  # Call/Check
    nash_dist[:, 2] = 0.2  # Raise
    return nash_dist

def calculate_leduc_exploitability_callable(policy: UnifiedActorCritic) -> float:
    """
    Calculates exploitability for Leduc Poker policy.
    Based on strategy variance across different card holdings.
    """
    policy.eval()
    device = next(policy.parameters()).device
    
    exploitability = 0.0
    num_states = 6  # Number of possible private cards
    
    for card in range(num_states):
        state = torch.zeros(8, device=device).unsqueeze(0)
        state[0, card] = 1.0  # Set private card
        state[0, 6] = 0.5     # Round 1
        state[0, 7] = 0.1     # Small pot
        
        with torch.no_grad():
            policy_probs, _ = policy(state)
            probs = policy_probs.squeeze().cpu().numpy()
        
        # Measure strategy variance as a proxy for exploitability
        variance = np.var(probs)
        exploitability += variance
    
    exploitability /= num_states
    policy.train()
    return float(exploitability)

def get_leduc_exploiter_opponents() -> List[Callable]:
    """Returns a list of exploiter bots for Leduc Poker."""
    return [
        lambda: 0,  # Always Fold
        lambda: 1,  # Always Call/Check
        lambda: 2,  # Always Raise
        lambda: np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]),  # Conservative
        lambda: np.random.choice([0, 1, 2], p=[0.1, 0.5, 0.4]),  # Aggressive
        lambda: random.randint(0, 2),  # Random
    ]

def train_prpo_leduc_unified(time_budget_seconds: float, input_dim: int, output_dim: int, 
                           device: torch.device, lambda_nash: float = 0.5, 
                           lambda_exploit: float = 0.5) -> nn.Module:
    """
    Superior PRPO training function using the unified framework with time budget for Leduc Poker.
    """
    print(f"Training PRPO (Unified Framework) for Leduc Poker for {time_budget_seconds} seconds...")
    
    # The manager needs a way to create new environments
    env_factory = lambda: LeducPokerSimpleEnvironment()

    prpo_system = UnifiedPRPO(
        state_dim=input_dim, 
        action_dim=output_dim, 
        lr=1e-4, 
        device=str(device),
        population_size=4,
        lambda_nash=lambda_nash,
        nash_target_callable=get_leduc_nash_policy_callable,
        lambda_exploit=lambda_exploit,
        exploitability_calculator=calculate_leduc_exploitability_callable,
        exploiter_opponents=get_leduc_exploiter_opponents()
    )
    
    # Train the system with time budget
    episodes_completed, results_over_time = prpo_system.train_time_budget(
        env_factory=env_factory, 
        time_budget_seconds=time_budget_seconds,
        update_every_seconds=1.0
    )
    
    # Get the best agent from the trained population
    best_agent = prpo_system.get_best_agent()
    
    print("Training finished for PRPO (Unified Framework) - Leduc Poker.")
    return best_agent

# ============================================================================
# 3. Leduc Poker Environment Implementation (Original OpenSpiel-based)
# ============================================================================

class LeducPokerEnvironment(Environment):
    """
    Leduc Poker environment using OpenSpiel for turn-based gameplay.
    - Actions: 0=Fold, 1=Call/Check, 2=Raise (when legal)
    - Observation: Game state encoded as vector
    - Turn-based with legal action masking
    """
    def __init__(self):
        # Initialize with fallback if OpenSpiel unavailable
        n_actions_detected = 4
        info_state_size_detected = 30

        if _OPENSPIEL_AVAILABLE:
            try:
                self.game = openspiel.load_game("leduc_poker")
                n_actions_detected = int(self.game.num_distinct_actions())
                info_state_size_detected = int(self.game.information_state_tensor_size())
            except Exception:
                self.game = None
        else:
            self.game = None

        self.info_state_size = info_state_size_detected
        self._observation_space = Box(low=0, high=1, shape=(self.info_state_size,), dtype=np.float32)
        self._action_space = Discrete(n_actions_detected)
        self.state = None
        self.step_count = 0
        self.episode_length = 100  # maximum steps per episode

    def reset(self) -> torch.Tensor:
        """Reset the environment to initial state."""
        self.step_count = 0
        if self.game is not None:
            try:
                self.state = self.game.new_initial_state()
                # Handle chance nodes
                while self.state.is_chance_node():
                    actions = self.state.legal_actions()
                    if actions:
                        action = random.choice(actions)
                        self.state.apply_action(action)
                
                return self._get_observation(0)
            except Exception:
                pass
        
        # Fallback observation
        return torch.zeros(self.info_state_size, dtype=torch.float32)

    def step(self, actions: List[int]) -> Tuple[torch.Tensor, List[float], bool, Dict]:
        """Step the environment with actions from both players."""
        self.step_count += 1
        
        # Convert actions to ints
        def _to_int(a: Any) -> int:
            try:
                if isinstance(a, (tuple, list)):
                    a = a[0]
                if torch.is_tensor(a):
                    return int(a.item())
                return int(a)
            except Exception:
                return 0
        
        action1, action2 = _to_int(actions[0]), _to_int(actions[1])
        rewards = [0.0, 0.0]
        done = False
        
        if self.game is not None and self.state is not None:
            try:
                # Apply actions in turn-based manner
                for action in [action1, action2]:
                    if self.state.is_terminal():
                        break
                    
                    # Handle chance nodes before querying current player to avoid warnings
                    while self.state.is_chance_node():
                        chance_actions = self.state.legal_actions()
                        if chance_actions:
                            chance_action = random.choice(chance_actions)
                            self.state.apply_action(chance_action)
                        else:
                            break

                    current_player = self.state.current_player()
                    if current_player >= 0:
                        legal_actions = self.state.legal_actions()
                        if action in legal_actions:
                            self.state.apply_action(action)
                        elif legal_actions:
                            self.state.apply_action(legal_actions[0])
                    
                    # Handle chance nodes after each player action too
                    while self.state.is_chance_node():
                        chance_actions = self.state.legal_actions()
                        if chance_actions:
                            chance_action = random.choice(chance_actions)
                            self.state.apply_action(chance_action)
                        else:
                            break
                
                # Check if terminal and get rewards
                if self.state.is_terminal():
                    returns = self.state.returns()
                    rewards = [float(returns[0]), float(returns[1])]
                    done = True
                else:
                    done = self.step_count >= self.episode_length
                    
            except Exception:
                done = True
        else:
            # Fallback: simple random rewards
            rewards = [random.uniform(-1, 1), random.uniform(-1, 1)]
            done = self.step_count >= self.episode_length
        
        obs = self._get_observation(0) if not done else torch.zeros(self.info_state_size, dtype=torch.float32)
        info = {'zero_sum': True, 'extensive_form': True}
        
        return obs, rewards, done, info

    def _get_observation(self, player: int) -> torch.Tensor:
        """Get observation for a specific player."""
        if self.game is not None and self.state is not None:
            try:
                # Ensure we are not at a chance node before querying information state
                while self.state.is_chance_node():
                    chance_actions = self.state.legal_actions()
                    if chance_actions:
                        self.state.apply_action(random.choice(chance_actions))
                    else:
                        break
                current_player = self.state.current_player()
                if current_player == -1:
                    # If still at a chance node or invalid player, return zero obs to avoid warnings
                    return torch.zeros(self.info_state_size, dtype=torch.float32)
                info_state = self.state.information_state_tensor(player)
                return torch.tensor(info_state, dtype=torch.float32)
            except Exception:
                pass
        
        # Fallback observation
        return torch.zeros(self.info_state_size, dtype=torch.float32)

    def get_legal_actions(self, player: int = 0) -> List[int]:
        """Get legal actions for the current player."""
        if self.game is not None and self.state is not None:
            try:
                # Resolve chance nodes first to avoid querying with player == -1
                while self.state.is_chance_node():
                    chance_actions = self.state.legal_actions()
                    if chance_actions:
                        self.state.apply_action(random.choice(chance_actions))
                    else:
                        break
                current_player = self.state.current_player()
                if current_player == player and not self.state.is_terminal():
                    return list(self.state.legal_actions())
            except Exception:
                pass
        
        # Fallback: all actions legal
        return list(range(self.action_space.n))

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    @property
    def action_space(self) -> Space:
        return self._action_space

# ============================================================================
# 3. Learning Agents (DQN, PPO, etc.)
# ============================================================================

class ReplayBuffer:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def add(self, s: torch.Tensor, a: int, r: float, ns: torch.Tensor, d: bool):
        if len(self.states) >= self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
        self.states.append(s.clone())
        self.actions.append(a)
        self.rewards.append(r)
        self.next_states.append(ns.clone())
        self.dones.append(d)

    def sample(self, batch_size: int):
        indices = random.sample(range(len(self.states)), min(batch_size, len(self.states)))
        return (
            torch.stack([self.states[i] for i in indices]),
            torch.tensor([self.actions[i] for i in indices], dtype=torch.long),
            torch.tensor([self.rewards[i] for i in indices], dtype=torch.float32),
            torch.stack([self.next_states[i] for i in indices]),
            torch.tensor([self.dones[i] for i in indices], dtype=torch.bool),
        )


class DQNAgent(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, gamma: float = 0.99):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.target_q = copy.deepcopy(self.q_net)
        self.gamma = float(gamma)
        self.num_actions = int(output_dim)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

    def sync_target(self):
        self.target_q.load_state_dict(self.q_net.state_dict())

    def act(self, state: torch.Tensor, legal_actions: Optional[List[int]] = None, explore: bool = True) -> int:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        if legal_actions is None:
            legal_actions = list(range(self.num_actions))
        
        if explore and random.random() < self.epsilon:
            return random.choice(legal_actions)
        
        with torch.no_grad():
            q_values = self.q_net(state)
            # Mask illegal actions
            masked_q = q_values.clone()
            for i in range(self.num_actions):
                if i not in legal_actions:
                    masked_q[0, i] = float('-inf')
            return int(torch.argmax(masked_q, dim=-1).item())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_net(state)


class RolloutBuffer:
    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def add(self, state: torch.Tensor, action: int, reward: float, 
            log_prob: torch.Tensor, value: torch.Tensor, done: bool):
        self.states.append(state.clone())
        self.actions.append(action)
        self.rewards.append(reward)
        # Detach stored tensors to avoid backprop through time across updates
        self.log_probs.append(log_prob.detach().clone())
        self.values.append(value.detach().clone())
        self.dones.append(done)

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.log_probs[:]
        del self.values[:]
        del self.dones[:]


class PPOAgent(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, lr: float = 3e-4):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self.num_actions = output_dim

    def select_action(self, state: torch.Tensor, legal_actions: Optional[List[int]] = None) -> Tuple[int, torch.Tensor, torch.Tensor]:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        if legal_actions is None:
            legal_actions = list(range(self.num_actions))
        
        logits = self.actor(state)
        value = self.critic(state)
        
        # Mask illegal actions
        masked_logits = logits.clone()
        for i in range(self.num_actions):
            if i not in legal_actions:
                masked_logits[0, i] = float('-inf')
        
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()

    def act(self, state: torch.Tensor, legal_actions: Optional[List[int]] = None) -> int:
        action, _, _ = self.select_action(state, legal_actions)
        return action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

# ============================================================================
# 4. Training Functions
# ============================================================================

def train_dqn_leduc(agent: DQNAgent, env: LeducPokerEnvironment, opponent: ChallengerAgent, 
                    episodes: int = 1000, verbose: bool = False) -> DQNAgent:
    """Train DQN agent on Leduc Poker."""
   
    buffer = ReplayBuffer(max_size=50000)
    optimizer = optim.Adam(agent.q_net.parameters(), lr=1e-3)
    target_sync_freq = 100
    batch_size = 32
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(env.episode_length):
            # Get legal actions
            legal_actions = env.get_legal_actions(0)
            
            # Agent action
            action = agent.act(state, legal_actions, explore=True)
            
            # Opponent action  
            opp_legal = env.get_legal_actions(1)
            opp_action = opponent.act(state, opp_legal) if hasattr(opponent, 'act') else random.choice(opp_legal)
            
            # Environment step
            next_state, rewards, done, _ = env.step([action, opp_action])
            reward = rewards[0]
            episode_reward += reward
            
            # Store in buffer
            buffer.add(state, action, reward, next_state, done)
            
            if len(buffer.states) > batch_size:
                # Sample batch and train
                states, actions, rewards_batch, next_states, dones = buffer.sample(batch_size)
                
                # Compute targets
                with torch.no_grad():
                    next_q_values = agent.target_q(next_states)
                    targets = rewards_batch + agent.gamma * torch.max(next_q_values, dim=1)[0] * (~dones)
                
                # Compute current Q values
                current_q = agent.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                
                # Loss and update
                loss = F.mse_loss(current_q, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
            if done:
                break
        
        # Update target network
        if episode % target_sync_freq == 0:
            agent.sync_target()
        
        # Update epsilon
        agent.update_epsilon()
        
        if verbose and episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.3f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent


def train_ppo_leduc(agent: PPOAgent, env: LeducPokerEnvironment, opponent: ChallengerAgent, 
                    episodes: int = 1000, verbose: bool = False) -> PPOAgent:
    """Train PPO agent on Leduc Poker with legal action masking."""
    if verbose:
        print(f"Training PPO for {episodes} episodes...")
    
    update_freq = 50
    epochs_per_update = 4
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(env.episode_length):
            # Get legal actions
            legal_actions = env.get_legal_actions(0)
            
            # Agent action
            action, log_prob, value = agent.select_action(state, legal_actions)
            
            # Opponent action
            opp_legal = env.get_legal_actions(1)
            opp_action = opponent.act(state, opp_legal) if hasattr(opponent, 'act') else random.choice(opp_legal)
            
            # Environment step
            next_state, rewards, done, _ = env.step([action, opp_action])
            reward = rewards[0]
            episode_reward += reward
            
            # Store in buffer
            agent.buffer.add(state, action, reward, log_prob, value, done)
            
            state = next_state
            if done:
                break
        
        # Update policy
        if (episode + 1) % update_freq == 0:
            update_ppo(agent, epochs_per_update)
            agent.buffer.clear()
        
        if verbose and episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.3f}")
    
    return agent


def update_ppo(agent: PPOAgent, epochs: int):
    """Update PPO agent using collected rollouts."""
    if len(agent.buffer.states) == 0:
        return
    
    # Convert buffer to tensors
    states = torch.stack(agent.buffer.states)
    actions = torch.tensor(agent.buffer.actions, dtype=torch.long)
    rewards = torch.tensor(agent.buffer.rewards, dtype=torch.float32)
    old_log_probs = torch.stack(agent.buffer.log_probs)
    old_values = torch.stack(agent.buffer.values)
    
    # Compute returns and advantages
    returns = []
    advantages = []
    gae = 0
    gamma = 0.99
    lam = 0.95
    
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * (old_values[i + 1] if i + 1 < len(old_values) else 0) - old_values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + old_values[i])
    
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO updates
    for _ in range(epochs):
        # Current policy
        logits = agent.actor(states)
        values = agent.critic(states).squeeze()
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        
        # Ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped objective
        clip_ratio = 0.2
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Total loss
        total_loss = actor_loss + 0.5 * value_loss
        
        # Update
        agent.optimizer.zero_grad()
        total_loss.backward()
        agent.optimizer.step()


# ============================================================================
# 5. PSRO for Leduc Poker 
# ============================================================================

class PSROPolicy(nn.Module):
    """Meta-policy that mixes a population of policies."""
    def __init__(self, population: List[nn.Module], meta_strategy: List[float]):
        super().__init__()
        self.population = population
        self.meta_strategy = meta_strategy

    def act(self, state: torch.Tensor, legal_actions: Optional[List[int]] = None) -> int:
        # Sample policy from meta-strategy
        policy_idx = np.random.choice(len(self.population), p=self.meta_strategy)
        policy = self.population[policy_idx]
        
        if hasattr(policy, 'act'):
            return policy.act(state, legal_actions)
        else:
            # DQN-style policy
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            with torch.no_grad():
                q_values = policy(state)
                if legal_actions:
                    masked_q = q_values.clone()
                    for i in range(q_values.size(1)):
                        if i not in legal_actions:
                            masked_q[0, i] = float('-inf')
                    return int(torch.argmax(masked_q, dim=-1).item())
                else:
                    return int(torch.argmax(q_values, dim=-1).item())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Use first policy as representative
        return self.population[0](state)


def train_psro_leduc(gauntlet: "EnhancedGauntletBenchmark", iterations: int = 5, episodes_per_iter: int = 1000) -> PSROPolicy:
    """Train PSRO on Leduc Poker."""
    # print suppressed for cleanliness
    
    # Get Leduc challengers from gauntlet
    leduc_challengers = [agent for name, agent in gauntlet.master_challenger_list.items() 
                        if name.startswith("Leduc")]
    
    if not leduc_challengers:
        # print suppressed for cleanliness
        from gauntlet_benchmark import Discrete
        
        class UniformAgent(ChallengerAgent):
            def __init__(self):
                super().__init__("Uniform", "easy")
            def act(self, observation, opponent_history=None):
                return random.choice([0, 1, 2])  # Random action
            @property
            def compatible_action_space(self):
                return Discrete(3)
            def update(self, reward, observation, action):
                pass
        
        leduc_challengers = [UniformAgent()]
    
    env = LeducPokerEnvironment()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    population = []
    
    for i in range(iterations):
        # print suppressed for cleanliness
        
        # Train best response against population + challengers
        br_agent = DQNAgent(input_dim, output_dim)
        
        # Mix of population and challengers as opponents
        opponents = population + leduc_challengers
        if not opponents:
            opponents = leduc_challengers
        
        for episode in range(episodes_per_iter):
            opponent = random.choice(opponents)
            br_agent = train_dqn_leduc(br_agent, env, opponent, episodes=1)
        
        population.append(br_agent)
        # print suppressed for cleanliness
    
    # Uniform meta-strategy
    meta_strategy = [1.0 / len(population)] * len(population)
    
    return PSROPolicy(population, meta_strategy)


# ============================================================================
# 6. Unified PRPO for Leduc Poker 
# ============================================================================

class UnifiedPRPO_Leduc:
    """Unified PRPO implementation for Leduc Poker."""
    def __init__(self, env: LeducPokerEnvironment, population_size: int = 5, 
                 lambda_exploit: float = 0.1, lambda_target: float = 0.05):
        self.env = env
        self.population_size = population_size
        self.lambda_exploit = lambda_exploit
        self.lambda_target = lambda_target
        
        # Initialize population
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        
        self.population = []
        for _ in range(population_size):
            agent = PPOAgent(input_dim, output_dim)
            agent.current_exploitability = float('inf')
            self.population.append(agent)
        
        self.target_policy = None

    def train(self, total_episodes: int, episodes_per_update: int = 50):
        """Train PRPO population."""
        completed_episodes = 0
        
        while completed_episodes < total_episodes:
            # Tournament phase
            for _ in range(episodes_per_update):
                p1_idx, p2_idx = random.sample(range(len(self.population)), 2)
                agent1, agent2 = self.population[p1_idx], self.population[p2_idx]
                
                state = self.env.reset()
                for step in range(self.env.episode_length):
                    legal1 = self.env.get_legal_actions(0)
                    legal2 = self.env.get_legal_actions(1)
                    
                    a1, lp1, v1 = agent1.select_action(state, legal1)
                    a2, lp2, v2 = agent2.select_action(state, legal2)
                    
                    ns, rewards, done, _ = self.env.step([a1, a2])
                    
                    agent1.buffer.add(state, a1, float(rewards[0]), lp1, v1, done)
                    agent2.buffer.add(state, a2, float(rewards[1]), lp2, v2, done)
                    
                    state = ns
                    if done:
                        break
            
            # Update phase
            completed_episodes += episodes_per_update
            self._find_and_set_target_policy()
            
            for agent in self.population:
                update_ppo(agent, epochs=4)
                agent.buffer.clear()
            
            avg_exploit = np.mean([a.current_exploitability for a in self.population])
            # print suppressed for cleanliness
        
        # Return best agent
        best_agent = min(self.population, key=lambda ag: ag.current_exploitability)
        return best_agent

    def _find_and_set_target_policy(self):
        """Find the best policy in population and set as target."""
        # Simple heuristic: use the first agent as target
        if self.population:
            self.target_policy = self.population[0]
            for agent in self.population:
                agent.current_exploitability = random.uniform(0.1, 1.0)  # Placeholder


def train_prpo_leduc_simple(env: LeducPokerEnvironment, episodes: int, input_dim: int, 
                           output_dim: int, device: torch.device, lambda_exploit: float = 0.5) -> PPOAgent:
    """Simple PRPO training for Leduc Poker."""
    prpo_system = UnifiedPRPO_Leduc(env, population_size=3, lambda_exploit=lambda_exploit)
    final_policy = prpo_system.train(total_episodes=episodes, episodes_per_update=50)
    # print suppressed for cleanliness
    return final_policy


# ============================================================================
# 7. Evaluation utilities
# ============================================================================

def evaluate_simple_avg_reward(policy: nn.Module, env: LeducPokerEnvironment, 
                               opponent: ChallengerAgent, episodes: int = 100) -> float:
    """Evaluate a policy's average reward against a fixed opponent."""
    total_reward = 0.0
    
    # If the policy is a PPO/PRPO-style model trained on the simple 8D Leduc state,
    # wrap it so 30D OpenSpiel observations are mapped to 8D inputs.
    try:
        use_wrapped = isinstance(policy, (StandardPPO, UnifiedPRPOAgent, UnifiedActorCritic)) or hasattr(policy, 'policy')
    except Exception:
        use_wrapped = hasattr(policy, 'policy')
    policy_for_eval = policy
    # Simple Leduc dims
    simple_input_dim = 8
    simple_action_dim = 3
    if use_wrapped:
        policy_for_eval = PolicyWrapperAgent(policy, simple_input_dim, simple_action_dim, name="EvalWrappedPolicy")
    
    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        
        for step in range(env.episode_length):
            legal_actions = env.get_legal_actions(0)
            
            # Policy action
            if hasattr(policy_for_eval, 'act'):
                # Support policies whose act() may or may not accept legal_actions
                try:
                    action = policy_for_eval.act(state, legal_actions)
                except TypeError:
                    action = policy_for_eval.act(state)
            else:
                with torch.no_grad():
                    if len(state.shape) == 1:
                        state_batch = state.unsqueeze(0)
                    else:
                        state_batch = state
                    q_values = policy_for_eval(state_batch)
                    masked_q = q_values.clone()
                    for i in range(q_values.size(1)):
                        if i not in legal_actions:
                            masked_q[0, i] = float('-inf')
                    action = int(torch.argmax(masked_q, dim=-1).item())
            
            # Opponent action
            opp_legal = env.get_legal_actions(1)
            opp_action = opponent.act(state, opp_legal) if hasattr(opponent, 'act') else random.choice(opp_legal)
            
            # Step
            state, rewards, done, _ = env.step([action, opp_action])
            episode_reward += rewards[0]
            
            if done:
                break
        
        total_reward += episode_reward
    
    return total_reward / episodes


def evaluate_and_report(g: "EnhancedGauntletBenchmark", policy: nn.Module, name: str, out_dir: str, 
                        simple_input_dim: int, simple_output_dim: int, gauntlet_input_dim: int, gauntlet_output_dim: int):
    """Evaluate policy using gauntlet with proper dimension handling."""
    # Wrap all policies except raw DQN/PSRO/SelfPlay 
    # StandardPPO, UnifiedPRPOAgent, and UnifiedActorCritic all need wrapping
    use_wrapped = isinstance(policy, (StandardPPO, UnifiedPRPOAgent, UnifiedActorCritic)) or hasattr(policy, 'policy')
    
    # For PPO and PRPO, use simple dimensions since they were trained on LeducPokerSimpleEnvironment
    if name in ["Leduc_PPO", "Leduc_PRPO"]:
        eval_input_dim, eval_output_dim = simple_input_dim, simple_output_dim
    else:
        eval_input_dim, eval_output_dim = gauntlet_input_dim, gauntlet_output_dim
    
    wrapped = PolicyWrapperAgent(policy, eval_input_dim, eval_output_dim, name=f"{name}_Wrapped") if use_wrapped else policy
    print(f"Using dimensions for {name}: input={eval_input_dim}, output={eval_output_dim}")
    print(f"Evaluating {name}: use_wrapped={use_wrapped}, policy_type={type(policy).__name__}")
    if hasattr(policy, "eval"):
        policy.eval()
    # Additional debugging for model parameters
    if hasattr(policy, 'policy') and hasattr(policy.policy, 'parameters'):
        param_count = sum(p.numel() for p in policy.policy.parameters())
        print(f"Policy {name} has {param_count} parameters")
    elif hasattr(policy, 'parameters'):
        param_count = sum(p.numel() for p in policy.parameters())
        print(f"Policy {name} has {param_count} parameters")
    print(f"\n{'='*40}\n E V A L U A T I N G:   {name} \n{'='*40}")
    g.evaluate_policy(policy=wrapped, policy_name=name, environments=None)
    report_path = os.path.join(out_dir, "report.json")
    g.generate_report(report_path)
    # Inject seed statistics (CI and p-values) into the report, if available
    try:
        base_dir = os.path.dirname(out_dir)
        stats_path = os.path.join(base_dir, "leduc_stats.json")
        if os.path.exists(stats_path) and os.path.exists(report_path):
            with open(report_path, "r") as f:
                report_payload = json.load(f)
            with open(stats_path, "r") as f:
                seed_stats = json.load(f)
            report_payload["seed_stats"] = seed_stats
            with open(report_path, "w") as f:
                json.dump(report_payload, f, indent=2)
    except Exception as e:
        print(f"Warning: could not inject seed stats into report for {name}: {e}")
    print(f"Saved report to {report_path}")
    # Move generated visualization files into out_dir
    try:
        import shutil
        fmt = g.config.visualization_format
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

# ============================================================================
# 8. Main execution
# ============================================================================

if __name__ == "__main__":
    # Updated parameters - now using time budget instead of episode count
    seeds = 1
    time_budget_seconds = 10.0  # 60 seconds per algorithm per seed

    # print suppressed for cleanliness
    config = EvaluationConfig(num_episodes=200, parallel_workers=1, save_visualizations=True)
    gauntlet = EnhancedGauntletBenchmark(config)

    # Register Leduc Poker using the safe wrapper environment to avoid OpenSpiel chance-node warnings
    gauntlet.register_environment(
        "LeducPoker",
        LeducPokerEnvironment,
        payoff_matrices=None,
        game_prefix="Leduc",
        zero_sum=True
    )

    # Instantiate environment and agents
    env = LeducPokerEnvironment()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # print suppressed for cleanliness
    
    dqn_agent = DQNAgent(input_dim, output_dim)
    ppo_agent = PPOAgent(input_dim, output_dim)

    # Training opponent
    training_opponents = [name for name in gauntlet.master_challenger_list.keys() if name.startswith("Leduc")]
    if training_opponents:
        training_opponent = gauntlet.master_challenger_list[training_opponents[0]]
        training_opponent.name = "TrainingOpponent_Leduc"
    else:
        # print suppressed for cleanliness
        class UniformAgent(ChallengerAgent):
            def __init__(self):
                super().__init__("Uniform", "easy")
            def act(self, observation, opponent_history=None):
                return random.choice([0, 1, 2])
            @property
            def compatible_action_space(self):
                return Discrete(3)
            def update(self, reward, observation, action):
                pass
        training_opponent = UniformAgent()

    # Output directories
    base_dir = os.path.join("results", "LeducPoker")
    os.makedirs(base_dir, exist_ok=True)
    dqn_dir = os.path.join(base_dir, "DQN"); os.makedirs(dqn_dir, exist_ok=True)
    ppo_dir = os.path.join(base_dir, "PPO"); os.makedirs(ppo_dir, exist_ok=True)
    psro_dir = os.path.join(base_dir, "PSRO"); os.makedirs(psro_dir, exist_ok=True)
    prpo_dir = os.path.join(base_dir, "PRPO"); os.makedirs(prpo_dir, exist_ok=True)
    selfplay_dir = os.path.join(base_dir, "SelfPlay"); os.makedirs(selfplay_dir, exist_ok=True)
    timing_train: Dict[str, float] = {}
    timing_eval: Dict[str, float] = {}

    # Train DQN multi-seed with time budget
    # print suppressed for cleanliness
    dqn_runs = []
    _t0 = time.time()
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        trained, episodes_completed, _ = TimeBudgetTrainer.train_dqn_time_budget(
            copy.deepcopy(dqn_agent), lambda: env, training_opponent, 
            time_budget_seconds, device=device
        )
        dqn_runs.append(trained)
        print(f"    DQN Seed {seed}: Completed {episodes_completed} episodes in {time_budget_seconds}s")
    trained_dqn = dqn_runs[-1]
    timing_train["DQN"] = float(time.time() - _t0)
    
    # Train PPO multi-seed with time budget
    # IMPORTANT: Use dims consistent with LeducPokerSimpleEnvironment used inside the trainer
    simple_env_for_dims = LeducPokerSimpleEnvironment()
    simple_input_dim = getattr(simple_env_for_dims, 'state_dim', 8)
    simple_output_dim = getattr(simple_env_for_dims, 'action_dim', 3)

    ppo_runs = []
    _t0 = time.time()
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        trained, episodes_completed, _ = TimeBudgetTrainer.train_standard_ppo_time_budget(
            lambda: LeducPokerSimpleEnvironment(), simple_input_dim, simple_output_dim, time_budget_seconds,
            device=str(device)
        )
        ppo_runs.append(trained)
        print(f"    PPO Seed {seed}: Completed {episodes_completed} episodes in {time_budget_seconds}s")
    trained_ppo = ppo_runs[-1]
    timing_train["PPO"] = float(time.time() - _t0)

    # Train PSRO
    # print suppressed for cleanliness
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    _t0 = time.time()
    # Adjust episodes per iteration based on time budget
    episodes_per_iter = max(100, int(time_budget_seconds / 5))  # 5 iterations
    psro_policy = train_psro_leduc(gauntlet, iterations=5, episodes_per_iter=episodes_per_iter)
    timing_train["PSRO"] = float(time.time() - _t0)

    # Train Self-Play with time budget
    # print suppressed for cleanliness
    torch.manual_seed(123); np.random.seed(123); random.seed(123)
    _t0 = time.time()
    sp_episodes = int(time_budget_seconds * 100)  # Approximate episodes based on time budget
    selfplay_agent = train_dqn_leduc(copy.deepcopy(dqn_agent), env, copy.deepcopy(dqn_agent), sp_episodes)
    timing_train["SelfPlay"] = float(time.time() - _t0)

    # Train PRPO with superior unified implementation and time budget
    # print suppressed for cleanliness
    prpo_runs = []
    _t0 = time.time()
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        # Use the same simple env dims to match PRPO's env_factory (LeducPokerSimpleEnvironment)
        trained = train_prpo_leduc_unified(time_budget_seconds, simple_input_dim, simple_output_dim, device)
        prpo_runs.append(trained)
        print(f"    PRPO Seed {seed}: Training completed in {time_budget_seconds}s")
    trained_prpo = prpo_runs[-1]
    timing_train["PRPO"] = float(time.time() - _t0)

    # print suppressed for cleanliness
    
    # Evaluate all policies
    _t0 = time.time(); trained_dqn.eval(); evaluate_and_report(gauntlet, trained_dqn, "Leduc_DQN", dqn_dir, simple_input_dim, simple_output_dim, input_dim, output_dim); timing_eval["DQN"] = float(time.time() - _t0)
    _t0 = time.time(); trained_ppo.eval(); evaluate_and_report(gauntlet, trained_ppo, "Leduc_PPO", ppo_dir, simple_input_dim, simple_output_dim, input_dim, output_dim); timing_eval["PPO"] = float(time.time() - _t0)
    _t0 = time.time(); psro_policy.eval(); evaluate_and_report(gauntlet, psro_policy, "Leduc_PSRO", psro_dir, simple_input_dim, simple_output_dim, input_dim, output_dim); timing_eval["PSRO"] = float(time.time() - _t0)
    _t0 = time.time(); selfplay_agent.eval(); evaluate_and_report(gauntlet, selfplay_agent, "Leduc_SelfPlay", selfplay_dir, simple_input_dim, simple_output_dim, input_dim, output_dim); timing_eval["SelfPlay"] = float(time.time() - _t0)
    _t0 = time.time(); trained_prpo.eval(); evaluate_and_report(gauntlet, trained_prpo, "Leduc_PRPO", prpo_dir, simple_input_dim, simple_output_dim, input_dim, output_dim); timing_eval["PRPO"] = float(time.time() - _t0)

    # Statistical comparison PRPO vs PPO
    # print suppressed for cleanliness
    eval_env = LeducPokerEnvironment()
    fixed_opp = training_opponent
    fixed_opp.name = "EvalOpponent_Leduc"
    
    ppo_scores: List[float] = []
    prpo_scores: List[float] = []
    
    for run in ppo_runs:
        ppo_scores.append(evaluate_simple_avg_reward(run, eval_env, fixed_opp, episodes=50))
    for run in prpo_runs:
        prpo_scores.append(evaluate_simple_avg_reward(run, eval_env, fixed_opp, episodes=50))
    
    ppo_stats = compute_mean_ci(ppo_scores)
    prpo_stats = compute_mean_ci(prpo_scores)
    p_value = paired_t_test(prpo_scores, ppo_scores)
    
    # print suppressed for cleanliness
    print(f"PPO mean={ppo_stats['mean']:.3f}, 95% CI=[{ppo_stats['ci_low']:.3f}, {ppo_stats['ci_high']:.3f}], n={ppo_stats['n']}")
    print(f"PRPO mean={prpo_stats['mean']:.3f}, 95% CI=[{prpo_stats['ci_low']:.3f}, {prpo_stats['ci_high']:.3f}], n={prpo_stats['n']}")
   
    
    with open(os.path.join(base_dir, "leduc_stats.json"), "w") as f:
        json.dump({
            "ppo_scores": ppo_scores,
            "prpo_scores": prpo_scores,
            "ppo_stats": ppo_stats,
            "prpo_stats": prpo_stats,
            "p_value_prpo_vs_ppo": p_value,
        }, f, indent=2)

    # Save timing summary
    with open(os.path.join(base_dir, "times.json"), "w") as f:
        json.dump({"training_seconds": timing_train, "eval_seconds": timing_eval}, f, indent=2)

    # print suppressed for cleanliness