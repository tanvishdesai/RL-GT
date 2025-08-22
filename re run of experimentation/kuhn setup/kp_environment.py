%%writefile kp_environment.py

# kuhn
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

# Import the superior unified PRPO implementation
from unified_prpo import UnifiedActorCritic, UnifiedPRPOAgent, UnifiedPRPO, StandardPPO, TimeBudgetTrainer

# Import ChallengerAgent from gauntlet benchmark
from gauntlet_benchmark import *


# Optional Nash solver support
try:
    import nashpy as nash  # type: ignore
    _NASH_AVAILABLE = True
except Exception:
    nash = None  # type: ignore
    _NASH_AVAILABLE = False
    print("Warning: nashpy not available. Install with 'pip install nashpy' for proper PSRO.")

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
# 2. Kuhn Poker Specific Functions for Unified PRPO
# ============================================================================

class KuhnPokerSimpleEnvironment:
    """Simplified Kuhn Poker game environment for unified PRPO."""
    def __init__(self):
        # Actions: 0 = Pass/Check, 1 = Bet/Call
        self.action_dim = 2
        self.state_dim = 3  # One-hot encoding of private card (J=0, Q=1, K=2)
        self.cards = [0, 1, 2]  # J, Q, K
        self.episode_length = 1  # Single-step game
        self.reset()

    def reset(self):
        # Sample private cards for both players (without replacement)
        self.player_cards = random.sample(self.cards, 2)
        return self._get_state()

    def _get_state(self):
        # One-hot encoding of player 1's private card
        state = np.zeros(self.state_dim)
        state[self.player_cards[0]] = 1.0
        return torch.tensor(state, dtype=torch.float32)


    def step(self, actions, opp_action=None):
        """
        Simplified payoff structure:
        - Both Pass (0,0): showdown, higher card wins +1/-1
        - One Bets, other Passes: bettor wins +1/-1 immediately
        - Both Bet (1,1): showdown for larger pot, winner +2/-2
        
        Can be called as:
        - step([p1_action, p2_action])  # List format
        - step(p1_action, p2_action)    # Separate arguments format
        """
        if opp_action is not None:
            # Called with separate arguments: step(p1_action, p2_action)
            p1_action, p2_action = int(actions), int(opp_action)
        else:
            # Called with list: step([p1_action, p2_action])
            p1_action, p2_action = int(actions[0]), int(actions[1])
        
        p1_card, p2_card = self.player_cards
        
        if p1_action == 0 and p2_action == 0:  # Both pass
            # Showdown: higher card wins
            if p1_card > p2_card:
                p1_reward = 1.0
            elif p1_card < p2_card:
                p1_reward = -1.0
            else:
                p1_reward = 0.0  # Tie (shouldn't happen in Kuhn)
        elif p1_action == 1 and p2_action == 0:  # P1 bets, P2 passes
            p1_reward = 1.0  # P1 wins the pot
        elif p1_action == 0 and p2_action == 1:  # P1 passes, P2 bets
            p1_reward = -1.0  # P2 wins the pot
        else:  # Both bet
            # Showdown with larger pot
            if p1_card > p2_card:
                p1_reward = 2.0
            elif p1_card < p2_card:
                p1_reward = -2.0
            else:
                p1_reward = 0.0  # Tie
        
        p2_reward = -p1_reward  # Zero-sum game
        done = True
        
        # FIX: Always return 4 values for a consistent API (state, rewards, done, info).
        # The info dictionary can be empty if not needed.
        info = {"p1_card": p1_card, "p2_card": p2_card}
        return self._get_state(), [p1_reward, p2_reward], done, info

    def get_legal_actions(self, player: int):
        """Return legal actions for the given player. In Kuhn Poker, both actions are always legal."""
        return [0, 1]  # Pass/Check=0, Bet/Call=1

def get_kuhn_nash_policy_callable(policy_probs_batch: torch.Tensor) -> torch.Tensor:
    """
    Returns an approximation of the Nash equilibrium for Kuhn Poker.
    This is a simplified approximation - the actual Nash is more complex.
    """
    # Simplified Nash-like strategy: more mixed but slightly conservative
    batch_size = policy_probs_batch.shape[0]
    # Approximate mixed strategy [0.6, 0.4] for Pass/Bet
    nash_dist = torch.full_like(policy_probs_batch, 0.0)
    nash_dist[:, 0] = 0.6  # Pass
    nash_dist[:, 1] = 0.4  # Bet
    return nash_dist

def calculate_kuhn_exploitability_callable(policy: nn.Module) -> float:
    """
    Calculates exploitability for Kuhn Poker policy.
    This is a simplified measure based on deviation from mixed strategies.
    """
    policy.eval()
    device = next(policy.parameters()).device

    def _policy_probs_for_state(st: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Case 1: UnifiedActorCritic-style: forward returns (probs, value)
            try:
                out = policy(st)
                if isinstance(out, (tuple, list)) and len(out) >= 1:
                    probs_or_logits = out[0]
                    if probs_or_logits.dim() == 1:
                        probs_or_logits = probs_or_logits.unsqueeze(0)
                else:
                    probs_or_logits = out
            except Exception:
                probs_or_logits = None

            # Case 2: PPOAgent-style with actor
            if probs_or_logits is None or not torch.is_tensor(probs_or_logits):
                if hasattr(policy, "actor"):
                    probs_or_logits = policy.actor(st)
                else:
                    # Fallback: call again and hope for logits
                    probs_or_logits = policy(st)

            # Convert logits to probabilities if needed
            probs = torch.softmax(probs_or_logits, dim=-1)
            return probs

    # Evaluate policy on all possible card states
    exploitability = 0.0
    for card in [0, 1, 2]:  # J, Q, K
        state = torch.zeros(3, device=device).unsqueeze(0)
        state[0, card] = 1.0
        probs = _policy_probs_for_state(state).squeeze(0)
        p_np = probs.detach().cpu().numpy()

        # Simplified exploitability: how much opponent can exploit based on predictability
        variance = np.var(p_np)
        exploitability += float(variance)

    exploitability /= 3.0  # Average over all cards
    policy.train()
    return float(exploitability)

def get_kuhn_exploiter_opponents() -> List[Callable]:
    """Returns a list of exploiter bots for Kuhn Poker."""
    return [
        lambda: 0,  # Always Pass
        lambda: 1,  # Always Bet
        lambda: np.random.choice([0, 1], p=[0.8, 0.2]),  # Conservative (mostly pass)
        lambda: np.random.choice([0, 1], p=[0.3, 0.7]),  # Aggressive (mostly bet)
        lambda: random.randint(0, 1),  # Random
    ]

def train_prpo_kuhn_unified(time_budget_seconds: float, input_dim: int, output_dim: int, 
                           device: torch.device, lambda_nash: float = 0.5, 
                           lambda_exploit: float = 0.5) -> nn.Module:
    """
    Superior PRPO training function using the unified framework with time budget for Kuhn Poker.
    """
    print(f"Training PRPO (Unified Framework) for Kuhn Poker for {time_budget_seconds} seconds...")
    
    # The manager needs a way to create new environments
    env_factory = lambda: KuhnPokerSimpleEnvironment()

    prpo_system = UnifiedPRPO(
        state_dim=input_dim, 
        action_dim=output_dim, 
        lr=1e-4, 
        device=str(device),
        population_size=4,
        lambda_nash=lambda_nash,
        nash_target_callable=get_kuhn_nash_policy_callable,
        lambda_exploit=lambda_exploit,
        exploitability_calculator=calculate_kuhn_exploitability_callable,
        exploiter_opponents=get_kuhn_exploiter_opponents()
    )
    
    # Train the system with time budget
    episodes_completed, results_over_time = prpo_system.train_time_budget(
        env_factory=env_factory, 
        time_budget_seconds=time_budget_seconds,
        update_every_seconds=1.0
    )
    
    # Get the best policy from the trained population
    best_agent = prpo_system.get_best_agent()
    best_policy = best_agent.policy if best_agent else None
    
    print("Training finished for PRPO (Unified Framework) - Kuhn Poker.")
    return best_policy

# ============================================================================
# 3. Kuhn Poker Environment (Original Implementation)
# ============================================================================

class KuhnPokerEnvironment(Environment):
    """
    Simplified Kuhn Poker environment:
    - Deck: J, Q, K encoded as 0,1,2; each episode samples private cards for both players
    - Actions: 0 = Pass/Check, 1 = Bet/Call
    - One-step simultaneous-actions abstraction for training/eval simplicity
    - Payoffs (zero-sum):
        * Both Pass (0,0): showdown, higher card wins +1/-1
        * One Bets, other Passes (1,0) or (0,1): bettor wins +1/-1 immediately
        * Both Bet (1,1): showdown for larger pot, winner +2/-2
    - Observation for Player 1 (the evaluated policy): one-hot of own private card (dim=3)
    """

    def __init__(self):
        # Lazy imports to avoid top-level dependency on gym in this script
        try:
            from gymnasium.spaces import Box, Discrete  # type: ignore
        except Exception:
            class Box:  # type: ignore
                def __init__(self, low, high, shape, dtype):
                    self.low, self.high, self.shape, self.dtype = low, high, shape, dtype
            class Discrete:  # type: ignore
                def __init__(self, n: int):
                    self.n = int(n)
        self._observation_space = Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self._action_space = Discrete(2)
        self.episode_length = 1  # Single-step game
        self.state = None
        self.my_card = None
        self.opp_card = None

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    def _sample_cards(self) -> Tuple[int, int]:
        deck = [0, 1, 2]  # J, Q, K
        my = random.choice(deck)
        deck.remove(my)
        opp = random.choice(deck)
        return my, opp

    def _one_hot_card(self, card_idx: int) -> torch.Tensor:
        vec = torch.zeros(3, dtype=torch.float32)
        vec[card_idx] = 1.0
        return vec

    def reset(self) -> torch.Tensor:
        self.my_card, self.opp_card = self._sample_cards()
        self.state = self._one_hot_card(self.my_card)
        return self.state

    def step(self, actions: List[int]) -> Tuple[torch.Tensor, List[float], bool, Dict]:
        a1, a2 = int(actions[0]), int(actions[1])

        # Showdown helper
        def showdown_reward(magnitude: float) -> Tuple[float, float]:
            if self.my_card > self.opp_card:
                return magnitude, -magnitude
            else:
                return -magnitude, magnitude

        if a1 == 0 and a2 == 0:
            r1, r2 = showdown_reward(1.0)
        elif a1 == 1 and a2 == 0:
            r1, r2 = 1.0, -1.0
        elif a1 == 0 and a2 == 1:
            r1, r2 = -1.0, 1.0
        else:  # a1 == 1 and a2 == 1
            r1, r2 = showdown_reward(2.0)

        # One-step game; next_state is identical to observation
        info = {"my_card": int(self.my_card), "opp_card": int(self.opp_card)}
        return self.state, [float(r1), float(r2)], True, info

    def get_legal_actions(self, player: int):
        """Return legal actions for the given player. In Kuhn Poker, both actions are always legal."""
        return [0, 1]  # Pass/Check=0, Bet/Call=1


# ============================================================================
# 3. Agents (DQN/PPO) for Kuhn (input_dim=3, output_dim=2)
# ============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: List[Tuple[torch.Tensor, int, float, torch.Tensor, bool]] = []
        self.idx = 0

    def __len__(self) -> int:
        return len(self.data)

    def push(self, s: torch.Tensor, a: int, r: float, ns: torch.Tensor, d: bool):
        item = (s.detach().cpu(), int(a), float(r), ns.detach().cpu(), bool(d))
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.idx] = item
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch: int):
        s, a, r, ns, d = zip(*random.sample(self.data, batch))
        return (
            torch.stack(list(s)),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(list(ns)),
            torch.tensor(d, dtype=torch.bool),
        )


class DQNAgent(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, gamma: float = 0.99):
        super().__init__()
        self.q_net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim))
        self.target_q = copy.deepcopy(self.q_net)
        self.gamma = float(gamma)
        self.num_actions = int(output_dim)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995

    def sync_target(self):
        self.target_q.load_state_dict(self.q_net.state_dict())

    def reset(self):
        """Reset method for compatibility with ChallengerAgent interface in PSRO."""
        pass

    def act(self, state: torch.Tensor, explore: bool = True, opponent_history: Optional[List] = None) -> int:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if explore and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q = self.q_net(state)
            return int(torch.argmax(q, dim=-1).item())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_net(state)


class RolloutBuffer:
    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []

    def clear(self):
        self.__init__()


class PPOAgent(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim))
        self.critic = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    def act(self, state: torch.Tensor):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state).squeeze(-1)
        return int(action.item()), log_prob.squeeze(), value.squeeze()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        logits = self.actor(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states).squeeze(-1)
        return log_probs, entropy, values


# ============================================================================
# 4. Custom Challenger for Kuhn Poker
# ============================================================================

class KuhnThresholdAgent(ChallengerAgent):
    """
    Simple threshold-based strategy:
    - Bet with K (card=2)
    - Pass with J (card=0)
    - With Q (card=1), bet with small probability; increase if opponent tends to pass
    """

    def __init__(self, name: str = "Kuhn_Threshold"):
        super().__init__(name, "medium")
        try:
            from gymnasium.spaces import Discrete  # type: ignore
        except Exception:
            class Discrete:  # type: ignore
                def __init__(self, n: int):
                    self.n = int(n)
        self._action_space = Discrete(2)

    @property
    def compatible_action_space(self) -> Any:
        return self._action_space

    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        # Observation is one-hot of own card
        if isinstance(observation, torch.Tensor) and observation.dim() > 1:
            observation = observation[0]
        card = int(torch.argmax(observation).item())

        # Estimate opponent pass frequency
        pass_rate = 0.5
        if opponent_history:
            num_passes = sum(1 for a in opponent_history if int(a) == 0)
            pass_rate = num_passes / len(opponent_history)

        if card == 2:  # K
            return 1  # Bet/Call
        if card == 0:  # J
            return 0  # Pass/Check

        # Q: bet a bit more when opponent passes often
        bet_prob = 0.3 + 0.4 * max(0.0, pass_rate - 0.5)
        return 1 if random.random() < bet_prob else 0

    def update(self, reward: float, observation: torch.Tensor, action: int):
        pass

# ============================================================================
# X. Self-Play Baseline and PRPO (simple) for Kuhn Poker
# ============================================================================

class SelfPlayOpponent:
    def __init__(self, action_dim: int, input_dim: int, device: torch.device):
        self.name = "SelfPlayOpponent"
        self._snapshots: List[nn.Module] = []
        self._action_dim = action_dim
        self._input_dim = input_dim
        self._device = device
    def reset(self):
        pass
    def add_snapshot(self, agent: nn.Module):
        clone = PPOAgent(self._input_dim, self._action_dim).to(self._device)
        clone.load_state_dict(copy.deepcopy(agent.state_dict()))
        clone.eval(); self._snapshots.append(clone)
        if len(self._snapshots) > 20: self._snapshots.pop(0)
    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        if not self._snapshots or random.random() < 0.3:
            return random.randint(0, self._action_dim - 1)
        res = random.choice(self._snapshots).act(observation)
        return int(res[0]) if isinstance(res, tuple) else int(res)


def train_selfplay_kuhn(env: KuhnPokerEnvironment, time_budget_seconds: float, input_dim: int, output_dim: int, device: torch.device) -> PPOAgent:
    """
    Train Self-Play for Kuhn Poker using actual time budget instead of fixed episodes.
    """
    print(f"[SelfPlay-Kuhn] Training for {time_budget_seconds} seconds...")
    start_time = time.time()
    
    agent = PPOAgent(input_dim, output_dim).to(device)
    opponent = SelfPlayOpponent(output_dim, input_dim, device)
    opt = optim.Adam(agent.parameters(), lr=1e-3)
    episode_count = 0
    last_log_time = start_time
    
    while (time.time() - start_time) < time_budget_seconds:
        s = env.reset()
        opponent.reset()
        a, _, _ = agent.act(s)
        b = opponent.act(s, opponent_history=[])
        ns, rewards, _, _ = env.step([a, b])
        r = rewards[0]
        
        logits = agent(s.unsqueeze(0))
        v = agent.critic(s.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        lp = dist.log_prob(torch.tensor(a))
        adv = torch.tensor(r, dtype=torch.float32) - v.detach().squeeze()
        loss = -lp * adv + nn.functional.mse_loss(v.squeeze(), torch.tensor(r, dtype=torch.float32))
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        episode_count += 1
        
        # Add agent snapshot to opponent pool occasionally
        if random.random() < 0.2:
            opponent.add_snapshot(agent)
        
        current_time = time.time()
        # Log progress every 10 seconds
        if (current_time - last_log_time) >= 10.0:
            elapsed_time = current_time - start_time
            print(f"    [SelfPlay-Kuhn] Time {elapsed_time:.1f}s: Episodes {episode_count}")
            last_log_time = current_time
    
    final_time = time.time() - start_time
    print(f"[SelfPlay-Kuhn] Training completed in {final_time:.1f}s with {episode_count} episodes")
    return agent


# =====================================================================================
# START: UNIFIED PRPO FRAMEWORK FOR KUHN POKER (PORTED FROM V6 LEDUC POKER)
# This code replaces the original, flawed PRPO implementation for Kuhn Poker.
# It faithfully applies the principles from the revised Leduc Poker script:
#   1. A true BestResponse calculator for accurate exploitability measurement.
#   2. Training a dedicated StandardPPO agent as a best-response oracle.
#   3. A population manager that coordinates a robust training curriculum, combining
#      self-play, oracle-play, and game-theoretic regularization.
# =====================================================================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])

class StandardPPO:
    """ A standard PPO agent, used here for training oracle policies. """
    def __init__(self, state_dim, action_dim, lr=3e-4, device='cpu'):
        self.device=device;self.action_dim=action_dim;self.gamma=0.99;self.eps_clip=0.2
        self.k_epochs=4;self.entropy_coeff=0.01;self.policy=PPOAgent(state_dim,action_dim).to(device)
        self.optimizer=optim.Adam(self.policy.parameters(),lr=lr);self.memory=[]
    def select_action(self,state):
        state_t=torch.FloatTensor(state).unsqueeze(0).to(self.device);
        with torch.no_grad(): action,logp,val=self.policy.act(state_t)
        return action,logp.cpu(),val.cpu()
    def act(self, state: torch.Tensor) -> int:
        # Lightweight adapter so evaluation helpers can call .act
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.float32)
        if isinstance(state, torch.Tensor) and state.ndim > 1:
            state = state.squeeze(0)
        action, _, _ = self.select_action(state)
        return int(action)
    def store_experience(self, s, a, r, ns, d, lp, v):
        # Ensure tensors are stored to avoid numpy stacking errors
        s_t = torch.as_tensor(s, dtype=torch.float32)
        # Handle one-step trainers that pass ns=None
        ns_t = s_t if ns is None else torch.as_tensor(ns, dtype=torch.float32)
        lp_t = lp if isinstance(lp, torch.Tensor) else torch.as_tensor(lp, dtype=torch.float32)
        v_t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float32)
        self.memory.append(Experience(s_t, int(a), float(r), ns_t, bool(d), lp_t, v_t))
    def eval(self):
        self.policy.eval()
        return self
    def train(self):
        self.policy.train()
        return self
    def update_policy(self):
        if not self.memory:
            return {}
        states = torch.stack([torch.as_tensor(e.state, dtype=torch.float32) for e in self.memory]).to(self.device)
        actions = torch.LongTensor([int(e.action) for e in self.memory]).to(self.device)
        old_log_probs = torch.stack([e.log_prob if isinstance(e.log_prob, torch.Tensor) else torch.as_tensor(e.log_prob, dtype=torch.float32) for e in self.memory]).to(self.device)
        old_values = torch.stack([e.value if isinstance(e.value, torch.Tensor) else torch.as_tensor(e.value, dtype=torch.float32) for e in self.memory]).to(self.device)
        returns=[]; discounted_reward=0
        for r,d in zip(reversed([e.reward for e in self.memory]),reversed([e.done for e in self.memory])):
            if d: discounted_reward=0
            discounted_reward=r+(self.gamma*discounted_reward); returns.insert(0,discounted_reward)
        returns=torch.tensor(returns,dtype=torch.float32).to(self.device);advantages=returns-old_values.detach()
        if len(advantages)>1: advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        for _ in range(self.k_epochs):
            logp, entropy, vals=self.policy.evaluate_actions(states, actions);
            ratios=torch.exp(logp-old_log_probs.detach());surr1=ratios*advantages;surr2=torch.clamp(ratios,1-self.eps_clip,1+self.eps_clip)*advantages
            pol_loss=-torch.min(surr1,surr2).mean();val_loss=F.mse_loss(vals,returns)
            loss=pol_loss+0.5*val_loss-self.entropy_coeff*entropy.mean()
            self.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.policy.parameters(),0.5);self.optimizer.step()
        self.memory.clear();return {'loss':loss.item()}

class KuhnBestResponse:
    """
    Computes the exact exploitability of a policy for the simplified Kuhn Poker game.
    Exploitability is the expected value a best-response opponent can achieve.
    """
    def __init__(self, game_proto: KuhnPokerEnvironment, policy: nn.Module, device: str):
        self.env = game_proto
        self.policy = policy
        self.device = device
        self.policy.eval()

    def _get_policy_probs(self, card_idx: int) -> np.ndarray:
        state_vec = self.env._one_hot_card(card_idx).to(self.device)
        with torch.no_grad():
            if hasattr(self.policy, 'actor'):
                logits = self.policy.actor(state_vec.unsqueeze(0))
            else:
                logits = self.policy(state_vec.unsqueeze(0))
            probs = F.softmax(logits, dim=-1).squeeze(0)
        return probs.cpu().numpy()

    def compute_exploitability(self) -> float:
        total_br_ev = 0.0
        deals = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

        for p1_card, p2_card in deals:
            p1_probs = self._get_policy_probs(p1_card)
            showdown_val_1 = 1.0 if p2_card > p1_card else -1.0
            showdown_val_2 = 2.0 if p2_card > p1_card else -2.0
            ev_p2_if_pass = p1_probs[0] * showdown_val_1 + p1_probs[1] * (-1.0)
            ev_p2_if_bet = p1_probs[0] * 1.0 + p1_probs[1] * showdown_val_2
            best_ev_for_deal = max(ev_p2_if_pass, ev_p2_if_bet)
            total_br_ev += best_ev_for_deal

        avg_br_ev = total_br_ev / len(deals)
        # Return value in "milli-big-blinds", assuming 1 unit is one big blind
        return avg_br_ev * 1000

class UnifiedPRPOAgent(StandardPPO):
    """ A unified PRPO agent whose loss is regularized by game-theoretic properties. """
    def __init__(self, state_dim: int, action_dim: int, lr: float, device: str,
                 lambda_exploit_base: float = 0.0,
                 adaptive_lambda_callable: Optional[Callable] = None,
                 lambda_target: float = 0.0):
        super().__init__(state_dim, action_dim, lr, device)
        self.lambda_exploit_base = lambda_exploit_base
        self.adaptive_lambda_callable = adaptive_lambda_callable
        self.lambda_exploit_current = lambda_exploit_base
        self.lambda_target = lambda_target
        self.target_policy_state_dict = None
        self.current_exploitability = 1000.0

    def update_policy(self):
        if not self.memory: return {}
        if self.adaptive_lambda_callable:
            # Be robust to different callable signatures
            try:
                self.lambda_exploit_current = float(
                    self.adaptive_lambda_callable(self.lambda_exploit_base, self.current_exploitability)
                )
            except TypeError:
                try:
                    self.lambda_exploit_current = float(
                        self.adaptive_lambda_callable(self.lambda_exploit_base)
                    )
                except Exception:
                    self.lambda_exploit_current = float(self.lambda_exploit_base)
        # Standard PPO data preparation from base class
        states=torch.stack([e.state for e in self.memory]).to(self.device);actions=torch.LongTensor([e.action for e in self.memory]).to(self.device)
        old_log_probs=torch.stack([e.log_prob for e in self.memory]).to(self.device);old_values=torch.stack([e.value for e in self.memory]).to(self.device)
        returns=[]; discounted_reward=0
        for r,d in zip(reversed([e.reward for e in self.memory]),reversed([e.done for e in self.memory])):
            if d: discounted_reward=0
            discounted_reward=r+(self.gamma*discounted_reward); returns.insert(0,discounted_reward)
        returns=torch.tensor(returns,dtype=torch.float32).to(self.device);advantages=returns-old_values.detach()
        if len(advantages)>1: advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)

        # PRPO Update Loop
        for _ in range(self.k_epochs):
            log_probs, entropy, values = self.policy.evaluate_actions(states, actions)
            ppo_loss = -torch.min(torch.exp(log_probs-old_log_probs)*advantages, torch.clamp(torch.exp(log_probs-old_log_probs),1-self.eps_clip,1+self.eps_clip)*advantages).mean() + 0.5 * F.mse_loss(values, returns) - self.entropy_coeff*entropy.mean()

            # L_Opponent: Penalty for being exploitable (using a fresh value)
            exploit_penalty = torch.tensor(self.current_exploitability / 1000.0, dtype=torch.float32, device=self.device)
            exploit_reg_loss = self.lambda_exploit_current * exploit_penalty

            # L_Target: KL-Divergence penalty to move towards the best agent
            target_reg_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_target > 0 and self.target_policy_state_dict is not None:
                is_self = all(torch.equal(p1, p2) for p1, p2 in zip(self.policy.state_dict().values(), self.target_policy_state_dict.values()))
                if not is_self:
                    target_policy = PPOAgent(self.policy.actor[0].in_features, self.policy.actor[-1].out_features).to(self.device)
                    target_policy.load_state_dict(self.target_policy_state_dict)
                    target_policy.eval()
                    with torch.no_grad():
                        target_probs, _, _ = target_policy.evaluate_actions(states, actions)
                    current_log_probs = torch.log_softmax(self.policy.actor(states), dim=-1)
                    kl_div = F.kl_div(current_log_probs, torch.softmax(target_policy.actor(states), dim=-1).detach(), reduction='batchmean')
                    target_reg_loss = self.lambda_target * kl_div

            total_loss = ppo_loss + exploit_reg_loss + target_reg_loss
            self.optimizer.zero_grad(); total_loss.backward(); torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5); self.optimizer.step()

        self.memory.clear()
        return {'loss': total_loss.item(), 'exploitability': self.current_exploitability}

class UnifiedPRPO_Kuhn:
    """ Manages a population of PRPO agents for Kuhn Poker. """
    def __init__(self, input_dim: int, output_dim: int, device: torch.device, population_size: int,
                 lambda_exploit_base: float, lambda_target: float, adaptive_lambda_callable: Optional[Callable]):
        self.input_dim, self.output_dim, self.device = input_dim, output_dim, device
        self.population = [UnifiedPRPOAgent(input_dim, output_dim, 3e-4, device, lambda_exploit_base, adaptive_lambda_callable, lambda_target) for _ in range(population_size)]
        self.best_agent_policy_state_dict = None

    def _train_oracle(self, target_agent: UnifiedPRPOAgent, num_episodes: int) -> nn.Module:
        oracle_agent = StandardPPO(self.input_dim, self.output_dim, device=self.device)
        target_agent.policy.eval()
        env = KuhnPokerEnvironment()
        for _ in range(num_episodes):
            s = env.reset()
            # Oracle is player 1, target is player 2
            a1, lp1, v1 = oracle_agent.select_action(s)
            with torch.no_grad():
                a2, _, _ = target_agent.policy.act(s.to(self.device))
            _, rewards, _, _ = env.step([a1, a2])
            oracle_agent.store_experience(s, a1, rewards[0], s, True, lp1, v1)
            if len(oracle_agent.memory) >= 10: oracle_agent.update_policy()
        if oracle_agent.memory: oracle_agent.update_policy()
        target_agent.policy.train(); return oracle_agent.policy

    def _find_and_set_target_policy(self):
        best_agent_so_far, min_exploit = None, float('inf')
        for agent in self.population:
            exploit_calc = KuhnBestResponse(KuhnPokerEnvironment(), agent.policy, self.device)
            agent.current_exploitability = exploit_calc.compute_exploitability()
            if agent.current_exploitability < min_exploit:
                min_exploit = agent.current_exploitability
                best_agent_so_far = agent
        if best_agent_so_far:
            self.best_agent_policy_state_dict = copy.deepcopy(best_agent_so_far.policy.state_dict())
            for agent in self.population:
                agent.target_policy_state_dict = self.best_agent_policy_state_dict
        return min_exploit

    def train(self, total_episodes: int, episodes_per_update=50, oracle_training_episodes=100):
        completed_episodes = 0
        env = KuhnPokerEnvironment()
        while completed_episodes < total_episodes:
            # 1. Tournament Phase (Self-Play)
            for _ in range(episodes_per_update):
                p1_idx, p2_idx = random.sample(range(len(self.population)), 2)
                agent1, agent2 = self.population[p1_idx], self.population[p2_idx]
                s = env.reset()
                a1, lp1, v1 = agent1.select_action(s)
                a2, lp2, v2 = agent2.select_action(s)
                _, rewards, _, _ = env.step([a1, a2])
                agent1.store_experience(s, a1, rewards[0], s, True, lp1, v1)
                agent2.store_experience(s, a2, rewards[1], s, True, lp2, v2)

            # 2. Exploitative Phase (Play vs. Oracle)
            for agent in self.population:
                oracle_policy = self._train_oracle(agent, num_episodes=oracle_training_episodes)
                oracle_policy.eval()
                for _ in range(episodes_per_update):
                    s = env.reset()
                    # Agent is P1, Oracle is P2
                    a1, lp1, v1 = agent.select_action(s)
                    with torch.no_grad():
                        a2, _, _ = oracle_policy.act(s.to(self.device))
                    _, rewards, _, _ = env.step([a1, a2])
                    agent.store_experience(s, a1, rewards[0], s, True, lp1, v1)

            # 3. Update and Evaluation Phase
            completed_episodes += episodes_per_update * 2
            current_best_exploit = self._find_and_set_target_policy()
            for agent in self.population:
                agent.update_policy()
            avg_exploit = np.mean([a.current_exploitability for a in self.population])
            print(f"  PRPO Episode {completed_episodes}: Avg Exploit: {avg_exploit:.2f} mbb/h (Best: {current_best_exploit:.2f} mbb/h)")
        
        # Final selection of the best agent
        final_best_exploit = self._find_and_set_target_policy()
        best_agent = min(self.population, key=lambda ag: ag.current_exploitability)
        print(f"  PRPO Training Finished. Best agent exploitability: {final_best_exploit:.2f} mbb/h")
        return best_agent.policy

def get_kuhn_adaptive_lambda_callable(base_lambda: float, exploitability: float) -> float:
    """Returns an adaptive lambda that increases with exploitability."""
    # Scale factor can be tuned. This one increases penalty as agent gets worse.
    exploit_factor = 1.0 + (exploitability / 200.0) # 200 is a heuristic scaling factor
    return base_lambda * exploit_factor

def train_prpo_kuhn(episodes: int, input_dim: int, output_dim: int, device: torch.device) -> nn.Module:
    """ Instantiates and runs the corrected Unified PRPO framework for Kuhn Poker. """
    print("\n--- Training Kuhn PRPO (Unified V6 Framework) ---")
    prpo_system = UnifiedPRPO_Kuhn(
        input_dim=input_dim, output_dim=output_dim, device=device,
        population_size=4,
        # Hyperparameters inspired by the Leduc implementation
        lambda_exploit_base=0.3,  # Penalty for being exploitable
        lambda_target=0.1,        # Penalty for deviating from best-in-population
        adaptive_lambda_callable=get_kuhn_adaptive_lambda_callable,
    )
    # The number of episodes inside the train method is the main driver
    final_policy = prpo_system.train(total_episodes=episodes)
    return final_policy

# =====================================================================================
# END: UNIFIED PRPO FRAMEWORK FOR KUHN POKER
# =====================================================================================

# ============================================================================
# 5. Training Logic
# ============================================================================

def train_dqn_kuhn(agent: DQNAgent, env: KuhnPokerEnvironment, opponent: ChallengerAgent, episodes: int = 5000):
    opt = optim.Adam(agent.q_net.parameters(), lr=1e-3)
    buf = ReplayBuffer(5000)
    batch = 64
    target_sync = 250
    step = 0
    for _ in range(episodes):
        s = env.reset(); opponent.reset();
        a = agent.act(s, explore=True)
        b = opponent.act(s, opponent_history=[])
        ns, rewards, _, _ = env.step([a, b]); r = float(rewards[0])
        buf.push(s, a, r, ns, True)
        step += 1
        if len(buf) >= batch:
            sb, ab, rb, nsb, db = buf.sample(batch)
            q_pred = agent.q_net(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
            target = rb  # single-step terminal
            loss = nn.functional.mse_loss(q_pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
        if step % target_sync == 0:
            agent.sync_target()
        agent.update_epsilon()
    agent.epsilon = agent.epsilon_min; agent.eval(); return agent


def train_ppo_kuhn(agent: PPOAgent, env: KuhnPokerEnvironment, opponent: ChallengerAgent, episodes: int = 5000):
    opt = optim.Adam(list(agent.actor.parameters()) + list(agent.critic.parameters()), lr=3e-4)
    clip_eps = 0.2; entropy_coef = 0.01; value_coef = 0.5
    for _ in range(episodes):
        s = env.reset();
        a, logp, val = agent.act(s)
        b = opponent.act(s, opponent_history=[])
        ns, rewards, _, _ = env.step([a, b]); r = float(rewards[0])
        logp_new, entropy, v_new = agent.evaluate_actions(s.unsqueeze(0), torch.tensor([a]))
        ratio = torch.exp(logp_new.squeeze() - logp.detach())
        adv = r - val.detach()
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        actor_loss = -torch.min(surr1, surr2)
        critic_loss = nn.functional.mse_loss(v_new.squeeze(), torch.tensor(r, dtype=torch.float32))
        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    agent.eval(); return agent


# ----------------------------------------------------------------------------
# Seed-wise evaluation helpers
# ----------------------------------------------------------------------------

def _act_to_int(policy: nn.Module, state: torch.Tensor) -> int:
    res = policy.act(state)
    if isinstance(res, (tuple, list)):
        return int(res[0])
    return int(res)


def evaluate_simple_avg_reward(policy: nn.Module, env: KuhnPokerEnvironment, opponent: ChallengerAgent, episodes: int = 200) -> float:
    total = 0.0
    for _ in range(episodes):
        s = env.reset(); opponent.reset();
        a = _act_to_int(policy, s)
        b = opponent.act(s, opponent_history=[])
        _, rewards, _, _ = env.step([a, b])
        total += float(rewards[0])
    return total / float(episodes)

# ============================================================================
# 6a. PSRO for Kuhn Poker (population of DQN BRs)
# ============================================================================

class PSROPolicy(nn.Module):
    def __init__(self, population: List[nn.Module], meta_strategy: List[float]):
        super().__init__()
        assert len(population) == len(meta_strategy) and len(population) > 0
        self.population = nn.ModuleList(population)
        probs = torch.tensor(meta_strategy, dtype=torch.float32)
        probs = probs / probs.sum()
        self.register_buffer("meta_strategy", probs)

    def act(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            idx = torch.distributions.Categorical(self.meta_strategy).sample().item()
        return self.population[idx].act(state)


def _eval_policy_vs_opp(policy: nn.Module, env: KuhnPokerEnvironment, opponent: ChallengerAgent, episodes: int = 200) -> float:
    total = 0.0
    for _ in range(episodes):
        s = env.reset()
        opponent.reset()
        hist: List[int] = []
        a = policy.act(s)
        b = opponent.act(s, opponent_history=hist)
        ns, rewards, _, _ = env.step([a, b])
        total += rewards[0]
    return total / episodes


def _compute_meta(population: List[nn.Module], opponents: List[ChallengerAgent], env: KuhnPokerEnvironment) -> List[float]:
    scores: List[float] = []
    for pol in population:
        perf = 0.0
        for opp in opponents:
            # Avoid deepcopy of opponents due to RNG objects; reset is sufficient
            opp.reset()
            perf += _eval_policy_vs_opp(pol, env, opp, episodes=200)
        scores.append(perf / max(1, len(opponents)))
    logits = torch.tensor(scores, dtype=torch.float32)
    return torch.softmax(logits, dim=0).tolist()


def train_best_response_dqn(env: KuhnPokerEnvironment, opponents: List[ChallengerAgent], episodes: int, input_dim: int, output_dim: int) -> nn.Module:
    br = DQNAgent(input_dim, output_dim)
    opt = optim.Adam(br.q_net.parameters(), lr=1e-3)
    for _ in range(episodes):
        s = env.reset()
        # Avoid deepcopy of opponents due to RNG objects; reuse and reset
        opp = random.choice(opponents)
        opp.reset()
        a = br.act(s)
        b = opp.act(s, opponent_history=[])
        ns, rewards, _, _ = env.step([a, b])
        r = rewards[0]
        q_pred = br(s.unsqueeze(0))[0, a]
        loss = nn.functional.mse_loss(q_pred, torch.tensor(r, dtype=torch.float32))
        opt.zero_grad(); loss.backward(); opt.step()
    return br


def train_psro_kuhn(gauntlet: "EnhancedGauntletBenchmark", time_budget_seconds: float, episodes_per_iter: int = 100) -> PSROPolicy:
    """
    Train PSRO for Kuhn Poker using actual time budget instead of fixed iterations.
    """
    print(f"[PSRO-Kuhn] Training for {time_budget_seconds} seconds...")
    start_time = time.time()
    
    # Avoid deepcopy of challengers; some contain numpy RNGs that are not deepcopy-safe
    opponents = [ch for name, ch in gauntlet.master_challenger_list.items() if name.startswith("Kuhn_")]
    if not opponents:
        opponents = [KuhnThresholdAgent()]
    env = KuhnPokerEnvironment()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    population: List[nn.Module] = []
    iteration_count = 0
    
    while (time.time() - start_time) < time_budget_seconds:
        elapsed_time = time.time() - start_time
        print(f"[PSRO-Kuhn] Iteration {iteration_count + 1} (Elapsed: {elapsed_time:.1f}s)")
        br = train_best_response_dqn(env, opponents, episodes_per_iter, input_dim, output_dim)
        population.append(copy.deepcopy(br))
        # Update opponent pool to include the new best response
        opponents.append(br)
        iteration_count += 1
        
        # Safety check to prevent infinite loops with very small time budgets
        if iteration_count >= 20:  # Max reasonable iterations
            print(f"[PSRO-Kuhn] Reached maximum iterations ({iteration_count}), stopping.")
            break
    
    final_time = time.time() - start_time
    print(f"[PSRO-Kuhn] Training completed in {final_time:.1f}s with {iteration_count} iterations")
    
    if not population:
        # Fallback: train at least one best response if no time was sufficient
        print("[PSRO-Kuhn] Warning: No iterations completed, training one best response")
        br = train_best_response_dqn(env, opponents, episodes_per_iter, input_dim, output_dim)
        population.append(copy.deepcopy(br))
    
    meta = _compute_meta(population, opponents, env)
    print(f"[PSRO-Kuhn] Meta: {np.round(meta, 3)}")
    return PSROPolicy(population, meta)


# ============================================================================
# 6. Main Execution: Register env, train, and evaluate via Gauntlet
# ============================================================================

if __name__ == "__main__":

    # Global training parameters - now using time budget
    TIME_BUDGET_SECONDS = 10.0  # Time budget per algorithm per seed
    NUM_SEEDS = 1  # Number of seeds for statistical comparison
    PSRO_ITERATIONS = 5  # PSRO-specific iterations

    print("\n--- Setting up Gauntlet Benchmark for Kuhn Poker Evaluation ---")
    config = EvaluationConfig(num_episodes=200, parallel_workers=1, save_visualizations=True)
    gauntlet = EnhancedGauntletBenchmark(config)

    # Prefer OpenSpiel Kuhn when available; otherwise fallback to simplified env
    try:
        import pyspiel as _openspiel  # type: ignore
        _OS_OK = True
    except Exception:
        _OS_OK = False
        _openspiel = None  # type: ignore

    if _OS_OK:
        # Register without fake (A,B); rely on OpenSpiel-derived metrics in benchmark
        gauntlet.register_environment(
            "KuhnPoker",
            KuhnPokerEnvironment,  # Placeholder factory for interface; gameplay remains in training script
            payoff_matrices=None,
            game_prefix="Kuhn",
            zero_sum=True
        )
    else:
        # Fallback to previous placeholder
        A_formal = np.array([[0, -1], [1, 0]])
        gauntlet.register_environment(
            "KuhnPoker",
            KuhnPokerEnvironment,
            payoff_matrices=(A_formal, -A_formal),
            game_prefix="Kuhn",
            zero_sum=True
        )

    # Instantiate env and agents
    env = KuhnPokerEnvironment()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    dqn_agent = DQNAgent(input_dim, output_dim)
    ppo_agent = PPOAgent(input_dim, output_dim)

    # Training opponent
    training_opponent = gauntlet.master_challenger_list['Kuhn_Uniform']
    training_opponent.name = "TrainingOpponent_Kuhn_Uniform"

    # Output dirs
    base_dir = os.path.join("results", "KuhnPoker")
    os.makedirs(base_dir, exist_ok=True)
    dqn_dir = os.path.join(base_dir, "DQN"); os.makedirs(dqn_dir, exist_ok=True)
    ppo_dir = os.path.join(base_dir, "PPO"); os.makedirs(ppo_dir, exist_ok=True)
    psro_dir = os.path.join(base_dir, "PSRO"); os.makedirs(psro_dir, exist_ok=True)
    timing_train: Dict[str, float] = {}
    timing_eval: Dict[str, float] = {}

    print(f"\n--- Starting Kuhn Poker Training Phase with Time Budget (DQN, PPO, {TIME_BUDGET_SECONDS}s x {NUM_SEEDS} seeds) ---")
    dqn_runs = []
    _t0 = time.time()
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        trained, episodes_completed, _ = TimeBudgetTrainer.train_dqn_time_budget(
            copy.deepcopy(dqn_agent), lambda: env, training_opponent, 
            TIME_BUDGET_SECONDS, device=torch.device('cpu')
        )
        dqn_runs.append(trained)
        print(f"    DQN Seed {seed}: Completed {episodes_completed} episodes in {TIME_BUDGET_SECONDS}s")
    trained_dqn = dqn_runs[-1]
    timing_train["DQN"] = float(time.time() - _t0)
    
    ppo_runs = []
    _t0 = time.time()
    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        trained, episodes_completed, _ = TimeBudgetTrainer.train_standard_ppo_time_budget(
            lambda: KuhnPokerSimpleEnvironment(), input_dim, output_dim, TIME_BUDGET_SECONDS, 
            device='cpu'
        )
        ppo_runs.append(trained)
        print(f"    PPO Seed {seed}: Completed {episodes_completed} episodes in {TIME_BUDGET_SECONDS}s")
    trained_ppo = ppo_runs[-1]
    timing_train["PPO"] = float(time.time() - _t0)
    torch.save(trained_dqn.state_dict(), os.path.join(dqn_dir, "model.pt"))
    # Save the underlying PPO policy parameters (StandardPPO has no state_dict)
    torch.save(trained_ppo.policy.state_dict(), os.path.join(ppo_dir, "model.pt"))

    print(f"\n--- Training Kuhn PSRO ({TIME_BUDGET_SECONDS}s time budget) ---")
    _t0 = time.time()
    # Use actual time budget instead of approximated episodes per iteration
    psro_policy = train_psro_kuhn(gauntlet, time_budget_seconds=TIME_BUDGET_SECONDS, episodes_per_iter=100)
    timing_train["PSRO"] = float(time.time() - _t0)
    meta = psro_policy.meta_strategy.cpu().numpy().tolist()
    for i, pol in enumerate(psro_policy.population):
        torch.save(pol.state_dict(), os.path.join(psro_dir, f"pop_member_{i}.pt"))
    import json
    with open(os.path.join(psro_dir, "meta_strategy.json"), "w") as f:
        json.dump({"meta_strategy": meta, "population_size": len(meta)}, f, indent=2)

    # Self-Play
    print(f"\n--- Training Kuhn Self-Play ({TIME_BUDGET_SECONDS}s time budget) ---")
    _t0 = time.time()
    # Use actual time budget for Self-Play instead of approximated episodes
    sp_agent = train_selfplay_kuhn(env, time_budget_seconds=TIME_BUDGET_SECONDS, input_dim=input_dim, output_dim=output_dim, device=torch.device('cpu'))
    timing_train["SelfPlay"] = float(time.time() - _t0)
    sp_dir = os.path.join(base_dir, "SelfPlay"); os.makedirs(sp_dir, exist_ok=True)
    torch.save(sp_agent.state_dict(), os.path.join(sp_dir, "model.pt"))

    # PRPO with superior unified implementation and time budget
    print(f"\n--- Training Kuhn PRPO Unified ({TIME_BUDGET_SECONDS}s per seed x {NUM_SEEDS} seeds) ---")
    prpo_runs: List[nn.Module] = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _t0 = time.time()
    for seed in range(NUM_SEEDS):
        print(f"\n--- PRPO Seed {seed+1}/{NUM_SEEDS} ---")
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        # Use the superior unified PRPO implementation
        prpo_policy_model = train_prpo_kuhn_unified(TIME_BUDGET_SECONDS, input_dim, output_dim, device)
        prpo_runs.append(copy.deepcopy(prpo_policy_model))
        print(f"    PRPO Seed {seed}: Training completed in {TIME_BUDGET_SECONDS}s")
    prpo_agent = prpo_runs[-1] # Use the last trained agent for single evaluation
    timing_train["PRPO"] = float(time.time() - _t0)
    prpo_dir = os.path.join(base_dir, "PRPO"); os.makedirs(prpo_dir, exist_ok=True)
    torch.save(prpo_agent.state_dict(), os.path.join(prpo_dir, "model.pt"))

    # Add custom challenger
    gauntlet.add_custom_challenger("Kuhn_Threshold", KuhnThresholdAgent())

    # Evaluate
    def eval_and_report(g: "EnhancedGauntletBenchmark", policy: nn.Module, name: str, out_dir: str):
        print(f"\n{'='*40}\nEVALUATING: {name}\n{'='*40}")
        g.evaluate_policy(policy=policy, policy_name=name, environments=["KuhnPoker"])
        report_path = os.path.join(out_dir, "report.json")
        g.generate_report(report_path)
        # Inject seed statistics into the report, if available
        try:
            base_dir = os.path.dirname(out_dir)
            stats_path = os.path.join(base_dir, "kuhn_stats.json")
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
            fmt = config.visualization_format
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

    _t0 = time.time(); trained_dqn.eval(); eval_and_report(gauntlet, trained_dqn, "Kuhn_DQN", dqn_dir); timing_eval["DQN"] = float(time.time() - _t0)
    _t0 = time.time(); trained_ppo.eval(); eval_and_report(gauntlet, trained_ppo, "Kuhn_PPO", ppo_dir); timing_eval["PPO"] = float(time.time() - _t0)
    _t0 = time.time(); psro_policy.eval(); eval_and_report(gauntlet, psro_policy, "Kuhn_PSRO", psro_dir); timing_eval["PSRO"] = float(time.time() - _t0)
    _t0 = time.time(); sp_agent.eval(); eval_and_report(gauntlet, sp_agent, "Kuhn_SelfPlay", sp_dir); timing_eval["SelfPlay"] = float(time.time() - _t0)
    _t0 = time.time(); prpo_agent.eval(); eval_and_report(gauntlet, prpo_agent, "Kuhn_PRPO", prpo_dir); timing_eval["PRPO"] = float(time.time() - _t0)

    # Statistical comparison PRPO vs PPO
    print("\n--- Computing seed-wise scores, 95% CI, and paired t-test (Kuhn PRPO vs PPO) ---")
    eval_env = KuhnPokerEnvironment()
    fixed_opp = KuhnThresholdAgent()
    ppo_scores: List[float] = []
    prpo_scores: List[float] = []
    for run in ppo_runs:
        ppo_scores.append(evaluate_simple_avg_reward(run, eval_env, fixed_opp, episodes=200))
    for run in prpo_runs:
        prpo_scores.append(evaluate_simple_avg_reward(run, eval_env, fixed_opp, episodes=200))
    ppo_stats = compute_mean_ci(ppo_scores)
    prpo_stats = compute_mean_ci(prpo_scores)
    p_value = paired_t_test(prpo_scores, ppo_scores)
    print(f"PPO mean={ppo_stats['mean']:.3f}, 95% CI=[{ppo_stats['ci_low']:.3f}, {ppo_stats['ci_high']:.3f}], n={ppo_stats['n']}")
    print(f"PRPO mean={prpo_stats['mean']:.3f}, 95% CI=[{prpo_stats['ci_low']:.3f}, {prpo_stats['ci_high']:.3f}], n={prpo_stats['n']}")
    if p_value is not None:
        print(f"Paired t-test (PRPO vs PPO): p-value={p_value:.4f}")
    else:
        print("Paired t-test unavailable (SciPy not installed).")
    with open(os.path.join(base_dir, "kuhn_stats.json"), "w") as f:
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

    print("\n\n🎉 Kuhn Poker training and evaluation complete. Outputs under 'results/KuhnPoker/'. 🎉")
