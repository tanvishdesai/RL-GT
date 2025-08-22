%%writefile matchingpennies_training_and_evaluation.py

# mp
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
from gauntlet_benchmark import ChallengerAgent, EnhancedGauntletBenchmark, EvaluationConfig

# Gymnasium/Gym imports for environment compatibility
try:
    from gymnasium.spaces import Box, Discrete, Space
    from gymnasium import Env as Environment
except ImportError:
    try:
        from gym.spaces import Box, Discrete, Space
        from gym import Env as Environment
    except ImportError:
        # Fallback: create minimal environment classes
        class Environment:
            pass
        
        class Space:
            def __init__(self, shape=None, n=None):
                self.shape = shape
                self.n = n
        
        class Box(Space):
            def __init__(self, low, high, shape, dtype):
                super().__init__(shape=shape)
                self.low = low
                self.high = high
                self.dtype = dtype
        
        class Discrete(Space):
            def __init__(self, n):
                super().__init__(n=n)

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
# 2. Matching Pennies Specific Functions for Unified PRPO
# ============================================================================

class MPEnvironment:
    """Matching Pennies game environment for use with unified PRPO."""
    def __init__(self):
        # Payoff matrix for Player 1 (row player)
        # Actions: 0 = Heads, 1 = Tails  
        # P1 wins if match, P2 wins if mismatch
        self.payoff_matrix = np.array([[1, -1], [-1, 1]])
        self.state_dim = 2  # State is the opponent's last action (Heads/Tails)
        self.action_dim = 2  # Actions are Heads or Tails
        self.nash_equilibrium = np.array([0.5, 0.5])
        self.episode_length = 10  # Number of steps per episode
        self.reset()

    @property
    def observation_space(self):
        """Return observation space compatible with gym interface."""
        return type('MockSpace', (), {
            'shape': (self.state_dim,),
            'n': self.state_dim
        })()

    @property
    def action_space(self):
        """Return action space compatible with gym interface."""
        from gym.spaces import Discrete
        return Discrete(self.action_dim)

    def reset(self):
        # Initialize with a random opponent action
        self.last_opponent_action = random.randint(0, self.action_dim - 1)
        return self._get_state()

    def _get_state(self):
        # State is a one-hot vector of the opponent's last action
        state = np.zeros(self.state_dim)
        state[self.last_opponent_action] = 1.0
        return torch.from_numpy(state).float()

    def step(self, actions):
        # Handle both list format [p1_action, p2_action] and tuple format (p1_action, p2_action)
        if isinstance(actions, (list, tuple)) and len(actions) == 2:
            p1_action, p2_action = actions[0], actions[1]
        else:
            # Fallback for single action (assume it's p1_action and generate random p2_action)
            p1_action = actions
            p2_action = random.randint(0, 1)
        
        # Get rewards from the payoff matrix
        p1_reward = self.payoff_matrix[p1_action, p2_action]
        p2_reward = -p1_reward  # Zero-sum game
        # The new state is determined by the opponent's current action
        self.last_opponent_action = p2_action
        # The game is stateless and ends after one turn for RL purposes
        done = True
        return self._get_state(), [p1_reward, p2_reward], done, {}

    def get_legal_actions(self, player_id: int = 0) -> List[int]:
        """Return all legal actions for Matching Pennies (all actions are always legal)."""
        return list(range(self.action_dim))

def get_mp_nash_policy_callable(policy_probs_batch: torch.Tensor) -> torch.Tensor:
    """Returns the uniform Nash equilibrium distribution for Matching Pennies."""
    return torch.full_like(policy_probs_batch, 0.5)

def calculate_mp_exploitability_callable(policy: UnifiedActorCritic) -> float:
    """Calculates exploitability against hard-coded Matching Pennies bots."""
    policy.eval()
    device = next(policy.parameters()).device
    # State is constant in MP, so we can use a dummy state
    state = torch.FloatTensor(np.zeros(2)).unsqueeze(0).to(device)
    state[0, random.randint(0, 1)] = 1.0
    
    with torch.no_grad():
        policy_probs, _ = policy(state)
        p = policy_probs.squeeze().cpu().numpy()  # p = [p_heads, p_tails]

    # Exploitability is the best possible reward an opponent can get.
    # Opponent plays Heads (0): their reward is p[1] - p[0]
    # Opponent plays Tails (1): their reward is p[0] - p[1]
    max_exploit = max(p[1] - p[0], p[0] - p[1])
    policy.train()
    return max(0.0, max_exploit)

def get_mp_exploiter_opponents() -> List[Callable]:
    """Returns a list of functions, each representing a simple exploiter bot."""
    return [
        lambda: 0,  # Always Heads
        lambda: 1,  # Always Tails
        lambda: np.random.choice([0, 1], p=[0.9, 0.1]),  # Biased Heads
        lambda: np.random.choice([0, 1], p=[0.1, 0.9]),  # Biased Tails
    ]

def train_prpo_mp_unified(time_budget_seconds: float, input_dim: int, output_dim: int, 
                         device: torch.device, lambda_nash: float = 1.0, 
                         lambda_exploit: float = 0.5) -> nn.Module:
    """
    Superior PRPO training function using the unified framework with time budget.
    """
    print(f"Training PRPO (Unified Framework) for Matching Pennies for {time_budget_seconds} seconds...")
    
    # The manager needs a way to create new environments
    env_factory = lambda: MPEnvironment()

    prpo_system = UnifiedPRPO(
        state_dim=input_dim, 
        action_dim=output_dim, 
        lr=1e-4, 
        device=str(device),
        population_size=4,
        lambda_nash=lambda_nash,
        nash_target_callable=get_mp_nash_policy_callable,
        lambda_exploit=lambda_exploit,
        exploitability_calculator=calculate_mp_exploitability_callable,
        exploiter_opponents=get_mp_exploiter_opponents()
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
    
    print("Training finished for PRPO (Unified Framework) - Matching Pennies.")
    return best_policy

# ============================================================================
# 3. Matching Pennies Environment Implementation (Original)
# ============================================================================

class MatchingPenniesEnvironment(Environment):
    """
    An environment for the zero-sum game Matching Pennies.
    - Actions: 0 for Heads, 1 for Tails.
    - Player 1 (the policy being evaluated) is the "Matcher". They win if the pennies match.
    - Player 2 (the challenger) is the "Mismatcher". They win if the pennies do not match.
    - Payoffs are (+1, -1) for a win/loss.
    """
    def __init__(self):
        # Observation: [own_last_action, opponent_last_action]
        self._observation_space = Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self._action_space = Discrete(2) # 0: Heads, 1: Tails
        self.state = None
        self.episode_length = 10  # Number of steps per episode

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    @property
    def action_space(self) -> Space:
        return self._action_space

    def reset(self) -> torch.Tensor:
        # Initial state represents no prior actions
        self.state = torch.zeros(2, dtype=torch.float32)
        return self.state

    def step(self, actions: List[int]) -> Tuple[torch.Tensor, List[float], bool, Dict]:
        action1, action2 = actions[0], actions[1]

        # Payoff logic: Player 1 wins if actions are the same (match)
        if action1 == action2:
            rewards = [1.0, -1.0]
        else:
            rewards = [-1.0, 1.0]

        self.state = torch.tensor([action1, action2], dtype=torch.float32)
        # The game is "done" after each step in this simple representation
        return self.state, rewards, True, {}

    def get_legal_actions(self, player_id: int = 0) -> List[int]:
        """Return all legal actions for Matching Pennies (all actions are always legal)."""
        return [0, 1]  # Heads, Tails

# ============================================================================
# 3. Algorithm Implementations for Matching Pennies
# ============================================================================

# --- Learning Agents (DQN and PPO can be reused) ---

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.storage: List[Tuple[torch.Tensor, int, float, torch.Tensor, bool]] = []
        self.index = 0

    def __len__(self) -> int:
        return len(self.storage)

    def push(self, s: torch.Tensor, a: int, r: float, ns: torch.Tensor, d: bool) -> None:
        item = (s.detach().cpu(), int(a), float(r), ns.detach().cpu(), bool(d))
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.index] = item
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.storage, batch_size)
        s, a, r, ns, d = zip(*batch)
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
        self.target_q_net = copy.deepcopy(self.q_net)
        self.gamma = float(gamma)
        self.num_actions = int(output_dim)
        self.epsilon: float = 1.0
        self.epsilon_min: float = 0.05
        self.epsilon_decay: float = 0.9995

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_net(state)

    def sync_target(self) -> None:
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def act(self, state: torch.Tensor, explore: bool = True) -> int:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if explore and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q = self.q_net(state)
            return int(torch.argmax(q, dim=-1).item())

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
    def __init__(self, input_dim, output_dim):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim))
        self.critic = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    def act(self, state: torch.Tensor):
        if len(state.shape) == 1: state = state.unsqueeze(0)
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


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float):
    T = rewards.size(0)
    advantages = torch.zeros(T, dtype=torch.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_value * (1.0 - float(dones[t])) - values[t]
        last_gae = delta + gamma * lam * (1.0 - float(dones[t])) * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:T]
    return advantages, returns


def train_dqn_pennies(agent: DQNAgent, env, opponent, episodes: int = 5000, device: torch.device = torch.device('cpu')) -> DQNAgent:
    agent.to(device)
    optimizer = optim.Adam(agent.q_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(5000)
    batch_size = 64
    target_update_interval = 250
    step = 0
    for _ in range(episodes):
        s = env.reset(); opponent.reset(); hist: List[int] = []
        # single-step
        a = agent.act(s, explore=True)
        b = opponent.act(s, opponent_history=hist)
        ns, rewards, done, _ = env.step([a, b])
        r = float(rewards[0])
        buffer.push(s, a, r, ns, bool(done))
        s = ns; step += 1
        if len(buffer) >= batch_size:
            sb, ab, rb, nsb, db = buffer.sample(batch_size)
            sb = sb.to(device); nsb = nsb.to(device)
            ab = ab.to(device); rb = rb.to(device); db = db.to(device)
            q_pred = agent.q_net(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next = agent.target_q_net(nsb).max(dim=1).values
                target = rb + agent.gamma * max_next * (~db)
            loss = nn.functional.mse_loss(q_pred, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % target_update_interval == 0:
            agent.sync_target()
        agent.update_epsilon()
    agent.epsilon = agent.epsilon_min; agent.eval(); return agent


def train_ppo_pennies(agent: PPOAgent, env, opponent, episodes: int = 5000, device: torch.device = torch.device('cpu')) -> PPOAgent:
    agent.to(device)
    optimizer = optim.Adam(list(agent.actor.parameters()) + list(agent.critic.parameters()), lr=3e-4)
    clip_eps = 0.2; entropy_coef = 0.01; value_coef = 0.5; gamma = 0.99; lam = 0.95
    update_epochs = 4; minibatch_size = 128
    rollout = RolloutBuffer()

    def flush_and_opt():
        if len(rollout.states) == 0:
            return
        states = torch.stack(rollout.states).to(device)
        actions = torch.tensor(rollout.actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rollout.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(rollout.dones, dtype=torch.bool, device=device)
        old_logp = torch.stack(rollout.log_probs).detach().to(device)
        values = torch.stack(rollout.values).detach().to(device)
        adv, rets = compute_gae(rewards, values, dones, gamma, lam)
        # Robust normalization for very small batch sizes (e.g., single-step rollouts)
        adv_std = adv.std(unbiased=False)
        adv = (adv - adv.mean()) / (adv_std + 1e-8)
        idxs = list(range(states.size(0)))
        for _ in range(update_epochs):
            random.shuffle(idxs)
            for i in range(0, len(idxs), minibatch_size):
                idx = idxs[i:i+minibatch_size]
                s, a, gae, ret, olp = states[idx], actions[idx], adv[idx], rets[idx], old_logp[idx]
                logp, entropy, v = agent.evaluate_actions(s, a)
                ratio = torch.exp(logp - olp)
                surr1 = ratio * gae
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(v, ret)
                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy.mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        rollout.clear()

    for _ in range(episodes):
        s = env.reset(); opponent.reset(); hist: List[int] = []
        a, logp, val = agent.act(s)
        b = opponent.act(s, opponent_history=hist)
        ns, rewards, done, _ = env.step([a, b]); r = float(rewards[0])
        rollout.states.append(s.detach().cpu()); rollout.actions.append(int(a))
        rollout.rewards.append(r); rollout.dones.append(bool(done))
        rollout.log_probs.append(logp.detach().cpu()); rollout.values.append(val.detach().cpu())
        flush_and_opt()
    agent.eval(); return agent

# --- Custom Challenger for Matching Pennies ---

class FrequencyCounterAgent(ChallengerAgent):
    """
    A custom challenger for Matching Pennies. As the "Mismatcher", it tries to
    predict the opponent's next move based on frequency and play the opposite.
    """
    def __init__(self, name="FrequencyCounter"):
        super().__init__(name, "medium")
        self.opponent_action_counts = np.zeros(2) # Counts for Heads, Tails

    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        if opponent_history:
            # Update counts from the most recent opponent action
            self.opponent_action_counts[opponent_history[-1]] += 1

        total_moves = np.sum(self.opponent_action_counts)
        if total_moves < 3: # Act randomly for the first few moves
            return random.randint(0, 1)

        # Predict opponent's most likely move
        predicted_opponent_move = np.argmax(self.opponent_action_counts)

        # As the Mismatcher, play the opposite action to win
        my_action = 1 - predicted_opponent_move
        return my_action

    def update(self, reward: float, observation: torch.Tensor, action: int):
        pass # Logic is handled in act()

    def reset(self):
        super().reset()
        self.opponent_action_counts = np.zeros(2)

    @property
    def compatible_action_space(self) -> Space:
        """Declares that this agent is designed for a 2-action space like Matching Pennies."""
        return Discrete(2)

# ============================================================================
# X. Self-Play Baseline and PRPO for Matching Pennies
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


def train_selfplay_pennies(env: MatchingPenniesEnvironment, time_budget_seconds: float, input_dim: int, output_dim: int, device: torch.device) -> PPOAgent:
    """
    Train Self-Play for Matching Pennies using actual time budget instead of fixed episodes.
    """
    print(f"[SelfPlay-Pennies] Training for {time_budget_seconds} seconds...")
    start_time = time.time()
    
    agent = PPOAgent(input_dim, output_dim).to(device)
    opponent = SelfPlayOpponent(output_dim, input_dim, device)
    opt = optim.Adam(agent.parameters(), lr=1e-3)
    episode_count = 0
    last_log_time = start_time
    
    while (time.time() - start_time) < time_budget_seconds:
        try:
            s = env.reset()
            opponent.reset()
            hist: List[int] = []
            for _ in range(1):
                # Use the log_prob already computed by agent.act instead of recomputing
                a, lp, val = agent.act(s)
                b = opponent.act(s, opponent_history=hist)
                ns, rewards, _, _ = env.step([a, b])
                r = rewards[0]
                
                # Simple advantage computation
                adv = torch.tensor(r, dtype=torch.float32) - val.detach()
                
                # Actor loss using existing log_prob
                actor_loss = -lp * adv
                
                # Critic loss
                critic_loss = nn.functional.mse_loss(val, torch.tensor(r, dtype=torch.float32))
                
                # Combined loss
                loss = actor_loss + critic_loss
                
                # Check for NaN/inf values
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss at episode {episode_count}, skipping update")
                    continue
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                s = ns
                
        except Exception as e:
            print(f"Error in self-play episode {episode_count}: {e}")
            continue
        
        episode_count += 1
        
        # Add agent snapshot to opponent pool occasionally
        if episode_count % 25 == 0: 
            opponent.add_snapshot(agent)
        
        current_time = time.time()
        # Log progress every 10 seconds
        if (current_time - last_log_time) >= 10.0:
            elapsed_time = current_time - start_time
            print(f"    [SelfPlay-Pennies] Time {elapsed_time:.1f}s: Episodes {episode_count}")
            last_log_time = current_time
    
    final_time = time.time() - start_time
    print(f"[SelfPlay-Pennies] Training completed in {final_time:.1f}s with {episode_count} episodes")
    return agent


def _mp_exploitability_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.detach(), dim=-1).squeeze()
    # best opponent plays opposite; exploitability ~ |p0 - p1|
    return torch.abs(probs[0] - probs[1])


Experience = namedtuple("Experience", ["state", "action", "reward", "done", "log_prob", "value"]) 


# ============================================================================
# X. Self-Play Baseline and PRPO for Matching Pennies
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


def train_selfplay_pennies(env: MatchingPenniesEnvironment, time_budget_seconds: float, input_dim: int, output_dim: int, device: torch.device) -> PPOAgent:
    """
    Train Self-Play for Matching Pennies using actual time budget instead of fixed episodes.
    """
    print(f"[SelfPlay-Pennies] Training for {time_budget_seconds} seconds...")
    start_time = time.time()
    
    agent = PPOAgent(input_dim, output_dim).to(device)
    opponent = SelfPlayOpponent(output_dim, input_dim, device)
    opt = optim.Adam(agent.parameters(), lr=1e-3)
    episode_count = 0
    last_log_time = start_time
    
    while (time.time() - start_time) < time_budget_seconds:
        try:
            s = env.reset()
            opponent.reset()
            hist: List[int] = []
            for _ in range(1):
                # Use the log_prob already computed by agent.act instead of recomputing
                a, lp, val = agent.act(s)
                b = opponent.act(s, opponent_history=hist)
                ns, rewards, _, _ = env.step([a, b])
                r = rewards[0]
                
                # Simple advantage computation
                adv = torch.tensor(r, dtype=torch.float32) - val.detach()
                
                # Actor loss using existing log_prob
                actor_loss = -lp * adv
                
                # Critic loss
                critic_loss = nn.functional.mse_loss(val, torch.tensor(r, dtype=torch.float32))
                
                # Combined loss
                loss = actor_loss + critic_loss
                
                # Check for NaN/inf values
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss at episode {episode_count}, skipping update")
                    continue
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                s = ns
                
        except Exception as e:
            print(f"Error in self-play episode {episode_count}: {e}")
            continue
        
        episode_count += 1
        
        # Add agent snapshot to opponent pool occasionally
        if episode_count % 25 == 0: 
            opponent.add_snapshot(agent)
        
        current_time = time.time()
        # Log progress every 10 seconds
        if (current_time - last_log_time) >= 10.0:
            elapsed_time = current_time - start_time
            print(f"    [SelfPlay-Pennies] Time {elapsed_time:.1f}s: Episodes {episode_count}")
            last_log_time = current_time
    
    final_time = time.time() - start_time
    print(f"[SelfPlay-Pennies] Training completed in {final_time:.1f}s with {episode_count} episodes")
    return agent


def _mp_exploitability_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.detach(), dim=-1).squeeze()
    # best opponent plays opposite; exploitability ~ |p0 - p1|
    return torch.abs(probs[0] - probs[1])


Experience = namedtuple("Experience", ["state", "action", "reward", "done", "log_prob", "value"])


class PRPOAgent(PPOAgent):
    """
    Corrected PRPO Agent that regularizes toward a fixed Nash Equilibrium.
    """
    def __init__(self, input_dim: int, output_dim: int, lambda_exploit_base: float, lambda_nash: float,
                 adaptive_lambda: Optional[Callable[[float, float], float]], device: torch.device):
        super().__init__(input_dim, output_dim)
        self.memory: List[Experience] = []
        self.gamma = 0.99
        self.k_epochs = 4
        self.eps_clip = 0.2
        self.entropy_coef = 0.01
        self.lambda_exploit_base = float(lambda_exploit_base)
        self.lambda_nash = float(lambda_nash)  # Regularize towards Nash Equilibrium
        self.adaptive_lambda = adaptive_lambda
        self.lambda_exploit_current = float(lambda_exploit_base)
        self.current_exploitability: float = 1.0
        self._device = device
        # Optimizer is created once for stable training
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=3e-4)

    def store(self, s, a, r, d, lp, v):
        self.memory.append(Experience(s.detach().cpu(), int(a), float(r), bool(d), lp.detach().cpu(), v.detach().cpu()))

    def clear(self):
        self.memory.clear()

    def update_policy(self) -> Dict[str, float]:
        if not self.memory:
            return {}
        if self.adaptive_lambda is not None:
            self.lambda_exploit_current = float(self.adaptive_lambda(self.lambda_exploit_base, self.current_exploitability))

        states = torch.stack([e.state for e in self.memory]).to(self._device)
        actions = torch.tensor([e.action for e in self.memory], dtype=torch.long, device=self._device)
        rewards = torch.tensor([e.reward for e in self.memory], dtype=torch.float32, device=self._device)
        old_logp = torch.stack([e.log_prob for e in self.memory]).to(self._device)
        old_values = torch.stack([e.value for e in self.memory]).to(self._device)

        # Compute advantages
        adv = rewards - old_values.detach()
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO update loop
        last_loss = 0.0
        for _ in range(self.k_epochs):
            logits = self.actor(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            values = self.critic(states).squeeze(-1)

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, rewards) # In MP, return = immediate reward
            ppo_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

            # --- CORRECTED REGULARIZATION ---
            exploit_reg = self.lambda_exploit_current * self.current_exploitability

            nash_reg = torch.tensor(0.0, device=self._device)
            if self.lambda_nash > 0.0:
                current_probs = F.softmax(logits, dim=-1)
                target_nash_dist = torch.full_like(current_probs, 0.5)
                current_log_probs = F.log_softmax(logits, dim=-1)
                kl_div = F.kl_div(current_log_probs, target_nash_dist, reduction='batchmean')
                nash_reg = self.lambda_nash * kl_div

            total_loss = ppo_loss + exploit_reg + nash_reg

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            last_loss = total_loss.item()

        self.clear()
        return {"loss": last_loss, "exploitability": self.current_exploitability}


class PenniesExploitabilityCalculator:
    def __init__(self, env: MatchingPenniesEnvironment, device: torch.device):
        self.env = env
        self.device = device
    def compute(self, policy: PPOAgent) -> float:
        s = self.env.reset()
        with torch.no_grad():
            logits = policy.actor(s.unsqueeze(0).to(self.device))
        return float(_mp_exploitability_from_logits(logits).item())


class PRPOManagerPennies:
    """
    Corrected PRPO Manager using a population and exploiter bots.
    """
    def __init__(self, env: MatchingPenniesEnvironment, input_dim: int, output_dim: int, device: torch.device,
                 population_size: int, lambda_exploit_base: float, lambda_nash: float,
                 adaptive_lambda: Optional[Callable[[float, float], float]]):
        self.env = env
        self.device = device
        self.population: List[PRPOAgent] = [
            PRPOAgent(input_dim, output_dim, lambda_exploit_base, lambda_nash, adaptive_lambda, device).to(device)
            for _ in range(population_size)
        ]
        self.exploit_calc = PenniesExploitabilityCalculator(env, device)
        # A pool of simple exploiter bots for robust training
        self.exploiter_bots: List[Callable[[], int]] = [
            lambda: 0,  # Always Heads
            lambda: 1,  # Always Tails
            lambda: int(np.random.choice([0, 1], p=[0.9, 0.1])), # Biased Heads
            lambda: int(np.random.choice([0, 1], p=[0.1, 0.9])), # Biased Tails
        ]

    def _update_exploitabilities(self) -> None:
        """Calculates and stores the current exploitability for each agent."""
        for agent in self.population:
            agent.eval()
            agent.current_exploitability = self.exploit_calc.compute(agent)
            agent.train()

    def train(self, total_episodes: int, games_per_iter: int = 10, exploit_ratio: float = 0.5) -> PRPOAgent:
        num_updates = max(1, total_episodes // (games_per_iter * len(self.population)))

        for _ in range(num_updates):
            # Phase 1 & 2: Data Collection (Tournament + Exploitative)
            for agent_idx, agent in enumerate(self.population):
                for _ in range(games_per_iter):
                    state = self.env.reset()
                    agent_action, logp, val = agent.act(state)

                    if random.random() < exploit_ratio:
                        # Exploitative game vs a hard-coded bot
                        bot = random.choice(self.exploiter_bots)
                        opp_action = bot()
                    else:
                        # Tournament game vs another agent in the population
                        opp_idx = random.choice([i for i in range(len(self.population)) if i != agent_idx])
                        opponent = self.population[opp_idx]
                        with torch.no_grad():
                            opp_action, _, _ = opponent.act(state)
                    
                    _, rewards, done, _ = self.env.step([agent_action, opp_action])
                    agent.store(state, agent_action, rewards[0], done, logp, val)

            # Phase 3: Update all agents in the population
            self._update_exploitabilities()
            for agent in self.population:
                agent.update_policy()

        # Final evaluation to find the best agent
        self._update_exploitabilities()
        best_agent = min(self.population, key=lambda ag: ag.current_exploitability)
        return best_agent


def train_prpo_pennies(env: MatchingPenniesEnvironment, episodes: int, input_dim: int, output_dim: int, device: torch.device,
                       lambda_nash: float = 1.0, lambda_exploit: float = 0.5) -> PPOAgent:
    """
    Main function to set up and run the corrected PRPO training.
    """
    def adaptive_lambda(base_lambda: float, current_exploitability: float) -> float:
        # Increase exploitability penalty if the agent is performing poorly
        return float(min(5.0 * base_lambda, base_lambda * (1.0 + 2.0 * current_exploitability)))

    mgr = PRPOManagerPennies(env, input_dim, output_dim, device,
                             population_size=5,
                             lambda_exploit_base=lambda_exploit,
                             lambda_nash=lambda_nash,  # Pass the Nash regularization weight
                             adaptive_lambda=adaptive_lambda)
    
    best_agent = mgr.train(total_episodes=episodes, games_per_iter=10, exploit_ratio=0.5)
    print("Training finished for PRPO (Matching Pennies, Unified Method).")
    return best_agent
# ============================================================================
# 4. Training and Evaluation Logic
# ============================================================================

def train_pennies_agent(agent, env, opponent, num_episodes=5000):
    print(f"Training {agent.__class__.__name__} against {opponent.name} in Matching Pennies...")
    if isinstance(agent, DQNAgent):
        return train_dqn_pennies(agent, env, opponent, episodes=num_episodes)
    if isinstance(agent, PPOAgent):
        return train_ppo_pennies(agent, env, opponent, episodes=num_episodes)
    return agent


def _act_to_int(policy: nn.Module, state: torch.Tensor) -> int:
    res = policy.act(state)
    if isinstance(res, (tuple, list)):
        return int(res[0])
    return int(res)


def evaluate_simple_avg_reward(policy: nn.Module, env: MatchingPenniesEnvironment, opponent: "ChallengerAgent", episodes: int = 200) -> float:
    total = 0.0
    for _ in range(episodes):
        s = env.reset(); opponent.reset(); hist: List[int] = []
        a = _act_to_int(policy, s)
        b = opponent.act(s, opponent_history=hist)
        s, rewards, _, _ = env.step([a, b])
        total += float(rewards[0])
    return total / float(episodes)

# ============================================================================
# 5. PSRO for Matching Pennies
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


def _eval_policy_vs_opp(policy: nn.Module, env: MatchingPenniesEnvironment, opponent: ChallengerAgent, episodes: int = 50) -> float:
    total = 0.0
    for _ in range(episodes):
        s = env.reset()
        opponent.reset()
        hist: List[int] = []
        for _ in range(1):  # single step
            a = policy.act(s)
            b = opponent.act(s, opponent_history=hist)
            hist.append(a)
            s, rewards, done, _ = env.step([a, b])
            total += rewards[0]
            if done:
                break
    return total / episodes


def _eval_pol_vs_pol(env: MatchingPenniesEnvironment, row: nn.Module, col: nn.Module, episodes: int = 100) -> float:
    total = 0.0
    for _ in range(episodes):
        s = env.reset()
        a = row.act(s)
        b = col.act(s)
        _, rewards, _, _ = env.step([a, b])
        total += rewards[0]
    return total / episodes


def _compute_meta(population: List[nn.Module], opponents: List[ChallengerAgent], env: MatchingPenniesEnvironment) -> List[float]:
    n = len(population)
    if n == 0:
        return []
    
    print(f"[PSRO-Pennies] Computing meta-strategy for {n} policies...")
    
    # Dynamically adjust episodes based on population size to prevent exponential slowdown
    if n <= 5:
        episodes_per_eval = 100
    elif n <= 10:
        episodes_per_eval = 60
    elif n <= 15:
        episodes_per_eval = 40
    else:
        episodes_per_eval = 20
    
    print(f"[PSRO-Pennies] Using {episodes_per_eval} episodes per evaluation ({n}×{n} = {n*n} total evaluations)")
    
    A = np.zeros((n, n), dtype=float)
    total_evaluations = n * n
    completed = 0
    
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 0.0
            else:
                A[i, j] = _eval_pol_vs_pol(env, population[i], population[j], episodes=episodes_per_eval)
            
            completed += 1
            if completed % max(1, total_evaluations // 10) == 0:  # Progress every 10%
                progress = (completed / total_evaluations) * 100
                print(f"[PSRO-Pennies] Meta-strategy computation progress: {progress:.0f}% ({completed}/{total_evaluations})")
    
    print(f"[PSRO-Pennies] Meta-strategy computation completed.")
    
    # Solve for mixed NE in zero-sum (row-player strategy)
    print(f"[PSRO-Pennies] Computing Nash equilibrium...")
    if _NASH_AVAILABLE:
        try:
            game = nash.Game(A, -A)
            
            # For very large populations, skip exact Nash and use uniform
            if n > 20:
                print(f"[PSRO-Pennies] Population too large ({n} > 20), skipping exact Nash computation.")
                raise TimeoutError("Population too large for Nash computation")
            
            # Use time-based timeout instead of signal for better compatibility
            nash_start_time = time.time()
            timeout_seconds = 30
            
            # Simple manual timeout check during Nash computation
            import time
            print(f"[PSRO-Pennies] Starting Nash computation with {timeout_seconds}s timeout...")
            
            eqs = list(game.support_enumeration())
            
            nash_time = time.time() - nash_start_time
            print(f"[PSRO-Pennies] Nash computation took {nash_time:.1f} seconds.")
            
            if nash_time > timeout_seconds:
                print(f"[PSRO-Pennies] Nash computation took too long ({nash_time:.1f}s), using uniform strategy.")
                raise TimeoutError("Nash computation took too long")
            
            if len(eqs) > 0 and len(eqs[0]) >= 1:
                row_sigma = np.array(eqs[0][0], dtype=float)
                row_sigma = np.clip(row_sigma, 0.0, 1.0)
                s = row_sigma.sum()
                meta = (row_sigma / s if s > 0 else np.ones(n) / n).tolist()
                _meta_last_matrix_pennies[:] = [row[:] for row in A.tolist()]
                _meta_last_equilibrium_pennies[:] = meta[:]
                print(f"[PSRO-Pennies] Nash equilibrium computed successfully.")
                return meta
            else:
                print(f"[PSRO-Pennies] No Nash equilibrium found, using uniform strategy.")
        except (TimeoutError, Exception) as e:
            print(f"[PSRO-Pennies] Nash computation failed ({e}), falling back to uniform strategy.")
    _meta_last_matrix_pennies[:] = [row[:] for row in A.tolist()]
    _meta_last_equilibrium_pennies[:] = (np.ones(n) / n).tolist()
    return _meta_last_equilibrium_pennies


_meta_last_matrix_pennies: List[List[float]] = []
_meta_last_equilibrium_pennies: List[float] = []


def train_best_response_dqn(env: MatchingPenniesEnvironment, opponents: List[ChallengerAgent], episodes: int, input_dim: int, output_dim: int) -> nn.Module:
    br = DQNAgent(input_dim, output_dim)
    opt = optim.Adam(br.q_net.parameters(), lr=1e-3)
    for _ in range(episodes):
        s = env.reset()
        # Avoid deepcopying challengers to prevent RNG reconstruction errors
        opp = random.choice(opponents)
        opp.reset()
        hist: List[int] = []
        a = br.act(s)
        b = opp.act(s, opponent_history=hist)
        hist.append(a)
        ns, rewards, done, _ = env.step([a, b])
        r = rewards[0]
        q_pred = br(s.unsqueeze(0))[0, a]
        loss = nn.functional.mse_loss(q_pred, torch.tensor(r, dtype=torch.float32))
        opt.zero_grad(); loss.backward(); opt.step()
    return br


def train_psro_pennies(gauntlet: "EnhancedGauntletBenchmark", time_budget_seconds: float, episodes_per_iter: int = 100) -> PSROPolicy:
    """
    Train PSRO for Matching Pennies using actual time budget instead of fixed iterations.
    """
    print(f"[PSRO-Pennies] Training for {time_budget_seconds} seconds...")
    start_time = time.time()
    
    # Use references to registered challengers (reset between uses); avoid deepcopy
    opponents = [ch for name, ch in gauntlet.master_challenger_list.items() if name.startswith("Pennies_")]
    if not opponents:
        opponents = [FrequencyCounterAgent()]
    env = MatchingPenniesEnvironment()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    population: List[nn.Module] = []
    iteration_count = 0
    
    while (time.time() - start_time) < time_budget_seconds:
        elapsed_time = time.time() - start_time
        print(f"[PSRO-Pennies] Iteration {iteration_count + 1} (Elapsed: {elapsed_time:.1f}s)")
        br = train_best_response_dqn(env, opponents, episodes_per_iter, input_dim, output_dim)
        population.append(copy.deepcopy(br))
        iteration_count += 1
        
        # Safety check to prevent infinite loops with very small time budgets
        if iteration_count >= 20:  # Max reasonable iterations
            print(f"[PSRO-Pennies] Reached maximum iterations ({iteration_count}), stopping.")
            break
    
    final_time = time.time() - start_time
    print(f"[PSRO-Pennies] Training completed in {final_time:.1f}s with {iteration_count} iterations")
    
    if not population:
        # Fallback: train at least one best response if no time was sufficient
        print("[PSRO-Pennies] Warning: No iterations completed, training one best response")
        br = train_best_response_dqn(env, opponents, episodes_per_iter, input_dim, output_dim)
        population.append(copy.deepcopy(br))
    
    meta = _compute_meta(population, opponents, env)
    print(f"[PSRO-Pennies] Meta: {np.round(meta, 3)}")
    print(f"[PSRO-Pennies] PSRO training completed successfully!")
    return PSROPolicy(population, meta)


if __name__ == "__main__":
    seeds = 1
    time_budget_seconds = 10.0  # 60 seconds per algorithm per seed

    print("\n--- Setting up Gauntlet Benchmark for Matching Pennies Evaluation ---")
    config = EvaluationConfig(num_episodes=200, parallel_workers=1, save_visualizations=True)
    gauntlet = EnhancedGauntletBenchmark(config)

    # Register env
    pennies_payoff_matrix = np.array([[1, -1], [-1, 1]])
    gauntlet.register_environment(
        "MatchingPennies",
        MatchingPenniesEnvironment,
        payoff_matrices=(pennies_payoff_matrix, -pennies_payoff_matrix),
        game_prefix="Pennies",
        zero_sum=True
    )

    # Agents
    env = MatchingPenniesEnvironment()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    dqn_agent = DQNAgent(input_dim, output_dim)
    ppo_agent = PPOAgent(input_dim, output_dim)
    training_opponent = gauntlet.master_challenger_list['Pennies_Uniform']
    training_opponent.name = "TrainingOpponent_Uniform"

    # Output dirs
    base_dir = os.path.join("results", "MatchingPennies")
    os.makedirs(base_dir, exist_ok=True)
    timing_train: Dict[str, float] = {}
    timing_eval: Dict[str, float] = {}
    dqn_dir = os.path.join(base_dir, "DQN"); os.makedirs(dqn_dir, exist_ok=True)
    ppo_dir = os.path.join(base_dir, "PPO"); os.makedirs(ppo_dir, exist_ok=True)
    psro_dir = os.path.join(base_dir, "PSRO"); os.makedirs(psro_dir, exist_ok=True)

    # Train multi-seed with time budget
    print(f"\n--- Starting Matching Pennies Training Phase DQN/PPO ({time_budget_seconds}s per seed x {seeds} seeds) ---")
    dqn_runs = []
    _t0 = time.time()
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        trained, episodes_completed, _ = TimeBudgetTrainer.train_dqn_time_budget(
            copy.deepcopy(dqn_agent), lambda: env, training_opponent, 
            time_budget_seconds, device=torch.device('cpu')
        )
        dqn_runs.append(trained)
        print(f"    DQN Seed {seed}: Completed {episodes_completed} episodes in {time_budget_seconds}s")
    trained_dqn = dqn_runs[-1]
    timing_train["DQN"] = float(time.time() - _t0)
    
    ppo_runs = []
    _t0 = time.time()
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        trained, episodes_completed, _ = TimeBudgetTrainer.train_standard_ppo_time_budget(
            lambda: MPEnvironment(), input_dim, output_dim, time_budget_seconds, 
            device='cpu'
        )
        ppo_runs.append(trained)
        print(f"    PPO Seed {seed}: Completed {episodes_completed} episodes in {time_budget_seconds}s")
    trained_ppo = ppo_runs[-1]
    timing_train["PPO"] = float(time.time() - _t0)
    torch.save(trained_dqn.state_dict(), os.path.join(dqn_dir, "model.pt"))
    # Save the underlying PPO policy parameters (StandardPPO has no state_dict)
    torch.save(trained_ppo.policy.state_dict(), os.path.join(ppo_dir, "model.pt"))

    # PSRO with time budget
    print(f"\n--- Training Pennies PSRO ({time_budget_seconds}s total) ---")
    _t0 = time.time()
    # Use actual time budget instead of approximated episodes per iteration
    psro_policy = train_psro_pennies(gauntlet, time_budget_seconds=time_budget_seconds, episodes_per_iter=100)
    timing_train["PSRO"] = float(time.time() - _t0)
    meta = psro_policy.meta_strategy.cpu().numpy().tolist()
    for i, pol in enumerate(psro_policy.population):
        torch.save(pol.state_dict(), os.path.join(psro_dir, f"pop_member_{i}.pt"))
    import json
    with open(os.path.join(psro_dir, "meta_strategy.json"), "w") as f:
        json.dump({"meta_strategy": meta, "population_size": len(meta)}, f, indent=2)
    # Save empirical meta-game as well
    try:
        with open(os.path.join(psro_dir, "meta_game.json"), "w") as f:
            json.dump({
                "payoff_row": _meta_last_matrix_pennies,
                "equilibrium_row": _meta_last_equilibrium_pennies
            }, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save meta_game.json: {e}")

    # Self-Play with time budget
    print(f"\n{'='*60}")
    print(f"PSRO TRAINING PHASE COMPLETED - STARTING SELF-PLAY")
    print(f"{'='*60}")
    # Use actual time budget for Self-Play instead of approximated episodes
    print(f"\n--- Training Pennies Self-Play ({time_budget_seconds}s time budget) ---")
    _t0 = time.time()
    sp_agent = train_selfplay_pennies(env, time_budget_seconds=time_budget_seconds, input_dim=input_dim, output_dim=output_dim, device=torch.device('cpu'))
    timing_train["SelfPlay"] = float(time.time() - _t0)
    sp_dir = os.path.join(base_dir, "SelfPlay"); os.makedirs(sp_dir, exist_ok=True)
    torch.save(sp_agent.state_dict(), os.path.join(sp_dir, "model.pt"))

    # PRPO with superior unified implementation and time budget
    print(f"\n--- Training Pennies PRPO Unified ({time_budget_seconds}s per seed x {seeds} seeds) ---")
    # Multi-seed PRPO for paired testing
    prpo_runs: List[nn.Module] = []
    _t0 = time.time()
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        prpo_tr = train_prpo_mp_unified(time_budget_seconds, input_dim, output_dim, torch.device('cpu'))
        prpo_runs.append(prpo_tr)
        print(f"    PRPO Seed {seed}: Training completed in {time_budget_seconds}s")
    prpo_agent = prpo_runs[-1]
    timing_train["PRPO"] = float(time.time() - _t0)
    prpo_dir = os.path.join(base_dir, "PRPO"); os.makedirs(prpo_dir, exist_ok=True)
    torch.save(prpo_agent.state_dict(), os.path.join(prpo_dir, "model.pt"))

    # Add custom challenger
    gauntlet.add_custom_challenger("Pennies_FrequencyCounter", FrequencyCounterAgent())

    # Evaluate
    def eval_and_report(g: "EnhancedGauntletBenchmark", policy: nn.Module, name: str, out_dir: str):
        print(f"\n{'='*40}\nEVALUATING: {name}\n{'='*40}")
        g.evaluate_policy(policy=policy, policy_name=name, environments=["MatchingPennies"])
        report_path = os.path.join(out_dir, "report.json")
        g.generate_report(report_path)
        # Inject seed statistics into the report, if available
        try:
            base_dir = os.path.dirname(out_dir)
            stats_path = os.path.join(base_dir, "pennies_stats.json")
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

    _t0 = time.time(); trained_dqn.eval(); eval_and_report(gauntlet, trained_dqn, "Pennies_DQN", dqn_dir); timing_eval["DQN"] = float(time.time() - _t0)
    _t0 = time.time(); trained_ppo.eval(); eval_and_report(gauntlet, trained_ppo, "Pennies_PPO", ppo_dir); timing_eval["PPO"] = float(time.time() - _t0)
    _t0 = time.time(); psro_policy.eval(); eval_and_report(gauntlet, psro_policy, "Pennies_PSRO", psro_dir); timing_eval["PSRO"] = float(time.time() - _t0)
    _t0 = time.time(); sp_agent.eval(); eval_and_report(gauntlet, sp_agent, "Pennies_SelfPlay", sp_dir); timing_eval["SelfPlay"] = float(time.time() - _t0)
    _t0 = time.time(); prpo_agent.eval(); eval_and_report(gauntlet, prpo_agent, "Pennies_PRPO", prpo_dir); timing_eval["PRPO"] = float(time.time() - _t0)

    # Statistical comparison PRPO vs PPO
    print("\n--- Computing seed-wise scores, 95% CI, and paired t-test (Pennies PRPO vs PPO) ---")
    eval_env = MatchingPenniesEnvironment()
    fixed_opp = gauntlet.master_challenger_list['Pennies_Uniform']
    fixed_opp.name = "EvalOpponent_Uniform"
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
    with open(os.path.join(base_dir, "pennies_stats.json"), "w") as f:
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

    print("\n\n🎉 Matching Pennies training and evaluation complete. Outputs under 'results/MatchingPennies/'. 🎉")