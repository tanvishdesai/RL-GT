# rps
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import os
import math
import argparse
from typing import Optional, List, Dict, Tuple, Any, TYPE_CHECKING, Callable
import time
from collections import namedtuple
import torch.nn.functional as F
import json


# Gym spaces (with safe fallback if not available)
try:
    from gym.spaces import Space, Discrete
except Exception:
    class Space:  # type: ignore
        pass
    class Discrete:  # type: ignore
        def __init__(self, n: int):
            self.n = int(n)

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
    _scipy_stats = None  # type: ignore
    _SCIPY_AVAILABLE = False

# ----------------------------------------------------------------------------
# Statistical helpers: mean/std/SEM/95% CI and paired t-test
# ----------------------------------------------------------------------------

def _t_critical_95(n: int) -> float:
    """Return two-tailed 95% t critical value for sample size n (df=n-1).
    Uses SciPy when available; otherwise falls back to a small lookup table
    and finally to z=1.96 as a conservative default for large df.
    """
    if n <= 1:
        return float("nan")
    df = n - 1
    if _SCIPY_AVAILABLE:
        try:
            return float(_scipy_stats.t.ppf(0.975, df))
        except Exception:
            pass
    # Lookup table for common dfs (two-tailed 95%)
    lookup = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980
    }
    if df in lookup:
        return float(lookup[df])
    # nearest higher known df, else z~1.96
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
    # Without SciPy, omit p-value to avoid inaccurate approximations
    return None


# ============================================================================
# 2. RPS-Specific Functions for Unified PRPO
# ============================================================================

class RPSEnvironment:
    """Rock Paper Scissors game environment for use with unified PRPO."""
    def __init__(self):
        self.payoff_matrix = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        self.state_dim = 3
        self.action_dim = 3
        self.nash_equilibrium = np.array([1/3, 1/3, 1/3])
        self.reset()

    def reset(self):
        self.last_opponent_action = random.randint(0, 2)
        return self._get_state()

    def _get_state(self):
        state = np.zeros(self.state_dim)
        state[self.last_opponent_action] = 1.0
        return state

    def step(self, p1_action: int, p2_action: int):
        p1_reward = self.payoff_matrix[p1_action, p2_action]
        p2_reward = -p1_reward
        self.last_opponent_action = p2_action
        return self._get_state(), [p1_reward, p2_reward], True

def get_rps_nash_policy_callable(policy_probs_batch: torch.Tensor) -> torch.Tensor:
    """Returns the uniform Nash equilibrium distribution for RPS."""
    return torch.full_like(policy_probs_batch, 1/3)

def calculate_rps_exploitability_callable(policy: UnifiedActorCritic) -> float:
    """Calculates exploitability against a suite of hard-coded RPS bots."""
    policy.eval()
    device = next(policy.parameters()).device
    # State is constant in RPS, so we can use a dummy state
    state = torch.FloatTensor(np.zeros(3)).unsqueeze(0).to(device)
    state[0, random.randint(0, 2)] = 1.0  # Use a dummy state
    
    with torch.no_grad():
        policy_probs, _ = policy(state)
        p = policy_probs.squeeze().cpu().numpy()

    # Exploitability is the best possible reward an opponent can get against our policy 'p'
    # Opponent plays Rock: reward is p[1]*(-1) + p[2]*(1) = p[2] - p[1]
    # Opponent plays Paper: reward is p[0]*(1) + p[2]*(-1) = p[0] - p[2]
    # Opponent plays Scissors: reward is p[0]*(-1) + p[1]*(1) = p[1] - p[0]
    max_exploit = max(p[2]-p[1], p[0]-p[2], p[1]-p[0])
    policy.train()
    return max(0.0, max_exploit)

def get_rps_exploiter_opponents() -> List[Callable]:
    """Returns a list of functions, each representing a simple exploiter bot."""
    return [
        lambda: 0,  # Always Rock
        lambda: 1,  # Always Paper
        lambda: 2,  # Always Scissors
        lambda: np.random.choice([0, 1, 2], p=[0.8, 0.1, 0.1]),  # Biased Rock
        lambda: np.random.choice([0, 1, 2], p=[0.1, 0.8, 0.1]),  # Biased Paper
        lambda: np.random.choice([0, 1, 2], p=[0.1, 0.1, 0.8]),  # Biased Scissors
    ]

def train_prpo_rps_unified(time_budget_seconds: float, input_dim: int, output_dim: int, 
                          device: torch.device, lambda_nash: float = 1.5, 
                          lambda_exploit: float = 1.0) -> nn.Module:
    """
    Superior PRPO training function using the unified framework with time budget.
    """
    print(f"Training PRPO (Unified Framework) for {time_budget_seconds} seconds...")
    
    # The manager needs a way to create new environments
    env_factory = lambda: RPSEnvironment()

    prpo_system = UnifiedPRPO(
        state_dim=input_dim, 
        action_dim=output_dim, 
        lr=1e-4, 
        device=str(device),
        population_size=4,
        lambda_nash=lambda_nash,
        nash_target_callable=get_rps_nash_policy_callable,
        lambda_exploit=lambda_exploit,
        exploitability_calculator=calculate_rps_exploitability_callable,
        exploiter_opponents=get_rps_exploiter_opponents()
    )
    
    # Train the system with time budget
    episodes_completed, results_over_time = prpo_system.train_time_budget(
        env_factory=env_factory, 
        time_budget_seconds=time_budget_seconds,
        update_every_seconds=1.0
    )
    
    # Get the best agent from the trained population
    best_agent = prpo_system.get_best_agent()
    
    print("Training finished for PRPO (Unified Framework).")
    return best_agent

# ============================================================================
# 3. MARL Algorithm Implementations (Updated with Time Budget Training)
# ============================================================================

# --- START: CORRECTED AGENT DEFINITION ---
# This agent is now a valid ChallengerAgent because it implements the required
# abstract property: `compatible_action_space`.
class RuleBasedAgent(ChallengerAgent):
    """
    A simple rule-based agent that counters the opponent's most frequent move.
    It correctly uses the opponent_history provided by the act method and declares
    its compatibility with Rock-Paper-Scissors.
    """
    def __init__(self, name: str = "RuleBasedAgent"):
        super().__init__(name, "easy")

    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        """Plays the move that beats the opponent's most frequent past move."""
        if not opponent_history:
            return random.randint(0, 2)
        
        # Find the most common action the opponent has taken
        most_frequent_move = max(set(opponent_history), key=opponent_history.count)
        
        # Play the counter-move (for RPS: 0 beats 2, 1 beats 0, 2 beats 1)
        return (most_frequent_move + 1) % 3

    def update(self, reward: float, observation: torch.Tensor, action: int):
        """Rule-based agent does not need to learn from rewards."""
        pass
        
    @property
    def compatible_action_space(self) -> Space:
        """Declares that this agent is designed for a 3-action space like RPS."""
        return Discrete(3)
# --- END: CORRECTED AGENT DEFINITION ---


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
        
        # Handle dimension mismatch: map 6D gauntlet RPS state [self_onehot(3), opp_onehot(3)] to 3D RPS input (opp_onehot)
        if ts.shape[-1] == 6 and self._input_dim == 3:
            ts = ts[..., 3:6]
            print(f"Info: Mapping 6D gauntlet state to 3D RPS input by selecting opponent one-hot: {ts.shape}")
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
    """DQN with target network, replay buffer, and epsilon-greedy policy."""
    def __init__(self, input_dim: int, output_dim: int, gamma: float = 0.99):
        super().__init__()
        self.q_net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, output_dim))
        self.target_q_net = copy.deepcopy(self.q_net)
        self.gamma = float(gamma)
        self.num_actions = int(output_dim)
        self.epsilon: float = 1.0
        self.epsilon_min: float = 0.05
        self.epsilon_decay: float = 0.9995

    def forward(self, state: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
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
    """PPO with GAE, clipped objective, mini-batches, entropy bonus."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, output_dim))
        self.critic = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def act(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        logits = self.actor(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state).squeeze(-1)
        return int(action.item()), log_prob.squeeze(), value.squeeze()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(states)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states).squeeze(-1)
        return log_probs, entropy, values


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
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


def train_dqn_rps(agent: DQNAgent, env, opponent, episodes: int = 5000, device: torch.device = torch.device('cpu')) -> DQNAgent:
    agent.to(device)
    target_update_interval = 250
    batch_size = 64
    buffer = ReplayBuffer(10000)
    optimizer = optim.Adam(agent.q_net.parameters(), lr=1e-3)
    global_step = 0
    for _ in range(episodes):
        state = env.reset()
        opponent.reset()
        agent_hist: List[int] = []
        for _ in range(10):
            a = agent.act(state, explore=True)
            b = opponent.act(state, opponent_history=agent_hist)
            agent_hist.append(a)
            next_state, rewards, done, _ = env.step([a, b])
            r = float(rewards[0])
            buffer.push(state, a, r, next_state, bool(done))
            state = next_state
            global_step += 1

            if len(buffer) >= batch_size:
                s_batch, a_batch, r_batch, ns_batch, d_batch = buffer.sample(batch_size)
                s_batch = s_batch.to(device)
                ns_batch = ns_batch.to(device)
                a_batch = a_batch.to(device)
                r_batch = r_batch.to(device)
                d_batch = d_batch.to(device)
                q_pred = agent.q_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next = agent.target_q_net(ns_batch).max(dim=1).values
                    target = r_batch + agent.gamma * max_next * (~d_batch)
                loss = nn.functional.mse_loss(q_pred, target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            if global_step % target_update_interval == 0:
                agent.sync_target()
            agent.update_epsilon()
            if done:
                break
    agent.epsilon = agent.epsilon_min
    agent.eval()
    return agent


def train_ppo_rps(agent: PPOAgent, env, opponent, episodes: int = 5000, device: torch.device = torch.device('cpu')) -> PPOAgent:
    agent.to(device)
    optimizer = optim.Adam(list(agent.actor.parameters()) + list(agent.critic.parameters()), lr=3e-4)
    clip_eps = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    gamma = 0.99
    lam = 0.95
    update_epochs = 4
    minibatch_size = 128
    rollout = RolloutBuffer()

    def flush_and_optimize():
        if len(rollout.states) == 0:
            return
        states = torch.stack(rollout.states).to(device)
        actions = torch.tensor(rollout.actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rollout.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(rollout.dones, dtype=torch.bool, device=device)
        old_log_probs = torch.stack(rollout.log_probs).detach().to(device)
        values = torch.stack(rollout.values).detach().to(device)
        adv, rets = compute_gae(rewards, values, dones, gamma, lam)
        # Robust normalization: avoid NaNs when batch has a single element
        adv_mean = adv.mean()
        adv_std = adv.std(unbiased=False)
        if torch.isnan(adv_std) or adv_std < 1e-8:
            adv = adv - adv_mean
        else:
            adv = (adv - adv_mean) / (adv_std + 1e-8)
        dataset = list(range(states.size(0)))
        for _ in range(update_epochs):
            random.shuffle(dataset)
            for i in range(0, len(dataset), minibatch_size):
                idx = dataset[i:i+minibatch_size]
                b_states = states[idx]
                b_actions = actions[idx]
                b_adv = adv[idx]
                b_rets = rets[idx]
                b_old_logp = old_log_probs[idx]
                logp, entropy, values_new = agent.evaluate_actions(b_states, b_actions)
                ratio = torch.exp(logp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(values_new, b_rets)
                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy.mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        rollout.clear()

    for _ in range(episodes):
        state = env.reset(); opponent.reset(); agent_hist: List[int] = []
        for _ in range(10):
            action, logp, value = agent.act(state)
            b = opponent.act(state, opponent_history=agent_hist)
            agent_hist.append(action)
            next_state, rewards, done, _ = env.step([action, b])
            r = float(rewards[0])
            rollout.states.append(state.detach().cpu())
            rollout.actions.append(int(action))
            rollout.rewards.append(r)
            rollout.dones.append(bool(done))
            rollout.log_probs.append(logp.detach().cpu())
            rollout.values.append(value.detach().cpu())
            state = next_state
            if done:
                break
        # Optimize after each episode (sufficient for small single-step games)
        flush_and_optimize()
    agent.eval(); return agent

# ============================================================================
# X. Self-Play Baseline and PRPO for RPS
# ============================================================================

class SelfPlayOpponent:
    """Opponent that plays using a buffer of past snapshots of the agent."""
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
        clone.eval()
        self._snapshots.append(clone)
        if len(self._snapshots) > 20:
            self._snapshots.pop(0)

    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        if not self._snapshots or random.random() < 0.3:
            return random.randint(0, self._action_dim - 1)
        opponent = random.choice(self._snapshots)
        action_or_tuple = opponent.act(observation)
        if isinstance(action_or_tuple, (tuple, list)):
            return int(action_or_tuple[0])
        return int(action_or_tuple)


def train_selfplay_rps(env, num_episodes: int, input_dim: int, output_dim: int, device: torch.device) -> PPOAgent:
    agent = PPOAgent(input_dim, output_dim).to(device)
    opponent = SelfPlayOpponent(output_dim, input_dim, device)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)

    print(f"Starting Self-Play RPS training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        try:
            state = env.reset()
            opponent.reset()
            agent_hist: List[int] = []
            for _ in range(10):
                a, logp, value = agent.act(state)
                b = opponent.act(state, opponent_history=agent_hist)
                agent_hist.append(int(a))
                next_state, rewards, done, _ = env.step([int(a), int(b)])
                reward = rewards[0]
                advantage = torch.tensor(reward, dtype=torch.float32) - value.detach().squeeze()
                actor_loss = -logp * advantage
                critic_loss = nn.functional.mse_loss(value.squeeze(), torch.tensor(reward, dtype=torch.float32))
                loss = actor_loss + critic_loss

                # Check for NaN/inf values
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss at episode {episode}, skipping update")
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                state = next_state
                if done:
                    break
        except Exception as e:
            print(f"Error in self-play episode {episode}: {e}")
            continue

        if (episode + 1) % 25 == 0:
            opponent.add_snapshot(agent)
        if (episode + 1) % 1000 == 0:
            print(f"Self-Play RPS progress: {episode+1}/{num_episodes} episodes completed")

    print("Training finished for Self-Play (RPS).")
    return agent


def _rps_exploitability_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.detach(), dim=-1).squeeze()
    return torch.max(torch.stack([probs[2]-probs[1], probs[0]-probs[2], probs[1]-probs[0]])).clamp(min=0.0)


# =========================================================================
# Matured PRPO (population-based) for RPS
# =========================================================================

Experience = namedtuple("Experience", ["state", "action", "reward", "done", "log_prob", "value"]) 


# =========================================================================
# The old PRPO implementation has been replaced with the superior 
# UnifiedPRPO framework imported from unified_prpo.py
# =========================================================================
# ============================================================================
# 3. Training Logic
# ============================================================================

def train_agent(agent, env, opponent, num_episodes=5000):
    """Wrapper that dispatches to upgraded trainers for DQN and PPO."""
    print(f"Starting training for {agent.__class__.__name__} against {opponent.name}...")
    if isinstance(agent, DQNAgent):
        return train_dqn_rps(agent, env, opponent, episodes=num_episodes)
    if isinstance(agent, PPOAgent):
        return train_ppo_rps(agent, env, opponent, episodes=num_episodes)
    # Fallback: no-op
    return agent


# ----------------------------------------------------------------------------
# Simple evaluation: average reward vs a fixed opponent with many episodes
# ----------------------------------------------------------------------------

def _act_to_int(policy: nn.Module, state: torch.Tensor) -> int:
    # Robustly convert state to a torch tensor for policies expecting tensors
    ts = torch.as_tensor(state, dtype=torch.float32)
    res = policy.act(ts)
    if isinstance(res, (tuple, list)):
        return int(res[0])
    return int(res)


def _safe_env_step(env, a: int, b: int):
    """Call env.step robustly and normalize return to (next_state, rewards, done, info)."""
    try:
        out = env.step([a, b])
    except Exception:
        out = env.step(a, b)
    if isinstance(out, tuple) and len(out) == 3:
        ns, rewards, done = out
        info = {}
    else:
        ns, rewards, done, info = out
    return ns, rewards, done, info


def evaluate_simple_avg_reward(policy: nn.Module, env, opponent: "ChallengerAgent", episodes: int = 200) -> float:
    total = 0.0
    for _ in range(episodes):
        s = env.reset(); opponent.reset(); hist: List[int] = []
        for _ in range(10):
            a = _act_to_int(policy, s)
            b = opponent.act(s, opponent_history=hist)
            hist.append(a)
            s, rewards, done, _ = _safe_env_step(env, a, b)
            total += float(rewards[0])
            if done:
                break
    return total / float(episodes)

# ============================================================================
# 4. PSRO Implementation
# ============================================================================

class PSROPolicy(nn.Module):
    """Mixture policy over a population of base policies with a meta-strategy."""
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


def _evaluate_policy_against_opponent(policy: nn.Module, env, opponent: ChallengerAgent, episodes: int = 50) -> float:
    total_reward = 0.0
    for _ in range(episodes):
        state = env.reset()
        opponent.reset()
        agent_history: List[int] = []
        for _ in range(10):
            a = _act_to_int(policy, state)
            b = opponent.act(state, opponent_history=agent_history)
            agent_history.append(a)
            next_state, rewards, done, _ = _safe_env_step(env, a, b)
            total_reward += rewards[0]
            state = next_state
            if done:
                break
    return total_reward / episodes


def _eval_policy_vs_policy(env, policy_row: nn.Module, policy_col: nn.Module, episodes: int = 50) -> float:
    """Row-player average payoff vs column-player in the given zero-sum env (RPS)."""
    total = 0.0
    for _ in range(episodes):
        s = env.reset()
        # single-step game
        a = _act_to_int(policy_row, s)
        b = _act_to_int(policy_col, s)
        _, rewards, _, _ = _safe_env_step(env, a, b)
        total += rewards[0]
    return total / episodes


def _compute_meta_strategy(population: List[nn.Module], opponents: List[ChallengerAgent], env) -> List[float]:
    # Build empirical meta-game among population
    n = len(population)
    if n == 0:
        return []
    
    print(f"[PSRO] Computing meta-strategy for {n} policies...")
    
    # Dynamically adjust episodes based on population size to prevent exponential slowdown
    if n <= 5:
        episodes_per_eval = 50
    elif n <= 10:
        episodes_per_eval = 30
    elif n <= 15:
        episodes_per_eval = 20
    else:
        episodes_per_eval = 10
    
    print(f"[PSRO] Using {episodes_per_eval} episodes per evaluation ({n}×{n} = {n*n} total evaluations)")
    
    A = np.zeros((n, n), dtype=float)
    total_evaluations = n * n
    completed = 0
    
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 0.0
            else:
                A[i, j] = _eval_policy_vs_policy(env, population[i], population[j], episodes=episodes_per_eval)
            
            completed += 1
            if completed % max(1, total_evaluations // 10) == 0:  # Progress every 10%
                progress = (completed / total_evaluations) * 100
                print(f"[PSRO] Meta-strategy computation progress: {progress:.0f}% ({completed}/{total_evaluations})")
    
    print(f"[PSRO] Meta-strategy computation completed.")

    # Solve for mixed NE in zero-sum (row-player strategy)
    print(f"[PSRO] Computing Nash equilibrium...")
    if _NASH_AVAILABLE:
        try:
            game = nash.Game(A, -A)
            
            # For very large populations, skip exact Nash and use uniform
            if n > 20:
                print(f"[PSRO] Population too large ({n} > 20), skipping exact Nash computation.")
                raise TimeoutError("Population too large for Nash computation")
            
            # Use time-based timeout instead of signal for better compatibility
            nash_start_time = time.time()
            timeout_seconds = 30
            
            # Simple manual timeout check during Nash computation
            import time
            print(f"[PSRO] Starting Nash computation with {timeout_seconds}s timeout...")
            
            eqs = list(game.support_enumeration())
            
            nash_time = time.time() - nash_start_time
            print(f"[PSRO] Nash computation took {nash_time:.1f} seconds.")
            
            if nash_time > timeout_seconds:
                print(f"[PSRO] Nash computation took too long ({nash_time:.1f}s), using uniform strategy.")
                raise TimeoutError("Nash computation took too long")
            
            if len(eqs) > 0 and len(eqs[0]) >= 1:
                row_sigma = np.array(eqs[0][0], dtype=float)
                row_sigma = np.clip(row_sigma, 0.0, 1.0)
                s = row_sigma.sum()
                meta = (row_sigma / s if s > 0 else np.ones(n) / n).tolist()
                # attach for downstream export
                _meta_last_matrix[:] = [row[:] for row in A.tolist()]
                _meta_last_equilibrium[:] = meta[:]
                print(f"[PSRO] Nash equilibrium computed successfully.")
                return meta
            else:
                print(f"[PSRO] No Nash equilibrium found, using uniform strategy.")
        except (TimeoutError, Exception) as e:
            print(f"[PSRO] Nash computation failed ({e}), falling back to uniform strategy.")
    # Fallback to uniform if solver unavailable
    _meta_last_matrix[:] = [row[:] for row in A.tolist()]
    _meta_last_equilibrium[:] = (np.ones(n) / n).tolist()
    return _meta_last_equilibrium


# --- storage for export ---
_meta_last_matrix: List[List[float]] = []
_meta_last_equilibrium: List[float] = []


def train_best_response_dqn(env, opponents: List[ChallengerAgent], episodes: int, input_dim: int, output_dim: int) -> nn.Module:
    br_agent = DQNAgent(input_dim, output_dim)
    optimizer = optim.Adam(br_agent.parameters(), lr=1e-3)

    for _ in range(episodes):
        state = env.reset()
        # sample opponent uniformly from the mixture
        opp = copy.deepcopy(random.choice(opponents))
        opp.reset()
        agent_history: List[int] = []
        for _ in range(10):
            a = br_agent.act(state)
            b = opp.act(state, opponent_history=agent_history)
            agent_history.append(a)
            next_state, rewards, done, _ = env.step([a, b])
            reward = rewards[0]

            state_batch = state.unsqueeze(0)
            q_pred = br_agent(state_batch)[0, a]
            with torch.no_grad():
                target_q = torch.tensor(reward, dtype=torch.float32)
            loss = nn.functional.mse_loss(q_pred, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state
            if done:
                break
    return br_agent


def train_psro(env, gauntlet: "EnhancedGauntletBenchmark", iterations: int, episodes_per_iter: int) -> PSROPolicy:
    # Build opponent population from the benchmark challengers for RPS
    opponent_pool: List[ChallengerAgent] = []
    for name, chal in gauntlet.master_challenger_list.items():
        if name.startswith("RPS_") or name.startswith("Student_"):
            opponent_pool.append(copy.deepcopy(chal))
    if not opponent_pool:
        # Fallback to a simple rule-based opponent
        opponent_pool = [RuleBasedAgent(name="FallbackOpponent")]

    # Initialize population with a single DQN BR to uniform opponents
    env_train = gauntlet._create_default_environment()
    input_dim = env_train.observation_space.shape[0]
    output_dim = env_train.action_space.n
    population: List[nn.Module] = []

    for it in range(iterations):
        print(f"[PSRO] Iteration {it+1}/{iterations}")
        # Train a best response policy against (approx) mixture of opponent pool
        br_policy = train_best_response_dqn(env_train, opponent_pool, episodes_per_iter, input_dim, output_dim)
        population.append(copy.deepcopy(br_policy))

    meta = _compute_meta_strategy(population, opponent_pool, env_train)
    print(f"[PSRO] Learned meta-strategy over {len(population)} policies: {np.round(meta, 3)}")
    print(f"[PSRO] PSRO training completed successfully!")
    return PSROPolicy(population, meta)

# ============================================================================
# 4. Main Execution
# ============================================================================

if __name__ == "__main__":
    # Updated parameters - now using time budget instead of episode count
    num_seeds = 2
    time_budget_seconds = 10.0  # 60 seconds per algorithm per seed

    base_results_dir = os.path.join("results", "RPS")
    os.makedirs(base_results_dir, exist_ok=True)
    training_times: Dict[str, float] = {}
    eval_times: Dict[str, float] = {}

    print("\n--- Setting up Gauntlet Benchmark for Evaluation ---")
    config = EvaluationConfig(
        num_episodes=200,
        parallel_workers=1,
        save_visualizations=True,
        compute_exploitability=True,
    )

    # Prepare environment for training
    gauntlet_train = EnhancedGauntletBenchmark(config)
    env_factory = gauntlet_train._create_default_environment
    training_env = env_factory()
    input_dim = training_env.observation_space.shape[0]
    output_dim = training_env.action_space.n
    # Ensure RPS-specific components use the correct RPS env dimensions
    _rps_probe = RPSEnvironment()
    rps_input_dim = _rps_probe.state_dim
    rps_output_dim = _rps_probe.action_dim
    
    # DEBUG: Print dimensions to identify the mismatch
    print(f"DEBUG: Gauntlet env dimensions - input_dim: {input_dim}, output_dim: {output_dim}")
    print(f"DEBUG: RPS env dimensions - rps_input_dim: {rps_input_dim}, rps_output_dim: {rps_output_dim}")
    print(f"DEBUG: Training env observation space: {training_env.observation_space}")
    print(f"DEBUG: RPS env state sample: {_rps_probe._get_state().shape}")

    # Instantiate learning agents
    dqn_agent = DQNAgent(input_dim, output_dim)  # DQN uses gauntlet dimensions  
    training_opponent = RuleBasedAgent(name="TrainingOpponent")
    
    # Note: PPO agents are created during training, not here

    # Train DQN multi-seed with time budget
    print(f"\n--- Training DQN ({time_budget_seconds}s per seed x {num_seeds} seeds) ---")
    dqn_runs: List[nn.Module] = []
    _t0 = time.time()
    for seed in range(num_seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        trained, episodes_completed, _ = TimeBudgetTrainer.train_dqn_time_budget(
            copy.deepcopy(dqn_agent), lambda: training_env, training_opponent, 
            time_budget_seconds, device=torch.device('cpu')
        )
        dqn_runs.append(trained)
        print(f"    DQN Seed {seed}: Completed {episodes_completed} episodes in {time_budget_seconds}s")
    trained_dqn_agent = dqn_runs[-1]
    training_times["DQN"] = float(time.time() - _t0)
    dqn_dir = os.path.join(base_results_dir, "DQN")
    os.makedirs(dqn_dir, exist_ok=True)
    torch.save(trained_dqn_agent.state_dict(), os.path.join(dqn_dir, "model.pt"))

    # Train PPO multi-seed with time budget
    print(f"\n--- Training PPO ({time_budget_seconds}s per seed x {num_seeds} seeds) ---")
    ppo_runs: List[nn.Module] = []
    _t0 = time.time()
    for seed in range(num_seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        trained, episodes_completed, _ = TimeBudgetTrainer.train_standard_ppo_time_budget(
            lambda: RPSEnvironment(), rps_input_dim, rps_output_dim, time_budget_seconds, 
            device='cpu'
        )
        ppo_runs.append(trained)
        print(f"    PPO Seed {seed}: Completed {episodes_completed} episodes in {time_budget_seconds}s")
    trained_ppo_agent = ppo_runs[-1]
    training_times["PPO"] = float(time.time() - _t0)
    ppo_dir = os.path.join(base_results_dir, "PPO")
    os.makedirs(ppo_dir, exist_ok=True)
    torch.save(trained_ppo_agent.policy.state_dict(), os.path.join(ppo_dir, "model.pt"))

    # Train PSRO with time budget (fewer iterations due to time constraint)
    print(f"\n--- Training PSRO ({time_budget_seconds}s total) ---")
    gauntlet_psro = EnhancedGauntletBenchmark(config)
    _t0 = time.time()
    # Adjust iterations and episodes per iteration for time budget
    iterations = max(3, int(time_budget_seconds / 20))  # At least 3 iterations
    episodes_per_iter = max(100, int(time_budget_seconds / iterations))
    psro_policy = train_psro(training_env, gauntlet_psro, iterations=iterations, episodes_per_iter=episodes_per_iter)
    training_times["PSRO"] = float(time.time() - _t0)
    psro_dir = os.path.join(base_results_dir, "PSRO")
    os.makedirs(psro_dir, exist_ok=True)
    # Save PSRO population
    meta = psro_policy.meta_strategy.cpu().numpy().tolist()
    for i, pol in enumerate(psro_policy.population):
        torch.save(pol.state_dict(), os.path.join(psro_dir, f"pop_member_{i}.pt"))
    with open(os.path.join(psro_dir, "meta_strategy.json"), "w") as f:
        import json
        json.dump({"meta_strategy": meta, "population_size": len(meta)}, f, indent=2)
    # --- NEW: export empirical meta-game matrix and NE ---
    try:
        import json
        with open(os.path.join(psro_dir, "meta_game.json"), "w") as f:
            json.dump({
                "payoff_row": _meta_last_matrix,
                "equilibrium_row": _meta_last_equilibrium
            }, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save meta_game.json: {e}")

    # Train Self-Play baseline with time budget  
    print(f"\n{'='*60}")
    print(f"PSRO TRAINING PHASE COMPLETED - STARTING SELF-PLAY")
    print(f"{'='*60}")
    # Use reasonable episode scaling for self-play (max 10k episodes)
    selfplay_episodes = min(10000, max(1000, int(time_budget_seconds * 10)))
    print(f"\n--- Training Self-Play ({time_budget_seconds}s, {selfplay_episodes} episodes) ---")
    _t0 = time.time()
    sp_agent = train_selfplay_rps(training_env, num_episodes=selfplay_episodes, input_dim=input_dim, output_dim=output_dim, device=torch.device('cpu'))
    training_times["SelfPlay"] = float(time.time() - _t0)
    sp_dir = os.path.join(base_results_dir, "SelfPlay")
    os.makedirs(sp_dir, exist_ok=True)
    torch.save(sp_agent.state_dict(), os.path.join(sp_dir, "model.pt"))

    # Train PRPO with superior unified implementation and time budget
    print(f"\n--- Training PRPO Unified ({time_budget_seconds}s per seed x {num_seeds} seeds) ---")
    prpo_runs: List[nn.Module] = []
    _t0 = time.time()
    for seed in range(num_seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        prpo_tr = train_prpo_rps_unified(time_budget_seconds, rps_input_dim, rps_output_dim, torch.device('cpu'))
        prpo_runs.append(prpo_tr)
        print(f"    PRPO Seed {seed}: Training completed in {time_budget_seconds}s")
    prpo_agent = prpo_runs[-1]
    training_times["PRPO"] = float(time.time() - _t0)
    prpo_dir = os.path.join(base_results_dir, "PRPO")
    os.makedirs(prpo_dir, exist_ok=True)
    torch.save(prpo_agent.policy.state_dict(), os.path.join(prpo_dir, "model.pt"))

    # Statistical evaluation across seeds vs a fixed opponent
    print("\n--- Computing seed-wise scores, 95% CI, and paired t-test (RPS PRPO vs PPO) ---")
    eval_env = RPSEnvironment()
    fixed_opp = RuleBasedAgent(name="EvalOpponent")
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

    with open(os.path.join(base_results_dir, "rps_stats.json"), "w") as f:
        json.dump({
            "ppo_scores": ppo_scores,
            "prpo_scores": prpo_scores,
            "ppo_stats": ppo_stats,
            "prpo_stats": prpo_stats,
            "p_value_prpo_vs_ppo": p_value,
        }, f, indent=2)

    # Evaluate each policy using a provided benchmark instance (reuse across calls)
    def evaluate_and_report(g: "EnhancedGauntletBenchmark", policy: nn.Module, name: str, out_dir: str):
        # Wrap all policies except raw DQN/PSRO/SelfPlay 
        # StandardPPO, UnifiedPRPOAgent, and UnifiedActorCritic all need wrapping
        use_wrapped = isinstance(policy, (StandardPPO, UnifiedPRPOAgent, UnifiedActorCritic)) or hasattr(policy, 'policy')
        # For PPO and PRPO, use RPS dimensions since they were trained on RPSEnvironment
        if name in ["RPS_PPO", "RPS_PRPO"]:
            eval_input_dim, eval_output_dim = rps_input_dim, rps_output_dim
        else:
            eval_input_dim, eval_output_dim = input_dim, output_dim
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
            stats_path = os.path.join(base_dir, "rps_stats.json")
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

    # Prepare a shared benchmark instance for evaluation and register custom challenger once
    gauntlet_eval = EnhancedGauntletBenchmark(config)
    gauntlet_eval.add_custom_challenger("Student_RuleBased", RuleBasedAgent())

    _t0 = time.time(); evaluate_and_report(gauntlet_eval, trained_dqn_agent, "RPS_DQN", dqn_dir); eval_times["DQN"] = float(time.time() - _t0)
    _t0 = time.time(); evaluate_and_report(gauntlet_eval, trained_ppo_agent, "RPS_PPO", ppo_dir); eval_times["PPO"] = float(time.time() - _t0)
    _t0 = time.time(); evaluate_and_report(gauntlet_eval, psro_policy, "RPS_PSRO", psro_dir); eval_times["PSRO"] = float(time.time() - _t0)
    _t0 = time.time(); evaluate_and_report(gauntlet_eval, sp_agent, "RPS_SelfPlay", sp_dir); eval_times["SelfPlay"] = float(time.time() - _t0)
    _t0 = time.time(); evaluate_and_report(gauntlet_eval, prpo_agent, "RPS_PRPO", prpo_dir); eval_times["PRPO"] = float(time.time() - _t0)

    # Save timing summary
    with open(os.path.join(base_results_dir, "times.json"), "w") as f:
        json.dump({"training_seconds": training_times, "eval_seconds": eval_times}, f, indent=2)

    print("\n\n🎉 RPS training and evaluation complete. Outputs saved under 'results/RPS/'. 🎉")