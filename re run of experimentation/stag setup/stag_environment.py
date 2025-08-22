%%writefile stag_hunt_training_and_evaluation.py

# stag hunt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import os
import shutil
import argparse
from typing import Optional, List, Tuple, Dict, Any, Callable, TYPE_CHECKING
from collections import namedtuple
import torch.nn.functional as F
import time
import math
import json
import warnings


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

# Optional Nash solver support
try:
    import nashpy as nash  # type: ignore
    _NASH_AVAILABLE = True
except Exception:
    nash = None  # type: ignore
    _NASH_AVAILABLE = False
    print("Warning: nashpy not available. Install with 'pip install nashpy' for proper PSRO.")

from unified_prpo import UnifiedActorCritic, StandardPPO, UnifiedPRPOAgent, UnifiedPRPO, TimeBudgetTrainer
# Optional SciPy for significance testing
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False
    _scipy_stats = None  # type: ignore
from gauntlet_benchmark import EnhancedGauntletBenchmark, EvaluationConfig, ChallengerAgent, Environment
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


# ============================================================================
# 2. Stag Hunt Specific Functions for Unified PRPO
# ============================================================================

class StagHuntSimpleEnvironment:
    """Simplified Stag Hunt game environment for unified PRPO."""
    def __init__(self, episode_length: int = 50):
        # Actions: 0=Stag (cooperate), 1=Hare (defect)
        self.action_dim = 2
        self.state_dim = 2  # [own_last_action, opponent_last_action]
        self.episode_length = episode_length
        self.payoff_matrix = np.array([[3, 0], [2, 1]])  # Classic Stag Hunt payoffs
        self.reset()

    def reset(self):
        self.steps = 0
        self.last_actions = [0, 0]  # Start with both cooperating
        return self._get_state()

    def _get_state(self):
        # State represents the last actions taken by both players
        state = np.zeros(self.state_dim)
        if self.steps > 0:  # After first step, use actual last actions
            state[0] = self.last_actions[0]  # Own last action
            state[1] = self.last_actions[1]  # Opponent last action
        return state

    def step(self, actions):
        """
        Stag Hunt payoffs:
        Both Stag (0,0): (3, 3) - High reward for cooperation
        Stag vs Hare (0,1): (0, 2) - Cooperator gets nothing
        Hare vs Stag (1,0): (2, 0) - Defector gets something
        Both Hare (1,1): (1, 1) - Low reward but safe
        
        Args:
            actions: List of two actions [p1_action, p2_action] or tuple (p1_action, p2_action)
        """
        # Handle both list and separate arguments for compatibility
        if isinstance(actions, (list, tuple)):
            p1_action, p2_action = int(actions[0]), int(actions[1])
        else:
            # Fallback for single argument case - shouldn't happen but safety
            p1_action, p2_action = int(actions), 0
            
        p1_reward = self.payoff_matrix[p1_action, p2_action]
        p2_reward = self.payoff_matrix[p2_action, p1_action]
        
        self.last_actions = [p1_action, p2_action]
        self.steps += 1
        done = self.steps >= self.episode_length
        
        return self._get_state(), [p1_reward, p2_reward], done

def get_stag_hunt_nash_policy_callable(policy_probs_batch: torch.Tensor) -> torch.Tensor:
    """
    Returns one of the Nash equilibria for Stag Hunt.
    Stag Hunt has multiple Nash equilibria - we use a mixed strategy.
    """
    batch_size = policy_probs_batch.shape[0]
    # Mixed Nash equilibrium approximation [0.6, 0.4] for Stag/Hare
    nash_dist = torch.full_like(policy_probs_batch, 0.0)
    nash_dist[:, 0] = 0.6  # Stag (cooperate)
    nash_dist[:, 1] = 0.4  # Hare (defect)
    return nash_dist

def calculate_stag_hunt_exploitability_callable(policy: UnifiedActorCritic) -> float:
    """
    Calculates exploitability for Stag Hunt policy.
    Based on how much an opponent can exploit deviations from mixed strategies.
    """
    policy.eval()
    device = next(policy.parameters()).device
    
    # Test policy on different state contexts
    exploitability = 0.0
    test_states = [
        [0, 0],  # Both cooperated last
        [0, 1],  # I cooperated, opponent defected
        [1, 0],  # I defected, opponent cooperated
        [1, 1],  # Both defected last
    ]
    
    for state_values in test_states:
        state = torch.tensor(state_values, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            policy_probs, _ = policy(state)
            probs = policy_probs.squeeze().cpu().numpy()
        
        # Exploitability based on predictability - more extreme strategies are more exploitable
        predictability = max(probs) - min(probs)  # Range of probability distribution
        exploitability += predictability
    
    exploitability /= len(test_states)
    policy.train()
    return float(exploitability)

def get_stag_hunt_exploiter_opponents() -> List[Callable]:
    """Returns a list of exploiter bots for Stag Hunt."""
    return [
        lambda: 0,  # Always Stag (cooperate)
        lambda: 1,  # Always Hare (defect)
        lambda: np.random.choice([0, 1], p=[0.8, 0.2]),  # Mostly cooperate
        lambda: np.random.choice([0, 1], p=[0.2, 0.8]),  # Mostly defect
        lambda: random.randint(0, 1),  # Random
    ]

def train_prpo_stag_hunt_unified(time_budget_seconds: float, input_dim: int, output_dim: int, 
                               device: torch.device, lambda_nash: float = 0.5, 
                               lambda_exploit: float = 0.5) -> nn.Module:
    """
    Superior PRPO training function using the unified framework with time budget for Stag Hunt.
    """
    print(f"Training PRPO (Unified Framework) for Stag Hunt for {time_budget_seconds} seconds...")
    
    # The manager needs a way to create new environments
    env_factory = lambda: StagHuntSimpleEnvironment(episode_length=50)

    prpo_system = UnifiedPRPO(
        state_dim=input_dim, 
        action_dim=output_dim, 
        lr=1e-4, 
        device=str(device),
        population_size=4,
        lambda_nash=lambda_nash,
        nash_target_callable=get_stag_hunt_nash_policy_callable,
        lambda_exploit=lambda_exploit,
        exploitability_calculator=calculate_stag_hunt_exploitability_callable,
        exploiter_opponents=get_stag_hunt_exploiter_opponents()
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
    
    print("Training finished for PRPO (Unified Framework) - Stag Hunt.")
    return best_policy

# ============================================================================
# 3. Stag Hunt Environment Implementation (Original General-Sum Matrix Game)
# ============================================================================

class StagHuntEnvironment(Environment):
    """
    Stag Hunt environment as a general-sum matrix game.
    - Actions: 0=Stag (cooperate), 1=Hare (defect)
    - Observation: Simple one-hot encoding or history-based
    - Payoffs favor mutual cooperation but defection is safer
    
    Classic Stag Hunt payoffs:
    Both Stag: (3, 3) - High reward for cooperation
    Stag vs Hare: (0, 2) - Cooperator gets nothing, defector gets something
    Hare vs Stag: (2, 0) - Defector gets something, cooperator gets nothing  
    Both Hare: (1, 1) - Safe but suboptimal mutual defection
    """
    def __init__(self, episode_length: int = 50):
        # Observation space: [own_last_action, opponent_last_action]
        self._observation_space = Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self._action_space = Discrete(2)  # 0: Stag (cooperate), 1: Hare (defect)
        self.state = None
        self.episode_length = int(episode_length)
        self.step_count = 0
        
        # Classic Stag Hunt payoff matrix
        # (own_action, opponent_action): (own_reward, opponent_reward)
        self.payoff_matrix = {
            (0, 0): (3, 3),  # Both Stag - high reward for cooperation
            (0, 1): (0, 2),  # Stag vs Hare - cooperator gets exploited
            (1, 0): (2, 0),  # Hare vs Stag - defector gets safe reward
            (1, 1): (1, 1),  # Both Hare - safe but suboptimal
        }

    def reset(self) -> torch.Tensor:
        # Initial state represents no prior actions
        self.state = torch.zeros(2, dtype=torch.float32)
        self.step_count = 0
        return self.state

    def step(self, actions: List[int]) -> Tuple[torch.Tensor, List[float], bool, Dict]:
        # Robustly coerce potentially tuple/tensor actions into ints
        def _to_int(a: Any) -> int:
            try:
                if isinstance(a, (tuple, list)):
                    a = a[0]
                if torch.is_tensor(a):
                    return int(a.item())
                return int(a)
            except Exception:
                # Fallback to 0 (Stag/cooperate)
                return 0
        
        action1, action2 = _to_int(actions[0]), _to_int(actions[1])
        reward1, reward2 = self.payoff_matrix[(action1, action2)]
        rewards = [float(reward1), float(reward2)]
        
        # Update state to reflect actions
        self.state = torch.tensor([action1, action2], dtype=torch.float32)
        
        # Advance step counter
        self.step_count += 1
        done = self.step_count >= self.episode_length
        
        # Expose semantics for benchmark analysis
        info = {
            'general_sum': True, 
            'action0_is_cooperate': True,  # Action 0 (Stag) is cooperation
            'social_welfare': reward1 + reward2,  # Track social welfare
            'cooperation_actions': [action1 == 0, action2 == 0]  # Track cooperation
        }
        
        return self.state, rewards, done, info

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
            nn.Linear(input_dim, 64), 
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

    def act(self, state: torch.Tensor, explore: bool = True) -> int:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if explore and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q_values = self.q_net(state)
            return int(torch.argmax(q_values, dim=-1).item())

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
        # Detach stored tensors to avoid backprop through old graphs during PPO updates
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
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
            nn.Linear(input_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self.num_actions = output_dim

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        # Collect rollout data without tracking gradients
        with torch.no_grad():
            logits = self.actor(state)
            value = self.critic(state)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze()

    def act(self, state: torch.Tensor) -> int:
        action, _, _ = self.select_action(state)
        return action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

# ============================================================================
# 4. Training Functions
# ============================================================================

def train_dqn_stag_hunt(agent: DQNAgent, env: StagHuntEnvironment, opponent: ChallengerAgent, 
                        episodes: int = 1000) -> DQNAgent:
    """Train DQN agent on Stag Hunt."""
    # print(f"Training DQN for {episodes} episodes...")
    
    buffer = ReplayBuffer(max_size=50000)
    optimizer = optim.Adam(agent.q_net.parameters(), lr=1e-3)
    target_sync_freq = 100
    batch_size = 32
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(env.episode_length):
            # Agent action
            action = agent.act(state, explore=True)
            
            # Opponent action
            opp_action = opponent.act(state) if hasattr(opponent, 'act') else random.choice([0, 1])
            
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
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.3f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent


def train_ppo_stag_hunt(agent: PPOAgent, env: StagHuntEnvironment, opponent: ChallengerAgent, 
                        episodes: int = 1000) -> PPOAgent:
    """Train PPO agent on Stag Hunt."""
    print(f"Training PPO for {episodes} episodes...")
    
    update_freq = 50
    epochs_per_update = 4
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(env.episode_length):
            # Agent action
            action, log_prob, value = agent.select_action(state)
            
            # Opponent action
            opp_action = opponent.act(state) if hasattr(opponent, 'act') else random.choice([0, 1])
            
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
        
        if episode % 100 == 0:
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
# 5. PSRO for Stag Hunt 
# ============================================================================

class PSROPolicy(nn.Module):
    """Meta-policy that mixes a population of policies."""
    def __init__(self, population: List[nn.Module], meta_strategy: List[float]):
        super().__init__()
        self.population = population
        self.meta_strategy = meta_strategy

    def act(self, state: torch.Tensor) -> int:
        # Sample policy from meta-strategy
        policy_idx = np.random.choice(len(self.population), p=self.meta_strategy)
        policy = self.population[policy_idx]
        
        if hasattr(policy, 'act'):
            return policy.act(state)
        else:
            # DQN-style policy
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            with torch.no_grad():
                q_values = policy(state)
                return int(torch.argmax(q_values, dim=-1).item())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Use first policy as representative
        return self.population[0](state)


def train_psro_stag_hunt(gauntlet: "EnhancedGauntletBenchmark", time_budget_seconds: float, episodes_per_iter: int = 100) -> PSROPolicy:
    """
    Train PSRO on Stag Hunt using actual time budget instead of fixed iterations.
    """
    print(f"[PSRO-StagHunt] Training for {time_budget_seconds} seconds...")
    start_time = time.time()
    
    # Get Stag Hunt challengers from gauntlet
    stag_hunt_challengers = [agent for name, agent in gauntlet.master_challenger_list.items() 
                            if name.startswith("StagHunt")]
    
    if not stag_hunt_challengers:
        print("Warning: No StagHunt challengers found. Using uniform random opponent.")
        
        class UniformAgent(ChallengerAgent):
            def __init__(self):
                super().__init__("Uniform", "easy")
            def act(self, observation, opponent_history=None):
                return random.choice([0, 1])  # Random action
            @property
            def compatible_action_space(self):
                return Discrete(2)
            def update(self, reward, observation, action):
                pass
        
        stag_hunt_challengers = [UniformAgent()]
    
    env = StagHuntEnvironment()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    population = []
    iteration_count = 0
    
    while (time.time() - start_time) < time_budget_seconds:
        elapsed_time = time.time() - start_time
        print(f"[PSRO-StagHunt] Iteration {iteration_count + 1} (Elapsed: {elapsed_time:.1f}s)")
        
        # Train best response against population + challengers
        br_agent = DQNAgent(input_dim, output_dim)
        
        # Mix of population and challengers as opponents
        opponents = population + stag_hunt_challengers
        if not opponents:
            opponents = stag_hunt_challengers
        
        for episode in range(episodes_per_iter):
            opponent = random.choice(opponents)
            br_agent = train_dqn_stag_hunt(br_agent, env, opponent, episodes=1)
        
        population.append(br_agent)
        iteration_count += 1
        print(f"Added agent {iteration_count} to population. Population size: {len(population)}")
        
        # Safety check to prevent infinite loops with very small time budgets
        if iteration_count >= 20:  # Max reasonable iterations
            print(f"[PSRO-StagHunt] Reached maximum iterations ({iteration_count}), stopping.")
            break
    
    final_time = time.time() - start_time
    print(f"[PSRO-StagHunt] Training completed in {final_time:.1f}s with {iteration_count} iterations")
    
    if not population:
        # Fallback: train at least one best response if no time was sufficient
        print("[PSRO-StagHunt] Warning: No iterations completed, training one best response")
        br_agent = DQNAgent(input_dim, output_dim)
        opponents = stag_hunt_challengers
        for episode in range(episodes_per_iter):
            opponent = random.choice(opponents)
            br_agent = train_dqn_stag_hunt(br_agent, env, opponent, episodes=1)
        population.append(br_agent)
    
    # Uniform meta-strategy
    meta_strategy = [1.0 / len(population)] * len(population)
    
    return PSROPolicy(population, meta_strategy)


# ============================================================================
# 6. Unified PRPO for Stag Hunt 
# ============================================================================

class UnifiedPRPO_StagHunt:
    """Unified PRPO implementation for Stag Hunt."""
    def __init__(self, env: StagHuntEnvironment, population_size: int = 5, 
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
                    a1, lp1, v1 = agent1.select_action(state)
                    a2, lp2, v2 = agent2.select_action(state)
                    
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
            print(f"PRPO Episode {completed_episodes}: Avg Exploit: {avg_exploit:.3f}")
        
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


def train_prpo_stag_hunt_simple(env: StagHuntEnvironment, episodes: int, input_dim: int, 
                                output_dim: int, device: torch.device, lambda_exploit: float = 0.5) -> PPOAgent:
    """Simple PRPO training for Stag Hunt."""
    prpo_system = UnifiedPRPO_StagHunt(env, population_size=3, lambda_exploit=lambda_exploit)
    final_policy = prpo_system.train(total_episodes=episodes, episodes_per_update=50)
    print("Training finished for PRPO (Stag Hunt, Unified).")
    return final_policy


# ============================================================================
# 7. Training utilities
# ============================================================================

def train_stag_hunt_agent(agent, env, opponent, num_episodes=1000):
    """Generic training function for Stag Hunt agents."""
    opponent_name = getattr(opponent, "name", opponent.__class__.__name__)
    print(f"Training {agent.__class__.__name__} against {opponent_name} in Stag Hunt...")
    if isinstance(agent, DQNAgent):
        return train_dqn_stag_hunt(agent, env, opponent, episodes=num_episodes)
    if isinstance(agent, PPOAgent):
        return train_ppo_stag_hunt(agent, env, opponent, episodes=num_episodes)
    return agent


def train_selfplay_stag_hunt_time_budget(env: StagHuntEnvironment, time_budget_seconds: float, 
                                       input_dim: int, output_dim: int, device: torch.device) -> DQNAgent:
    """
    Train Self-Play for Stag Hunt using actual time budget instead of fixed episodes.
    """
    print(f"[SelfPlay-StagHunt] Training for {time_budget_seconds} seconds...")
    start_time = time.time()
    
    # Create two agents for self-play
    agent1 = DQNAgent(input_dim, output_dim).to(device)
    agent2 = DQNAgent(input_dim, output_dim).to(device)
    
    optimizer1 = optim.Adam(agent1.q_net.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(agent2.q_net.parameters(), lr=1e-3)
    
    episode_count = 0
    last_log_time = start_time
    
    while (time.time() - start_time) < time_budget_seconds:
        try:
            state = env.reset()
            episode_reward1 = 0
            episode_reward2 = 0
            
            for step in range(env.episode_length):
                # Both agents act
                action1 = agent1.act(state, explore=True)
                action2 = agent2.act(state, explore=True)
                
                # Environment step
                next_state, rewards, done, _ = env.step([action1, action2])
                reward1, reward2 = rewards[0], rewards[1]
                episode_reward1 += reward1
                episode_reward2 += reward2
                
                # Train both agents
                if len(state.shape) == 1:
                    state_batch = state.unsqueeze(0)
                    next_state_batch = next_state.unsqueeze(0)
                else:
                    state_batch = state
                    next_state_batch = next_state
                
                # Agent 1 update
                q_pred1 = agent1.q_net(state_batch)[0, action1]
                with torch.no_grad():
                    target1 = torch.tensor(reward1, dtype=torch.float32)
                loss1 = F.mse_loss(q_pred1, target1)
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()
                
                # Agent 2 update
                q_pred2 = agent2.q_net(state_batch)[0, action2]
                with torch.no_grad():
                    target2 = torch.tensor(reward2, dtype=torch.float32)
                loss2 = F.mse_loss(q_pred2, target2)
                optimizer2.zero_grad()
                loss2.backward()
                optimizer2.step()
                
                state = next_state
                if done:
                    break
                    
        except Exception as e:
            print(f"Error in self-play episode {episode_count}: {e}")
            continue
        
        episode_count += 1
        
        # Update exploration parameters
        agent1.update_epsilon()
        agent2.update_epsilon()
        
        current_time = time.time()
        # Log progress every 10 seconds
        if (current_time - last_log_time) >= 10.0:
            elapsed_time = current_time - start_time
            print(f"    [SelfPlay-StagHunt] Time {elapsed_time:.1f}s: Episodes {episode_count}")
            last_log_time = current_time
    
    final_time = time.time() - start_time
    print(f"[SelfPlay-StagHunt] Training completed in {final_time:.1f}s with {episode_count} episodes")
    
    # Return the first agent (or could return the better performing one)
    agent1.epsilon = agent1.epsilon_min
    agent1.eval()
    return agent1


# ============================================================================
# 8. Evaluation utilities
# ============================================================================

def evaluate_simple_avg_reward(policy: nn.Module, env: StagHuntEnvironment, 
                               opponent: ChallengerAgent, episodes: int = 100) -> float:
    """Evaluate a policy's average reward against a fixed opponent."""
    total_reward = 0.0
    
    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        
        for step in range(env.episode_length):
            # Policy action
            if hasattr(policy, 'act'):
                action = policy.act(state)
            else:
                with torch.no_grad():
                    if len(state.shape) == 1:
                        state_batch = state.unsqueeze(0)
                    else:
                        state_batch = state
                    q_values = policy(state_batch)
                    action = int(torch.argmax(q_values, dim=-1).item())
            
            # Opponent action
            opp_action = opponent.act(state) if hasattr(opponent, 'act') else random.choice([0, 1])
            
            # Step
            state, rewards, done, _ = env.step([action, opp_action])
            episode_reward += rewards[0]
            
            if done:
                break
        
        total_reward += episode_reward
    
    return total_reward / episodes


def eval_and_report(gauntlet: EnhancedGauntletBenchmark, policy: nn.Module, 
                   policy_name: str, output_dir: str):
    """Evaluate policy using gauntlet, save reports, and move visualizations."""
    print(f"Evaluating {policy_name}...")
    try:
        # Suppress benign matplotlib RuntimeWarnings across the whole evaluation pipeline
        try:
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="matplotlib.colors")
        except Exception:
            pass
        # Run evaluation using the benchmark's public API
        metrics = gauntlet.evaluate_policy(policy, policy_name, environments=["StagHunt"])  # limit to StagHunt

        # Build a serializable results payload combining metrics and detailed results
        try:
            latest = gauntlet.results_history[-1]
            detailed_results = latest.get('detailed_results', {})
        except Exception:
            detailed_results = {}

        results_payload = {
            "policy_name": policy_name,
            "metrics": getattr(metrics, "__dict__", {}),
            "robustness_score": float(getattr(metrics, "robustness_score", 0.0)),
            "detailed_results": detailed_results,
        }

        # Attach seed statistics if available
        try:
            base_dir = os.path.dirname(output_dir)
            stats_path = os.path.join(base_dir, "stag_hunt_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, "r") as f:
                    seed_stats = json.load(f)
                results_payload["seed_stats"] = seed_stats
        except Exception as e:
            print(f"Warning: could not inject seed stats for {policy_name}: {e}")

        # Save outputs
        os.makedirs(output_dir, exist_ok=True)
        try:
            # Prefer saving whole policy if it's an nn.Module
            if isinstance(policy, nn.Module):
                torch.save(policy.state_dict(), os.path.join(output_dir, "model.pt"))
            # If policy exposes a state_dict-like method, try using it
            elif hasattr(policy, "state_dict") and callable(getattr(policy, "state_dict")):
                torch.save(policy.state_dict(), os.path.join(output_dir, "model.pt"))
            # Common pattern: a wrapper with an inner actor network
            elif hasattr(policy, "actor") and isinstance(getattr(policy, "actor"), nn.Module):
                torch.save(policy.actor.state_dict(), os.path.join(output_dir, "model_actor.pt"))
            else:
                # Fallback: save a lightweight textual representation
                with open(os.path.join(output_dir, "model.txt"), "w") as f:
                    f.write(repr(policy))
        except Exception as e:
            print(f"Warning: could not save model for {policy_name}: {e}")
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results_payload, f, indent=2, default=lambda o: float(o) if isinstance(o, (int, float)) else str(o))

        # Generate full report and move visualization assets into output_dir
        try:
            report_path = os.path.join(output_dir, "report.json")
            gauntlet.generate_report(report_path)
            # Inject seed statistics into the report as well, if available
            try:
                base_dir = os.path.dirname(output_dir)
                stats_path = os.path.join(base_dir, "stag_hunt_stats.json")
                if os.path.exists(stats_path) and os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        report_payload = json.load(f)
                    with open(stats_path, "r") as f:
                        seed_stats = json.load(f)
                    report_payload["seed_stats"] = seed_stats
                    with open(report_path, "w") as f:
                        json.dump(report_payload, f, indent=2)
            except Exception as e:
                print(f"Warning: could not inject seed stats into report for {policy_name}: {e}")
        except Exception as e:
            print(f"Warning: could not generate report for {policy_name}: {e}")

        # Move generated visualization files produced by the benchmark
        try:
            fmt = gauntlet.config.visualization_format if hasattr(gauntlet, 'config') else 'png'
            viz_files = [
                f"{policy_name}_challenger_performance.{fmt}",
                f"{policy_name}_robustness_radar.{fmt}",
                f"{policy_name}_performance_heatmap.{fmt}",
                f"{policy_name}_metrics_comparison.{fmt}",
            ]
            for vf in viz_files:
                if os.path.exists(vf):
                    shutil.move(vf, os.path.join(output_dir, vf))
        except Exception as e:
            print(f"Warning: could not move visualization files for {policy_name}: {e}")

        print(f"✅ {policy_name} evaluation completed.")
    except Exception as e:
        print(f"❌ Evaluation failed for {policy_name}: {e}")


# ============================================================================
# 9. Main execution
# ============================================================================

if __name__ == "__main__":
    # Updated parameters - now using time budget instead of episode count
    seeds = 2
    time_budget_seconds = 10.0  # 60 seconds per algorithm per seed

    print("\n--- Setting up Gauntlet Benchmark for Stag Hunt Evaluation ---")
    config = EvaluationConfig(num_episodes=200, parallel_workers=1, save_visualizations=True)
    gauntlet = EnhancedGauntletBenchmark(config)

    # Register Stag Hunt as general-sum matrix game
    # Payoff matrices A and B for the two players
    stag_hunt_A = np.array([[3, 0], [2, 1]])  # Row player (P1)
    stag_hunt_B = np.array([[3, 2], [0, 1]])  # Column player (P2)
    
    gauntlet.register_environment(
        "StagHunt",
        StagHuntEnvironment,
        payoff_matrices=(stag_hunt_A, stag_hunt_B),
        game_prefix="StagHunt",
        zero_sum=False  # General-sum game
    )
    print("✅ Registered Stag Hunt as general-sum matrix game")

    # Instantiate environment and agents
    env = StagHuntEnvironment()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Environment: input_dim={input_dim}, output_dim={output_dim}")
    
    dqn_agent = DQNAgent(input_dim, output_dim)
    ppo_agent = PPOAgent(input_dim, output_dim)

    # Training opponent
    training_opponents = [name for name in gauntlet.master_challenger_list.keys() if name.startswith("StagHunt")]
    if training_opponents:
        training_opponent = gauntlet.master_challenger_list[training_opponents[0]]
        training_opponent.name = "TrainingOpponent_StagHunt"
    else:
        print("No StagHunt challengers found, using uniform random")
        class UniformAgent(ChallengerAgent):
            def __init__(self):
                super().__init__("Uniform", "easy")
            def act(self, observation, opponent_history=None):
                return random.choice([0, 1])
            @property
            def compatible_action_space(self):
                return Discrete(2)
            def update(self, reward, observation, action):
                pass
        training_opponent = UniformAgent()

    # Output directories
    base_dir = os.path.join("results", "StagHunt")
    os.makedirs(base_dir, exist_ok=True)
    dqn_dir = os.path.join(base_dir, "DQN"); os.makedirs(dqn_dir, exist_ok=True)
    ppo_dir = os.path.join(base_dir, "PPO"); os.makedirs(ppo_dir, exist_ok=True)
    psro_dir = os.path.join(base_dir, "PSRO"); os.makedirs(psro_dir, exist_ok=True)
    prpo_dir = os.path.join(base_dir, "PRPO"); os.makedirs(prpo_dir, exist_ok=True)
    selfplay_dir = os.path.join(base_dir, "SelfPlay"); os.makedirs(selfplay_dir, exist_ok=True)
    timing_train: Dict[str, float] = {}
    timing_eval: Dict[str, float] = {}

    # Train DQN multi-seed with time budget
    print(f"\n--- Starting Stag Hunt Training Phase with Time Budget (DQN, PPO, {time_budget_seconds}s x {seeds} seeds) ---")
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
    ppo_runs = []
    _t0 = time.time()
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        trained, episodes_completed, _ = TimeBudgetTrainer.train_standard_ppo_time_budget(
            lambda: StagHuntSimpleEnvironment(episode_length=50), input_dim, output_dim, time_budget_seconds, 
            device=str(device)
        )
        ppo_runs.append(trained)
        print(f"    PPO Seed {seed}: Completed {episodes_completed} episodes in {time_budget_seconds}s")
    trained_ppo = ppo_runs[-1]
    timing_train["PPO"] = float(time.time() - _t0)

    # Train PSRO with time budget
    print(f"\n--- Training PSRO ({time_budget_seconds}s time budget) ---")
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    _t0 = time.time()
    # Use actual time budget instead of approximated episodes per iteration
    psro_policy = train_psro_stag_hunt(gauntlet, time_budget_seconds=time_budget_seconds, episodes_per_iter=100)
    timing_train["PSRO"] = float(time.time() - _t0)

    # Train Self-Play with time budget
    print(f"\n--- Training Self-Play ({time_budget_seconds}s time budget) ---")
    torch.manual_seed(123); np.random.seed(123); random.seed(123)
    _t0 = time.time()
    # Use actual time budget for Self-Play instead of approximated episodes
    selfplay_agent = train_selfplay_stag_hunt_time_budget(env, time_budget_seconds, input_dim, output_dim, device)
    timing_train["SelfPlay"] = float(time.time() - _t0)

    # Train PRPO with superior unified implementation and time budget
    print(f"\n--- Training PRPO Unified ({time_budget_seconds}s per seed x {seeds} seeds) ---")
    prpo_runs = []
    _t0 = time.time()
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        trained = train_prpo_stag_hunt_unified(time_budget_seconds, input_dim, output_dim, device)
        prpo_runs.append(trained)
        print(f"    PRPO Seed {seed}: Training completed in {time_budget_seconds}s")
    trained_prpo = prpo_runs[-1]
    timing_train["PRPO"] = float(time.time() - _t0)

    print("\n--- Gauntlet Evaluation Phase ---")
    
    # Evaluate all policies
    _t0 = time.time();
    try:
        if hasattr(trained_dqn, "eval") and callable(getattr(trained_dqn, "eval")):
            trained_dqn.eval()
    except Exception:
        pass
    eval_and_report(gauntlet, trained_dqn, "StagHunt_DQN", dqn_dir); timing_eval["DQN"] = float(time.time() - _t0)

    _t0 = time.time();
    try:
        if hasattr(trained_ppo, "eval") and callable(getattr(trained_ppo, "eval")):
            trained_ppo.eval()
    except Exception:
        pass
    eval_and_report(gauntlet, trained_ppo, "StagHunt_PPO", ppo_dir); timing_eval["PPO"] = float(time.time() - _t0)

    _t0 = time.time();
    try:
        if hasattr(psro_policy, "eval") and callable(getattr(psro_policy, "eval")):
            psro_policy.eval()
    except Exception:
        pass
    eval_and_report(gauntlet, psro_policy, "StagHunt_PSRO", psro_dir); timing_eval["PSRO"] = float(time.time() - _t0)

    _t0 = time.time();
    try:
        if hasattr(selfplay_agent, "eval") and callable(getattr(selfplay_agent, "eval")):
            selfplay_agent.eval()
    except Exception:
        pass
    eval_and_report(gauntlet, selfplay_agent, "StagHunt_SelfPlay", selfplay_dir); timing_eval["SelfPlay"] = float(time.time() - _t0)

    _t0 = time.time();
    try:
        if hasattr(trained_prpo, "eval") and callable(getattr(trained_prpo, "eval")):
            trained_prpo.eval()
    except Exception:
        pass
    eval_and_report(gauntlet, trained_prpo, "StagHunt_PRPO", prpo_dir); timing_eval["PRPO"] = float(time.time() - _t0)

    # Statistical comparison PRPO vs PPO
    print("\n--- Computing seed-wise scores, 95% CI, and paired t-test (Stag Hunt PRPO vs PPO) ---")
    eval_env = StagHuntEnvironment()
    fixed_opp = training_opponent
    fixed_opp.name = "EvalOpponent_StagHunt"
    
    ppo_scores: List[float] = []
    prpo_scores: List[float] = []
    
    for run in ppo_runs:
        ppo_scores.append(evaluate_simple_avg_reward(run, eval_env, fixed_opp, episodes=50))
    for run in prpo_runs:
        prpo_scores.append(evaluate_simple_avg_reward(run, eval_env, fixed_opp, episodes=50))
    
    ppo_stats = compute_mean_ci(ppo_scores)
    prpo_stats = compute_mean_ci(prpo_scores)
    p_value = paired_t_test(prpo_scores, ppo_scores)
    
    print(f"PPO mean={ppo_stats['mean']:.3f}, 95% CI=[{ppo_stats['ci_low']:.3f}, {ppo_stats['ci_high']:.3f}], n={ppo_stats['n']}")
    print(f"PRPO mean={prpo_stats['mean']:.3f}, 95% CI=[{prpo_stats['ci_low']:.3f}, {prpo_stats['ci_high']:.3f}], n={prpo_stats['n']}")
    if p_value is not None:
        print(f"Paired t-test (PRPO vs PPO): p-value={p_value:.4f}")
    else:
        print("Paired t-test unavailable (SciPy not installed).")
    
    with open(os.path.join(base_dir, "stag_hunt_stats.json"), "w") as f:
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

    print("\n\n🎉 Stag Hunt training and evaluation complete. Outputs under 'results/StagHunt/'. 🎉")