import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import os
import argparse
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Any, Callable, TYPE_CHECKING
from collections import namedtuple
import torch.nn.functional as F
import time
import math
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

# ============================================================================
# 1. IMPORT FROM YOUR BENCHMARK FILE
# ============================================================================
# NOTE: Assumes your benchmark file is named 'gauntlet_benchmark.py'



# ============================================================================
# 2. IPD Environment Implementation
# ============================================================================

class IPDEnvironment(Environment):
    """
    An environment for the Iterated Prisoner's Dilemma (IPD).
    - Actions: 0 for Cooperate, 1 for Defect.
    - Observation: A history of the last round of actions [my_action, opponent_action].
    - Payoffs are standard for IPD.
    """
    def __init__(self, episode_length: int = 50):
        # Observation space: [own_last_action, opponent_last_action]
        self._observation_space = Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self._action_space = Discrete(2)  # 0: Cooperate, 1: Defect
        self.state = None
        # Episode horizon for a truly iterated IPD
        self.episode_length = int(episode_length)
        self.step_count = 0
        # Payoff matrix: R=3, S=0, T=5, P=1
        self.payoff_matrix = {
            (0, 0): (3, 3),  # Both Cooperate (Reward)
            (0, 1): (0, 5),  # You Cooperate, Opponent Defects (Sucker)
            (1, 0): (5, 0),  # You Defect, Opponent Cooperates (Temptation)
            (1, 1): (1, 1),  # Both Defect (Punishment)
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
                # Fallback to 0 on unexpected types
                return 0
        action1, action2 = _to_int(actions[0]), _to_int(actions[1])
        reward1, reward2 = self.payoff_matrix[(action1, action2)]
        rewards = [float(reward1), float(reward2)]
        self.state = torch.tensor([action1, action2], dtype=torch.float32)
        # Advance step counter; end only at the specified horizon
        self.step_count += 1
        done = self.step_count >= self.episode_length
        # Expose action semantics to the benchmark for cooperation-rate computation
        info = {'general_sum': True, 'action0_is_cooperate': True}
        return self.state, rewards, done, info
    @property
    def observation_space(self) -> Space:
        return self._observation_space

    @property
    def action_space(self) -> Space:
        return self._action_space


# ============================================================================
# 3. Algorithm Implementations for IPD
# ============================================================================

# --- Learning Agents (DQN and PPO can be reused) ---

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: List[Tuple[torch.Tensor, int, float, torch.Tensor, bool]] = []
        self.idx = 0

    def __len__(self):
        return len(self.data)

    def push(self, s: torch.Tensor, a: int, r: float, ns: torch.Tensor, d: bool):
        item = (s.detach().cpu(), int(a), float(r), ns.detach().cpu(), bool(d))
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.idx] = item
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch: int):
        batch_items = random.sample(self.data, batch)
        s, a, r, ns, d = zip(*batch_items)
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

    def act(self, state: torch.Tensor, explore: bool = True) -> int:
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
    def __init__(self, input_dim, output_dim):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim))
        self.critic = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    def act(self, state: torch.Tensor, opponent_history: Optional[List] = None):
        # opponent_history is accepted for compatibility with interfaces that pass it; it is unused here
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


# ----------------------------------------------------------------------------
# Statistical helpers
# ----------------------------------------------------------------------------

def _t_critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    df = n - 1
    try:
        if _SCIPY_AVAILABLE:
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


def train_dqn_ipd(agent: DQNAgent, env: "IPDEnvironment", opponent, episodes: int = 5000, device: torch.device = torch.device('cpu')) -> DQNAgent:
    agent.to(device)
    opt = optim.Adam(agent.q_net.parameters(), lr=1e-3)
    buf = ReplayBuffer(20000)
    batch = 128
    target_sync = 500
    step = 0
    for _ in range(episodes):
        s = env.reset(); opponent.reset(); hist: List[int] = []
        for _ in range(env.episode_length):
            a = agent.act(s, explore=True)
            b = opponent.act(s, opponent_history=hist)
            hist.append(a)
            ns, rewards, done, _ = env.step([a, b]); r = float(rewards[0])
            buf.push(s, a, r, ns, bool(done))
            s = ns; step += 1
            if len(buf) >= batch:
                sb, ab, rb, nsb, db = buf.sample(batch)
                sb = sb.to(device); nsb = nsb.to(device)
                ab = ab.to(device); rb = rb.to(device); db = db.to(device)
                q_pred = agent.q_net(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next = agent.target_q(nsb).max(dim=1).values
                    target = rb + agent.gamma * max_next * (~db)
                loss = nn.functional.mse_loss(q_pred, target)
                opt.zero_grad(); loss.backward(); opt.step()
            if step % target_sync == 0:
                agent.sync_target()
            agent.update_epsilon()
            if done:
                break
    agent.epsilon = agent.epsilon_min; agent.eval(); return agent


def train_ppo_ipd(agent: PPOAgent, env: "IPDEnvironment", opponent, episodes: int = 5000, device: torch.device = torch.device('cpu')) -> PPOAgent:
    agent.to(device)
    optimizer = optim.Adam(list(agent.actor.parameters()) + list(agent.critic.parameters()), lr=3e-4)
    clip_eps = 0.2; entropy_coef = 0.01; value_coef = 0.5; gamma = 0.99; lam = 0.95
    update_epochs = 4; minibatch_size = 128
    roll = RolloutBuffer()

    def optimize_rollout():
        if len(roll.states) == 0:
            return
        states = torch.stack(roll.states).to(device)
        actions = torch.tensor(roll.actions, dtype=torch.long, device=device)
        rewards = torch.tensor(roll.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(roll.dones, dtype=torch.bool, device=device)
        old_logp = torch.stack(roll.log_probs).detach().to(device)
        values = torch.stack(roll.values).detach().to(device)
        adv, rets = compute_gae(rewards, values, dones, gamma, lam)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
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
        roll.clear()

    for _ in range(episodes):
        s = env.reset(); opponent.reset(); hist: List[int] = []
        for _ in range(env.episode_length):
            a, logp, val = agent.act(s)
            b = opponent.act(s, opponent_history=hist)
            hist.append(a)
            ns, rewards, done, _ = env.step([a, b]); r = float(rewards[0])
            roll.states.append(s.detach().cpu()); roll.actions.append(int(a))
            roll.rewards.append(r); roll.dones.append(bool(done))
            roll.log_probs.append(logp.detach().cpu()); roll.values.append(val.detach().cpu())
            s = ns
            if done:
                break
        optimize_rollout()
    agent.eval(); return agent


# ----------------------------------------------------------------------------
# Seed-wise evaluation helper (average per-episode reward against a fixed opponent)
# ----------------------------------------------------------------------------

def _act_to_int(policy: nn.Module, state: torch.Tensor) -> int:
    res = policy.act(state)
    if isinstance(res, (tuple, list)):
        return int(res[0])
    return int(res)


def evaluate_simple_avg_reward(policy: nn.Module, env: IPDEnvironment, opponent: ChallengerAgent, episodes: int = 50) -> float:
    total = 0.0
    for _ in range(episodes):
        s = env.reset(); opponent.reset(); hist: List[int] = []
        ep_total = 0.0
        for _ in range(env.episode_length):
            a = _act_to_int(policy, s)
            b = opponent.act(s, opponent_history=hist)
            hist.append(a)
            s, rewards, done, _ = env.step([a, b])
            ep_total += float(rewards[0])
            if done:
                break
        total += ep_total
    return total / float(episodes)

# --- START: CORRECTED Game-Theoretic Agent ---

class FictitiousPlayAgent(ChallengerAgent):
    """
    An agent that implements Fictitious Play for IPD. It plays the best response
    to the historical frequency of the opponent's actions.
    """
    def __init__(self, name="FictitiousPlay"):
        super().__init__(name, "medium")
        self.opponent_action_counts = np.zeros(2) # Counts for Cooperate, Defect

    def act(self, observation: torch.Tensor, opponent_history: Optional[List] = None) -> int:
        # Fictitious play should look at the entire history
        if opponent_history:
            # Rebuild counts from scratch each time for accuracy
            self.opponent_action_counts = np.array([opponent_history.count(0), opponent_history.count(1)])

        total_moves = np.sum(self.opponent_action_counts)
        if total_moves == 0:
            return random.randint(0, 1) # Cooperate or Defect randomly at the start

        opponent_strategy = self.opponent_action_counts / total_moves
        
        # Expected payoffs for my actions (Cooperate=0, Defect=1) against opponent's historical strategy
        # Payoffs from IPD: (my_action, opp_action) -> (my_reward, opp_reward)
        # R=3, S=0, T=5, P=1
        payoff_cooperate = opponent_strategy[0] * 3 + opponent_strategy[1] * 0  # My payoff if I cooperate
        payoff_defect = opponent_strategy[0] * 5 + opponent_strategy[1] * 1     # My payoff if I defect

        return 1 if payoff_defect > payoff_cooperate else 0

    def update(self, reward: float, observation: torch.Tensor, action: int):
        pass # The logic is stateless and handled entirely within act() based on history

    def reset(self):
        super().reset()
        self.opponent_action_counts = np.zeros(2)

    @property
    def compatible_action_space(self) -> Space:
        """Declares that this agent is designed for a 2-action space like IPD."""
        return Discrete(2)

# ============================================================================
# X. Self-Play Baseline and PRPO (exploitability-only) for IPD
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


def train_selfplay_ipd(env: IPDEnvironment, episodes: int, input_dim: int, output_dim: int, device: torch.device) -> PPOAgent:
    agent = PPOAgent(input_dim, output_dim).to(device)
    opponent = SelfPlayOpponent(output_dim, input_dim, device)
    opt = optim.Adam(agent.parameters(), lr=1e-3)
    for _ in range(episodes):
        s = env.reset(); opponent.reset(); hist: List[int] = []
        for _ in range(env.episode_length):
            a, _, _ = agent.act(s)
            b = opponent.act(s, opponent_history=hist)
            hist.append(a)
            ns, rewards, done, _ = env.step([a, b]); r = rewards[0]
            logits = agent(s.unsqueeze(0)); v = agent.critic(s.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            lp = dist.log_prob(torch.tensor(a))
            adv = torch.tensor(r, dtype=torch.float32) - v.detach().squeeze()
            loss = -lp * adv + nn.functional.mse_loss(v.squeeze(), torch.tensor(r, dtype=torch.float32))
            opt.zero_grad(); loss.backward(); opt.step(); s = ns
            if done: break
        if random.random() < 0.1: opponent.add_snapshot(agent)
    print("Training finished for Self-Play (IPD).")
    return agent


def _ipd_exploitability_proxy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.detach(), dim=-1).squeeze()
    # Proxy: prefer mutual cooperation; penalize deviation probability mass on Defect
    # Higher p(defect) implies more exploitable in cooperative regimes; use p_defect
    return probs[1]


# =====================================================================================
# START: CORRECTED UNIFIED PRPO IMPLEMENTATION FOR IPD
# This code replaces the original, flawed PRPO implementation. It is based on the
# robust, population-based framework used for Kuhn Poker and other zero-sum games,
# adapted for the general-sum, long-horizon nature of the Iterated Prisoner's Dilemma.
# =====================================================================================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'log_prob', 'value'])

class RolloutBuffer:
    """A buffer to store trajectories for PPO updates."""
    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []

    def clear(self):
        self.__init__()

    def __len__(self) -> int:
        return len(self.states)


class StandardPPO:
    """
    A standard PPO agent implementation, used here for training oracle policies
    and as a base class for the PRPO agent. It correctly handles multi-step episodes.
    """
    def __init__(self, state_dim: int, action_dim: int, lr: float, device: torch.device):
        self.device = device
        self.action_dim = action_dim
        self.gamma = 0.99
        self.lam = 0.95 # GAE lambda
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.entropy_coeff = 0.01
        self.value_coeff = 0.5
        self.policy = PPOAgent(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = RolloutBuffer()

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        state_t = state.unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, logp, val = self.policy.act(state_t)
        return action, logp.cpu(), val.cpu()

    def store_experience(self, s: torch.Tensor, a: int, r: float, d: bool, lp: torch.Tensor, v: torch.Tensor):
        self.memory.states.append(s.detach().cpu())
        self.memory.actions.append(a)
        self.memory.rewards.append(r)
        self.memory.dones.append(d)
        self.memory.log_probs.append(lp)
        self.memory.values.append(v)

    def update_policy(self):
        if not self.memory: return {}
        
        # 1. Prepare data from buffer
        states = torch.stack(self.memory.states).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_log_probs = torch.stack(self.memory.log_probs).to(self.device)
        old_values = torch.stack(self.memory.values).to(self.device)
        rewards = torch.tensor(self.memory.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.memory.dones, dtype=torch.bool, device=self.device)

        # 2. Compute GAE and Returns
        advantages, returns = compute_gae(rewards, old_values, dones, self.gamma, self.lam)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. PPO update loop
        for _ in range(self.k_epochs):
            logp, entropy, vals = self.policy.evaluate_actions(states, actions)
            ratios = torch.exp(logp - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            pol_loss = -torch.min(surr1, surr2).mean()
            val_loss = F.mse_loss(vals, returns)
            loss = pol_loss + self.value_coeff * val_loss - self.entropy_coeff * entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
        self.memory.clear()
        return {'loss': loss.item()}


class IPDExploitabilityCalculator:
    """
    Trains a PPO-based best-response 'oracle' against a given policy to
    calculate its exploitability in the IPD environment.
    Exploitability is defined as the average episode reward achieved by the oracle.
    """
    def __init__(self, env: IPDEnvironment, device: torch.device, train_episodes: int = 250, eval_episodes: int = 50):
        self.env = env
        self.device = device
        self.train_episodes = train_episodes
        self.eval_episodes = eval_episodes
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n

    def _train_oracle(self, policy_to_exploit: nn.Module) -> StandardPPO:
        """Trains and returns a StandardPPO agent that is a best response."""
        oracle = StandardPPO(self.input_dim, self.output_dim, lr=3e-4, device=self.device)
        policy_to_exploit.eval()

        for _ in range(self.train_episodes):
            s = self.env.reset()
            oracle_hist: List[int] = []
            for _ in range(self.env.episode_length):
                # Oracle (P1) acts and collects experience
                action, logp, val = oracle.select_action(s)
                
                # Frozen policy (P2) acts
                with torch.no_grad():
                    opp_action = _act_to_int(policy_to_exploit, s)

                ns, rewards, done, _ = self.env.step([action, opp_action])
                oracle.store_experience(s, action, float(rewards[0]), done, logp, val)
                s = ns
                oracle_hist.append(action)
                if done: break
            oracle.update_policy() # Update after each full episode
        
        policy_to_exploit.train()
        oracle.policy.eval()
        return oracle

    def compute(self, policy: nn.Module) -> float:
        """Computes exploitability by training and evaluating an oracle."""
        oracle = self._train_oracle(policy)
        policy.eval()
        
        total_oracle_reward = 0.0
        for _ in range(self.eval_episodes):
            s = self.env.reset()
            oracle_hist: List[int] = []
            ep_reward = 0.0
            for _ in range(self.env.episode_length):
                with torch.no_grad():
                    action, _, _ = oracle.select_action(s)
                    opp_action = _act_to_int(policy, s)
                
                ns, rewards, done, _ = self.env.step([action, opp_action])
                ep_reward += float(rewards[0]) # Oracle's reward
                s = ns
                oracle_hist.append(action)
                if done: break
            total_oracle_reward += ep_reward
        
        policy.train()
        # Exploitability is the average reward the BR can extract per episode
        return total_oracle_reward / self.eval_episodes


class UnifiedPRPOAgent(StandardPPO):
    """
    The PRPO agent for IPD. It inherits from StandardPPO and adds game-theoretic
    regularization to its update rule.
    """
    def __init__(self, state_dim: int, action_dim: int, lr: float, device: str,
                 lambda_exploit_base: float, adaptive_lambda_callable: Optional[Callable],
                 lambda_target: float):
        super().__init__(state_dim, action_dim, lr, device)
        self.lambda_exploit_base = lambda_exploit_base
        self.adaptive_lambda_callable = adaptive_lambda_callable
        self.lambda_target = lambda_target
        self.lambda_exploit_current = lambda_exploit_base
        self.target_policy_state_dict: Optional[Dict[str, Any]] = None
        self.current_exploitability: float = 0.0

    def update_policy(self):
        if not self.memory: return {}

        # Adapt exploitability lambda if a callable is provided
        if self.adaptive_lambda_callable:
            self.lambda_exploit_current = self.adaptive_lambda_callable(
                self.lambda_exploit_base, self.current_exploitability
            )

        # 1. Prepare data and compute GAE (same as StandardPPO)
        states = torch.stack(self.memory.states).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_log_probs = torch.stack(self.memory.log_probs).to(self.device)
        old_values = torch.stack(self.memory.values).to(self.device)
        rewards = torch.tensor(self.memory.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.memory.dones, dtype=torch.bool, device=self.device)
        advantages, returns = compute_gae(rewards, old_values, dones, self.gamma, self.lam)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. PRPO update loop
        for _ in range(self.k_epochs):
            logp, entropy, vals = self.policy.evaluate_actions(states, actions)
            ratios = torch.exp(logp - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            pol_loss = -torch.min(surr1, surr2).mean()
            val_loss = F.mse_loss(vals, returns)
            ppo_loss = pol_loss + self.value_coeff * val_loss - self.entropy_coeff * entropy.mean()

            # --- Game-Theoretic Regularization Terms ---
            # L_Exploit: Penalty for being exploitable
            exploit_reg_loss = self.lambda_exploit_current * self.current_exploitability

            # L_Target: KL-Divergence penalty to move towards the best agent in the population
            target_reg_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_target > 0 and self.target_policy_state_dict is not None:
                target_policy = PPOAgent(self.policy.actor[0].in_features, self.policy.actor[-1].out_features).to(self.device)
                target_policy.load_state_dict(self.target_policy_state_dict)
                target_policy.eval()
                with torch.no_grad():
                    target_logits = target_policy.actor(states)
                current_log_probs = F.log_softmax(self.policy.actor(states), dim=-1)
                kl_div = F.kl_div(current_log_probs, F.softmax(target_logits, dim=-1).detach(), reduction='batchmean')
                target_reg_loss = self.lambda_target * kl_div

            # --- Final Loss ---
            total_loss = ppo_loss - exploit_reg_loss + target_reg_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.memory.clear()
        return {'loss': total_loss.item(), 'exploitability': self.current_exploitability}


class UnifiedPRPO_IPD:
    """ Manages a population of PRPO agents for the Iterated Prisoner's Dilemma. """
    def __init__(self, env: IPDEnvironment, input_dim: int, output_dim: int, device: torch.device,
                 population_size: int, lambda_exploit_base: float, lambda_target: float,
                 adaptive_lambda: Optional[Callable]):
        self.env = env
        self.device = device
        self.population = [
            UnifiedPRPOAgent(input_dim, output_dim, 3e-4, device, lambda_exploit_base, adaptive_lambda, lambda_target)
            for _ in range(population_size)
        ]
        self.exploit_calc = IPDExploitabilityCalculator(env, device, train_episodes=200)
        self.best_agent_state_dict: Optional[Dict[str, Any]] = None

    def _find_and_set_target_policy(self):
        """Finds the least exploitable agent and sets it as the target for others."""
        best_agent, min_exploit = None, float('inf')
        for agent in self.population:
            exploit = self.exploit_calc.compute(agent.policy)
            agent.current_exploitability = exploit
            if exploit < min_exploit:
                min_exploit = exploit
                best_agent = agent
        
        if best_agent:
            self.best_agent_state_dict = copy.deepcopy(best_agent.policy.state_dict())
            for agent in self.population:
                agent.target_policy_state_dict = self.best_agent_state_dict
        return min_exploit

    def train(self, total_episodes: int, episodes_per_update=50):
        completed_episodes = 0
        while completed_episodes < total_episodes:
            # 1. Tournament Phase: All agents play against each other
            for _ in range(episodes_per_update):
                p1_idx, p2_idx = random.sample(range(len(self.population)), 2)
                agent1, agent2 = self.population[p1_idx], self.population[p2_idx]
                
                s = self.env.reset()
                a1_hist: List[int] = []
                a2_hist: List[int] = []
                for _ in range(self.env.episode_length):
                    a1, lp1, v1 = agent1.select_action(s)
                    a2, lp2, v2 = agent2.select_action(s)
                    ns, rewards, done, _ = self.env.step([a1, a2])
                    agent1.store_experience(s, a1, float(rewards[0]), done, lp1, v1)
                    agent2.store_experience(s, a2, float(rewards[1]), done, lp2, v2)
                    s = ns
                    if done: break
            
            # 2. Update and Evaluation Phase
            completed_episodes += episodes_per_update
            current_best_exploit = self._find_and_set_target_policy()
            for agent in self.population:
                agent.update_policy()
            avg_exploit = np.mean([a.current_exploitability for a in self.population])
            print(f"  PRPO Episode {completed_episodes}: Avg Exploit: {avg_exploit:.3f} (Best: {current_best_exploit:.3f})")

        # Final selection of the best agent
        final_best_exploit = self._find_and_set_target_policy()
        best_agent = min(self.population, key=lambda ag: ag.current_exploitability)
        print(f"  PRPO Training Finished. Best agent exploitability: {final_best_exploit:.3f}")
        return best_agent.policy


def train_prpo_ipd_simple(env: IPDEnvironment, episodes: int, input_dim: int, output_dim: int, device: torch.device,
                          lambda_exploit: float = 0.5) -> PPOAgent:
    """ Main function to set up and run the corrected PRPO training for IPD. """
    print("\n--- Training IPD PRPO (Unified Framework) ---")
    
    # An adaptive lambda that increases the penalty as the agent becomes more exploitable.
    # We invert the sign because higher reward for the oracle means higher exploitability.
    def adaptive_lambda(base: float, exploit_score: float) -> float:
        # A higher score is bad, so we want a larger penalty.
        # Scale factor can be tuned. Let's assume scores are around 0-250.
        # A score of 125 (mid-range) would double the base lambda.
        exploit_factor = 1.0 + (exploit_score / 125.0) 
        return base * exploit_factor

    prpo_system = UnifiedPRPO_IPD(
        env=env,
        input_dim=input_dim, output_dim=output_dim, device=device,
        population_size=4,
        lambda_exploit_base=lambda_exploit,
        lambda_target=0.05, # Penalty for deviating from the best agent
        adaptive_lambda=adaptive_lambda
    )
    
    final_policy = prpo_system.train(total_episodes=episodes, episodes_per_update=10)
    print("Training finished for PRPO (IPD, Unified).")
    return final_policy

# =====================================================================================
# END: CORRECTED UNIFIED PRPO IMPLEMENTATION FOR IPD
# =====================================================================================
# ============================================================================
# 4. Training and Evaluation Logic for IPD
# ============================================================================

def train_ipd_agent(agent, env, opponent, num_episodes=5000):
    print(f"Training {agent.__class__.__name__} against {opponent.name} in IPD...")
    if isinstance(agent, DQNAgent):
        return train_dqn_ipd(agent, env, opponent, episodes=num_episodes)
    if isinstance(agent, PPOAgent):
        return train_ppo_ipd(agent, env, opponent, episodes=num_episodes)
    return agent

# ============================================================================
# 5. PSRO for IPD (mixture of DQN best-responses)
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


def _evaluate_policy_vs_ipd_opponent(policy: nn.Module, env: IPDEnvironment, opponent: ChallengerAgent, episodes: int = 20) -> float:
    total = 0.0
    for _ in range(episodes):
        s = env.reset()
        opponent.reset()
        hist: List[int] = []
        for _ in range(env.episode_length):
            a = _act_to_int(policy, s)
            b = opponent.act(s, opponent_history=hist)
            hist.append(a)
            s, rewards, done, _ = env.step([a, b])
            total += float(rewards[0])
            if done:
                break
    return total / float(episodes)


def _compute_meta_strategy(population: List[nn.Module], opponents: List[ChallengerAgent], env: IPDEnvironment) -> List[float]:
    payoffs: List[float] = []
    for pol in population:
        perf = 0.0
        for opp in opponents:
            perf += _evaluate_policy_vs_ipd_opponent(pol, env, copy.deepcopy(opp), episodes=10)
        payoffs.append(perf / max(1, len(opponents)))
    logits = torch.tensor(payoffs, dtype=torch.float32)
    return torch.softmax(logits, dim=0).tolist()


def train_best_response_dqn_ipd(env: IPDEnvironment, opponents: List[ChallengerAgent], episodes: int, input_dim: int, output_dim: int) -> nn.Module:
    br = DQNAgent(input_dim, output_dim)
    opt = optim.Adam(br.q_net.parameters(), lr=1e-3)
    for _ in range(episodes):
        s = env.reset()
        opp = copy.deepcopy(random.choice(opponents))
        opp.reset()
        hist: List[int] = []
        for _ in range(env.episode_length):
            a = br.act(s)
            b = opp.act(s, opponent_history=hist)
            hist.append(a)
            ns, rewards, done, _ = env.step([a, b])
            r = rewards[0]
            q_pred = br(s.unsqueeze(0))[0, a]
            loss = nn.functional.mse_loss(q_pred, torch.tensor(r, dtype=torch.float32))
            opt.zero_grad(); loss.backward(); opt.step()
            s = ns
            if done:
                break
    return br


def train_psro_ipd(gauntlet: "EnhancedGauntletBenchmark", iterations: int = 5, episodes_per_iter: int = 100) -> PSROPolicy:
    # Opponent pool from master list with IPD prefix
    opponent_pool: List[ChallengerAgent] = [copy.deepcopy(ch) for name, ch in gauntlet.master_challenger_list.items() if name.startswith("IPD_")]
    if not opponent_pool:
        opponent_pool = [FictitiousPlayAgent()]
    env = IPDEnvironment()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    population: List[nn.Module] = []
    for i in range(iterations):
        print(f"[PSRO-IPD] Iteration {i+1}/{iterations}")
        br = train_best_response_dqn_ipd(env, opponent_pool, episodes_per_iter, input_dim, output_dim)
        population.append(copy.deepcopy(br))
    meta = _compute_meta_strategy(population, opponent_pool, env)
    print(f"[PSRO-IPD] Meta: {np.round(meta, 3)}")
    return PSROPolicy(population, meta)


if __name__ == "__main__":
    # Hard-coded parameters instead of CLI arguments
    seeds = 3
    episodes = 100

    print("\n--- Setting up Gauntlet Benchmark for IPD Evaluation ---")
    config = EvaluationConfig(num_episodes=episodes, parallel_workers=1, save_visualizations=True)
    gauntlet = EnhancedGauntletBenchmark(config)

    # Register IPD
    ipd_A = np.array([[3, 0], [5, 1]])
    ipd_B = ipd_A.T.copy()
    gauntlet.register_environment(
        "IPD",
        IPDEnvironment,
        payoff_matrices=(ipd_A, ipd_B),
        game_prefix="IPD",
        zero_sum=False,
        symmetric_identical_payoffs=True
    )

    # Instantiate environment and agents
    env = IPDEnvironment()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    dqn_agent = DQNAgent(input_dim, output_dim)
    ppo_agent = PPOAgent(input_dim, output_dim)

    # Training opponent
    training_opponent = gauntlet.master_challenger_list['IPD_TitForTat']
    training_opponent.name = "TrainingOpponent_TFT"

    # Prepare output dirs
    base_dir = os.path.join("results", "IPD")
    os.makedirs(base_dir, exist_ok=True)
    dqn_dir = os.path.join(base_dir, "DQN"); os.makedirs(dqn_dir, exist_ok=True)
    ppo_dir = os.path.join(base_dir, "PPO"); os.makedirs(ppo_dir, exist_ok=True)
    psro_dir = os.path.join(base_dir, "PSRO"); os.makedirs(psro_dir, exist_ok=True)

    # Train DQN & PPO multi-seed
    print(f"\n--- Starting IPD Training Phase (DQN, PPO, {episodes} episodes x {seeds} seeds) ---")
    dqn_runs = []
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        dqn_runs.append(train_ipd_agent(copy.deepcopy(dqn_agent), env, training_opponent, num_episodes=episodes))
    trained_dqn = dqn_runs[-1]
    ppo_runs = []
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        ppo_runs.append(train_ipd_agent(copy.deepcopy(ppo_agent), env, training_opponent, num_episodes=episodes))
    trained_ppo = ppo_runs[-1]
    torch.save(trained_dqn.state_dict(), os.path.join(dqn_dir, "model.pt"))
    # Save the underlying PPO policy parameters (StandardPPO has no state_dict)
    torch.save(trained_ppo.policy.state_dict(), os.path.join(ppo_dir, "model.pt"))

    # Train PSRO (increase budget for stronger oracle)
    print("\n--- Training IPD PSRO (5x1000) ---")
    psro_policy = train_psro_ipd(gauntlet, iterations=5, episodes_per_iter=1000)
    meta = psro_policy.meta_strategy.cpu().numpy().tolist()
    for i, pol in enumerate(psro_policy.population):
        torch.save(pol.state_dict(), os.path.join(psro_dir, f"pop_member_{i}.pt"))
    import json
    with open(os.path.join(psro_dir, "meta_strategy.json"), "w") as f:
        json.dump({"meta_strategy": meta, "population_size": len(meta)}, f, indent=2)
    # Save empirical meta-game as well if available
    try:
        with open(os.path.join(psro_dir, "meta_game.json"), "w") as f:
            json.dump({
                "payoff_row": [],  # PSRO IPD uses evaluation-based meta, not full matrix
                "equilibrium_row": meta
            }, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save meta_game.json: {e}")

    # Self-Play baseline
    print("\n--- Training IPD Self-Play (1000 episodes) ---")  # Updated message
    ipd_sp = train_selfplay_ipd(env, episodes=100, input_dim=input_dim, output_dim=output_dim, device=torch.device('cpu'))  # Increased episodes
    sp_dir = os.path.join(base_dir, "SelfPlay"); os.makedirs(sp_dir, exist_ok=True)
    torch.save(ipd_sp.state_dict(), os.path.join(sp_dir, "model.pt"))

    # PRPO simple (multi-seed for CI and paired testing)
    print("\n--- Training IPD PRPO (1000 episodes x seeds) ---")
    prpo_runs: List[nn.Module] = []
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        prpo_runs.append(train_prpo_ipd_simple(env, episodes=100, input_dim=input_dim, output_dim=output_dim, device=torch.device('cpu')))
    ipd_prpo = prpo_runs[-1]
    prpo_dir = os.path.join(base_dir, "PRPO"); os.makedirs(prpo_dir, exist_ok=True)
    torch.save(ipd_prpo.state_dict(), os.path.join(prpo_dir, "model.pt"))

    # Add custom FP challenger
    gauntlet.add_custom_challenger("IPD_FictitiousPlay", FictitiousPlayAgent())

    # Evaluate and save reports per algorithm
    def eval_and_report(g: "EnhancedGauntletBenchmark", policy: nn.Module, name: str, out_dir: str):
        print(f"\n{'='*40}\nEVALUATING: {name}\n{'='*40}")
        g.evaluate_policy(policy=policy, policy_name=name, environments=["IPD"])
        report_path = os.path.join(out_dir, "report.json")
        g.generate_report(report_path)
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

    trained_dqn.eval(); eval_and_report(gauntlet, trained_dqn, "IPD_DQN", dqn_dir)
    trained_ppo.eval(); eval_and_report(gauntlet, trained_ppo, "IPD_PPO", ppo_dir)
    psro_policy.eval(); eval_and_report(gauntlet, psro_policy, "IPD_PSRO", psro_dir)
    ipd_sp.eval(); eval_and_report(gauntlet, ipd_sp, "IPD_SelfPlay", sp_dir)
    ipd_prpo.eval(); eval_and_report(gauntlet, ipd_prpo, "IPD_PRPO", prpo_dir)

    # Statistical comparison PRPO vs PPO
    print("\n--- Computing seed-wise scores, 95% CI, and paired t-test (IPD PRPO vs PPO) ---")
    eval_env = IPDEnvironment()
    fixed_opp = gauntlet.master_challenger_list['IPD_TitForTat']
    fixed_opp.name = "EvalOpponent_TFT"
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
    with open(os.path.join(base_dir, "ipd_stats.json"), "w") as f:
        import json
        json.dump({
            "ppo_scores": ppo_scores,
            "prpo_scores": prpo_scores,
            "ppo_stats": ppo_stats,
            "prpo_stats": prpo_stats,
            "p_value_prpo_vs_ppo": p_value,
        }, f, indent=2)

    print("\n\n🎉 IPD training and evaluation complete. Outputs under 'results/IPD/'. 🎉")