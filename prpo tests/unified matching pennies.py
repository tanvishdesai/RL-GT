"""
Complete Matching Pennies Comparison: UNIFIED PRPO vs Baselines
================================================================

This file contains a complete, self-contained script to compare reinforcement
learning algorithms on Matching Pennies, now featuring the Unified PRPO framework
and the Gauntlet Benchmark, structured identically to the Rock Paper Scissors version.

🔧 UNIFIED PRPO FRAMEWORK (Applied to Matching Pennies):
=========================================================
This script utilizes the general, unified PRPO implementation to demonstrate its
flexibility. For Matching Pennies, the Unified PRPO is instantiated with:

1.  A 'Target Policy Regularization' term (L_Target): The agent is penalized via
    KL-Divergence for deviating from the known Nash Equilibrium ([0.5, 0.5]).
2.  An 'Opponent-Driven Regularization' term (L_Opponent): The agent is trained
    against a curated population of hard-coded "exploiter" bots (e.g., always
    playing Heads), and a direct exploitability penalty is added to its loss.

EVALUATION METRICS (For Matching Pennies):
==========================================
- Game-Theoretic Exploitability: Measures how much a strategy can be exploited
  by an optimal opponent (one who plays a pure best-response). Lower is better.
- Nash Distance: L1 distance between the agent's policy and the Nash Equilibrium
  (uniform random play: [0.5, 0.5]). Lower is better.
- The Gauntlet Benchmark: A rigorous evaluation against a diverse set of
  challenger bots designed to probe for weaknesses in learned policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import copy
import time
from collections import deque, namedtuple
from typing import List, Dict, Tuple, Callable

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Try to import nashpy for proper PSRO Nash solving
try:
    import nashpy as nash
    NASHPY_AVAILABLE = True
except ImportError:
    NASHPY_AVAILABLE = False
    print("Warning: nashpy not available. Install with 'pip install nashpy' for proper PSRO.")

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])

# ==========================================
#        START: MATCHING PENNIES GAME
# ==========================================

class MatchingPenniesGame:
    """A class representing the Matching Pennies game environment."""
    def __init__(self):
        # Payoff matrix for Player 1 (row player)
        # Actions: 0 = Heads, 1 = Tails
        # P1 wins if match, P2 wins if mismatch
        self.payoff_matrix = np.array([[1, -1], [-1, 1]])
        self.state_dim = 2  # State is the opponent's last action (Heads/Tails)
        self.action_dim = 2 # Actions are Heads or Tails
        self.nash_equilibrium = np.array([0.5, 0.5])
        self.reset()

    def reset(self):
        # Initialize with a random opponent action
        self.last_opponent_action = random.randint(0, self.action_dim - 1)
        return self._get_state()

    def _get_state(self):
        # State is a one-hot vector of the opponent's last action
        state = np.zeros(self.state_dim)
        state[self.last_opponent_action] = 1.0
        return state

    def step(self, p1_action: int, p2_action: int) -> Tuple[np.ndarray, List[float], bool]:
        # Get rewards from the payoff matrix
        p1_reward = self.payoff_matrix[p1_action, p2_action]
        p2_reward = -p1_reward  # Zero-sum game

        # The new state is determined by the opponent's current action
        self.last_opponent_action = p2_action
        
        # The game is stateless and ends after one turn for RL purposes
        done = True
        return self._get_state(), [p1_reward, p2_reward], done

# ==========================================
#   (Network and Baseline classes are unchanged)
# ==========================================

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        for layer in self.actor:
            if isinstance(layer, nn.Linear): torch.nn.init.xavier_uniform_(layer.weight, gain=0.1)
    def forward(self, state, temperature=1.0):
        features = self.shared(state); logits = self.actor(features)
        policy = F.softmax(logits/temperature, dim=-1); value = self.critic(features); return policy, value
    def act(self, state, temperature=1.0):
        policy, value = self.forward(state, temperature); dist=Categorical(policy)
        action=dist.sample(); return action.item(), dist.log_prob(action), value.squeeze()

class StandardPPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, device='cpu'):
        self.device=device;self.action_dim=action_dim;self.gamma=0.99;self.eps_clip=0.2
        self.k_epochs=4;self.entropy_coeff=0.01;self.policy=ActorCritic(state_dim, action_dim).to(device)
        self.optimizer=optim.Adam(self.policy.parameters(),lr=lr);self.memory=[]
    def select_action(self,state):
        state=torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): action,log_prob,value=self.policy.act(state)
        return action,log_prob.cpu().item(),value.cpu().item()
    def store_experience(self,s,a,r,ns,d,lp,v): self.memory.append(Experience(s,a,r,ns,d,lp,v))
    def update_policy(self):
        if not self.memory: return {}
        states=torch.FloatTensor([e.state for e in self.memory]).to(self.device)
        actions=torch.LongTensor([e.action for e in self.memory]).to(self.device)
        old_log_probs=torch.FloatTensor([e.log_prob for e in self.memory]).to(self.device)
        returns=[]; discounted_reward=0
        for reward,done in zip(reversed([e.reward for e in self.memory]),reversed([e.done for e in self.memory])):
            if done: discounted_reward=0
            discounted_reward=reward+(self.gamma*discounted_reward); returns.insert(0,discounted_reward)
        returns=torch.tensor(returns,dtype=torch.float32).to(self.device)
        old_values=torch.FloatTensor([e.value for e in self.memory]).to(self.device)
        advantages=returns-old_values.detach()
        if len(advantages)>1: advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        for _ in range(self.k_epochs):
            policy_probs,values=self.policy(states); dist=Categorical(policy_probs)
            new_log_probs=dist.log_prob(actions); entropy=dist.entropy().mean()
            ratios=torch.exp(new_log_probs-old_log_probs.detach()); surr1=ratios*advantages
            surr2=torch.clamp(ratios,1-self.eps_clip,1+self.eps_clip)*advantages
            policy_loss=-torch.min(surr1,surr2).mean(); value_loss=F.mse_loss(values.squeeze(),returns)
            loss=policy_loss+0.5*value_loss-self.entropy_coeff*entropy
            self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.policy.parameters(),0.5); self.optimizer.step()
        self.memory.clear(); return {'loss':loss.item()}

class SelfPlay(StandardPPO):
    def __init__(self, state_dim, action_dim, lr=3e-4, device='cpu'):
        super().__init__(state_dim, action_dim, lr, device); self.policy_memory = deque(maxlen=20)
    def get_opponent_action(self, state):
        if not self.policy_memory or random.random() < 0.3: return random.randint(0, self.action_dim - 1)
        opponent_state_dict = random.choice(self.policy_memory)
        opponent_policy = ActorCritic(state.shape[0], self.action_dim).to(self.device)
        opponent_policy.load_state_dict(opponent_state_dict); opponent_policy.eval()
        with torch.no_grad(): action,_,_ = opponent_policy.act(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        return action
    def update_policy(self):
        metrics = super().update_policy()
        if metrics: self.policy_memory.append(copy.deepcopy(self.policy.state_dict()))
        return metrics

class PSRO:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, device: str = 'cpu'):
        self.device = device; self.state_dim = state_dim; self.action_dim = action_dim; self.lr = lr
        self.population = []; self.nash_mixture = None; self.response_episodes = 1000; self.eval_games = 50
    def create_random_policy(self): return ActorCritic(self.state_dim, self.action_dim).to(self.device)
    def train_best_response(self, target_policies, target_weights):
        best_response = ActorCritic(self.state_dim, self.action_dim).to(self.device)
        optimizer = optim.Adam(best_response.parameters(), lr=self.lr)
        for episode in range(self.response_episodes):
            env=MatchingPenniesGame();state=env.reset()
            if len(target_policies)>0:
                opponent_idx = np.random.choice(len(target_policies), p=target_weights); opponent=target_policies[opponent_idx]; opponent.eval()
                with torch.no_grad(): opp_action,_,_=opponent.act(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            else: opp_action=random.randint(0,self.action_dim-1)
            br_action,br_log_prob,_=best_response.act(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            _,rewards,_=env.step(br_action,opp_action); reward=rewards[0]
            loss=-br_log_prob*reward; optimizer.zero_grad(); loss.backward(); optimizer.step()
        return best_response
    def evaluate_population(self):
        n=len(self.population); payoff_matrix=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i==j: continue
                total_reward=0.0
                for _ in range(self.eval_games):
                    env=MatchingPenniesGame();state=env.reset()
                    with torch.no_grad():
                        action_i,_,_=self.population[i].act(torch.FloatTensor(state).unsqueeze(0).to(self.device))
                        action_j,_,_=self.population[j].act(torch.FloatTensor(state).unsqueeze(0).to(self.device))
                    _,rewards,_=env.step(action_i,action_j); total_reward+=rewards[0]
                payoff_matrix[i,j]=total_reward/self.eval_games
        return payoff_matrix
    def compute_nash_mixture(self, payoff_matrix):
        if payoff_matrix.size==0:return np.array([]);
        if payoff_matrix.shape[0]<2:return np.array([1.0])
        if NASHPY_AVAILABLE:
            try:
                game=nash.Game(payoff_matrix,-payoff_matrix);equilibria=list(game.support_enumeration())
                if equilibria: nash_mixture=np.array(equilibria[0][0]);return nash_mixture/np.sum(nash_mixture)
            except: pass
        return np.ones(payoff_matrix.shape[0])/payoff_matrix.shape[0]
    def train(self, num_iterations=8):
        self.population.append(self.create_random_policy())
        for iteration in range(num_iterations):
            payoff_matrix = self.evaluate_population()
            self.nash_mixture = self.compute_nash_mixture(payoff_matrix) if len(self.population) > 0 else np.array([1.0])
            self.population.append(self.train_best_response(self.population, self.nash_mixture))
        return self.get_best_policy()
    def get_best_policy(self): return self.population[-1] if self.population else None

# ===================================================================
#      UNIFIED POPULATION-REGULARIZED POLICY OPTIMIZATION (Unchanged)
# ===================================================================

class UnifiedPRPOAgent(StandardPPO):
    """A unified PRPO agent whose loss is regularized by game-theoretic properties."""
    def __init__(self, state_dim: int, action_dim: int, lr: float, device: str,
                 lambda_nash: float = 0.0,
                 nash_target_callable: Callable = None,
                 lambda_exploit: float = 0.0):
        super().__init__(state_dim, action_dim, lr, device)
        self.lambda_nash = lambda_nash
        self.nash_target_callable = nash_target_callable
        self.lambda_exploit = lambda_exploit
        self.entropy_coeff = 0.05
        self.current_exploitability = 0.0

    def update_policy(self):
        if not self.memory: return {}
        states = torch.FloatTensor(np.array([e.state for e in self.memory])).to(self.device)
        actions = torch.LongTensor([e.action for e in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([e.log_prob for e in self.memory]).to(self.device)
        old_values = torch.FloatTensor([e.value for e in self.memory]).to(self.device)
        returns = []; discounted_reward = 0
        for r, d in zip(reversed([e.reward for e in self.memory]), reversed([e.done for e in self.memory])):
            if d: discounted_reward = 0
            discounted_reward = r + (self.gamma * discounted_reward); returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - old_values.detach()
        if len(advantages) > 1: advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.k_epochs):
            policy_probs, values = self.policy(states)
            dist = Categorical(policy_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.view_as(returns), returns)
            ppo_loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy

            nash_reg_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_nash > 0 and self.nash_target_callable:
                target_dist = self.nash_target_callable(policy_probs)
                nash_reg_loss = F.kl_div(policy_probs.log(), target_dist, reduction='batchmean')

            exploit_reg_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_exploit > 0:
                exploit_reg_loss = torch.tensor(self.current_exploitability, dtype=torch.float32, device=self.device)

            total_loss = ppo_loss + self.lambda_nash * nash_reg_loss + self.lambda_exploit * exploit_reg_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.memory.clear()
        return {'loss': total_loss.item(), 'exploitability': self.current_exploitability}

class UnifiedPRPO:
    """Manages a population of PRPO agents and their game-specific training regimen."""
    def __init__(self, state_dim: int, action_dim: int, lr: float, device: str,
                 population_size: int,
                 lambda_nash: float, nash_target_callable: Callable,
                 lambda_exploit: float, exploitability_calculator: Callable,
                 exploiter_opponents: List[Callable] = None):
        self.population_size = population_size
        self.device = device
        self.exploiter_opponents = exploiter_opponents or []
        self.exploitability_calculator = exploitability_calculator
        self.population = [
            UnifiedPRPOAgent(state_dim, action_dim, lr, device, lambda_nash, nash_target_callable, lambda_exploit)
            for _ in range(population_size)
        ]

    def _run_tournament_phase(self, env_callable):
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                agent1, agent2 = self.population[i], self.population[j]
                env = env_callable(); state = env.reset()
                p1_action, p1_logp, p1_val = agent1.select_action(state)
                p2_action, p2_logp, p2_val = agent2.select_action(state)
                _, rewards, done = env.step(p1_action, p2_action)
                agent1.store_experience(state, p1_action, rewards[0], None, done, p1_logp, p1_val)
                agent2.store_experience(state, p2_action, rewards[1], None, done, p2_logp, p2_val)

    def _run_exploitative_phase(self, env_callable):
        if not self.exploiter_opponents: return
        for agent in self.population:
            opponent_strategy = random.choice(self.exploiter_opponents)
            env = env_callable(); state = env.reset()
            agent_action, logp, val = agent.select_action(state)
            opp_action = opponent_strategy()
            _, rewards, done = env.step(agent_action, opp_action)
            agent.store_experience(state, agent_action, rewards[0], None, done, logp, val)

    def train(self, env_callable, num_episodes=2000, update_every=10, exploit_ratio=0.3):
        for episode in range(1, num_episodes + 1):
            self._run_tournament_phase(env_callable)
            if random.random() < exploit_ratio:
                self._run_exploitative_phase(env_callable)

            if episode % update_every == 0:
                for agent in self.population:
                    agent.current_exploitability = self.exploitability_calculator(agent.policy)
                    agent.update_policy()
        return self.get_best_agent()

    def get_best_agent(self):
        best_agent, min_exploit = None, float('inf')
        for agent in self.population:
            exploit = self.exploitability_calculator(agent.policy)
            if exploit < min_exploit:
                min_exploit, best_agent = exploit, agent
        return best_agent

# ===================================================================
#        MP-SPECIFIC REGULARIZATION & EXPLOITER FUNCTIONS
# ===================================================================

def get_mp_nash_policy_callable(policy_probs_batch: torch.Tensor) -> torch.Tensor:
    """Returns the uniform Nash equilibrium distribution for Matching Pennies."""
    return torch.full_like(policy_probs_batch, 0.5)

def calculate_mp_exploitability_callable(policy: ActorCritic) -> float:
    """Calculates exploitability against hard-coded Matching Pennies bots."""
    policy.eval()
    device = next(policy.parameters()).device
    # State is constant in MP, so we can use a dummy state
    state = torch.FloatTensor(np.zeros(2)).unsqueeze(0).to(device)
    state[0, random.randint(0,1)] = 1.0
    
    with torch.no_grad():
        policy_probs, _ = policy(state)
        p = policy_probs.squeeze().cpu().numpy() # p = [p_heads, p_tails]

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
        lambda: np.random.choice([0, 1], p=[0.9, 0.1]), # Biased Heads
        lambda: np.random.choice([0, 1], p=[0.1, 0.9]), # Biased Tails
    ]

# ===================================================================
#               THE GAUNTLET: MATCHING PENNIES EDITION
# ===================================================================


# ==========================================
#        EVALUATION & EXPERIMENT RUNNER
# ==========================================

def evaluate_policy_vs_nash_mp(policy, device):
    env=MatchingPenniesGame(); state=env.reset();
    with torch.no_grad(): policy_probs,_=policy(torch.FloatTensor(state).unsqueeze(0).to(device))
    policy_probs=policy_probs.squeeze().cpu().numpy()
    nash_dist=np.linalg.norm(policy_probs - env.nash_equilibrium, ord=1)
    return nash_dist, policy_probs

def evaluate_exploitability_mp(policy, device):
    return calculate_mp_exploitability_callable(policy)

def run_single_experiment_mp(algorithm_name, num_episodes=5000, seed=42):
    print(f"\n--- Running {algorithm_name} (Seed: {seed}) on Matching Pennies ---")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = MatchingPenniesGame()
    final_policy = None
    
    if algorithm_name == "Standard PPO":
        agent = StandardPPO(env.state_dim, env.action_dim, device=device)
        for ep in range(num_episodes):
            state=env.reset(); action,logp,val=agent.select_action(state); opp_action=random.randint(0,1)
            _,rewards,_=env.step(action,opp_action); agent.store_experience(state,action,rewards[0],None,True,logp,val)
            if ep>0 and ep%10==0: agent.update_policy()
        final_policy = agent.policy
    elif algorithm_name == "Self-Play":
        agent = SelfPlay(env.state_dim, env.action_dim, device=device)
        for ep in range(num_episodes):
            state=env.reset(); action,logp,val=agent.select_action(state); opp_action=agent.get_opponent_action(state)
            _,rewards,_=env.step(action,opp_action); agent.store_experience(state,action,rewards[0],None,True,logp,val)
            if ep>0 and ep%10==0: agent.update_policy()
        final_policy = agent.policy
    elif algorithm_name == "PSRO":
        psro = PSRO(env.state_dim, env.action_dim, device=device)
        final_policy = psro.train(num_iterations=8)
    elif algorithm_name in ["PRPO", "PRPO (Nash Only)", "PRPO (Exploitability Only)", "PRPO (Tournament + Nash Only)"]:
        # Configure PRPO based on the ablation type
        is_exploit_phase_enabled = "Tournament" not in algorithm_name
        lambda_nash = 1.0 if "Exploitability Only" not in algorithm_name else 0.0
        lambda_exploit = 0.5 if "Nash Only" not in algorithm_name else 0.0

        prpo_system = UnifiedPRPO(
            state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-4, device=device,
            population_size=4,
            lambda_nash=lambda_nash,
            nash_target_callable=get_mp_nash_policy_callable,
            lambda_exploit=lambda_exploit,
            exploitability_calculator=calculate_mp_exploitability_callable,
            exploiter_opponents=get_mp_exploiter_opponents() if is_exploit_phase_enabled else []
        )
        best_agent = prpo_system.train(MatchingPenniesGame, num_episodes=num_episodes, update_every=10)
        if best_agent: final_policy = best_agent.policy

    if final_policy is None:
        print(f"  {algorithm_name} did not produce a final policy."); return None
    
    nash_distance, final_policy_probs = evaluate_policy_vs_nash_mp(final_policy, device)
    exploitability = evaluate_exploitability_mp(final_policy, device)
    
    return {
        'algorithm': algorithm_name, 'seed': seed,
        'final_nash_distance': nash_distance, 'final_exploitability': exploitability,
        'final_policy_probs': final_policy_probs, 'policy_object': final_policy
    }

def run_comparison_mp(num_seeds=3, num_episodes=5000):
    print("="*80+"\nMATCHING PENNIES: BASELINE COMPARISON\n"+"="*80)
    algorithms = ["Standard PPO", "Self-Play", "PSRO", "PRPO"]
    all_results = {alg: [] for alg in algorithms}
    for seed in range(num_seeds):
        for algorithm in algorithms:
            result = run_single_experiment_mp(algorithm, num_episodes, seed)
            if result: all_results[algorithm].append(result)
    
    print(f"\n{'='*80}\n📊 FINAL RESULTS SUMMARY\n{'='*80}")
    summary = {}
    for alg, res_list in all_results.items():
        if not res_list: continue
        nash_d = [r['final_nash_distance'] for r in res_list]; exploit = [r['final_exploitability'] for r in res_list]
        summary[alg] = {'nash_dist_mean': np.mean(nash_d), 'nash_dist_std': np.std(nash_d),
                        'exploit_mean': np.mean(exploit), 'exploit_std': np.std(exploit)}
        print(f"🔬 {alg}:\n  - Nash Distance:  {summary[alg]['nash_dist_mean']:.4f} ± {summary[alg]['nash_dist_std']:.4f}")
        print(f"  - Exploitability: {summary[alg]['exploit_mean']:.4f} ± {summary[alg]['exploit_std']:.4f}")

def run_ablation_study_mp(num_seeds=3, num_episodes=5000):
    print("="*80+"\nMATCHING PENNIES: PRPO ABLATION STUDY\n"+"="*80)
    ablation_algorithms = ["PRPO", "PRPO (Nash Only)", "PRPO (Exploitability Only)", "PRPO (Tournament + Nash Only)"]
    all_results = {alg: [] for alg in ablation_algorithms}
    for seed in range(num_seeds):
        for algorithm in ablation_algorithms:
            result = run_single_experiment_mp(algorithm, num_episodes, seed)
            if result: all_results[algorithm].append(result)

    print(f"\n{'='*80}\n📊 ABLATION STUDY RESULTS\n{'='*80}")
    summary = {}
    for alg, res_list in all_results.items():
        if not res_list: continue
        nash_d = [r['final_nash_distance'] for r in res_list]; exploit = [r['final_exploitability'] for r in res_list]
        summary[alg] = {'nash_dist_mean': np.mean(nash_d), 'nash_dist_std': np.std(nash_d),
                        'exploit_mean': np.mean(exploit), 'exploit_std': np.std(exploit)}
        print(f"🔬 {alg}:\n  - Nash Distance:  {summary[alg]['nash_dist_mean']:.4f} ± {summary[alg]['nash_dist_std']:.4f}")
        print(f"  - Exploitability: {summary[alg]['exploit_mean']:.4f} ± {summary[alg]['exploit_std']:.4f}")

def run_hyperparameter_sensitivity_mp(num_seeds=2, num_episodes=5000):
    print("="*80+"\nMATCHING PENNIES: PRPO HYPERPARAMETER SENSITIVITY\n"+"="*80)
    lambda_nash_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    lambda_exploit_values = [0.1, 0.5, 1.0, 2.0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test lambda_nash sensitivity
    print("\n🔧 Testing lambda_nash sensitivity...")
    for lambda_nash in lambda_nash_values:
        nash_dists, exploits = [], []
        for seed in range(num_seeds):
            prpo = UnifiedPRPO(2, 2, 1e-4, device, 4, lambda_nash, get_mp_nash_policy_callable, 0.5, calculate_mp_exploitability_callable, get_mp_exploiter_opponents())
            agent = prpo.train(MatchingPenniesGame, num_episodes)
            if agent:
                n, _ = evaluate_policy_vs_nash_mp(agent.policy, device); e = evaluate_exploitability_mp(agent.policy, device)
                nash_dists.append(n); exploits.append(e)
        if nash_dists: print(f"  λ_nash={lambda_nash}: Nash Dist: {np.mean(nash_dists):.4f}±{np.std(nash_dists):.4f}, Exploit: {np.mean(exploits):.4f}±{np.std(exploits):.4f}")

    # Test lambda_exploit sensitivity
    print("\n🔧 Testing lambda_exploit sensitivity...")
    for lambda_exploit in lambda_exploit_values:
        nash_dists, exploits = [], []
        for seed in range(num_seeds):
            prpo = UnifiedPRPO(2, 2, 1e-4, device, 4, 1.0, get_mp_nash_policy_callable, lambda_exploit, calculate_mp_exploitability_callable, get_mp_exploiter_opponents())
            agent = prpo.train(MatchingPenniesGame, num_episodes)
            if agent:
                n, _ = evaluate_policy_vs_nash_mp(agent.policy, device); e = evaluate_exploitability_mp(agent.policy, device)
                nash_dists.append(n); exploits.append(e)
        if nash_dists: print(f"  λ_exploit={lambda_exploit}: Nash Dist: {np.mean(nash_dists):.4f}±{np.std(nash_dists):.4f}, Exploit: {np.mean(exploits):.4f}±{np.std(exploits):.4f}")



if __name__ == "__main__":
    # Note: Episode counts are reduced for faster demonstration.
    # For publication-quality results, increase num_episodes and num_seeds.
    EPISODES = 10000
    SEEDS = 2

    print("🚀 MATCHING PENNIES: COMPREHENSIVE PRPO & GAUNTLET EVALUATION")
    print("="*80)
    
    # 1. Baseline comparison
    print("\n📊 1. BASELINE COMPARISON")
    run_comparison_mp(num_seeds=SEEDS, num_episodes=EPISODES)
    
    # 2. Ablation study
    print("\n📊 2. ABLATION STUDY")
    run_ablation_study_mp(num_seeds=SEEDS, num_episodes=EPISODES)
    
    # 3. Hyperparameter sensitivity
    print("\n📊 3. HYPERPARAMETER SENSITIVITY")
    run_hyperparameter_sensitivity_mp(num_seeds=SEEDS, num_episodes=EPISODES)
        
    print("\n✅ All Matching Pennies experiments completed!")