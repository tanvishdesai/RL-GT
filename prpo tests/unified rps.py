"""
Complete Rock Paper Scissors Comparison: UNIFIED PRPO vs Baselines
==================================================================

This file contains a complete, self-contained script to compare reinforcement
learning algorithms on Rock Paper Scissors, now featuring a unified PRPO framework.

🔧 UNIFIED PRPO FRAMEWORK:
===========================
This script replaces the previous, game-specific PRPO with a general, unified
implementation. This demonstrates PRPO as a flexible framework for injecting
game-theoretic knowledge into policy optimization.

For Rock-Paper-Scissors, the Unified PRPO is instantiated with:
1.  A 'Target Policy Regularization' term (L_Target): The agent is penalized via
    KL-Divergence for deviating from the known Nash Equilibrium ([1/3, 1/3, 1/3]).
2.  An 'Opponent-Driven Regularization' term (L_Opponent): The agent is trained
    against a curated population of hard-coded "exploiter" bots, and a direct
    exploitability penalty is added to its loss function.

This instantiation showcases PRPO's ability to use both a known analytical target
and an empirical measure of robustness simultaneously.

EVALUATION METRICS:
===================
- Game-Theoretic Exploitability: Standard definition of exploitability against
  pure best-response strategies. This measures how much a strategy can be exploited
  by an optimal opponent. Lower values indicate better performance.
- Nash Distance: KL divergence between the agent's policy and the Nash Equilibrium
  (uniform random play). Lower values indicate closer approximation to optimal play.
- Training Efficiency: Exploitability vs wall-clock time to analyze computational
  cost and sample efficiency trade-offs.

KEY DISTINCTIONS FROM EXISTING METHODS:
======================================
PRPO differs from existing multi-agent training methods in several key ways:

1. PSRO (Policy Space Response Oracles): PSRO generates entirely new policies
   and computes a meta-strategy over the population. PRPO internalizes the
   population/opponent data into the loss function of a single agent.

2. Prioritized Fictitious Self-Play (PFSP): PFSP uses opponent selection
   strategies to prioritize training against stronger opponents. PRPO goes
   beyond opponent selection by adding explicit regularization terms to the
   loss function based on game-theoretic properties.

3. Standard Self-Play: Standard self-play trains against past versions of
   the agent. PRPO incorporates both theoretical knowledge (Nash Equilibrium)
   and empirical robustness measures (exploitability) directly into the
   optimization objective.

The core innovation of PRPO is that it transforms population-based training
data into explicit regularization terms that guide the optimization process,
rather than just using it for opponent selection or meta-strategy computation.
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
# (Game and Network classes are unchanged)
# ==========================================
class RockPaperScissorsGame:
    def __init__(self):
        self.payoff_matrix = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
        self.state_dim = 3; self.action_dim = 3
        self.nash_equilibrium = np.array([1/3, 1/3, 1/3]); self.reset()
    def reset(self):
        self.last_opponent_action = random.randint(0,2); return self._get_state()
    def _get_state(self):
        state = np.zeros(self.state_dim); state[self.last_opponent_action]=1.0; return state
    def step(self,p1_action, p2_action):
        p1_reward=self.payoff_matrix[p1_action,p2_action]; p2_reward=-p1_reward
        self.last_opponent_action=p2_action; return self._get_state(),[p1_reward,p2_reward],True

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

# ==========================================
# (Baselines: StandardPPO, SelfPlay, PSRO are unchanged)
# ==========================================
class StandardPPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, device='cpu'):
        self.device=device;self.action_dim=action_dim;self.gamma=0.99;self.eps_clip=0.2
        self.k_epochs=4;self.entropy_coeff=0.01;self.policy=ActorCritic(state_dim, action_dim).to(device)
        self.optimizer=optim.Adam(self.policy.parameters(),lr=lr);self.memory=[]
        self.policy_history=[]
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
    # This class remains unchanged from the original file provided.
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, device: str = 'cpu'):
        self.device = device; self.state_dim = state_dim; self.action_dim = action_dim; self.lr = lr
        self.population = []; self.nash_mixture = None; self.response_episodes = 1000; self.eval_games = 50
    def create_random_policy(self): return ActorCritic(self.state_dim, self.action_dim).to(self.device)
    def train_best_response(self, target_policies, target_weights):
        best_response = ActorCritic(self.state_dim, self.action_dim).to(self.device)
        optimizer = optim.Adam(best_response.parameters(), lr=self.lr)
        for episode in range(self.response_episodes):
            env=RockPaperScissorsGame();state=env.reset()
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
                    env=RockPaperScissorsGame();state=env.reset()
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
    def train(self, num_iterations=5):
        self.population.append(self.create_random_policy())
        results_over_time = []  # Add this line for computational cost analysis
        start_time = time.time()  # Add this line for timing
        
        for iteration in range(num_iterations):
            payoff_matrix = self.evaluate_population()
            if len(self.population)>0: self.nash_mixture=self.compute_nash_mixture(payoff_matrix)
            else: self.nash_mixture=np.array([1.0])
            self.population.append(self.train_best_response(self.population,self.nash_mixture))
            
            # Add computational cost analysis logging
            current_best_policy = self.get_best_policy()
            if current_best_policy:
                current_exploit = calculate_rps_exploitability_callable(current_best_policy)
                elapsed_time = time.time() - start_time
                results_over_time.append({
                    'iteration': iteration + 1,
                    'time': elapsed_time,
                    'exploitability': current_exploit
                })
                print(f"  PSRO Iteration {iteration + 1}, Time {elapsed_time:.0f}s, Exploitability: {current_exploit:.4f}")
        
        print(f"  PSRO: Training completed. Processed {num_iterations} iterations.")
        return num_iterations, results_over_time  # Modified return statement
    def get_best_policy(self): return self.population[-1] if self.population else None

# ===================================================================
#      START: UNIFIED POPULATION-REGULARIZED POLICY OPTIMIZATION
# ===================================================================

class UnifiedPRPOAgent(StandardPPO):
    """A unified PRPO agent whose loss is regularized by game-theoretic properties."""
    def __init__(self, state_dim: int, action_dim: int, lr: float, device: str,
                 # Regularization configuration
                 lambda_nash: float = 0.0,
                 nash_target_callable: Callable = None,
                 lambda_exploit: float = 0.0):
        super().__init__(state_dim, action_dim, lr, device)
        # Store regularization configuration
        self.lambda_nash = lambda_nash
        self.nash_target_callable = nash_target_callable
        self.lambda_exploit = lambda_exploit
        self.entropy_coeff = 0.05 # Higher entropy for exploration
        # This value is updated externally by the population manager before each update
        self.current_exploitability = 0.0

    def update_policy(self):
        if not self.memory: return {}
        # Standard PPO data preparation
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

        # PRPO Update Loop
        for _ in range(self.k_epochs):
            policy_probs, values = self.policy(states)
            dist = Categorical(policy_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # --- Core PPO Loss ---
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.view_as(returns), returns)
            ppo_loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy

            # --- Target Policy (Nash) Regularization ---
            nash_reg_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_nash > 0 and self.nash_target_callable:
                target_dist = self.nash_target_callable(policy_probs)
                nash_reg_loss = F.kl_div(policy_probs.log(), target_dist, reduction='batchmean')

            # --- Exploitability Regularization ---
            exploit_reg_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_exploit > 0:
                # Penalty is proportional to the agent's current measured exploitability
                exploit_reg_loss = torch.tensor(self.current_exploitability, dtype=torch.float32, device=self.device)

            # --- Combine Losses into the Unified PRPO Objective ---
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
                 # PRPO agent configuration
                 lambda_nash: float, nash_target_callable: Callable,
                 lambda_exploit: float, exploitability_calculator: Callable,
                 # Game-specific training configuration
                 exploiter_opponents: List[Callable] = None):
        self.population_size = population_size
        self.device = device
        self.exploiter_opponents = exploiter_opponents or []
        self.exploitability_calculator = exploitability_calculator
        self.tournament_results = []
        # Create the population using the unified agent
        self.population = [
            UnifiedPRPOAgent(
                state_dim, action_dim, lr, device,
                lambda_nash, nash_target_callable, lambda_exploit
            ) for _ in range(population_size)
        ]

    def _run_tournament_phase(self):
        """Agents play against each other within the population."""
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                agent1, agent2 = self.population[i], self.population[j]
                env = RockPaperScissorsGame(); state = env.reset()
                p1_action, p1_logp, p1_val = agent1.select_action(state)
                p2_action, p2_logp, p2_val = agent2.select_action(state)
                _, rewards, done = env.step(p1_action, p2_action)
                agent1.store_experience(state, p1_action, rewards[0], None, done, p1_logp, p1_val)
                agent2.store_experience(state, p2_action, rewards[1], None, done, p2_logp, p2_val)

    def _run_exploitative_phase(self):
        """Agents play against a curated list of exploiter opponents."""
        if not self.exploiter_opponents: return
        for agent in self.population:
            # Pick a random exploiter bot to train against
            opponent_strategy = random.choice(self.exploiter_opponents)
            env = RockPaperScissorsGame(); state = env.reset()
            agent_action, logp, val = agent.select_action(state)
            opp_action = opponent_strategy()
            _, rewards, done = env.step(agent_action, opp_action)
            agent.store_experience(state, agent_action, rewards[0], None, done, logp, val)

    def train(self, num_episodes=2000, update_every=10, exploit_ratio=0.3):
        print(f"  UnifiedPRPO: Training for {num_episodes} episodes...")
        results_over_time = []  # Add this line for computational cost analysis
        
        for episode in range(1, num_episodes + 1):
            # Run tournament games
            self._run_tournament_phase()
            # Run games against exploiters
            if random.random() < exploit_ratio:
                self._run_exploitative_phase()

            # Update policies periodically
            if episode % update_every == 0:
                for agent in self.population:
                    # **CRITICAL STEP**: Calculate exploitability for each agent
                    # and update it before the policy update.
                    agent.current_exploitability = self.exploitability_calculator(agent.policy)
                    agent.update_policy()

                if episode % (update_every * 10) == 0:
                    avg_exploit = np.mean([a.current_exploitability for a in self.population])
                    print(f"    PRPO Episode {episode}: Avg Exploitability: {avg_exploit:.4f}")
                    
                    # Add computational cost analysis logging
                    results_over_time.append({
                        'episode': episode,
                        'avg_exploitability': avg_exploit
                    })

        print(f"  PRPO: Training completed. Processed {num_episodes} episodes.")
        return num_episodes, results_over_time  # Modified return statement

    def get_best_agent(self):
        """Selects the best agent based on lowest final exploitability."""
        print("  PRPO: Performing final evaluation to select best agent...")
        best_agent, min_exploit = None, float('inf')
        for agent in self.population:
            exploit = self.exploitability_calculator(agent.policy)
            if exploit < min_exploit:
                min_exploit, best_agent = exploit, agent
        print(f"    - Best Agent Final Exploitability: {min_exploit:.4f}")
        return best_agent

# ===================================================================
#      END: UNIFIED POPULATION-REGULARIZED POLICY OPTIMIZATION
# ===================================================================

# ==========================================
#   RPS-SPECIFIC REGULARIZATION FUNCTIONS
# ==========================================

def get_rps_nash_policy_callable(policy_probs_batch: torch.Tensor) -> torch.Tensor:
    """Returns the uniform Nash equilibrium distribution for RPS."""
    return torch.full_like(policy_probs_batch, 1/3)

def calculate_rps_exploitability_callable(policy: ActorCritic) -> float:
    """Calculates exploitability against a suite of hard-coded RPS bots."""
    policy.eval()
    device = next(policy.parameters()).device
    state = torch.FloatTensor(np.zeros(3)).unsqueeze(0).to(device) # State is constant in RPS
    state[0, random.randint(0,2)] = 1.0 # Use a dummy state
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
        lambda: np.random.choice([0, 1, 2], p=[0.8, 0.1, 0.1]), # Biased Rock
        lambda: np.random.choice([0, 1, 2], p=[0.1, 0.8, 0.1]), # Biased Paper
        lambda: np.random.choice([0, 1, 2], p=[0.1, 0.1, 0.8]), # Biased Scissors
    ]

# ==========================================
#   THE GAUNTLET: ROBUSTNESS BENCHMARK
# ==========================================


# ==========================================
# (Evaluation and Experiment Runner are largely unchanged, but now call the UnifiedPRPO)
# ==========================================
def evaluate_policy_vs_nash(policy,device):
    env=RockPaperScissorsGame(); state=env.reset();
    with torch.no_grad(): policy_probs,_=policy(torch.FloatTensor(state).unsqueeze(0).to(device))
    policy_probs=policy_probs.squeeze().cpu().numpy()
    nash_dist=np.linalg.norm(policy_probs - env.nash_equilibrium, ord=1); return nash_dist, policy_probs

def evaluate_exploitability(policy,device): return calculate_rps_exploitability_callable(policy)

def run_single_experiment(algorithm_name, num_episodes=2000, seed=42):
    print(f"\n--- Running {algorithm_name} (Seed: {seed}) ---")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = RockPaperScissorsGame()
    start_time = time.time()
    final_policy = None; prpo_results = None
    
    # For computational cost analysis
    results_over_time = []
    EVALUATION_INTERVAL = 100  # Evaluate every 100 episodes

    if algorithm_name == "Standard PPO":
        agent = StandardPPO(env.state_dim, env.action_dim, device=device)
        for ep in range(num_episodes):
            state=env.reset(); action,logp,val=agent.select_action(state); opp_action=random.randint(0,2)
            _,rewards,_=env.step(action,opp_action); agent.store_experience(state,action,rewards[0],None,True,logp,val)
            if ep%10==0: agent.update_policy()
            
            # Computational cost analysis
            if ep % EVALUATION_INTERVAL == 0 and ep > 0:
                current_nash_dist, _ = evaluate_policy_vs_nash(agent.policy, device)
                current_exploit = evaluate_exploitability(agent.policy, device)
                elapsed_time = time.time() - start_time
                results_over_time.append({
                    'episode': ep, 'time': elapsed_time, 
                    'nash_distance': current_nash_dist, 'exploitability': current_exploit
                })
                print(f"  Episode {ep}, Time {elapsed_time:.0f}s, Nash Dist: {current_nash_dist:.4f}, Exploit: {current_exploit:.4f}")
        final_policy = agent.policy
    elif algorithm_name == "Self-Play":
        agent = SelfPlay(env.state_dim, env.action_dim, device=device)
        for ep in range(num_episodes):
            state=env.reset(); action,logp,val=agent.select_action(state); opp_action=agent.get_opponent_action(state)
            _,rewards,_=env.step(action,opp_action); agent.store_experience(state,action,rewards[0],None,True,logp,val)
            if ep%10==0: agent.update_policy()
            
            # Computational cost analysis
            if ep % EVALUATION_INTERVAL == 0 and ep > 0:
                current_nash_dist, _ = evaluate_policy_vs_nash(agent.policy, device)
                current_exploit = evaluate_exploitability(agent.policy, device)
                elapsed_time = time.time() - start_time
                results_over_time.append({
                    'episode': ep, 'time': elapsed_time, 
                    'nash_distance': current_nash_dist, 'exploitability': current_exploit
                })
                print(f"  Episode {ep}, Time {elapsed_time:.0f}s, Nash Dist: {current_nash_dist:.4f}, Exploit: {current_exploit:.4f}")
        final_policy = agent.policy
    elif algorithm_name == "PSRO":
        psro = PSRO(env.state_dim, env.action_dim, device=device)
        iterations_completed, psro_results_over_time = psro.train(num_iterations=8)
        final_policy = psro.get_best_policy()
        results_over_time = psro_results_over_time  # Capture PSRO results
    elif algorithm_name == "PRPO":
        # **MODIFICATION**: Instantiate and run the Unified PRPO framework
        prpo_system = UnifiedPRPO(
            state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-4, device=device,
            population_size=4,
            # RPS-specific configuration
            lambda_nash=1.5,
            nash_target_callable=get_rps_nash_policy_callable,
            lambda_exploit=1.0,
            exploitability_calculator=calculate_rps_exploitability_callable,
            exploiter_opponents=get_rps_exploiter_opponents()
        )
        episodes_completed, prpo_results_over_time = prpo_system.train(num_episodes=num_episodes, update_every=10)
        best_agent = prpo_system.get_best_agent()
        if best_agent: final_policy = best_agent.policy
        prpo_results = {'tournament_results': prpo_system.tournament_results}
        results_over_time = prpo_results_over_time  # Capture PRPO results
    elif algorithm_name == "PRPO (Nash Only)":
        # Ablation study: Nash regularization only
        prpo_system = UnifiedPRPO(
            state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-4, device=device,
            population_size=4,
            # RPS-specific configuration with only Nash regularization
            lambda_nash=1.5,
            nash_target_callable=get_rps_nash_policy_callable,
            lambda_exploit=0.0,  # Disable exploitability regularization
            exploitability_calculator=calculate_rps_exploitability_callable,
            exploiter_opponents=get_rps_exploiter_opponents()
        )
        episodes_completed, prpo_results_over_time = prpo_system.train(num_episodes=num_episodes, update_every=10)
        best_agent = prpo_system.get_best_agent()
        if best_agent: final_policy = best_agent.policy
        prpo_results = {'tournament_results': prpo_system.tournament_results}
        results_over_time = prpo_results_over_time  # Capture PRPO results
    elif algorithm_name == "PRPO (Tournament + Nash Only)":
        # Cleaner ablation study: Tournament + Nash regularization only (no exploitative phase)
        prpo_system = UnifiedPRPO(
            state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-4, device=device,
            population_size=4,
            # RPS-specific configuration with only tournament and Nash regularization
            lambda_nash=1.5,
            nash_target_callable=get_rps_nash_policy_callable,
            lambda_exploit=0.0,  # Disable exploitability regularization
            exploitability_calculator=calculate_rps_exploitability_callable,
            exploiter_opponents=[]  # Disable exploitative phase entirely
        )
        episodes_completed, prpo_results_over_time = prpo_system.train(num_episodes=num_episodes, update_every=10)
        best_agent = prpo_system.get_best_agent()
        if best_agent: final_policy = best_agent.policy
        prpo_results = {'tournament_results': prpo_system.tournament_results}
        results_over_time = prpo_results_over_time  # Capture PRPO results
    elif algorithm_name == "PRPO (Exploitability Only)":
        # Ablation study: Exploitability regularization only
        prpo_system = UnifiedPRPO(
            state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-4, device=device,
            population_size=4,
            # RPS-specific configuration with only exploitability regularization
            lambda_nash=0.0,  # Disable Nash regularization
            nash_target_callable=get_rps_nash_policy_callable,
            lambda_exploit=1.0,
            exploitability_calculator=calculate_rps_exploitability_callable,
            exploiter_opponents=get_rps_exploiter_opponents()
        )
        episodes_completed, prpo_results_over_time = prpo_system.train(num_episodes=num_episodes, update_every=10)
        best_agent = prpo_system.get_best_agent()
        if best_agent: final_policy = best_agent.policy
        prpo_results = {'tournament_results': prpo_system.tournament_results}
        results_over_time = prpo_results_over_time  # Capture PRPO results

    training_time = time.time() - start_time
    if final_policy is None: print(f"  {algorithm_name} did not produce a final policy."); return None

    nash_distance, final_policy_probs = evaluate_policy_vs_nash(final_policy, device)
    exploitability = evaluate_exploitability(final_policy, device)
    results = {'algorithm': algorithm_name, 'seed': seed, 'training_time': training_time,
               'final_nash_distance': nash_distance, 'final_exploitability': exploitability,
               'final_policy': final_policy_probs, 'results_over_time': results_over_time,
               'policy_object': final_policy}  # Add the policy object for Gauntlet evaluation
    if prpo_results: results.update(prpo_results)
    print(f"  Training Time: {training_time:.2f}s"); print(f"  Final Policy: {final_policy_probs}")
    print(f"  Nash Distance: {nash_distance:.4f}"); print(f"  Exploitability: {exploitability:.4f}")
    return results

def run_comparison(num_seeds=3, num_episodes=2000):
    # This function remains unchanged.
    print("="*80+"\nROCK PAPER SCISSORS: UNIFIED PRPO vs BASELINES COMPARISON\n"+"="*80)
    algorithms = ["Standard PPO", "Self-Play", "PSRO", "PRPO"]
    all_results = {alg: [] for alg in algorithms}
    for seed in range(num_seeds):
        print(f"\n🎲 SEED {seed + 1}/{num_seeds}")
        for algorithm in algorithms:
            try:
                result = run_single_experiment(algorithm, num_episodes, seed)
                if result: all_results[algorithm].append(result)
            except Exception as e: print(f"  Error in {algorithm}: {e}")
    print(f"\n{'='*80}\n📊 FINAL RESULTS SUMMARY\n{'='*80}")
    summary = {}
    for alg, res_list in all_results.items():
        if not res_list: continue
        nash_d = [r['final_nash_distance'] for r in res_list]; exploit = [r['final_exploitability'] for r in res_list]
        summary[alg] = {'nash_dist_mean': np.mean(nash_d), 'nash_dist_std': np.std(nash_d),
                        'exploit_mean': np.mean(exploit), 'exploit_std': np.std(exploit)}
        print(f"🔬 {alg}:\n  - Nash Distance:  {summary[alg]['nash_dist_mean']:.4f} ± {summary[alg]['nash_dist_std']:.4f}")
        print(f"  - Exploitability: {summary[alg]['exploit_mean']:.4f} ± {summary[alg]['exploit_std']:.4f}")
    return all_results, summary

def run_ablation_study(num_seeds=3, num_episodes=2000):
    """Run ablation studies for PRPO components."""
    print("="*80+"\nROCK PAPER SCISSORS: PRPO ABLATION STUDY\n"+"="*80)
    ablation_algorithms = ["PRPO", "PRPO (Nash Only)", "PRPO (Exploitability Only)", "PRPO (Tournament + Nash Only)"]
    all_results = {alg: [] for alg in ablation_algorithms}
    for seed in range(num_seeds):
        print(f"\n🎲 SEED {seed + 1}/{num_seeds}")
        for algorithm in ablation_algorithms:
            try:
                result = run_single_experiment(algorithm, num_episodes, seed)
                if result: all_results[algorithm].append(result)
            except Exception as e: print(f"  Error in {algorithm}: {e}")
    print(f"\n{'='*80}\n📊 ABLATION STUDY RESULTS\n{'='*80}")
    summary = {}
    for alg, res_list in all_results.items():
        if not res_list: continue
        nash_d = [r['final_nash_distance'] for r in res_list]; exploit = [r['final_exploitability'] for r in res_list]
        summary[alg] = {'nash_dist_mean': np.mean(nash_d), 'nash_dist_std': np.std(nash_d),
                        'exploit_mean': np.mean(exploit), 'exploit_std': np.std(exploit)}
        print(f"🔬 {alg}:\n  - Nash Distance:  {summary[alg]['nash_dist_mean']:.4f} ± {summary[alg]['nash_dist_std']:.4f}")
        print(f"  - Exploitability: {summary[alg]['exploit_mean']:.4f} ± {summary[alg]['exploit_std']:.4f}")
    return all_results, summary

def run_hyperparameter_sensitivity(num_seeds=2, num_episodes=2000):
    """Run hyperparameter sensitivity analysis for PRPO."""
    print("="*80+"\nROCK PAPER SCISSORS: PRPO HYPERPARAMETER SENSITIVITY\n"+"="*80)
    
    # Test different lambda_nash values
    lambda_nash_values = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0]
    lambda_exploit_values = [0.1, 0.5, 1.0, 1.5, 2.0]
    
    all_results = {}
    
    # Test lambda_nash sensitivity
    print("\n🔧 Testing lambda_nash sensitivity...")
    for lambda_nash in lambda_nash_values:
        print(f"\n--- Testing lambda_nash = {lambda_nash} ---")
        results_for_lambda = []
        for seed in range(num_seeds):
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            env = RockPaperScissorsGame()
            
            prpo_system = UnifiedPRPO(
                state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-4, device=device,
                population_size=4,
                lambda_nash=lambda_nash,
                nash_target_callable=get_rps_nash_policy_callable,
                lambda_exploit=1.0,  # Keep exploitability fixed
                exploitability_calculator=calculate_rps_exploitability_callable,
                exploiter_opponents=get_rps_exploiter_opponents()
            )
            prpo_system.train(num_episodes=num_episodes, update_every=10)
            best_agent = prpo_system.get_best_agent()
            if best_agent:
                nash_distance, _ = evaluate_policy_vs_nash(best_agent.policy, device)
                exploitability = evaluate_exploitability(best_agent.policy, device)
                results_for_lambda.append({
                    'lambda_nash': lambda_nash, 'seed': seed,
                    'nash_distance': nash_distance, 'exploitability': exploitability
                })
        all_results[f'lambda_nash_{lambda_nash}'] = results_for_lambda
    
    # Test lambda_exploit sensitivity
    print("\n🔧 Testing lambda_exploit sensitivity...")
    for lambda_exploit in lambda_exploit_values:
        print(f"\n--- Testing lambda_exploit = {lambda_exploit} ---")
        results_for_lambda = []
        for seed in range(num_seeds):
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            env = RockPaperScissorsGame()
            
            prpo_system = UnifiedPRPO(
                state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-4, device=device,
                population_size=4,
                lambda_nash=1.5,  # Keep Nash fixed
                nash_target_callable=get_rps_nash_policy_callable,
                lambda_exploit=lambda_exploit,
                exploitability_calculator=calculate_rps_exploitability_callable,
                exploiter_opponents=get_rps_exploiter_opponents()
            )
            prpo_system.train(num_episodes=num_episodes, update_every=10)
            best_agent = prpo_system.get_best_agent()
            if best_agent:
                nash_distance, _ = evaluate_policy_vs_nash(best_agent.policy, device)
                exploitability = evaluate_exploitability(best_agent.policy, device)
                results_for_lambda.append({
                    'lambda_exploit': lambda_exploit, 'seed': seed,
                    'nash_distance': nash_distance, 'exploitability': exploitability
                })
        all_results[f'lambda_exploit_{lambda_exploit}'] = results_for_lambda
    
    # Print summary
    print(f"\n{'='*80}\n📊 HYPERPARAMETER SENSITIVITY RESULTS\n{'='*80}")
    
    # Lambda Nash results
    print("\n🔧 Lambda Nash Sensitivity:")
    for lambda_nash in lambda_nash_values:
        key = f'lambda_nash_{lambda_nash}'
        if key in all_results and all_results[key]:
            nash_dists = [r['nash_distance'] for r in all_results[key]]
            exploits = [r['exploitability'] for r in all_results[key]]
            print(f"  λ_nash={lambda_nash}: Nash Dist: {np.mean(nash_dists):.4f}±{np.std(nash_dists):.4f}, "
                  f"Exploit: {np.mean(exploits):.4f}±{np.std(exploits):.4f}")
    
    # Lambda Exploit results
    print("\n🔧 Lambda Exploit Sensitivity:")
    for lambda_exploit in lambda_exploit_values:
        key = f'lambda_exploit_{lambda_exploit}'
        if key in all_results and all_results[key]:
            nash_dists = [r['nash_distance'] for r in all_results[key]]
            exploits = [r['exploitability'] for r in all_results[key]]
            print(f"  λ_exploit={lambda_exploit}: Nash Dist: {np.mean(nash_dists):.4f}±{np.std(nash_dists):.4f}, "
                  f"Exploit: {np.mean(exploits):.4f}±{np.std(exploits):.4f}")
    
    return all_results


if __name__ == "__main__":
    print("🚀 ROCK PAPER SCISSORS: COMPREHENSIVE PRPO EVALUATION")
    print("="*80)
    
    # 1. Baseline comparison
    print("\n📊 1. BASELINE COMPARISON")
    run_comparison(num_seeds=3, num_episodes=10000)
    
    # 2. Ablation study
    print("\n📊 2. ABLATION STUDY")
    run_ablation_study(num_seeds=3, num_episodes=10000)
    
    # 3. Hyperparameter sensitivity
    print("\n📊 3. HYPERPARAMETER SENSITIVITY")
    run_hyperparameter_sensitivity(num_seeds=2, num_episodes=10000)
       
    print("\n✅ All experiments completed!")