%%writefile unified_prpo.py
"""
Unified PRPO (Population-based Regularized Policy Optimization) Framework
========================================================================

This module contains the superior PRPO implementation extracted from the test files.
It provides a game-agnostic framework that can be instantiated for any two-player 
zero-sum game by providing game-specific functions.

Key Components:
- UnifiedPRPOAgent: PPO agent with game-theoretic regularization
- UnifiedPRPO: Population manager for coordinated training
- TimeBudgetTrainer: Time-based training instead of episode-based

The framework supports:
1. Nash equilibrium regularization
2. Exploitability penalties
3. Population-based tournament training
4. Exploitative training against hard-coded bots
5. Time-budget based training paradigms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import time
from collections import namedtuple
from typing import List, Dict, Tuple, Callable, Optional
import copy

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])

class UnifiedActorCritic(nn.Module):
    """A standalone UnifiedActorCritic network for PRPO agents."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(UnifiedActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )
        # Use small gain initialization to encourage exploration towards uniform policy early on
        for layer in self.actor:
            if isinstance(layer, nn.Linear): 
                torch.nn.init.xavier_uniform_(layer.weight, gain=0.1)

    def forward(self, state, temperature=1.0):
        features = self.shared(state)
        logits = self.actor(features)
        policy = F.softmax(logits / temperature, dim=-1)
        value = self.critic(features)
        return policy, value

    def act(self, state, temperature=1.0):
        try:
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            result = self.forward(state, temperature)
            if isinstance(result, (tuple, list)) and len(result) == 2:
                policy, value = result
                dist = Categorical(policy)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                # Ensure all outputs are properly shaped scalars
                action_item = int(action.item())
                
                # Make sure log_prob is a scalar tensor
                if log_prob.dim() > 0:
                    log_prob = log_prob.squeeze()
                if log_prob.dim() == 0:
                    log_prob_tensor = log_prob
                else:
                    log_prob_tensor = log_prob[0] if log_prob.numel() > 0 else torch.tensor(0.0)
                
                # Make sure value is a scalar tensor
                if value.dim() > 1:
                    value = value.squeeze()
                if value.dim() == 1 and value.shape[0] == 1:
                    value = value.squeeze(0)
                elif value.dim() > 1 or (value.dim() == 1 and value.shape[0] > 1):
                    value = value.flatten()[0]
                
                return action_item, log_prob_tensor, value
            else:
                # Fallback if forward returns unexpected format
                return 0, torch.tensor(0.0), torch.tensor(0.0)
        except Exception as e:
            print(f"Error in UnifiedActorCritic.act: {e}")
            return 0, torch.tensor(0.0), torch.tensor(0.0)

class StandardPPO:
    """Base PPO implementation that UnifiedPRPOAgent will inherit from."""
    def __init__(self, state_dim, action_dim, lr=3e-4, device='cpu'):
        self.device = device
        self.action_dim = action_dim
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.entropy_coeff = 0.01
        self.policy = UnifiedActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []

    def select_action(self, state):
        try:
            if isinstance(state, (list, np.ndarray)):
                state = torch.FloatTensor(state)
            elif not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)
            
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            
            with torch.no_grad():
                result = self.policy.act(state)
                if isinstance(result, (tuple, list)) and len(result) == 3:
                    action, log_prob, value = result
                    
                    # Safe conversion to scalars
                    action_scalar = int(action)
                    
                    # Handle log_prob - ensure it's a scalar tensor
                    if isinstance(log_prob, torch.Tensor):
                        if log_prob.numel() == 1:
                            log_prob_scalar = float(log_prob.item())
                        else:
                            log_prob_scalar = float(log_prob.flatten()[0].item())
                    else:
                        log_prob_scalar = float(log_prob)
                    
                    # Handle value - ensure it's a scalar tensor  
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            value_scalar = float(value.item())
                        else:
                            value_scalar = float(value.flatten()[0].item())
                    else:
                        value_scalar = float(value)
                    
                    return action_scalar, log_prob_scalar, value_scalar
                else:
                    # Fallback if policy.act returns unexpected format
                    return 0, 0.0, 0.0
        except Exception as e:
            print(f"Error in select_action: {e}")
            return 0, 0.0, 0.0
    
    def act(self, state):
        """Simplified act method for gauntlet evaluation compatibility."""
        action, _, _ = self.select_action(state)
        return action
    
    def to(self, device):
        """Move agent to device."""
        self.device = device
        self.policy.to(device)
        return self
    
    def eval(self):
        """Set policy to evaluation mode."""
        self.policy.eval()
        return self
    
    def train(self):
        """Set policy to training mode."""
        self.policy.train()
        return self

    def store_experience(self, s, a, r, ns, d, lp, v):
        # Ensure tensors are properly converted
        try:
            # Convert state to numpy or keep as tensor
            if isinstance(s, torch.Tensor):
                s = s.detach().cpu()
            elif isinstance(s, (list, np.ndarray)):
                s = np.array(s, dtype=np.float32)
                
            # Convert log_prob and value to scalars if they are tensors
            if isinstance(lp, torch.Tensor):
                lp = float(lp.item()) if lp.numel() == 1 else float(lp.flatten()[0].item())
            if isinstance(v, torch.Tensor):
                v = float(v.item()) if v.numel() == 1 else float(v.flatten()[0].item())
                
            # Store experience - use 7-field version
            self.memory.append(Experience(s, int(a), float(r), s if ns is None else ns, bool(d), float(lp), float(v)))
        except Exception as e:
            print(f"Error storing experience: {e}")
            # Fallback storage
            self.memory.append(Experience(s, int(a), float(r), s, bool(d), 0.0, 0.0))

    def update_policy(self):
        if not self.memory:
            return {}
        
        try:
            # Safely extract data from memory
            states_list = []
            actions_list = []
            old_log_probs_list = []
            old_values_list = []
            
            for e in self.memory:
                # Handle state conversion
                if isinstance(e.state, torch.Tensor):
                    state = e.state.clone().detach()
                elif isinstance(e.state, np.ndarray):
                    state = torch.FloatTensor(e.state)
                else:
                    state = torch.FloatTensor(e.state)
                states_list.append(state)
                
                actions_list.append(int(e.action))
                old_log_probs_list.append(float(e.log_prob))
                old_values_list.append(float(e.value))
            
            states = torch.stack(states_list).to(self.device)
            actions = torch.LongTensor(actions_list).to(self.device)
            old_log_probs = torch.FloatTensor(old_log_probs_list).to(self.device)
            old_values = torch.FloatTensor(old_values_list).to(self.device)
            
            returns = []
            discounted_reward = 0
            for reward, done in zip(reversed([e.reward for e in self.memory]), 
                                   reversed([e.done for e in self.memory])):
                if done:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                returns.insert(0, discounted_reward)
                
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            advantages = returns - old_values.detach()
            
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(self.k_epochs):
                policy_probs, values = self.policy(states)
                dist = Categorical(policy_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                ratios = torch.exp(new_log_probs - old_log_probs.detach())
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Ensure values are properly shaped for loss computation
                values_flat = values.view(-1)
                returns_flat = returns.view(-1)
                value_loss = F.mse_loss(values_flat, returns_flat)
                
                loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
            self.memory.clear()
            return {'loss': float(loss.item())}
            
        except Exception as e:
            print(f"Error in update_policy: {e}")
            self.memory.clear()
            return {'loss': 0.0}

class UnifiedPRPOAgent(StandardPPO):
    """A unified PRPO agent whose loss is regularized by game-theoretic properties."""
    def __init__(self, state_dim: int, action_dim: int, lr: float, device: str,
                 # Regularization configuration
                 lambda_nash: float = 0.0,
                 nash_target_callable: Optional[Callable] = None,
                 lambda_exploit: float = 0.0):
        super().__init__(state_dim, action_dim, lr, device)
        # Store regularization configuration
        self.lambda_nash = lambda_nash
        self.nash_target_callable = nash_target_callable
        self.lambda_exploit = lambda_exploit
        self.entropy_coeff = 0.05  # Higher entropy for exploration
        # This value is updated externally by the population manager before each update
        self.current_exploitability = 0.0

    def update_policy(self):
        if not self.memory:
            return {}
        
        try:    
            # Standard PPO data preparation - safer tensor handling
            states_list = []
            for e in self.memory:
                if isinstance(e.state, torch.Tensor):
                    state = e.state.detach().cpu().numpy()
                elif isinstance(e.state, np.ndarray):
                    state = e.state
                else:
                    state = np.array(e.state, dtype=np.float32)
                states_list.append(state)
            
            states = torch.FloatTensor(np.array(states_list)).to(self.device)
            actions = torch.LongTensor([int(e.action) for e in self.memory]).to(self.device)
            old_log_probs = torch.FloatTensor([float(e.log_prob) for e in self.memory]).to(self.device)
            old_values = torch.FloatTensor([float(e.value) for e in self.memory]).to(self.device)
            
            returns = []
            discounted_reward = 0
            for r, d in zip(reversed([e.reward for e in self.memory]), 
                           reversed([e.done for e in self.memory])):
                if d:
                    discounted_reward = 0
                discounted_reward = r + (self.gamma * discounted_reward)
                returns.insert(0, discounted_reward)
                
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            advantages = returns - old_values.detach()
            
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
                
                # Ensure values are properly shaped for loss computation
                values_flat = values.view(-1) 
                returns_flat = returns.view(-1)
                value_loss = F.mse_loss(values_flat, returns_flat)
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
                    exploit_reg_loss = torch.tensor(
                        self.current_exploitability, 
                        dtype=torch.float32, 
                        device=self.device
                    )

                # --- Combine Losses into the Unified PRPO Objective ---
                total_loss = (ppo_loss + 
                             self.lambda_nash * nash_reg_loss + 
                             self.lambda_exploit * exploit_reg_loss)

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

            self.memory.clear()
            return {'loss': float(total_loss.item()), 'exploitability': float(self.current_exploitability)}
        
        except Exception as e:
            print(f"Error in PRPO update_policy: {e}")
            self.memory.clear()
            return {'loss': 0.0, 'exploitability': 0.0}

class UnifiedPRPO:
    """Manages a population of PRPO agents and their game-specific training regimen."""
    def __init__(self, state_dim: int, action_dim: int, lr: float, device: str,
                 population_size: int,
                 # PRPO agent configuration
                 lambda_nash: float, nash_target_callable: Optional[Callable],
                 lambda_exploit: float, exploitability_calculator: Callable,
                 # Game-specific training configuration
                 exploiter_opponents: Optional[List[Callable]] = None):
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

    # In unified_prpo.py, inside the UnifiedPRPO class
    def _run_tournament_phase(self, env_factory):
        """Agents play against each other within the population."""
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                try:
                    agent1, agent2 = self.population[i], self.population[j]
                    env = env_factory()
                    state = env.reset()
                    p1_action, p1_logp, p1_val = agent1.select_action(state)
                    p2_action, p2_logp, p2_val = agent2.select_action(state)
                    
                    # FIX: Pass actions as a list to match the more common environment API.
                    # This avoids ambiguity in how the step function interprets the arguments.
                    result = env.step([p1_action, p2_action])
                    if len(result) == 4:
                        _, rewards, done, _ = result
                    else:
                        # Fallback if environment returns unexpected format
                        rewards = [0.0, 0.0]
                        done = True
        
                    agent1.store_experience(state, p1_action, rewards[0], None, done, p1_logp, p1_val)
                    agent2.store_experience(state, p2_action, rewards[1], None, done, p2_logp, p2_val)
                except Exception as e:
                    print(f"    Error in tournament phase: {e}")
                    # Continue with next pair


    def _run_exploitative_phase(self, env_factory):
        """Agents play against a curated list of exploiter opponents."""
        if not self.exploiter_opponents:
            return
            
        for agent in self.population:
            try:
                # Pick a random exploiter bot to train against
                opponent_strategy = random.choice(self.exploiter_opponents)
                env = env_factory()
                state = env.reset()
                agent_action, logp, val = agent.select_action(state)
                opp_action = opponent_strategy()
                # FIX: Pass actions as a list to match the environment API
                result = env.step([agent_action, opp_action])
                if len(result) == 4:
                    _, rewards, done, _ = result
                else:
                    # Fallback if environment returns unexpected format
                    rewards = [0.0, 0.0]
                    done = True
                
                agent.store_experience(state, agent_action, rewards[0], None, done, logp, val)
            except Exception as e:
                print(f"    Error in exploitative phase: {e}")
                # Continue with next agent

    def train_episodes(self, env_factory, num_episodes=2000, update_every=10, exploit_ratio=0.3):
        """Episode-based training (original paradigm)."""
        print(f"  UnifiedPRPO: Training for {num_episodes} episodes...")
        results_over_time = []
        
        for episode in range(1, num_episodes + 1):
            # Run tournament games
            self._run_tournament_phase(env_factory)
            # Run games against exploiters
            if random.random() < exploit_ratio:
                self._run_exploitative_phase(env_factory)

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
                    results_over_time.append({
                        'episode': episode,
                        'avg_exploitability': avg_exploit
                    })

        print(f"  PRPO: Training completed. Processed {num_episodes} episodes.")
        return num_episodes, results_over_time

    def train_time_budget(self, env_factory, time_budget_seconds, update_every_seconds=1.0, exploit_ratio=0.3):
        """Time-budget based training (new paradigm)."""
        print(f"  UnifiedPRPO: Training for {time_budget_seconds} seconds...")
        start_time = time.time()
        results_over_time = []
        episode_count = 0
        last_update_time = start_time
        
        while (time.time() - start_time) < time_budget_seconds:
            # Run tournament games
            self._run_tournament_phase(env_factory)
            # Run games against exploiters
            if random.random() < exploit_ratio:
                self._run_exploitative_phase(env_factory)
            
            episode_count += 1
            current_time = time.time()
            
            # Update policies based on time intervals
            if (current_time - last_update_time) >= update_every_seconds:
                for agent in self.population:
                    agent.current_exploitability = self.exploitability_calculator(agent.policy)
                    agent.update_policy()
                
                avg_exploit = np.mean([a.current_exploitability for a in self.population])
                elapsed_time = current_time - start_time
                print(f"    PRPO Time {elapsed_time:.1f}s: Episodes {episode_count}, Avg Exploitability: {avg_exploit:.4f}")
                
                results_over_time.append({
                    'time': elapsed_time,
                    'episode': episode_count,
                    'avg_exploitability': avg_exploit
                })
                
                last_update_time = current_time

        total_time = time.time() - start_time
        print(f"  PRPO: Training completed. Processed {episode_count} episodes in {total_time:.1f} seconds.")
        return episode_count, results_over_time

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

class TimeBudgetTrainer:
    """A training coordinator that runs different algorithms with time budgets."""
    
    @staticmethod
    def train_standard_ppo_time_budget(env_factory, state_dim, action_dim, time_budget_seconds, 
                                      device='cpu', lr=3e-4):
        """Train Standard PPO with time budget."""
        print(f"  StandardPPO: Training for {time_budget_seconds} seconds...")
        start_time = time.time()
        
        agent = StandardPPO(state_dim, action_dim, lr, device)
        episode_count = 0
        results_over_time = []
        last_log_time = start_time
        
        while (time.time() - start_time) < time_budget_seconds:
            try:
                env = env_factory()
                state = env.reset()
                action, logp, val = agent.select_action(state)
                opp_action = random.randint(0, action_dim - 1)
                # FIX: Use consistent action format - pass as list instead of separate arguments
                result = env.step([action, opp_action])
                if len(result) == 4:
                    _, rewards, _, _ = result
                else:
                    # Fallback handling if step returns different format
                    rewards = [0.0, 0.0]
                    
                agent.store_experience(state, action, rewards[0], None, True, logp, val)
            except Exception as e:
                print(f"    Error in PPO training episode: {e}")
                # Continue with next episode
 
            
            if episode_count % 10 == 0:
                agent.update_policy()
            
            episode_count += 1
            current_time = time.time()
            
            # Log progress every 10 seconds
            if (current_time - last_log_time) >= 10.0:
                elapsed_time = current_time - start_time
                print(f"    PPO Time {elapsed_time:.1f}s: Episodes {episode_count}")
                results_over_time.append({
                    'time': elapsed_time,
                    'episode': episode_count
                })
                last_log_time = current_time
        
        total_time = time.time() - start_time
        print(f"  PPO: Training completed. Processed {episode_count} episodes in {total_time:.1f} seconds.")
        return agent, episode_count, results_over_time

    @staticmethod 
    def train_dqn_time_budget(agent, env_factory, opponent, time_budget_seconds, device='cpu'):
        """Train DQN with time budget."""
        print(f"  DQN: Training for {time_budget_seconds} seconds...")
        start_time = time.time()
        
        from collections import deque
        import random
        
        # Simple replay buffer for time-budget training
        buffer = deque(maxlen=10000)
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        episode_count = 0
        results_over_time = []
        last_log_time = start_time
        
        while (time.time() - start_time) < time_budget_seconds:
            env = env_factory()
            state = env.reset()
            opponent.reset() if hasattr(opponent, 'reset') else None
            
            agent_hist = []
            for _ in range(10):  # Multi-step episodes for complex games
                if hasattr(agent, 'act'):
                    a = agent.act(state, explore=True)
                else:
                    # Fallback for different agent interfaces
                    with torch.no_grad():
                        q_vals = agent(state.unsqueeze(0) if len(state.shape) == 1 else state)
                        if random.random() < 0.1:  # epsilon
                            a = random.randint(0, q_vals.shape[-1] - 1)
                        else:
                            a = q_vals.argmax().item()
                
                # Get opponent action
                if hasattr(opponent, 'act'):
                    b = opponent.act(state, opponent_history=agent_hist)
                else:
                    b = random.randint(0, agent.num_actions if hasattr(agent, 'num_actions') else 2)
                
                agent_hist.append(a)
                result = env.step([a, b])
                if len(result) == 4:
                    next_state, rewards, done, _ = result
                    r = float(rewards[0])
                else:
                    # Fallback if environment returns different format
                    next_state = state
                    r = 0.0
                    done = True
                
                buffer.append((state, a, r, next_state, bool(done)))
                state = next_state
                
                # Training step
                if len(buffer) >= 64:
                    batch = random.sample(buffer, 64)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    
                    states = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in states]).to(device)
                    next_states = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in next_states]).to(device)
                    actions = torch.tensor(actions, device=device)
                    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                    dones = torch.tensor(dones, device=device)
                    
                    q_pred = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        if hasattr(agent, 'target_q_net'):
                            max_next = agent.target_q_net(next_states).max(dim=1).values
                        else:
                            max_next = agent(next_states).max(dim=1).values
                        target = rewards + 0.99 * max_next * (~dones)
                    
                    loss = F.mse_loss(q_pred, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                if done:
                    break
            
            episode_count += 1
            current_time = time.time()
            
            # Log progress every 10 seconds
            if (current_time - last_log_time) >= 10.0:
                elapsed_time = current_time - start_time
                print(f"    DQN Time {elapsed_time:.1f}s: Episodes {episode_count}")
                results_over_time.append({
                    'time': elapsed_time,
                    'episode': episode_count
                })
                last_log_time = current_time
        
        total_time = time.time() - start_time
        print(f"  DQN: Training completed. Processed {episode_count} episodes in {total_time:.1f} seconds.")
        return agent, episode_count, results_over_time
