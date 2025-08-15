import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from collections import defaultdict
from typing import Optional, List, Tuple, Dict

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
    def __init__(self):
        # Observation space: [own_last_action, opponent_last_action]
        self._observation_space = Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self._action_space = Discrete(2) # 0: Cooperate, 1: Defect
        self.state = None
        # Payoff matrix: R=3, S=0, T=5, P=1
        self.payoff_matrix = {
            (0, 0): (3, 3),  # Both Cooperate (Reward)
            (0, 1): (0, 5),  # You Cooperate, Opponent Defects (Sucker)
            (1, 0): (5, 0),  # You Defect, Opponent Cooperates (Temptation)
            (1, 1): (1, 1)   # Both Defect (Punishment)
        }

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
        reward1, reward2 = self.payoff_matrix[(action1, action2)]
        rewards = [float(reward1), float(reward2)]
        self.state = torch.tensor([action1, action2], dtype=torch.float32)
        # The game is "done" after each step in this simple representation
        return self.state, rewards, True, {}

# ============================================================================
# 3. Algorithm Implementations for IPD
# ============================================================================

# --- Learning Agents (DQN and PPO can be reused) ---

class DQNAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNAgent, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, output_dim))

    def forward(self, x):
        return self.network(x)

    def act(self, state: torch.Tensor) -> int:
        if len(state.shape) == 1: state = state.unsqueeze(0)
        with torch.no_grad():
            return self.forward(state).argmax().item()

class PPOAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, output_dim))
        self.critic = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.actor(x)

    def act(self, state: torch.Tensor) -> int:
        if len(state.shape) == 1: state = state.unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.forward(state), dim=-1)
            return torch.distributions.Categorical(probs).sample().item()

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
# --- END: CORRECTED Game-Theoretic Agent ---

# ============================================================================
# 4. Training and Evaluation Logic for IPD
# ============================================================================

def train_ipd_agent(agent, env, opponent, num_episodes=3000):
    print(f"Training {agent.__class__.__name__} against {opponent.name} in IPD...")
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)
    
    for episode in range(num_episodes):
        state = env.reset()
        opponent.reset()
        agent_action_history = []
        
        # Simple fixed-length episodes for training
        for _ in range(20):
            agent_action = agent.act(state)
            opponent_action = opponent.act(state, opponent_history=agent_action_history)
            agent_action_history.append(agent_action)

            next_state, rewards, _, _ = env.step([agent_action, opponent_action])
            reward = rewards[0]

            if isinstance(agent, DQNAgent):
                q_pred = agent(state.unsqueeze(0))[0, agent_action]
                loss = nn.functional.mse_loss(q_pred, torch.tensor(reward, dtype=torch.float32))
            elif isinstance(agent, PPOAgent):
                action_logits = agent(state.unsqueeze(0))
                critic_value = agent.critic(state.unsqueeze(0))
                advantage = reward - critic_value.detach().squeeze()
                log_prob = torch.distributions.Categorical(logits=action_logits).log_prob(torch.tensor(agent_action))
                actor_loss = -log_prob * advantage
                critic_loss = nn.functional.mse_loss(critic_value.squeeze(), torch.tensor(reward, dtype=torch.float32))
                loss = actor_loss + critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state
    
    print("Training finished.")
    return agent


if __name__ == "__main__":
    print("\n--- Setting up Gauntlet Benchmark for IPD Evaluation ---")
    config = EvaluationConfig(num_episodes=500, parallel_workers=1, save_visualizations=True)
    gauntlet = EnhancedGauntletBenchmark(config)

    # --- Register the IPD Environment with the Benchmark ---
    # The benchmark will use this factory to create IPD environments during evaluation.
    # The payoff matrix is used for advanced metrics like Nash Convergence.
    ipd_payoff_matrix = np.array([[3, 0], [5, 1]])
    gauntlet.register_environment("IPD", IPDEnvironment, payoff_matrix=ipd_payoff_matrix)

    # --- Instantiate Agents ---
    env = IPDEnvironment()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    dqn_agent = DQNAgent(input_dim, output_dim)
    ppo_agent = PPOAgent(input_dim, output_dim)
    
    # --- START: CORRECTED Training Opponent Selection ---
    # Fetch a suitable IPD opponent from the benchmark's master list.
    training_opponent = gauntlet.master_challenger_list['IPD_TitForTat']
    training_opponent.name = "TrainingOpponent_TFT"
    # --- END: CORRECTED Training Opponent Selection ---

    # --- Train Agents ---
    print("\n--- Starting IPD Training Phase ---")
    trained_dqn = train_ipd_agent(copy.deepcopy(dqn_agent), env, training_opponent)
    trained_ppo = train_ipd_agent(copy.deepcopy(ppo_agent), env, training_opponent)

    # --- Set up Policies for Evaluation ---
    policies_to_evaluate = {
        "IPD_DQN": trained_dqn,
        "IPD_PPO": trained_ppo
    }
    gauntlet.add_custom_challenger("IPD_FictitiousPlay", FictitiousPlayAgent())

    # --- Run Gauntlet Evaluation ---
    print("\n--- Starting IPD Benchmark Evaluation Phase ---")
    for name, policy in policies_to_evaluate.items():
        print(f"\n{'='*40}\nEVALUATING: {name.upper()}\n{'='*40}")
        policy.eval()
        # The benchmark will now automatically test against 'IPD_AlwaysCooperate',
        # 'IPD_AlwaysDefect', 'IPD_TitForTat', 'IPD_Uniform', AND our custom
        # 'IPD_FictitiousPlay' because their action spaces all match.
        gauntlet.evaluate_policy(policy=policy, policy_name=name, environments=["IPD"])

    print("\n\n🎉 IPD evaluation run has finished. 🎉")
    gauntlet.generate_report("ipd_evaluation_report.json")
    print("\nFinal IPD summary report saved to 'ipd_evaluation_report.json'")