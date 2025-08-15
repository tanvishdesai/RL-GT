import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from typing import Optional, List, Tuple, Dict

# ============================================================================
# 1. IMPORT FROM YOUR BENCHMARK FILE
# ============================================================================
# NOTE: Assumes your benchmark file is named 'gauntlet_benchmark.py'



# ============================================================================
# 2. Matching Pennies Environment Implementation
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

# ============================================================================
# 3. Algorithm Implementations for Matching Pennies
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
# 4. Training and Evaluation Logic
# ============================================================================

def train_pennies_agent(agent, env, opponent, num_episodes=4000):
    print(f"Training {agent.__class__.__name__} against {opponent.name} in Matching Pennies...")
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)
    
    for episode in range(num_episodes):
        state = env.reset()
        opponent.reset()
        agent_action_history = []
        
        for _ in range(10): # Short episodes
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
    print("\n--- Setting up Gauntlet Benchmark for Matching Pennies Evaluation ---")
    config = EvaluationConfig(num_episodes=1000, parallel_workers=1, save_visualizations=True)
    gauntlet = EnhancedGauntletBenchmark(config)

    # --- Register the Matching Pennies Environment ---
    # The payoff matrix for Player 1 (Matcher)
    pennies_payoff_matrix = np.array([[1, -1], [-1, 1]])
    # Add the game_prefix to explicitly link this environment to "Pennies_" challengers.
    gauntlet.register_environment(
        "MatchingPennies", 
        MatchingPenniesEnvironment, 
        payoff_matrix=pennies_payoff_matrix,
        game_prefix="Pennies"  # This is the crucial addition
    )

    # --- Instantiate Agents ---
    env = MatchingPenniesEnvironment()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    dqn_agent = DQNAgent(input_dim, output_dim)
    ppo_agent = PPOAgent(input_dim, output_dim)
    
    # Select a simple opponent from the benchmark's list for training
    training_opponent = gauntlet.master_challenger_list['Pennies_Uniform']
    training_opponent.name = "TrainingOpponent_Uniform"

    # --- Train Agents ---
    print("\n--- Starting Matching Pennies Training Phase ---")
    trained_dqn = train_pennies_agent(copy.deepcopy(dqn_agent), env, training_opponent)
    trained_ppo = train_pennies_agent(copy.deepcopy(ppo_agent), env, training_opponent)

    # --- Set up Policies for Evaluation ---
    policies_to_evaluate = {
        "Pennies_DQN": trained_dqn,
        "Pennies_PPO": trained_ppo
    }
    # Add our new custom challenger, ensuring its name starts with the environment name
    gauntlet.add_custom_challenger("Pennies_FrequencyCounter", FrequencyCounterAgent())

    # --- Run Gauntlet Evaluation ---
    print("\n--- Starting Matching Pennies Benchmark Evaluation Phase ---")
    for name, policy in policies_to_evaluate.items():
        print(f"\n{'='*40}\nEVALUATING: {name.upper()}\n{'='*40}")
        policy.eval()
        # The benchmark will now automatically test against 'Pennies_AlwaysHeads',
        # 'Pennies_Uniform', and our custom 'Pennies_FrequencyCounter' agent.
        gauntlet.evaluate_policy(policy=policy, policy_name=name, environments=["MatchingPennies"])

    print("\n\n🎉 Matching Pennies evaluation run has finished. 🎉")
    gauntlet.generate_report("matching_pennies_report.json")
    print("\nFinal Matching Pennies summary report saved to 'matching_pennies_report.json'")