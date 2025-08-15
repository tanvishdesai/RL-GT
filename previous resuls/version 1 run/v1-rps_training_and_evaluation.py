import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from typing import Optional, List

# ============================================================================
# 1. IMPORT FROM YOUR BENCHMARK FILE
# ============================================================================
# NOTE: Assumes your benchmark file is named 'gauntlet_benchmark.py'
# We import all the necessary components for clarity.


# ============================================================================
# 2. MARL Algorithm Implementations
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


class DQNAgent(nn.Module):
    """A simple Deep Q-Network agent, with an 'act' method for evaluation."""
    def __init__(self, input_dim, output_dim):
        super(DQNAgent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)
    
    def act(self, state: torch.Tensor) -> int:
        """Selects the best action based on Q-values for evaluation."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()


class PPOAgent(nn.Module):
    """A PPO agent, with an 'act' method for evaluation."""
    def __init__(self, input_dim, output_dim):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        """The forward pass for training returns actor logits."""
        return self.actor(x)

    def act(self, state: torch.Tensor) -> int:
        """Selects an action stochastically based on policy logits for evaluation."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            action_logits = self.forward(state)
            probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            return dist.sample().item()

# ============================================================================
# 3. Training Logic
# ============================================================================

def train_agent(agent, env, opponent, num_episodes=2000):
    """A robust training loop for a single agent against a fixed opponent."""
    print(f"Starting training for {agent.__class__.__name__} against {opponent.name}...")
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    
    for episode in range(num_episodes):
        state = env.reset()
        opponent.reset()
        # History needs to be managed per episode
        agent_action_history = [] 

        for step in range(10): # A few steps per episode
            # Agent selects action
            agent_action = agent.act(state)

            # Opponent selects action, using the agent's history
            opponent_action = opponent.act(state, opponent_history=agent_action_history)
            
            # Record agent's action for the opponent's next turn
            agent_action_history.append(agent_action)

            # Environment step
            next_state, rewards, done, _ = env.step([agent_action, opponent_action])
            reward = rewards[0]

            # --- Learning Update ---
            state_batch = state.unsqueeze(0)
            if isinstance(agent, DQNAgent):
                q_pred = agent(state_batch)[0, agent_action]
                with torch.no_grad():
                    target_q = reward + 0.99 * agent(next_state.unsqueeze(0)).max()
                loss = nn.functional.mse_loss(q_pred, target_q)
            
            elif isinstance(agent, PPOAgent):
                action_logits = agent(state_batch)
                critic_value = agent.critic(state_batch)
                dist = torch.distributions.Categorical(logits=action_logits)
                log_prob = dist.log_prob(torch.tensor(agent_action))
                advantage = reward - critic_value.detach().squeeze()
                
                actor_loss = -log_prob * advantage
                critic_loss = nn.functional.mse_loss(critic_value.squeeze(), torch.tensor(reward, dtype=torch.float32))
                loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            if done:
                break
            
    print(f"Training finished for {agent.__class__.__name__}.")
    return agent

# ============================================================================
# 4. Main Execution
# ============================================================================

if __name__ == "__main__":
    # --- Set up for evaluation using the BENCHMARK's components ---
    print("\n--- Setting up Gauntlet Benchmark for Evaluation ---")
    
    config = EvaluationConfig(
        num_episodes=200,
        parallel_workers=1,
        save_visualizations=True,
        # Exploitability can be enabled, as the benchmark now has built-in adversaries
        compute_exploitability=True,
    )
    gauntlet = EnhancedGauntletBenchmark(config)
    
    # --- Use the BENCHMARK'S default environment for consistency ---
    print("Initializing environment from the benchmark...")
    env_factory = gauntlet._create_default_environment
    training_env = env_factory()
    input_dim = training_env.observation_space.shape[0]
    output_dim = training_env.action_space.n
    
    # Instantiate learning agents
    dqn_agent = DQNAgent(input_dim, output_dim)
    ppo_agent = PPOAgent(input_dim, output_dim)
    
    # A dedicated opponent for the training phase
    training_opponent = RuleBasedAgent(name="TrainingOpponent")

    # --- Train the learning agents ---
    print("\n--- Starting Training Phase ---")
    trained_dqn_agent = train_agent(copy.deepcopy(dqn_agent), training_env, training_opponent)
    trained_ppo_agent = train_agent(copy.deepcopy(ppo_agent), training_env, training_opponent)

    # --- Create the dictionary of policies to be evaluated ---
    policies_to_evaluate = {
        "DQN": trained_dqn_agent,
        "PPO": trained_ppo_agent
    }

    # --- Add the custom rule-based agent as a challenger for the evaluation phase ---
    # The benchmark will now automatically check if its action space is compatible.
    gauntlet.add_custom_challenger("Student_RuleBased", RuleBasedAgent())

    # --- Run the final evaluation ---
    print("\n--- Starting Benchmark Evaluation Phase ---")
    
    for policy_name, policy in policies_to_evaluate.items():
        print(f"\n{'='*40}")
        print(f" E V A L U A T I N G:   {policy_name.upper()} ")
        print(f"{'='*40}")
        
        policy.eval() # Set model to evaluation mode

        try:
            # The benchmark will automatically find all compatible challengers from its
            # master list (e.g., RPS_AlwaysRock, RPS_AdaptiveCounter) AND our
            # custom "Student_RuleBased" challenger for the default RPS environment.
            metrics = gauntlet.evaluate_policy(
                policy=policy, 
                policy_name=policy_name,
                environments=None # This correctly defaults to the registered RPS environment
            )
            print(f"\n✅ Evaluation for {policy_name} completed.")
        except Exception as e:
            print(f"\n❌ An error occurred during the evaluation of {policy_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n🎉 Benchmark evaluation run has finished for all policies. 🎉")
    final_report = gauntlet.generate_report("final_evaluation_report.json")
    print("\nFinal summary report saved to 'final_evaluation_report.json'")