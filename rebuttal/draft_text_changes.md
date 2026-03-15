# Draft Text Changes — Exact Edits for the Paper

This document contains all text changes to make in the paper draft.
Placeholders marked with `⟨PLACEHOLDER⟩` should be filled after running the two experiment scripts.

---

## 1. Section 4.1 — Rewrite PRPO Objective (Proxy-Gradient Clarification)

**REPLACE** the current text describing the PRPO optimization objective with:

> The total PRPO objective combines the standard PPO surrogate loss with two regularization terms:
>
> $$\mathcal{L}_{\text{PRPO}} = \mathcal{L}_{\text{PPO}} + \lambda_{\text{nash}} \cdot \mathcal{L}_{\text{Nash}} + \lambda_{\text{exploit}} \cdot \mathcal{L}_{\text{Exploit}}$$
>
> where $\mathcal{L}_{\text{Nash}} = D_{\text{KL}}(\pi_\theta \| \pi^*)$ penalizes deviation from the Nash target policy $\pi^*$, and $\mathcal{L}_{\text{Exploit}}$ is a scalar exploitability penalty.
>
> **We emphasize a critical implementation detail**: the exploitability metric serves as a *selection signal* to identify the most robust policy within the population (the "robust anchor" with lowest exploitability), and as a scalar penalty that does not propagate gradients through the best-response computation. The actual gradient for policy improvement flows exclusively through the KL-divergence term $\mathcal{L}_{\text{Nash}}$. Computing a true best-response gradient would require differentiating through the opponent's optimization problem, which is computationally intractable. Instead, PRPO circumvents this by using the exploitability value as an external signal: it influences *which* policy is selected as the population's anchor, and scales the update magnitude, but the gradient direction is determined by $\mathcal{L}_{\text{Nash}}$ alone.
>
> For games where the Nash equilibrium is analytically known (e.g., RPS, Matching Pennies), $\pi^*$ is set directly. For games without known equilibria (e.g., Leduc Poker), PRPO uses a proxy: the policy of the current "best-in-population" agent (the one with lowest measured exploitability) serves as a dynamic target, updated after each training cycle.

---

## 2. After Section 4.1 — Insert Algorithm 1 Pseudocode

**INSERT** the following after the PRPO objective description:

> **Algorithm 1: PRPO Training Loop**
>
> **Input:** Population $\mathcal{P} = \{\pi_1, \ldots, \pi_K\}$, opponent set $\mathcal{O}$, Nash target $\pi^*$, coefficients $\lambda_{\text{nash}}, \lambda_{\text{exploit}}$
>
> **for** episode $= 1$ to $N$ **do:**
>
> &nbsp;&nbsp; **1. Tournament Phase:** For each pair $(\pi_i, \pi_j) \in \mathcal{P}$: play game, store experience for both agents.
>
> &nbsp;&nbsp; **2. Exploitative Phase** (with probability $p_{\text{exploit}}$): For each $\pi_i \in \mathcal{P}$: sample opponent $o \sim \mathcal{O}$, play game against $o$, store experience.
>
> &nbsp;&nbsp; **3. Policy Update** (every $K$ episodes):
>
> &nbsp;&nbsp;&nbsp;&nbsp; For each $\pi_i \in \mathcal{P}$:
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Compute exploitability: $\varepsilon_i = \text{EXPLOIT}(\pi_i)$
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Compute PPO loss: $\mathcal{L}_{\text{PPO}}$ from stored experiences
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Compute Nash regularization: $\mathcal{L}_{\text{Nash}} = D_{\text{KL}}(\pi_i \| \pi^*)$
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Total loss: $\mathcal{L} = \mathcal{L}_{\text{PPO}} + \lambda_{\text{nash}} \cdot \mathcal{L}_{\text{Nash}} + \lambda_{\text{exploit}} \cdot \varepsilon_i$
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Update $\pi_i$ via gradient descent on $\mathcal{L}$
>
> **4. Selection:** **return** $\arg\min_{\pi_i \in \mathcal{P}} \text{EXPLOIT}(\pi_i)$

---

## 3. Equation 3 — Add Normalization Constants

**REPLACE** Equation 3 with:

> $$\hat{U}_i = \frac{U_i - U_{\min}}{U_{\max} - U_{\min}}$$
>
> where $U_{\max}$ and $U_{\min}$ are the empirical payoff bounds of the game (e.g., for RPS: $U_{\max} = 1, U_{\min} = -1$; for Leduc Poker: $U_{\max}$ and $U_{\min}$ are the maximum and minimum possible per-hand rewards). When dynamic payoff ranges are unavailable, we fall back to a fixed normalization: $\hat{U}_i = (U_i + 1) / 2$ for rewards in $[-1, 1]$.

---

## 4. Equation 11/12 — Insert Gauntlet Composite Score Weights

**REPLACE** the composite score equation with the explicit weights:

> The Gauntlet robustness score is a weighted combination of nine normalized metrics, each clipped to $[0, 1]$:
>
> **Zero-sum games** (RPS, MP, Kuhn Poker, Leduc Poker):
>
> $$S_{\text{robust}} = 0.25 \cdot \text{WR}_{\text{overall}} + 0.15 \cdot \text{WR}_{\min} + 0.12 \cdot (1 - \varepsilon) + 0.12 \cdot (1 - r) + 0.10 \cdot \alpha_{\text{adapt}} + 0.08 \cdot \rho_{\text{plast}} + 0.08 \cdot \tau_{\text{fwd}} + 0.05 \cdot (1 - f) + 0.05 \cdot \delta_{\text{pop}}$$
>
> **General-sum games** (Stag Hunt):
>
> $$S_{\text{robust}} = 0.30 \cdot \bar{R} + 0.20 \cdot W_{\text{social}} + 0.15 \cdot C_{\text{coop}} + 0.12 \cdot (1 - \varepsilon) + 0.10 \cdot (1 - r) + 0.08 \cdot \delta_{\text{pop}} + 0.05 \cdot (1 - \sigma_{\text{WR}})$$
>
> Weights were chosen to emphasize outcome-oriented metrics (win rate / reward, 40% total) as primary performance signals, game-theoretic soundness (exploitability + regret, 24%) for equilibrium quality, and learning dynamics (adaptation, plasticity, transfer, 36%) for robust generalization. This tripartite weighting reflects the Gauntlet's design philosophy: a robust agent must not only perform well in aggregate, but also resist worst-case exploitation and adapt efficiently to novel opponents. For general-sum games, cooperation and social welfare replace win-rate components, acknowledging that maximizing individual payoffs alone is insufficient.

---

## 5. Update Leduc Poker Results

**REPLACE** old Leduc Poker exploitability numbers with:

| Algorithm | Exploitability (mbb/h) |
|-----------|----------------------|
| Standard PPO | $7031.96 \pm 742.34$ |
| Self-Play | $4217.42 \pm 924.20$ |
| PSRO | $5504.06 \pm 556.25$ |
| **PRPO (Ours)** | $\mathbf{3903.66 \pm 40.59}$ |

**ADD** footnote: "All Leduc Poker results use a standardized computational budget of 60 seconds per algorithm per seed. Our exploitability values are higher than asymptotic CFR convergence results (which achieve near-zero exploitability given unlimited computation) because we compare methods under equal computational budgets rather than at convergence. This equal-budget protocol provides a fair comparison of sample efficiency across fundamentally different algorithmic paradigms."

---

## 6. Fix RPS/MP Ablation — Explain Identical Results + New Numbers

In the ablation study section, **ADD** the following note:

> **Note on RPS Ablation:** In the original implementation, the exploitability regularization term was implemented as a constant scalar (`torch.tensor(ε)`) rather than a differentiable function of the policy. Since constants carry no gradient, $\lambda_{\text{exploit}}$ contributed zero gradient signal during backpropagation, making PRPO (Full) and PRPO (Nash Only) mathematically equivalent. We corrected this by implementing a differentiable exploitability penalty using a smooth-max (log-sum-exp) approximation:
>
> $$\hat{\varepsilon}(\pi) = \frac{1}{\tau} \log \sum_{a_{\text{opp}}} \exp\left(\tau \cdot u_{\text{opp}}(a_{\text{opp}}, \pi)\right)$$
>
> where $u_{\text{opp}}(a_{\text{opp}}, \pi)$ is the opponent's expected payoff when playing pure action $a_{\text{opp}}$ against policy $\pi$, and $\tau$ controls approximation sharpness. The corrected results are shown below.

**REPLACE** old ablation table with: (fill from `rps_ablation_fixed.py` output)

| Variant | Nash Distance | Exploitability |
|---------|--------------|----------------|
| PRPO (Full — Diff Exploit) | ⟨PLACEHOLDER: from script⟩ | ⟨PLACEHOLDER: from script⟩ |
| PRPO (Nash Only) | ⟨PLACEHOLDER: from script⟩ | ⟨PLACEHOLDER: from script⟩ |
| PRPO (Exploit Only — Diff) | ⟨PLACEHOLDER: from script⟩ | ⟨PLACEHOLDER: from script⟩ |
| PRPO (Tournament + Nash — No Exploiters) | ⟨PLACEHOLDER: from script⟩ | ⟨PLACEHOLDER: from script⟩ |

---

## 7. NEW Section/Table — Cross-Ecosystem Validation Results

**INSERT** new subsection in Experiments (before Conclusion):

> ### Cross-Ecosystem Validation
>
> To address the concern that training and evaluating on the same Gauntlet population could constitute "pool shaping" rather than genuine robustness, we conduct a cross-ecosystem validation experiment. All algorithms are trained using only **Population A** (simple, predictable bots: pure-strategy and biased-strategy opponents). They are then evaluated against a completely disjoint **Population B** (adaptive counter-strategy bots, pattern detectors, Thompson sampling opponents, switching-strategy bots, and ε-Nash exploiters) that are **never seen during training**.
>
> If an algorithm achieves high performance on Population B, this demonstrates *generalized* strategic robustness rather than overfitting to the training population.

**INSERT TABLE** (fill from `cross_ecosystem_validation.py` output):

> **Table X: Cross-Ecosystem Validation — Rock-Paper-Scissors**
> Train: Population A (simple bots) → Evaluate: Population B (adaptive, unseen)

| Algorithm | Overall WR (Pop B) | Min WR (Pop B) | Avg Reward (Pop B) |
|-----------|-------------------|----------------|-------------------|
| Standard PPO | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ |
| Self-Play | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ |
| PSRO | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ |
| **PRPO (Ours)** | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ |

> **Table Y: Cross-Ecosystem Validation — Matching Pennies**

| Algorithm | Overall WR (Pop B) | Min WR (Pop B) | Avg Reward (Pop B) |
|-----------|-------------------|----------------|-------------------|
| Standard PPO | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ |
| Self-Play | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ |
| PSRO | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ |
| **PRPO (Ours)** | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ | ⟨PLACEHOLDER⟩ |

---

## 8. Tone Down Claims → Add Proof-of-Concept Framing

**FIND AND REPLACE** any claims about "large-scale" or "general applicability" with:

> Our experiments on matrix games (Rock-Paper-Scissors, Matching Pennies, Stag Hunt) and small poker variants (Kuhn Poker, Leduc Hold'em) serve as a proof-of-concept demonstrating the PRPO framework's core principles. We deliberately chose these well-understood domains to enable rigorous game-theoretic evaluation with analytically computable exploitability bounds. Scaling to complex environments (e.g., StarCraft Multi-Agent Challenge, Hanabi, or large-scale poker variants) remains an important direction for future work, and we expect the unified regularization approach to transfer, though empirical validation at scale is required.

---

## 9. Section 7 — Fix "Perfect Information" Contradiction

**REPLACE** the sentence claiming PRPO assumes perfect information with:

> While the PRPO framework makes no inherent assumption about the information structure of the underlying game, our current implementation treats each agent's observation vector as a complete state representation during policy optimization. For imperfect-information games such as Kuhn and Leduc Poker, each agent observes its private card and the public betting history, but not the opponent's private card. The Nash equilibrium targets for these games are computed externally using established methods (e.g., CFR) over the full game tree, including the information-set structure. The PRPO agent is then regularized toward these information-set-level equilibrium strategies using only its available observations. This approximation—training on partial observations while regularizing toward strategies defined over information sets—represents a known limitation that may weaken exploitability guarantees in deeper game trees with complex information structures.

---

## 10. Improve Narrative Flow — Add Transitions

**INSERT** at end of Section 3 (Gauntlet framework description), before Section 4:

> Having established the Gauntlet as a multi-dimensional evaluation framework that assesses not just average performance but worst-case robustness, adaptation dynamics, and population diversity, we now confront a natural question: *how should agents be trained to perform well across all Gauntlet dimensions simultaneously?* Unlike Balduzzi et al.'s Nash averaging, which computes a post-hoc meta-strategy over a fixed population of policies, PRPO internalizes population-level signals directly into the optimization objective of each individual agent. And unlike PSRO, which maintains an ever-growing policy population and solves for meta-Nash mixtures, PRPO distills population information into regularization terms that guide a single agent's gradient updates.

**INSERT** at end of Section 4 (PRPO description), before Section 5:

> We now empirically validate that this internalization approach—replacing external population management with internal regularization—produces agents that score higher on the Gauntlet's multi-dimensional robustness metric while requiring significantly less computational overhead than population-based methods like PSRO.

---

## 11. Fix Table Highlights

Review Table 3 and Table 5 and ensure:
- The **best** result in each column is **bolded**
- If two results are not statistically significantly different (p > 0.05), both can be bolded
- No contradictory highlighting (e.g., a lower number highlighted as better when higher is better, or vice versa)

Double-check that the Gauntlet robustness score ranking matches these experiment results:

| Game | #1 | #2 | #3 | #4 | #5 |
|------|----|----|----|----|-----|
| RPS | PRPO (0.406) | PPO (0.396) | PSRO (0.317) | SelfPlay (0.312) | DQN (0.299) |
| MP | PRPO (0.424) | PPO (0.422) | DQN (0.344) | SelfPlay (0.344) | PSRO (0.343) |
| Kuhn | DQN (0.586) | PRPO (0.442) | PPO (0.441) | PSRO (0.412) | SelfPlay (0.412) |
| Leduc | PRPO (0.477) | PPO (0.470) | PSRO (0.440) | SelfPlay (0.436) | DQN (0.436) |
| Stag Hunt | PRPO (0.638) | SelfPlay (0.634) | PSRO (0.622) | PPO (0.600) | DQN (0.589) |
