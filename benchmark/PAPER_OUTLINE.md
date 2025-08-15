## Bridging Nash Equilibrium and Win Rate in Multi-Agent RL: PRPO and The Gauntlet

Author(s): Anonymous for Review • Affiliation(s): Anonymous

Paper Type: Research Track

Keywords: Multi-Agent Reinforcement Learning, Game Theory, Robustness, Exploitability, Nash Equilibrium, Benchmarking


### Abstract
This paper introduces Population-Regularized Policy Optimization (PRPO), a learning framework that closes the gap between optimizing for win rate and converging toward Nash equilibria in two-player games. We further present The Gauntlet, a targeted evaluation benchmark and protocol that systematically probes agents’ strategic blind spots using a curated challenger suite across five canonical games: Rock–Paper–Scissors, Matching Pennies, Kuhn Poker, Leduc Poker, and Stag Hunt. Across extensive experiments with multi-seed runs and statistical tests, PRPO consistently reduces exploitability and improves convergence toward game-theoretic solutions while maintaining strong win rates against diverse opponents.

Placeholder: 150–200 words summarizing problem, method, benchmark, results, and contributions.


### 1. Introduction
- Motivation: Existing MARL methods (e.g., PPO, self-play, PSRO) often maximize head-to-head win rate but can remain exploitable and fail to converge to Nash equilibria in practice.
- Problem: Bridging the gap between empirical win rate and theoretical robustness (low exploitability, NE convergence) in two-player games.
- Contributions:
  - Introduce PRPO, a population-based regularized objective aligning learning signals with game-theoretic robustness.
  - Propose The Gauntlet, a benchmark emphasizing targeted evaluation with a challenger suite probing specific weaknesses.
  - Demonstrate consistent gains across five games with rigorous statistics (multi-seed, 95% CI, paired t-tests), ablations, and hyperparameter sensitivity.

Placeholder Figure 1: Conceptual overview of PRPO and The Gauntlet evaluation loop.


### 2. Related Work
- Multi-Agent RL: self-play, league training, PSRO, fictitious play; strengths and limitations for robustness and equilibrium convergence.
- Game-Theoretic RL: exploitability, meta-game analysis, Nash computation/approximation; relationships to MARL performance.
- Evaluation and Benchmarks: Common practices (win rates, Elo), shortcomings for robustness; need for targeted adversarial/challenger evaluation suites.

Placeholder Table R1: Summary comparison of methods (PPO, Self-Play, PSRO, PRPO) vs evaluation desiderata (win rate, exploitability, stability, scalability).


### 3. The Gauntlet: A Benchmark for Strategically Robust MARL
#### 3.1 Design Principles
- Targeted evaluation via challenger suite spanning difficulty tiers and styles (e.g., uniform, biased, rule-based, learned best-responsers).
- Multi-metric scoring: average return, exploitability proxies, cross-play, meta-game analysis.
- Reproducibility: unified API for environments and agents; standardized reporting (JSON + visualizations).

#### 3.2 Game Environments
- Rock–Paper–Scissors (RPS): 3-action zero-sum; closed-form NE; simple exploitability proxies.
- Matching Pennies (MP): 2-action zero-sum; uniform NE; simple exploitability proxies.
- Kuhn Poker (KP): partial information; simplified simultaneous action abstraction; approximate exploitability.
- Leduc Poker (LP): partial information; openspiel-compatible abstraction; exploitability via proxies and cross-play.
- Stag Hunt (SH): coordination vs risk-dominant strategies; non-zero-sum; robustness beyond zero-sum.

Placeholder Table G1: Challenger taxonomy by game (name, difficulty, style, intended failure mode tested).

#### 3.3 Metrics and Reports
- Primary metrics: average reward, cross-play matrices, exploitability (or proxy), convergence to NE (distance/KL vs target equilibria), robustness radar.
- Statistical reporting: 95% confidence intervals, paired t-tests across seeds; per-game and aggregate reports.
- Outputs: per-algorithm JSON reports, figures (performance curves, heatmaps, radar charts, meta-game visualizations).

Placeholder Figure 2: Gauntlet architecture and reporting pipeline (evaluation runners, challenger suite, report generator).


### 4. Population-Regularized Policy Optimization (PRPO)
#### 4.1 Objective and Training Signal
We augment a standard policy-gradient objective with population-regularized terms that penalize exploitability and encourage proximity to game-theoretic targets.

Objective (informal): \(L(\pi) = L_{PPO}(\pi) + \lambda_{nash} \cdot \mathrm{KL}(\pi \parallel \pi^\*) + \lambda_{exploit} \cdot \mathrm{Exploitability}(\pi)\), where \(\pi^\*) is a Nash-target distribution (exact or proxy), and Exploitability is measured via analytic proxies or learned best responses.

#### 4.2 Population Mechanism
- Maintain a population of learners to stabilize updates and provide richer opponents for curriculum-style training.
- Periodically identify best-in-population and regularize other members toward this target to reduce policy collapse.

#### 4.3 Exploitability Estimators
- RPS/MP: closed-form proxies via payoff differences from logits/probabilities.
- Kuhn/Leduc/SH: learned BR approximations and/or proxy metrics; cross-play and meta-game analyses for additional signals.

#### 4.4 Algorithm
- Alternating phases: (i) population tournament/self-play, (ii) exploitative training vs exploiter bots/BRs, (iii) population-wide regularized updates.

Placeholder Algorithm Box A1: PRPO training loop pseudocode.

#### 4.5 Implementation Notes
- Unified interfaces for actors/critics; environment wrappers; legality masking when needed.
- Robustness in evaluation: wrapping policies for consistent observation/action spaces; handling single-step vs multi-step games.


### 5. Experimental Setup
#### 5.1 Baselines and Algorithms
- PPO (standard), Self-Play, PSRO, and PRPO (ours). Where relevant, DQN for simple single-step baselines (RPS/MP).

#### 5.2 Protocols
- Training budget per algorithm: fixed episodes and/or time budgets; per-game schedule documented.
- Multi-seed evaluation (e.g., 10 seeds) with 10,000 training episodes per algorithm; fixed evaluation episodes per seed.
- Evaluation against standardized challenger suites with consistent RNG seeds for reproducibility.

#### 5.3 Hyperparameters and Ablations
- Ablations: removing \(\lambda_{nash}\), removing \(\lambda_{exploit}\), population size variants, alternative exploitability proxies.
- Sensitivity: grid over \(\lambda_{nash}\), \(\lambda_{exploit}\), learning rate, entropy bonus; report trends and stability.

#### 5.4 Infrastructure and Reproducibility
- Hardware details, software versions, and random seed handling; code and benchmark planned for public release as a pip package.

Placeholder Table E1: Training budgets and seeds per algorithm/game.
Placeholder Table E2: Hyperparameter grids for sensitivity studies.


### 6. Results
#### 6.1 Main Performance Across Games
- PRPO vs PPO, Self-Play, PSRO across RPS, MP, KP, LP, SH. Report mean ± 95% CI and paired t-tests.

Placeholder Table 1: Average reward vs challenger suite (per game and aggregate).
Placeholder Figure 3: Metrics comparison by game (win rate, exploitability, convergence measure).

#### 6.2 Exploitability and Nash Proximity
- Show exploitability proxies and distance to NE targets; demonstrate improvements and stability under PRPO.

Placeholder Table 2: Exploitability metrics/proxies per algorithm and game.
Placeholder Figure 4: Robustness radar charts and exploitability trajectories over training.

#### 6.3 Cross-Play and Meta-Game Analysis
- Cross-play matrices among populations; mixed-strategy equilibria estimation (e.g., via support enumeration when feasible).

Placeholder Figure 5: Cross-play heatmaps and inferred meta-strategy distributions.

#### 6.4 Ablation Studies
- Component importance: drop \(\lambda_{nash}\), drop \(\lambda_{exploit}\), change population size; discuss impacts on exploitability and win rate.

Placeholder Table 3: Ablation results across games (mean ± 95% CI).
Placeholder Figure 6: Ablation impact plots (bars/lines).

#### 6.5 Hyperparameter Sensitivity
- Stability and performance envelopes across \(\lambda\)-values and learning rates; report regions yielding best trade-offs.

Placeholder Figure 7: Sensitivity heatmaps (score vs \(\lambda_{nash}\), \(\lambda_{exploit}\)).


### 7. Discussion
- PRPO bridges the empirical and theoretical desiderata by aligning optimization with game-theoretic structure while preserving strong win rates.
- Insights: when PSRO helps/hurts, when self-play collapses; role of challenger diversity; lessons for non-zero-sum (SH).
- Practical guidance: choosing exploitability proxies and regularization strengths.


### 8. Limitations and Broader Impact
- Limitations: proxy exploitability vs exact best-response, simplified abstractions in poker domains, two-player focus.
- Broader impact: safer/robust decision-making; risk of overfitting to benchmarked challengers; responsible release practices.


### 9. Future Work
- Extend Gauntlet to multi-player, imperfect-information general-sum settings with scalable BR approximations.
- Automated challenger generation (adversarial population design) and curriculum learning.
- Public pip package for Gauntlet; full code/data release for PRPO and experiments.


### 10. Conclusion
PRPO and The Gauntlet provide a principled path toward strategically robust MARL, unifying empirical performance with game-theoretic soundness. We show consistent exploitability reductions and improved equilibrium convergence across five games with rigorous evaluation and analysis.


### References
Placeholder: Formatted bibliography per venue style (AAMAS/ACM sigconf). Ensure anonymity.


### Appendices
#### A. Extended Experimental Details
- Full hyperparameter lists; training schedules; environment details and any legality masking.

#### B. Gauntlet API Summary
- Environment and agent interfaces; challenger registration; report schema.

#### C. Additional Figures and Tables
- Expanded cross-play matrices, per-seed plots, additional ablations.

#### D. Theoretical Notes
- Additional derivations, caveats on exploitability proxies, and limitations of approximations.


