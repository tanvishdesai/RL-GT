### Pre-Writing Checklist

#### Tier 1: The Core Narrative & Claims
Before writing a single sentence of the paper, you should be able to crisply answer these questions. Write down one-paragraph answers to each. This will become the backbone of your abstract and introduction.

1.  **What is the central problem?** (e.g., "Existing MARL evaluations based on win-rate produce brittle policies that fail against diverse strategies...")
2.  **What is your core insight or solution?** (e.g., "We propose to fix this by combining a benchmark with a diverse challenger suite to expose weaknesses (The Gauntlet) with a new learning algorithm that directly optimizes for robustness by regularizing towards game-theoretic equilibria (PRPO).")
3.  **What is your single most important claim?** (e.g., "Across five canonical games, PRPO produces policies with statistically significant lower exploitability and higher strategic robustness scores on The Gauntlet compared to standard baselines like PPO, Self-Play, and PSRO.")

---

#### Tier 2: All Figures and Tables (Finalized & Publication-Ready)
This is the most time-consuming part. You should have a script that can generate every single one of these from your raw experimental data. **Do not create these by hand.**

**For the Main Paper:**

*   **[ ] Figure 1: Gauntlet Overview Diagram.** A conceptual diagram showing the flow: a policy enters, faces the diverse challenger suite (rule-based, adaptive, neural, etc.), and produces a multi-faceted report (robustness radar, heatmaps, etc.).
*   **[ ] Figure 2: PRPO Objective Diagram.** A conceptual diagram showing the PPO loss augmented with your two regularization terms (\(L_{target}\) and \(L_{exploit}\)).
*   **[ ] Table 1: Main Results - Composite Scores.** The master table. For each of the 5 games, it must show:
    *   Algorithms: DQN, PPO, Self-Play, PSRO, PRPO (Ours).
    *   Metrics: **Robustness Score**, **Average Win-Rate vs. Gauntlet**, **Exploitability/NashConv**.
    *   Format: **Mean ± 95% CI** across all 10 seeds.
    *   Statistical Significance: P-values for PRPO vs. each baseline on the key metrics.
*   **[ ] Table 2: Exploitability/NashConv Deep Dive.** A table focusing *only* on the game-theoretic metrics, showing the final exploitability values (with CIs and p-values) for each algorithm in each game.
*   **[ ] Figure 3-5: Per-Game Performance Panels.** For your most illustrative games (e.g., RPS, Leduc Poker, and Stag Hunt), generate the key visualizations from the Gauntlet report:
    *   Bar chart of win-rate against each challenger.
    *   The final **Robustness Radar Chart** for PRPO vs. the best baseline.
*   **[ ] Figure 6: Ablation Study Results.** A bar chart showing the impact on a key metric (e.g., Robustness Score or Exploitability) for:
    *   Full PRPO
    *   PRPO without the \(\lambda_{nash}\) term
    *   PRPO without the \(\lambda_{exploit}\) term
    (This figure is **non-negotiable** and provides the evidence for *why* PRPO works).
*   **[ ] Figure 7: Hyperparameter Sensitivity.** A heatmap showing the Robustness Score as you sweep \(\lambda_{target}\) and \(\lambda_{exploit}\). This shows your method is stable.

**For the Appendix:**

*   **[ ] Full Cross-Play Matrices:** Heatmaps for the final populations of PRPO and PSRO.
*   **[ ] Full Results Tables:** Detailed tables with per-challenger win rates for every algorithm on every game.
*   **[ ] Learning Curves:** Plots showing a key metric (e.g., average reward or exploitability) over training time/episodes for all algorithms.

---

#### Tier 3: Code and Reproducibility Artifacts

*   **[ ] Finalized Codebase:** Create a "paper-submission" branch in Git. Clean up the code, add comments where necessary, and freeze it. No more changes.
*   **[ ] A Single Script to Run All Experiments:** A shell script (`run_all.sh`) that can re-run all training and evaluation experiments from scratch.
*   **[ ] A Single Script to Generate All Figures/Tables:** A Python script (`generate_paper_figures.py`) that takes the raw data from the experiments and produces every single figure and table listed above in PDF/PNG format. This is critical for consistency.
*   **[ ] `requirements.txt` / `environment.yml`:** A file specifying all dependencies and their exact versions.
*   **[ ] A Detailed `README.md`:** Instructions on how to set up the environment and use your two scripts (`run_all.sh` and `generate_paper_figures.py`) to reproduce your entire paper.

---

#### Tier 4: Supporting Textual Content

*   **[ ] Bibliography File:** Start a `.bib` file (e.g., `references.bib`) and add the key papers you will be citing for PSRO, exploitability, MARL benchmarks, etc.
*   **[ ] Formal Definitions:** Write down the precise mathematical definitions you will use for Nash Equilibrium, Exploitability, Correlated Equilibrium, etc., in a separate text file to ensure you use them consistently.

Once you have checked every box on this list, you are no longer just "doing research." You are ready to *write the paper*. The writing process will be dramatically faster and less stressful because you are simply describing the complete and finalized body of evidence you have already prepared.