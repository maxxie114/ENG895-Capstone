# Dynamic Benchmarking of LLMs: Moving Beyond Static Evaluation

**ENG 895 Capstone Paper — First Draft**
San Francisco State University

---

## Research Question

> Compared with static evaluation benchmarks like MMLU, how do dynamic evaluations (LiveBench) and formal metalinguistic evaluations (the Beguš et al. linguistic dataset) alter the performance rankings and assessed reasoning capabilities of contemporary LLMs?

**Hypothesis:** While frontier models will continue to outperform baseline models across all formats, the performance gap will shrink under dynamic and metalinguistic conditions where models cannot rely on data contamination.

---

## Overview

This paper investigates the limitations of static LLM benchmarks and evaluates whether dynamic and metalinguistic evaluation methods provide a more accurate picture of model capabilities. As LLMs ingest massive portions of the internet, static benchmarks face a high risk of test set leakage — high scores may reflect statistical familiarity with training data rather than genuine reasoning ability.

---

## Paper Structure

### I. Introduction and Background
- The rapid advancement and deployment of LLMs
- The vulnerability of static evaluation and the data contamination problem
- Thesis and research question

### II. Literature Review: Assessing the Evaluation Frameworks
- **Traditional Static Benchmarking** — MMLU & GLUE
- **The Shift Toward Dynamic Benchmarks** — LiveBench
- **Evaluating Metalinguistic Abilities in LLMs** — Beguš et al. (2025)

### III. Methodology
- **Models Evaluated** (all released early 2026):
  - OpenAI GPT-5.4 (released March 5, 2026)
  - Anthropic Claude Sonnet 4.6 (released February 17, 2026)
  - MiniMax M2.7 (released March 18, 2026)
- **Benchmark 1 — Static Evaluation:** MMLU (57-subject multiple-choice)
- **Benchmark 2 — Dynamic Evaluation:** LiveBench (frequently-updated, contamination-resistant)
- **Benchmark 3 — Metalinguistic Evaluation:** Beguš et al. 120-item dataset (syntax, phonology, recursion, ambiguity)

### IV. Expected Results and Analysis
- Performance metrics and rank stability across benchmarks
- Analysis of metalinguistic capabilities
- Evidence of data contamination impacts (MMLU vs. LiveBench performance delta)

### V. Conclusion
- Synthesis of findings across evaluation formats
- Implications for computational linguistics
- Recommendations for living/dynamic benchmarks

---

## Repository Contents

| File | Description |
|------|-------------|
| `ENG895 Outline V2.pdf` | Detailed capstone outline (latest version) |
| `ENG895 Capstone Outline (1).pdf` | Original capstone outline |
| `citations.txt` | All references in APA format |
| `hendrycks-MMLU.pdf` | Hendrycks et al. (2021) — MMLU |
| `wang-GLUE.pdf` | Wang et al. (2018) — GLUE |
| `white-LiveBench.pdf` | White et al. (2024) — LiveBench |
| `liang-HELM.pdf` | Liang et al. (2022) — HELM |
| `begus-LLM-Metalinguistic.pdf` | Beguš et al. (2025) — Metalinguistic abilities |
| `jimenez-SWE-bench.pdf` | Jimenez et al. (2024) — SWE-bench |
| `li-Arena-Hard.pdf` | Li et al. (2024) — Arena-Hard |
| `zhou-WebArena.pdf` | Zhou et al. (2023) — WebArena |
| `openai-GPT-5.4.pdf` | OpenAI (2026) — Introducing GPT-5.4 |
| `anthropic-Claude_Sonnet_4.6.pdf` | Anthropic (2026) — Claude Sonnet 4.6 System Card |
| `minimax-M2.7.pdf` | MiniMax (2026) — MiniMax M2.7 |
| `glm_team-GLM-5.pdf` | GLM-5 Team (2026) — GLM-5 |

---

## References

Anthropic. (2026, February 17). *System card: Claude Sonnet 4.6*. https://www.anthropic.com/claude-sonnet-4-6-system-card

Beguš, G., Dąbkowski, M., & Rhodes, R. (2025). Large linguistic models: Investigating LLMs' metalinguistic abilities. *IEEE Transactions on Artificial Intelligence, 6*(4), 3454–3467.

GLM-5 Team, Zeng, A., Lv, X., Hou, Z., Du, Z., Zheng, Q., et al. (2026, February 17). *GLM-5: From vibe coding to agentic engineering*. arXiv preprint arXiv:2602.15763.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). Measuring massive multitask language understanding. *International Conference on Learning Representations (ICLR)*.

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024). SWE-bench: Can language models resolve real-world GitHub issues? *The Twelfth International Conference on Learning Representations (ICLR)*.

Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Wu, T., Zhu, B., Gonzalez, J. E., & Stoica, I. (2024). From crowdsourced data to high-quality benchmarks: Arena-Hard and BenchBuilder pipeline. *arXiv preprint arXiv:2406.11939*.

Liang, P., Bommasani, R., Lee, T., Tsipras, D., Soylu, D., Yasunaga, M., ... & Koreeda, Y. (2022). Holistic evaluation of language models. *arXiv preprint arXiv:2211.09110*.

MiniMax. (2026, March 18). *MiniMax M2.7: Early echoes of self-evolution*. https://www.minimax.io/news/minimax-m27-en

OpenAI. (2026, March 5). *Introducing GPT-5.4*. https://openai.com/index/introducing-gpt-5-4/

Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S. R. (2018). GLUE: A multi-task benchmark and analysis platform for natural language understanding. *Proceedings of the 2018 EMNLP Workshop BlackboxNLP*, 353–355.

White, C., Dooley, S., Roberts, M., Pal, A., Feuer, B., Jain, S., ... & Goldblum, M. (2024). LiveBench: A challenging, contamination-limited LLM benchmark. *arXiv preprint arXiv:2406.19314*.

Zhou, S., Xu, F. F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., ... & Neubig, G. (2023). WebArena: A realistic web environment for building autonomous agents. *arXiv preprint arXiv:2307.13854*.
