# Q3b Citation Verification

## 1. 2310.11207 — Huang et al. 2023
- Confirmed title: "Can Large Language Models Explain Themselves? A Study of LLM-Generated Self-Explanations"
- Authors (page 1): Shiyuan Huang, Siddarth Mamidanna, Shreedhar Jangam, Yilun Zhou, Leilani H. Gilpin (UC Santa Cruz / MIT CSAIL)
- Year: 2023 (per citation; not printed on page 1, arXiv ID prefix 2310 = Oct 2023)
- Quote: "we study different ways to elicit the self-explanations, evaluate their faithfulness on a set of evaluation metrics, and compare them to traditional explanation methods such as occlusion or LIME saliency maps. Through an extensive set of experiments, we find that ChatGPT's self-explanations perform on par with traditional ones, but are quite different from them according to various agreement metrics, meanwhile being much cheaper to produce"
- Tag: VERIFIED

## 2. 2401.07927 — Madsen et al. 2024
- Confirmed title: "Are self-explanations from Large Language Models faithful?"
- Authors (page 1): Andreas Madsen, Sarath Chandar, Siva Reddy (Mila – Quebec AI Institute / Polytechnique Montréal / McGill University)
- Year: 2024 (arXiv ID prefix 2401 = Jan 2024)
- Quote: "Figure 8: Faithfulness evaluation using self-consistency checks, evaluated using Llama2-70B. Results show that Llama2-70B is not affected by prompt variations, but the faithfulness for each explanation type is task-dependent."
- Tag: VERIFIED

## 3. 2311.07466 — Parcalabescu & Frank (ID CORRECTED from 2311.13725)
- NOTE: the ID given in the task, 2310.11207... i.e. **2311.13725**, resolves to an unrelated paper ("Studying Artist Sentiments around AI-generated Artwork" by Ali & Breazeal, MIT). The correct arXiv ID for the CC-SHAP paper, found via web search and confirmed by grepping its PDF, is **2311.07466**.
- Confirmed title: "On Measuring Faithfulness or Self-consistency of Natural Language Explanations"
- Authors (page 1): Letitia Parcalabescu, Anette Frank (Computational Linguistics Department, Heidelberg University)
- Year: 2024 (ACL 2024; PDF header shows arXiv:2311.07466v4, 18 Sep 2024)
- Quote: "constructing a Comparative Consistency Bank for self-consistency tests that for the first time compares existing tests on a common suite of 11 open LLMs and 5 tasks – including iii) our new self-consistency measure CC-SHAP. CC-SHAP is a fine-grained measure (not a test) of LLM self-consistency. It compares how a model's input contributes to the predicted answer and to generating the explanation."
- Tag: VERIFIED (under corrected ID 2311.07466 — the cited ID 2311.13725 is UNVERIFIED/wrong, points to a different paper)

## 4. 2305.04388 — Turpin et al. 2023
- Confirmed title: "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting"
- Authors (page 1): Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman (NYU Alignment Research Group / Cohere / Anthropic)
- Year: 2023 (arXiv ID prefix 2305 = May 2023)
- Quote: "With BIG-Bench Hard (§3), we investigate two biasing features: (1) Answer is Always A, where we reorder all multiple-choice answer options in a few-shot prompt so the correct one is always "(A)", and (2) Suggested Answer... Our main findings are as follows: 1. Adding biasing features heavily influences model CoT predictions on BBH tasks, causing accuracy to drop as much as 36%, despite the biasing features never being referenced in the CoT explanations."
- Tag: VERIFIED

## 5. 2307.13702 — Lanham et al. 2023
- Confirmed title: "Measuring Faithfulness in Chain-of-Thought Reasoning"
- Authors (page 1): Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamile Lukosiute, Karina Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, [et al.] (large author list, page 1 truncated)
- Year: 2023 (arXiv:2307.13702v1, 17 Jul 2023)
- Quote: "Post-hoc reasoning: The model's reasoning may be post-hoc, i.e., produced after a certain conclusion has already been guaranteed... In this work, we test for post-hoc reasoning by truncating the chain of thought or adding mistakes to it. We find great variation in how much LLMs use CoT on different tasks, not using CoT at all for some tasks while relying upon it heavily for other tasks."
- Tag: VERIFIED

## 6. 2505.05410 — Chen et al. 2025
- Confirmed title: "Reasoning Models Don't Always Say What They Think"
- Authors (page 1): Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, Vlad Mikulik, Samuel R. Bowman, Jan Leike, Jared Kaplan, Ethan Perez (Alignment Science Team, Anthropic)
- Year: 2025 (arXiv:2505.05410v1, 8 May 2025)
- Quote: "CoTs of reasoning models often lack faithfulness and can conceal misalignment. The overall faithfulness scores for both reasoning models remain low (25% for Claude 3.7 Sonnet and 39% for DeepSeek R1) (Figure 1)."
- Tag: VERIFIED

## 7. 2410.13787 — Binder et al. 2024
- Confirmed title: "LOOKING INWARD: LANGUAGE MODELS CAN LEARN ABOUT THEMSELVES BY INTROSPECTION"
- Authors (page 1): Felix J Binder, James Chua, Tomek Korbak, Henry Sleight, John Hughes, Robert Long, Ethan Perez, Miles Turpin, Owa[in ...] (UC San Diego / Stanford / Truthful AI / Independent / MATS / Speechmatics / Eleos AI / Anthropic / Scale AI / NYU — list truncated on page)
- Year: 2024 (arXiv ID prefix 2410 = Oct 2024)
- Quote: "Llama 70B predicts its own behavior more accurately (48.5%) than GPT-4o (31.8%), despite GPT-4o's superior capabilities... The same pattern holds the other way around: GPT-4o predicts itself better (49.4%) than Llama 70B does (36.6%)."
- Tag: VERIFIED

## 8. 2504.16574 — PIS
- Confirmed title: "PIS: Linking Importance Sampling and Attention Mechanisms for Efficient Prompt Compression"
- Authors (page 1): Lizhe Chen, Binjia Zhou, Yuyao Ge, Jiayi Chen, Shiguang Ni (Tsinghua University / Zhejiang University / CAS Key Laboratory of AI Security)
- Year: 2025 (arXiv ID prefix 2504 = Apr 2025)
- Quote: "This variance-based criterion improves reasoning quality... Additionally, relying solely on attention scores may lead to the removal of important tokens with high attention scores. To address this, we introduce TF-IDF scores as a corrective measure for attention scores of each token."
- Tag: VERIFIED

## 9. 2512.12411 — "Detecting the Disturbance"
- Confirmed title: "Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs"
- Authors (page 1): Ely Hahami, Ishaan Sinha, Lavik Jain, Josh Kaplan, Jon Hahami
- Year: printed as arXiv:2512.12411v2 [cs.AI] 1 Mar 2026 (submission ID prefix 2512 = Dec 2025)
- Quote: "on tasks requiring differential sensitivity, we find robust evidence for partial introspection: models localize which of 10 sentences received an injection at up to 88% accuracy (vs. 10% chance) and discriminate relative injection strengths at 83% accuracy (vs. 50% chance). These capabilities are confined to early-layer injections and collapse to chance thereafter"
- Tag: VERIFIED
