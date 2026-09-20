## 0. Handbook position (aii-handbook-auto-mechanistic-interpretability)

Honest scope note: the handbook (generated 2026-07-27) does NOT cover attention-as-explanation,
attention sinks, positional attention artifacts, or input-level saliency; its SKILL.md and
volatile.md contain no occurrence of "attention", "sink", "rollout" or "saliency" (grep, 2026-09-20).
Its "NOT for" line explicitly excludes post-hoc XAI (SHAP/LIME/saliency). Only transferable points
are listed; each names the handbook section it comes from.

1. [Organizing principles] Causal effect of a component is "a volatile random variable rather than
   a fixed property" (handbook S4) -> any per-sentence importance score (attention, LOO delta) must
   be reported as a distribution over examples, not a single-draw ranking.
2. [Frontier / validity] "small perturbations in input data or hyperparameters yield vastly
   different circuits" (S4) -> the shuffled-order elicitation control and the
   layer-range sensitivity check are the sentence-level analogue of seed/hyperparameter sweeps.
3. [Ground rules] Activation patching is the gold-standard causal metric; attribution-style
   approximations are first-order and their "dominant error stems from the non-linearities in the
   downstream network" (S19) -> leave-one-sentence-out ablation (an intervention) outranks attention
   (an approximation) as ground truth; attention is the thing being evaluated, never the referee.
4. [Frontier / reasoning-trace] "models can appear faithful yet remain hard to monitor when they
   leave out key factors" (S12) -> a verbalized keep-list can be faithful for the sentences it
   names and still omit load-bearing sentences; omission, not contradiction, is the expected
   failure mode of the verbalized signal.
5. [Frontier / reasoning-trace] The dominant unfaithfulness metric "confuses unfaithfulness with
   incompleteness" (S13) -> a sentence dropped by the verbalized list but high in attention is
   evidence of incompleteness, not proof of unfaithfulness; the report must keep the two apart.
6. [Frontier / applied] Decodability != actionability: 98.2% AUROC internal probe vs 45.1% output
   sensitivity, a "53-percentage-point knowledge-action gap" (S3) -> an internal signal that
   "knows" a sentence matters does not imply the model's verbalized output will act on it; this
   is exactly the gap the hypothesis tries to exploit, so it is a supportive framing, not a novelty.
7. [Critical rules] "Report the distribution and stability metrics, not the modal circuit" (S4
   row) -> per-sentence metrics get bootstrap CIs; disagreement rate reported per model, per
   elicitation variant, with n.
8. [Critical rules] Closing the loop needs "output-level correction AND collateral disruption of
   already-correct cases, against a random-perturbation control" (S3 row) -> the compressed-prompt
   comparison must include a random-protect control (protect the same number of random sentences)
   and count examples that flip from correct to incorrect after protection.
9. [What counts as DEEP] The venue bar is "specific falsifiable hypotheses, and how the evidence
   provided does and does not support them" or "clear practical benefits over well-implemented
   baselines" (S14) -> the hypothesis's falsifiable form is
   P(LOO-critical | DISAGREE) > P(LOO-critical | AGREE-DROP); the baseline is LLMLingua-2 /
   attention-only / verbalized-only compression at matched budget.
10. [Standing directive] "Map-silence means not-yet-checked, NOT open" -> the handbook's silence on
    attention-vs-verbalization disagreement is not evidence of novelty; Question 3 below runs the
    dated saturation search the handbook demands.
