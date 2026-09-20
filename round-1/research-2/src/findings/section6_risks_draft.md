## 6. Open risks for the experiment executor (decision rules fixed BEFORE looking at results)

| Symptom (pilot, 5-10 examples) | Interpretation | Fallback |
|---|---|---|
| Jaccard(orig, shuffled) < 0.5 | verbalized selection is mostly index/position noise | use only STABLE selections; if stable set < K/2, raise K or run 3 permutations and majority-vote |
| Disagreement rate < 5% | signals redundant; protective rule changes nothing | raise K granularity (finer split, e.g. clause-level) or lower budget ratio |
| Disagreement rate > 60% | signals unrelated; disagreement is not informative | check sink handling and layer range first; then switch attention to EHPC alternative |
| Spearman(attention, LOO delta) < 0.1 | attention not a usable internal signal for this model | EHPC head-selected alternative; if still < 0.1, report attention as failed signal and run black-box two-round protocol only |
| No-context accuracy > 50% of full-context accuracy | parametric memory dominates; LOO deltas compressed | filter to examples the model cannot answer closed-book; report both subsets |
| Placeholder vs deletion LOO deltas differ by > 2x on > 25% of sentences | dangling-reference confound | use placeholder variant as primary ground truth |
| Verbalized parse failure > 10% after one re-prompt | model cannot follow the JSON format at 1.5B | switch to enumerated yes/no per sentence (round-2 format) as primary elicitation |
| LOO-critical set empty on > 30% of examples | model answers from any single sentence or not at all | restrict to HotpotQA 'bridge' with both gold facts required; drop 'comparison' |
