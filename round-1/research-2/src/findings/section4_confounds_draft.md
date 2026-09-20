## 4. Confound table (draft; citation numbers filled at synthesis)

| Confound | Corrupts | Control that removes it | Cite |
|---|---|---|---|
| Attention sink (BOS / special tok) | attention | drop BOS+template toks from | [] |
|  |  | denominator; value-norm weight | |
| Early-position attention bias | attention | per-index z-score across | [] |
|  |  | examples; question after passage | |
| Sentence-length bias | attention | MEAN not SUM over sentence | [] |
|  |  | tokens; sum as sensitivity | |
| Question not visible to passage | attention | append question AFTER passage; | [] |
|  |  | query = question toks + last tok | |
| Index/position bias (verbalized) | verbalized | shuffled-order rerun, map back, | [] |
|  |  | keep only stable selections | |
| Count bias (verbalized) | verbalized | fixed K = round(r*n) matches | [] |
|  |  | attention top-K | |
| Lead-sentence bias | both | report rate excluding S1; | [] |
|  |  | per-position selection freq | |
| Parametric-memory answering | LOO | no-context baseline; delta | [] |
|  |  | relative to full-minus-none gain | |
| Dangling references after drop | LOO | placeholder "[...]" vs deletion | [] |
| Small-n noise | all | paired bootstrap / McNemar; | [] |
|  |  | CIs; pre-registered threshold | |
