# Confound-proof recipes for two sentence-importance signals

## Summary

Research recipe document for the disagreement-as-protection prompt-compression hypothesis. Provides, with 63 sources whose quoted passages were re-checked verbatim against the fetched URLs: (1) a PRIMARY sentence-level attention salience recipe for Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct (eager attention, question after passage, layers 19-27 of 28, pilot last-token vs question-token query, BOS/special-token exclusion plus VATP value-norm weighting, mean over heads/query rows/sentence tokens, Sentinel-style context-only normalisation, per-index position z-score, K = round(budget_ratio*n)) and an ALTERNATIVE EHPC-style evaluator-head recipe with a pre-registered switch trigger (Spearman with LOO < 0.1); rollout is rejected. (2) Exact verbalized elicitation prompts (task-agnostic and query-aware), fixed-K JSON output, strict parse + one re-prompt, a shuffled-order control with a both-orderings STABILITY rule, per-position and per-id bias diagnostics, a no-lead control, and a two-round one-sentence-at-a-time self-critique protocol as the black-box alternative. (3) A leave-one-sentence-out protocol on HotpotQA distractor bridge questions restricted to the two gold paragraphs (HF fields verified), n+2 passes per example (~420 per model for 30 examples), teacher-forced gold-answer log-prob with no-context and full-context baselines, placeholder-vs-deletion control for referential dangling, a pre-registered 'breaks performance' rule (answer flip OR |delta| > 2x median non-gold |delta|), the per-sentence comparison table, and paired bootstrap / McNemar testing. (4) A confound table (sink, position, length, question visibility, index/count/lead bias, parametric memory, dangling references, small n, model-dependent attention, single-config fragility) mapping each to the signal it corrupts, the control, and the citation. (5) A calibrated novelty statement: verbalized/internal confidence gaps, self-explanation unfaithfulness, explanation disagreement and query-by-committee are all established; the closest prior work closes the gap by steering/alignment (2603.25052, 2512.11998) or fuses two internal signals for compression (MIST 2608.31066); no paper uses attention-vs-verbalized disagreement at sentence level as a protective compression rule validated by LOO — verdict 'no direct prior claim found', moderate confidence. (6) Pre-registered fallbacks for unstable selections, out-of-range disagreement rates, failed attention, parametric memory and parse failures. Corrections: CC-SHAP is arXiv 2311.07466 (not 2311.13725); Referential Dangling is 2608.04569; Llama-3.2-3B config read from the unsloth mirror (meta-llama gated). Handbook coverage of attention-as-explanation is nil; only transferable points are used.

## Research Findings

## Scope and handbook position

The mechanistic-interpretability handbook does not cover attention-as-explanation, attention sinks or input saliency; only its transferable points apply: per-component causal effects are volatile random variables so every per-sentence score must be reported as a distribution with stability metrics; intervention (ablation) outranks approximation (attention) as ground truth; a verbalized trace can be faithful for what it names and still omit load-bearing factors, so omission — not contradiction — is the expected failure of the verbalized signal; decodability of an internal signal does not imply the output acts on it; and the venue bar is a falsifiable hypothesis, here P(LOO-critical | DISAGREE) > P(LOO-critical | AGREE-DROP), against matched-budget baselines with a random-protect control. Handbook silence on the hypothesis is not evidence of novelty; the dated search in Section 5 is.

## 1. Attention recipe (PRIMARY)

Evidence. Beyond the bottom two layers, Llama-2-7B "heavily attends to the initial token across all layers and heads" [1] (the planner's "more than half of total attention" figure does not appear in the current arXiv text and is UNVERIFIED); massive activations "lead to the concentration of attention probabilities to their corresponding tokens" [2]; sink tokens have value norms "much smaller than other tokens", so VATP scores tokens by "the product of value vector norm and attention score" [3]; sinks act "like key biases ... not contribut[ing] to the value computation" [4]. Context use is U-shaped with primacy and recency bias [5]. Sentinel reads attention "from the final prompt position", sums per sentence and normalises over a context-only denominator that "improves comparability across sentences" [6]; AttentionRAG collapses the query to a single hint token and selects sentences containing top-k attended tokens [7]; QUITO uses "a trigger token" [8]; AttnComp keeps documents whose "cumulative attention weights exceed a predefined threshold" [9]. EHPC selects per-model evaluator heads by a needle probe (Llama-3.1-8B: layer 13, heads [18, 13, 21, 8, 11, 1, 4, 3]) and beats LLMLingua-2 39.1 -> 49.6 at 2,048 tokens [10]. Rollout/flow beat raw attention only in an encoder classification setting [11]. Attention can be permuted or adversarially replaced with near-unchanged predictions [12], is at best a plausible rationale under proper tests [13], and ContextCite's attention baseline "approaches" its method on Llama-3-8B but "fares quite poorly with Phi-3-mini" — attention quality is model-dependent [14]. SDPA "does not support output_attentions=True. Please set your attention to eager" [15]; attentions come back as (batch, heads, seq, seq) [16]; both target models have 28 layers (Qwen2.5-1.5B: 12 heads / 2 KV heads [17]; Llama-3.2-3B: 24 heads / 8 KV heads, read from the unsloth mirror because meta-llama is gated [18]). A 600x600 map over 28 layers in fp16 is ~231 MiB (Qwen) / ~461 MiB (Llama).

Steps (each with the confound it removes):
1. `attn_implementation="eager"`, `output_attentions=True`, ONE pass over [template][passage][question][assistant header][gold answer]; this pass also yields the full-context teacher-forced log-prob for Section 3 [15, 16]. Removes: missing attention under SDPA; duplicate compute.
2. Question AFTER the passage. Removes: causal mask hiding the question from passage tokens [5, 6].
3. Layer band = layers 19-27 of 28 (0-indexed) for both models [17, 18]; report 9-18 and all-layers as sensitivity rows. Removes: early sink/positional heads; single-config fragility.
4. Query rows: pilot (a) last prompt token [6] vs (b) mean over question-token rows on 5 held-out examples; pick by Spearman with LOO; fix before the main run. Removes: query choice tuned on hypothesis data.
5. Sink handling: drop BOS and every chat-template special token from columns and denominator [1, 4]; weight each remaining column by the L2 norm of its value vector at that layer/head (hook v_proj; share KV head across GQA query heads) [3]; fallback: drop non-special columns > 3 SD above passage mean after manual inspection. Removes: sink mass as salience.
6. Mean over heads; mean over query rows. Removes: head-count / question-length scaling.
7. Sentence score = MEAN over sentence tokens; SUM (Sentinel's numerator [6]) as sensitivity. Removes: length bias.
8. Context-only normalisation over passage tokens [6]. Removes: template mass dilution.
9. Position residual: z-score each sentence within its index across examples. Removes: primacy/recency [5].
10. K = round(budget_ratio * n), identical to the verbalized K. Removes: count mismatch.

ALTERNATIVE (EHPC-style) [10]: per model, run ~10 held-out needle passages, score each (layer, head) by attention from the answer position onto the needle (specials excluded), keep the top 8 heads, per-token score = sum over selected heads, average-pool over a 16-token window (retune), sentence = mean over tokens, then apply steps 5, 8, 9, 10. Switch trigger: primary Spearman with LOO < 0.1 on the pilot. Do NOT use rollout: only encoder-setting evidence [11], L n x n products per example, and the decoder-only over-smoothing claims are UNVERIFIED.

Conclusion: attention mass is a noisy, model-dependent proxy [12, 14], usable only as the object under test, never as the referee; LOO ablation is the ground truth; if attention fails the pilot, report it as a failed signal and use the EHPC alternative.

## 2. Verbalized elicitation

Evidence. LLMLingua-2's labeler prompt is "Compress the given text ... You can ONLY remove unimportant words ... Do not reorder", chunked to <= 512 tokens, with the top-5% variation-rate examples dropped because GPT-4 "sometimes generates hallucinated content" [19]; Selective Context uses self-information, not a verbal judgment [20]; RECOMP uses a dual encoder and an "extractive oracle" [21]; CompAct asks for "0 to 3 sentences ... Additionally ... 0 to 3 sentences" — soft count, no ids [22]. Order flips pairwise judgments (Vicuna "could beat ChatGPT on 66 over 80 tested queries"), fixed by evaluating "each candidate in both positions across two runs" [23]; option-id token bias moves gpt-3.5-turbo by -6.3 and llama-30b by +15.2 points, and llama-30B picks A/B/C/D 34.6/27.3/22.3/15.8% [24]; listwise LLM ranking [25] is de-biased by permutation self-consistency, which "marginalize[s] out different list orders" for up to 9-24% gains, five permutations giving 67% of the gain of twenty [26]; "Sentence position bias dominates the learning signal for news summarization" [27]. No round-number/count-bias paper and no 2025-2026 numbered-id sentence-selection compressor were found (UNVERIFIED).

Prompt (temperature 0; sentences pre-split with the same boundaries as attention; ids S1..Sn; K = round(budget_ratio * n)). Variant A (task-agnostic): "You will be given a passage split into numbered sentences. Select the sentences that carry the passage's most important, load-bearing factual content - the sentences a reader would need to reconstruct the key facts and their relationships, independent of any specific question. Passage title: {TITLE} S1: ... Sn: ... Select exactly {K} sentence ids. Do not explain. Output only JSON of the form {"keep": ["S3", "S7"]} and nothing else - no prose, no markdown fences." Variant B (query-aware): same, with "Question: {QUESTION}" shown and the instruction "Select the sentences needed to answer the question, including sentences that state entities or facts the answer sentence depends on (for example bridge entities)." Parse strictly (exactly K distinct valid ids); re-prompt once with the failure reason; second failure = parse_failure, excluded, never coerced.

Control and stability rule: run each prompt in original order and once more with sentence contents randomly permuted and fresh ids; map back; report Jaccard(orig, shuffled), per-original-position selection frequency, per-id frequency in the shuffled run [23, 24, 26]. A sentence is INTROSPECTIVELY KEPT only if selected in both orderings; selected in one = UNSTABLE, analysed separately. No-lead control: report the disagreement rate with and without sentence 1 of each paragraph [27].

Two-round self-critique (black-box): round 1 = the prompt above; round 2 shows each dropped sentence ALONE with the title and asks "Would removing this sentence make it impossible to answer a typical factual question about the passage that a reader could otherwise answer using the full passage? Answer with exactly one word, yes or no, on the first line, then one short sentence giving one reason." A "yes" is the black-box disagreement flag. No prior drop-then-critique compressor found; closest are CompAct's iterative re-inclusion [22] and Referential Dangling's reinsertion repair [42].

## 3. LOO protocol

Evidence. ContextCite fits a LASSO linear surrogate on logit-scaled response probability, using "just 32 ablations even though the context comprises 98 sources", sweeping 32-512, and evaluates by top-k log-prob drop [14]; AttriBoT speeds exact LOO by caching, hierarchical attribution and proxy models (">300x speedup") and tests on HotpotQA [52]; for n <= ~15, exact LOO (n+1 passes) is below the surrogate's ~32-ablation floor and has no surrogate error (inference, not a quote). HotpotQA on HF: `hotpotqa/hotpot_qa`, configs distractor/fullwiki, fields id, question, answer, type, level, supporting_facts{title, sent_id}, context{title, sentences} [53]; the paper defines sentence-level supporting facts and bridge entities [54]; 2WikiMultihopQA adds evidence triples [55]; MuSiQue's paragraph_support_idx is paragraph-level only [56, 57]. ERASER comprehensiveness = m(x)_j - m(x \ r)_j [58]. Closed-book accuracy is non-trivial (56.1% vs 88.3% oracle [5]; 0.34-0.47 on FanOutQA [59]). Independent selection "can split dependent evidence pairs", leaving 34-54% of bridge answer paths incomplete; reinsertion recovers 29-34 points [42]. Use add_generation_prompt [60] and left padding for decoder-only batches [61]. Use paired bootstrap [62] and paired tests on the same data [63].

Protocol: HotpotQA distractor validation, type bridge, level medium/hard, ONLY the two gold paragraphs, 8-15 sentences, 200-500 tokens, 30 seeded examples plus 5-10 held out for pilots and the needle probe. Passes per example: no-context, full-context (= attention pass), n LOO deletions, optionally n "[...]"-placeholder passes; ~420 passes of <= 600 tokens per model (throughput UNVERIFIED — record it). Teacher-force the gold answer after the assistant header; verify tokenizer round-trip including the leading-space variant; score sum log p over answer tokens. Record per sentence: attention (primary raw/z, alternative), keep flags (orig, shuffled, stable; A and B), round-2 flag, delta_i = logp_full - logp_LOO_i, normalised delta / (logp_full - logp_nocontext), greedy answer and normalised EM/F1, answer_flip, gold flag, index, length, placeholder delta. Breaks performance (fixed before results, tuned once on the pilot): answer_flip OR |delta_i| > 2 x median |delta| over non-gold sentences of the same example; report 1.5x/3x sensitivity. Metrics: precision/recall of each signal's top-K vs LOO-critical and vs gold; Spearman(attention, delta); disagreement rate; P(LOO-critical | DISAGREE) vs P(LOO-critical | AGREE-DROP) with bootstrap CIs; accuracy of attention-only, verbalized-only, intersection-drop, disagreement-protect and random-protect variants by paired bootstrap / McNemar.

## 4. Confound table (confound -> signal hit -> control -> cite)

Sink/BOS/template -> attention -> drop specials, value-norm weight -> [1-4]. Early-position attention bias -> attention -> per-index z-score -> [5]. Sentence length -> attention -> mean not sum -> [6]. Question hidden from passage -> attention -> question after passage -> [5, 6]. Index/position bias -> verbalized -> shuffled rerun, stable-only -> [23, 24, 26]. Count bias -> verbalized -> fixed K -> [22]. Lead bias -> both -> rate excluding S1 -> [27]. Parametric memory -> LOO -> no-context baseline, normalised delta -> [5, 59]. Dangling references -> LOO -> placeholder vs deletion -> [42]. Small n -> all -> paired bootstrap / McNemar, CIs -> [62, 63]. Model-dependent attention -> attention -> pilot Spearman, EHPC switch -> [14, 10]. Single-config fragility -> attention -> layer-band sensitivity rows -> [11].

## 5. Calibrated novelty statement

Established: verbalized and internal confidence are imperfectly aligned — P(IK) is learnable but poorly calibrated on new tasks [28], verbalized confidence can be better calibrated [29] yet overconfident [30], internals "may encode the correct answer, yet consistently generate an incorrect one" [31], and hidden-state probes detect lies [32] and semantic entropy [33]. Self-explanations are often unfaithful or inconsistent with attribution: they "perform on par with traditional ones, but are quite different from them according to various agreement metrics" [44], faithfulness "is task-dependent" [46], CC-SHAP measures input-contribution consistency between answer and explanation [47], biasing features shift CoT answers unmentioned [48], CoT reliance varies by task [49], reasoning-model faithfulness is 25-39% [50], and introspection is real but partial [51, 45]; sentence-level attributions and self-rationales support different understanding [36]. Explanation methods disagree, but the disagreement problem is documented, not operationalised [37]; disagreement-as-informativeness is query-by-committee [39], re-instantiated for one LLM over perturbed prompts [40] and as agreement-gated selective prediction [38]. Closest operationalisations of the gap CLOSE it by steering [34] or alignment training [35]. In compression, MIST fuses two internal signals [41], PIS corrects attention with TF-IDF because "relying solely on attention scores may lead to the removal of important tokens" [43], and Referential Dangling repairs single-signal drops with dependency constraints [42]. No paper pairs attention salience with the same model's verbalized essential-sentence list and uses their DISAGREEMENT as the keep rule validated against LOO. Verdict: no direct prior claim found; closest are [34, 35, 41]; the hypothesis adds a verbalized channel, sentence-level operation in compression, and disagreement-as-protection validated by ablation. Confidence moderate: search covered 2019-2026 scholarly and general engines, but unindexed workshop work may exist.

## 6. Open risks (pre-registered fallbacks)

Jaccard(orig, shuffled) < 0.5 -> selection is position noise -> stable-only or 3-permutation majority, raise K. Disagreement rate < 5% -> no room -> finer granularity or lower budget; > 60% -> signals unrelated -> re-check sink handling, then EHPC. Spearman(attention, LOO) < 0.1 -> EHPC; if still < 0.1, report attention as failed and run the black-box two-round protocol only. No-context accuracy > 50% of full-context -> filter to examples the model cannot answer closed-book. Placeholder vs deletion deltas differ > 2x on > 25% of sentences -> use placeholder LOO as primary. Parse failure > 10% -> switch to per-sentence yes/no elicitation. LOO-critical set empty on > 30% of examples -> restrict to bridge questions needing both gold facts. Correction to the plan: the CC-SHAP id is 2311.07466, not 2311.13725 [47].


## Sources

[1] [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/pdf/2309.17453) (Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis; 2023) — Introduces StreamingLLM; shows initial tokens act as 'attention sinks' absorbing a large, content-independent share of attention mass.

> Beyond the bottom two layers, the model heavily attends to the initial token across all layers and heads

Locator: Figure 2 caption

[2] [Massive Activations in Large Language Models](https://arxiv.org/pdf/2402.17762) (Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu; 2024) — Identifies a small set of massive-magnitude activations at specific tokens (often BOS/delimiters) that act as fixed bias terms and pull attention toward themselves.

> these massive activations lead to the concentration of attention probabilities to their corresponding tokens, and further, implicit bias terms in the

Locator: arxiv.org/pdf/2402.17762, Abstract

> massive activations are closely connected with self-attention. In particular, we show massive activations cause attention to be attracted to the tokens associated with them.

Locator: arxiv.org/pdf/2402.17762, Introduction

[3] [Unveiling and Harnessing Hidden Attention Sinks: Enhancing Large Language Models without Training through Attention Calibration (VATP: Value-Aware Token Pruning)](https://arxiv.org/pdf/2406.12335) (2024) — Shows attention-sink tokens have near-zero value-vector norm despite large attention scores, and proposes weighting attention by value-vector norm (VATP) for token pruning.

> scores, the value vector norms of the attention sink tokens are much smaller than other tokens

Locator: arxiv.org/pdf/2406.12335, section on Observations

> VATP uses the product of value vector norm and attention score to evaluate the importance of each token’s KV cache.

Locator: arxiv.org/pdf/2406.12335, Figure 2 caption / methods

[4] [When Attention Sink Emerges in Language Models: An Empirical View](https://arxiv.org/pdf/2410.10781) (Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, Min Lin; 2024) — Studies when/why attention sinks emerge during pretraining; argues sinks act like key biases storing extra, non-informative attention mass, tied to softmax normalization.

> attention sink acts more like key biases, storing extra attention scores, which could be non-informative and not contribute to the value computation.

Locator: arxiv.org/pdf/2410.10781, Abstract

[5] [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/pdf/2307.03172) (Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang; 2023) — Reports closed-book (no-context) vs oracle accuracy on a multi-document QA task (NaturalQuestions-based, not literally HotpotQA); illustrates the parametric-memory pitfall for LOO ground truth.

> we observe a distinctive U-shaped performance curve (Figure 1); language model performance is highest when relevant information occurs at the very beginning (primacy bias) or end of its

Locator: arxiv.org/pdf/2307.03172, Introduction

> occurs at the very beginning (primacy bias) or end of its input context (recency bias), and performance degrades significantly when models must access and use information located in the middle of its input context.

Locator: Abstract/Figure 1

> when predicting without any documents (i.e., the closed-book setting; 56.1%)

Locator: §1 Introduction

[6] [Sentinel: Decoding Context Utilization via Attention Probing for Efficient LLM Context Compression](https://arxiv.org/pdf/2505.23277) (2025) — Sentence-level attention-probing compressor: uses last-prompt-position query attention, sum-over-sentence-tokens numerator normalized by context-only denominator; beats raw attention baseline on LongBench.

> represents the attention weight assigned from the final prompt position to token t at layer l and head h.

Locator: arxiv.org/pdf/2505.23277, section 2.2.2

> denotes the set of tokens originating from the retrieved context C, excluding prompt and query tokens. This normalization improves comparability across sentences by restricting

Locator: arxiv.org/pdf/2505.23277, section 2.2.2, Eq. 1

> Raw Attention (Qwen2.5-0.5B) 34.92 38.96 21.32 31.74

Locator: arxiv.org/pdf/2505.23277, Table 2

[7] [AttentionRAG: Attention-Guided Context Pruning in Retrieval-Augmented Generation](https://arxiv.org/pdf/2503.10720) (Yixiong Fang, Tianran Sun, Yuling Shi, Xiaodong Gu; 2025) — Reformulates the query as a next-token-prediction 'answer hint prefix' so a single anchor token carries query semantics for efficient query-context attention; selects sentences by top-k attended tokens.

> a compressed context by selecting sentences from the original context with the top-k attended tokens.

Locator: arxiv.org/pdf/2503.10720, Introduction

> achieves up to 6.3x context compression while

Locator: arxiv.org/pdf/2503.10720, Abstract

[8] [QUITO: Accelerating Long-Context Reasoning through Query-Guided Context Compression](https://arxiv.org/pdf/2408.00274) (Wenshan Wang, Yihang Wang, Yixing Fan, Huaming Liao, Jiafeng Guo; 2024) — Uses a single 'trigger token' to compute the attention distribution of the context relative to the question, then filters context by that distribution.

> we take a trigger token to calculate the attention distribution of the context in response to the question. Based on the distribution, we propose three different filtering

Locator: arxiv.org/pdf/2408.00274, Abstract

[9] [AttnComp: Attention-Guided Adaptive Context Compression for Retrieval-Augmented Generation](https://arxiv.org/pdf/2509.17486) (Lvzhou Luo, Yixuan Cao, Ping Luo; 2025) — Uses cross-attention from query to context documents plus a Top-P compression rule to retain the minimal document set whose cumulative attention exceeds a threshold; also yields a confidence score.

> compression algorithm to retain the minimal set of documents whose cumulative attention weights exceeds a predefined threshold.

Locator: arxiv.org/pdf/2509.17486, Abstract

[10] [Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference (EHPC)](https://arxiv.org/pdf/2501.12959) (Weizhi Fei, Xueyan Niu, Guoqing Xie, Yingqing Liu, Bo Bai, Wei Han; 2025) — Identifies a small per-model set of early-layer 'evaluator heads' via a needle-in-a-haystack pilot, uses their cumulative/pooled attention scores to select important tokens, and applies this training-free to prompt compression.

> scores as evaluator heads from the layer with the highest cumulative score

Locator: arxiv.org/pdf/2501.12959, 'Practicality' section

[11] [Quantifying Attention Flow in Transformers](https://arxiv.org/pdf/2005.00928) (Samira Abnar, Willem Zuidema; 2020) — Proposes attention rollout and attention flow as post-hoc corrections to raw attention; shows both correlate better with ablation/gradient-based importance than raw attention, in an encoder-style classification setting.

> compared to raw attention, both yield higher correlations with importance scores of input tokens obtained

Locator: arxiv.org/pdf/2005.00928, Abstract

> Raw 0.69±0.27 0.10±0.43 -0.11±0.49 -0.09±0.52 0.20±0.45 0.29±0.39

Locator: arxiv.org/pdf/2005.00928, Table 1

[12] [Attention is not Explanation](https://arxiv.org/pdf/1902.10186) (Sarthak Jain, Byron C. Wallace; 2019) — Shows adversarial or randomly permuted attention distributions can leave BiLSTM model predictions almost unchanged, questioning attention as a faithful explanation.

> permuting attention weights often induces only minimal changes in output

Locator: arxiv.org/pdf/1902.10186, Introduction

[13] [Attention is not not Explanation](https://arxiv.org/pdf/1908.04626) (Sarah Wiegreffe, Yuval Pinter; 2019) — Responds to Jain & Wallace; argues attention can serve as a plausible (though not necessarily faithful) explanation under the right diagnostic tests.

> If we accept the Rudin and Riedl deﬁnitions of explainability as providing a plausible, but not necessarily faithful rationale for model prediction, then the argument against attention mechanisms because they are not

Locator: arxiv.org/pdf/1908.04626, section 6

[14] [ContextCite: Attributing Model Generation to Context](https://arxiv.org/pdf/2409.00729) (Benjamin Cohen-Wang, Harshay Shah, Kristian Georgiev, Aleksander Madry; 2024) — Proposes a linear-surrogate context-attribution method; benchmarks it against an attention-sum baseline, finding attention's reliability for context attribution is model-dependent (works reasonably for Llama-3-8B, poorly for Phi-3-mini).

> We use a simple but effective baseline that computes an attribution score for each source by summing the average attention weight of individual tokens in the source across all heads in all layers

Locator: arxiv.org/pdf/2409.00729, Baselines section

> While the attention baseline approaches the performance of CONTEXTCITE with Llama-3-8B, it fares quite poorly with Phi-3-mini suggesting that attention is not consistently reliable for context attribution.

Locator: arxiv.org/pdf/2409.00729, Results section

> the surrogate model in Figure 2 uses just 32 ablations even though the context comprises 98 sources (in this case, sentences).

Locator: §3 Context attribution with CONTEXTCITE

[15] [transformers integrations/sdpa_attention.py (source)](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/integrations/sdpa_attention.py) — Runtime warning in the SDPA attention backend confirming that output_attentions=True is unsupported under SDPA and requires attn_implementation='eager'.

> Please set your attention to `eager` if you want any of these features.

Locator: sdpa_attention.py, function sdpa_attention_forward

[16] [Llama - HuggingFace Transformers docs](https://huggingface.co/docs/transformers/en/model_doc/llama) — Documents the 'attentions' output field, returned only when output_attentions=True is passed, shape (batch_size, num_heads, sequence_length, sequence_length).

> returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`

Locator: LlamaModel forward outputs

[17] [Qwen2.5-1.5B-Instruct config.json](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json) — Model config confirming 28 layers, 12 attention heads, 2 key-value heads (GQA) for Qwen2.5-1.5B-Instruct.

> "num_attention_heads": 12, "num_hidden_layers": 28, "num_key_value_heads": 2

Locator: config.json

[18] [Llama-3.2-3B-Instruct config.json (unsloth mirror; meta-llama/Llama-3.2-3B-Instruct is gated, returned HTTP 401)](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct/raw/main/config.json) — Model config confirming 28 layers, 24 attention heads, 8 key-value heads (GQA), head_dim 128 for Llama-3.2-3B-Instruct, fetched from the unsloth mirror because the canonical meta-llama repo is gated.

> "num_attention_heads": 24, "num_hidden_layers": 28, "num_key_value_heads": 8

Locator: config.json

[19] [LLMLingua-2: A Fast and Faithful Task-Agnostic Prompt Compression](https://arxiv.org/pdf/2403.12968) (Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, et al.; 2024) — GPT-4 distillation instruction for word-level compression; chunks contexts at <=512 tokens; Variation Rate filter for hallucination/labeling quality control.

> Compress the given text to short expressions, and such that you (GPT-4) can reconstruct it as close as possible to the original. Unlike the usual text compression, I need you to comply with the 5 conditions below: 1. You can ONLY remove unimportant words.

Locator: §3.1 Instruction Design

> we first segment each long context into multiple chunks, each containing no more than 512 tokens and ending with a period.

Locator: §3.2 Data Annotation

> we exclude the examples with the top 5% highest variation rates.

Locator: §3.3 Quality Control, Variation Rate

[20] [Compressing Context to Enhance Inference Efficiency of Large Language Models (Selective Context)](https://arxiv.org/pdf/2310.06201) (Yucheng Li, Bo Dong, Chenghua Lin, Frank Guerin; 2023) — Prunes context by self-information (surprisal) computed by a base causal LM, not by LLM verbal judgment.

> Selective Context evaluates informativeness of lexical units (i.e., tokens, phrases, or sentences) with self-information (Shannon, 1948) computed by a base causal language model.

Locator: Abstract

[21] [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://arxiv.org/pdf/2310.04408) (Fangyuan Xu, Weijia Shi, Eunsol Choi; 2023) — Extractive compressor uses a trained dual-encoder to score sentences by inner product with the query; also defines an extractive oracle by brute-force substitution.

> embeddings respectively. Their inner product represents how helpful it would be for the LM M to prepend si to the input x to generate y.

Locator: §2 Extractive Compressor

> an extractive oracle which selects a sentence in evidence documents that leads to the best task performance

Locator: Introduction

[22] [CompAct: Compressing Retrieved Documents Actively for Question Answering](https://arxiv.org/pdf/2407.09014) (2024) — LLM sentence-level selection prompt allows 0-3 + 0-3 sentences (soft band, not fixed K); iterative compression can re-add previously omitted content.

> we devise an iterative architecture where the compressed contexts are updated at each iteration

Locator: Section 3, framework overview

[23] [Large Language Models are not Fair Evaluators](https://arxiv.org/pdf/2305.17926) (Peiyi Wang, Lei Li, Liang Chen, et al.; 2023) — Documents strong positional bias in pairwise LLM judging; proposes Balanced Position Calibration (average score over both orderings).

> we evaluate each candidate in both positions across two runs and compute the final score as the average of the two runs.

Locator: §1, Balanced Position Calibration (BPC)

> Vicuna-13B could beat ChatGPT on 66 over 80 tested queries with ChatGPT as an evaluator.

Locator: Abstract

[24] [Large Language Models Are Not Robust Multiple Choice Selectors](https://arxiv.org/pdf/2309.03882) (Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, Minlie Huang; 2023) — Identifies option-ID token bias (selection bias) in MCQ answering; proposes PriDe debiasing; quantifies large accuracy swings from moving the correct option.

> moving the golden answers to position D degrades the accuracy of gpt-3.5-turbo by 6.3 (from 67.2 to 60.9). When moving to A, llama-30b is boosted by 15.2 and surpasses gpt-3.5-turbo (68.2 vs. 65.3)

Locator: §1 Introduction

> llama-30B selects A/B/C/D 34.6% / 27.3% / 22.3% / 15.8% of the time

Locator: §1 Introduction

[25] [Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents (RankGPT)](https://arxiv.org/pdf/2304.09542) (Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, Zhaochun Ren; 2023) — Zero-shot LLM listwise passage re-ranking; competitive with supervised SOTA; introduces permutation distillation.

> even superior results to state-of-the-art supervised methods on popular IR benchmarks

Locator: Abstract

[26] [Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models](https://arxiv.org/pdf/2310.07712) (Raphael Tang, Xinyu Zhang, Xueguang Ma, Jimmy Lin, Ferhan Ture; 2023) — Marginalizes out list order by repeated shuffling and aggregation to reduce positional bias in listwise ranking.

> we propose permutation self-consistency, a form of self-consistency over the ranking list outputs of black-box LLMs. Our key idea is to marginalize out different list orders in the prompt to produce an order-independent

Locator: Abstract

> increasing the scores of GPT-3.5, GPT-4, and LLaMA v2 (70B; Touvron et al., 2023) by up to 4–17%, 9–24%, and 8–16%, respectively

Locator: Introduction

[27] [Content Selection in Deep Learning Models of Summarization](https://arxiv.org/pdf/1810.12343) (Chris Kedzie, Kathleen McKeown, Hal Daumé III; 2018) — Establishes lead bias dominance in news summarization; sentence-shuffling ablation shows models rely heavily on position, not content.

> signal for news summarization, though not for other domains

Locator: §1 main results

> are not able to identify important information without position

Locator: §5 Document Shuffling

[28] [Language Models (Mostly) Know What They Know](https://arxiv.org/pdf/2207.05221) (Saurav Kadavath, Tom Conerly, Amanda Askell, et al.; 2022) — Introduces P(True) and P(IK) as ways for LLMs to self-evaluate the correctness/knowability of their own answers.

> we investigate whether models can be trained to predict "P(IK)", the probability that "I know" the answer to a question, without reference to any particular proposed answer. Models perform well at predicting P(IK) and partially generalize across tasks, though they struggle with calibration of P(IK) on new tasks.

Locator: Abstract

[29] [Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models](https://arxiv.org/pdf/2305.14975) (Katherine Tian, Eric Mitchell, et al.; 2023) — Shows verbalized confidence scores from RLHF-LMs are often better calibrated than raw conditional token probabilities.

> verbalized confidences emitted as output tokens are

Locator: Abstract

[30] [Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs](https://arxiv.org/pdf/2306.13063) (Miao Xiong, Zhiyuan Hu, et al.; 2024) — Systematic black-box framework for eliciting verbalized confidence, showing LLMs are overconfident when verbalizing.

> LLMs, when verbalizing their confidence, tend to be overconfident, potentially imitating human patterns of expressing confidence.

Locator: Abstract

[31] [LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations](https://arxiv.org/pdf/2410.02707) (Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, Yonatan Belinkov; 2024) — Shows LLM internal representations encode truthfulness info beyond what is used in generation, including cases where the model internally encodes the right answer but outputs the wrong one.

> we reveal a discrepancy between LLMs’ internal encoding and external behavior: they may encode the correct answer, yet consistently generate an incorrect one.

Locator: Abstract

[32] [The Internal State of an LLM Knows When It's Lying](https://arxiv.org/pdf/2304.13734) (Amos Azaria, Tom Mitchell; 2023) — Trains a classifier (SAPLMA) on LLM hidden-layer activations to detect statement truthfulness.

> our trained classifier achieves an average of 71% to 83% accuracy labeling which

Locator: Abstract

[33] [Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs](https://arxiv.org/pdf/2406.15927) (Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, Yarin Gal; 2024) — Proposes cheap linear probes on hidden states that approximate expensive semantic-entropy-based hallucination detection.

> We propose semantic entropy probes (SEPs), a cheap and reliable method for

Locator: Abstract

[34] [Closing the Confidence-Faithfulness Gap in Large Language Models](https://arxiv.org/pdf/2603.25052) (Miranda Muqing Miao, Lyle Ungar; 2026) — Shows the geometric relationship between internal accuracy signal and verbalized confidence, and closes the gap by steering verbalized output to match an internal accuracy estimate (adaptive steering vs SteerConf).

> by reading the model’s internal estimate of its own competence and steering its verbalized output to match, we are able to close much of the faithfulness gap of verbalized confidence

Locator: Section 3 results discussion, near Table with ECE/Brier/MAE

[35] [Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models](https://arxiv.org/pdf/2512.11998) (Glenn Zhang, Treasure Mayowa, Jason Fan, Yicheng Fu, Aaron Sandoval, Sean O'Brien, Kevin Zhu; 2026) — Uses Direct Preference Optimization (DCA) to align an LLM's verbalized confidence with its internal (token-probability-derived) confidence rather than ground-truth accuracy.

> is not well aligned with its verbalized confidence, leading to misleading results with different calibration methods

Locator: Abstract

[36] [Not All Explanations Simulate Equally: Comparing Verbalized Feature Attributions and Self-Generated Rationales](https://arxiv.org/pdf/2606.01148) (2026) — Compares sentence-level and token-level feature-attribution explanations against self-generated rationales via simulatability; sentence-level attributions give stronger simulatability gains than token-level.

> attributions provide stronger simulatability gains than token-level attributions, LLM

Locator: Contributions list, Introduction

[37] [The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective](https://arxiv.org/pdf/2202.01602) (Satyapriya Krishna, Tessa Han, Alex Gu, Steven Wu, et al.; 2024) — Formalizes and empirically studies disagreement among post-hoc explanation methods (SHAP, LIME, etc.) across 6 explainers and 4 datasets; interviews 25 practitioners on how they resolve disagreement qualitatively. Does not propose disagreement as an automated downstream risk score.

> we formalize and investigate the disagreement problem in explainable ML, a novel area of research that examines conflicts among post hoc explanation methods.

Locator: Introduction

[38] [Selective Prediction from Agreement: A Lipschitz-Consistent Version Space Approach](https://arxiv.org/pdf/2605.02611) (Mohamadsadegh Khosravani; 2026) — Selective classification method that abstains unless all Lipschitz-consistent classification heads (a version space) agree on the label; disagreement among an ensemble of consistent heads triggers abstention. Image classification (CIFAR-10/SVHN), not LLM/explanation disagreement.

> The model therefore predicts only when the label is forced (i.e., all consistent heads agree), and abstains otherwise.

Locator: Abstract

[39] [Query by Committee](https://dl.acm.org/doi/pdf/10.1145/130385.130417) (H. S. Seung, M. Opper, H. Sompolinsky; 1992) — Foundational query-by-committee active-learning paper: disagreement among a committee of learners estimates the information value of a query, giving exponential generalization-error decay.

> in which a committee of students is trained on the same data set. The next query is chosen according to the principle of maximal disagreement

Locator: Abstract

> of disagreement among a committee of learners can serve as an estimate of this information value

Locator: Introduction

[40] [Active Instruction Tuning: Improving Cross-Task Generalization by Training on Prompt Sensitive Tasks](https://arxiv.org/pdf/2311.00288) (Po-Nien Kung, Fan Yin, Di Wu, Kai-Wei Chang, Nanyun Peng; 2023) — Defines 'Prompt Uncertainty' as the disagreement of a single model's predictions across the original vs. perturbed task instructions, motivated by BALD (Bayesian Active Learning by Disagreement); used for task-level selection in instruction tuning, not sentence-level content salience.

> We represent the informativeness of new tasks with the disagreement of the current model

Locator: Abstract

[41] [Every Token Leaves a Ripple in the Stream of Thought: Eliciting Model-Internal Token Saliency for Chain-of-Thought Compression (MIST)](https://arxiv.org/pdf/2608.31066) (Tianyi Zhao, Yinhan He, Wendy Zheng, Chen Chen; 2026) — Proposes MIST, combining two model-internal saliency axes (necessity: residual-stream ablation effect; sufficiency: residual-stream patching effect) into one fused importance score for pruning CoT reasoning tokens. Both signals are internal; no verbalized signal; fusion not disagreement.

> MIST (Model-Internal Saliency for Token-level CoT compression), which defines token importance along two complementary axes: necessity, the drop in answer likelihood when a token’s internal contribution is removed, and sufficiency, the gain in answer likelihood when that contribution alone is provided

Locator: Abstract

[42] [Relevant but Incomplete: Referential Dangling as a Paradigm-Level Failure Mode in Hard Prompt Compression](https://arxiv.org/pdf/2608.04569) (Zhengpei Hu, Kai Li, Dapeng Fu, Xuechao Zou, Yuanhao Tang, Yue Li, Tengfei Cao, Jianqiang Huang; 2026) — Identifies 'referential dangling': independent per-unit importance scoring in hard prompt compressors splits dependent evidence pairs, leaving retained spans uninterpretable; measured up to 60% dangling rate on HotpotQA bridge questions across 6 compressors, with 29-34pp accuracy recovery from fixing it.

> independent selection can split dependent evidence pairs, retaining one member while deleting the other. When the retained text span contains an

Locator: Abstract/Introduction

> leaves the answer path incomplete in 34 to 54% of bridge examples across three multi-hop question

Locator: Abstract

[43] [PIS: Linking Importance Sampling and Attention Mechanisms for Efficient Prompt Compression](https://arxiv.org/pdf/2504.16574) (Lizhe Chen, Binjia Zhou, Yuyao Ge, Jiayi Chen, Shiguang Ni; 2025) — Treats attention scores as importance-sampling weights for prompt compression but notes relying solely on attention can drop important tokens; addresses this via attention-score variance across layers rather than a second, independent (e.g. verbalized) signal.

> relying solely on attention scores may lead to the removal of important tokens with high attention scores. To address this, we introduce TF-IDF scores as a corrective measure for attention scores of each token.

Locator: arxiv.org/pdf/2504.16574, methodology section

[44] [Can Large Language Models Explain Themselves? A Study of LLM-Generated Self-Explanations](https://arxiv.org/pdf/2310.11207) (Shiyuan Huang, Siddarth Mamidanna, Shreedhar Jangam, Yilun Zhou, Leilani H. Gilpin; 2023) — Compares ChatGPT's self-generated feature-attribution explanations against occlusion and LIME saliency maps; self-explanations perform on par with faithfulness metrics but disagree substantially with the traditional methods.

> we study different ways to elicit the self-explanations, evaluate their faithfulness on a set of evaluation metrics, and compare them to traditional explanation methods such as occlusion or LIME saliency maps. Through an extensive set of experiments, we find that ChatGPT’s

Locator: arxiv.org/pdf/2310.11207, abstract

[45] [Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs](https://arxiv.org/pdf/2512.12411) (Ely Hahami, Ishaan Sinha, Lavik Jain, Josh Kaplan, Jon Hahami; 2026) — Finds LLM introspection of injected concept perturbations is partial: near-chance on naive binary detection but well above chance (up to 88%/83%) on localization/strength-comparison tasks, confined to early layers.

> on tasks requiring differential sensitivity, we find robust evidence for partial introspection: models localize which of 10 sentences received an injection at up to 88% accuracy (vs. 10% chance) and discriminate relative injection strengths at 83% accuracy (vs. 50% chance). These capabilities are confined to early-layer injections and collapse to chance thereafter

Locator: arxiv.org/pdf/2512.12411, abstract

[46] [Are self-explanations from Large Language Models faithful?](https://arxiv.org/pdf/2401.07927) (Andreas Madsen, Sarath Chandar, Siva Reddy; 2024) — Introduces self-consistency checks to measure faithfulness of LLM self-explanations; finds faithfulness varies by explanation type and task, not just by model.

> Figure 8: Faithfulness evaluation using self-consistency checks, evaluated using Llama2-70B. Results show that Llama2-70B is not affected by prompt variations, but the faithfulness for each explanation type is task-dependent.

Locator: arxiv.org/pdf/2401.07927, Figure 8 caption

[47] [On Measuring Faithfulness or Self-consistency of Natural Language Explanations](https://arxiv.org/pdf/2311.07466) (Letitia Parcalabescu, Anette Frank; 2024) — Introduces CC-SHAP, a fine-grained Shapley-value-based measure of self-consistency between a model's input contributions when producing a prediction versus when producing its explanation. NOTE: the ID given in the source task (2311.13725) is incorrect and resolves to an unrelated paper (Ali & Breazeal, 'Studying Artist Sentiments around AI-generated Artwork'); the correct ID, confirmed by page-1 title and content match, is 2311.07466 (Parcalabescu & Frank, ACL 2024).

> including iii) our new self-consistency measure CC-SHAP. CC-SHAP is a fine-grained

Locator: arxiv.org/pdf/2311.07466, abstract

[48] [Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/pdf/2305.04388) (Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman; 2023) — Shows CoT explanations can be systematically unfaithful: biasing features (e.g. reordering answer options) shift model predictions by up to 36% without being mentioned in the explanation.

> Adding biasing features heavily influences model CoT predictions on BBH tasks, causing accuracy to drop as much as 36%, despite the biasing features never being referenced in the CoT explanations.

Locator: arxiv.org/pdf/2305.04388, Introduction, main findings

[49] [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/pdf/2307.13702) (Tamera Lanham, Anna Chen, Ansh Radhakrishnan, et al.; 2023) — Tests several possible faithfulness failure modes of CoT (post-hoc reasoning, etc.); finds models vary greatly in how much they use/rely on CoT depending on the task.

> We find great variation in how much LLMs use CoT on different tasks, not using CoT at all for some tasks while relying upon it heavily for other tasks

Locator: arxiv.org/pdf/2307.13702, Section on faithfulness failures

[50] [Reasoning Models Don't Always Say What They Think](https://arxiv.org/pdf/2505.05410) (Yanda Chen, Joe Benton, Ansh Radhakrishnan, et al., Ethan Perez; 2025) — Measures how often reasoning-model CoT verbalizes hints actually used to change the answer; overall faithfulness scores are low (25% Claude 3.7 Sonnet, 39% DeepSeek R1).

> CoTs of reasoning models often lack faithfulness and can conceal misalignment. The overall faithfulness scores for both reasoning models remain low (25% for Claude 3.7 Sonnet and 39% for DeepSeek R1) (Figure 1).

Locator: arxiv.org/pdf/2505.05410, results summary

[51] [Looking Inward: Language Models Can Learn About Themselves by Introspection](https://arxiv.org/pdf/2410.13787) (Felix J Binder, James Chua, Tomek Korbak, et al.; 2024) — Shows a model trained on its own self-prediction data predicts its own behavior more accurately than a cross-trained model predicting it, evidence for (limited) introspection.

> Llama 70B predicts its own behavior more accurately (48.5%) than GPT-4o (31.8%), despite GPT-4o’s superior capabilities

Locator: arxiv.org/pdf/2410.13787, Section 3.2.1

[52] [AttriBoT: A Bag of Tricks for Efficiently Approximating Leave-One-Out Context Attribution](https://arxiv.org/pdf/2411.15102) (2024) — Speeds up exact LOO via KV caching, hierarchical attribution, and proxy models; >300x speedup, tested on HotpotQA examples.

> AttriBoT can provide a >300× speedup while remaining more faithful to a target model’s LOO error than prior context attribution methods.

Locator: Abstract

> We empirically test the assumptions underlying the AttriBoT’s underlying methods on

Locator: Figure 1 caption

[53] [hotpotqa/hotpot_qa dataset card (Hugging Face)](https://huggingface.co/datasets/hotpotqa/hotpot_qa) — Confirms configs (distractor/fullwiki), splits (train/validation), and field schema (id, question, answer, type, level, supporting_facts.title/sent_id, context.title/sentences).

> { "title": [ "Arthur's Magazine", "First for Women" ], "sent_id": [ 0, 0 ] }

Locator: dataset viewer, supporting_facts column

[54] [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/pdf/1809.09600) (Zhilin Yang, Peng Qi, Saizhen Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, Christopher D. Manning; 2018) — Original HotpotQA paper defining sentence-level supporting facts, bridge entities, and comparison questions.

> facts required for reasoning, allowing QA systems to reason with strong supervision and explain the predictions

Locator: Introduction

[55] [Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps (2WikiMultiHopQA)](https://arxiv.org/pdf/2011.01060) (Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, Akiko Aizawa; 2020) — Introduces evidence triples (subject, property, object) alongside sentence-level supporting facts; four question types.

> Evidence information in our dataset is a set of triples, where each triple is a structured data (subject entity, property, object entity) obtained from the Wikidata

Locator: §1 Introduction

[56] [dgslibisey/MuSiQue dataset card (Hugging Face)](https://huggingface.co/datasets/dgslibisey/MuSiQue) — Confirms paragraph_support_idx field in question_decomposition, i.e. MuSiQue support annotation is paragraph-level not sentence-level.

> { "id": 482757, "question": "The Collegian >> owned by", "answer": "Houston Baptist University", "paragraph_support_idx": 5 }

Locator: dataset viewer, question_decomposition column

[57] [MuSiQue: Multihop Questions via Single-hop Question Composition](https://arxiv.org/pdf/2108.00573) (Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal; 2022) — MuSiQue paper; corroborates paragraph-level supporting-paragraph annotation concept (does not contain literal field name paragraph_support_idx in prose).

> MuSiQue constitutes unique 21020 single-hop questions, 4132 answers to multihop questions, 19841 answers to

Locator: §6 dataset statistics

[58] [ERASER: A Benchmark to Evaluate Rationalized NLP Models](https://arxiv.org/pdf/1911.03429) (Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, et al.; 2019) — Defines comprehensiveness and sufficiency faithfulness metrics via contrast examples with rationale removed.

> A high score here implies that the rationales were indeed inﬂuential in the prediction, while a low score suggests that they were not.

Locator: §4.2 Measuring faithfulness, Comprehensiveness

[59] [FanOutQA: A Multi-Hop, Multi-Document Question Answering Benchmark for Large Language Models](https://arxiv.org/pdf/2402.14116) (Andrew Zhu, Alyssa Hwang, Liam Dugan, Chris Callison-Burch; 2024) — HotpotQA-adjacent multi-hop benchmark reporting explicit closed-book LLM accuracy figures.

> Using only knowledge encoded in their parameters, models’ loose string accuracy ranged from 0.341 (Claude) to 0.470 (Mixtral), with none reaching our estimated human baseline of 0.685

Locator: §4.2 Closed Book Results

[60] [Chat templates (Hugging Face Transformers docs)](https://huggingface.co/docs/transformers/chat_templating) — Explains add_generation_prompt and the assistant-header boundary; adjacent to but not an exact citation for the 'leading space' tokenization pitfall.

> This argument adds tokens to the end of the chat that indicate the start of an

Locator: add_generation_prompt section

[61] [Text generation (Hugging Face Transformers docs)](https://huggingface.co/docs/transformers/llm_tutorial) — Confirms padding_side='left' requirement for decoder-only batched generation.

> because a LLM is not trained to continue generation from padding tokens.

Locator: Default generate section

[62] [Statistical Significance Tests for Machine Translation Evaluation](https://aclanthology.org/W04-3250.pdf) (Philipp Koehn; 2004) — Introduces paired bootstrap resampling for comparing two systems on the same test set.

> to compute statistical signiﬁcance of test results, and validate them on the concrete example of the BLEU score. Even for small test sizes of only 300 sentences, our methods may give us assurances that test result differences are real

Locator: Abstract

[63] [The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing](https://arxiv.org/pdf/1809.01448) (Rotem Dror, Gili Baumer, Segev Shlomov, Roi Reichart; 2018) — Recommends paired significance tests when comparing systems on the same dataset.

> on the same dataset, one should use the paired version of the statistical signiﬁcance test (such as the matched-pair t-test).

Locator: Introduction

## Verification

Numbered citations resolve to unique listed sources. Passage checks test text occurrence, not claim truth or entailment. Author/year metadata and locators are not independently verified. Details: `research_verification.json`.

- Source [1]: text found — Beyond the bottom two layers, the model heavily attends to the initial token across all layers and h
- Source [2]: text found — these massive activations lead to the concentration of attention probabilities to their correspondin
- Source [2]: text found — massive activations are closely connected with self-attention. In particular, we show massive activa
- Source [3]: text found — scores, the value vector norms of the attention sink tokens are much smaller than other tokens
- Source [3]: text found — VATP uses the product of value vector norm and attention score to evaluate the importance of each to
- Source [4]: text found — attention sink acts more like key biases, storing extra attention scores, which could be non-informa
- Source [5]: text found — we observe a distinctive U-shaped performance curve (Figure 1); language model performance is highes
- Source [5]: text found — occurs at the very beginning (primacy bias) or end of its input context (recency bias), and performa
- Source [5]: text found — when predicting without any documents (i.e., the closed-book setting; 56.1%)
- Source [6]: text found — represents the attention weight assigned from the final prompt position to token t at layer l and he
- Source [6]: text found — denotes the set of tokens originating from the retrieved context C, excluding prompt and query token
- Source [6]: text found — Raw Attention (Qwen2.5-0.5B) 34.92 38.96 21.32 31.74
- Source [7]: text found — a compressed context by selecting sentences from the original context with the top-k attended tokens
- Source [7]: text found — achieves up to 6.3x context compression while
- Source [8]: text found — we take a trigger token to calculate the attention distribution of the context in response to the qu
- Source [9]: text found — compression algorithm to retain the minimal set of documents whose cumulative attention weights exce
- Source [10]: text found — scores as evaluator heads from the layer with the highest cumulative score
- Source [11]: text found — compared to raw attention, both yield higher correlations with importance scores of input tokens obt
- Source [11]: text found — Raw 0.69±0.27 0.10±0.43 -0.11±0.49 -0.09±0.52 0.20±0.45 0.29±0.39
- Source [12]: text found — permuting attention weights often induces only minimal changes in output
- Source [13]: text found — If we accept the Rudin and Riedl deﬁnitions of explainability as providing a plausible, but not nece
- Source [14]: text found — We use a simple but effective baseline that computes an attribution score for each source by summing
- Source [14]: text found — While the attention baseline approaches the performance of CONTEXTCITE with Llama-3-8B, it fares qui
- Source [14]: text found — the surrogate model in Figure 2 uses just 32 ablations even though the context comprises 98 sources 
- Source [15]: text found — Please set your attention to `eager` if you want any of these features.
- Source [16]: text found — returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of
- Source [17]: text found — "num_attention_heads": 12, "num_hidden_layers": 28, "num_key_value_heads": 2
- Source [18]: text found — "num_attention_heads": 24, "num_hidden_layers": 28, "num_key_value_heads": 8
- Source [19]: text found — Compress the given text to short expressions, and such that you (GPT-4) can reconstruct it as close 
- Source [19]: text found — we first segment each long context into multiple chunks, each containing no more than 512 tokens and
- Source [19]: text found — we exclude the examples with the top 5% highest variation rates.
- Source [20]: text found — Selective Context evaluates informativeness of lexical units (i.e., tokens, phrases, or sentences) w
- Source [21]: text found — embeddings respectively. Their inner product represents how helpful it would be for the LM M to prep
- Source [21]: text found — an extractive oracle which selects a sentence in evidence documents that leads to the best task perf
- Source [22]: text found — we devise an iterative architecture where the compressed contexts are updated at each iteration
- Source [23]: text found — we evaluate each candidate in both positions across two runs and compute the final score as the aver
- Source [23]: text found — Vicuna-13B could beat ChatGPT on 66 over 80 tested queries with ChatGPT as an evaluator.
- Source [24]: text found — moving the golden answers to position D degrades the accuracy of gpt-3.5-turbo by 6.3 (from 67.2 to 
- Source [24]: text found — llama-30B selects A/B/C/D 34.6% / 27.3% / 22.3% / 15.8% of the time
- Source [25]: text found — even superior results to state-of-the-art supervised methods on popular IR benchmarks
- Source [26]: text found — we propose permutation self-consistency, a form of self-consistency over the ranking list outputs of
- Source [26]: text found — increasing the scores of GPT-3.5, GPT-4, and LLaMA v2 (70B; Touvron et al., 2023) by up to 4–17%, 9–
- Source [27]: text found — signal for news summarization, though not for other domains
- Source [27]: text found — are not able to identify important information without position
- Source [28]: text found — we investigate whether models can be trained to predict "P(IK)", the probability that "I know" the a
- Source [29]: text found — verbalized confidences emitted as output tokens are
- Source [30]: text found — LLMs, when verbalizing their confidence, tend to be overconfident, potentially imitating human patte
- Source [31]: text found — we reveal a discrepancy between LLMs’ internal encoding and external behavior: they may encode the c
- Source [32]: text found — our trained classifier achieves an average of 71% to 83% accuracy labeling which
- Source [33]: text found — We propose semantic entropy probes (SEPs), a cheap and reliable method for
- Source [34]: text found — by reading the model’s internal estimate of its own competence and steering its verbalized output to
- Source [35]: text found — is not well aligned with its verbalized confidence, leading to misleading results with different cal
- Source [36]: text found — attributions provide stronger simulatability gains than token-level attributions, LLM
- Source [37]: text found — we formalize and investigate the disagreement problem in explainable ML, a novel area of research th
- Source [38]: text found — The model therefore predicts only when the label is forced (i.e., all consistent heads agree), and a
- Source [39]: text found — in which a committee of students is trained on the same data set. The next query is chosen according
- Source [39]: text found — of disagreement among a committee of learners can serve as an estimate of this information value
- Source [40]: text found — We represent the informativeness of new tasks with the disagreement of the current model
- Source [41]: text found — MIST (Model-Internal Saliency for Token-level CoT compression), which defines token importance along
- Source [42]: text found — independent selection can split dependent evidence pairs, retaining one member while deleting the ot
- Source [42]: text found — leaves the answer path incomplete in 34 to 54% of bridge examples across three multi-hop question
- Source [43]: text found — relying solely on attention scores may lead to the removal of important tokens with high attention s
- Source [44]: text found — we study different ways to elicit the self-explanations, evaluate their faithfulness on a set of eva
- Source [45]: text found — on tasks requiring differential sensitivity, we find robust evidence for partial introspection: mode
- Source [46]: text found — Figure 8: Faithfulness evaluation using self-consistency checks, evaluated using Llama2-70B. Results
- Source [47]: text found — including iii) our new self-consistency measure CC-SHAP. CC-SHAP is a fine-grained
- Source [48]: text found — Adding biasing features heavily influences model CoT predictions on BBH tasks, causing accuracy to d
- Source [49]: text found — We find great variation in how much LLMs use CoT on different tasks, not using CoT at all for some t
- Source [50]: text found — CoTs of reasoning models often lack faithfulness and can conceal misalignment. The overall faithfuln
- Source [51]: text found — Llama 70B predicts its own behavior more accurately (48.5%) than GPT-4o (31.8%), despite GPT-4o’s su
- Source [52]: text found — AttriBoT can provide a >300× speedup while remaining more faithful to a target model’s LOO error tha
- Source [52]: text found — We empirically test the assumptions underlying the AttriBoT’s underlying methods on
- Source [53]: text found — { "title": [ "Arthur's Magazine", "First for Women" ], "sent_id": [ 0, 0 ] }
- Source [54]: text found — facts required for reasoning, allowing QA systems to reason with strong supervision and explain the 
- Source [55]: text found — Evidence information in our dataset is a set of triples, where each triple is a structured data (sub
- Source [56]: text found — { "id": 482757, "question": "The Collegian >> owned by", "answer": "Houston Baptist University", "pa
- Source [57]: text found — MuSiQue constitutes unique 21020 single-hop questions, 4132 answers to multihop questions, 19841 ans
- Source [58]: text found — A high score here implies that the rationales were indeed inﬂuential in the prediction, while a low 
- Source [59]: text found — Using only knowledge encoded in their parameters, models’ loose string accuracy ranged from 0.341 (C
- Source [60]: text found — This argument adds tokens to the end of the chat that indicate the start of an
- Source [61]: text found — because a LLM is not trained to continue generation from padding tokens.
- Source [62]: text found — to compute statistical signiﬁcance of test results, and validate them on the concrete example of the
- Source [63]: text found — on the same dataset, one should use the paired version of the statistical signiﬁcance test (such as 

## Follow-up Questions

- Which exact layers/heads give the best Spearman with LOO on Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct, and does the last-token or question-token query win?
- Is the attention-vs-verbalized disagreement rate stable across the task-agnostic and query-aware elicitation variants and across original vs shuffled orderings?
- Does the black-box two-round self-critique flag match attention-based disagreement, and does either predict LOO-critical sentences better than the other?
- Does placeholder-replacement LOO differ materially from deletion LOO on bridge questions (referential dangling), and which should be the primary ground truth?

---
*Generated by AI Inventor Pipeline*
