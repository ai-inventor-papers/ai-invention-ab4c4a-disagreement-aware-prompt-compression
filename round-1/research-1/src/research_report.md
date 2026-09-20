# Spec sheet for three prompt compression baselines

## Summary

Implementation-grade spec (spec.md, ~6.5k words) for reproducing three prompt compressors as sentence-level, matched-budget baselines on Qwen2.5-1.5B-Instruct / Llama-3.2-3B-Instruct for multi-hop QA, plus the evaluation protocol and a calibrated related-work map. Per baseline it gives: source and code status, the exact scoring unit and formula, how the budget is enforced natively, 10-line pseudocode for the sentence-level version, what is dropped and whether the drop is neutral or a stated handicap, transfer notes for the two target models, and published anchor tables.

Key decisions: (1) LLMLingua-2 — use the official xlm-roberta-large checkpoint on 510-token chunks; sentence score = mean word P(preserve) (budget-independent, tie-free); token-level deletion is dropped (expected neutral-to-favourable for a 1-3B reader; measured once by an official token-level run at rate 0.5). (2) LongLLMLingua — sentence score = mean NLL of question + " We can get the answer to this question in the given documents." conditioned on title+sentence, ranked ASCENDING (the code's direction; the paper prose says 'higher r_k' but Eq. 2 is a cross-entropy); no chat template; drop token-level contrastive PPL, dynamic ratio, reordering, recovery. This equals the paper's own 'LongLLMLingua r_k' ablation row, a quantified 1.5-3.3 MultiDoc-point handicap; reordering is dropped in every arm and the paper's LongBench numbers are already without it. (3) EHPC — no public repo, but the NeurIPS supplemental zip contains the authors' code (GemFilter fork); it confirms: attention from the last N_o=16 (Llama) / 4 (Phi) positions averaged over the window, summed over 8 heads of one middle layer (13/32, 14/32, 17/32), avg-pooled kernel 32, top-k tokens in original order plus the window appended; the attention sink is NOT masked; the budget INCLUDES the question tail. Sentence version: per-sentence mean of the un-pooled score, BOS column excluded (an addition), question outside the budget. Transfer recipe: 50 needle probes (400-1500 tokens), eager attention, Eq. 2 with the window-averaged rows, best layer + top-4 heads, mass-fraction and SNR diagnostics, last-3-layers-mean fallback; expect the layer in the middle (≈11-15 of 28), not early.

Protocol: LongBench qa_f1_score (lowercase, strip punctuation/articles/whitespace; bag-of-tokens F1; max over golds) as headline plus EM; LongBench template verbatim, 32 new tokens, greedy; budget B = floor(rho * passage tokens), rho in {0.3,0.5,0.7}, passages only (question/instruction always kept); shared sentence index; greedy fill with first-fail stop; realised kept fraction logged and required within 0.05 across arms; controls full/random/question-only; n >= 100 per dataset. Ratio mapping: LLMLingua '4x' = keep 25%; EHPC uses token targets; Referential Dangling uses r = |C~|/|C|; the published 3,000-token rows bracket rho=0.3.

Anchors (all from the papers' own tables): LLMLingua-2 MultiDoc 33.1 (2k) / 38.7 (3k) vs full prompt 38.7; LongLLMLingua 43.2 / 46.2 (r_k-only 41.7 / 42.9); EHPC EMI 50.7 / 49.1; EHPC NMI per-dataset (Table 6): Llama-3.1-8B HotpotQA 20.64 @1024 vs 16.23 full-KV, Phi-3.5 44.97 @1024. The flagged '20.64 vs 39.0' inconsistency is resolved: different tables, settings and metric columns. None of the three papers reports HotpotQA/2Wiki/MuSiQue individually in its compression-vs-compression table; only EHPC's KV-cache table does.

Related work: FrugalPrompt (arXiv 2510.16439; GlobEnc/DecompX token attribution on a 110M BERT, top-k% tokens, 20% reduction near-lossless on 4 short tasks; query-agnostic), MIST (arXiv 2608.31066, Aug 2026; residual-stream necessity/sufficiency ablation for CoT-token pruning, needs the gold answer, uses Qwen2.5-1.5B-Instruct; the two axes rank tokens differently), Referential Dangling (arXiv 2608.04569, Aug 2026; independent selection splits dependent evidence in 32-60% of HotpotQA bridge cases across six compressors incl. LLMLingua-2 and LongLLMLingua; equal-budget restoration +29-34 points; a diagnostic and add-on, not a competitor). Ends with a 10-item executor checklist and a confidence statement; UNVERIFIED items are marked.

## Research Findings

# Spec sheet: LLMLingua-2, LongLLMLingua and EHPC as sentence-level matched-budget baselines

The full implementation-grade spec is `spec.md` in this artifact (sections per baseline: source, unit and score, budget enforcement, sentence-level pseudocode, omissions, transfer notes, anchor tables; then protocol, related work, a 10-item checklist and a confidence statement). Verbatim quotes with locators are in `notes/*.md`. This answer condenses the findings and decisions.

## 1. LLMLingua-2 (Findings of ACL 2024, arXiv 2403.12968; code github.com/microsoft/LLMLingua)

**Unit and score.** A binary token-classification model: "We approach prompt compression as a token classification task (i.e., preserve or discard), and take the predicted probability of each token being labeled as preserve as the compression metric" [1]. The encoder is xlm-roberta-large (355M, "LLMLingua-2") or multilingual-BERT (110M, "LLMLingua-2-small"), trained with per-word cross-entropy on GPT-4-distilled MeetingBank labels; multi-token words get the mean of their sub-word probabilities [1]. It is explicitly question-agnostic: "LLMLingua-2 is designed for question-agnostic compression" [1].

**Budget enforcement.** Paper: keep the top τN words by p_i in original order [1]. Code: a percentile threshold `np.percentile(new_token_probs, int(100 * reduce_rate + 1))` with `reduce_rate = 1 − rate`; `force_tokens` set a word's probability to 1.0; `drop_consecutive` suppresses repeated forced tokens; the encoder runs on chunks with `self.max_seq_len = 512`, split at `chunk_end_tokens` [2]. `compress_prompt_llmlingua2` returns `compressed_prompt`, `origin_tokens`, `compressed_tokens`, `ratio` ("5.1x") and `rate` [2]. One docstring sentence calls `rate` "≥ 1.0" but the code treats it as the kept fraction; follow the code [2].

**API.** `PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True)` then `compress_prompt(prompt, rate=0.33, force_tokens=['\n','?'])` [3]; the two HF checkpoints carry the same usage snippet and no benchmark numbers [4].

**Sentence-level version (decision).** Run the official checkpoint on the full passage in 510-token sentence-aligned chunks, take word-level P(preserve) (mean over sub-words, index 1 = preserve [2]), and score each sentence by the **mean word P(preserve)**. Mean is budget-independent (one pass serves ratios 0.3/0.5/0.7) and equals the expected fraction of the sentence's words the native method would keep; the alternative "fraction above the global threshold" couples the score to the ratio and ties heavily at 0.3, so it is only an ablation.

**Omitted.** Token deletion inside kept sentences, `force_tokens`, `drop_consecutive`. Direction: whole sentences remove the telegraphic-text penalty for a 1-3B reader and the dependency-splitting failure documented for token-level LLMLingua-2 [22, 23], but also remove LLMLingua-2's ability to spread a budget over more sentences' keywords. Expected neutral-to-favourable; measured directly by one official token-level run at rate 0.5 on 50 examples (checklist item 3).

**Anchors (GPT-3.5-Turbo-0613 answering, LongBench; HotpotQA/2WikiMQA/MuSiQue appear only inside the "MultiDoc" aggregate [1]).** 2,000-token constraint: LLMLingua-2 SingleDoc 29.8 / MultiDoc 33.1 / Avg 39.1 (1,954 tokens, 5x); LLMLingua 22.4 / 32.1 / 34.6; Selective-Context 16.2 / 34.8 / 24.8; LongLLMLingua 39.0 / 42.2 / 48.0. 3,000-token: LLMLingua-2 35.5 / 38.7 / 42.4; LongLLMLingua 40.7 / 46.2 / 48.8. Original prompt 39.7 / 38.7 / 44.0 (10,295 tokens); zero-shot 15.6 / 31.3 / 23.5 [1]. With Mistral-7B answering, LongBench-SingleDoc is 26.8 (2k) / 27.3 (3k) vs 24.5 for the full prompt — compression can beat the full prompt for a small reader [1]. The repo's LongBench evaluation reuses the standard `normalize_answer` / `qa_f1_score` [7] and maps `hotpotqa`, `2wikimqa`, `musique` to `qa_f1_score` individually; aggregation into MultiDoc happens only at reporting time [24].

## 2. LongLLMLingua (ACL 2024, arXiv 2310.06839; same repo)

**Coarse score (Eq. 2).** `r_k = −(1/N_c) Σ_i log p(x_i^{que,restrict} | x_k^doc)`: the mean NLL of the question plus a restrictive statement conditioned on document k [6]. The statement, verbatim from code: `" We can get the answer to this question in the given documents."` [2]. The prose says documents with "higher r_k" are kept, but the code sorts the conditional cross-entropy **ascending** (`sort_direct = 1` unless `condition_in_question == "none"`), so lowest NLL = most relevant = kept first; follow the code [2, 6]. `get_condition_ppl(..., "after")` evaluates `text + question` with the loss restricted to the question tokens — plain string concatenation, no chat template [2].

**Fine score (Eq. 3).** `s_i = perplexity(x_i | x_<i) − perplexity(x_i | x^que, x_<i)`, shown in Appendix A to be proportional to conditional PMI; code path `condition_compare=True` [2, 6].

**Dynamic ratio, reordering, recovery.** Eq. 5 gives a linear budget schedule over the rank index with δτ = 0.3; Eq. 4 reorders documents by r_k; `control_context_budget` consumes documents in rank order until `target_token` (plus `context_budget="+100"`) is spent, then `reorder_context="original"` restores input order while any other value (the README's "sort") leaves rank order [2, 6]. The paper states "For a fair comparison, we only used reordering in the NaturalQuestions Multi-document QA" — the LongBench numbers are without reordering [6]. Subsequence recovery restores original substrings post hoc [6].

**Question outside the budget.** `target_token = (ins + que + Σctx) · rate − ins − que`; instruction and question are separate arguments described as "highly sensitive to compression" [2, 5]; τ_ins = 0.85, τ_que = 0.9, segment size 200, "We set the granular control coefficient k to 2" [6].

**Small LM.** Paper: LLaMA-2-7B-Chat as compressor, GPT-3.5-Turbo-0613 and LongChat-13B-16k as answerers, GPT2-dolly in one ablation [6]; repo default `model_name="NousResearch/Llama-2-7b-hf"` (base, not chat) [2]. The loader is generic `AutoModelForCausalLM` [2], nothing is Llama-specific and no chat template is used, so Qwen2.5-1.5B-Instruct / Llama-3.2-3B-Instruct drop in (BOS differences are irrelevant because only the question-token loss is averaged). No mention of Qwen or Llama-3 in README/DOCUMENT.md (unverified beyond the generic loader).

**Sentence-level version (decision).** Per sentence, `r_s` = mean NLL of the question + restrictive statement conditioned on `title\nsentence`; rank ascending; fill the budget; emit in original order. Drop Eq. 3, Eq. 5, reordering and recovery. This equals the paper's own "LongLLMLingua r_k" ablation row, which trails the full method by 1.5 MultiDoc points at 2,000 tokens (41.7 vs 43.2) and 3.3 at 3,000 (42.9 vs 46.2), and by 3-5 accuracy points on NaturalQuestions 2x [6]. This is a quantified, stated handicap, so the arm is labelled "LongLLMLingua-r_k (sentence)". Reordering is dropped in every arm because the LongBench passages are already randomly ordered [13] and our attention pipeline needs original positions; "lost in the middle" therefore hurts every arm equally.

**Anchors [6].** LongBench 2,000-token: LongLLMLingua 39.9 / 43.2 / 48.3 (1,822 tokens, 6x); r_k 37.8 / 41.7 / 46.3; 3,000-token: 40.7 / 46.2 / 48.8; r_k 37.6 / 42.9 / 46.5. NaturalQuestions accuracy by gold position (1st/5th/10th/15th/20th; reorder): original 75.7/57.3/54.1/55.4/63.1; LongLLMLingua 2x 77.2/72.9/70.8/70.5/70.6 (76.2); 4x 75.0/71.8/71.2/71.2/74.7 (75.5); r_k 2x 73.9/67.7/68.7/66.0/65.6; without question-awareness 42.1/40.3/39.7/40.1/40.3. MuSiQue is also run standalone in the appendix (numbers not extracted; unverified).

## 3. EHPC / Evaluator Heads (NeurIPS 2025, arXiv 2501.12959 — v1 Jan 2025, v2 Feb 2025 [9])

**Code.** No public GitHub repository: the paper cites none, GitHub API searches return only an unofficial 1-star reimplementation [11], and HF Papers lists no code [25]. The NeurIPS checklist says "the code is provided in the supplementary material" [8], and the proceedings page links a Supplemental zip [12]. I downloaded it: it contains the authors' `AttentionCompressor/` (a GemFilter fork pinned to transformers 4.43.3) with `needle_probe.py`, the scoring function `standard_dis_index`, attention hijacks for Llama-3.1/Mistral-Nemo/Phi-3.5, and the stock LongBench eval [10]. So the spec is complete enough to reimplement, and the code settles the open questions below.

**Head identification.** Needle-in-a-Haystack pilot; the per-head score is the accumulated attention on the evidence tokens (paper Eq. 2, not Eq. 1); "We selected 8 heads with the highest scores as evaluator heads from the layer with the highest cumulative score" [8]. Code: Paul Graham essays truncated to `ctx_len × 3.66` chars (16k tokens), the sandwich/Dolores-Park needle inserted at depths 0.1-0.9 (nine probes), per-(layer, head) attention from the last `window_size` query positions summed over needle tokens, averaged over depths; layer = argmax of the row sum; `select_num = 8` [10]. Reported: Llama-3.1-8B-Instruct layer 13 heads [18,13,21,8,11,1,4,3]; CodeLlama-7B layer 14; Phi-3.5-mini layer 17 — all 32-layer models, so 40-53% depth, which the paper calls "one early and significant layer" [8].

**Utility score (Eq. 1).** Sum over evaluator heads of the attention rows of the last N_o positions averaged over the window, then average-pooled with kernel r: "we used the average pooling operation" [8]; N_o = 16, r = 32 for Llama-3.1-8B, N_o = 4, r = 32 for Phi-3.5 [8]. Code: `softmax(QK^T/√d)[:, :, -window_size:, :-window_size].mean(-2)`, sum over `config['heads']`, `F.avg_pool1d(kernel_size, padding=kernel_size//2, stride=1)`, `topk(k − window_size)`, then `new_input_ids = cat([select_input_ids (sorted), input_ids[-window_size:]])` [10].

**Two facts the paper leaves implicit, settled by the code [10].** (a) The attention sink is **not masked**: the key slice starts at position 0 and nothing zeroes the BOS column. (b) The budget `k` **includes the question tail** (the appended window), and question tokens outside the window can be dropped — the opposite of LongLLMLingua's accounting.

**Sentence-level version (decision).** On the target model itself (EHPC's "NMI" setting), capture layer L*'s attention for the last 16 prompt positions, sum over the 4 discovered heads, zero the BOS column and the instruction/question positions, score each sentence by the **mean** per-token attention (length-neutral; sum would favour long sentences), rank, fill, emit in original order. Pooling is dropped because sentence units already give contiguity, which the paper says is pooling's purpose [8]. The sink exclusion is an addition to the published method and is reported as such.

**Transfer recipe for Qwen2.5-1.5B-Instruct (28 layers × 12 heads) and Llama-3.2-3B-Instruct (28 × 24), neither in the paper or code [10].** 50 probes (HotpotQA-train filler, 400-1,500 tokens, needle at depth U(0.05,0.95), LongBench-shaped prompt), `output_attentions=True` with eager attention (≈1.5-3 GB at 1,500 tokens), Eq. 2 with the window-averaged rows and the BOS column excluded, average over probes; L* = argmax layer, H* = top-4 heads (paper: 8 of 32 = 25%); log the mass fraction and needle-vs-random-span SNR; expected L* in the middle (layers ≈11-15 of 28), not early. Fallback if top-4 mass < 2× uniform or SNR < 2: mean attention over the last three layers, labelled "EHPC-fallback". At test time hook only layer L* so memory is O(N_o × N).

**Anchors.** Table 4 (EMI: Llama-3.1-8B compresses, "another model" answers — which one is unverified), category aggregates only: 2,000-token EHPC 44.5 SingleDoc / 50.7 MultiDoc / 49.6 Avg (2,004 tokens); LongLLMLingua 39.0 / 42.2 / 48.0; LLMLingua-2 29.8 / 33.1 / 39.1; 3,000-token EHPC 44.2 / 49.1 / 49.7; LongLLMLingua 40.7 / 46.2 / 48.8; LLMLingua-2 35.5 / 38.7 / 42.2 [8]. Table 6 (NMI, per dataset, F1): Llama-3.1-8B-Instruct full KV HotpotQA 16.23 / 2WikiMQA 16.05 / MuSiQue 11.22; EHPC-1024 20.64 / 16.97 / 13.99; SnapKV-1024 14.81 / 15.73 / 10.69; GemFilter-1024 19.12 / 17.01 / 13.01; EHPC-2048 19.35 / 16.23 / 13.02; Phi-3.5-mini full KV 21.70 / 25.70 / 11.68; EHPC-1024 44.97 / 32.79 / 20.27; EHPC-2048 27.06 / 25.22 / 14.05 [8]. The flagged "20.64 vs 39.0" is not a contradiction: 20.64 is EHPC's HotpotQA F1 in Table 6 (NMI, 1,024 tokens, Llama-3.1-8B answering), 39.0 is LongLLMLingua's SingleDoc aggregate in Table 4 (EMI); within Table 6 EHPC-1024 beats every baseline and the full-KV row on HotpotQA [8]. Table 6 is also the best absolute-level anchor for iteration 2: a small instruct model answering its own compressed context lands at HotpotQA F1 15-25 (Llama-3.1-8B) and 2WikiMQA/MuSiQue 10-33, so n ≥ 100 per dataset is needed to resolve 3-point effects.

## 4. Evaluation protocol

**Metric.** LongBench scores HotpotQA, 2WikiMultihopQA and MuSiQue with F1 (200 examples each; 9,151 / 4,887 / 11,214 words; "built from three Wikipedia-based multi-hop QA datasets"; passages randomly ordered) [13]. `normalize_answer` = "Lower text and remove punctuation, articles and extra whitespace"; `qa_f1_score` is bag-of-tokens F1 on the normalized whitespace-split strings [14]; the per-example score is the max over gold answers [15]; results are `round(100 × mean, 2)`. EM is the SQuAD companion on the same normalization. The LLMLingua repo's copy of the scorer additionally keeps only the first output line for the QA datasets [24]. Prompt template, identical for the three datasets: "Answer the question based on the given passages. Only give me the answer and do not output any other words." … `Question: {input}\nAnswer:` [16]; `max_new_tokens = 32` [17].

**Ratio definitions.** LLMLingua family: 1/τ with τ = compressed/original words, so "4x" = keep 25%; tables also give a token target (2,000/3,000) with the realised token count and 1/τ per row [1, 6]. EHPC: token budgets (2,000/3,000 for EMI; 1,024/2,048 KV-equivalent for NMI) with realised counts [8]. Referential Dangling: kept fraction r = |C̃|/|C| reached by binary search [23]. Mapping onto ρ ∈ {0.3, 0.5, 0.7}: ρ = 0.5 ≈ 2x, 0.3 ≈ 3.3x, 0.7 ≈ 1.4x; a 2,000-token target on a ~10k-token LongBench prompt is ≈ 0.2 (5x) and 3,000 ≈ 0.3 (3x), so the published 3,000-token rows bracket our ρ = 0.3 arm and only LongLLMLingua's NQ 2x row matches ρ = 0.5.

**Question in the budget.** LongLLMLingua: no [2, 5]; LLMLingua-2: question-agnostic, evaluators pass passages only [1, 24]; EHPC: the last window (16-32 tokens) of the question is counted and the rest is compressible [10]. Convention adopted: the budget covers passages only; instruction and question are always kept and never counted (≤ 32-token departure in EHPC's favour).

**Matched-budget protocol (normative, spec §4.4).** B_e = ⌊ρ · n_passage_tokens⌋ in the answering model's tokenizer; one shared sentence segmentation (titles as own units) for every arm; each arm ranks sentences by its own score, fills in descending score with a first-fail stop and earlier-position tie-break; output in original order with paragraph breaks; instruction + question prepended/appended verbatim; realised ρ̂ logged per arm and required to agree within 0.05 on the dataset mean, with mean kept-sentence length as the short-sentence-bias diagnostic; answering with the same 1-3B model, LongBench template, greedy, 32 new tokens; F1 (headline) + EM, paired bootstrap CIs vs the full-passage arm; controls = full passage, random sentences at the same budget (3 seeds), question only; optional OpenRouter LLM-judge as a secondary check.

## 5. Related work (calibrated)

**FrugalPrompt** (Raiyan, Ishmam, Al Imran, Moni; arXiv 2510.16439 v5, code Starscream-11813/Frugal-ICL). Per-token attribution scores from GlobEnc and DecompX on a single 110M BERT with a task-specific scoring function; "rank them to retain the top-k% tokens, and obtain a sparse frugalized prompt" [18]; inputs beyond 512 tokens are scored by sentence-level chunking; headline "a 20% prompt reduction preserves performance for most models" across IMDb sentiment, summarization, CosmosQA and GSM8K [19] — e.g. Llama-3-8B classification 0.949 → 0.942 at 80% retention but GSM8K pass@1 0.786 → 0.500 [19]. Not question-conditioned, not long-context multi-hop, attribution from an external encoder: a query-agnostic classifier-style compressor in LLMLingua-2's family; its Corollary 1 loss bound (summed deleted saliency + a quadratic interaction term) is the formal counterpart of the dependency loss Referential Dangling measures.

**MIST** (Zhao, He, Zheng, Chen; "Every Token Leaves a Ripple in the Stream of Thought", arXiv 2608.31066, 31 Aug 2026). Model-internal token saliency for chain-of-thought compression: necessity = "the drop in answer likelihood when a token's internal contribution is removed", sufficiency = the gain when that contribution alone is patched into a no-chain pass [20]; the two rankings are weakly correlated (Spearman −0.07, top-30% overlap 0.28) on 100 MATH "reasoning chains with Qwen2.5-1.5B-Instruct" [21]. It compresses generated reasoning traces for model adaptation, needs the gold answer at scoring time, and is token-level — not an input-context compressor — but it is the closest interventional (ablation-based) salience definition to attention-based salience, uses our exact 1.5B model, and is direct precedent that two salience signals rank the same tokens differently.

**Referential Dangling** (Hu et al.; "Relevant but Incomplete", arXiv 2608.04569, Aug 2026; code cslikai.cn/Referential-Dangling). A diagnostic, not a compressor: "independent selection can split dependent evidence pairs, retaining one member while deleting the other" [22]. Under Beaver at r = 0.30 the answer path is incomplete in 34.2% (HotpotQA) / 53.5% (2Wiki) / 54.2% (MuSiQue) of bridge examples; "six hard compressors" (Beaver, PartPrompt, Selective-Context, LLMLingua-2, LongLLMLingua, DAC) dangle at 32-60% on a shared HotpotQA bridge set, only 5.4% of examples dangle under all six [23]; reinserting the omitted supporting paragraph at equal budget raises Qwen3-8B accuracy 29-34 points, and an automatic dependency classifier gives +4.7 on HotpotQA at ratio 0.30 → 0.31 [22, 23]. Metric is accuracy, not F1. It is the strongest recent argument for matched-budget evaluation, uses the same kept-fraction definition as iteration 2, and its dependency classifier is an add-on arm rather than a competitor.

## Confidence

High (quoted equations/code): all three scoring rules, budget mechanics, LongLLMLingua's ascending sort and question exclusion, EHPC's head lists, window/kernel values, absent sink masking and in-budget question tail. Medium: the sign of the sentence-level LLMLingua-2 effect (argued, to be measured); EHPC Table 4's answering model. Low/open: whether evaluator heads are as clean in 1.5-3B models as in 8B models — hence the fallback. Unverified: LongLLMLingua's standalone MuSiQue appendix numbers, EHPC's pilot sample count, the LLMLingua `scripts/*.sh` contents.


## Sources

[1] [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://arxiv.org/html/2403.12968) (Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, Dongmei Zhang; 2024) — Primary source for LLMLingua-2: token-classification formulation, encoders, word-probability aggregation, compression strategy (top tau*N words), question-agnostic statement, LongBench/ZeroSCROLLS/Mistral-7B tables.

> We approach prompt compression as a token classification task (i.e., preserve or discard), and take the predicted probability of each token being labeled as preserve as the compression metric.

Locator: Section 1, contributions

> represent the probability of a word by averaging over the predicted probabilities of all subword tokens

Locator: Section 4.2, footnote

> LLMLingua-2 is designed for question-agnostic compression

Locator: Appendix K

[2] [microsoft/LLMLingua — llmlingua/prompt_compressor.py (main branch)](https://raw.githubusercontent.com/microsoft/LLMLingua/main/llmlingua/prompt_compressor.py) — Official code for LLMLingua, LongLLMLingua and LLMLingua-2: percentile threshold on P(preserve), force_tokens/drop_consecutive, 512-token chunking, get_condition_ppl with the restrictive statement, ascending sort of conditional perplexity, question/instruction excluded from target_token, generic AutoModelForCausalLM loader, reorder_context handling.

> We can get the answer to this question in the given documents.

Locator: get_rank_results / get_distance_longllmlingua

> self.max_seq_len = 512

Locator: init_llmlingua2

> sort_direct = -1 if condition_in_question

Locator: get_distance_longllmlingua

[3] [microsoft/LLMLingua README](https://raw.githubusercontent.com/microsoft/LLMLingua/main/README.md) — Documented PromptCompressor API calls for LLMLingua-2 (use_llmlingua2=True, HF checkpoint ids, rate, force_tokens) and LongLLMLingua (question=, rank_method='longllmlingua', condition_in_question, reorder_context='sort', dynamic_context_compression_ratio, condition_compare, context_budget); phi-2 and GPTQ model support.

> model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

Locator: Quick Start, LLMLingua-2 snippet

[4] [microsoft/llmlingua-2-xlm-roberta-large-meetingbank (Hugging Face model card)](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank) — Official LLMLingua-2 checkpoint card: XLM-RoBERTa-large token classifier, p_preserve as the compression metric, usage snippet with compress_prompt_llmlingua2(rate, force_tokens, chunk_end_tokens, drop_consecutive); no benchmark numbers on the card.

[5] [microsoft/LLMLingua DOCUMENT.md (parameter reference)](https://raw.githubusercontent.com/microsoft/LLMLingua/main/DOCUMENT.md) — Parameter semantics: context vs instruction vs question sensitivity, default model_name, granular division principle for multi-document QA.

> also highly sensitive to compression

Locator: Parameters

[6] [LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression](https://arxiv.org/html/2310.06839v2) (Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu; 2024) — Primary source for LongLLMLingua: Eq. 2 (question-conditioned document perplexity with restrictive statement), Eq. 3 (contrastive perplexity), Eq. 4 (reordering), Eq. 5 (dynamic ratio), Appendix A PMI derivation, implementation details (LLaMA-2-7B-Chat, GPT-3.5-Turbo-0613, delta-tau 0.3, tau_ins 0.85, tau_que 0.9), NaturalQuestions Table 1, LongBench Table 2 including the 'LongLLMLingua r_k' rows, ablation Table 3.

> we only used reordering in the NaturalQuestions Multi-document QA

Locator: Appendix B.2

> granular control coefficient

Locator: Appendix B.2

> we propose contrastive perplexity, i.e., the distribution shift caused by the condition of the question

Locator: Section 4.1, fine-grained compression

[7] [microsoft/LLMLingua — experiments/llmlingua2/evaluation/metrics.py](https://raw.githubusercontent.com/microsoft/LLMLingua/main/experiments/llmlingua2/evaluation/metrics.py) — LLMLingua-2's vendored LongBench metrics: normalize_answer (lower, strip punctuation, articles, whitespace), f1_score, qa_f1_score — byte-identical to LongBench's.

> Lower text and remove punctuation, articles and extra whitespace.

Locator: normalize_answer docstring

[8] [Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference (NeurIPS 2025 camera-ready)](https://proceedings.neurips.cc/paper_files/paper/2025/file/e356ed5f27885c79c7cb597bb1107c94-Paper-Conference.pdf) (Weizhi Fei, Xueyan Niu, Guoqing Xie, Yingqing Liu, Bo Bai, Wei Han; 2025) — Primary source for EHPC: Eq. 1 utility score, Eq. 2 evidence-accumulation score, single-layer top-8 head selection, layer/head indices per model, N_o and kernel values, average pooling, attention-sink discussion without masking, Table 1/4 (EMI aggregates), Table 6 (NMI per-dataset HotpotQA/2WikiMQA/MuSiQue), NeurIPS checklist statements on code.

> heads with the highest scores as evaluator heads

Locator: Section 4, Generalizability

> we used the average pooling operation

Locator: Appendix D

> the code is provided in the supplementary material

Locator: NeurIPS Paper Checklist, item 5

[9] [arXiv abstract page for 2501.12959 (Evaluator Heads / EHPC)](https://arxiv.org/abs/2501.12959) (2025) — Version history (v1 Jan 2025, v2 Feb 2025) and title consistency with the NeurIPS camera-ready.

[10] [EHPC NeurIPS 2025 supplemental material (AttentionCompressor code)](https://proceedings.neurips.cc/paper_files/paper/2025/file/e356ed5f27885c79c7cb597bb1107c94-Supplemental-Conference.zip) (2025) — Authors' code (GemFilter fork, transformers 4.43.3): needle_probe.py (Paul Graham haystack, needle at depths 0.1-0.9, per-head evidence attention, argmax layer, top-8 heads), gem_filter_utils.standard_dis_index (window-averaged softmax attention, sum over configured heads, avg_pool1d, topk(k - window_size)), my_generation.get_layer_context (selected ids in original order plus the last window_size ids), LongBench pred.py defaults; confirms no BOS/sink masking and that the budget includes the question tail. Binary zip, inspected locally; not text-matchable.

[11] [comsa33/ehpc-research (unofficial third-party EHPC reimplementation)](https://github.com/comsa33/ehpc-research) — The only GitHub repository returned by API searches for 'EHPC prompt compression' / 'evaluator heads prompt compression'; 1 star, not by the authors.

[12] [NeurIPS 2025 proceedings page for Efficient Prompt Compression with Evaluator Heads](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e356ed5f27885c79c7cb597bb1107c94-Abstract-Conference.html) (2025) — Lists the paper PDF and the Supplemental-Conference.zip download; DOI 10.52202/085713-5171.

[13] [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/pdf/2308.14508) (Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, Juanzi Li; 2024) — Multi-Doc QA construction from HotpotQA / 2WikiMultihopQA / MuSiQue (200 examples each, F1 metric, average lengths), random ordering of passages, automatic metrics F1 and ROUGE-L.

> built from three Wikipedia-based

Locator: Section 3, Multi-Doc QA

[14] [THUDM/LongBench — LongBench/metrics.py](https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/metrics.py) — Reference normalize_answer, f1_score and qa_f1_score implementations used by LongBench for HotpotQA/2WikiMQA/MuSiQue.

> Lower text and remove punctuation, articles and extra whitespace.

Locator: normalize_answer docstring

[15] [THUDM/LongBench — LongBench/eval.py](https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/eval.py) — dataset2metric mapping (hotpotqa/2wikimqa/musique -> qa_f1_score) and the scorer taking the max over gold answers and reporting round(100*mean, 2).

> score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))

Locator: scorer

[16] [THUDM/LongBench — config/dataset2prompt.json](https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/config/dataset2prompt.json) — Exact prompt template for hotpotqa, 2wikimqa and musique.

> Answer the question based on the given passages. Only give me the answer and do not output any other words.

Locator: hotpotqa / 2wikimqa / musique entries

[17] [THUDM/LongBench — config/dataset2maxlen.json](https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/config/dataset2maxlen.json) — Max generation length 32 tokens for hotpotqa, 2wikimqa, musique.

> "hotpotqa": 32

Locator: json entry

[18] [FrugalPrompt: Reducing Contextual Overhead in Large Language Models via Token Attribution](https://arxiv.org/abs/2510.16439) (Syed Rifat Raiyan, Md Farhan Ishmam, Abdullah Al Imran, Mohammad Ali Moni; 2025) — Abstract and metadata (v5, Sep 2026): GlobEnc/DecompX token attribution, top-k% retention, four tasks, asymmetric retention-performance patterns.

> rank them to retain the top-k% tokens, and obtain a sparse frugalized prompt

Locator: Abstract

[19] [FrugalPrompt (PDF)](https://arxiv.org/pdf/2510.16439) (Syed Rifat Raiyan, Md Farhan Ishmam, Abdullah Al Imran, Mohammad Ali Moni; 2025) — Method details: 110M BERT encoder, task-specific scoring function, sentence-level chunking beyond 512 tokens, top ceil(k/100*n) rule, Corollary 1 bound, results table (Llama-3 8B GlobEnc 80%: CLS 0.942 vs 0.949, GSM8K 0.500 vs 0.786), datasets IMDb / CosmosQA / GSM8k, Frugal-ICL code link.

> a 20% prompt reduction preserves

Locator: Section 1, contributions

[20] [Every Token Leaves a Ripple in the Stream of Thought: Eliciting Model-Internal Token Saliency for Chain-of-Thought Compression (MIST)](https://arxiv.org/abs/2608.31066) (Tianyi Zhao, Yinhan He, Wendy Zheng, Chen Chen; 2026) — Abstract: MIST defines necessity (drop in answer likelihood when a token's residual contribution is removed) and sufficiency (gain when that contribution alone is provided) for token-level CoT compression; four benchmarks, four models.

> the drop in answer likelihood when a token's internal contribution is removed

Locator: Abstract

[21] [MIST (PDF)](https://arxiv.org/pdf/2608.31066) (Tianyi Zhao, Yinhan He, Wendy Zheng, Chen Chen; 2026) — Figure 1/2 details: residual-stream patching into a no-chain forward pass; 100 MATH chains with Qwen2.5-1.5B-Instruct; necessity and sufficiency rankings weakly correlated (Spearman -0.07, top-30% overlap 0.28); TokenSkip-style auxiliary scorers as the contrast.

> reasoning chains with Qwen2.5-1.5B-Instruct

Locator: Figure 2 caption

[22] [Relevant but Incomplete: Referential Dangling as a Paradigm-Level Failure Mode in Hard Prompt Compression](https://arxiv.org/abs/2608.04569) (Zhengpei Hu, Kai Li, Dapeng Fu, Xuechao Zou, Yuanhao Tang, Yue Li, Tengfei Cao, Jianqiang Huang; 2026) — Abstract: independent unit selection splits dependent evidence pairs; Beaver at ratio 0.30 leaves 34-54% of bridge examples incomplete; six compressors dangle up to 60%; equal-budget restoration +29-34 points with Qwen3-8B; automatic classifier +4.7 on HotpotQA at 0.30->0.31; code link.

> independent selection can split dependent evidence pairs, retaining one member while deleting the other

Locator: Abstract

[23] [Referential Dangling (PDF)](https://arxiv.org/pdf/2608.04569) (Zhengpei Hu, Kai Li, Dapeng Fu, Xuechao Zou, Yuanhao Tang, Yue Li, Tengfei Cao, Jianqiang Huang; 2026) — Section 4: ratio r = |C~|/|C| reached by binary search; contexts of ~40 paragraphs / 5.7k tokens; Table 1 per-dataset dangling rates (34.2 / 53.5 / 54.2); six compressors (Beaver, PartPrompt, Selective-Context, LLMLingua-2, LongLLMLingua, DAC) at 32-60%; Table 2 Jaccard overlaps; 5.4% dangle under all six; GPT-5.5 / GLM-5.2 / Qwen3 answer models with accuracy metric.

> six hard compressors that assign

Locator: Section 4.2

[24] [microsoft/LLMLingua — experiments/llmlingua2/evaluation/eval_longbench.py](https://raw.githubusercontent.com/microsoft/LLMLingua/main/experiments/llmlingua2/evaluation/eval_longbench.py) — LLMLingua-2's LongBench evaluation: per-dataset dataset2metric keys for hotpotqa/2wikimqa/musique (qa_f1_score), first-line truncation of QA predictions, max over gold answers, tiktoken-based token counting.

> "hotpotqa": qa_f1_score

Locator: dataset2metric

[25] [Hugging Face Papers page for 2501.12959](https://huggingface.co/papers/2501.12959) — No models, datasets or spaces citing the EHPC paper; no code link.

## Verification

Numbered citations resolve to unique listed sources. Passage checks test text occurrence, not claim truth or entailment. Author/year metadata and locators are not independently verified. Details: `research_verification.json`.

- Source [1]: text found — We approach prompt compression as a token classification task (i.e., preserve or discard), and take 
- Source [1]: text found — represent the probability of a word by averaging over the predicted probabilities of all subword tok
- Source [1]: text found — LLMLingua-2 is designed for question-agnostic compression
- Source [2]: text found — We can get the answer to this question in the given documents.
- Source [2]: text found — self.max_seq_len = 512
- Source [2]: text found — sort_direct = -1 if condition_in_question
- Source [3]: text found — model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
- Source [5]: text found — also highly sensitive to compression
- Source [6]: text found — we only used reordering in the NaturalQuestions Multi-document QA
- Source [6]: text found — granular control coefficient
- Source [6]: text found — we propose contrastive perplexity, i.e., the distribution shift caused by the condition of the quest
- Source [7]: text found — Lower text and remove punctuation, articles and extra whitespace.
- Source [8]: text found — heads with the highest scores as evaluator heads
- Source [8]: text found — we used the average pooling operation
- Source [8]: text found — the code is provided in the supplementary material
- Source [13]: text found — built from three Wikipedia-based
- Source [14]: text found — Lower text and remove punctuation, articles and extra whitespace.
- Source [15]: text found — score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
- Source [16]: text found — Answer the question based on the given passages. Only give me the answer and do not output any other
- Source [17]: text found — "hotpotqa": 32
- Source [18]: text found — rank them to retain the top-k% tokens, and obtain a sparse frugalized prompt
- Source [19]: text found — a 20% prompt reduction preserves
- Source [20]: text found — the drop in answer likelihood when a token's internal contribution is removed
- Source [21]: text found — reasoning chains with Qwen2.5-1.5B-Instruct
- Source [22]: text found — independent selection can split dependent evidence pairs, retaining one member while deleting the ot
- Source [23]: text found — six hard compressors that assign
- Source [24]: text found — "hotpotqa": qa_f1_score

## Follow-up Questions

- Do Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct have a single layer whose top-4 heads capture a clearly larger share of needle attention than uniform (the paper's 8-of-32 pattern in 8B models), or does evidence attention spread across layers in small models so that the last-3-layers fallback becomes the de-facto EHPC arm?
- How large is the sentence-level vs token-level gap for LLMLingua-2 with a 1-3B reader at rate 0.5 on HotpotQA — does whole-sentence selection help (grammar) or hurt (fewer sentences covered), and does the same sign hold for LongLLMLingua's fine-grained contrastive step?
- Which answering model produced EHPC's Table 4 (EMI) numbers, and do its 2,000-token MultiDoc gains over LongLLMLingua survive when both are re-run with the same answering model and the question excluded from the budget?

---
*Generated by AI Inventor Pipeline*
