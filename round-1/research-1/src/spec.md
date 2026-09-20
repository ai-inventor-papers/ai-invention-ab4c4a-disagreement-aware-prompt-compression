# Spec sheet: three prompt compressors as sentence-level matched-budget baselines

Implementation-grade specification for reproducing LLMLingua-2, LongLLMLingua and EHPC (Evaluator Heads) as
**sentence-level, matched-budget** baselines on Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct, for the
iteration-2 comparison against disagreement-protected compression on HotpotQA / 2WikiMultihopQA / MuSiQue.
Every factual claim is tied to a primary source (arXiv HTML/PDF, the official repos, the NeurIPS supplemental
code, the LongBench repo). Verbatim quotes and locators live in `notes/*.md`; `research_out.json` carries the
numbered citations. Items marked **UNVERIFIED** could not be confirmed from a primary source.

Date of research: 2026-09-20.

## 0. Headline decisions (read this if nothing else)

| Baseline | Native unit / score | Sentence-level score used here | Kept from the original | Dropped (and why the drop is neutral or is a stated handicap) |
|---|---|---|---|---|
| LLMLingua-2 | word; `P(preserve)` from a fine-tuned XLM-RoBERTa-large token classifier (query-agnostic) | mean of word-level `P(preserve)` over the sentence, from the **official checkpoint** run on the full passage in 512-token chunks | the real model, its tokenisation, its word-probability aggregation | token deletion inside kept sentences, `force_tokens`, `drop_consecutive` — neutral-to-favourable for LLMLingua-2 with a 1-3B reader (grammar preserved); measure the gap once with the official token-level run at ρ=0.5 |
| LongLLMLingua | document: `r_k` = mean NLL of (question + "We can get the answer to this question in the given documents.") given the document; token: contrastive PPL | `r_s` = the same coarse score with the **sentence (plus its paragraph title)** as the conditioning document; rank ascending (lowest NLL first) | the question-aware conditional-perplexity signal, the exact restrictive sentence, plain-string concatenation (no chat template) | token-level contrastive PPL, dynamic ratio, **reordering**, subsequence recovery. Corresponds to the paper's own "LongLLMLingua r_k" ablation row, which trails the full method by 1.5-3.3 MultiDoc points on LongBench (a stated handicap); reordering is dropped in *every* arm so its absence is not arm-specific |
| EHPC | token; sum over evaluator heads of attention from the last N_o prompt positions to each token, avg-pooled | per-sentence **mean** of the un-pooled per-token score, computed on the target model's own evaluator heads found by a needle probe; BOS/sink column excluded | the evaluator-head idea, the observation-window scoring, single-layer head choice | pooling (sentence aggregation replaces it), token-level top-k, in-budget question tail; the sink exclusion is an *addition* the paper does not make and must be reported as such |

Budget protocol (Section 4): budget `B = ⌊ρ · n_passage_tokens⌋`, ρ ∈ {0.3, 0.5, 0.7}, passages only; the
LongBench instruction and the question are always kept and never counted; every arm fills whole sentences in
descending score with a first-fail stop; output in original order; realised kept fraction logged per arm and
required to agree within 0.05 across arms; LongBench `qa_f1_score` (max over golds) as headline plus EM.

## 1. LLMLingua-2 (Pan et al., Findings of ACL 2024; arXiv 2403.12968)

**Source.** Paper: https://arxiv.org/html/2403.12968. Code: https://github.com/microsoft/LLMLingua
(`llmlingua/prompt_compressor.py`, `experiments/llmlingua2/`). Checkpoints:
`microsoft/llmlingua-2-xlm-roberta-large-meetingbank` (355M encoder, "LLMLingua-2") and
`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` (110M, "LLMLingua-2-small"). Training data:
`microsoft/MeetingBank-LLMCompressed` (GPT-4-distilled extractive labels on MeetingBank).

**Unit and score.** Binary token classification: "We formulate prompt compression as a binary token
classification problem (i.e., preserve or discard)"; `p(x_i, Θ) = softmax(W h_i + b)` over {preserve, discard}
with `h = f_θ(x)` from the bidirectional encoder (Eq. 5-6); loss is per-word cross-entropy (Eq. 7). Multi-token
words: "we preserve the integrity of multi-token words and represent the probability of a word by averaging over
the predicted probabilities of all subword tokens" (footnote 2, Sec. 4.2); code:
`__token_prob_to_word_prob(..., convert_mode="mean")`. Label index 1 is *preserve* (`probs[j, :, 1]` in
`__get_context_prob`). The score is **question-agnostic**: "LLMLingua-2 is designed for question-agnostic
compression" (Appendix K); the paper attributes its LongBench gap to LongLLMLingua to "the additional information
that they leverage from the question".

**Budget enforcement (native).** Sec. 4.2: target ratio 1/τ, τ = compressed words / original words; keep the top
Ñ = τN words by p_i "and maintain their original order". Code: `threshold = np.percentile(new_token_probs,
int(100 * reduce_rate + 1))` with `reduce_rate = 1 − rate`; keep words with `word_prob > threshold`.
`force_tokens` (and `force_reserve_digit`) set a word's probability to 1.0; `drop_consecutive` zeroes a forced
token that repeats immediately. The encoder is run on chunks: `max_seq_len = 512`, `__chunk_context` splits at
the nearest `chunk_end_tokens` (default `[".", "\n"]`) before the 510-token mark. `compress_prompt_llmlingua2`
returns `compressed_prompt`, `origin_tokens`, `compressed_tokens`, `ratio` ("5.1x"), `rate` (percent kept).
Docstring caveat: one sentence describes `rate` as "≥ 1.0" but the code and default (`rate=0.5`) treat it as
the kept fraction in [0, 1] — follow the code.

**What the sentence-level reproduction computes.**
```
model = AutoModelForTokenClassification("microsoft/llmlingua-2-xlm-roberta-large-meetingbank"); tok = its tokenizer
for each example:
    passage_text = "\n".join(paragraphs)                      # titles kept as their own line
    chunks = chunk_on_sentence_boundaries(passage_text, tok, max_tokens=510)   # mirrors __chunk_context
    word_probs = []                                            # one P(preserve) per whitespace word, in order
    for chunk in chunks:
        ids = tok(chunk, return_tensors="pt", truncation=True, max_length=512)
        p_keep = softmax(model(**ids).logits, -1)[0, :, 1]     # index 1 = preserve
        word_probs += mean_over_subwords(p_keep, ids)          # convert_mode="mean", skip CLS/SEP
    for each sentence s (from the shared sentence index, Section 4.4):
        score[s] = mean(word_probs[w] for w in words_of(s))     # length-normalised, budget-independent
    rank sentences by score desc; fill budget B with the shared first-fail rule; emit in original order
```
Why *mean* P(preserve) and not "fraction of words above the global threshold": the mean is budget-independent
(one scoring pass serves ρ = 0.3/0.5/0.7), has no ties, and equals the expected fraction of the sentence's words
LLMLingua-2 would keep under a uniformly random threshold, so it ranks sentences by how much of them the native
method would retain. The fraction-above-threshold variant couples the score to ρ and produces many exact ties at
ρ = 0.3; report it only as an ablation. Use the xlm-roberta-large checkpoint as the baseline (LLMLingua-2-small
gives near-identical LongBench numbers and is an optional speed arm).

**What is omitted and why it does not tilt the comparison.** Omitted: token-level deletion inside kept
sentences, `force_tokens`/`drop_consecutive`, the +0/−1 word threshold offset. Direction of the effect: token
deletion is where LLMLingua-2 both wins (it can spread a fixed budget over more sentences' keywords) and loses
(telegraphic text is harder for a small reader; MultiDoc QA is the category where LLMLingua-2 sits closest to
Selective-Context in its own Table 2). Whole-sentence selection removes the grammar penalty, so for a 1-3B reader
the sentence-level variant is expected neutral-to-favourable for LLMLingua-2; it also removes the
dependency-splitting failure that Referential Dangling documents for token-level LLMLingua-2. Because the sign is
argued, not measured, the checklist includes one direct measurement: run the official `compress_prompt_llmlingua2`
(token-level, `force_tokens=['\n','.','?','!',',']`, `drop_consecutive=True`) at `rate=0.5` on the same 50
examples and report both numbers.

**Transfer notes.** Nothing to transfer: the compressor is a fixed encoder independent of the answering model.
Tokenisation for the *budget* is the answering model's tokenizer (Section 4.4); the encoder's own word pieces are
used only for scoring. Run on CPU or a small GPU; 512-token chunks × ~20 chunks per HotpotQA example.

**Anchor numbers (paper Table 2, answering model GPT-3.5-Turbo-0613, LongBench; F1 for QA columns).**
HotpotQA / 2WikiMQA / MuSiQue are **never reported individually** in the paper — only the "MultiDoc" aggregate.

| Constraint | Method | SingleDoc | MultiDoc | Avg | Tokens | 1/τ |
|---|---|---|---|---|---|---|
| 2,000 tok | LLMLingua-2 | 29.8 | 33.1 | 39.1 | 1,954 | 5x |
| 2,000 tok | LLMLingua-2-small | 29.5 | 32.0 | 38.2 | 1,891 | 5x |
| 2,000 tok | LLMLingua (†) | 22.4 | 32.1 | 34.6 | 1,950 | 5x |
| 2,000 tok | Selective-Context (†) | 16.2 | 34.8 | 24.8 | 1,925 | 5x |
| 2,000 tok | LongLLMLingua (†) | 39.0 | 42.2 | 48.0 | 1,809 | 6x |
| 3,000 tok | LLMLingua-2 | 35.5 | 38.7 | 42.4 | 3,392 | 3x |
| 3,000 tok | LLMLingua-2-small | 35.5 | 38.1 | 41.9 | 3,278 | 3x |
| 3,000 tok | LongLLMLingua (†) | 40.7 | 46.2 | 48.8 | 3,283 | 3x |
| — | Original prompt | 39.7 | 38.7 | 44.0 | 10,295 | — |
| — | Zero-shot (no context) | 15.6 | 31.3 | 23.5 | 214 | 48x |

(† = numbers copied by the LLMLingua-2 authors from the LongLLMLingua paper.) Sanity reading: at 3-5x, a
question-agnostic compressor loses 0-6 MultiDoc points against the full prompt, while the question-aware
LongLLMLingua *gains* 3.5-7.5 points — the size of the question-aware advantage is the first thing a tiny
reproduction should recover in sign. Mistral-7B as answering model (Table 4): LongBench-SingleDoc 26.8 (2k) /
27.3 (3k) vs 24.5 full prompt — compression can beat the full prompt for a small reader.

## 2. LongLLMLingua (Jiang et al., ACL 2024; arXiv 2310.06839)

**Source.** Paper: https://arxiv.org/html/2310.06839v2. Code: same repo, `PromptCompressor.compress_prompt(context=[...],
instruction=..., question=..., target_token=..., rank_method="longllmlingua", condition_in_question="after_condition",
reorder_context="sort", dynamic_context_compression_ratio=0.3, condition_compare=True, context_budget="+100")`;
parameter semantics in `DOCUMENT.md`. Small LM in the paper: LLaMA-2-7B-Chat ("we apply LLaMA-2-7B-Chat"); repo
default `model_name="NousResearch/Llama-2-7b-hf"` (base, not chat) — a documented discrepancy. Answering models:
GPT-3.5-Turbo-0613 (16k variant above 4k tokens) and LongChat-13B-16k, greedy.

**Unit and score.**
- Coarse (document) score, Eq. 2: `r_k = −(1/N_c) Σ_i log p(x_i^{que,restrict} | x_k^doc)`, the mean NLL of the
  question followed by the restrictive statement, conditioned on document k. Restrictive statement, verbatim from
  code: `" We can get the answer to this question in the given documents."` appended to the question. Direction:
  the paper's prose says documents with "higher r_k" are kept, but Eq. 2 is a cross-entropy and the code sorts
  `context_ppl` **ascending** (`sort_direct = 1` for `condition_in_question != "none"`), so **lowest NLL = most
  relevant = kept first**. Follow the code. `get_condition_ppl(text, question, "after")` computes `get_ppl(text +
  question, condition_mode="after", condition_pos_id=len(text)−1)`, i.e. one forward pass on the plain
  concatenation with the loss restricted to the question tokens; no chat template anywhere.
- Fine (token) score, Eq. 3: `s_i = perplexity(x_i | x_<i) − perplexity(x_i | x^que, x_<i)`, shown in Appendix A
  to be proportional to conditional PMI; code path `condition_compare=True` → `condition_flag`, thresholds via
  `get_estimate_threshold_base_distribution(ppl, ratio, condition_flag)`.
- Ablation anchor: the paper's Table 1/2 include a "LongLLMLingua r_k" row = coarse score only (used as a
  retrieval ranker). That row is the closest published analogue of our sentence-level version.

**Budget enforcement (native).** `target_token` applies to the *context list*; the instruction and question are
separate arguments and are not compressed away: `target_token = (ins + que + Σ ctx) · rate − ins − que` when
`rate` is given; `context_budget="+100"` adds slack; τ_ins = 0.85, τ_que = 0.9 are separate (near-1) rates in the
paper. Documents are consumed in rank order until the budget is exhausted (`control_context_budget`), then the
survivors get a linear dynamic ratio (Eq. 5, δτ = 0.3): `τ_k = clip((1 − 2·I(r_k)/K') · δτ + τ^doc, 0, 1)`, then
iterative token-level compression (segment size 200), then `reorder_context`: `"original"` restores input order,
`"two_stage"` alternates, any other value (the README's `"sort"`) leaves the documents in **rank order** (most
relevant first). Subsequence recovery restores original substrings from the LLM's answer post hoc.

**What the sentence-level reproduction computes.**
```
lm = AutoModelForCausalLM(target_model); tok = its tokenizer          # Qwen2.5-1.5B-Instruct or Llama-3.2-3B-Instruct
Q = question + " We can get the answer to this question in the given documents."
for each example:
    for each sentence s (shared index) with paragraph title t:
        doc = f"{t}\n{s}" if t else s                                 # sentence as the 'document'
        ids = tok(doc + Q)                                            # plain concatenation, no chat template
        loss = per_token_nll(lm, ids)[len(tok(doc)):]                 # loss only on the Q tokens (condition_mode='after')
        r[s] = loss.mean()                                            # Eq. 2 at sentence granularity
    rank sentences by r ascending (lowest NLL first); fill budget B with the shared first-fail rule; emit in ORIGINAL order
```
Batch all sentences of an example with left padding; ~40 sentences × ~40 Q tokens per example, one pass each.
Conditioning on the sentence *with its paragraph title* keeps the entity anchor that LongBench passages put in
the title line; it is the sentence-level analogue of conditioning on the whole document and costs nothing.

**What is omitted and why it does not tilt the comparison.**
- *Token-level contrastive compression (Eq. 3), dynamic ratio (Eq. 5), τ_ins/τ_que, subsequence recovery*: these
  refine *within* kept documents; at sentence granularity there is nothing left to refine. The paper's own
  "LongLLMLingua r_k" rows measure the cost: LongBench MultiDoc 41.7 (r_k) vs 43.2 (full) at 2,000 tokens and
  42.9 vs 46.2 at 3,000 tokens — a 1.5-3.3 point handicap for the coarse-only signal, and on NQ 2x the r_k row
  trails the full method by 3-5 points at every gold position. This is a *known, quantified* handicap, not a
  neutral drop; report our arm as "LongLLMLingua-r_k (sentence)" so the reader knows which published row it
  corresponds to, and do not describe it as the full LongLLMLingua system.
- *Reordering*: dropped in every arm, because the attention-salience pipeline depends on original positions and
  the LongBench passages are already randomly ordered. The paper's Table 1 shows reorder adds +0.5 to +4 points
  for LongLLMLingua on NaturalQuestions (e.g. 2x: 76.2 reordered vs 77.2/70.6 at 1st/20th position), and the
  paper only used it for NQ ("For a fair comparison, we only used reordering in the NaturalQuestions
  Multi-document QA"). So the LongBench anchors below are *without* reordering, matching our setting exactly;
  all arms keep original order, so "lost in the middle" penalises every arm equally.

**Transfer notes (Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct).** The repo loads any `AutoModelForCausalLM`
(`load_model` uses `AutoConfig/AutoTokenizer/AutoModelForCausalLM` generically; README documents phi-2 and GPTQ
Llama-2). No chat template is involved, so instruct models work as plain LMs. Practicalities: Llama tokenizers
prepend BOS, Qwen2.5 does not — irrelevant because only the question-token loss is averaged; use `torch.bfloat16`,
left padding, `attention_mask` for batching; do **not** pass the LongBench instruction inside `doc` (the paper
conditions on the document alone). No Qwen/Llama-3 mention exists in README/DOCUMENT.md (UNVERIFIED beyond the
generic loader); nothing in the scoring path is Llama-2-specific.

**Anchor numbers.** LongBench, GPT-3.5-Turbo, no reordering (Table 2); NaturalQuestions 20-doc accuracy (Table 1).

| Setting | Method | SingleDoc | MultiDoc | Avg | Tokens | 1/τ |
|---|---|---|---|---|---|---|
| LB 2,000 tok | LongLLMLingua | 39.9 | 43.2 | 48.3 | 1,822 | 6x |
| LB 2,000 tok | LongLLMLingua r_k (coarse only) | 37.8 | 41.7 | 46.3 | 1,960 | 5x |
| LB 2,000 tok | LLMLingua | 22.4 | 32.1 | 34.6 | 1,950 | 5x |
| LB 2,000 tok | SBERT retrieval | 33.8 | 35.9 | 25.8 | 1,947 | 5x |
| LB 3,000 tok | LongLLMLingua | 40.7 | 46.2 | 48.8 | 3,283 | 3x |
| LB 3,000 tok | LongLLMLingua r_k | 37.6 | 42.9 | 46.5 | 3,424 | 3x |
| LB — | Original prompt | 39.7 | 38.7 | 44.0 | 10,295 | — |

| NQ (GPT-3.5), accuracy at gold position | 1st | 5th | 10th | 15th | 20th | Reorder | Tokens |
|---|---|---|---|---|---|---|---|
| Original prompt | 75.7 | 57.3 | 54.1 | 55.4 | 63.1 | — | 2,946 |
| LongLLMLingua 2x | 77.2 | 72.9 | 70.8 | 70.5 | 70.6 | 76.2 | 1,429 |
| LongLLMLingua 4x | 75.0 | 71.8 | 71.2 | 71.2 | 74.7 | 75.5 | 748 |
| LongLLMLingua r_k 2x | 73.9 | 67.7 | 68.7 | 66.0 | 65.6 | 74.3 | 1,548 |
| LLMLingua 2x | 39.7 | 39.5 | 40.4 | 37.1 | 42.3 | 41.5 | 1,410 |
| − w/o question-awareness (Table 3, 2x) | 42.1 | 40.3 | 39.7 | 40.1 | 40.3 | — | — |

Note the LLMLingua-2 paper's copy of the LongLLMLingua 2,000-token row (39.0 / 42.2 / 48.0, 1,809 tokens)
differs slightly from LongLLMLingua's own Table 2 (39.9 / 43.2 / 48.3, 1,822 tokens); both are quoted verbatim
from their respective papers. MuSiQue is also run as a standalone benchmark in the LongLLMLingua appendix
(C.5, validation set, official Trivedi et al. scripts) — its numbers were not extracted (UNVERIFIED).

## 3. EHPC — Evaluator Heads Prompt Compression (Fei et al., NeurIPS 2025; arXiv 2501.12959)

**Source.** Paper: arXiv 2501.12959 (v1 Jan 2025, v2 Feb 2025; same title as the camera-ready "Efficient Prompt
Compression with Evaluator Heads for Long-Context Transformer Inference"); NeurIPS 2025 PDF at
proceedings.neurips.cc (hash `e356ed5f27885c79c7cb597bb1107c94`); OpenReview `yOs12gdsaL` (forum page is behind a
bot check). **Code: no public GitHub repository** (GitHub API searches for "EHPC prompt compression" and
"evaluator heads prompt compression" return one unofficial 1-star reimplementation, `comsa33/ehpc-research`; HF
Papers and PapersWithCode list no code). **However, the NeurIPS supplemental zip contains the authors' code**
(`AttentionCompressor/`, 1.4 MB, a fork of GemFilter pinned to `transformers==4.43.3`, `flash-attn==2.6.3`):
`needle_probe.py`, `my_utils/my_generation.py`, `my_baseline/GemFilter/gem_filter_utils.py`, attention hijacks
for Llama-3.1 / Mistral-Nemo / Phi-3.5 only, and the stock LongBench `pred.py/eval.py/metrics.py`. Everything
below that says "code" refers to this zip (`notes/ehpc_supplemental_code_addendum.md`).

**Evaluator-head identification (paper Sec. 3.4 + `needle_probe.py`).** Probe = Needle-in-a-Haystack: Paul Graham
essays truncated to `ctx_len × 3.66` chars (default 16,000 tokens); needle `"The best thing to do in San Francisco
is eat a sandwich and sit in Dolores Park on a sunny day."` inserted by sentence index at depths 0.1 … 0.9 (nine
probes, one haystack); prompt `"<|im_start|> This is a very long story book: <book> … </book>.\nBased on the
content of the book, Question: What is the best thing to do in San Francisco?\nAnswer:"`. Per (layer, head)
score = paper Eq. 2, `â_h^l = Σ_{j∈I_e} a_h^l[j]`, where in code `a_h^l` is the attention of the last
`window_size` query positions averaged over the window (not just the single last row); scores are averaged over
the nine depths; `score_layers = avg_score.sum(1)`; **one layer = argmax**; **top-8 heads** in that layer
(`select_num = 8`), plus a printed mass fraction `topk(8).sum() / layer.sum()`. Reported heads (Appendix D /
README): Llama-3.1-8B-Instruct layer 13 heads [18,13,21,8,11,1,4,3]; CodeLlama-7B layer 14 heads
[24,3,18,7,29,2,9,1]; Phi-3.5-mini layer 17 heads [7,17,30,2,6,16,25,18] — all 32-layer models, so the chosen
layer sits at 40-53 % depth (the paper calls this "one early and significant layer").

**Utility score (paper Eq. 1 + `standard_dis_index`).** `s = Σ_{(l,h)∈C_f} Pool( Σ_{N_r≤i≤N} A_{l,h}[i,:] / N_o , r )`:
softmax attention rows of the last `N_o` prompt positions (the "observation window"; in LongBench prompts that is
the tail `…Question: {input}\nAnswer:`), averaged over the window, summed over the evaluator heads, then 1-D
average-pooled with kernel `r` (stride 1, padding r/2). Hyper-parameters: N_o = 16, r = 32 (Llama-3.1-8B);
N_o = 4, r = 32 (Phi-3.5); `pred.py` CLI defaults are `--window 32 --kernel 4` (code defaults differ from the
paper's stated values; the paper's values are authoritative for its tables).

**Budget enforcement (native).** `top_k = torch.topk(pooled, k − window_size)`; selected indices sorted (original
order) and the last `window_size` input ids appended: `new_input_ids = cat([select_input_ids,
input_ids[-window_size:]])`. So the budget `k` (1,024/2,048 in the NMI tables; 2,000/3,000 in the EMI table)
**includes the question tail**, and question tokens outside the window can be dropped — unlike LongLLMLingua.
Token ids, not spans, are re-fed to the model; readability comes only from pooling. **Attention sink: not
masked** — the key slice `[:, :, -window_size:, :-window_size]` starts at position 0 and nothing zeroes the BOS
column; the paper cites the sink phenomenon only as motivation for attention sparsity.

**What the sentence-level reproduction computes.**
```
# one-off per target model: head discovery (Section 3, transfer recipe below) -> layer L*, heads H* (|H*| = 4)
lm = AutoModelForCausalLM(target_model, attn_implementation="eager")   # needed for attention weights
for each example:
    prompt = LongBench_template(passages, question)                    # instruction + passages + question, as answered
    A = attention_weights(lm, prompt, layer=L*)                       # via a forward hook on layer L*; shape [heads, N, N]
    W = last N_o (=16) positions of the prompt                        # observation window = end of the question
    s = A[H*][:, W, :].mean(dim=1).sum(dim=0)                         # Eq. 1 without pooling; length N
    s[0] = 0                                                          # exclude the BOS / first-token sink column (our addition)
    s[instruction_positions] = 0; s[question_positions] = 0           # only passage tokens are candidates
    for each sentence t (shared index): score[t] = s[tokens_of(t)].mean()   # length-normalised mean
    rank sentences by score desc; fill budget B with the shared first-fail rule; emit in original order
```
Mean, not sum: the sum favours long sentences mechanically; the mean is length-neutral and the realised-budget
check (Section 4.4) catches any residual short-sentence bias. Pooling is dropped because sentence aggregation
already enforces contiguity; the plan's kernel `r` therefore has no analogue and should not be reintroduced.

**What is omitted and why it does not tilt the comparison.** (i) *Pooling and token-level top-k* — replaced by
sentence units for every arm; the paper's stated purpose of pooling is readability ("continuous rather than
isolated"), which whole sentences deliver by construction. (ii) *Question tail inside the budget* — our protocol
keeps the question outside the budget for all arms, a ≤ 32-token accounting difference in EHPC's favour. (iii)
*The sink exclusion is an addition*: EHPC as published lets the BOS column absorb mass; at sentence level the
sink is not inside any passage sentence, so masking it changes nothing in ranking except through pooling, which
we drop — state this in the write-up rather than claiming fidelity on this point. (iv) *NMI vs EMI*: the paper's
EMI table uses Llama-3.1-8B as compressor and "another model" as answerer (UNVERIFIED which; the protocol cites
LongLLMLingua's GPT-3.5 setup); our arms use the same 1-3B model for both, which is EHPC's NMI setting.

**Transfer recipe for Qwen2.5-1.5B-Instruct (28 layers, 12 query heads, 2 KV heads) and Llama-3.2-3B-Instruct
(28 layers, 24 query heads, 8 KV heads).** Neither is in the paper or the code (`--model` choices are Llama-3.1-8B,
CodeLlama-7B, Mistral-Nemo, Phi-3.5 only), so the heads must be re-discovered:
1. Build 50 probes: filler = concatenated Wikipedia-style paragraphs from the *training* split of HotpotQA
   (not the eval examples) cut to a random length in 400-1,500 tokens; needle = one random declarative sentence
   with a unique answer entity (e.g. "The best thing to do in San Francisco is …"), inserted at depth
   U(0.05, 0.95) on a sentence boundary; question asks for the needle's entity; same prompt scaffold as the
   LongBench template so the observation window has the same shape as at test time.
2. Run each probe once with `output_attentions=True` (eager attention; 28 × 12 × 1500² × 2 B ≈ 1.5 GB for Qwen,
   3 GB for Llama-3.2-3B — fine). For every (layer, head) compute Eq. 2 with the window-averaged rows (N_o = 16),
   with the BOS column excluded before summing needle mass; average over the 50 probes.
3. Layer L* = argmax of the per-layer sum; H* = top-4 heads of L* (paper uses 8 of 32 = 25 %; 4 of 12 = 33 % for
   Qwen, 4 of 24 = 17 % for Llama-3.2 bracket that). Record the mass fraction `top4.sum() / layer.sum()`; also
   report the top-4 heads' needle mass divided by their mass on a random non-needle span of equal length
   (signal-to-noise). Expected L*: middle of the network (paper: 40-53 % depth → layers ≈ 11-15 of 28); do not
   assume "early" (layers 0-5).
4. Fallback if no head stands out (top-4 mass fraction < 2× the uniform expectation 4/heads, or needle SNR < 2):
   use the mean attention over all heads of the **last three layers** with the same window; label the arm
   "EHPC-fallback" and say why.
5. Freeze (L*, H*) per model before touching the evaluation examples; report them in the paper.
6. Attention for scoring at test time: register a forward hook on layer L* only (or recompute `softmax(Q_W K^T /
   √d)` for the 16 window rows from that layer's captured q/k), so memory is O(N_o × N), not O(N²) per layer;
   `output_attentions=True` on 8-10k-token contexts would not fit.

**Anchor numbers.** The only per-dataset HotpotQA/2WikiMQA/MuSiQue numbers in the paper are in Table 6 (NMI:
the compressor is also the answering model; budgets 1,024/2,048 tokens; baselines are KV-cache methods), plus a
copy of the Phi row in Appendix Table 9. The headline compression benchmark (Table 4, EMI, 2,000/3,000 tokens,
LongLLMLingua-style protocol) reports only category aggregates.

| Table 4 (EMI), constraint | Method | SingleDoc | MultiDoc | Avg | Tokens | κ |
|---|---|---|---|---|---|---|
| 2,000 | EHPC | 44.5 | 50.7 | 49.6 | 2,004 | 5x |
| 2,000 | LongLLMLingua | 39.0 | 42.2 | 48.0 | 1,809 | 6x |
| 2,000 | LLMLingua-2 | 29.8 | 33.1 | 39.1 | 1,954 | 5x |
| 3,000 | EHPC | 44.2 | 49.1 | 49.7 | 2,892 | 3x |
| 3,000 | LongLLMLingua | 40.7 | 46.2 | 48.8 | 3,283 | 3x |
| 3,000 | LLMLingua-2 | 35.5 | 38.7 | 42.2 | 3,392 | 3x |
| — | Original prompt | 39.7 | 38.7 | 44.0 | 10,295 | — |

| Table 6 (NMI), model / budget | Method | HotpotQA | 2WikiMQA | MuSiQue | LongBench Avg |
|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct, full KV | All KV | 16.23 | 16.05 | 11.22 | 38.32 |
| Llama-3.1-8B-Instruct, 1,024 | EHPC | 20.64 | 16.97 | 13.99 | 35.23 |
| Llama-3.1-8B-Instruct, 1,024 | SnapKV | 14.81 | 15.73 | 10.69 | 35.25 |
| Llama-3.1-8B-Instruct, 1,024 | GemFilter | 19.12 | 17.01 | 13.01 | 34.50 |
| Llama-3.1-8B-Instruct, 2,048 | EHPC | 19.35 | 16.23 | 13.02 | 37.86 |
| Phi-3.5-mini, full KV | All KV | 21.70 | 25.70 | 11.68 | 34.62 |
| Phi-3.5-mini, 1,024 | EHPC | 44.97 | 32.79 | 20.27 | 39.22 |
| Phi-3.5-mini, 1,024 | SnapKV | 20.72 | 26.02 | 13.74 | 33.68 |
| Phi-3.5-mini, 2,048 | EHPC | 27.06 | 25.22 | 14.05 | 36.46 |

Resolution of the flagged "20.64 vs 39.0" inconsistency: 20.64 is EHPC's HotpotQA F1 in Table 6 (NMI,
Llama-3.1-8B-Instruct answering its own 1,024-token compressed prompt), while 39.0 is LongLLMLingua's
**SingleDoc aggregate** in Table 4 (EMI, GPT-3.5-class answerer). They are different tables, metrics columns,
answering models and budgets; within Table 6 EHPC-1024 beats every listed baseline and the full-KV row on
HotpotQA. The genuinely useful anchor for iteration 2 is Table 6: with a small instruct model answering its own
compressed context, HotpotQA F1 sits in the 15-25 range for Llama-3.1-8B and 20-45 for Phi-3.5 — the absolute
level a 1-3B reader should be expected to land in, and a warning that the 2WikiMQA/MuSiQue F1 of small readers
is low (10-33) so n ≥ 100 per dataset is needed to see 3-point effects.

## 4. Evaluation protocol (what this subfield expects, and what iteration 2 will do)

### 4.1 Metric and normalization (LongBench convention)
LongBench (arXiv 2308.14508) scores HotpotQA, 2WikiMultihopQA and MuSiQue with token-level F1 (`qa_f1_score`); there is no EM column in LongBench itself. `normalize_answer` = lowercase → strip ASCII punctuation → remove articles `a|an|the` → collapse whitespace; F1 is computed on whitespace-split tokens of the normalized strings (bag-of-tokens overlap, `Counter & Counter`), and the per-example score is the MAX over the list of gold answers. Reported numbers are `round(100 * mean, 2)`. EM is the SQuAD companion metric: `normalize_answer(pred) == normalize_answer(gold)` (max over golds). Iteration 2 reports BOTH, F1 as the headline (comparable to the papers), EM as the strict secondary.

LongBench prompt for all three datasets (identical string): "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:" with max_new_tokens = 32 for hotpotqa / 2wikimqa / musique. Iteration 2 uses this exact template and 32 new tokens, greedy decoding, so numbers are in the papers' language.

LongBench facts to keep in mind: 200 examples per dataset; passages are randomly ordered in the multi-doc context ("these passages are randomly ordered to form the multi-document context"); average lengths 9,151 / 4,887 / 11,214 words. A tiny reproduction on 1-3B models with 4k-8k-token inputs will NOT match the published absolute numbers (those use GPT-3.5-Turbo or Llama-3.1-8B as the answering model); the anchor numbers are for ordering and effect-size sanity, not absolute matching.

### 4.2 Compression-ratio definitions across the three papers
- LLMLingua-2 and LongLLMLingua report either a token budget (LongBench: 2,000- and 3,000-token constraints) or "1/τ" / "Nx" (e.g. 2x = keep 1/2, 4x = keep 1/4). Their `compress_prompt` API takes `rate` (kept fraction, LLMLingua-2) or `target_token` (LongLLMLingua) and returns origin_tokens / compressed_tokens / ratio.
- EHPC reports target token counts (LongBench 2,000 / 3,000 constraints) — a budget in tokens, not a fraction.
- Referential Dangling (arXiv 2608.04569) uses r = |C̃|/|C| (kept fraction) and reaches the target by binary search.
Mapping onto iteration 2's kept-fraction ρ ∈ {0.3, 0.5, 0.7}: ρ = 0.5 ≈ "2x"; ρ = 0.3 ≈ "3.3x"; ρ = 0.7 ≈ "1.4x". A 2,000-token target on a ~9k-token HotpotQA input is ≈ 0.22 (≈4.5x); 3,000 tokens ≈ 0.33 (≈3x). So published 2,000/3,000-token rows bracket our ρ = 0.3 arm; our 0.5 and 0.7 arms are milder than anything in the LongBench tables, and only LongLLMLingua's NQ 2x row is a direct analogue of ρ = 0.5.

### 4.3 Is the question inside the budget?
- LongLLMLingua: no. `context` is compressed; `instruction` and `question` are separate arguments, subtracted from
  the budget (`target_token = (ins + que + Σctx) · rate − ins − que`) and kept at their own near-1 rates
  (τ_ins = 0.85, τ_que = 0.9).
- LLMLingua-2: question-agnostic; it compresses whatever string it is given, so evaluation scripts pass the
  passages only and re-attach instruction/question.
- EHPC: partly yes. The budget `k` counts the last `window_size` (16-32) prompt tokens that are appended
  verbatim; question tokens outside the window are scored like any other and may be dropped.
Convention adopted for iteration 2 (matches LongLLMLingua, the LongBench-style evaluations, and Referential
Dangling's r = |C̃|/|C| on the context): **the budget covers passages only; instruction and question are always
kept and never counted.** For EHPC this is a ≤ 32-token departure in EHPC's favour; say so.

### 4.4 Matched-budget sentence-level protocol (normative)
For each example e with passage token count n_e (tokens of the answering model's tokenizer, passages only):
1. B_e = floor(ρ · n_e), ρ ∈ {0.3, 0.5, 0.7}.
2. Segment passages into sentences once (same segmenter for every arm, e.g. spaCy `sentencizer` or nltk punkt; paragraph title lines are their own unit), keep the original order index for every sentence.
3. Every arm produces a score per sentence (its own signal). Greedy fill: iterate sentences in DESCENDING score; add a sentence if its token count fits in the remaining budget; stop at the first sentence that does not fit (strict "first-fail stop" so all arms use the same rule; do not skip-and-continue, which would let short-sentence-favouring arms pack more). Tie-break by original position (earlier first).
4. Emit kept sentences in ORIGINAL order (no reordering in any arm), with paragraph boundaries preserved; prepend the LongBench instruction and append the question unchanged.
5. Log realised kept fraction ρ̂_e = kept_tokens / n_e per arm; assert |ρ̂_arm − ρ̂_arm'| ≤ 0.05 pairwise on the dataset mean, and report the mean ρ̂ per arm in the results table.
6. Answering model: the same 1-3B model, greedy, max_new_tokens = 32, LongBench template; metric = LongBench F1 (max over golds) + EM; optional LLM-judge (OpenRouter) as a secondary correctness check on a fixed subset.
7. Control arms: full passage (ρ = 1), random-sentence selection at the same budget (3 seeds), and question-only (no passages) — the last two bound the useful range.
8. Report per-dataset mean ± bootstrap 95% CI over examples (n ≥ 100 per dataset), paired with the full-passage arm.

Why the "first-fail stop" rule: it makes every arm's realised budget a deterministic function of its own ranking plus the shared sentence lengths; the ±5% check catches any arm that systematically prefers short sentences (a known artefact of length-normalised attention scores).

## 5. Related work — three orthogonal recent methods (calibrated, one paragraph each)

**FrugalPrompt (Raiyan, Ishmam, Al Imran, Moni; arXiv 2510.16439, v5 Sep 2026; code github.com/Starscream-11813/Frugal-ICL).** Signal: per-token attribution scores from GlobEnc and DecompX, computed on a single 110M-parameter BERT encoder for a task-specific scoring function φτ; tokens are ranked and the top ⌈k/100·n⌉ are kept in original order, with inputs longer than the 512-token encoder window scored by sentence-level chunking. Headline: "Across four NLP tasks, a 20% prompt reduction preserves performance for most models with negligible parameter overhead." The four tasks are sentiment classification (IMDb), summarization, CosmosQA, and GSM8K; at 80% retention with Llama-3 8B, classification accuracy moves 0.949 → 0.942 while GSM8K pass@1 falls 0.786 → 0.500, which the authors read as an asymmetry between tasks tolerant of contextual sparsity and those needing exhaustive context. What it is NOT: not question-conditioned (the encoder attributes to a task label, not to a query), not evaluated on long-context multi-hop QA, and the attribution comes from an external encoder rather than the answering model — so it is a training-free classifier-style compressor in the LLMLingua-2 family, not a model-internal salience method. Relevance to iteration 2: an alternative query-agnostic token scorer; the paper's own theory (Corollary 1) bounds loss by the summed saliency of deleted tokens plus a quadratic interaction term — the interaction term is the formal version of the dependency-loss that Referential Dangling measures empirically.

**MIST (Zhao, He, Zheng, Chen; "Every Token Leaves a Ripple in the Stream of Thought", arXiv 2608.31066, 31 Aug 2026).** Signal: model-internal saliency of each chain-of-thought token measured by residual-stream interventions — necessity = drop in answer log-likelihood when that token's residual contribution is removed; sufficiency = gain in answer log-likelihood when only that contribution is patched into a no-chain forward pass; the two rankings are weakly correlated (mean Spearman −0.07, top-30% overlap 0.28 on 100 MATH chains with Qwen2.5-1.5B-Instruct) and are combined into one score. What it is NOT: it compresses generated reasoning traces (to build shorter CoT training data, TokenSkip-style), not input passages; it needs the gold answer at scoring time to compute answer likelihood; and it is token-level, not sentence-level. Relevance: it is the closest "causal" (ablation-based) salience definition to our disagreement-protected idea and uses the same 1.5B model, so it fixes the vocabulary — our attention-based scores are correlational proxies, MIST's are interventional — and its finding that necessity and sufficiency disagree is direct precedent that two salience signals can rank the same tokens differently.

**Referential Dangling (Hu, Li, Fu, Zou, Tang, Li, Cao, Huang; "Relevant but Incomplete", arXiv 2608.04569, Aug 2026; code cslikai.cn/Referential-Dangling).** Signal: not a compressor but a diagnostic plus a repair. Diagnosis: independent unit selection splits dependent evidence pairs — the retained span holds the answer while the deleted span defines the entity needed to read it. Under Beaver at r = |C̃|/|C| = 0.30 the answer path is incomplete in 34.2% (HotpotQA) / 53.5% (2Wiki) / 54.2% (MuSiQue) of bridge examples; across six compressors (Beaver, PartPrompt, Selective-Context, LLMLingua-2, LongLLMLingua, DAC) dangling rates are 32–60% on a shared HotpotQA bridge set (n=184), and only 5.4% of examples dangle under all six while 95% dangle under at least one. Repair: reinserting the omitted supporting paragraph while removing the same number of tokens of non-supporting text raises Qwen3-8B accuracy by 29–34 points at equal budget; a compact classifier that ranks omitted sentences by whether retained text needs them gives +4.7 accuracy on HotpotQA at ratio 0.30 → 0.31. What it is NOT: it does not propose a new relevance score; its metric is accuracy (LLM answer models Qwen3-4B/8B, GPT-5.5, GLM-5.2), not F1. Relevance: it is the strongest recent argument that sentence-level scores must be evaluated at MATCHED budget (they already do "restore one, remove equal tokens") and it defines the kept-fraction ratio the same way iteration 2 does; its dependency classifier is a natural add-on arm, not a competitor.

## 6. Checklist for the iteration-2 experiment executor

1. [ ] Sentence index: one shared segmenter (spaCy `sentencizer` or NLTK punkt) over each LongBench example's
       passages, title lines as their own units, original-order ids stored; every arm ranks *these* units.
2. [ ] Budget per example: `B = ⌊ρ · n_passage_tokens⌋` in the answering model's tokenizer, ρ ∈ {0.3, 0.5, 0.7};
       instruction + question outside the budget for all arms.
3. [ ] LLMLingua-2 arm: official `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`, run on 510-token
       sentence-aligned chunks of the full passage text, word probability = mean over sub-words, sentence score =
       mean word P(preserve). Plus one token-level official run (`rate=0.5`, `drop_consecutive=True`) on 50
       examples to measure the sentence-vs-token gap.
4. [ ] LongLLMLingua-r_k arm: per sentence, mean NLL of `question + " We can get the answer to this question in
       the given documents."` conditioned on `title\nsentence`, plain concatenation, no chat template, rank
       ascending; label the arm "LongLLMLingua-r_k (sentence)".
5. [ ] EHPC arm: 50 needle probes per model → (L*, H*, mass fraction, needle SNR) frozen and logged before
       evaluation; window N_o = 16; sink column excluded; sentence score = mean per-token attention; fallback =
       last-3-layers mean attention if no head stands out; hook on layer L* only at test time.
6. [ ] Greedy fill in descending score, first-fail stop, ties by earlier position; output in original order with
       paragraph breaks; identical for all arms (including the proposed method and random/full/question-only).
7. [ ] Log realised ρ̂ per example and arm; assert dataset-mean |ρ̂_a − ρ̂_b| ≤ 0.05 for all arm pairs; print
       the per-arm mean sentence length as the short-sentence-bias diagnostic.
8. [ ] Answering: LongBench template verbatim, `max_new_tokens = 32`, greedy, first line of the output scored;
       metrics = LongBench `qa_f1_score` (normalize_answer, max over golds, ×100) and EM; bootstrap 95 % CI
       paired against the full-passage arm; n ≥ 100 per dataset (200 = the full LongBench split).
9. [ ] Controls: full passage (ρ = 1), random sentences at the same budget (3 seeds), question only.
10. [ ] Report the anchor rows next to the results: LLMLingua-2 Table 2 (MultiDoc 33.1/38.7), LongLLMLingua Table 2
       (43.2/46.2 and r_k 41.7/42.9), EHPC Table 4 (50.7/49.1) and Table 6 (Llama-3.1-8B HotpotQA 20.64 @1,024),
       with the explicit statement that absolute levels are not comparable (different answering models, 10k-token
       contexts, token-level budgets) and only the ordering question-aware > question-agnostic is expected to
       transfer.

## 7. Confidence and what would change it

High confidence (quoted code and equations): LLMLingua-2 scoring and budget mechanics; LongLLMLingua Eq. 2/3/5,
the restrictive sentence, the ascending sort, question exclusion from the budget; EHPC Eq. 1/2, head lists,
window/kernel values, no sink masking, question tail inside the budget (from the supplemental code). Medium:
the claim that sentence-level LLMLingua-2 is neutral-to-favourable (argued from the papers' evidence, to be
measured by checklist item 3). Medium: EHPC Table 4's answering model (not named in the text). Low / open:
whether evaluator heads exist as cleanly in 1.5B-3B models as in 8B models — the recipe's fallback exists for
this reason. Anything marked UNVERIFIED above (LongLLMLingua's standalone MuSiQue appendix numbers, EHPC's
pilot sample count, the contents of `scripts/*.sh` in the LLMLingua eval) was not confirmed from a primary source.
