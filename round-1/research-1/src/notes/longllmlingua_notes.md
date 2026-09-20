# LongLLMLingua — Implementation-Grade Notes

Paper: Jiang, Wu, Luo, Li, Lin, Yang, Qiu. "LongLLMLingua: Accelerating and Enhancing LLMs in Long
Context Scenarios via Prompt Compression." arXiv:2310.06839, ACL 2024 (long.91).
Code: github.com/microsoft/LLMLingua, module `llmlingua/prompt_compressor.py` (class
`PromptCompressor`), fetched at `main` branch (2026-09-20 snapshot).

All quotes below are verbatim from the primary sources fetched in this session (arXiv HTML v2
of 2310.06839, and the `main`-branch raw files of the LLMLingua GitHub repo). Locators reference
line offsets as reported by the grep tool against the fetched raw text (character offsets shown
where line numbers aren't meaningful for a converted HTML/markdown page).

---

## (a) Coarse-grained (document-level) score: question-aware contrastive perplexity r_k

**Paper — problem setup (Section 2, Eq. 1)** defines the general prompt-compression objective:

> min_x̃ D_φ(y, ỹ) + λ‖x̃‖₀, (1)

**Paper — Section 4.1, "Question-Aware Coarse-Grained Compression" (before Eq. 2)** states the
general philosophy of the metric:

> "In coarse-grained compression, we aim to figure out a metric r_k to evaluate the importance of
> each document x^doc_k = {x^doc_{k,i}}_{i=1}^{N_k}, where N_k is the number of tokens in x^doc_k.
> We only keep x^doc_k with higher r_k as the intermediate compressed results."

It then explains why raw document perplexity conditioned on the question is inadequate, and
proposes reversing which side is conditioned on which:

> "We propose to use the perplexity of the question x^que conditioned on different contexts
> x^doc_k p(x^que|x^doc_k) to represent the association between them. We also append a
> restrictive statement x^restrict after x^que to strengthen the interconnection of x^que and
> x^doc_k."

The restrictive statement is given in a footnote:

> "Specifically, 'We can get the answer to this question in the given documents'."

**Eq. 2 (exact)**:

> r_k = − (1/N_c) Σ_i^{N_c} log p(x^{que,restrict}_i | x^doc_k),  k ∈ {1,2,⋯,K}   (2)

where "x^{que,restrict}_i is the i-th token in the concatenated sequence of x^que and x^restrict
and N_c in [sic] the number of tokens."

Source: https://arxiv.org/html/2310.06839v2 — Section 4.1 "Question-Aware Coarse-Grained
Compression" (char offset ≈10,562 for Eq. 1/problem formulation; ≈char offset of the
"Question-Aware Coarse-Grained Compression" heading match for the r_k text and Eq. 2).

**Direction/sign discrepancy — flagged, not resolved by the paper text alone.** r_k as defined in
Eq. 2 is a negative-average-log-likelihood (i.e., a per-token cross-entropy / log-perplexity of
the question given the document): the *better* a document explains/predicts the restated
question (higher p), the *smaller* r_k is. The prose quoted above says documents with **higher**
r_k are kept, which taken literally means keeping documents that make the question *less*
predictable — the opposite of the intuitive "keep the most relevant document" reading. This is
either loose prose (describing "importance" generically, before the sign convention of Eq. 2 is
fixed) or a genuine inconsistency between prose and formula in the paper; I cannot resolve it
from the paper text alone, so I am flagging it explicitly rather than silently picking a
direction. UNVERIFIED (paper-prose direction only): whether "higher r_k" is a wording error.

**Code — resolves the direction empirically.** In `llmlingua/prompt_compressor.py`, function
`get_rank_results`, nested function `get_distance_longllmlingua` (raw file
https://raw.githubusercontent.com/microsoft/LLMLingua/main/llmlingua/prompt_compressor.py, char
offset ≈67637):

```
def get_distance_longllmlingua(corpus, query):
    context_ppl = [
        self.get_condition_ppl(
            d,
            query + " We can get the answer to this question in the given documents.",
            condition_in_question,
        )
        - dl * 2 / 250 * 0
        for d, dl in zip(corpus, context_tokens_length)
    ]
    sort_direct = -1 if condition_in_question == "none" else 1
    ys = sorted(enumerate(context_ppl), key=lambda x: sort_direct * x[1])
    return ys
```

The exact restrictive-statement string used in code is `" We can get the answer to this question
in the given documents."` — matching the paper's footnote (leading space, trailing period; the
question is concatenated immediately before it).

For `rank_method="longllmlingua"`, `condition_in_question` defaults to `"after"` (see below), so
`condition_in_question != "none"` and `sort_direct = 1`, i.e. **`sorted(..., key=lambda x:
1 * x[1])` is an ASCENDING sort on `context_ppl`.** Since `context_ppl` is the raw
`CrossEntropyLoss(reduction="none").mean()` value (a positive average negative-log-likelihood,
i.e., the same quantity as r_k in Eq. 2 up to the `-dl*2/250*0` term, which is unconditionally
zeroed by the trailing `* 0` — a no-op length-penalty term left in the code but disabled),
**documents are ranked and kept in order of ascending (i.e., LOWEST-first) conditional
perplexity/r_k.** Lowest conditional perplexity of the question given the document = the document
makes the question most predictable = highest relevance = kept/ranked first when the budget is
tight. This is the code-level ground truth for "the direction," and it is the intuitive
relevance-ranking behavior; it contradicts the literal wording of the paper's "higher r_k" prose
quoted above.

`condition_in_question` default resolution (same file, `compress_prompt`, char offset ≈25515):

```
if rank_method == "longllmlingua":
    if condition_in_question == "none":
        condition_in_question = "after"
elif rank_method == "llmlingua":
    condition_in_question = (
        "none" if "_condition" not in condition_in_question else "none_condition"
    )
```

`get_condition_ppl` (char offset ≈39654/39970), which implements p(question | document) via a
single forward pass with the loss masked to only the question-side tokens:

```
def get_condition_ppl(
    self,
    text: str,
    question: str,
    condition_in_question: str = "none",
    granularity: str = "sentence",
):
    if condition_in_question == "none":
        return self.get_ppl(text, granularity=granularity)
    elif condition_in_question == "before":
        return self.get_ppl(
            question + text,
            granularity=granularity,
            condition_mode="after",
            condition_pos_id=self.get_token_length(question) - 1,
        )
    elif condition_in_question == "after":
        return self.get_ppl(
            text + question,
            granularity=granularity,
            condition_mode="after",
            condition_pos_id=self.get_token_length(text) - 1,
        )
```

`get_ppl`'s `condition_mode="after"` masks the loss to only the tokens after `condition_pos_id`
(i.e., only the question tokens' loss counts, conditioned on the preceding document tokens as
context/KV-cache):

```
if condition_mode == "before":
    loss = loss[:condition_pos_id]
elif condition_mode == "after":
    loss = loss[condition_pos_id:]
res = loss.mean() if granularity == "sentence" else loss
```

Source: https://raw.githubusercontent.com/microsoft/LLMLingua/main/llmlingua/prompt_compressor.py
— functions `get_ppl`, `get_condition_ppl`, `get_rank_results`/`get_distance_longllmlingua`,
`compress_prompt` (condition_in_question defaulting block).

---

## (b) Fine-grained token-level contrastive perplexity

**Eq. 3 (exact)**, Section 4.1 "Question-Aware Fine-Grained Compression":

> s_i = perplexity(x_i | x_<i) − perplexity(x_i | x^que, x_<i).   (3)

Surrounding text:

> "A straightforward solution for the awareness of x^que is to simply concatenate it at the
> beginning of the whole context. However, this will result in low perplexities of relevant
> tokens in the context following the condition of question x^que, further reducing their
> differentiation from other tokens. In this paper, we propose contrastive perplexity, i.e., the
> distribution shift caused by the condition of the question, to represent the association
> between the token and the question."

> "Additionally, we provide the derivation of its mathematical significance in the Appendix A,
> concluding that it is equivalent to conditional pointwise mutual information (Church and Hanks,
> 1989)."

**Appendix A derivation (Eqs. 6–8, exact)**:

> s_i = perplexity(x_i|x_<i) − perplexity(x_i|x^que,x_<i)
>     = q(x_i) log p(x_i|x^que,x_<i) − q(x_i) log p(x_i|x_<i)
>     = q(x_i) log [ p(x_i|x^que,x_<i) / p(x_i|x_<i) ]   (6)

> p(x^que|x_i,x_<i) = [p(x_i|x^que,x_<i) p(x^que)] / p(x_i|x_<i)
>                   = p(x^que) · [p(x_i|x^que,x_<i) / p(x_i|x_<i)]   (7)

> s_i ∝ p(x^que|x_i,x_<i)   (8)

> "The probability distribution p(x^que) of the question and the ground-truth distribution q(x_i)
> of x_i are constants, hence s_i can be considered as the representation of Eq. (7). ... we
> observe that the form of contrastive perplexity is equivalent to conditional pointwise mutual
> information (Church and Hanks, 1989)."

Source: https://arxiv.org/html/2310.06839v2 — Section 4.1 "Question-Aware Fine-Grained
Compression" (Eq. 3, char offset ≈16,488/17,815) and "Appendix A Derivation Of Question-Aware
Fine-Grained Compression" (Eqs. 6–8, near end of document, after the References section).

**Code path.** The `condition_compare` boolean and `condition_flag` control whether this
subtractive (contrastive) token-level scoring is used. Relevant signature and default-flag
resolution in `compress_prompt` (char offset ≈25515):

```
if condition_compare and "_condition" not in condition_in_question:
    condition_in_question += "_condition"
...
condition_flag = "_condition" in condition_in_question
condition_in_question = condition_in_question.replace("_condition", "")
```

and `iterative_compress_prompt(self, context, target_token, iterative_size=200, keep_split=False,
split_token_id=13, start=0, dynamic_ratio=None, condition_compare=False, ...)` (signature located
at char offset ≈44564, in the same file) is the function that performs the iterative token-level
compression loop and receives `condition_flag`/`condition_compare` to decide whether to subtract
the question-conditioned perplexity from the unconditioned perplexity per Eq. 3.
`get_estimate_threshold_base_distribution` (char offset ≈53120) picks the perplexity cutoff for a
target ratio and flips its sort order via a `condition_flag` argument:

```
def get_estimate_threshold_base_distribution(
    self, ppl, ratio: float, condition_flag: bool = False
):
    if ratio == 1.0:
        return float("-inf")
    ppl = ppl[ppl != 10000]
    target_token = max(0, min(len(ppl) - 1, int(len(ppl) * ratio) - 1))
    return (
        ppl.sort(descending=not condition_flag)
        .values[target_token]
        .detach()
        .cpu()
        .item()
    )
```

i.e., when `condition_flag` is True (contrastive/question-aware scoring, Eq. 3's s_i, where a
*higher* value means more question-relevant and should be kept), the sort is **ascending**
(`descending=False`) so the threshold sits above the lowest-scoring (least relevant) tokens to
drop; when False (plain LLMLingua single-perplexity scoring, where low-perplexity tokens are
uninformative and should be dropped), the sort is descending. This is consistent with "keep
higher-scoring tokens" in both regimes once the sign convention of each score is accounted for.

Source: https://raw.githubusercontent.com/microsoft/LLMLingua/main/llmlingua/prompt_compressor.py
— `compress_prompt` (condition_compare/condition_flag setup),
`iterative_compress_prompt` signature, `get_estimate_threshold_base_distribution`.

---

## (c) Dynamic compression ratio, document reordering, subsequence recovery

### Dynamic ratio — paper formula (Eqs. 4–5, exact)

**Reordering (Eq. 4)**, Section 4.2 "How to reduce information loss in the middle?":

> "After the coarse-grained compression, we have obtained a set of documents {x^doc_k}_{k=1}^{K'}
> with their corresponding importance scores {r_k}_{k=1}^{K'} indicating their association with
> the question x^que. Therefore, we reorder documents using their importance scores to better
> leverage LLMs' information perception difference in positions:"

> (x^ins, x^doc_1, ⋯, x^doc_{K'}, x^que) —r_k→ (x^ins, x^doc_{r1}, ⋯, x^doc_{rK'}, x^que)   (4)

**Dynamic ratio (Eq. 5)**, Section 4.3 "How to achieve adaptive granular control during
compression?":

> "The more relevant to the question a document is, the more budget (i.e., lower compression
> ratio) we should allocate to it. Therefore, we bridge coarse-grained compression to fine-grained
> compression and use the importance scores {r_k}_{k=1}^{K'} obtained from coarse-grained
> compression to guide the budget allocation in fine-grained compression. ... we first determine
> the initial budget for the retained documents τ^doc using the budget controller of LLMLingua.
> During fine-grained compression, we follow the iterative token-level compression algorithm in
> LLMLingua but dynamically assign the compression budget τ^doc_k to each document x^doc_k
> according to the ranking index I(r_k) (e.g., 0, 1) of its importance score from the
> coarse-grained compression. In this paper, we employ a linear scheduler for the adaptive
> allocation. Budget of each token x_i can be formulated as:"

> τ_i = τ^doc_k, ∀x_i ∈ x^doc_k,
> τ^doc_k = max( min( (1 − 2·I(r_k)/K') · δτ + τ^doc , 1 ), 0 )   (5)

> "where i and k is the index of token and document, K' denotes the number of documents, and δτ
> is a hyper-parameter that controls the overall budget for dynamic allocation."

Source: https://arxiv.org/html/2310.06839v2 — Sections 4.2 ("How to reduce information loss in
the middle?", Eq. 4) and 4.3 ("How to achieve adaptive granular control during compression?",
Eq. 5).

**Tuned value of δτ used in the paper's experiments** (Appendix B.2 "Other Implementation
Details"):

> "We set the granular control coefficient k to 2. We use the pre-defined compression rates
> τ_ins=0.85 and τ_que=0.9 for instructions and questions. The segment size used in the iterative
> token-level compression is set to 200. The δτ used in dynamic compression ratio is set to 0.3.
> For a fair comparison, we only used reordering in the NaturalQuestions Multi-document QA and
> noted this in Table 1. We use "We can get the answer to this question in the given documents."
> as the guideline sentence in Eq. (3)."

Source: https://arxiv.org/html/2310.06839v2 — Appendix B.2.

### Dynamic ratio — code (`control_context_budget`, exact)

`llmlingua/prompt_compressor.py`, function `control_context_budget` (char offset ≈44564–46869
region for the ranking→budget→dynamic-ratio pipeline):

```
demostrations_sort = self.get_rank_results(
    context, question, rank_method, condition_in_question, context_tokens_length,
)
if target_token < 0:
    target_token = 100
target_token = eval("target_token" + context_budget)
res = []
used = force_context_ids if force_context_ids is not None else []
...
for idx, _ in demostrations_sort:
    if idx >= len(context_tokens_length):
        continue
    target_token -= context_tokens_length[idx]
    if idx not in used:
        used.append(idx)
    if target_token < 0 or (
        force_context_number is not None and len(res) >= force_context_number
    ):
        break
original_used = used
if reorder_context == "original":
    used = sorted(used)
elif reorder_context == "two_stage":
    l, r = [_ for idx, _ in enumerate(used) if idx % 2 == 0], [
        _ for idx, _ in enumerate(used) if idx % 2 == 1
    ]
    used = l + r[::-1]
if dynamic_context_compression_ratio > 0:
    N = len(used)
    dynamic_ratio = [
        i * (abs(dynamic_context_compression_ratio) / (N - 1))
        if N > 1 else 0
        for i in range(-(N - 1), N, 2)
    ][::-1]
    dynamic_ratio_map = {i: j for i, j in zip(original_used, dynamic_ratio)}
    dynamic_ratio = [dynamic_ratio_map[i] for i in used]
else:
    dynamic_ratio = [0.0] * len(used)
res = [context[idx] for idx in used if idx < len(context)]
return res, dynamic_ratio, used
```

Notes:
- `context_budget` (default `"+100"`) is applied via Python `eval` on the string
  `"target_token" + context_budget`, e.g. `eval("target_token+100")`, giving the extra
  document-level budget referenced in the README/DOCUMENT.md call signature.
- `target_token` (the overall token budget) is consumed greedily walking `demostrations_sort`
  (the relevance-ranked order from part (a)); documents are added to `used` until the budget is
  exhausted — i.e. the *budget controller* both selects which documents survive coarse-grained
  compression AND fixes their traversal order for the ramp below.
- `dynamic_ratio` is a **linear ramp** over the N surviving (`used`) documents, built as
  `i * (ratio/(N-1))` for `i` in `range(-(N-1), N, 2)` then **reversed**. For `reorder_context`
  in {"original","two_stage"} the ramp is then re-indexed from `original_used` order
  (relevance rank order) back onto `used` order (the reorder-context order) via
  `dynamic_ratio_map`. Concretely, this assigns the *largest positive* delta to the
  most-relevant document (rank 0 in `original_used`) and the *most negative* delta to the
  least-relevant retained document, matching Eq. 5's `(1 − 2·I(r_k)/K')·δτ` term (at
  I(r_k)=0 → +δτ; at I(r_k)=K'−1 → ≈−δτ).
- This ramp (`dynamic_ratio`) is consumed downstream in `get_dynamic_compression_ratio`
  (char offset ≈39970 region) via `tau = target_token / (sum(context_length)+1)` and
  `get_ratio(base, delta) = max(min(1, base+delta), 0)`, i.e. exactly Eq. 5's
  `max(min(δτ·(...)+τ^doc, 1), 0)` clamp, just written with `base=τ^doc` (here `tau`) and
  `delta=dynamic_ratio[idx]` (here already carrying the `δτ` scaling baked in from
  `control_context_budget`).

Source: https://raw.githubusercontent.com/microsoft/LLMLingua/main/llmlingua/prompt_compressor.py
— `control_context_budget`, `get_dynamic_compression_ratio`.

### Reordering — "lost in the middle" motivation (paper, exact)

> "As demonstrated in Figure 1b, LLM achieves the highest performance when relevant information
> occurs at the beginning and significantly degrades if relevant information is located in the
> middle of long contexts."

and, from the Introduction:

> "(3) LLMs exhibit position bias Kamradt (2023), also known as the 'lost in the middle' issue
> (Liu et al., 2024), suggesting that the placement of key i..." [truncated by fetch; full
> sentence not re-verified beyond this point — mark as UNVERIFIED for the remainder of that
> sentence].

Source: https://arxiv.org/html/2310.06839v2 — Section 4.2 opening and Section 1 (Introduction),
paragraph enumerating the three challenges.

### Subsequence recovery (post-hoc restoration) — Section 4.4, exact

> "During the generation process, LLMs tend to replicate entities found in the prompt, such as
> names, places, and organizations. Compressing these entities at the token level doesn't affect
> the LLMs' understanding of semantic content but can lead to errors in the generated content.
> Therefore, we propose a subsequence recovery method to restore the original content in LLMs'
> responses. This method relies on the subsequence relationship among tokens in the original
> prompt, compressed prompt, and LLMs' response... The overall procedure includes: i) Iterate
> through tokens y_l in LLMs' response and select the longest substring ỹ_key,l = {y_l,
> y_{l+1},...,y_r} that appears in the compressed prompt x̃. ii) Find the maximum common shortest
> subsequence x_{i,j} = {x_i,...,x_j} in the original prompt x, corresponding to the
> representation ỹ_key,l in the original prompt (accelerated using prefix trees or sequence
> automata). iii) Replace the matched tokens ỹ_key,l in LLMs' response with the corresponding
> subsequence x_{i,j} from the original prompt."

Source: https://arxiv.org/html/2310.06839v2 — Section 4.4 "How to improve the integrity of key
information?" plus "Algorithm 1 Token-level Subsquence Recovery Algorithm" immediately following.

### README API call (exact, matches the task prompt's expected call)

`README.md` (main branch), section "To try LongLLMLingua in your scenarios":

```python
from llmlingua import PromptCompressor
llm_lingua = PromptCompressor()
compressed_prompt = llm_lingua.compress_prompt(
    prompt_list,
    question=question,
    rate=0.55,  # Set the special parameter for LongLLMLingua
    condition_in_question="after_condition",
    reorder_context="sort",
    dynamic_context_compression_ratio=0.3,  # or 0.4
    condition_compare=True,
    context_budget="+100",
    rank_method="longllmlingua",
)
```

(DOCUMENT.md's equivalent snippet uses `ratio=0.55` instead of `rate=0.55` — both are accepted
parameter names in different repo versions; `DOCUMENT.md`'s "Function Call" parameter table
below lists the canonical name as `ratio: float = 0.5`, so `rate=` in the top-level README may be
a doc-drift artifact. Flagging as a **minor cross-file inconsistency**, not resolved further.)

`reorder_context="sort"` is *not* the value shown in the DOCUMENT.md long-form parameter
description (`reorder_context: str = "original"` is the documented default), but the DOCUMENT.md
Advanced Usage / LlamaIndex integration example uses `"sort"` as an enabling value with the
comment `# Enables document reordering`. The code's actual accepted values, per
`control_context_budget`, are only checked against `"original"` and `"two_stage"` explicitly
(`if reorder_context == "original": ... elif reorder_context == "two_stage": ...`); no branch for
literal `"sort"` was found in the fetched excerpt of `control_context_budget` — **UNVERIFIED**:
whether `"sort"` maps to one of these two branches, is a third un-excerpted branch elsewhere in
the file, or is legacy/alias naming in the README not present in the current `main` code path
that was fetched. This should be checked directly against the full file (only ~78.8K/~its total
chars were retrieved via targeted greps, not the complete file byte-for-byte) before being relied
upon.

Sources:
https://raw.githubusercontent.com/microsoft/LLMLingua/main/README.md (section "To try
LongLLMLingua in your scenarios");
https://raw.githubusercontent.com/microsoft/LLMLingua/main/DOCUMENT.md (sections "Basic Usage",
"Integration with LlamaIndex", "Detailed of Pramater"/"Function Call");
https://raw.githubusercontent.com/microsoft/LLMLingua/main/llmlingua/prompt_compressor.py
(`control_context_budget`).

---

## (d) Small LM: paper vs. repo default, generality, chat-template independence, segmentation

### Which small model — paper (exact)

Section 5, "Implementation details":

> "In this paper, we use GPT-3.5-Turbo-0613 [footnote: For experiments with original prompts
> exceeding 4k tokens, we utilize GPT-3.5-Turbo-16k-0613.] and LongChat-13B-16k as the target
> LLMs, both accessible via OpenAI [platform.openai.com] and HuggingFace
> [huggingface.co/lmsys/longchat-13b-16k]. To ensure stable and reproducible results, we employ
> greedy decoding and set the temperature to 0 in all experiments. For the small language models
> used for compression, we apply LLaMA-2-7B-Chat [ai.meta.com/llama/], which has been aligned by
> supervised fine-tuning and RLHF. We implement our approach with PyTorch 1.13.1 and HuggingFace
> Transformers. We set up hyperparameters following LLMLingua except for the segment size used in
> iterative token-level compression set to 200 here."

So: **target LLM = GPT-3.5-Turbo-0613** (GPT-3.5-Turbo-16k-0613 for >4k-token prompts) and
**LongChat-13B-16k**; **small compressor LM (M_S) = LLaMA-2-7B-Chat** (an aligned/RLHF'd chat
checkpoint) in the main experiments. A **GPT2-small** variant (specifically **GPT2-dolly**) is
used only in one ablation row ("w/ GPT2-small"):

Appendix B.2, exact:

> "We use the GPT2-dolly [huggingface.co/lgaalves/gpt2-dolly] as the small language model in w/
> GPT2-small ablation experiments."

Source: https://arxiv.org/html/2310.06839v2 — Section 5 "Implementation details"; Appendix B.2
"Other Implementation Details".

### Which small model — repo default (exact)

`llmlingua/prompt_compressor.py`, `PromptCompressor.__init__` docstring and signature (char
offset ≈1417 for docstring, ≈3063 for signature):

> "Args: model_name (str, optional): The name of the language model to be loaded. Default is
> 'NousResearch/Llama-2-7b-hf'."

```
def __init__(
    self,
    model_name: str = "NousResearch/Llama-2-7b-hf",
    device_map: str = "cuda",
    model_config: dict = {},
    open_api_config: dict = {},
    use_llmlingua2: bool = False,
    use_slingua: bool = False,
    llmlingua2_config: dict = {},
):
```

`DOCUMENT.md` "Detailed of Pramater" → "Initialization" section confirms the same default and
names it explicitly as a base (non-chat) checkpoint:

> "model_name (str): Name of the small language model from Huggingface, use
> 'microsoft/llmlingua-2-xlm-roberta-large-meetingbank' or
> 'microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank' for LLMLingua-2. Defaults to
> 'NousResearch/Llama-2-7b-hf'."

**Discrepancy noted**: the repo's *default* constructor argument
(`NousResearch/Llama-2-7b-hf`) is the plain (non-instruction-tuned) Llama-2-7B base model in HF
format, whereas the *paper's* experiments use `LLaMA-2-7B-Chat`. Both are Llama-2-7B checkpoints
but differ in RLHF/chat alignment; the repo default is not identical to what the paper reports
using for its headline numbers.

### Arbitrary AutoModelForCausalLM support (exact)

```
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
```

`load_model` (char offset ≈4586):

```
def load_model(
    self, model_name: str, device_map: str = "cuda", model_config: dict = {}
):
    trust_remote_code = model_config.get("trust_remote_code", True)
    if "trust_remote_code" not in model_config:
        model_config["trust_remote_code"] = trust_remote_code
    config = AutoConfig.from_pretrained(model_name, **model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **model_config)
    if model_config.get("pad_to_left", True):
        tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        config.pad_token_id if config.pad_token_id else tokenizer.eos_token_id
    )
    MODEL_CLASS = (
        AutoModelForTokenClassification
        if any("ForTokenClassification" in ar for ar in config.architectures)
        else AutoModelForCausalLM
    )
    ...
```

This confirms the repo loads **any** Hugging Face `model_name` generically via `AutoConfig` /
`AutoTokenizer` / (`AutoModelForCausalLM` unless the architecture string contains
"ForTokenClassification", in which case `AutoModelForTokenClassification` is used for the
LLMLingua-2 token-classifier compressors) — i.e., the causal-LM small-model slot is not
hardcoded to Llama and should accept any causal-LM architecture `transformers` supports
(quantized GPTQ variants and `microsoft/phi-2` are explicitly documented as supported in
README.md: "Or use the phi-2 model... Or use the quantation [sic] model, like
TheBloke/Llama-2-7b-Chat-GPTQ, only need <8GB GPU memory."). **UNVERIFIED**: no explicit mention
of Qwen or Llama-3 was found in `README.md` or `DOCUMENT.md` (both fetched in full in this
session); GitHub Issues were not searched in this session (out of scope of the fetched primary
sources), so their absence there should not be read as confirmation they are unsupported —
architecturally, any `AutoModelForCausalLM`-compatible checkpoint should work per the code above.

### Chat-template independence (exact)

`get_condition_ppl` (quoted fully in part (a) above) builds inputs by **plain string
concatenation** — `question + text` or `text + question` — with no chat-template, role tags, or
special-token wrapping. `get_ppl` tokenizes with a bare `self.tokenizer(text,
return_tensors="pt")` call (char offset ≈5719 region):

```
def get_ppl(
    self, text: str, granularity: str = "sentence", input_ids=None,
    attention_mask=None, past_key_values=None, return_kv=False, end=None,
    condition_mode: str = "none", condition_pos_id: int = 0,
):
    if input_ids is None:
        tokenized_text = self.tokenizer(text, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].to(self.device)
        attention_mask = tokenized_text["attention_mask"].to(self.device)
    ...
```

No call to `tokenizer.apply_chat_template` was found anywhere in the fetched excerpts of
`prompt_compressor.py` (targeted greps covered the model-loading, `get_ppl`,
`get_condition_ppl`, `get_rank_results`/ranking, `control_context_budget`,
`control_sentence_budget`, and `iterative_compress_prompt` regions — not the full ~79KB file
byte-for-byte). This is consistent with the small compressor LM being used purely as a
perplexity/entropy oracle rather than as a chat-completion model.

### Instruction/question kept intact; context compressed (exact)

`DOCUMENT.md`, "Detailed of Pramater" → "Parameters":

> "context (str or List[str]): Contexts, documents, or demonstrations in the prompt, exhibiting
> low sensitivity to compression. instruction (str): General instruction within the prompt,
> displaying high sensitivity to compression. question (str): General question within the
> prompt, also highly sensitive to compression."

Same idea restated in DOCUMENT.md's "Principles" section:

> "Prompt Sensitivity: Different components of a prompt, like instructions and questions, vary
> in sensitivity to compression. Contexts or documents, for example, are less sensitive. It's
> advisable to separate these components in the prompt for demonstrations, instructions, and
> questions."

Code enforcement — `compress_prompt`, target-token budget computation explicitly *subtracts out*
the instruction and (optionally) question token counts, so the compression-target math is applied
to context only, and instruction/question survive at (near-)full length modulated only by their
own separate fixed rates (`τ_ins=0.85`, `τ_que=0.9`, per Appendix B.2 above):

```
context_tokens_length = [self.get_token_length(c) for c in context]
instruction_tokens_length, question_tokens_length = self.get_token_length(
    instruction
), self.get_token_length(question)
if target_token == -1:
    target_token = (
        (instruction_tokens_length + question_tokens_length + sum(context_tokens_length))
        * rate
        - instruction_tokens_length
        - (question_tokens_length if concate_question else 0)
    )
```

`context_budget` (default `"+100"`) is then added on top of this per-context budget in
`control_context_budget` via `target_token = eval("target_token" + context_budget)` (quoted in
full in part (c) above) — i.e., an *extra* +100-token allowance specifically for the surviving
context documents, beyond the proportional `rate`-derived budget.

### Segmentation into "documents" (`context` argument)

Per `DOCUMENT.md`'s "Function Call" signature: `context: List[str]` — the caller passes contexts
as a Python list of strings, one string per document/demonstration; this is the unit that
coarse-grained ranking (part a) and dynamic-ratio allocation (part c) operate over. "Granular
Division" principle, `DOCUMENT.md`:

> "For multi-document QA and few-shot learning, divide demonstrations and contexts into
> independent granularities. This helps with budget control and document reordering."

Sources for (d):
https://raw.githubusercontent.com/microsoft/LLMLingua/main/llmlingua/prompt_compressor.py
(`__init__` docstring/signature, `load_model`, `get_ppl`, `get_condition_ppl`, `compress_prompt`
target_token block, `control_context_budget`);
https://raw.githubusercontent.com/microsoft/LLMLingua/main/DOCUMENT.md (Principles, Parameters,
Function Call sections);
https://raw.githubusercontent.com/microsoft/LLMLingua/main/README.md (phi-2 / GPTQ quantized
model support);
https://arxiv.org/html/2310.06839v2 (Section 5 "Implementation details"; Appendix B.2).

---

## (e) Published numbers

**Target LLMs used for all main-table numbers**: GPT-3.5-Turbo-0613 (GPT-3.5-Turbo-16k-0613 for
prompts >4k tokens) and LongChat-13B-16k — see quote in part (d). **Compression-ratio notation**:
"2x"/"4x" etc. denote `1/τ`, i.e. original-token-count divided by compressed-token-count (a
literal column header in Table 1/Table 2 is `1/τ`); "Tokens" column reports the resulting
compressed prompt's average token count. This is stated directly in the Table 1 caption (quoted
below) and via the explicit `1/τ` column header — I did not find separate prose spelling out
"2x = half the tokens" beyond this operational (`1/τ`) definition, so treat "2x/4x" as exactly
`1/τ` (ratio of original to compressed tokens), not literally "twice as many tokens removed" in
some other sense.

### Table 1 — NaturalQuestions multi-document QA (20 documents), by ground-truth position

Table 1 caption (exact):

> "Table 1: Performance of different methods with different compression ratios (raw size /
> compressed size = 1/τ) on NaturalQuestions (20 documents) (Liu et al., 2024). Reorder: we
> reorder the documents with relevance metrics of different baselines as our document reordering
> strategy described in Sec. 4.2. In the case of OpenAI, it corresponds to LongContextReorder in
> the LangChain framework (Chase, 2022). For results reported under 1st to 20th, we do not use the
> reordering strategy for all methods."

Columns: accuracy at ground-truth position **1st / 5th / 10th / 15th / 20th**, plus **Reorder**
(with the paper's own reordering strategy applied), separately for **GPT3.5-Turbo** and
**LongChat-13b** target models, plus **Tokens**, **1/τ**, **Latency**, **Speedup**.

**2x constraint** (exact table rows):

| Method | GPT3.5 1st | 5th | 10th | 15th | 20th | Reorder | LongChat 1st | 5th | 10th | 15th | 20th | Reorder | Tokens | 1/τ | Latency | Speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 53.7 | 49.3 | 47.9 | 49.9 | 46.9 | 50.3 | 50.9 | 44.9 | 44.1 | 42.9 | 43.2 | 46.0 | 1,545 | 1.9x | 2.1 | 1.9x |
| Gzip | 64.6 | 63.8 | 60.5 | 58.3 | 57.3 | 64.4 | 61.9 | 55.7 | 52.7 | 50.8 | 50.9 | 59.3 | 1,567 | 1.9x | 2.1 | 1.9x |
| SBERT | 72.5 | 67.9 | 63.3 | 65.0 | 66.2 | 68.7 | 65.8 | 57.5 | 54.9 | 53.4 | 55.7 | 61.4 | 1,549 | 1.9x | 2.2 | 1.9x |
| OpenAI | 73.0 | 65.6 | 66.5 | 65.4 | 65.5 | 69.9 | 65.9 | 57.5 | 56.2 | 54.2 | 55.7 | 61.7 | 1,550 | 1.9x | 4.9 | 0.8x |
| LongLLMLingua r_k | 73.9 | 67.7 | 68.7 | 66.0 | 65.6 | 74.3 | 68.5 | 59.1 | 56.8 | 55.3 | 56.9 | 65.2 | 1,548 | 1.9x | 2.3 | 1.8x |
| Selective-Context | 45.4 | 39.0 | 33.8 | 33.5 | 41.5 | - | 53.2 | 26.3 | 25.4 | 24.2 | 33.3 | - | 1,478 | 2.0x | 7.4 | 0.6x |
| LLMLingua | 39.7 | 39.5 | 40.4 | 37.1 | 42.3 | 41.5 | 38.7 | 37.3 | 35.7 | 34.1 | 37.5 | 37.1 | 1,410 | 2.1x | 2.8 | 1.5x |
| **LongLLMLingua** | **77.2** | **72.9** | **70.8** | **70.5** | **70.6** | **76.2** | **68.7** | **59.4** | **57.3** | **55.9** | **58.4** | **66.1** | 1,429 | 2.1x | 2.9 | 1.4x |

**4x constraint** (exact table rows):

| Method | GPT3.5 1st | 5th | 10th | 15th | 20th | Reorder | LongChat 1st | 5th | 10th | 15th | 20th | Reorder | Tokens | 1/τ | Latency | Speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 40.6 | 38.6 | 38.2 | 37.4 | 36.6 | 36.3 | 39.5 | 37.5 | 36.8 | 36.4 | 35.5 | 37.7 | 798 | 3.7x | 1.5 | 2.7x |
| Gzip | 63.1 | 61.0 | 59.8 | 61.1 | 60.1 | 62.3 | 57.6 | 52.9 | 51.0 | 50.1 | 50.4 | 57.2 | 824 | 3.6x | 1.5 | 2.7x |
| SBERT | 66.9 | 61.1 | 59.0 | 61.2 | 60.3 | 64.4 | 62.6 | 56.6 | 55.1 | 53.9 | 55.0 | 59.1 | 808 | 3.6x | 1.6 | 2.5x |
| OpenAI | 63.8 | 64.6 | 65.4 | 64.1 | 63.7 | 63.7 | 61.2 | 56.0 | 55.1 | 54.4 | 55.0 | 58.8 | 804 | 3.7x | 4.3 | 1.0x |
| LongLLMLingua r_k | 71.1 | 70.7 | 69.3 | 68.7 | 68.5 | 71.5 | 67.8 | 59.4 | 57.7 | 57.7 | 58.6 | 64.0 | 807 | 3.7x | 1.7 | 2.4x |
| Selective-Context | 31.4 | 19.5 | 24.7 | 24.1 | 43.8 | - | 38.2 | 17.2 | 15.9 | 16.0 | 27.3 | - | 791 | 3.7x | 6.8 | 0.6x |
| LLMLingua | 25.5 | 27.5 | 23.5 | 26.5 | 30.0 | 27.0 | 32.1 | 30.8 | 29.9 | 28.9 | 32.4 | 30.5 | 775 | 3.8x | 1.8 | 2.2x |
| **LongLLMLingua** | **75.0** | **71.8** | **71.2** | **71.2** | **74.7** | **75.5** | **68.7** | **60.5** | **59.3** | **58.3** | **61.3** | **66.7** | 748 | 3.9x | 2.1 | 2.0x |
| Original Prompt | 75.7 | 57.3 | 54.1 | 55.4 | 63.1 | - | 68.6 | 57.4 | 55.3 | 52.5 | 55.0 | - | 2,946 | - | 4.1 | - |
| Zero-shot | (blank in source) | | 56.1 | | | | | 35.0 | | | 15 | 196x | 1.1 | 3.7x |

Note on the Zero-shot row: the raw extracted markdown for this row is sparse/misaligned
("Zero-shot |  |  | 56.1 |  |  |  |  | 35.0 |  |  | 15 | 196x | 1.1 | 3.7x") — several cells are
empty in the converted table, likely because the zero-shot condition doesn't vary by ground-truth
position for some columns or the HTML→text conversion dropped values. **UNVERIFIED**: the exact
per-position Zero-shot values beyond "56.1" (GPT3.5, some column) and "35.0" (LongChat, some
column); Tokens=15, 1/τ=196x, Latency=1.1, Speedup=3.7x are legible. Recommend checking the PDF
table directly if per-position zero-shot numbers are needed.

Also note the **Original Prompt row is only given under the 4x-constraint block** in the fetched
text (i.e., it is the shared, uncompressed baseline printed once, at 2,946 tokens); it was not
separately duplicated under the 2x block in the extracted text.

Source: https://arxiv.org/html/2310.06839v2 — Table 1 (Section 5, "Main results" / immediately
preceding "Dataset & evaluation metric").

### Table 2 — LongBench, by task category, at 3,000- and 2,000-token constraints

Table 2 caption (exact): "Table 2: Performance of different methods under different compression
ratios on LongBench (Bai et al., 2023) using GPT-3.5-Turbo in 2,000 tokens constraint." (Note:
this single caption in the source covers the whole table, which in fact contains BOTH the
3,000-token and 2,000-token blocks — the caption text itself only explicitly names the 2,000
constraint; this may be a paper caption that doesn't fully describe the combined table.)

Columns: **SingleDoc, MultiDoc, Summ., FewShot, Synth., Code, AVG, Tokens, 1/τ, Latency,
Speedup.**

**3,000 tokens constraint** (exact):

| Method | SingleDoc | MultiDoc | Summ. | FewShot | Synth. | Code | AVG | Tokens | 1/τ | Latency | Speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 32.3 | 34.3 | 25.3 | 57.9 | 45.1 | 48.9 | 40.6 | 3,417 | 3x | 7.5 | 2.1x |
| SBERT | 35.3 | 37.4 | 26.7 | 63.4 | 51.0 | 34.5 | 41.4 | 3,399 | 3x | 7.7 | 2.0x |
| OpenAI | 34.5 | 38.6 | 26.8 | 63.4 | 49.6 | 37.6 | 41.7 | 3,421 | 3x | 13.3 | 1.2x |
| LongLLMLingua r_k | 37.6 | 42.9 | 26.9 | 68.2 | 49.9 | 53.4 | 46.5 | 3,424 | 3x | 8.2 | 1.9x |
| Selective-Context | 23.3 | 39.2 | 25.0 | 23.8 | 27.5 | 53.1 | 32.0 | 3,328 | 3x | 50.6 | 0.3x |
| LLMLingua | 31.8 | 37.5 | 26.2 | 67.2 | 8.3 | 53.2 | 37.4 | 3,421 | 3x | 9.2 | 1.7x |
| **LongLLMLingua** | **40.7** | **46.2** | **27.2** | **70.6** | **53.0** | **55.2** | **48.8** | 3,283 | 3x | 10.0 | 1.6x |

**2,000 tokens constraint** (exact):

| Method | SingleDoc | MultiDoc | Summ. | FewShot | Synth. | Code | AVG | Tokens | 1/τ | Latency | Speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 30.1 | 29.4 | 21.2 | 19.5 | 12.4 | 29.1 | 23.6 | 1,985 | 5x | 4.6 | 3.4x |
| SBERT | 33.8 | 35.9 | 25.9 | 23.5 | 18.0 | 17.8 | 25.8 | 1,947 | 5x | 4.8 | 3.4x |
| OpenAI | 34.3 | 36.3 | 24.7 | 32.4 | 26.3 | 24.8 | 29.8 | 1,991 | 5x | 10.4 | 1.5x |
| LongLLMLingua r_k | 37.8 | 41.7 | 26.9 | 66.3 | 53.0 | 52.4 | 46.3 | 1,960 | 5x | 4.7 | 3.3x |
| Selective-Context | 16.2 | 34.8 | 24.4 | 15.7 | 8.4 | 49.2 | 24.8 | 1,925 | 5x | 47.1 | 0.3x |
| LLMLingua | 22.4 | 32.1 | 24.5 | 61.2 | 10.4 | 56.8 | 34.6 | 1,950 | 5x | 5.9 | 2.6x |
| **LongLLMLingua** | **39.9** | **43.2** | **27.4** | **69.8** | **53.0** | **56.7** | **48.3** | 1,822 | 6x | 6.1 | 2.6x |
| Original Prompt | 39.7 | 38.7 | 26.5 | 67.0 | 37.8 | 54.2 | 44.0 | 10,295 | - | 15.6 | - |
| Zero-shot | 15.6 | 31.3 | 15.6 | 40.7 | 1.6 | 36.2 | 23.5 | 214 | 48x | 1.6 | 9.8x |

Note: at the 2,000-token constraint, LongLLMLingua's actual `1/τ` is printed as **6x** (not 5x
like the retrieval baselines), because its compressed output (1,822 tokens) is smaller than the
other methods' outputs at the same nominal "2,000 token constraint" target — i.e., the constraint
is a *budget cap*, not an exact output length, and LongLLMLingua ends up more compressed than the
target in this row.

**HotpotQA / 2WikiMultihopQA / MuSiQue individually?** The main-paper Table 2 reports **only the
aggregate "MultiDoc" column** — no per-dataset breakdown for HotpotQA, 2WikiMultihopQA, or
MuSiQue appears in this table. MuSiQue is additionally treated as its **own, separate**
experiment (not as a LongBench sub-task readout) — see Appendix B.1:

> "MuSiQue: The multi-hop question-answer dataset is composed of 39,876, 4,834, and 4,918 problems
> in the training, validation, and testing datasets, respectively... For our experiments, we
> utilized the validation set and evaluation scripts provided by Trivedi et al. (2022) for this
> dataset."

and the table of contents lists a dedicated "C.5 MuSiQue" appendix subsection (title only
confirmed via TOC extraction; its numeric contents were not fetched in this session —
**UNVERIFIED** beyond the TOC entry: "C.5 MuSiQue" exists as a section, but its actual
per-dataset numbers were not retrieved).

Source: https://arxiv.org/html/2310.06839v2 — Table 2 (Section 5) for the main aggregate numbers;
Appendix B.1 "Dataset Details" → "MuSiQue" subsection for the MuSiQue-as-separate-benchmark
framing; document Table-of-Contents listing for "C.5 MuSiQue" (existence only, contents
UNVERIFIED).

### Ablation tables (Tables 3 & 4) — reported for completeness, not requested numerically but corroborate (c)/(a)/(b)

**Table 3** (NaturalQuestions, 2x constraint, GPT-3.5-Turbo), exact:

| Variant | 1st | 5th | 10th | 15th | 20th |
|---|---|---|---|---|---|
| LongLLMLingua | 77.2 | 72.9 | 70.8 | 70.5 | 70.6 |
| − w/o Question-awareness | 42.1 | 40.3 | 39.7 | 40.1 | 40.3 |
| − w/ SBERT | 73.2 | 68.5 | 65.7 | 66.1 | 66.7 |
| − w/ p(x^doc_k\|x^{que,restrict}_i) [swapped conditioning] | 56.0 | 52.6 | 53.4 | 51.6 | 51.1 |
| − w/o restrict | 75.1 | 72.2 | 70.3 | 70.3 | 70.2 |
| − w/o Question-aware Fine-grained | 75.8 | 71.0 | 68.9 | 68.4 | 69.3 |
| − w/o Dynamic Compression Ratio | 74.4 | 70.7 | 68.7 | 67.9 | 68.1 |
| − w/o Subsequence Recovery | 76.7 | 71.7 | 69.4 | 69.3 | 69.7 |
| − w/ Document Reordering [applied elsewhere too?] | 76.2 | 76.2 | 76.2 | 76.2 | 76.2 |
| − w/ GPT2-small | 74.6 | 71.7 | 70.1 | 69.8 | 68.5 |
| LLMLingua | 39.7 | 39.5 | 40.4 | 37.1 | 42.3 |
| − w/ Subsequence Recovery | 43.8 | 44.1 | 43.5 | 43.3 | 44.4 |

**Table 4** (LongBench, 2,000-token constraint, GPT-3.5-Turbo), exact:

| Variant | SingleDoc | MultiDoc | Summ. | FewShot | Synth. | Code | AVG | Tokens | 1/τ |
|---|---|---|---|---|---|---|---|---|---|
| LongLLMLingua | 39.9 | 43.2 | 27.4 | 69.8 | 53.0 | 56.7 | 48.3 | 1,822 | 6x |
| − w/o Question-awareness | 27.1 | 38.7 | 25.4 | 62.0 | 18.0 | 53.3 | 37.4 | 1,945 | 5x |
| − w/ SBERT | 34.0 | 38.7 | 24.1 | 57.9 | 32.5 | 31.1 | 36.4 | 1,790 | 6x |
| − w/ p(x^doc_k\|x^{que,restrict}_i) | 22.5 | 28.9 | 23.2 | 53.0 | 22.5 | 33.3 | 30.6 | 1,794 | 6x |
| − w/o restrict | 37.8 | 39.5 | 26.4 | 64.8 | 52.5 | 55.8 | 46.1 | 1,834 | 6x |
| − w/o Question-aware Fine-grained | 35.7 | 41.1 | 26.4 | 62.9 | 44.5 | 54.8 | 44.2 | 1,807 | 6x |
| − w/o Dynamic Compression Ratio | 36.1 | 40.6 | 26.9 | 67.2 | 48.0 | 55.8 | 45.7 | 1,851 | 6x |
| − w/o Subsequence Recovery | 38.6 | 41.8 | 27.3 | 69.0 | 53.8 | 56.6 | 47.8 | 1,809 | 6x |
| − w/o Document Reordering | 39.0 | 42.2 | 27.4 | 69.3 | 53.8 | 56.6 | 48.0 | 1,809 | 6x |
| − w/ GPT2-small | 35.9 | 39.4 | 25.0 | 60.6 | 42.0 | 55.4 | 43.0 | 1,892 | 5x |

Source: https://arxiv.org/html/2310.06839v2 — Table 3 caption ("Ablation study on
NaturalQuestions with 2x constraint using GPT-3.5-Turbo.") and Table 4 caption ("Ablation on
LongBench (Bai et al., 2023) using GPT-3.5-Turbo in 2,000 tokens constraint."), both in Appendix
B.2/immediately after it.

### Headline claims (Abstract, exact)

> "For instance, in the NaturalQuestions benchmark, LongLLMLingua boosts performance by up to
> 21.4% with around 4x fewer tokens in GPT-3.5-Turbo, leading to substantial cost savings. It
> achieves a 94.0% cost reduction in the LooGLE benchmark. Moreover, when compressing prompts of
> about 10k tokens at ratios of 2x-6x, LongLLMLingua can accelerate end-to-end latency by
> 1.4x-2.6x."

The "21.4%" figure is elaborated in the Main-results prose:

> "LongLLMLingua gains a performance boost of 21.4% on NaturalQuestions with the ground-truth
> document at the 10th position, while the number of tokens input to GPT3.5-Turbo is ∼4x less."

Cross-checking against Table 1: 4x-constraint, GPT3.5-Turbo, 10th position: LongLLMLingua = 71.2
vs. Original Prompt = 54.1 at 10th position → absolute gain = 71.2 − 54.1 = **17.1 points**, and
71.2/54.1 − 1 ≈ **31.6% relative gain** — neither matches "21.4%" exactly by my own arithmetic on
the quoted table cells. **UNVERIFIED / flagged discrepancy**: I cannot reconcile the abstract's
"21.4%" figure against the Table-1 4x/10th-position cells as extracted here without further
digging (possibly the 21.4% refers to a different comparison — e.g., against LongChat-13b's
10th-position original-prompt cell, 55.3, giving 59.3/55.3−1 ≈ 7.2%, which also doesn't match; or
it could be an absolute-point difference computed on a slightly different table version/decimal
than what the HTML conversion preserved). This specific cross-check should be redone directly
against the PDF's Table 1 if the exact 21.4% derivation matters downstream.

Source: https://arxiv.org/html/2310.06839v2 — Abstract; Section 5 "Main results" prose
paragraph (1).

---

## Sources

All URLs actually fetched (via `aii_fast_web_fetch.py fetch` or `grep`) in this session:

1. https://raw.githubusercontent.com/microsoft/LLMLingua/main/README.md
2. https://raw.githubusercontent.com/microsoft/LLMLingua/main/DOCUMENT.md
3. https://api.github.com/repos/microsoft/LLMLingua/contents/ (repo file listing)
4. https://raw.githubusercontent.com/microsoft/LLMLingua/main/llmlingua/prompt_compressor.py
   (multiple targeted greps: function-definition search; `condition_in_question`/restrictive
   statement search; `AutoModelForCausalLM`/`from_pretrained` search; `__init__`/model-name
   search; `condition_mode`/`get_ppl` search; `control_context_budget` search)
5. https://arxiv.org/abs/2310.06839 (abstract page, confirms v2 is the ACL 2024 version)
6. https://arxiv.org/html/2310.06839v2 (multiple targeted greps covering: Introduction/Eq.1
   problem formulation; "Question-Aware Coarse-Grained Compression" + Eq.2; "Question-Aware
   Fine-Grained Compression" + Eq.3; "How to reduce information loss in the middle" + Eq.4;
   "How to achieve adaptive granular control" + Eq.5; "Implementation details" + Table 1;
   "Table 2:" caption region covering Tables 1–3 data; "Other Implementation Details" (Appendix
   B.2) + Table 4 + dataset details (Appendix B.1); "Derivation Of Question-Aware..." (Appendix A,
   Eqs. 6–8))

Not fetched / out of scope for this pass (would be needed to fully resolve the UNVERIFIED items
above): GitHub Issues for Qwen/Llama-3 mentions; the LongLLMLingua PDF directly (only the HTML
rendering was used, which is noted by arXiv itself as "experimental" and visibly drops/garbles
some table cells, e.g. the Table 1 Zero-shot row); Appendix C.5 (MuSiQue) numeric contents;
full byte-for-byte contents of `prompt_compressor.py` beyond the targeted regex windows retrieved
(the `reorder_context="sort"` code path was not located in the excerpts pulled).
