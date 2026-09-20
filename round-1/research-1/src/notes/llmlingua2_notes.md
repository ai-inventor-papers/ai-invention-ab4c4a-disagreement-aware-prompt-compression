# LLMLingua-2 — Implementation-Grade Notes

Paper: Pan, Wu, Jiang, Xia, Luo, Zhang, Lin, Ruhle, Yang, Lin, Zhao, Qiu, Zhang (2024),
"LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression",
arXiv:2403.12968 (Findings of ACL 2024). Code: github.com/microsoft/LLMLingua.
All quotes below are from primary sources (arXiv HTML render of v2, the GitHub repo, and the
two HuggingFace model cards) that were actually fetched in this session. Locators use the
character offset reported by the grep tool against the fetched page text, plus section/table
numbers from the rendered paper where identifiable.

---

## (a) Scoring unit and score

**Scoring unit = word/token via binary token classification.** The paper frames compression as
predicting `{preserve, discard}` for every token/word using a bidirectional Transformer encoder,
not an LLM causal-LM entropy score:

> "We approach prompt compression as a token classification task (i.e., preserve or discard), and
> take the predicted probability of each token being labeled as preserve as the compression
> metric."
(arXiv:2403.12968, Sec. 1 "Contributions", https://arxiv.org/html/2403.12968, offset ~9220)

> "We formulate prompt compression as a binary token classification problem (i.e., preserve or
> discard) to guarantee the faithfulness of the compressed prompt to the original content, and
> meantime ensure the low latency of the compression model itself. For the token classification
> model, we employ a Transformer encoder as the feature extractor to leverage information from
> the bidirectional contexts of each token."
(Sec. 4 "Compressor", https://arxiv.org/html/2403.12968, offset ~22877, immediately after Table 1)

**Formal score definition** (Sec. 4.1 "Architecture", Eq. 5–6):

> "𝒉 = f_θ(𝒙), p(x_i,Θ) = softmax(W h_i + b), where 𝒉={h_i} denotes feature vectors for all
> words, p(x_i,Θ)∈ℝ² denotes the probability distribution of labels {preserve, discard} for the
> i-th word x_i, and Θ={θ,W,b} represent all the trainable parameters."
(https://arxiv.org/html/2403.12968, offset ~26791, Eq. 5–6)

Training loss is per-word cross-entropy averaged over the sequence (Eq. 7):

> "ℒ(Θ) = (1/N) Σ_{i=1}^{N} CrossEntropy(y_i, p(x_i,Θ))."
(Sec. 4.1 "Training", https://arxiv.org/html/2403.12968, offset ~24882 / ~26791)

**Base encoders.** Two encoders are used, giving two released checkpoints:

> "We use xlm-roberta-large (Conneau et al., 2020) and multilingual-BERT (Devlin et al., 2019)
> for the feature encoder f_θ in our compressor, which we refer to as LLMLingua-2 and
> LLMLingua-2-small, respectively."
(Sec. 5 "Implementation Details", https://arxiv.org/html/2403.12968, offset ~29922/34001)

> "We use xlm-roberta-large which has 355M parameters as the feature encoder f_θ in LLMLingua-2.
> ... For LLMLingua-2-small, the feature encoder is the multilingual-BERT which has 110M
> parameters."
(Appendix H "Model Size and Training Details", https://arxiv.org/html/2403.12968, offset ~70851)

**Training data = GPT-4-distilled MeetingBank labels.**

> "We propose a data distillation procedure to derive knowledge from an LLM (GPT-4) to compress
> the prompts without losing crucial information. We introduce an extractive text compression
> dataset, containing pairs of original texts from MeetingBank (Hu et al., 2023) and their
> compressed versions. We publicly release the dataset."
(Sec. 1 "Contributions", https://arxiv.org/html/2403.12968, offset ~9220/10106)

> "Having obtained pairs of original texts and their compressed versions from data distillation
> (Sec. 3.1), the goal of data annotation is to assign a binary label to each token in the
> original texts to determine if it should be preserved or discarded after compression."
(Sec. 3.2 "Data Annotation", https://arxiv.org/html/2403.12968, offset ~17585)

Dataset released as `microsoft/MeetingBank-LLMCompressed` on HF (confirmed from both model
cards, see (c) below) and from the experiments README:

> "We release our collected GPT-4 compression result at [HF](https://huggingface.co/datasets/microsoft/MeetingBank-LLMCompressed) after review."
(github.com/microsoft/LLMLingua, `experiments/llmlingua2/README.md`, raw fetch)

**Compression-metric definition (ratio τ):**

> "Our approach to compressing the original prompt 𝒙={x_i}_{i=1}^{N} with a target compression
> ratio 1/τ involves a three-step process, where τ is defined as the quotient of the number of
> words in the compressed prompt and the number of words in the original prompt 𝒙. First, we
> derive the target number of tokens to be preserved in the compressed prompt 𝒙̃: Ñ = τN. Next, we
> use the token classification model to predict the probability p_i of each word x_i being
> labeled as preserve … Finally, we retain the top Ñ words in the original prompt 𝒙 with the
> highest p_i and maintain their original order to form the compressed prompt 𝒙̃."
(Sec. 4.2 "Compression Strategy", https://arxiv.org/html/2403.12968, offset ~25073/26791)

Footnote on multi-token words (word-vs-token-piece aggregation, matches code's `mean` mode — see (b)):

> "To address tokenization-related challenges that arise when applying our approach across
> various LLMs and SLMs, we preserve the integrity of multi-token words and represent the
> probability of a word by averaging over the predicted probabilities of all subword tokens."
(footnote 2 to Sec. 4.2, https://arxiv.org/html/2403.12968, offset ~25073)

---

## (b) Budget enforcement (from `llmlingua/prompt_compressor.py`, github.com/microsoft/LLMLingua, `main` branch, raw fetch)

**Rank-and-threshold, not a fixed hard rule.** `rate` is converted into a percentile threshold on
word/token preservation probability, and everything at/above the threshold is kept (this
matches the paper's "top Ñ words with highest p_i" but is implemented as a percentile cut,
which is equivalent up to ties):

Inside `__compress(...)` (method on `PromptCompressor`):
```
if self.use_slingua:
    threshold = 0.5  # slingua use fixed threshold 0.5 for binary token classification
else:
    threshold = np.percentile(
        new_token_probs, int(100 * reduce_rate + 1)
    )
keep_words = []
word_labels = []
assert len(words) == len(word_probs)
for word, word_prob in zip(words, word_probs):
    if word_prob > threshold or (
        threshold == 1.0 and word_prob == threshold
    ):
        ...
        keep_words.append(word)
        word_labels.append(1)
    else:
        word_labels.append(0)
```
(`llmlingua/prompt_compressor.py`, function `__compress`, raw fetch of
`https://raw.githubusercontent.com/microsoft/LLMLingua/main/llmlingua/prompt_compressor.py`,
offset ~74609 in the grepped text)

`reduce_rate` here is `1 - rate` (i.e. the fraction to drop), set by the caller:
```
compressed_context, word_list, word_label_list = self.__compress(
    context_chunked, reduce_rate=max(0, 1 - rate), token_to_word=token_to_word, ...
)
```
(`compress_prompt_llmlingua2`, same file, offset ~35439/37851 — this pattern recurs at every
call site of `__compress`, both with and without context-level filtering and with
`target_token`)

Word probability is obtained from per-subword-token probability via `__token_prob_to_word_prob`,
default `mean`:
```
def __token_prob_to_word_prob(self, token_probs, convert_mode="mean"):
    if convert_mode == "mean":
        word_probs = [sum(p) / len(p) for p in token_probs]
    elif convert_mode == "first":
        word_probs = [p[0] for p in token_probs]
    else:
        raise NotImplementedError()
    return word_probs
```
(`llmlingua/prompt_compressor.py`, offset ~74651)

**`force_tokens` / `force_reserve_digit` / `drop_consecutive`.** All three are handled inside
`__merge_token_to_word` (assigning `prob = 1.0` to forced/digit tokens so they always clear the
threshold) and inside `__compress`'s `drop_consecutive` branch (dropping a forced token if it
repeats immediately after being kept once):

```
elif is_begin_of_new_word(token, self.model_name, force_tokens, token_map):
    pure_token = get_pure_token(token, self.model_name)
    prob_no_force = prob
    if pure_token in force_tokens or pure_token in set(token_map.values()):
        prob = 1.0
    token = replace_added_token(token, token_map)
    words.append(token)
    word_probs.append(
        [1.0 if force_reserve_digit and bool(re.search(r"\d", token)) else prob]
    )
```
(`__merge_token_to_word`, `llmlingua/prompt_compressor.py`, offset ~74609)

```
if drop_consecutive:
    threshold = np.percentile(word_probs, int(100 * reduce_rate))
    is_token_between = False
    prev = None
    for i, (word, word_prob) in enumerate(zip(words, word_probs)):
        if word in force_tokens:
            if is_token_between:
                is_token_between = False
            elif not is_token_between and word == prev:
                word_probs[i] = 0.0
            prev = word
        else:
            is_token_between |= word_prob > threshold
```
(`__compress`, `llmlingua/prompt_compressor.py`, offset ~74609, appears just before the
`new_token_probs` / final-threshold block quoted above)

Docstring definitions (from `compress_prompt_llmlingua2`'s docstring, same file, offset ~23391/32659):

> "force_reserve_digit (bool, optional): Whether to forcibly reserve tokens that containing digit
> (0,...,9). Default is False."
> "drop_consecutive (bool, optinal): Whether to drop tokens which are in 'force_tokens' but
> appears consecutively in compressed prompt. Default is False."
> "chunk_end_tokens (List[str], optinal): The early stop tokens for segmenting chunk. Default is
> [".", "\n"]."

**Chunking at 512 tokens for the encoder.** `init_llmlingua2` hard-codes the encoder's max
sequence length:
```
def init_llmlingua2(
    self, max_batch_size: int = 50, max_force_token: int = 100,
):
    seed_everything(42)
    self.max_batch_size = max_batch_size
    self.max_seq_len = 512
    self.max_force_token = max_force_token
```
(`llmlingua/prompt_compressor.py`, offset ~3922)

Chunking function, splitting on `chunk_end_tokens` near the 512-token boundary (leaving 2 tokens
for CLS/SEP):
```
def __chunk_context(self, origin_text, chunk_end_tokens):
    # leave 2 token for CLS and SEP
    max_len = self.max_seq_len - 2
    origin_list = []
    origin_tokens = self.tokenizer.tokenize(origin_text)
    n = len(origin_tokens)
    st = 0
    while st < n:
        if st + max_len > n - 1:
            chunk = self.tokenizer.convert_tokens_to_string(origin_tokens[st:n])
            origin_list.append(chunk)
            break
        else:
            ed = st + max_len
            for j in range(0, ed - st):
                if origin_tokens[ed - j] in chunk_end_tokens:
                    ed = ed - j
                    break
            chunk = self.tokenizer.convert_tokens_to_string(
                origin_tokens[st : ed + 1]
            )
            origin_list.append(chunk)
            st = ed + 1
    return origin_list
```
(`__chunk_context`, `llmlingua/prompt_compressor.py`, offset ~72724)

The classifier's forward pass (`__get_context_prob`) batches these chunks through a
`TokenClfDataset(..., max_len=self.max_seq_len)` and a `DataLoader(..., batch_size=self.max_batch_size)`,
then runs `self.model(input_ids=ids, attention_mask=mask)` and `F.softmax(logits, dim=-1)`,
taking `probs[..., 1]` as `P(preserve)` per token:
```
outputs = self.model(input_ids=ids, attention_mask=mask)
loss, logits = outputs.loss, outputs.logits
probs = F.softmax(logits, dim=-1)
for j in range(ids.shape[0]):
    _probs = probs[j, :, 1]
```
(`__get_context_prob`, `llmlingua/prompt_compressor.py`, offset ~70738)

**What `compress_prompt(rate=..., use_llmlingua2=True, ...)` returns.** `compress_prompt`
dispatches to `compress_prompt_llmlingua2` when `self.use_llmlingua2` is set:
```
if self.use_llmlingua2:
    return self.compress_prompt_llmlingua2(
        context, rate=rate, target_token=target_token, ...
    )
```
(`compress_prompt`, `llmlingua/prompt_compressor.py`, offset ~25049)

`compress_prompt_llmlingua2` returns:
```
res = {
    "compressed_prompt": "\n\n".join(compressed_context),
    "compressed_prompt_list": compressed_context,
    "origin_tokens": n_original_token,
    "compressed_tokens": n_compressed_token,
    "ratio": f"{ratio:.1f}x",
    "rate": f"{1 / ratio * 100:.1f}%",
    "saving": f", Saving ${saving:.1f} in GPT-4.",
}
```
(`compress_prompt_llmlingua2`, `llmlingua/prompt_compressor.py`, offset ~35439/37851; `ratio` and
`saving` are computed just above as
`ratio = 1 if n_compressed_token == 0 else n_original_token / n_compressed_token` and
`saving = (n_original_token - n_compressed_token) * 0.06 / 1000`)

Docstring's own description of `rate`, confirming rate is a *target*, not exact:
> "rate (float, optional): The minimum compression rate target to be achieved. Default is 0.5.
> The actual compression rate generally exceeds the specified target, but there can be
> fluctuations due to differences in tokenizers. If specified, it should be a float greater than
> or equal to 1.0, representing the target compression rate."
(`compress_prompt_llmlingua2` docstring, `llmlingua/prompt_compressor.py`, offset ~30333)

Note: this docstring text ("greater than or equal to 1.0") is internally inconsistent with the
function default (`rate: float = 0.5`, i.e. a *fraction* to keep, not a ratio ≥ 1) and with the
call-site code (`reduce_rate=max(0, 1 - rate)`, which only makes sense for `rate` in [0,1]).
**UNVERIFIED / apparent doc bug**: treat `rate` as "fraction of tokens to keep" per the code, not
per this one docstring sentence.

---

## (c) API and checkpoints

Exact `PromptCompressor(...)` call for LLMLingua-2, from the GitHub README:

> ```python
> from llmlingua import PromptCompressor
> llm_lingua = PromptCompressor(
>     model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
>     use_llmlingua2=True, # Whether to use llmlingua-2
> )
> compressed_prompt = llm_lingua.compress_prompt(prompt, rate=0.33, force_tokens = ['\n', '?'])
> ## Or use LLMLingua-2-small model
> llm_lingua = PromptCompressor(
>     model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
>     use_llmlingua2=True, # Whether to use llmlingua-2
> )
> ```
(`README.md`, github.com/microsoft/LLMLingua, main branch, raw fetch)

The two HF checkpoint IDs (confirmed on both model cards and matching the README):
- `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` (license: mit; base: XLM-RoBERTa-large,
  355M params — "Model size 0.6B params" per the HF card's Safetensors info, F32 tensor type)
- `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` (license: apache-2.0; base:
  BERT multilingual base cased, 110M params — "Model size 0.2B params" per its HF card)

Both cards state:
> "It is a [XLM-RoBERTa (large-sized model)] finetuned to perform token classification for task
> agnostic prompt compression. The probability $p_{preserve}$ of each token $x_i$ is used as the
> metric for compression."
(https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank; the parallel
sentence, substituting "BERT multilingual base model (cased)", appears on
https://huggingface.co/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank)

Note: the xlm-roberta-large card's own page **H1 title text is
"LLMLingua-2-Bert-base-Multilingual-Cased-MeetingBank"** — i.e. the wrong model name is used as
the page heading on the xlm-roberta-large card (an apparent copy-paste artifact on HF's side).
Quoted exactly as rendered:
> "#  LLMLingua-2-Bert-base-Multilingual-Cased-MeetingBank"
(https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank, page H1)
Flagging this as **UNVERIFIED / likely a card metadata bug**, not a claim about the underlying
model, which the rest of that same card correctly describes as XLM-RoBERTa-large.

Both cards' "Usage" sections give the identical runnable snippet (only `model_name` and the
implicit checkpoint differ):
```python
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True
)
...
results = compressor.compress_prompt_llmlingua2(
    original_prompt,
    rate=0.6,
    force_tokens=['\n', '.', '!', '?', ','],
    chunk_end_tokens=['.', '\n'],
    return_word_label=True,
    drop_consecutive=True
)
```
(https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank and
https://huggingface.co/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank, "Usage"
section; `bert-base-multilingual-cased` card has the identical snippet with `model_name` swapped)

**Does the model card report numbers?** No benchmark table on either HF card — both cards only
give the model description, a "Usage" code snippet, and a BibTeX citation block; no LongBench/
MeetingBank numbers are reproduced on the card itself (they only link to the paper and to
`microsoft/MeetingBank-QA-Summary` / `microsoft/MeetingBank-LLMCompressed` datasets). This is
based on the full fetched card text (8.2–8.4k chars each); no numeric results table appears in
either.

`device_map`: no explicit `device_map=...` example was found in the fetched README/model-card
snippets (**UNVERIFIED** for a concrete `device_map` value beyond the constructor's own default;
`PromptCompressor.load_model` in the source takes `device_map: str = "cuda"` as its default
per the source signature at `llmlingua/prompt_compressor.py` offset ~3922, but no README/card
snippet was found explicitly overriding it for the LLMLingua-2 checkpoints).

---

## (d) Published numbers

All numbers below are the paper's own tables, quoted verbatim (pipe-delimited as rendered by the
arXiv HTML converter). Target LLM: **GPT-3.5-Turbo-0613** by default; a **Mistral-7B-v0.1**
target is used only in Table 4/13.

> "Unless specified otherwise, all reported metrics use GPT-3.5-Turbo-0613 as the target LLM for
> downstream tasks, with greedy decoding at a temperature of 0 for enhanced stability across
> experiments."
(Sec. 5 "Implementation Details", https://arxiv.org/html/2403.12968, offset ~29922)

**Compression-ratio notation in every table is `1/τ`**, e.g. "5x" or "3x" (not raw token counts
alone; τ itself, per Sec. 4.2, is compressed/original word-count ratio, so `1/τ` is the
"compress by Nx" reading). Table columns generally give `AVG | Tokens | 1/τ`.

### Table 2 — LongBench + ZeroSCROLLS, 2,000-token constraint

> "Methods | LongBench | ZeroSCROLLS
> ---|---|---
> SingleDoc | MultiDoc | Summ. | FewShot | Synth. | Code | AVG | Tokens | 1/τ | AVG | Tokens | 1/τ
> 2,000-token constraint
> Task(Question)-Aware Compression
> SBERT† | 33.8 | 35.9 | 25.9 | 23.5 | 18.0 | 17.8 | 25.8 | 1,947 | 5x | 20.5 | 1,773 | 6x
> OpenAI† | 34.3 | 36.3 | 24.7 | 32.4 | 26.3 | 24.8 | 29.8 | 1,991 | 5x | 20.6 | 1,784 | 5x
> LongLLMLingua† | 39.0 | 42.2 | 27.4 | 69.3 | 53.8 | 56.6 | 48.0 | 1,809 | 6x | 32.5 | 1,753 | 6x
> Task(Question)-Agnostic Compression
> Selective-Context† | 16.2 | 34.8 | 24.4 | 15.7 | 8.4 | 49.2 | 24.8 | 1,925 | 5x | 19.4 | 1,865 | 5x
> LLMLingua† | 22.4 | 32.1 | 24.5 | 61.2 | 10.4 | 56.8 | 34.6 | 1,950 | 5x | 27.2 | 1,862 | 5x
> LLMLingua-2-small | 29.5 | 32.0 | 24.5 | 64.8 | 22.3 | 56.2 | 38.2 | 1,891 | 5x | 33.3 | 1,862 | 5x
> LLMLingua-2 | 29.8 | 33.1 | 25.3 | 66.4 | 21.3 | 58.9 | 39.1 | 1,954 | 5x | 33.4 | 1,898 | 5x"
(Sec. 5 "Results on Out-of-Domain Benchmarks", Table 2, https://arxiv.org/html/2403.12968,
offset ~26791)

### Table 2 (continued) — 3,000-token constraint, plus shared "Original Prompt" / "Zero-Shot" rows

> "3,000-tokens constraint
> Task(Question)-Aware Compression
> SBERT† | 35.3 | 37.4 | 26.7 | 63.4 | 51.0 | 34.5 | 41.4 | 3,399 | 3x | 24.0 | 3,340 | 3x
> OpenAI† | 34.5 | 38.6 | 26.8 | 63.4 | 49.6 | 37.6 | 41.7 | 3,421 | 3x | 22.4 | 3,362 | 3x
> LongLLMLingua† | 40.7 | 46.2 | 27.2 | 70.6 | 53.0 | 55.2 | 48.8 | 3,283 | 3x | 32.8 | 3,412 | 3x
> Task(Question)-Agnostic Compression
> Selective-Context† | 23.3 | 39.2 | 25.0 | 23.8 | 27.5 | 53.1 | 32.0 | 3,328 | 3x | 20.7 | 3,460 | 3x
> LLMLingua† | 31.8 | 37.5 | 26.2 | 67.2 | 8.3 | 53.2 | 37.4 | 3,421 | 3x | 30.7 | 3,366 | 3x
> LLMLingua-2-small | 35.5 | 38.1 | 26.2 | 67.5 | 23.9 | 60.0 | 41.9 | 3,278 | 3x | 33.4 | 3,089 | 3x
> LLMLingua-2 | 35.5 | 38.7 | 26.3 | 69.6 | 21.4 | 62.8 | 42.4 | 3,392 | 3x | 33.5 | 3,206 | 3x
> Original Prompt | 39.7 | 38.7 | 26.5 | 67.0 | 37.8 | 54.2 | 44.0 | 10,295 | - | 34.7 | 9,788 | -
> Zero-Shot | 15.6 | 31.3 | 15.6 | 40.7 | 1.6 | 36.2 | 23.5 | 214 | 48x | 10.8 | 32 | 306x
> Table 2: Out-of-domain evaluation on general long-context scenarios. †: numbers reported in
> Jiang et al. (2023b)."
(https://arxiv.org/html/2403.12968, offset ~26791/29922; note the "Original Prompt" and
"Zero-Shot" rows are shared across both the 2,000- and 3,000-token sub-tables in the rendered
HTML — they are listed once, after the 3,000-token block)

### Table 3 — GSM8K / BBH (reasoning, in-context learning; not part of LongBench)

> "Methods | GSM8K | BBH
> 1-shot constraint | half-shot constraint | 1-shot constraint | half-shot constraint
> EM | Tokens | 1/τ | EM | Tokens | 1/τ | EM | Tokens | 1/τ | EM | Tokens | 1/τ
> Selective-Context† | 53.98 | 452 | 5x | 52.99 | 218 | 11x | 54.27 | 276 | 3x | 54.02 | 155 | 5x
> LLMLingua† | 79.08 | 446 | 5x | 77.41 | 171 | 14x | 70.11 | 288 | 3x | 61.60 | 171 | 5x
> LLMLingua-2-small | 78.92 | 437 | 5x | 77.48 | 161 | 14x | 69.54 | 263 | 3x | 60.35 | 172 | 5x
> LLMLingua-2 | 79.08 | 457 | 5x | 77.79 | 178 | 14x | 70.02 | 269 | 3x | 61.94 | 176 | 5x
> Full-Shot | 78.85 | 2,366 | - | 78.85 | 2,366 | - | 70.07 | 774 | - | 70.07 | 774 | -
> Zero-Shot | 48.75 | 11 | 215x | 48.75 | 11 | 215x | 32.32 | 16 | 48x | 32.32 | 16 | 48x
> Table 3: Out-of-domain evaluation on reasoning and in-context learning."
(https://arxiv.org/html/2403.12968, offset ~29922)

### Table 4 — Mistral-7B as target LLM (MeetingBank + LongBench-SingleDoc)

> "Methods | MeetingBank | LongBench-SingleDoc
> QA | Summ. | Tokens | 1/τ | 2,000-token cons. | Tokens | 1/τ | 3,000-token cons. | Tokens | 1/τ
> Selective-Context | 58.13 | 26.84 | 1,222 | 2.5x | 22.0 | 2,038 | 7.1x | 26.0 | 3,075 | 4.7x
> LLMLingua | 50.45 | 23.63 | 1,176 | 2.5x | 19.5 | 2,054 | 7.1x | 20.8 | 3,076 | 4.7x
> LLMLingua-2-small | 75.97 | 29.93 | 984 | 3.0x | 25.3 | 1,949 | 7.4x | 27.9 | 2,888 | 5.0x
> LLMLingua-2 | 76.22 | 30.18 | 970 | 3.0x | 26.8 | 1,967 | 7.4x | 27.3 | 2,853 | 5.1x
> Original Prompt | 66.95 | 26.26 | 3,003 | - | 24.5 | 14,511 | - | 24.5 | 14,511 | -
> Table 4: Evaluation with Mistral-7B as the Target LLM on MeetingBank and LongBench single doc
> QA task. We report Rouge1 (Lin, 2004) for summary."
(https://arxiv.org/html/2403.12968, offset ~29922)

### Table 5 — Latency (MeetingBank, V100-32G GPU)

> "1/τ | 1x | 2x | 3x | 5x
> End2End w/o Compression | 14.9
> End2End w/ LLMLingua-2 | - | 9.4 (1.6x) | 7.5 (2.1x) | 5.2 (2.9x)
> Selective-Context | - | 15.9 | 15.6 | 15.5
> LLMLingua | - | 2.9 | 2.1 | 1.5
> LLMLingua-2 | - | 0.5 | 0.4 | 0.4
> Table 5: Latency (s) comparison on MeetingBank."
(Sec. 5 "Latency Evaluation", https://arxiv.org/html/2403.12968, offset ~29922)

Also: "Table 5 shows the latency of different systems on a V100-32G GPU with different
compression ratios. It shows that LLMLingua-2 has a much smaller computation overhead than
other compression methods, and can achieve an end-to-end speedup ranging from 1.6x to 2.9x.
Additionally, our method can reduce GPU memory costs by 8x."
(same location) — and the abstract's headline number: "our model is 3x-6x faster than existing
prompt compression methods, while accelerating the end-to-end latency by 1.6x-2.9x with
compression ratios of 2x-5x." (Abstract, offset ~4434/10106)

### Table 1 — In-domain MeetingBank (QA EM + summary ROUGE/BLEU/BERTScore)

> "Methods | QA | Summary | Length
> EM | BLEU | Rouge1 | Rouge2 | RougeL | BERTScore | Tokens | 1/τ
> Selective-Context | 66.28 | 10.83 | 39.21 | 18.73 | 27.67 | 84.48 | 1,222 | 2.5x
> LLMLingua | 67.52 | 8.94 | 37.98 | 14.08 | 26.58 | 86.42 | 1,176 | 2.5x
> LLMLingua-2-small | 85.82 | 17.41 | 48.33 | 23.07 | 34.36 | 88.77 | 984 | 3.0x
> LLMLingua-2 | 86.92 | 17.37 | 48.64 | 22.96 | 34.24 | 88.27 | 970 | 3.1x
> Original | 87.75 | 22.34 | 47.28 | 26.66 | 35.15 | 88.96 | 3,003 | 1.0x
> Table 1: In-domain evaluation of different methods on MeetingBank."
(Sec. 4, https://arxiv.org/html/2403.12968, offset ~22877)

### MeetingBank + ZeroSCROLLS at fixed 5x ratio with extended-training ablation (Table 6, Appendix)

> "Methods | LongBench | ZeroSCROLLS
> SingleDoc | MultiDoc | Summ. | FewShot | Synth. | Code | AVG | Tokens | 1/τ | AVG | Tokens | 1/τ
> LLMLingua-2-small | 29.5 | 32.0 | 24.5 | 64.8 | 22.3 | 56.2 | 38.2 | 1,891 | 5x | 33.3 | 1,862 | 5x
> LLMLingua-2 | 29.8 | 33.1 | 25.3 | 66.4 | 21.3 | 58.9 | 39.1 | 1,954 | 5x | 33.4 | 1,898 | 5x
> LLMLingua-2‡ | 30.7 | 33.9 | 25.4 | 66.6 | 22.6 | 58.1 | 39.5 | 1,853 | 5x | 33.4 | 1,897 | 5x
> Original Prompt | 39.7 | 38.7 | 26.5 | 67.0 | 37.8 | 54.2 | 44.0 | 10,295 | - | 34.7 | 9,788 | -
> Zero-Shot | 15.6 | 31.3 | 15.6 | 40.7 | 1.6 | 36.2 | 23.5 | 214 | 48x | 10.8 | 32 | 306x
> Table 6: Out-of-domain evaluation on general long-context benchmarks with the 2,000-token
> constraint. LLMLingua-2‡: We expand the constructed text compression dataset using 50k
> examples from TriviaQA-wiki."
(Appendix, "Secondly, we expand the constructed text compression dataset...", offset ~38363)

### GPT-4-compression comparison (Table 9, Appendix O) and Chinese LongBench (Table 10, Appendix J)

> "Methods | QA | Length
> EM | Tokens | 1/τ
> GPT-4 Compression | 84.86 | 1,221 | 2.5x
> LLMLingua-2-small | 85.82 | 984 | 3.0x
> LLMLingua-2 | 86.92 | 970 | 3.1x
> Original | 87.75 | 3,003 | 1.0x
> Table 9: Comparison with GPT-4 compressed prompt on MeetingBank."
(https://arxiv.org/html/2403.12968, offset ~75644)

> "Methods | LongBench-Zh
> SingleDoc | MultiDoc | Summ. | FewShot | Synth. | AVG | Tokens | 1/τ
> Task(Question)-Agnostic Compression
> LLMLingua | 35.2 | 20.4 | 11.8 | 24.3 | 51.4 | 28.6 | 3,060 | 5x
> LLMLingua-2 | 46.7 | 23.0 | 15.3 | 32.8 | 72.6 | 38.1 | 3,023 | 5x
> Original Prompt | 61.2 | 28.7 | 16.0 | 29.2 | 77.5 | 42.5 | 14,940 | -
> Table 10: Out-of-domain evaluation on LongBench Chinese benchmarks."
(Appendix J "Multilingual Generalization Ability", https://arxiv.org/html/2403.12968, offset ~75644)

### NaturalQuestions / LongLLMLingua integration (Table 11, Appendix K)

> "Methods | 1st | 5th | 10th | 15th | 20th | Reorder | Tokens | 1/τ
> 4x constraint
> Question-Aware Compression
> BM25† | 40.6 | 38.6 | 38.2 | 37.4 | 36.6 | 36.3 | 798 | 3.7x
> Gzip† | 63.1 | 61.0 | 59.8 | 61.1 | 60.1 | 62.3 | 824 | 3.6x
> SBERT† | 66.9 | 61.1 | 59.0 | 61.2 | 60.3 | 64.4 | 808 | 3.6x
> OpenAI† | 63.8 | 64.6 | 65.4 | 64.1 | 63.7 | 63.7 | 804 | 3.7x
> LLMLingua-2+ | 74.0 | 70.4 | 67.0 | 66.9 | 65.3 | 71.9 | 739 | 3.9x
> LongLLMLingua† | 75.0 | 71.8 | 71.2 | 71.2 | 74.7 | 75.5 | 748 | 3.9x
> Question-Agnostic Compression
> Selective-Context† | 31.4 | 19.5 | 24.7 | 24.1 | 43.8 | - | 791 | 3.7x
> LLMLingua† | 25.5 | 27.5 | 23.5 | 26.5 | 30.0 | 27.0 | 775 | 3.8x
> LLMLingua-2 | 48.6 | 44.5 | 43.6 | 40.9 | 39.9 | 46.2 | 748 | 3.9x
> Original Prompt | 75.7 | 57.3 | 54.1 | 55.4 | 63.1 | - | 2,946 | -
> Table 11: Performance comparison on NaturalQuestions (20 documents) (Liu et al., 2024).
> LLMLingua-2+ denotes LLMLingua-2 with LongLLMLingua coarse level compression."
(https://arxiv.org/html/2403.12968, offset ~75644; text notes elsewhere: "LLMLingua-2 with
LongLLMLingua coarse-grained compression achieves an average performance gain of 25.3% on
NaturalQuestions ... compared to LLMLingua-2", offset ~73178)

### Sample-wise Dynamic Compression Ratio (Table 12, Appendix L)

> "Methods | LongBench-SingleDoc
> QA Score | Tokens | 1/τ | QA Score | Tokens | 1/τ
> Target Token Constraint | 2,000 Tokens | 3,000 Tokens
> LLMLingua-2 | 29.8 | 1,954 | 7.4x | 35.5 | 3,392 | 4.3x
> Compression Ratio Constraint | 7x | 5x
> LLMLingua-2 FR† | 25.1 | 2,131 | 6.8x | 27.4 | 3,185 | 4.5x
> LLMLingua-2 DCR‡ | 29.5 | 2,125 | 6.8x | 32.2 | 3,164 | 4.5x
> Original Prompt | 39.7 | 14,511 | 1x | 39.7 | 14,511 | 1x
> Table 12: ... FR† assigns each example with the same fixed compression rate. DCR‡ assigns
> dynamic compression rate to different examples within the corpus level constraint."
(Appendix L, https://arxiv.org/html/2403.12968, offset ~75644/76258)

### Mistral-7B on ≤8K-token subset (Table 13, Appendix P)

> "Methods | MeetingBank | LongBench-SingleDoc
> QA | Summ. | Tokens | 1/τ | 2,000-token cons. | Tokens | 1/τ | 3,000-token cons. | Tokens | 1/τ
> Selective-Context | 62.43 | 19.25 | 703 | 2.4x | 29.3 | 1,829 | 2.5x | 34.6 | 2,855 | 1.6x
> LLMLingua | 51.78 | 24.57 | 714 | 2.4x | 29.9 | 1,862 | 2.5x | 30.7 | 3,016 | 1.5x
> LLMLingua-2 | 81.75 | 30.83 | 651 | 2.6x | 35.0 | 1,889 | 2.4x | 36.3 | 2,841 | 1.6x
> Original Prompt | 71.27 | 27.56 | 1,700 | - | 31.4 | 4,595 | - | 31.4 | 4,595 | -
> Table 13: Evaluation with Mistral-7B as the Target LLM ... We discarded samples where the input
> text has more than 8K tokens."
(Appendix P, https://arxiv.org/html/2403.12968, offset ~76258)

### HotpotQA / 2WikiMQA / MuSiQue: individually or only aggregated?

Searched the full rendered paper text (81,707 chars) for `hotpotqa|2wikimqa|musique` (and the
capitalized display forms `HotpotQA|2WikiMQA|MuSiQue`) — **zero matches**:

> "No matches found for pattern: hotpotqa|2wikimqa|musique|HotpotQA|2WikiMQA|MuSiQue"
(regex grep over https://arxiv.org/html/2403.12968, full document)

So **individually they do not appear anywhere in the paper** — only the aggregate "MultiDoc"
(Multi-Document QA) column of Table 2/6/12 is reported. They *do* appear individually only in
the LongBench **evaluation code** (see (e) below), as three of the ~20 `dataset2metric` keys the
repo's `eval_longbench.py` can score, but the paper text/tables never break them out.

---

## (e) Evaluation code

**LongBench scoring reuses LongBench's own metric functions** (`qa_f1_score`,
`normalize_answer`, etc.), vendored into the LLMLingua repo's
`experiments/llmlingua2/evaluation/metrics.py`
(https://raw.githubusercontent.com/microsoft/LLMLingua/main/experiments/llmlingua2/evaluation/metrics.py):

```python
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
...
def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)
```
This is a byte-for-byte match to the standard SQuAD-style/LongBench `qa_f1_score`/
`normalize_answer` implementation; no custom modification was found in this vendored copy.

`experiments/llmlingua2/evaluation/eval_longbench.py`
(https://raw.githubusercontent.com/microsoft/LLMLingua/main/experiments/llmlingua2/evaluation/eval_longbench.py)
imports these directly and maps each of the ~20 LongBench task names to a metric function:

```python
from metrics import (
    classification_score, code_sim_score, count_score, qa_f1_score,
    qa_f1_zh_score, retrieval_score, retrieval_zh_score, rouge_score, rouge_zh_score,
)
dataset2metric = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score, "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score, "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score,
    "musique": qa_f1_score, "dureader": rouge_zh_score, "gov_report": rouge_score,
    "qmsum": rouge_score, "multi_news": rouge_score, "vcsum": rouge_zh_score,
    "trec": classification_score, "triviaqa": qa_f1_score, "samsum": rouge_score,
    "lsht": classification_score, "passage_retrieval_en": retrieval_score,
    "passage_count": count_score, "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
}
```
Note: `hotpotqa`, `2wikimqa`, and `musique` are indeed present here as individual dataset keys
(each scored with plain `qa_f1_score`), confirming the aggregation into "MultiDoc" happens only
at the paper's table-reporting stage, not in the underlying per-example scoring code.

Scoring/aggregation (`scorer`, then averaged into a per-task score and a final `avg`):
```python
def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in [... "hotpotqa", "2wikimqa", "musique", ...]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)
...
score_list = [s["score"] for s in scores.values()]
scores["avg"] = sum(score_list) / len(score_list)
```

**Ratio definition in the eval/compression pipeline.** `experiments/llmlingua2/evaluation/compress.py`
(https://raw.githubusercontent.com/microsoft/LLMLingua/main/experiments/llmlingua2/evaluation/compress.py)
is a thin CLI wrapper that instantiates `PromptCompressor(model_name=..., use_llmlingua2=True)`
and calls `compress_prompt_llmlingua2(origin, rate=args.compression_rate, target_token=args.target_token, ...)`,
i.e. the eval scripts use the exact same `rate`/`target_token` knobs and the same
`ratio`/`rate` string fields documented in (b) — there is no separate ratio computation in the
eval code; it just reads `comp_dict["compressed_prompt"]` and writes it back into the dataset
JSON under `--save_key` (default `compressed_prompt`). Default CLI args:
```
--compressor default="llmcomp"
--model_name default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
--compression_rate type=float default=0.5
--target_token type=int default=-1
--use_token_level_filter default=True
--use_context_level_filter default=False
--force_tokens default=None (comma-separated)
--drop_consecutive default=True
--force_reserve_digit default=False
```
(`compress.py`, argparse block, raw fetch)

`experiments/llmlingua2/evaluation/utils.py` (`load_model_and_tokenizer`, `query_llm`) shows the
target-LLM side is queried via the (legacy) `openai==0.28` Azure OpenAI Completion/ChatCompletion
API with `tiktoken.encoding_for_model("gpt-4")` as the tokenizer for truncating prompts to
`n_max_token` (default 8100 in `eval_longbench.py`) minus the answer budget — this is the origin
of the "Tokens" column figures (`token_ids = tokenizer.encode(prompt)`; truncated by taking the
first and last `half` tokens if over budget) in `eval_longbench.py`.

**Evaluation directory listing** (confirms exact filenames), from the GitHub API
(https://api.github.com/repos/microsoft/LLMLingua/contents/experiments/llmlingua2/evaluation):
`compress.py`, `eval_bbh.py`, `eval_gsm8k.py`, `eval_longbench.py`, `eval_meetingbank_qa.py`,
`eval_meetingbank_summary.py`, `eval_zero_scrolls.py`, `metrics.py`, `utils.py`, plus a `scripts/`
subdirectory (holding `compress.sh` / `evaluate.sh` per the experiments README, not individually
fetched in this session — **UNVERIFIED** contents of `scripts/compress.sh` and
`scripts/evaluate.sh` themselves, only their existence and role as described in the
experiments README).

---

## Question-aware vs. question-agnostic — confirmed

LLMLingua-2 is explicitly **question-agnostic** (unlike LongLLMLingua, its sibling method):

> "While LLMLingua-2 is designed for question-agnostic compression, it can also be integrated
> with LongLLMLingua to preserve more key information relevant to the question in these
> scenarios."
(Appendix K "Integration with LongLLMLingua", https://arxiv.org/html/2403.12968, offset ~72438)

And from the introduction, contrasting task-aware (question-aware) vs. task-agnostic framing:

> "For example, LongLLMLingua (Jiang et al., 2023b) applies a question-aware coarse-to-fine
> compression approach to estimate the information entropy of the tokens and adapts the
> estimation according to the question." [describing the *baseline*, not LLMLingua-2 itself]
(Sec. 2 "Related Works", https://arxiv.org/html/2403.12968, offset ~10442)

> "This paper focuses on task-agnostic prompt compression for better generalizability and
> efficiency."
(Abstract, https://arxiv.org/html/2403.12968, offset ~3330)

And the paper's own explanation of *why* LLMLingua-2 underperforms LongLLMLingua on LongBench
(Table 2), directly attributing the gap to the question information LLMLingua-2 does not use:

> "While our approach has shown promising results, it falls short when compared to other
> task-aware compression methods like LongLLMlingua (Jiang et al., 2023a) on Longbench. We
> attribute this performance gap to the additional information that they leverage from the
> question."
(Sec. 5 "Results on Out-of-Domain Benchmarks", https://arxiv.org/html/2403.12968, offset ~34001)

## Limitations statement (paper's own "Limitations" section, verbatim)

> "Our text compression dataset was constructed using only training examples from MeetingBank, a
> dataset of summarization over meeting transcripts. This raises concerns about the
> generalization ability of our compressor. Here we discuss this question from two perspectives.
>
> Firstly, we have conducted extensive out-of-domain evaluation on four benchmarks in the paper,
> including LongBench (Bai et al., 2023), ZeroSCROLLS (Shaham et al., 2023), GSM8K (Cobbe et al.,
> 2021), and Big Bench Hard (BBH) (bench authors, 2023), which cover multiple tasks from document
> QA to math problems and in-context learning. The experimental results show that even our
> LLMLingua-2-small model that is of BERT-base size achieves superior performance than the two
> LLaMA-2-7B based baselines Selective-Context (Li et al., 2023) and LLMLingua (Jiang et al.,
> 2023a). This demonstrates that our learned prompt compression model has good generalization
> ability to data from different domains.
>
> Secondly, we expand the constructed text compression dataset using 50k examples from
> TriviaQA-wiki. Then train an LLMLingua-2 compressor with the expanded dataset to see whether
> there would be further performance gain. Table 6 shows the results under the 2,000-token
> constraint. We can see that training the compressor with more data does bring further
> performance gain (LLMLingua-2‡). However, the improvement seems not that significant. We
> conjecture that this is because although the semantics of texts from different domains may
> vary a lot, their redundancy pattern might be similar. Such pattern or knowledge may be learned
> during in-domain training, and then act as an anchor that can transfer across different
> domains. We leave this for future work."
(Section "Limitations", after Sec. 6 "Conclusion", https://arxiv.org/html/2403.12968, offset ~38363)

---

## Things flagged as UNVERIFIED

1. **`rate` docstring vs. code semantics mismatch** in `compress_prompt_llmlingua2`'s own
   docstring ("a float greater than or equal to 1.0, representing the target compression rate")
   contradicts the function default (`rate: float = 0.5`) and the call-site arithmetic
   (`reduce_rate=max(0, 1 - rate)`). The code's actual behavior (rate = fraction of tokens to
   *keep*, in [0,1]) is what's used throughout; the docstring sentence appears to be a
   copy-paste error from `compress_prompt`'s docstring for the original LLMLingua rate
   convention. Flagged, not resolved against a second primary source (e.g. a changelog or issue).
2. **`device_map` for LLMLingua-2 specifically**: no README/model-card snippet was found that
   passes an explicit `device_map=...` for the two LLMLingua-2 checkpoints (only the source
   default `device_map: str = "cuda"` in `load_model`'s signature was seen).
3. **`experiments/llmlingua2/evaluation/scripts/compress.sh` and `scripts/evaluate.sh`** — listed
   in the GitHub API directory listing and referenced by name in the experiments README, but
   their contents were not fetched in this session.
4. Direct fetch of `https://arxiv.org/abs/2403.12968` (the plain abstract page) **timed out**
   after 90s and returned no content; the abstract text was instead obtained from the full
   `https://arxiv.org/html/2403.12968` render, which includes the abstract verbatim, so this is
   not a substantive gap, just a note that the `/abs/` URL specifically was unreachable in this
   session.
5. The apparent mislabeled H1 heading on the xlm-roberta-large HF model card (see (c)) is
   reported as observed, but not independently cross-checked against the HF page's HTML
   `<title>` tag or a cached older revision — flagged as an artifact of the live page at fetch
   time, 2026-09-20.

---

## Sources (every URL actually fetched in this session)

- https://arxiv.org/html/2403.12968 (full paper render, v2; primary source for sections (a),
  (d), question-agnostic confirmation, and Limitations)
- https://raw.githubusercontent.com/microsoft/LLMLingua/main/README.md
- https://raw.githubusercontent.com/microsoft/LLMLingua/main/llmlingua/prompt_compressor.py
- https://raw.githubusercontent.com/microsoft/LLMLingua/main/experiments/llmlingua2/README.md
- https://api.github.com/repos/microsoft/LLMLingua/contents/experiments/llmlingua2/evaluation
- https://raw.githubusercontent.com/microsoft/LLMLingua/main/experiments/llmlingua2/evaluation/compress.py
- https://raw.githubusercontent.com/microsoft/LLMLingua/main/experiments/llmlingua2/evaluation/eval_longbench.py
- https://raw.githubusercontent.com/microsoft/LLMLingua/main/experiments/llmlingua2/evaluation/metrics.py
- https://raw.githubusercontent.com/microsoft/LLMLingua/main/experiments/llmlingua2/evaluation/utils.py
- https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank
- https://huggingface.co/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank

**Attempted but unreachable:** https://arxiv.org/abs/2403.12968 (fetch timed out at 90s; not
needed since https://arxiv.org/html/2403.12968 covers the same content in full).
