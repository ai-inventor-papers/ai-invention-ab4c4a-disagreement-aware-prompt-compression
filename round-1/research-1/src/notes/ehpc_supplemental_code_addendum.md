# EHPC — addendum from the NeurIPS 2025 supplemental code (orchestrator, 2026-09-20)

Source: https://proceedings.neurips.cc/paper_files/paper/2025/file/e356ed5f27885c79c7cb597bb1107c94-Supplemental-Conference.zip
(1,431,123 bytes; downloaded and unzipped; top-level folder `AttentionCompressor/`, forked from GemFilter per its README
"Acknowledgments"; `requirements.txt` pins `transformers==4.43.3`, `flash-attn==2.6.3`).

Files: `needle_probe.py`, `my_utils/my_generation.py`, `my_baseline/GemFilter/gem_filter_utils.py`,
`my_baseline/GemFilter/llama_select_attention.py` (+ mistral/phi3 variants), `eval/LongBench/{pred.py,eval.py,metrics.py,config/*}`,
`eval/needle/{needle_in_haystack.py,utils.py,PaulGrahamEssays/*.txt}`. No CodeLlama/Qwen/Llama-3.2 hijack modules.

## Scoring function (gem_filter_utils.py::standard_dis_index) — exact code
```
attn_weights = torch.matmul(queries, data.transpose(-1, -2)) / math.sqrt(queries.shape[-1])
... causal mask applied only inside the last window_size x window_size block ...
inner_product = torch.nn.functional.softmax(attn_weights, dim=-1)
inner_product = (inner_product[:, :, -window_size:, : -window_size]).mean(dim=-2) # align to snap kv
if self.probe_context is not None:
    a = inner_product[:, :,self.probe_context[0]:self.probe_context[0]+self.probe_context[1]].sum(dim=-1).squeeze()
    self.attention_socres = a
self.all_attention_scores = inner_product
if sum_over_heads:
    if self.config and self.config['heads']:
        indices = self.config['heads']
        inner_product = torch.sum(inner_product[:,indices,:], dim=1, keepdim=True)
    else:
        inner_product = torch.sum(inner_product, dim=1, keepdim=True)
if pool == 'avg_pool':
    inner_product = F.avg_pool1d(inner_product, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
elif pool == 'max_pool':
    inner_product = F.max_pool1d(inner_product, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
top_k = torch.topk(inner_product[-window_size:], k-window_size, dim=-1) # align to snapkv
```
Reading: queries = last `window_size` positions of the prompt (the "observation window"; in LongBench prompts this is the tail
"...Question: {input}\nAnswer:"); keys = all earlier positions. Score per key = mean over the window queries of softmax attention,
summed over the configured evaluator heads, then 1-D average-pooled (stride 1, kernel `kernel_size`), then top-(k − window_size).
`get_layer_context` (my_generation.py) sorts the selected indices (original order) and appends the last `window_size` input ids:
`new_input_ids = torch.cat([select_input_ids, input_ids[-window_size:]])`. The selected TOKEN ids are re-fed to the full model
(token-level, not span-level; readability comes only from pooling).

## Attention sink / BOS: NOT masked
The key slice `[: -window_size]` starts at position 0, so the BOS/sink column is scored like any other token; nothing zeroes
it. With avg-pooling (padding kernel//2) the sink's mass also leaks into its neighbours. Confirms the paper's silence = no masking.

## Question in the budget
`topk` (CLI default 2048 in eval/LongBench/pred.py; paper budgets 1024/2048 for NMI, 2,000/3,000 for EMI) counts the window
tokens: `k − window_size` scored tokens + the `window_size` tail. Question tokens beyond the tail window are scored and may be
dropped. So EHPC's budget INCLUDES the (tail of the) question; LongLLMLingua's does not.

## Head identification (needle_probe.py) — exact recipe
- Haystack: concatenation of Paul Graham essays truncated to `ctx_len * 3.66` chars (`ctx_len` default 16000 tokens).
- Needle: "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n", inserted by
  sentence index at depths [0.1, 0.2, ..., 0.9] (9 probes, one per depth; no random seeds, one haystack).
- Prompt: "\n<|im_start|> This is a very long story book: <book> %s </book>.\n" + "Based on the content of the book, Question:
  What is the best thing to do in San Francisco?\nAnswer:"
- Per (layer, head) score = sum over needle token positions of the window-averaged attention (`attention_socres`), collected
  from every layer (`get_layer_context_arocss` loops layers 0..31), averaged over the 9 depths.
- `score_layers = avg_score.sum(1)`; best layer = argmax; `select_num = 8`; heads = `torch.topk(score_selected_layer, 8)`.
- The script also prints `import_heads = topk(...).values.sum() / score_selected_layer.sum()` — the mass fraction captured by
  the 8 heads; a useful "does a head stand out" diagnostic for the transfer recipe.
- Comment in the script for Phi-3.5: `select_layer_idx = 19  # 19 out of 32` and Mistral-Nemo `19 # 19 out of 40`, versus the
  README/paper's layer 17 for Phi — the probe run truncates the model to `select_layer_idx+1` layers, so this is the probe
  depth, not necessarily the chosen layer.

## Hyperparameters actually wired in eval/LongBench/pred.py
CLI defaults: `--topk 2048 --window 32 --pool avg_pool --kernel 4 --select_layer_idx 14 --heads [24,3,18,7,29,2,9,1]`
(the CodeLlama heads). The README/paper state window 16 / kernel 32 for Llama-3.1-8B and window 4 / kernel 32 for Phi-3.5.
Prompts longer than `model2maxlen` are truncated in the middle before compression; `build_chat` wraps the LongBench prompt for
chat models. Supported `--model` choices include only llama-3.1-8b-instruct, codellama-7b-hf, mistral-nemo-instruct-2407,
phi-3.5-mini-instruct variants (plus legacy LongBench names) — no Llama-3.2 or Qwen2.5.

## LongBench metric in the supplemental
`eval/LongBench/metrics.py` and `eval.py` are the stock LongBench files (qa_f1_score with normalize_answer; max over golds).
