# EHPC (Evaluator Heads Prompt Compression) — Implementation-Grade Notes

Paper: **Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference**
Authors: Weizhi Fei, Xueyan Niu, Guoqing Xie, Yingqing Liu, Bo Bai, Wei Han (Tsinghua Univ. / Huawei Theory Lab)
arXiv: 2501.12959 (v1: 22 Jan 2025, v2: 5 Feb 2025 — same title in both versions and in the NeurIPS camera-ready; no v3 exists as of this check). NeurIPS 2025 (39th Conference), OpenReview id `yOs12gdsaL`, DOI `10.52202/085713-5171`.

Primary sources actually fetched and quoted below: the **NeurIPS 2025 camera-ready PDF** (`proceedings.neurips.cc/.../Paper-Conference.pdf`) — this is verified identical in title/author list/abstract to the arXiv v2 HTML and to the `openreview.net/pdf?id=yOs12gdsaL` search snippet — plus the arXiv abstract page, the arXiv v2 HTML, the NeurIPS abstract/proceedings page, GitHub's search API, Hugging Face Papers, and PapersWithCode. The OpenReview `forum` page itself returned a bot/CAPTCHA wall ("Verifying your browser… Complete the check below to continue") and could not be read directly; this is noted as a blocker below, but its content (reviews) was not needed for any of the (a)-(e) items since the camera-ready PDF and its own NeurIPS Paper Checklist cover code-availability and all technical claims.

---

## (a) Code availability

**No GitHub URL appears anywhere in the paper's own text** (Introduction, Conclusion, References were all fetched/grepped for `github|source code|release the code|will release` — the only GitHub hit in the whole PDF is a citation to a third party's benchmark repo, not the authors' code):

> "Greg Kamradt. Needle In A Haystack - Pressure Testing LLMs. https://github.com/gkamradt/LLMTest_NeedleInAHaystack, 2024."
— References section, NeurIPS PDF.

The paper's own **NeurIPS Paper Checklist** (appendix of the camera-ready PDF) gives two different, slightly inconsistent, answers about code:

> "4. Experimental result reproducibility ... Answer: [Yes] Justification: All the information needed to reproduce the numerical experiments is given in Sections 3 and 4. In Section 5, we outline details for implementation and the baselines. Additionally, we plan to make our code open source at the time of publication to facilitate reproducibility."
— NeurIPS PDF, Paper Checklist item 4 (checklist locator, offset ~59746 chars into the extracted text).

> "5. Open access to data and code ... Answer: [Yes] Justification: Yes, the code is provided in the supplementary material with instructions to reproduce the experiments."
— NeurIPS PDF, Paper Checklist item 5 (offset ~62403).

Item 4 phrases this as a future plan ("at the time of publication"); item 5 says it is already in the supplementary material. Consistent with item 5, the **NeurIPS proceedings abstract page does list a supplemental-material download**:

> Links present on the page: `[Paper](.../e356ed5f27885c79c7cb597bb1107c94-Paper-Conference.pdf)` `[Supplemental](.../e356ed5f27885c79c7cb597bb1107c94-Supplemental-Conference.zip)`
— https://proceedings.neurips.cc/paper_files/paper/2025/hash/e356ed5f27885c79c7cb597bb1107c94-Abstract-Conference.html

I confirmed this zip file actually exists and is downloadable (`curl -I`, 2026-09-20): `HTTP/1.1 200 OK`, `Content-Length: 1431123`, `Content-Type: application/zip`, `Content-Disposition: ... filename=NeurIPS-2025-efficient-prompt-compression-with-evaluator-heads-for-long-context-transformer-inference-Supplemental-Conference.zip`. **I did not download/unzip or inspect its contents** — that was out of scope for this task, so I cannot verify what is actually inside it beyond the filename/size. Direct URL: `https://proceedings.neurips.cc/paper_files/paper/2025/file/e356ed5f27885c79c7cb597bb1107c94-Supplemental-Conference.zip`.

**No official public GitHub repository was found.** Queries run (all on 2026-09-20):
- `aii_fast_web_search.py --query "EHPC evaluator heads prompt compression github"` (general/exa)
- `aii_fast_web_search.py --query "EHPC evaluator heads prompt compression code availability github official implementation"` (general/exa)
- `aii_fast_web_search.py --query "site:github.com \"evaluator heads\" EHPC Fei Niu prompt compression official code"` (general/exa)
- `aii_fast_web_search.py --query "openreview yOs12gdsaL EHPC code supplementary"` (general/exa)
- `aii_fast_web_search.py --query "\"Evaluator Heads\" prompt compression EHPC NeurIPS 2025 Fei" --mode scholarly` (openalex) — returned only an unrelated 2026 paper, 0 citations
- GitHub Search API directly: `https://api.github.com/search/repositories?q=EHPC+prompt+compression` → `"total_count":1`
- GitHub Search API directly: `https://api.github.com/search/repositories?q=evaluator+heads+prompt+compression` → `"total_count":1` (same repo)
- HuggingFace Papers page (`huggingface.co/papers/2501.12959`): "Models citing this paper 0" / "Datasets citing this paper 0" / "Spaces citing this paper 0" — no linked code.
- PapersWithCode page (`paperswithcode.com/paper/efficient-prompt-compression-with-evaluator`): no code-implementation links or benchmark table present, only the abstract mirror.

Both GitHub API searches return the **same single, unofficial, third-party repo**:

> `comsa33/ehpc-research` — description: "\"Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference\" 논문의 핵심 아이디어를 구현한 순수 Python 라이브러리입니다." [translation: "A pure-Python library implementing the core idea of the paper..."] — Stars: 1, Forks: 1, Watchers: 1.
— https://github.com/comsa33/ehpc-research (fetched HTML + GitHub API JSON)

**Conclusion for (a):** No official author-released public GitHub implementation found as of 2026-09-20. The only "official" code artifact located is the NeurIPS Supplemental-Conference.zip referenced above (existence confirmed via HTTP HEAD, contents not inspected). One unofficial, low-adoption (1 star) third-party reimplementation exists at github.com/comsa33/ehpc-research.

---

## (b) Evaluator-head identification

**Synthetic probe used to find evaluator heads:** the paper reuses the public "Needle-in-a-Haystack" (NIAH) benchmark (Kamradt 2024), not a bespoke dataset. It gives no explicit sample count, haystack-length range, or number of needles for the *identification* pilot experiment itself in the main text (the sample-count/length details given in Appendix F, "Speedup ratios," are for a *separate* timing experiment, not the head-identification pilot — see caveat below).

> "As a practical example, we used the "Needle-in-a-Haystack" benchmark Kamradt [2024], a well-established long-context retrieval benchmark, to demonstrate our pilot experiments (as illustrated in Figure 1). Let x be the synthesized long context and let e represent the "needle" (evidence) inserted at a specific position for f to identify."
— NeurIPS PDF, Section 3.4 "Pilot experiments for detecting evaluator heads."

Figure 1's caption (the illustrative example) describes the needle/haystack setup:

> "[BOS] … The best thing … is eat … on a sunny day … When cigarettes first appeared, … my parents smoked … "Needle": should be retained "Haystack": can be discarded / Instruction: Please answer the question based on the content of the book. Question: What is the best thing to do in San Francisco?"
— NeurIPS PDF, Figure 1 caption / inline figure text, Section 1.

**UNVERIFIED:** exact number of NIAH samples/haystack-length range used specifically for the *head-detection* pilot experiment (as opposed to the Appendix-F timing benchmark, which explicitly states "sequence lengths ranging from 8k to 32k, running five examples for each length" — that number belongs to the *speedup* experiment in Appendix F, not to head identification in Sec. 3.4, and I did not find the pilot experiment's own sample count/haystack-length stated anywhere in the fetched text).

**Per-head identification score.** Contrary to the task brief's assumption that this is "Eq. 1," in the actual paper **Eq. (1) is the compression *utility* score** (covered under (c) below), and the accumulated-evidence identification score used to *find* evaluator heads is **Eq. (2)**:

> "We extract the last row of the attention matrix, a_h^l = A_h^l[N, :] ∈ R^N, to represent the scores for the importance of each token, as these scores directly influence the computation of the final hidden state. To assess whether the heads are focusing on relevant information, we compute the accumulated score of the evidence as
> â_h^l = Σ_{j∈I_e} a_h^l[j].  (2)"
— NeurIPS PDF, Section 3.4.

Layer/head selection rule (unnumbered display equation immediately following Eq. 2, quoted exactly):

> "Finally, we selected the layer with the highest score and identified the top-k heads from this layer as the evaluator heads, i.e., C_f = arg max_{|Λ|≤k} ‖e_Λ S‖_F  s.t. (i, j) ∉ λ, i = max_{1≤l≤L}(S · 1_{L×1})_l, ∀j, where ‖·‖_F is the Frobenius norm, and e_Λ = (e_ij) is the incidence matrix such that e_ij = 0 if (i, j) ∉ Λ and e_ij = 1 if (i, j) ∈ Λ."
— NeurIPS PDF, Section 3.4 (unnumbered display equation, end of section).

This confirms: **a single best layer is chosen** (the layer maximizing the row-sum of the evidence-score matrix S ∈ R^{H×L}), and **top-k heads within that one layer** are then kept, with **k = 8** in every reported experiment:

> "The results over two models are presented in Table 2. We selected 8 heads with the highest scores as evaluator heads from the layer with the highest cumulative score."
— NeurIPS PDF, Section 4 "Generalizability."

**Layer/head indices reported for each model** (Appendix D, "Detected evaluator heads" — all three models have 32 layers, 32 heads/layer):

> "We conducted pilot experiments using the "Needle-in-a-Haystack" benchmark across three popular LLMs: Llama-3.1-8B-Instruct, CodeLlama-7B, and Phi-3.5-mini-3.8B-Instruct.
> • For Llama-3.1-8B-Instruct, which has 32 layers and 32 heads, the selected layer is 13, and the chosen heads are [18, 13, 21, 8, 11, 1, 4, 3].
> • For CodeLlama-7B, also with 32 layers and 32 heads, the selected layer is 14, and the selected heads are [24, 3, 18, 7, 29, 2, 9, 1].
> • Finally, for Phi-3.5-mini-3.8B-Instruct, which features 32 layers and 32 heads, the selected layer is 17, and the chosen heads are [7, 17, 30, 2, 6, 16, 25, 18]."
— NeurIPS PDF, Appendix D.

So the selected layer is **layer 13/32 (Llama-3.1-8B-Instruct), 14/32 (CodeLlama-7B), 17/32 (Phi-3.5-mini-3.8B-Instruct)** — i.e., roughly the lower-middle third of the network (≈40-53% depth), which the paper itself characterizes as "early":

> "The distribution of these evaluator heads is sparse and predominantly concentrated in some certain layers. This provides empirical support to find evaluator heads in one early and significant layer."
— NeurIPS PDF, Section 4 "Existence."

**Exact list of models used anywhere in the paper** (collected across the pilot experiment, generalizability/robustness studies, the main compression benchmark, and the acceleration benchmark):
- Llama-3.1-8B-Instruct (pilot/head-detection; also the compressor model for the main LongBench/ZeroSCROLLS benchmark, referred to there simply as "Llama-3.1-8B")
- CodeLlama-7B (pilot/head-detection only)
- Phi-3.5-mini-3.8B-Instruct (pilot/head-detection; also "Phi 3.5 Mini 3.8B Instruct" / "Phi-3.8B" in the acceleration benchmark, Table 6 — the paper uses "Phi-3.8B" and "Phi 3.5 Mini 3.8B Instruct" interchangeably for what appears to be the same model)
- GPT-3.5-Turbo / ChatGPT-3.5-Turbo — mentioned as the kind of "black-box LLM" prompt compression targets generally, and used explicitly as the **latency baseline** in Table 5 ("ChatGPT-3.5-Turbo (all tokens)"), but I could **not find an explicit sentence stating GPT-3.5-Turbo is the model that *answers* the compressed prompts in the Table 4 accuracy benchmark** — see UNVERIFIED note under (e) below.

---

## (c) Utility score for compression

**Eq. (1)** — the token-utility score actually used to decide which tokens to keep (quoted exactly, including the surrounding definitions):

> "Our prompt compression strategy selects salient tokens based on their averaged attention scores on the identified evaluator heads. Given a transformer-based language model f and the evaluator heads C_f identified through the pilot experiment delineated in Section 3.4. Let A_h^l ∈ R^{N×N} denote the matrix of attention scores for the layer l and the attention head h using f. For a long input prompt x, we utilize the attention scores from C_f to compute the utility scores s ∈ R^N for the input during the pre-filling stage according to
> s = Σ_{(l_j,h_j)∈C_f} Pool( Σ_{N_r≤i≤N} A_{l_j,h_j}[i, :] / N_o , r ),  (1)
> where Pool(·, ·) denotes a pooling operation, and r represents the kernel size, N_o is the observed window length and N_r is the length of remaining part such that N = N_r + N_o."
— NeurIPS PDF, Section 3.2 "Prompt compression using evaluator heads."

So the score for each key position is: **sum, over the evaluator heads in C_f, of the (pooled) average of the attention rows of the last N_o query positions onto that key position** — i.e. attention *from* the final N_o positions of the prompt *to* each earlier token, averaged over those N_o query rows and summed over heads, then pooled. This is structurally the same "observation window" idea as SnapKV (cited explicitly, see below).

**Query/instruction tokens:** the last N_o positions of the prompt act as the "observers" whose attention rows are averaged into the score — the method explicitly borrows this from SnapKV:

> "we adopt a pooling operation, as described in [Li et al., 2024], to group neighboring tokens with similar scores, thus generating a continuous sequence to enhance readability."
— NeurIPS PDF, Section 3.2 (Li et al. 2024 = SnapKV).

**UNVERIFIED:** I found no sentence stating that the instruction/question tokens are treated *specially* beyond being (implicitly) part of whatever sits in the last N_o positions of the prompt — there is no explicit claim that the question is excluded from scoring, given a separate/boosted weight, or kept outside the compression budget (contrast with LongLLMLingua's explicit "Question-Aware" mechanism, which the paper does describe for the *baseline*, not for EHPC itself):

> "LongLLMLingua [Jiang et al., 2023b] further incorporates task information (such as questions for document QA) and employs a Question-Aware strategy to enhance the density of key information in long contexts."
— NeurIPS PDF, Appendix I ("Baselines of prompt compression").

**Pooling / kernel:** average pooling is used throughout (max pooling tested and found equivalent), with model-specific (N_o, r) hyperparameters (Appendix D, "Using evaluator heads"):

> "The hyperparameters applied during the evaluation of heads include the size of the observed windows, the pooling operation, and the kernel size for pooling. In all experiments, we used the average pooling operation, as the difference between average pooling and maximum pooling was experimentally negligible. For Llama-3.1-8B-Instruct, we set the size of the observed windows and the kernel size for pooling to 16 and 32, respectively. For Phi-3.5-mini-3.8B-Instruct, the size of the observed windows and the kernel size for pooling were set to 4 and 32, respectively. A larger kernel size typically results in a more continuous compressed context, which is why we generally prefer to use a larger kernel size."
— NeurIPS PDF, Appendix D.

So: **N_o = 16, r = 32 for Llama-3.1-8B-Instruct; N_o = 4, r = 32 for Phi-3.5-mini-3.8B-Instruct** (CodeLlama-7B's (N_o, r) is not given in this sub-section — only its layer/heads are given, see (b)).

**Attention sink / BOS masking:** the paper discusses attention sinks only as *motivating background* for why raw attention scores are sparse enough to be useful for token selection — I found **no explicit masking step for the BOS/sink token anywhere in Eq. (1), the algorithm box (Appendix B), or the hyperparameter appendix (Appendix D)**. Quote of the only sink/BOS discussion in the paper (already given above under (a)'s neighboring context, repeated here for locator completeness):

> "This approach is feasible primarily because the attention scores of tokens in long texts are sparse, explained by the widely studied "attention sink" phenomenon (Xiao et al., 2023; Gu et al., 2024). This phenomenon is characterized by LLMs' frequently assigning high attention weights to the semantically inconsequential initial token, <BOS>."
— NeurIPS PDF / arXiv v2 HTML, Section 1 "Introduction."

**UNVERIFIED / not found:** an explicit "we mask/exclude the BOS or first-K sink tokens when computing s" step. Marking this as not confirmed rather than assuming EHPC does or doesn't handle sinks specially.

---

## (d) Budget enforcement

**Top-tokens-by-score, kept in original order, re-pooled for contiguity** (Algorithm described in Section 3.2 and formalized as Appendix Algorithm B / Eq. (5)):

> "Subsequently, we employ the scores s to remove non-essential tokens, constructing x̂ from the retained tokens in their original order. Although the compressed prompt retains its natural language form, it may lack certain contextual elements. To mitigate this, we adopt a pooling operation, as described in [Li et al., 2024], to group neighboring tokens with similar scores, thus generating a continuous sequence to enhance readability."
— NeurIPS PDF, Section 3.2.

> "we apply the pooling operation to perform clustering: ŝ = Pool(s, k), (5) where Pool(·, ·) represents a pooling operation such as max(·) and Average(·), and k is the kernel size. This trick ensures that the identified tokens are continuous rather than isolated, resulting in more coherent semantics."
— NeurIPS PDF, Appendix (Eq. 5, immediately preceding Appendix D "Hyper-parameters").

Formal algorithm signature (Appendix B):

> "Require: Input prompt x = (x1, x2, . . . , xN), and transformer-based LLM f. Ensure: Output compressed prompt x̂ = (x_{i1}, x_{i2}, . . . , x_{iM}) with M tokens. Detect the evaluator heads at layer l_e, C_f = {l_j, h_j} by the pilot experiments. Get A_{l_j,h_j} during the pre-filling stage at layer l_e layer. Get the scores by evaluator head[s]..."
— NeurIPS PDF, Appendix B "Algorithm" (truncated by fetch window at "evaluator head[s]..."; the top-M / thresholding step itself was not captured verbatim in the fetched excerpt — UNVERIFIED for the exact wording of the top-M selection line, though the *effect* — M = target token budget, e.g. 2,000/3,000 on LongBench/ZeroSCROLLS, 1,024/2,048 in the acceleration benchmark — is clearly stated via Table 4/6's "# Tokens" and "target lengths" language quoted below).

Target budgets used are explicit: **2,000 and 3,000 tokens** for the LongBench/ZeroSCROLLS compression benchmark (Table 4), and **1,024 and 2,048 tokens** for the KV-cache-equivalent acceleration benchmark (Table 6):

> "we report the averaged results for ZeroSCROLLS and LongBench, as well as the average performance for the sub-tasks on LongBench, with target compressed prompt lengths of 2, 000 and 3, 000 tokens."
— NeurIPS PDF, Section 5.1 "Results."

> "For each method, we set the target lengths to 1024 and 2048 tokens for prompt compression and KV cache compression, respectively."
— NeurIPS PDF, Section 5.2 "Implementation detail."

**Single forward pass / partial pre-filling.** Compression itself uses only the layers up to (and including) the evaluator-head layer l_e in one pre-filling pass — it is not an iterative, multi-call, chunked process like LLMLingua/LongLLMLingua:

> "While our prompt compression strategy necessitates the processing of prompts by an LLM, it leverages the computationally efficient pre-filling stage, enabling fast compression."
— NeurIPS PDF, Section 3.2.

> "In the NMI setting, where f is used for both compression and inference, our method involves two pre-filling stages and one decoding stage... The first pre-filling stage utilizes only L/κ1 layers to compress the prompts, resulting in the complexity of O(LHd_k N^2/κ1)."
— NeurIPS PDF, Section 3.5 "Complexity" / Figure 3 caption region (κ1 = L / max_l (S·1) — i.e. total layers divided by the detected evaluator-head layer's rank).

**Reported speedups** — compression-time and end-to-end latency (Table 5, "Averaged time (in seconds) for different methods applied to a subset of LongBench," 2,048-token target, quoted verbatim from the table):

| Method | Compression time (s) | Inference time (2048 tok) (s) | Total (s) |
|---|---|---|---|
| LLMLingua | 7.51 | 1.09 | 8.60 |
| LongLLMLingua | 67.44 | 1.31 | 68.74 |
| LLMLingua-2 | 1.27 | 1.15 | 2.37 |
| EHPC (ours) | 0.88 | 1.11 | 1.99 |
| ChatGPT-3.5-Turbo (all tokens) | — | — | 2.16 |
— NeurIPS PDF, Table 5.

> "The results in Table 5 indicate that our method is significantly faster than LongLLMLingua and also outperforms LLMLingua-2, which is known for its efficiency. The lower latency of our compression strategy is attributed to its reliance on the efficient pre-filling stage."
— NeurIPS PDF, Section 5.1 "Compression latency."

Timing setup detail:

> "The subset contains 10 examples from RepoBench-P, with an average of 14,354 tokens. Each example was repeated 5 times to reduce randomness... Each prompt compression method reduces the average prompt length from 14,354 tokens to 2,048 tokens. The experiments were carried out on a GPU with 48GB of VRAM. For a fair comparison, we used the same language model, Llama-3.1-8B, for all methods except LLMLingua-2, which requires a specialized model."
— NeurIPS PDF, Section 5.1.

---

## (e) Published numbers — LongBench results (THE important part)

### Table 1 (headline summary, average-only, 2,048-token constraint)

> "Table 1: Overall comparison of the proposed method in terms of average performance and latency on the LongBench dataset, under the constraint of a compressed prompt length of 2048 tokens. For comprehensive results, please see Table 4 and Table 5."

| Method | Performance | Latency | Training-free |
|---|---|---|---|
| LongLLMLingua | 48.0 | 67.44 | ✓ |
| LLMLingua | 34.6 | 7.51 | ✓ |
| LLMLingua-2 | 39.1 | 1.27 | ✗ |
| EHPC (ours) | **49.6** | **0.88** | ✓ |
— NeurIPS PDF, Table 1.

### Table 4 — full compression-benchmark table (this is the paper's *main* prompt-compression result table; note its columns are **category aggregates**, not individual datasets — HotpotQA/2WikiMQA/MuSiQue are folded into the single "MultiDoc" column here)

> "Table 4: Performance of various prompt compression methods under different compressed length constraints on LongBench and ZeroSCROLLS. Higher values indicate better performance. The best scores are highlighted in boldface."
Columns: `SingleDoc | MultiDoc | Summ. | FewShot | Synth. | Code | Avg. | #Tokens(LB) | κ2(LB) | Avg.(ZS) | #Tokens(ZS) | κ2(ZS)`

Original Prompt (uncompressed): 39.7 / 38.7 / 26.5 / 67.0 / 37.8 / 54.2 / **44.0** / 10,295 / – / 32.5 / 9,788 / –
Zero-shot (no context): 15.6 / 31.3 / 15.6 / 40.7 / 1.6 / 36.2 / **23.5** / 214 / 48× / 10.8 / 32 / 306×

**3,000-token constraint:**
| Method | SingleDoc | MultiDoc | Summ. | FewShot | Synth. | Code | Avg. | #Tok(LB) | κ2 | Avg(ZS) | #Tok(ZS) | κ2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 32.3 | 34.3 | 25.3 | 57.9 | 45.1 | 48.9 | 40.6 | 3,417 | 3× | 19.8 | 3,379 | 3× |
| SBERT | 35.3 | 37.4 | 26.7 | 63.4 | 51.0 | 34.5 | 41.4 | 3,399 | 3× | 24.0 | 3,340 | 3× |
| OpenAI (embedding) | 34.5 | 38.6 | 26.8 | 63.4 | 49.6 | 37.6 | 41.7 | 3,421 | 3× | 22.4 | 3,362 | 3× |
| Selective-Context | 23.3 | 39.2 | 25.0 | 23.8 | 27.5 | 53.1 | 32.0 | 3,328 | 3× | 20.7 | 3,460 | 3× |
| LLMLingua | 31.8 | 37.5 | 26.2 | 67.2 | 8.3 | 53.2 | 37.4 | 3,421 | 3× | 30.7 | 3,366 | 3× |
| LLMLingua-2 | 35.5 | 38.7 | 26.3 | 69.6 | 21.4 | 62.8 | 42.2 | 3,392 | 3× | 33.5 | 3,206 | 3× |
| LongLLMLingua | 40.7 | 46.2 | 27.2 | 70.6 | 53.0 | 55.2 | 48.8 | 3,283 | 3× | 32.8 | 3,412 | 3× |
| **EHPC (EMI)** | **44.2** | **49.1** | 25.1 | 68.8 | 54.0 | 63.0 | **49.7** | 2,892 | 3× | **36.7** | 3,005 | 3× |

**2,000-token constraint:**
| Method | SingleDoc | MultiDoc | Summ. | FewShot | Synth. | Code | Avg. | #Tok(LB) | κ2 | Avg(ZS) | #Tok(ZS) | κ2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25 | 30.1 | 29.4 | 21.2 | 19.5 | 12.4 | 29.1 | 23.6 | 1,985 | 5× | 20.1 | 1,799 | 5× |
| SBERT | 33.8 | 35.9 | 25.9 | 23.5 | 18.0 | 17.8 | 25.8 | 1,947 | 5× | 20.5 | 1,773 | 6× |
| OpenAI (embedding) | 34.3 | 36.3 | 24.7 | 32.4 | 26.3 | 24.8 | 29.8 | 1,991 | 5× | 20.6 | 1,784 | 5× |
| Selective-Context | 16.2 | 34.8 | 24.4 | 15.7 | 8.4 | 49.2 | 24.8 | 1,925 | 5× | 19.4 | 1,865 | 5× |
| LLMLingua | 22.4 | 32.1 | 24.5 | 61.2 | 10.4 | 56.8 | 34.6 | 1,950 | 5× | 27.2 | 1,862 | 5× |
| LLMLingua-2 | 29.8 | 33.1 | 25.3 | 66.4 | 21.3 | 58.9 | 39.1 | 1,954 | 5× | 33.4 | 1,898 | 5× |
| LongLLMLingua | 39.0 | 42.2 | 27.4 | 69.3 | 53.8 | 56.6 | 48.0 | 1,809 | 6× | 32.5 | 1,753 | 6× |
| **EHPC (EMI)** | **44.5** | **50.7** | 24.8 | 68.9 | 51.5 | 61.9 | **49.6** | 2,004 | 5× | **34.6** | 2,041 | 5× |

(all numbers transcribed verbatim from the grepped/fetched Table 4 text of the NeurIPS PDF; "κ2" is the paper's own compression-ratio symbol.)

**Compression-ratio definition ("1/x"):** κ2 (also called κ2 elsewhere in Sec. 3.5/5.2) is the *original-length ÷ compressed-length* ratio, reported per-row next to the actually-achieved token count (e.g. "1,809 / 6×" for LongLLMLingua at the "2,000-token constraint" row means its real compressed length averaged 1,809 tokens against an original ~10,295–9,788-token prompt, i.e. roughly a 5.7-6× reduction) — the nominal "2,000/3,000 tokens" figures are **targets**, and the "#Tokens" column shows the *actual* average post-compression length each method achieved (methods don't hit the target exactly; EHPC's 2,004 and 2,892 are the closest-to-target of all rows shown).

**Target/downstream LLM for Table 4:** the text states EHPC is run in its **EMI (Extended Model Inference)** setting for this benchmark and that the compressor model is Llama-3.1-8B:

> "We compare our EMI setting, which employs a local LLM for prompt compression and another model for inferring the compressed prompts, against the following baselines... For prompt compression, we utilize the Llama-3.1-8B model."
— NeurIPS PDF, Section 5.1 "Baselines and implementation details" / "Implementation Detail."

**UNVERIFIED:** I could not find an explicit sentence in the fetched text naming the *specific* downstream/answering model ("another model" above) used to score accuracy in Table 4. The paper says this benchmark follows the protocol "established by Jiang et al. [2023b]" (i.e., LongLLMLingua), whose own published protocol uses GPT-3.5-turbo-0301 as the answering model, and Table 5 independently uses "ChatGPT-3.5-Turbo (all tokens)" as its uncompressed-latency reference point — both are circumstantial evidence pointing to GPT-3.5-Turbo, but I did not find EHPC's own text explicitly naming GPT-3.5-Turbo as the Table-4 answering model, so this is flagged rather than stated as confirmed.

### Resolving the flagged inconsistency ("HotpotQA 20.64 for EHPC-1024 at ~5x vs 39.0 for LongLLMLingua")

This comparison mixes **two different tables, two different target models, and two different budget definitions** — it is not actually a contradiction inside the paper, it is a comparison across incompatible rows:

1. **"39.0" is not a HotpotQA number at all.** It is LongLLMLingua's **SingleDoc-QA** *category-aggregate* score in **Table 4** at the "2,000 tokens constraint" (EMI setting, target model = an "another model" per Sec. 5.1, likely GPT-3.5-Turbo per the LongLLMLingua protocol reference — see UNVERIFIED note above). Table 4 has **no per-dataset HotpotQA/2WikiMQA/MuSiQue columns at all** — multi-doc QA datasets are folded into the single "MultiDoc" aggregate column (LongLLMLingua's MultiDoc value at 2,000 tokens is **42.2**, not 39.0 either).

2. **"20.64" is a real number, but from a completely different benchmark: Table 6**, "Acceleration of LLM inference," which is the **NMI (Native Model Inference)** self-inference benchmark under a **1,024-token KV-cache-equivalent budget**, with **Llama-3.1-8B-Instruct itself** (not GPT-3.5) as both compressor and answering model, and LongLLMLingua is **not even a baseline in Table 6** (its baselines are All-KV / H2O / SnapKV / GemFilter). Full Table 6 HotpotQA/2WikiMQA/MuSiQue rows, quoted verbatim (columns: NarrativeQA, Qasper, MultiQA-en | HotpotQA, 2WikiMQA, Musique | ... | Average):

**LLaMA 3.1 8B Instruct, budget = 1024:**
| Method | HotpotQA | 2WikiMQA | Musique | Average |
|---|---|---|---|---|
| All KV (uncompressed) | 16.23 | 16.05 | 11.22 | 38.32 |
| H2O-1024 | 16.06 | 15.16 | 10.15 | 34.77 |
| SnapKV-1024 | 14.81 | 15.73 | 10.69 | 35.25 |
| GemFilter-1024 | 19.12 | 17.01 | 13.01 | 34.50 |
| **EHPC (ours)-1024** | **20.64** | 16.97 | **13.99** | 35.23 |

**LLaMA 3.1 8B Instruct, budget = 2048:**
| Method | HotpotQA | 2WikiMQA | Musique | Average |
|---|---|---|---|---|
| H2O-2048 | 16.17 | 15.22 | 9.93 | 35.45 |
| SnapKV-2048 | 15.73 | 16.03 | 11.66 | 35.80 |
| GemFilter-2048 | 19.58 | 17.03 | 14.11 | 35.87 |
| EHPC (ours)-2048 | 19.35 | 16.23 | 13.02 | **37.86** |

**Phi 3.5 Mini 3.8B Instruct, budget = 1024:**
| Method | HotpotQA | 2WikiMQA | Musique | Average |
|---|---|---|---|---|
| All KV | 21.70 | 25.70 | 11.68 | 34.62 |
| H2O-1024 | 20.75 | 20.90 | 9.90 | 31.38 |
| SnapKV-1024 | 20.72 | 26.02 | 13.74 | 33.68 |
| GemFilter-1024 | 24.22 | 26.10 | 9.70 | 32.74 |
| **EHPC (ours)-1024** | **44.97** | **32.79** | **20.27** | **39.22** |

**Phi 3.5 Mini 3.8B Instruct, budget = 2048:**
| Method | HotpotQA | 2WikiMQA | Musique | Average |
|---|---|---|---|---|
| H2O-2048 | 20.03 | 22.51 | 10.30 | 31.94 |
| SnapKV-2048 | 21.80 | 26.07 | 12.57 | 34.20 |
| GemFilter-2048 | 21.38 | 19.72 | 10.13 | 31.69 |
| **EHPC (ours)-2048** | 27.06 | 25.22 | 14.05 | **36.46** |
— NeurIPS PDF, Table 6, quoted verbatim.

So **within Table 6's own comparison** — its actual apples-to-apples table — EHPC-1024 (20.64 on HotpotQA) *beats every listed budget-1024 baseline* (H2O 16.06, SnapKV 14.81, GemFilter 19.12) *and even the uncompressed All-KV baseline* (16.23), fully consistent with the paper's SoTA claim. The apparent contradiction only arises from comparing it against a number ("39.0") that is (i) from an unrelated table, (ii) a SingleDoc-QA aggregate rather than a HotpotQA-specific score, (iii) evaluated in a different setting (EMI vs NMI), and (iv) not even the same metric column.

3. There is a **third, smaller comparison** with per-dataset HotpotQA/2WikiMHQA/MuSiQue numbers, in **Appendix G.2, Table 9** ("Comparison with additional baselines" — Minference and CritePrefill), again on the **Phi** model directly (self-inference), at what the row label calls "(1024)":

> "Model: Phi | Method: CritePrefill | HotpotQA 4.72 | 2WikiMHQA 31.70 | MuSiQue 20.42 ... | Method: Minference | HotpotQA 18.81 | 2WikiMHQA 21.26 | MuSiQue 9.80 ... | Method: Our (1024) | HotpotQA 44.97 | 2WikiMHQA 32.79 | MuSiQue 20.27 ..."
— NeurIPS PDF, Appendix G.2, Table 9.

(Note: "Our (1024)" HotpotQA=44.97/2WikiMHQA=32.79/MuSiQue=20.27 numerically matches EHPC-1024's Phi-model row in Table 6 exactly — the two tables report the same underlying EHPC run, just against different baseline sets.)

**Bottom line for (e):** the paper's *only* table that breaks LongBench Multi-Doc QA into HotpotQA/2WikiMQA/MuSiQue is **Table 6 (and its appendix companion, Table 9)** — the self-inference/NMI acceleration benchmark, target model = Llama-3.1-8B-Instruct or Phi-3.5-Mini-3.8B-Instruct itself, budgets 1024/2048. The paper's *headline* LongBench/ZeroSCROLLS **prompt-compression** benchmark against Selective-Context/LLMLingua/LongLLMLingua/LLMLingua-2 (Table 4, EMI setting, budgets 2,000/3,000, compressor = Llama-3.1-8B) reports only **category aggregates** (SingleDoc/MultiDoc/Summ./FewShot/Synth./Code/Avg.) and never breaks MultiDoc QA down to the individual HotpotQA/2WikiMQA/MuSiQue level. Any per-dataset HotpotQA number compared against Table-4-style baselines (Selective-Context, LLMLingua, LongLLMLingua, LLMLingua-2) **does not exist in this paper** — trying to build one requires pulling LongLLMLingua et al. numbers from a different source paper and EHPC numbers from Table 6, which is not a valid apples-to-apples comparison the authors themselves make.

**Question-in-budget:** UNVERIFIED — no sentence found stating whether the question/instruction text is counted inside the "2,000/3,000 tokens" budget of Table 4 or added on top of it (contrast: LongLLMLingua's own paper is explicit that the question is kept outside the budget; EHPC's paper does not repeat or contradict this for its own method in the fetched text).

---

## Blockers / things not verified

- **OpenReview `forum?id=yOs12gdsaL` page returned a bot-check wall** ("Verifying your browser... Please complete the verification above") on direct fetch and could not be read; reviews/rebuttal content (which might contain author comments on code release) was not accessible this way. The `openreview.net/pdf?id=...` variant appeared in web-search snippets with the same title/author/abstract as the NeurIPS PDF, so it is very likely an identical camera-ready mirror, but this was not independently fetched byte-for-byte.
- Exact NIAH sample count / haystack-length range for the head-*identification* pilot experiment (Sec. 3.4) — not found (only the separate Appendix-F speedup-timing experiment gives "8k to 32k... five examples for each length").
- Exact downstream/answering model for Table 4's accuracy numbers — not explicitly named in the fetched text (circumstantial evidence points to GPT-3.5-Turbo via the "established by Jiang et al. [2023b]" protocol reference and Table 5's ChatGPT-3.5-Turbo latency baseline, but no direct quote confirms it).
- Whether instruction/question tokens are excluded from, or counted inside, the compression budget — not found.
- Whether BOS/attention-sink tokens are explicitly masked when computing the utility score (Eq. 1) — not found; the paper only cites the sink phenomenon as motivating background.
- Contents of the NeurIPS Supplemental-Conference.zip (existence/downloadability confirmed via HTTP HEAD; contents not inspected).
- CodeLlama-7B's (N_o, r) pooling hyperparameters — Appendix D gives these only for Llama-3.1-8B-Instruct and Phi-3.5-mini-3.8B-Instruct, not for CodeLlama-7B.

---

## Sources (all URLs actually fetched/grepped during this task, 2026-09-20)

1. https://arxiv.org/abs/2501.12959 — arXiv abstract page (title/version history: v1 22 Jan 2025, v2 5 Feb 2025, same title both versions)
2. https://arxiv.org/html/2501.12959v2 — arXiv v2 full HTML (abstract, intro, Table 1, section TOC)
3. https://proceedings.neurips.cc/paper_files/paper/2025/file/e356ed5f27885c79c7cb597bb1107c94-Paper-Conference.pdf — NeurIPS 2025 camera-ready PDF (primary source for essentially all quotes above; fetched in full via multiple offsetted fetches + targeted regex greps)
4. https://proceedings.neurips.cc/paper_files/paper/2025/hash/e356ed5f27885c79c7cb597bb1107c94-Abstract-Conference.html — NeurIPS abstract/proceedings page (confirms Supplemental zip link, DOI)
5. https://proceedings.neurips.cc/paper_files/paper/2025/file/e356ed5f27885c79c7cb597bb1107c94-Supplemental-Conference.zip — confirmed to exist via `curl -I` (200 OK, 1,431,123 bytes, application/zip); contents not inspected
6. https://openreview.net/forum?id=yOs12gdsaL — blocked by browser-verification wall; not readable (BLOCKER)
7. https://github.com/comsa33/ehpc-research — unofficial third-party reimplementation (HTML fetch)
8. https://api.github.com/search/repositories?q=EHPC+prompt+compression — GitHub API search (1 result)
9. https://api.github.com/search/repositories?q=evaluator+heads+prompt+compression — GitHub API search (1 result, same repo)
10. https://huggingface.co/papers/2501.12959 — HF Papers page (no linked code/models/datasets/spaces)
11. https://paperswithcode.com/paper/efficient-prompt-compression-with-evaluator — PapersWithCode mirror (no code links)
12. Web searches (via aii_fast_web_search.py, general/exa mode unless noted): "EHPC evaluator heads prompt compression github"; "EHPC evaluator heads prompt compression code availability github official implementation"; "site:github.com \"evaluator heads\" EHPC Fei Niu prompt compression official code"; "openreview yOs12gdsaL EHPC code supplementary"; "\"Evaluator Heads\" prompt compression EHPC NeurIPS 2025 Fei" (scholarly/openalex mode)
