from typing import Union, List, Dict, Tuple, Optional
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel, BatchEncoding
from torch.nn.utils.rnn import pad_sequence

def apply_prompt_template(
    prompt: str, 
    tokenizer: PreTrainedTokenizer,
    prompt_template: str
) -> str:
    if prompt_template == 'chat':
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif prompt_template == 'base':
        return f"Question: {prompt}\n Answer: "
    elif prompt_template == 'no':
        return prompt
    else:
        raise ValueError("Invalid prompt template type. Choose 'chat', 'base', or 'none'.")

PromptLike = Dict[str, List[str]]
TokenisedPrompt = Dict[str, List[BatchEncoding]]

def template_tokenize_single_prompt(prompt: str, tokenizer: PreTrainedTokenizer, *, prompt_template: str = "no", tokenize_kwargs: Dict = None) -> BatchEncoding:
    if tokenize_kwargs is None:
        tokenize_kwargs = {}
    templated = apply_prompt_template(prompt, tokenizer, prompt_template)
    return tokenizer(templated, **tokenize_kwargs)

def template_tokenize_prompts(prompts: PromptLike, tokenizer: PreTrainedTokenizer, *, prompt_template: str = "no", tokenize_kwargs: Dict = None) -> TokenisedPrompt:
    return {k: [template_tokenize_single_prompt(p, tokenizer, prompt_template=prompt_template, tokenize_kwargs=tokenize_kwargs) for p in v] for k, v in prompts.items()}

GenResult = Tuple[
    torch.LongTensor,       # generated token ids,  (P+T,) where P is prompt length and T is generated length
    torch.FloatTensor,      # probs over vocab,     (T , |V|)
    str                     # decoded text
]                        

def _process_chunk(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    enc_list: List[BatchEncoding],
    max_new_tokens: int,
    pad_id: int,            # kept for call signature; no longer used
    device: torch.device,
) -> List[GenResult]:
    """
    Greedy‑generate continuations for a micro‑batch and return
    (tokens_(P+T,), probs_(T,V), decoded_text) for each item.
    """

    # ------------------------------------------------------------------
    # 1. Let the tokenizer do the padding
    # ------------------------------------------------------------------
    batch = tokenizer.pad(
        enc_list,                    # list[BatchEncoding]
        padding="longest",
        return_tensors="pt"
    )
    input_ids  = batch["input_ids"      ].to(device)
    attn_mask  = batch["attention_mask" ].to(device)

    # ------------------------------------------------------------------
    # 2. Greedy generation (no sampling, return per‑step scores)
    # ------------------------------------------------------------------
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,    # ← tell it what “stop” looks like
            early_stopping=True,                    # ← stop when you hit EOS
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    seqs   = out.sequences                       # (B , P+T)
    scores = torch.stack(out.scores, dim=1)      # (B , T , |V|)

    # ------------------------------------------------------------------
    # 3. Convert logits to probabilities (float32 for accuracy)
    # ------------------------------------------------------------------
    probs = torch.softmax(scores.float(), dim=-1).cpu()   # (B , T , |V|)

    # ------------------------------------------------------------------
    # 4. Decode full sequences to text
    # ------------------------------------------------------------------
    texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in seqs]

    # ------------------------------------------------------------------
    # 5. Pack results
    # ------------------------------------------------------------------
    return [
        (ids.cpu(), pr, txt)
        for ids, pr, txt in zip(seqs, probs, texts)
    ]


def _flatten(tokenised: TokenisedPrompt) -> Tuple[List[BatchEncoding], List[Tuple[str | None, int]]]:
    """
    Flattens List[...] or Dict[str, List[...]] into a single list.
    Returns the flat list *and* a mapping so we can rebuild the
    original structure afterwards.
    """
    flat, index = [], []
    for k, lst in tokenised.items():
        for i, enc in enumerate(lst):
            flat.append(enc)
            index.append((k, i))
    return flat, index

def get_tokens_and_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tokenized_prompts: TokenisedPrompt,
    *,
    max_new_tokens: int = 128,
    micro_batch_size: int | None = None,
) -> Dict[str, List[GenResult]]: # GenResult = (tokens, probs, decoded_text)
    """
    Batched greedy generation returning (tokens, probs, decoded_text).
    """
    device   = next(model.parameters()).device
    pad_id   = model.config.pad_token_id or model.config.eos_token_id
    flat_enc, mapping = _flatten(tokenized_prompts)

    if not micro_batch_size or micro_batch_size <= 0:
        micro_batch_size = len(flat_enc)

    flat_results: List[GenResult] = []
    for start in range(0, len(flat_enc), micro_batch_size):
        chunk = flat_enc[start : start + micro_batch_size]
        print(f"Processing micro‑batch {start // micro_batch_size + 1}"
              f" of {len(flat_enc) // micro_batch_size} …", flush=True)

        flat_results.extend(
            _process_chunk(model, tokenizer, chunk, max_new_tokens, pad_id, device)
        )
        torch.cuda.empty_cache()

    # ── rebuild nested structure identical to tokenized_prompts ────────
    nested: Dict[str, List[GenResult]] = {}
    for (key, pos), res in zip(mapping, flat_results):
        nested.setdefault(key, []).append(res)
    return nested


# ----------------------------------------------------------------------
#--- Teacher forcing version of the above function
# ----------------------------------------------------------------------
def get_teacher_forcing_tokens_and_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    non_deactivated_token_and_logits: Dict[str, List[GenResult]],
    *,
    micro_batch_size: int = 32,
) -> Dict[str, List[GenResult]]:
    """
    Run teacher forcing with the deactivated model using tokens from non_deactivated_token_and_logits.
    Returns the same structure but with probabilities from the deactivated model.
    If sample_alternative_tokens=True, also samples what tokens the deactivated model would have chosen.
    """
    device = next(model.parameters()).device
    pad_id = model.config.pad_token_id or model.config.eos_token_id
    
    # Flatten the non_deactivated results to process in batches
    flat_items = []
    mapping = []
    
    for key, gen_results in non_deactivated_token_and_logits.items():
        for i, (tokens, probs, decoded_text) in enumerate(gen_results):
            flat_items.append((tokens, probs, decoded_text))
            mapping.append((key, i))
    
    # Process in micro-batches
    flat_results: List[GenResult] = []
    
    for start in range(0, len(flat_items), micro_batch_size):
        end = min(start + micro_batch_size, len(flat_items))
        chunk = flat_items[start:end]
        
        print(f"Processing teacher forcing micro-batch {start // micro_batch_size + 1}"
              f" of {(len(flat_items) + micro_batch_size - 1) // micro_batch_size} …", flush=True)
        
        # Process this chunk
        chunk_results = _process_teacher_forcing_chunk(
            model, tokenizer, chunk, pad_id, device
        )
        flat_results.extend(chunk_results)
        torch.cuda.empty_cache()
    
    # Rebuild nested structure identical to non_deactivated_token_and_logits
    nested: Dict[str, List[GenResult]] = {}
    for (key, pos), res in zip(mapping, flat_results):
        nested.setdefault(key, []).append(res)
    
    return nested


def _process_teacher_forcing_chunk(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    chunk: List[GenResult],  # List of (tokens, probs, decoded_text)
    pad_id: int,
    device: torch.device,
) -> List[GenResult]:
    """
    Process a chunk of teacher forcing items.
    """
    enc_list: List[BatchEncoding] = [
        {"input_ids": seq} for seq, _, _ in chunk
    ]
    batch = tokenizer.pad(
        enc_list,
        padding="longest",
        return_tensors="pt",
    )
    # after building `batch`:
    input_ids     = batch["input_ids"].to(device)            # (B, L_pad)
    attention_msk = batch["attention_mask"].to(device)       # (B, L_pad)

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_msk,
            return_dict=True,
        ).logits                                             # (B, L_pad, |V|)

    probs = torch.softmax(logits.float(), dim=-1).cpu()       # ensure float32

    results = []
    for i, (original_tokens, original_probs, original_decoded_text) in enumerate(chunk):
        T = original_probs.shape[0]                           # generated length
        L_eff = int(attention_msk[i].sum().item())
        # Guard: T must be <= L_eff - 1
        assert 0 < T <= L_eff - 1, f"Bad lengths: T={T}, L_eff={L_eff}"

        start = (L_eff - T) - 1
        end   = L_eff - 1                                     # exclusive
        item_probs = probs[i, start:end, :]                   # (T, |V|)

        # (optional) deterministic debug path:
        # sampled_tokens = item_probs.argmax(dim=-1)
        
        sampled_tokens = item_probs.argmax(dim=-1)
        # sampled_tokens = torch.multinomial(item_probs, num_samples=1).squeeze(-1)
        sampled_decoded = tokenizer.decode(sampled_tokens, skip_special_tokens=True)

        # OPTIONAL: sanity asserts to catch misalignment early
        # 1) Make sure we're not reading from padding
        assert attention_msk[i, start].item() == 1 and attention_msk[i, end-1].item() == 1
        # 2) Length check
        assert item_probs.shape[0] == T

        results.append((sampled_tokens, item_probs, sampled_decoded))


        print(f"Original tokens shape: {original_tokens.shape}")
        print(f"Original probs shape:  {original_probs.shape}")
        # print(f"Inferred prompt_len:   {prompt_len}")
        print(f"Start index: {start}, End index: {end}, Logits shape: {logits[i].shape}")
        print(f"Generated text: {sampled_decoded}")

    
    return results