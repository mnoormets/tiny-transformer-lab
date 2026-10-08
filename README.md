# Tiny Transformer Lab

AI-assisted learning implementation of a small decoder-only Transformer in PyTorch.
Manual multi-head causal attention, learned positional embeddings, pre-norm residual
blocks, character-level next-token loss, AdamW, gradient clipping and CPU training.
No pretrained model downloads, customer data or GPU claims. A small-model LoRA/full fine-tuning comparison is included below.

## Run
Install PyTorch for your platform from https://pytorch.org/get-started/locally/ and
pytest in an isolated environment. `python -m pytest -q`; `python -m model.train
--steps 150`; `python -m model.generate`.

The run records the seed, config, exact corpus hash, runtime versions, loss curve,
unigram baseline and a fixed contiguous held-out validation suffix. Checkpoints
are ignored from Git. The tokenizer vocabulary is derived from the complete corpus;
no validation targets are optimized. Shared synthetic sentence templates make the
validation task easy: results do not demonstrate unseen-document or general-language
performance. This model is deliberately too small for useful text generation.

## Explain before claiming the skill
1. Why must logits at position 4 not depend on a token at position 5?
2. Why are targets shifted by one character?
3. What does dividing attention scores by sqrt(head_dim) do?
4. Why compare validation loss with a unigram model and the untrained network?
5. Change heads from 4 to 2, predict tensor shapes, then run the invariance test.

This is a from-scratch architecture learning demo using PyTorch tensor/autograd
primitives, not a distributed-training system, QLoRA pipeline or production LLM.

## Low-rank adapter experiment
Run `python -m model.train --steps 150`, then `python -m model.adapt --steps 150`.
This executes real CPU adaptation of our own 28,706-parameter model; it is not
QLoRA on Llama or a GPU cluster experiment. LoRALinear computes W*x plus
(alpha/rank)*B*A*x. A starts random and B starts zero, preserving base outputs.
Only attention qkv/projection adapters train; all base parameters stay frozen.

The fixed run updates 1,536 adapter parameters. New-template final test loss:
frozen base 1.8010, LoRA 0.5132, full fine-tuning 0.2744. On the original template,
loss changes from 0.5737 to 1.3992 (LoRA) and 2.4925 (full): adaptation improves the
new domain but degrades the old domain. Frozen base weights alone do not prevent
changed model behavior. The saved adapter can be removed to restore the base.

Reports include dataset/checkpoint hashes, exact split, seeds, parameter counts,
loss curves, timing, unchanged-base checks and adapter/merged-logit verification.
The repeated template makes this an easy synthetic task, not a language benchmark.
The report is in adapter-report.json; checkpoints stay in ignored artifacts/.

Saved adapters are loaded with weights_only=True and verified against a hash of
the matching base model and its configuration. Loading validates all tensor keys,
shapes, dtypes and finite values on a copy before returning the adapted model.
Merging returns a separate inference model, preserving the original adapter model.
Fourteen tests cover base output equivalence, frozen trainable sets, corrupted or
mismatched adapters, merge/reload equivalence and original Transformer behavior.

Algorithm source: https://arxiv.org/abs/2106.09685. This implementation is based on
the low-rank update principle; no performance claims from that paper are attributed
to this small project. Hosted CI trains only ten steps as a smoke check; the local
150-step result is recorded separately.
