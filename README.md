# Tiny Transformer Lab

AI-assisted learning implementation of a small decoder-only Transformer in PyTorch.
Manual multi-head causal attention, learned positional embeddings, pre-norm residual
blocks, character-level next-token loss, AdamW, gradient clipping and CPU training.
No pretrained model downloads, customer data, GPU claims or fine-tuning claims.

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
