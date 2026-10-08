# Validation - 8 October 2026
Five tests passed: causal future-token isolation, next-token batch boundaries,
finite gradients and actual learning, checkpoint round-trip, shapes/generation.
A real 150-step CPU experiment completed. Exact environment, data hash, parameters,
training curve and held-out results are in training-report.json. PyTorch emitted
an optional NumPy initialization warning; no NumPy conversions are used here.
GPU, fine-tuning, distributed training and real-language quality are not tested.
