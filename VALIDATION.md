# Validation - 8 October 2026
Fourteen tests passed: causal future-token isolation, next-token batch boundaries,
finite gradients and actual learning, checkpoint round-trip, shapes/generation.
A real 150-step CPU experiment completed. Exact environment, data hash, parameters,
training curve and held-out results are in training-report.json. PyTorch emitted
an optional NumPy initialization warning; no NumPy conversions are used here.
GPU, pretrained large-model fine-tuning, distributed training and real-language quality are not tested.

A real 150-step LoRA/full fine-tuning comparison completed on CPU. 1,536 LoRA
parameters trained versus 28,706 for full tuning. The original base weights stayed
bitwise unchanged. Saved/restored adapter logits matched exactly; merged logits
maximum error was 4.77e-6. Final new-domain loss improved in both methods, while
original-domain loss worsened (recorded). See adapter-report.json for the complete
measured comparison and caveats. Shared templates preclude a generalization claim.
