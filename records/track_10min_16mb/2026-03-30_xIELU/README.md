# Full Competition Stack: 11L + XSA + SmearGate + BigramHash + EMA + Int6 GPTQ-lite

## Summary
Full competition architecture incorporating ALL proven techniques from top leaderboard submissions.
Targeting 8xH100 SXM, 600s wallclock, ~16MB artifact.

## Techniques (cumulative ~0.10 BPB improvement over baseline)

| Technique | Impact | Status |
|-----------|--------|--------|
| 11 layers (was 9) | ~-0.03 BPB | Done |
| 3x MLP (hidden=1536) | ~-0.01 BPB | Done |
| SmearGate (learned prev-token blending) | ~-0.005 BPB | Done |
| BigramHash (2048-bucket hash embedding) | ~-0.005 BPB | Done |
| Orthogonal init + muP scaling | ~-0.005 BPB | Done |
| Weight decay (Muon 0.04, Adam 0.04) | ~-0.005 BPB | Done |
| EMA (decay=0.997, every step) | ~-0.006 BPB | Done |
| XSA on last 4 layers | ~-0.005 BPB | Done |
| Partial RoPE (16/64 dims) + NTK scaling | ~-0.002 BPB | Done |
| LN Scale (1/sqrt(layer+1)) | ~-0.002 BPB | Done |
| Sliding window eval (stride=64) | ~-0.02 BPB | Done |
| Int6 GPTQ-lite (5 clip percentiles) | best compression | Done |
| Seq len 2048, batch 786k tokens | more throughput | Done |
| Tuned LRs (matrix=0.025, scalar=0.025) | better convergence | Done |
| Warmdown 3500 iters, grad clip 0.3 | stability | Done |

## Architecture
- 11 layers, 512 dim, 8 heads (4 KV), relu² MLP 3x
- SmearGate + BigramHash at embedding layer
- XSA on last 4 layers, partial RoPE 16/64, LN Scale
- U-Net skip connections, tied embeddings, logit softcap=30
- EMA weight averaging, int6 GPTQ-lite + zlib

## How to Train (8xH100, RunPod)
```bash
bash run_runpod.sh
```

## Runs
- **v0**: seed 1337, 8xH100, 600s cap (pending)
- **v1**: seed 42 (pending)
- **v2**: seed 2025 (pending)
