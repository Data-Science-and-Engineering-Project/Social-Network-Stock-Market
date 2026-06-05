# Known issues — slumrun

## GAT OOM on 2024 Q2-Q4 for `multiq_A` and `multiq_B`

**State**: 41 / 44 quarters complete per multiq corner. The 3 missing GAT
quarters are 2024Q2, 2024Q3, 2024Q4. SAGE has full coverage (44/44). The
bipartite corner is fully complete (46/46 × both edge cols × both models).

**Root cause**: GAT activation memory grew with the multi-quarter union graph
size. On RTX 3090 (24 GB) the failed quarters peak at 23.5 / 24.0 / 25.0 GiB
respectively (PyTorch allocated + autograd buffer for `loss.backward()`). bf16
autocast is already on (`USE_AMP` in `_common.py`); the issue is GPU capacity,
not numerical precision.

**Retry recipe** (for when bigger GPUs are free):

The smart-resume in `_common.py:load_done_set` requires both `sage_acc` and
`gat_acc` to be non-null per row. After dedupe, the 3 failed quarters per
multiq corner are the only rows missing GAT, so a re-submission picks them
up automatically — nothing else gets recomputed.

Either edit the GPU line in the sbatch:
```bash
# in SocialNetwork/runandcompare_A.sbatch and runandcompare_B.sbatch
#SBATCH --gpus=rtx_5090:1   # 32 GB, fits with ~5 GB headroom on the worst quarter
```

…or override at submission time without editing the file:
```bash
sbatch --gpus=rtx_5090:1 SocialNetwork/runandcompare_A.sbatch
sbatch --gpus=rtx_5090:1 SocialNetwork/runandcompare_B.sbatch
```

Both jobs together should finish in ~5 minutes (3 quarters × 2 edge cols × 2
models per job, mostly GAT — SAGE is already done).

**Verification after retry**: the metrics CSV row for each affected quarter
should have non-null `gat_acc`; the cusip_scores parquet should gain rows
for those `(year, quarter, model="gat")` triples (atomic upsert via
`append_cusip_scores` already handles the replace).
