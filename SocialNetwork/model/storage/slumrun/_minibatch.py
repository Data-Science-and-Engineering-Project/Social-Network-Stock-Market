"""Mini-batch (NeighborLoader) replacements for _common's train/eval/score.

Why this exists
---------------
Full-graph training (model(data.x, data.edge_index)) blows up GAT on the
multi-quarter graphs (multiq_A / multiq_B): even a 24GB GPU OOMs because GAT
materialises attention coefficients for every edge on every layer. Switching
to neighbour sampling caps each step's memory at O(batch_size * fanout^L)
instead of O(num_nodes + num_edges), which empirically takes peak GPU usage
from ~18GB → 2-4GB without changing the model architecture.

What stays the same
-------------------
- WeightedSAGE / WeightedGAT (and their weighted v2 variants) are reused as-is.
- forward_fn(model, batch) signature is identical — a NeighborLoader batch is
  also a torch_geometric.data.Data, so model(batch.x, batch.edge_index, ...)
  works without changes.
- Metrics (accuracy, macro_f1, peak_gb, train_s) and CUSIP score parquet
  output are produced exactly as before — eval just runs through the loader
  instead of a single full-graph forward.

How patching works
------------------
_common.run_approach() looks up train_one / eval_subsets / compute_cusip_scores
in _common's own globals at call time. Reassigning those module attributes
here redirects every call inside run_approach to the mini-batch versions
without touching _common.py itself. Call patch_common() once before run_all().

Defaults
--------
- num_neighbors=[20, 10] matches num_layers=2 (one fanout per layer). 20/10
  is the standard PyG default for citation-style graphs and is plenty for our
  bipartite Δ-graphs where each fund touches ~hundreds of CUSIPs.
- batch_size=512 for training: small enough to fit GAT activations on a
  16-24GB GPU, large enough that 150 epochs finish in reasonable wall-time.
- batch_size=2048 for eval: no backward pass, so we can push more nodes per
  step.
- Override via env vars FGNN_FANOUT (e.g. "20,10"), FGNN_BS_TRAIN, FGNN_BS_EVAL.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import NeighborLoader

import SocialNetwork.src.Korkevados.storage.slumrun._common as _c


def _fanout():
    raw = os.environ.get("FGNN_FANOUT", "20,10")
    return [int(x) for x in raw.split(",") if x.strip()]


BS_TRAIN = int(os.environ.get("FGNN_BS_TRAIN", "512"))
BS_EVAL = int(os.environ.get("FGNN_BS_EVAL", "2048"))


def _predict_all(model, data, forward_fn):
    """Mini-batch inference over every node; returns fp32 logits on CPU.

    Eval is forward-only, so we use a larger batch_size than training. The
    loader iterates seed nodes (input_nodes=None ⇒ all nodes); each batch's
    first batch.batch_size rows correspond to the seeds, and batch.n_id maps
    them back to global node IDs.
    """
    loader = NeighborLoader(
        data,
        num_neighbors=_fanout(),
        batch_size=BS_EVAL,
        input_nodes=None,
        shuffle=False,
    )
    model.eval()
    n = data.num_nodes
    num_classes = _c.NUM_CLASSES
    out_logits = torch.zeros(n, num_classes, dtype=torch.float32)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(_c.DEVICE)
            with _c._autocast_ctx():
                logits = forward_fn(model, batch)
            bs = batch.batch_size
            global_ids = batch.n_id[:bs].cpu()
            out_logits[global_ids] = logits[:bs].float().cpu()
    return out_logits


def train_one_mb(model, data, train_mask, val_mask, *, forward_fn,
                 epochs=None, lr=_c.LR, verbose=False):
    """Mini-batch replacement for _common.train_one.

    - data stays on CPU; NeighborLoader streams batches to DEVICE.
    - Loss is computed only on the seed (train) nodes of each batch
      (batch.y[:batch.batch_size]) — the sampled k-hop neighbours are context
      for message passing, not training targets.
    - Validation runs a full mini-batch inference pass and tracks best
      val-acc state, mirroring the original full-graph behaviour.
    """
    if epochs is None:
        epochs = _c.EPOCHS
    model = model.to(_c.DEVICE)
    train_mask_cpu = train_mask.cpu().bool()
    val_mask_cpu = val_mask.cpu().bool()

    train_loader = NeighborLoader(
        data,
        num_neighbors=_fanout(),
        batch_size=BS_TRAIN,
        input_nodes=train_mask_cpu,
        shuffle=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_acc = 0.0
    best_state = None
    y_cpu = data.y.cpu()
    for ep in range(1, epochs + 1):
        model.train()
        last_loss = float("nan")
        for batch in train_loader:
            batch = batch.to(_c.DEVICE)
            opt.zero_grad(set_to_none=True)
            with _c._autocast_ctx():
                logits = forward_fn(model, batch)
                bs = batch.batch_size
                loss = F.cross_entropy(logits[:bs], batch.y[:bs])
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

        # Per-epoch eval. Full mini-batch inference; expensive for 150 epochs
        # but matches the original train_one's best-val tracking.
        if val_mask_cpu.any():
            logits_all = _predict_all(model, data, forward_fn)
            pred = logits_all.argmax(dim=-1)
            val_acc = (pred[val_mask_cpu] == y_cpu[val_mask_cpu]).float().mean().item()
        else:
            val_acc = 0.0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and ep % 25 == 0:
            print(f"  ep {ep:3d}  loss={last_loss:.4f}  val_acc={val_acc:.3f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_subsets_mb(model, data, mask, *, forward_fn):
    """Mini-batch replacement for _common.eval_subsets.

    Computes per-subset (all / stocks / investors) accuracy and macro-F1 from
    a single mini-batch inference pass.
    """
    logits = _predict_all(model, data, forward_fn)
    pred = logits.argmax(dim=-1)
    y = data.y.cpu()
    is_cik = data.is_cik.cpu()
    mask = mask.cpu()
    out = {}
    for label, sel in [
        ("all", mask),
        ("stocks", mask & (~is_cik)),
        ("investors", mask & is_cik),
    ]:
        if sel.sum() == 0:
            out[label] = {"n": 0}
            continue
        yt = y[sel].numpy()
        yp = pred[sel].numpy()
        out[label] = {
            "n": int(sel.sum()),
            "accuracy": accuracy_score(yt, yp),
            "macro_f1": f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
        }
    return out


def compute_cusip_scores_mb(model, data, meta, year, quarter, *, forward_fn):
    """Mini-batch replacement for _common.compute_cusip_scores.

    Runs full-coverage mini-batch inference, takes softmax over fp32 logits,
    and slices the CUSIP block (rows [n_cik : n_cik + n_cusip]) for ranking.
    """
    logits = _predict_all(model, data, forward_fn)
    probs = F.softmax(logits, dim=-1).numpy()
    n_cik = meta["n_cik"]
    cusip_ids = meta["cusip_ids"]
    stock_probs = probs[n_cik:n_cik + len(cusip_ids)]
    df = pd.DataFrame({
        "cusip":   list(cusip_ids),
        "year":    int(year),
        "quarter": int(quarter),
        "score":   stock_probs[:, 2],
    })
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def patch_common():
    """Redirect _common's full-graph train/eval helpers to the mini-batch ones.

    _common.run_approach calls train_one / eval_subsets / compute_cusip_scores
    as module-level names, so reassigning them on the _common module object
    is enough — no edits to _common.py.
    """
    _c.train_one = train_one_mb
    _c.eval_subsets = eval_subsets_mb
    _c.compute_cusip_scores = compute_cusip_scores_mb
    print(f"[mini-batch] patched _common — fanout={_fanout()} "
          f"bs_train={BS_TRAIN} bs_eval={BS_EVAL}", flush=True)
