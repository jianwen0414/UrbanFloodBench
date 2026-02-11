"""
Production Training Script — Full 60-epoch run with both models.

Usage:
    python train_production.py
    python train_production.py --epochs 80 --hidden_channels 128
    python train_production.py --resume checkpoints/checkpoint_epoch_20.pt

This runs outside the notebook to avoid kernel timeouts and memory
issues during long training runs.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_PATH
from src.trainer import TrainConfig, UnifiedTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Production training for UnifiedFloodModel"
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden_channels", type=int, default=192)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--num_gru_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--val_event_id", type=str, default="3,9,15",
                        help="Comma-separated val event IDs (must exist in BOTH models)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed precision (for CPU training)")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=str(RAW_DATA_PATH),
        model_ids=["1", "2"],           # Both urban models
        val_event_id=args.val_event_id,

        # Architecture
        hidden_channels=args.hidden_channels,
        num_gnn_layers=args.num_gnn_layers,
        num_gru_layers=args.num_gru_layers,
        dropout=args.dropout,

        # Training
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-5,
        grad_clip_norm=1.0,            # standard; let BPTT gradients flow
        batch_accumulation=1,

        # ── Scheduled Sampling ────────────────────────────────────
        # Very short warmup (3 epochs) forces the model to face its
        # own errors early.  Previous runs showed that models trained
        # with long TF warmup (10 epochs) achieve low train loss but
        # completely collapse during autoregressive validation.
        tf_warmup_epochs=3,
        tf_decay_epochs=30,
        tf_min_ratio=0.0,

        # ── Push-Forward + Progressive K Curriculum ───────────────
        # v4: Increased K target from 20 → 30 to better match the
        # 40+ step validation rollout.  With depth-based targets
        # and gradient flow (no detach), the model can handle it.
        pushforward_K=30,              # target K (reached after K_ramp_epochs)
        use_push_forward=True,
        temporal_scheme="linear",
        progressive_K=True,
        K_start=3,                     # start with short trajectories
        K_ramp_epochs=30,              # reach full K by epoch 30

        # ── Randomized Spinup ─────────────────────────────────────
        randomize_spinup=True,
        spinup_min=3,
        spinup_max=10,

        # ── Noise Injection — DISABLED ────────────────────────────
        training_noise_std=0.0,

        # ── Loss ──────────────────────────────────────────────────
        loss_variant="huber",          # outlier-robust
        huber_delta=0.3,               # v4: tighter from 0.5 → 0.3 for depth targets
        clamp_weights=5.0,
        alpha=0.5,

        # ── LR Scheduler ─────────────────────────────────────────
        # v4: Plain cosine decay.  CosineWarmRestarts caused
        # destructive LR resets at epoch 15 & 45 that undid
        # learned dynamics.  Single smooth decay is safer.
        scheduler="cosine",
        cosine_eta_min=1e-6,

        # ── Early Stopping ────────────────────────────────────────
        early_stop_patience=30,
        early_stop_min_delta=1e-5,

        # ── Mixed Precision ──────────────────────────────────────
        use_amp=False,

        # ── Checkpointing ────────────────────────────────────────
        checkpoint_dir="experiments/checkpoints",
        save_every_n_epochs=5,
        save_best=True,

        # ── Logging ──────────────────────────────────────────────
        log_dir="logs",
        verbose=True,

        # ── Device ───────────────────────────────────────────────
        device=args.device,
    )

    print("=" * 60)
    print("  PRODUCTION TRAINING (v4 — Depth-Based Targets)")
    print("=" * 60)
    print(f"  Epochs          : {cfg.epochs}")
    print(f"  Hidden channels : {cfg.hidden_channels}")
    print(f"  GNN layers      : {cfg.num_gnn_layers}")
    print(f"  GRU layers      : {cfg.num_gru_layers}")
    print(f"  LR              : {cfg.lr}")
    print(f"  Models          : {cfg.model_ids}")
    print(f"  Val events      : {cfg.val_event_id}")
    print(f"  AMP             : {cfg.use_amp}")
    print(f"  Push-forward K  : {cfg.K_start} → {cfg.pushforward_K} (over {cfg.K_ramp_epochs} epochs)")
    print(f"  Scheduler       : {cfg.scheduler}")
    print(f"  Resume from     : {args.resume or 'scratch'}")
    print("=" * 60)

    trainer = UnifiedTrainer(cfg)
    trainer.setup()
    history = trainer.train(resume_from=args.resume)

    print("\n" + history.summary_str())
    print(f"\nBest checkpoint saved to: {cfg.checkpoint_dir}/best_model.pt")
    print("Done! Now run validation_run.ipynb to evaluate.")


if __name__ == "__main__":
    main()
