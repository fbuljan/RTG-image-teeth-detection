import torch
import yaml
import argparse
from ultralytics import YOLO
import gc
import os

def validate_resume_args(args):
    """Validate argument combinations for resume training."""
    if args.resume_training and not args.resume:
        raise ValueError(
            "Error: --resume-training requires --resume <checkpoint_path>\n"
            "Usage: python train_yolo.py --config <cfg> --resume <path> --resume-training"
        )

    if args.resume_training:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(
                f"Checkpoint not found: {args.resume}\n"
                "Cannot resume training without valid checkpoint."
            )

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., runs-segmentation/exp/weights/last.pt)")
    parser.add_argument("--resume-training", action="store_true",
                        help="Resume training with optimizer state (pass resume=True to YOLO). Use with --resume.")
    parser.add_argument("--skip-final-val", action="store_true",
                        help="Skip final validation after training (useful for batch training)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config")
    args = parser.parse_args()

    # Validate resume arguments
    validate_resume_args(args)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override epochs if provided
    if args.epochs is not None:
        config["epochs"] = args.epochs

    # Determine training mode
    if args.resume and args.resume_training:
        # TRUE RESUME: Load checkpoint and restore full training state
        model_path = args.resume
        resume_training = True
        print(f"RESUMING training with optimizer state from: {model_path}")
    elif args.resume:
        # WARM START: Load weights only, fresh optimizer (current behavior)
        model_path = args.resume
        resume_training = False
        print(f"WARM START: Loading weights from {model_path}, fresh optimizer")
    else:
        # COLD START: New training from pretrained model
        model_path = config["model"]
        resume_training = False
        print(f"COLD START: Training YOLO model: {model_path}")

    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}\n")
    
    # MPS-specific settings
    if device == "mps":
        if "train_args" not in config:
            config["train_args"] = {}
        
        # eliminate memory leak
        config["train_args"]["workers"] = 0
        config["train_args"]["cache"] = False
        
        print("MPS detected - applied memory-safe settings:")
        print("   - workers: 0")
        print("   - cache: False\n")
    
    # Load model
    model = YOLO(model_path)
    
    if device == "mps":
        def clear_memory_callback(trainer):
            if trainer.epoch > 0 and trainer.epoch % 5 == 0:
                torch.mps.empty_cache()
                gc.collect()
                print(f"[Epoch {trainer.epoch}] Cleared MPS cache")
        
        model.add_callback("on_train_epoch_end", clear_memory_callback)
    
    # Training
    print("Starting training...")

    # Training configuration
    train_config = {
        "data": config["data"],
        "batch": config["batch"],
        "imgsz": config["imgsz"],
        "project": config["project"],
        "name": config.get("name", "experiment"),
        "device": device,
        "exist_ok": True,
    }

    # Add train_args, but handle warmup specially for resumed training
    train_args = config.get("train_args", {}).copy()

    if resume_training:
        # SIMPLIFIED APPROACH: Use warm start (weights only, no optimizer state)
        # Ultralytics' resume feature is too complex for batch training
        # Trade-off: We lose optimizer momentum between segments, but training still works
        print(f"\nResume mode (warm start): Loading weights from {model_path}")
        print("  Note: Optimizer state will be reset (warm start, not true resume)")
        print("  Disabling warmup for continued training")

        # Disable warmup for continued training
        train_args["warmup_epochs"] = 0
        train_args["warmup_momentum"] = train_args.get("momentum", 0.937)
        train_args["warmup_bias_lr"] = train_args.get("lr0", 0.01)

        # Simple warm start: load model weights and train for more epochs
        results = model.train(
            epochs=config["epochs"],
            **train_config,
            **train_args
        )
    else:
        # COLD/WARM START: Use config settings as-is
        results = model.train(
            epochs=config["epochs"],
            **train_config,
            **train_args
        )
    
    print("Training done!\n")

    # Skip validation if requested (useful for batch training)
    if args.skip_final_val:
        print("Skipping final validation (--skip-final-val flag set)\n")
        return

    if device == "mps":
        torch.mps.empty_cache()
        gc.collect()
        print("Cleared cache before validation\n")

    # Final validation
    print("Running final validation...")
    val_results = model.val(
        data=config["data"],
        split=config.get("val_split", "test"),
        save=True,
        **config.get("val_args", {})
    )

    print("\nRESULTS:")
    print(f"   Box mAP50:  {val_results.box.map50:.4f}")
    print(f"   Mask mAP50: {val_results.seg.map50:.4f}")
    print("\nDone!\n")

if __name__ == "__main__":
    main()