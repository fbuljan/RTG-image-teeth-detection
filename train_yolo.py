import torch
import yaml
import argparse
from ultralytics import YOLO
import gc

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g., runs-segmentation/exp/weights/last.pt)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override epochs if provided
    if args.epochs is not None:
        config["epochs"] = args.epochs

    # Determine model path (use resume checkpoint if provided)
    model_path = args.resume if args.resume else config["model"]
    print(f"Training YOLO model: {model_path}")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")

    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}\n")
    
    # MPS-specific settings - KLJUČNO!
    if device == "mps":
        if "train_args" not in config:
            config["train_args"] = {}
        
        # Ovo je najvažnije - eliminira memory leak
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

    # For batch training: don't use resume=True (it checks saved epoch target)
    # Instead, load weights from checkpoint and start fresh training for N more epochs
    # The model already has trained weights loaded from last.pt
    results = model.train(
        data=config["data"],
        epochs=config["epochs"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        project=config["project"],
        name=config.get("name", "experiment"),
        device=device,
        exist_ok=True,
        **config.get("train_args", {})
    )
    
    print("Training done!\n")
    
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