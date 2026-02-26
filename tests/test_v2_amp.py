
import sys
import os
from pathlib import Path

# Fix path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import FloodDataset
from src.train_v2 import train_model_v2
from src.config import RAW_DATA_PATH

def test_amp_training():
    print("Testing AMP Training...")
    ds = FloodDataset(str(RAW_DATA_PATH), mode="train")
    
    # Run a very short training loop with AMP enabled
    # Small model to ensure it fits and fast
    try:
        train_model_v2(
            "2", ds, 
            epochs=1, 
            hidden_channels=32, 
            num_gnn_layers=2, 
            heads=2, 
            use_amp=True,
            checkpoint_dir="checkpoints_test_amp"
        )
        print("AMP Training Test Passed.")
    except Exception as e:
        print(f"AMP Training Test Failed: {e}")
        raise e

if __name__ == "__main__":
    test_amp_training()
