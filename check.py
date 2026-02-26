import json
import torch
from src.config import RAW_DATA_PATH
from src.dataset import FloodDataset
ds = FloodDataset(str(RAW_DATA_PATH), mode="train")
stds = ds.compute_node_stds(model_id="2")
s1 = torch.tensor(stds["2"]["1d"])
s2 = torch.tensor(stds["2"]["2d"])
print(f"1D active nodes: {(s1 >= 0.01).sum()} / {len(s1)}")
print(f"2D active nodes: {(s2 >= 0.01).sum()} / {len(s2)}")
