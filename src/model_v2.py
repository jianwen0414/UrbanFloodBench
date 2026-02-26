"""
model_v2 — SAGEConv Unified Engine (V2).

Based on V1's proven SAGEConv architecture but with V2's improvements:
- Input skip connection: decoder receives [GRU_output; raw_features]
- Configurable heads (not used for SAGEConv, kept for API compat)
- Deeper network support (4+ layers)

Replaces the GATv2Conv prototype which was too memory-heavy for 4GB VRAM.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HeteroConv, SAGEConv, LayerNorm

_PIPE_ET = ("1d", "pipe", "1d")
_SPREAD_ET = ("2d", "spread", "2d")
_LINK_12_ET = ("1d", "link", "2d")
_LINK_21_ET = ("2d", "link", "1d")

class UnifiedGATModel(nn.Module):
    """Heterogeneous GNN-GRU predicting delta-depth.
    
    Named UnifiedGATModel for backward compatibility but uses SAGEConv
    internally to fit within 4GB VRAM constraints.
    """
    def __init__(
        self,
        in_channels_1d: int,
        in_channels_2d: int,
        hidden_channels: int = 192,
        heads: int = 4,         # kept for API compat, not used by SAGEConv
        num_gnn_layers: int = 3,
        dropout: float = 0.05,
        **kwargs: Any,
    ):
        super().__init__()
        self.hidden = hidden_channels
        self.in_channels_1d = in_channels_1d
        self.in_channels_2d = in_channels_2d
        
        # Encoders
        self.enc_1d = nn.Sequential(
            nn.Linear(in_channels_1d, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.01)
        )
        self.enc_2d = nn.Sequential(
            nn.Linear(in_channels_2d, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.01)
        )
        
        # SAGEConv Layers (memory-efficient, proven in V1)
        self.convs = nn.ModuleList()
        self.norms_1d = nn.ModuleList()
        self.norms_2d = nn.ModuleList()
        
        for _ in range(num_gnn_layers):
            conv = HeteroConv({
                _PIPE_ET: SAGEConv(hidden_channels, hidden_channels),
                _SPREAD_ET: SAGEConv(hidden_channels, hidden_channels),
                _LINK_12_ET: SAGEConv((hidden_channels, hidden_channels), hidden_channels),
                _LINK_21_ET: SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            }, aggr="sum")
            self.convs.append(conv)
            self.norms_1d.append(LayerNorm(hidden_channels))
            self.norms_2d.append(LayerNorm(hidden_channels))
            
        self.dropout = nn.Dropout(dropout)
        
        # GRU
        self.gru_1d = nn.GRUCell(hidden_channels, hidden_channels)
        self.gru_2d = nn.GRUCell(hidden_channels, hidden_channels)
        self.gru_norm_1d = nn.LayerNorm(hidden_channels)
        self.gru_norm_2d = nn.LayerNorm(hidden_channels)
        
        # Decoder with input skip connection: [GRU_output; raw_features]
        self.head_1d = nn.Sequential(
            nn.Linear(hidden_channels + in_channels_1d, hidden_channels // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_channels // 2, 1)
        )
        self.head_2d = nn.Sequential(
            nn.Linear(hidden_channels + in_channels_2d, hidden_channels // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m, nn.GRUCell):
            nn.init.orthogonal_(m.weight_hh)
            nn.init.xavier_uniform_(m.weight_ih)
            
    def init_hidden(self, n_1d: int, n_2d: int, device: torch.device):
        return {
            "1d": torch.zeros(n_1d, self.hidden, device=device),
            "2d": torch.zeros(n_2d, self.hidden, device=device)
        }
        
    def forward(self, x_dict, edge_index_dict, hidden_dict=None):
        if hidden_dict is None:
            hidden_dict = self.init_hidden(x_dict["1d"].size(0), x_dict["2d"].size(0), x_dict["1d"].device)
            
        z = {
            "1d": self.enc_1d(x_dict["1d"]),
            "2d": self.enc_2d(x_dict["2d"])
        }
        
        for i, conv in enumerate(self.convs):
            out = conv(z, edge_index_dict)
            
            # Residual + Norm
            for k in ["1d", "2d"]:
                if k in out:
                    h = out[k] + z[k]  # Residual
                    h = self.norms_1d[i](h) if k == "1d" else self.norms_2d[i](h)
                    h = F.leaky_relu(h, 0.01)
                    h = self.dropout(h)
                    z[k] = h
        
        # GRU Update
        h_1d = self.gru_1d(z["1d"], hidden_dict["1d"])
        h_2d = self.gru_2d(z["2d"], hidden_dict["2d"])
        
        # Decode: [GRU_Normed; Raw Input]
        cat_1d = torch.cat([self.gru_norm_1d(h_1d), x_dict["1d"]], dim=-1)
        cat_2d = torch.cat([self.gru_norm_2d(h_2d), x_dict["2d"]], dim=-1)
        
        delta_1d = self.head_1d(cat_1d).squeeze(-1)
        delta_2d = self.head_2d(cat_2d).squeeze(-1)
        
        return {"1d": delta_1d, "2d": delta_2d}, {"1d": h_1d, "2d": h_2d}
