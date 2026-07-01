import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence



# ---------- End-to-end HyperMIL (intra -> inter -> tiny MIL) ----------
class HyperMIL(nn.Module):
    """
    End-to-end HyperMIL:
      - Hypergraph feature extractor (intra-channel hg) : X_b [F_b, M] -> Z_w_b [T_b, d]
      - Tiny MIL attention head : padded to (B, T_max, d) + mask -> logits (B, C)

    Inputs:
      bag_feats: (B, T_raw, M_full)    # batch-padded raw frames/features
      lengths:  list[int]               # valid frame lengths F_b for each sample
    Outputs:
      logits: (B, C)
      aux: {
        "Zw_intra": (B, T_max, d),
        "Wmask":    (B, T_max) bool,
        "win_lengths": [T_b],
        "attn":     (B, T_max)
      }
    """
    def __init__(self, intra_hg: nn.Module, timemil: nn.Module):
        super().__init__()
        self.intra_hg = intra_hg
        self.timemil = timemil

    def forward(self, bag_feats: torch.Tensor, lengths:list, **timemil_kwargs):
        device = bag_feats.device
        B, T_raw, M_full = bag_feats.shape
        Zw_intra_list, Zw_inter_list, T_list = [], [], []
        
        if lengths == None: #equal-length dataset
            for b in range(B):
                X_b = bag_feats[b]
                Z_w_b = self.intra_hg(X_b)                      # [B, T, d]
                Zw_intra_list.append(Z_w_b)
    
            Zw_intra_pad = torch.stack(Zw_intra_list, dim=0)

            Mw = None

        else: #variable length dataset

            for b in range(B):
                X_b = bag_feats[b][:lengths[b]]                 # [F_b, M_full] strip outer padding
                Z_w_b = self.intra_hg(X_b)                      # [T_b, d]
                Zw_intra_list.append(Z_w_b)
                T_list.append(Z_w_b.size(0))

            # Pad to batch
            Zw_intra_pad = pad_sequence(Zw_intra_list, batch_first=True)  # (B, T_max, d)
            T_max = Zw_intra_pad.size(1)
            Mw = torch.zeros(B, T_max, dtype=torch.bool, device=device)
            for b, Tb in enumerate(T_list):
                Mw[b, :Tb] = True
                

        logits = self.timemil(Zw_intra_pad, Mw, **timemil_kwargs)  # -> (B, C)

        aux = {
            "Zw_intra": Zw_intra_pad,
            "Wmask": Mw,
            "win_lengths": T_list,
            }
        return logits, aux