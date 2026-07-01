import torch
import torch.nn as nn
import torch.nn.functional as F

class HypergraphFeatureExtractor(nn.Module):
    """
    Hypergraph construction over channels in a sequence X [F, M].
    Hyperedges = motif prototypes (learned).
    Output will be Zw [T, d] where T is the number of windows.

    Options
    -------
    K:         # prototypes for motif edges (0 disables motif edges)
    tau:       temperature for soft motif assignments
    time_decay: float in (0, ∞]; if set, edges get weight * exp(-|Δ|/time_decay)
    use_exact_norms: recompute degrees on the fly (slower, adaptive); else fixed per template
    """
    def __init__(
        self,
        M: int,
        d: int,
        out_dim: int,
        K: int = 8,
        num_conv: int = 1,
        tau: float = 0.2,
        time_decay: float = None,
        eps: float = 1e-6,
        activity_gate: bool = True,
        identity: bool = False,
    ):
        super().__init__()
        self.M = M
        self.d = d
        self.out_dim = out_dim
        self.K = K
        self.num_conv = num_conv
        self.tau = tau
        self.time_decay = time_decay
        self.eps = eps
        self.activity_gate = activity_gate
        self.identity = identity

        # Motif prototypes
        if K > 0:
            self.channel_emb = torch.nn.Parameter(torch.randn(self.M, self.d) * (1.0 / self.d**0.5))
            self.prototypes  = torch.nn.Parameter(torch.randn(self.K, self.d) * (1.0 / self.d**0.5))
            self.gamma = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("prototypes", None)
        
        if self.identity == True:
            self.Theta = nn.Identity()
        else:
            self.Theta = nn.Sequential(
                nn.Linear(M, out_dim),
                nn.LeakyReLU(),
                nn.Linear(out_dim, out_dim),
            )

        # Edge weights (per edge type block)
        self.w_motif = nn.Parameter(torch.tensor(1.0)) if K > 0 else None


    def _edges_motif_intra(self, Z, device):
        """
        Intra-HG soft assignment: channels (AUs) -> K prototypes.
        Z: [T, M]  (window features: time x channels)
        Returns:
            H_t: [T, M, K]  soft incidence over AUs per window
            w:   [K]        motif edge weights
        """
        if self.K <= 0:
            return None, None

        T, M = Z.shape
        assert M == self.M, f"Z has {M} channels, expected {self.M}"

        # 1) AU→prototype memberships (shared across time)
        Cn = F.normalize(self.channel_emb, dim=-1)   # [M, d]
        Pn = F.normalize(self.prototypes,  dim=-1)   # [K, d]
        S  = Cn @ Pn.t()                              # [M, K]
        A  = F.softmax(S / self.tau, dim=-1)          # [M, K] rows sum to 1

        # 2) Optional per-window activity gating (time × channel) -> AUs contibution to prototypes
        if getattr(self, "activity_gate", True):
            G = torch.sigmoid(self.gamma * (Z - Z.mean(dim=0, keepdim=True)))   # [T, M]
            H_t = G.unsqueeze(-1) * A.unsqueeze(0)                               # [T, M, K]
        else:
            H_t = A.unsqueeze(0).expand(T, -1, -1).contiguous()                  # [T, M, K]

        # 3) Motif edge weights
        if isinstance(self.w_motif, torch.nn.Parameter):
            w = self.w_motif
            if w.ndim == 0:  # scalar -> broadcast
                w = torch.full((self.K,), float(self.w_motif.item()), device=device)
            else:
                assert w.shape == (self.K,), "w_motif must be scalar or shape [K]"
        else:
            w = torch.full((self.K,), float(self.w_motif), device=device)

        return H_t, w


    def _per_window_norms(self, H_t: torch.Tensor, w_e: torch.Tensor):
        """
        H_t: [T, M, E] -> De_inv_t: [T, E], Dv_inv_sqrt_t: [T, M]
        """
        T, M, E = H_t.shape
        De_t = torch.clamp(H_t.sum(dim=1), min=self.eps)                  # [T, E]
        # Dv_t(i) = sum_e H_t(i,e) * w_e
        Dv_t = torch.clamp(torch.einsum("tme,e->tm", H_t, w_e), min=self.eps)  # [T, M]
        De_inv_t = 1.0 / De_t
        Dv_inv_sqrt_t = 1.0 / torch.sqrt(Dv_t)
        return De_inv_t, Dv_inv_sqrt_t
    
    def _hypergraph_conv(self, x: torch.Tensor, H_t: torch.Tensor, w_e: torch.Tensor):                                                  # [T, M]
        # Exact per-window norms
        De_inv_t, Dv_inv_sqrt_t = self._per_window_norms(H_t, w_e)   # [T,E], [T,M]

        # m_e = Dv^{-1/2} * H_t^T * x  (implemented as H_t^T (Dv^{-1/2} ∘ x))
        x_tilde = Dv_inv_sqrt_t * x                              # [T, M]
        # edge agg: sum over nodes i: H_t(i,e) * x_tilde(i)
        m_e = torch.einsum("tme,tm->te", H_t, x_tilde)          # [T, E]

        # edge weights and degree normalization
        m_e = m_e * w_e.unsqueeze(0) * De_inv_t            # [T, E]

        # back to nodes: x' = Dv^{-1/2} ∘ (H_t * m_e)
        x_prime = torch.einsum("tme,te->tm", H_t, m_e)          # [T, M]
        x_prime = Dv_inv_sqrt_t * x_prime 
        
        edge_factor = (w_e * De_inv_t)            # [T,E]
        edge_msg = edge_factor * m_e               # [T,E]
        per_au_sum = torch.einsum("tme,te->tm", H_t, edge_msg)  # [T,M]
        contrib = Dv_inv_sqrt_t * per_au_sum         # [T,M]
            
        return x_prime     

    # ---------- forward ----------
    def forward(self, X: torch.Tensor):
        """
        X: [f, M] AU time series for a single video
        Returns:
            Z_w: [T, d] window embeddings (instances for MIL)
        """
        assert X.dim() == 2 and X.shape[1] == self.M, f"X must be [F,{self.M}]"
        f, M = X.shape
        device = X.device

        Z = X

        # Build edge blocks
        H, w_e = self._edges_motif_intra(Z, device)

        if H is None:
            return self.Theta(Z)

        Z_w = Z
        for _ in range(self.num_conv):
            Z_w = self._hypergraph_conv(Z_w, H, w_e)    # [T, M]
            
        # Final projection to window embedding space
        Z_w = self.Theta(Z_w)                         # [T, d]
        
        return Z_w 
