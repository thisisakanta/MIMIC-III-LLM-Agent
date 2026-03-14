import torch
import torch.nn as nn


class NeurologyGRUBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]


class CardiologyCNNBranch(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, emb_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.squeeze(-1)


class RespiratoryGRUAttentionBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        scores = self.attn(out)
        w = torch.softmax(scores, dim=1)
        ctx = (w * out).sum(dim=1)
        return ctx


class GRUDCell(nn.Module):
    """PyTorch GRU-D cell using value, mask, and time-gap inputs."""

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Learnable global empirical mean per feature for missing-value decay.
        self.x_mean = nn.Parameter(torch.zeros(feature_dim))

        # Decay terms for input and hidden state.
        self.gamma_x_weight = nn.Parameter(torch.zeros(feature_dim))
        self.gamma_x_bias = nn.Parameter(torch.zeros(feature_dim))
        self.gamma_h = nn.Linear(feature_dim, hidden_dim)

        # GRU gates with additional masking pathway.
        self.x_z = nn.Linear(feature_dim, hidden_dim)
        self.h_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.m_z = nn.Linear(feature_dim, hidden_dim, bias=False)

        self.x_r = nn.Linear(feature_dim, hidden_dim)
        self.h_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.m_r = nn.Linear(feature_dim, hidden_dim, bias=False)

        self.x_h = nn.Linear(feature_dim, hidden_dim)
        self.h_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.m_h = nn.Linear(feature_dim, hidden_dim, bias=False)

    def _decay_x(self, delta):
        return torch.exp(-torch.relu(delta * self.gamma_x_weight + self.gamma_x_bias))

    def _decay_h(self, delta):
        return torch.exp(-torch.relu(self.gamma_h(delta)))

    def forward(self, x_t, m_t, d_t, h_prev, x_last):
        gamma_x = self._decay_x(d_t)
        gamma_h = self._decay_h(d_t)

        x_hat = gamma_x * x_last + (1.0 - gamma_x) * self.x_mean.unsqueeze(0)
        x_t = m_t * x_t + (1.0 - m_t) * x_hat
        h_prev = gamma_h * h_prev

        z_t = torch.sigmoid(self.x_z(x_t) + self.h_z(h_prev) + self.m_z(m_t))
        r_t = torch.sigmoid(self.x_r(x_t) + self.h_r(h_prev) + self.m_r(m_t))
        h_tilde = torch.tanh(self.x_h(x_t) + self.h_h(r_t * h_prev) + self.m_h(m_t))
        h_t = z_t * h_prev + (1.0 - z_t) * h_tilde

        x_last = m_t * x_t + (1.0 - m_t) * x_last
        return h_t, x_last


class MetabolicGRUDLikeBranch(nn.Module):
    """GRU-D encoder over [values, masks, deltas]."""

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        if input_dim % 3 != 0:
            raise ValueError(
                "Metabolic branch expects concatenated [values, masks, deltas], "
                f"so input_dim must be divisible by 3. Got {input_dim}."
            )
        self.feature_dim = input_dim // 3
        self.hidden_dim = hidden_dim
        self.cell = GRUDCell(self.feature_dim, hidden_dim)

    def forward(self, x):
        x_val, x_mask, x_delta = torch.split(x, self.feature_dim, dim=-1)
        batch_size = x.shape[0]
        h = x.new_zeros(batch_size, self.hidden_dim)
        x_last = self.cell.x_mean.unsqueeze(0).expand(batch_size, -1).clone()

        for t in range(x.shape[1]):
            h, x_last = self.cell(
                x_val[:, t, :],
                x_mask[:, t, :],
                x_delta[:, t, :],
                h,
                x_last,
            )
        return h


class StaticMLPBranch(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, emb_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiBranchMortalityModel(nn.Module):
    def __init__(
        self,
        neuro_dim: int,
        cardio_dim: int,
        resp_dim: int,
        meta_dim: int,
        static_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.neuro = NeurologyGRUBranch(neuro_dim, hidden_dim=64)
        self.cardio = CardiologyCNNBranch(cardio_dim, emb_dim=64)
        self.resp = RespiratoryGRUAttentionBranch(resp_dim, hidden_dim=64)
        self.meta = MetabolicGRUDLikeBranch(meta_dim, hidden_dim=32)
        self.static = StaticMLPBranch(static_dim, emb_dim=16)

        fusion_in = 64 + 64 + 64 + 32 + 16
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, neuro, cardio, resp, meta, static):
        e_neuro = self.neuro(neuro)
        e_cardio = self.cardio(cardio)
        e_resp = self.resp(resp)
        e_meta = self.meta(meta)
        e_static = self.static(static)

        z = torch.cat([e_neuro, e_cardio, e_resp, e_meta, e_static], dim=1)
        return self.fusion(z).squeeze(-1)
