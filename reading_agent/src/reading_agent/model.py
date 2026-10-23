"""The reading agent: tabular encoder -> stacked recurrent state -> pointer policy / anytime predictor."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedCell(nn.Module):
    """Multi-layer RNN unrolled one step at a time (needed because the policy picks the next input)."""

    def __init__(self, d, layers=2, cell="gru", dropout=0.2):
        super().__init__()
        cls = {"gru": nn.GRUCell, "rnn": nn.RNNCell, "lstm": nn.LSTMCell}[cell]
        self.cells = nn.ModuleList([cls(d, d) for _ in range(layers)])
        self.cell, self.layers, self.drop = cell, layers, nn.Dropout(dropout)

    def init(self, h0):  # h0: (layers, B, d)
        if self.cell == "lstm":
            return [(h0[l], torch.zeros_like(h0[l])) for l in range(self.layers)]
        return [h0[l] for l in range(self.layers)]

    def top(self, state):
        s = state[-1]
        return s[0] if self.cell == "lstm" else s

    def step(self, state, x):
        new, inp = [], x
        for l, cell in enumerate(self.cells):
            s = cell(inp, state[l])
            new.append(s)
            inp = self.drop(s[0] if self.cell == "lstm" else s)
        return new

    @staticmethod
    def blend(old, new, keep):  # keep: (B,1) bool -> use old where keep
        if isinstance(old[0], tuple):
            return [tuple(torch.where(keep, o, n) for o, n in zip(os_, ns)) for os_, ns in zip(old, new)]
        return [torch.where(keep, o, n) for o, n in zip(old, new)]


def mlp(i, h, o, dropout=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o))


class ReadingAgent(nn.Module):
    def __init__(self, n_num, cat_cards, col_name_emb, field_name_emb, n_out,
                 d=128, layers=2, cell="gru", dropout=0.2, max_items=8, e5_dim=384):
        super().__init__()
        self.d, self.n_num, self.n_cat, self.layers = d, n_num, len(cat_cards), layers
        self.register_buffer("col_name_emb", torch.as_tensor(col_name_emb, dtype=torch.float32))
        self.register_buffer("field_name_emb", torch.as_tensor(field_name_emb, dtype=torch.float32)
                             if len(field_name_emb) else torch.zeros(1, e5_dim))
        self.name_proj = nn.Linear(e5_dim, d)
        # tabular value encoders
        self.num_w = nn.Parameter(torch.randn(max(n_num, 1), d) * 0.1)
        self.num_miss = nn.Parameter(torch.zeros(max(n_num, 1), d))
        offsets = [0]
        for c in cat_cards:
            offsets.append(offsets[-1] + c)
        self.register_buffer("cat_off", torch.tensor(offsets[:-1], dtype=torch.long))
        self.cat_emb = nn.Embedding(max(offsets[-1], 1), d)
        n_tab = n_num + self.n_cat
        self.tab_mlp = mlp(max(n_tab, 1) * d, 2 * d, d, dropout) if n_tab else None
        self.tab_null = nn.Parameter(torch.zeros(d))
        self.tab_norm = nn.LayerNorm(d)
        # text items
        self.item_proj = nn.Linear(e5_dim, d)
        self.pos_emb = nn.Embedding(max_items, d)
        self.item_norm = nn.LayerNorm(d)
        # recurrent state, initialised from the table (encoder -> initial hidden state)
        self.h0_proj = nn.Linear(d, layers * d)
        self.rnn = StackedCell(d, layers, cell, dropout)
        # heads on s_t = [h_top ; mean(read items) ; h_tab]
        self.predictor = mlp(3 * d, 2 * d, n_out, dropout)
        self.q = nn.Linear(3 * d, d)
        self.k = nn.Linear(d, d)
        self.k_stop = nn.Parameter(torch.randn(d) * 0.1)
        self.value = mlp(3 * d, d, 1, dropout)
        # expected-gain head: predicts E[loss reduction | state, item] (label-free value of information)
        self.key_mode = "content"
        self.meta_key = nn.Linear(e5_dim, d)
        self.gain = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1))

    def item_keys(self, b, items):
        """What the policy may look at *before* reading an item: its content ('content') or only which
        field / position it is ('meta' = true acquisition setting)."""
        if self.key_mode == "content":
            return items
        return self.meta_key(self.field_name_emb)[b["item_field"]] + self.pos_emb(b["item_pos"])

    def gain_scores(self, s, keys):
        """(B, K) predicted loss reduction from reading each item next."""
        K = keys.shape[1]
        return self.gain(torch.cat([s.unsqueeze(1).expand(-1, K, -1), keys], -1)).squeeze(-1)

    # ---- encoders -------------------------------------------------------
    def encode(self, b):
        B = b["item_mask"].shape[0]
        toks = []
        names = self.name_proj(self.col_name_emb) if self.col_name_emb.numel() else None
        if self.n_num:
            toks.append(names[: self.n_num] + b["x_num"].unsqueeze(-1) * self.num_w + b["m_num"].unsqueeze(-1) * self.num_miss)
        if self.n_cat:
            toks.append(names[self.n_num:] + self.cat_emb(b["x_cat"] + self.cat_off))
        if toks:
            h_tab = self.tab_norm(self.tab_mlp(torch.cat(toks, 1).flatten(1)))
        else:
            h_tab = self.tab_null.expand(B, -1)
        fields = self.name_proj(self.field_name_emb)[b["item_field"]]
        items = self.item_norm(self.item_proj(b["items"].float()) + fields + self.pos_emb(b["item_pos"]))
        return h_tab, items

    def init_state(self, h_tab):
        h0 = torch.tanh(self.h0_proj(h_tab)).view(-1, self.layers, self.d).transpose(0, 1).contiguous()
        return self.rnn.init(h0)

    def summary(self, state, read_sum, read_cnt, h_tab):
        mean_read = read_sum / read_cnt.clamp(min=1)
        return torch.cat([self.rnn.top(state), mean_read, h_tab], -1)

    def policy_logits(self, s, items, avail):
        """Pointer attention over unread items + a learned STOP key. Returns (B, K+1); index K = STOP."""
        q = self.q(s)
        keys = torch.cat([self.k(items), self.k_stop.expand(items.shape[0], 1, -1)], 1)
        logits = torch.einsum("bd,bkd->bk", q, keys) / math.sqrt(self.d)
        mask = torch.cat([avail, torch.ones_like(avail[:, :1])], 1)
        return logits.masked_fill(mask == 0, float("-inf"))

    # ---- teacher-forced pass over a fixed reading order -------------------
    def forward_order(self, b, order, n_read):
        """Read items in `order` (B,K); returns predictions after 0..K reads: (K+1, B, n_out).
        Rows stop updating after n_read[b] items."""
        h_tab, items = self.encode(b)
        B, K, d = items.shape
        state = self.init_state(h_tab)
        read_sum = items.new_zeros(B, d)
        read_cnt = items.new_zeros(B, 1)
        preds = [self.predictor(self.summary(state, read_sum, read_cnt, h_tab))]
        for t in range(K):
            active = (n_read > t).float().unsqueeze(1)
            x = items[torch.arange(B), order[:, t]]
            new = self.rnn.step(state, x)
            state = StackedCell.blend(state, new, active == 0)
            read_sum = read_sum + active * x
            read_cnt = read_cnt + active
            preds.append(self.predictor(self.summary(state, read_sum, read_cnt, h_tab)))
        return torch.stack(preds)


class SoftAttentionReader(nn.Module):
    """Baseline: table-conditioned soft attention over *all* items (no reading decisions)."""

    def __init__(self, agent_kwargs, heads=4):
        super().__init__()
        self.enc = ReadingAgent(**agent_kwargs)
        d = self.enc.d
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
        self.head = mlp(3 * d, 2 * d, agent_kwargs["n_out"])

    def forward(self, b):
        h_tab, items = self.enc.encode(b)
        pad = b["item_mask"] == 0
        all_pad = pad.all(1)
        pad = pad.clone()
        pad[all_pad, 0] = False  # avoid NaN rows; zeroed below
        a, w = self.attn(h_tab.unsqueeze(1), items, items, key_padding_mask=pad)
        a = a.squeeze(1) * (~all_pad).float().unsqueeze(1)
        m = (items * b["item_mask"].unsqueeze(-1)).sum(1) / b["item_mask"].sum(1, keepdim=True).clamp(min=1)
        return self.head(torch.cat([a, m, h_tab], -1)), w.squeeze(1)
