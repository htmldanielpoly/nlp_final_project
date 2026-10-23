"""Training for the reading agent.

Phase A  supervised warm-up: random reading orders and prefix losses train an "anytime" predictor.
Phase B  expected one-step gain regression with epsilon-greedy roll-ins; the predictor is frozen.
         The original oracle-imitation policy remains an optional experimental baseline.
"""
import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .model import ReadingAgent, StackedCell


@dataclass
class Config:
    d: int = 128
    layers: int = 2
    cell: str = "gru"
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 60
    patience: int = 6
    min_epochs: int = 0          # minimum completed warm-up epochs before patience may stop training
    early_stop_ema: float = 0.0  # previous EMA retention, [0, 1); 0 uses raw validation loss
    sched: str = "none"           # "plateau": halve lr when val loss stalls 2 epochs (noisy val curves on small data)
    full_weight: float = 0.5      # weight on the full-read loss vs. the mean over prefixes
    # policy
    policy_epochs: int = 15
    policy_lr: float = 3e-3
    dagger_beta0: float = 1.0     # prob. of following the oracle during roll-in (decays per epoch)
    dagger_decay: float = 0.7
    seed: int = 0

    def __post_init__(self):
        for name in ("epochs", "patience", "batch_size", "policy_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.min_epochs, bool)
                or not isinstance(self.min_epochs, (int, np.integer))
                or not 0 <= self.min_epochs <= self.epochs):
            raise ValueError("min_epochs must be an integer between 0 and epochs")
        if not math.isfinite(self.early_stop_ema) or not 0 <= self.early_stop_ema < 1:
            raise ValueError("early_stop_ema must be finite and in [0, 1)")
        for name in ("lr", "policy_lr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.full_weight) or not 0 <= self.full_weight <= 1:
            raise ValueError("full_weight must be finite and in [0, 1]")
        if self.sched not in ("none", "plateau"):
            raise ValueError("sched must be 'none' or 'plateau'")


def to_batch(p, idx=None):
    sl = slice(None) if idx is None else idx
    b = {
        "x_num": torch.from_numpy(p.x_num[sl]),
        "m_num": torch.from_numpy(p.m_num[sl]),
        "x_cat": torch.from_numpy(p.x_cat[sl]),
        "items": torch.from_numpy(p.items[sl].astype(np.float32)),
        "item_field": torch.from_numpy(p.item_field[sl]),
        "item_pos": torch.from_numpy(p.item_pos[sl]),
        "item_mask": torch.from_numpy(p.item_mask[sl]),
    }
    if p.y is not None:
        b["y"] = torch.from_numpy(np.asarray(p.y[sl]))
    return b


def sample_loss(pred, y, task):
    """Per-sample loss: cross-entropy (classification) or squared error on the standardised target."""
    if task == "regression":
        return (pred.squeeze(-1) - y.float()) ** 2
    return F.cross_entropy(pred, y.long(), reduction="none")


def random_order(mask, gen=None):
    """Random permutation of the real items, padding pushed to the end."""
    r = torch.rand(mask.shape, generator=gen) + (1 - mask) * 10
    return r.argsort(1)


def natural_order(mask):
    return (torch.arange(mask.shape[1]).expand_as(mask) + (1 - mask) * 100).argsort(1)


def build_model(pipe, n_out, cfg: Config):
    return ReadingAgent(n_num=pipe.n_num_in, cat_cards=pipe.cat_cardinalities,
                        col_name_emb=pipe.col_name_emb, field_name_emb=pipe.field_name_emb, n_out=n_out,
                        d=cfg.d, layers=cfg.layers, cell=cfg.cell, dropout=cfg.dropout, max_items=pipe.max_items)


# ---------------------------------------------------------------- Phase A
def prefix_loss(model, b, task, cfg, order):
    n_valid = b["item_mask"].sum(1).long()
    preds = model.forward_order(b, order, n_valid)               # (K+1, B, out)
    K1 = preds.shape[0]
    losses = torch.stack([sample_loss(preds[t], b["y"], task) for t in range(K1)])  # (K+1, B)
    valid = (torch.arange(K1).unsqueeze(1) <= n_valid.unsqueeze(0)).float()
    mean_prefix = (losses * valid).sum(0) / valid.sum(0)
    full = losses.gather(0, n_valid.unsqueeze(0)).squeeze(0)
    return ((1 - cfg.full_weight) * mean_prefix + cfg.full_weight * full).mean(), full


@torch.no_grad()
def eval_full_loss(model, prep, task, bs=1024):
    if len(prep) == 0:
        raise ValueError("validation data must contain at least one row")
    if isinstance(bs, bool) or not isinstance(bs, (int, np.integer)) or bs < 1:
        raise ValueError("validation batch size must be a positive integer")
    model.eval()
    tot = 0.0
    for s in range(0, len(prep), bs):
        b = to_batch(prep, slice(s, s + bs))
        n_valid = b["item_mask"].sum(1).long()
        preds = model.forward_order(b, natural_order(b["item_mask"]), n_valid)
        final = preds.gather(0, n_valid.view(1, -1, 1).expand(1, -1, preds.shape[-1])).squeeze(0)
        losses = sample_loss(final, b["y"], task)
        if not torch.isfinite(losses).all():
            raise RuntimeError("warm-up validation produced non-finite per-row losses")
        tot += losses.sum().item()
    return tot / len(prep)


def train_warmup(model, tr, va, task, cfg: Config, log=print):
    """Return the best raw validation loss and restore its checkpoint.

    EMA is optional and controls only the patience criterion, not checkpoint selection.
    ``min_epochs`` is a hard lower bound on completed epochs; stalled epochs before it
    still count towards patience. History and zero-based ``best_epoch`` are attached to
    the model for reporting/checkpoint manifests.
    """
    cfg.__post_init__()  # also catch invalid fields changed after Config construction
    if len(tr) == 0 or len(va) == 0:
        raise ValueError("warm-up training and validation data must both contain rows")
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, threshold=1e-4,
                                                        threshold_mode="abs", min_lr=cfg.lr / 32)
             if cfg.sched == "plateau" else None)
    patience = cfg.patience + (4 if sched else 0)  # leave room for the lr to drop before stopping
    best, best_state, bad = math.inf, None, 0
    best_stop, ema = math.inf, None
    model.training_history = []
    model.best_epoch = None
    model.best_val_loss = math.inf
    model.early_stopped = False
    n = len(tr)
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).numpy()
        tot = 0.0
        for s in range(0, n, cfg.batch_size):
            b = to_batch(tr, perm[s:s + cfg.batch_size])
            loss, _ = prefix_loss(model, b, task, cfg, random_order(b["item_mask"], gen))
            if not torch.isfinite(loss).all():
                raise RuntimeError(f"warm-up epoch {ep} produced a non-finite training loss")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # L6 p92
            opt.step()
            tot += loss.item() * len(b["y"])
        vl = eval_full_loss(model, va, task)
        if not math.isfinite(vl):
            raise RuntimeError(f"warm-up epoch {ep} produced a non-finite validation loss")
        if sched:
            sched.step(vl)
        raw_improved = vl < best
        if raw_improved:
            best, best_state = vl, copy.deepcopy(model.state_dict())
            model.best_epoch, model.best_val_loss = ep, best
        ema = vl if ema is None else cfg.early_stop_ema * ema + (1 - cfg.early_stop_ema) * vl
        if ema < best_stop - 1e-4:
            best_stop, bad = ema, 0
        else:
            bad += 1
        model.training_history.append(dict(epoch=ep, train_loss=tot / n, val_loss=vl,
                                           stop_loss=ema, lr=opt.param_groups[0]["lr"],
                                           checkpoint_selected=raw_improved))
        log(f"  warmup ep {ep:2d} train {tot / n:.4f} val {vl:.4f}{' *' if bad == 0 else ''}")
        if ep + 1 >= cfg.min_epochs and bad >= patience:
            model.early_stopped = True
            break
    if best_state is None:  # val loss never finite (e.g. exploding regression targets)
        raise RuntimeError("warm-up never produced a finite validation loss")
    model.load_state_dict(best_state)
    return best


# ---------------------------------------------------------------- roll-outs
class Rollout:
    """Step-wise reading state for a batch."""

    def __init__(self, model, b):
        self.m, self.b = model, b
        self.h_tab, self.items = model.encode(b)
        B, K, d = self.items.shape
        self.B, self.K = B, K
        self.state = model.init_state(self.h_tab)
        self.read_sum = self.items.new_zeros(B, d)
        self.read_cnt = self.items.new_zeros(B, 1)
        self.avail = b["item_mask"].clone()
        self.done = torch.zeros(B, dtype=torch.bool)
        self.n_read = torch.zeros(B, dtype=torch.long)
        self.final_pred = None

    def summary(self):
        return self.m.summary(self.state, self.read_sum, self.read_cnt, self.h_tab)

    def candidate_losses(self, y, task):
        """Loss after reading each candidate item (oracle look-ahead). (B, K), inf where unavailable."""
        out = torch.full((self.B, self.K), math.inf)
        ar = torch.arange(self.B)
        for k in range(self.K):
            ok = self.avail[:, k] > 0
            if not ok.any():
                continue
            x = self.items[:, k]
            new = self.m.rnn.step(self.state, x)
            s = self.m.summary(new, self.read_sum + x, self.read_cnt + 1, self.h_tab)
            l = sample_loss(self.m.predictor(s), y, task)
            out[ok, k] = l[ok]
        return out

    def apply(self, action):
        """action: (B,) in [0, K]; K = STOP. Already-done rows are ignored."""
        stop = (action == self.K) | self.done | (self.avail.sum(1) == 0)
        a = action.clamp(max=self.K - 1)
        x = self.items[torch.arange(self.B), a]
        new = self.m.rnn.step(self.state, x)
        self.state = StackedCell.blend(self.state, new, stop.unsqueeze(1))
        go = (~stop).float().unsqueeze(1)
        self.read_sum = self.read_sum + go * x
        self.read_cnt = self.read_cnt + go
        rows = torch.nonzero(~stop).squeeze(1)
        self.avail[rows, a[rows]] = 0
        self.n_read += (~stop).long()
        self.done = stop


def oracle_action(ro, y, task, lam):
    """Greedy oracle: read the item that lowers the loss most, unless the gain is below the cost lam."""
    cur = sample_loss(ro.m.predictor(ro.summary()), y, task)
    cand = ro.candidate_losses(y, task)
    best_l, best_k = cand.min(1)
    stop = (cur - best_l < lam) | torch.isinf(best_l)
    return torch.where(stop, torch.full_like(best_k, ro.K), best_k)


# ---------------------------------------------------------------- Phase B
def train_policy_imitation(model, data, task, lam, cfg: Config, log=print):
    """DAgger: roll in with a mixture of oracle and current policy, label every visited state with the oracle."""
    for p in model.parameters():
        p.requires_grad_(False)
    pol_params = list(model.q.parameters()) + list(model.k.parameters()) + [model.k_stop]
    for p in pol_params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(pol_params, lr=cfg.policy_lr)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    n = len(data)
    beta = cfg.dagger_beta0
    for ep in range(cfg.policy_epochs):
        model.eval()  # predictor/GRU frozen & deterministic
        perm = torch.randperm(n, generator=gen).numpy()
        tot, cnt, agree = 0.0, 0, 0
        for s in range(0, n, cfg.batch_size):
            b = to_batch(data, perm[s:s + cfg.batch_size])
            y = b["y"]
            with torch.no_grad():
                ro = Rollout(model, b)
            losses = []
            for t in range(ro.K + 1):
                if ro.done.all():
                    break
                with torch.no_grad():
                    a_star = oracle_action(ro, y, task, lam)
                    s_t = ro.summary()
                logits = model.policy_logits(s_t, ro.items, ro.avail)
                active = ~ro.done
                if active.any():
                    losses.append(F.cross_entropy(logits[active], a_star[active], reduction="sum"))
                    agree += (logits[active].argmax(1) == a_star[active]).sum().item()
                    cnt += active.sum().item()
                with torch.no_grad():
                    a_pol = torch.distributions.Categorical(logits=logits).sample()
                    use_oracle = torch.rand(ro.B, generator=gen) < beta
                    ro.apply(torch.where(use_oracle, a_star, a_pol))
            if losses:
                loss = torch.stack(losses).sum() / max(1, ro.B)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * ro.B
        beta *= cfg.dagger_decay
        log(f"  policy ep {ep:2d} loss {tot / n:.4f} oracle-agreement {agree / max(cnt, 1):.3f} beta {beta:.2f}")
    for p in model.parameters():
        p.requires_grad_(True)


def train_policy_gain(model, data, task, cfg: Config, explore=0.3, loss_kind="mse", epochs=None, log=print):
    """Learn E[loss reduction | state, item] by regression on observed one-step gains (label-free at test time).

    Roll-in follows the current gain estimates (epsilon-greedy), so the training states match the states the
    policy visits at test time (DAgger-style, fixes exposure bias). Reading continues until items run out,
    so the gain head sees every depth."""
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(model.gain.parameters()) + (list(model.meta_key.parameters()) if model.key_mode == "meta" else [])
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg.policy_lr)
    gen = torch.Generator().manual_seed(cfg.seed + 2)
    n = len(data)
    for ep in range(epochs or cfg.policy_epochs):
        model.eval()
        perm = torch.randperm(n, generator=gen).numpy()
        tot, cnt = 0.0, 0
        for s in range(0, n, cfg.batch_size):
            b = to_batch(data, perm[s:s + cfg.batch_size])
            y = b["y"]
            with torch.no_grad():
                ro = Rollout(model, b)
            keys = model.item_keys(b, ro.items)  # outside no_grad: meta_key must receive gradients
            losses, bcnt = [], 0
            for t in range(ro.K):
                if (ro.avail.sum(1) == 0).all():
                    break
                with torch.no_grad():
                    s_t = ro.summary()
                    cur = sample_loss(model.predictor(s_t), y, task)
                    cand = ro.candidate_losses(y, task)
                    target = cur.unsqueeze(1) - cand          # observed gain of each available item
                ok = ro.avail > 0
                g = model.gain_scores(s_t, keys)
                err = g - target.nan_to_num(posinf=0, neginf=0)
                if loss_kind == "huber":
                    losses.append(F.huber_loss(g[ok], (g - err)[ok], reduction="sum", delta=0.1))
                else:
                    losses.append((err ** 2)[ok].sum())
                bcnt += ok.sum().item()
                with torch.no_grad():
                    greedy = g.masked_fill(~ok, -math.inf).argmax(1)
                    rnd = random_order(ro.avail, gen)[:, 0]
                    a = torch.where(torch.rand(ro.B, generator=gen) < explore, rnd, greedy)
                    a = torch.where(ok.any(1), a, torch.full_like(a, ro.K))
                    ro.apply(a)
            if losses:
                loss = torch.stack(losses).sum() / max(bcnt, 1)  # per-batch mean (was a running count)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += torch.stack(losses).sum().item()
            cnt += bcnt
        log(f"  gain ep {ep:2d} mse {tot / max(cnt, 1):.5f}")
    for p in model.parameters():
        p.requires_grad_(True)


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, prep, mode="all", lam=0.0, budget=None, seed=0, bs=1024, task=None):
    """mode: 'none' (table only) | 'all' (read everything, natural order) | 'policy' (greedy pointer policy)
             | 'gain' (read the item with the highest predicted gain while it exceeds `lam`)
             | 'random' (random items; `budget` = per-row number of reads) | 'oracle' (greedy oracle, uses y)
    Returns (outputs, n_read)."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    outs, reads = [], []
    for s in range(0, len(prep), bs):
        b = to_batch(prep, slice(s, s + bs))
        mask = b["item_mask"]
        n_valid = mask.sum(1).long()
        if mode in ("none", "all", "random", "natural"):
            if mode == "none":
                n = torch.zeros_like(n_valid)
                order = natural_order(mask)
            elif mode == "all":
                n, order = n_valid, natural_order(mask)
            else:
                bud = torch.as_tensor(budget[s:s + bs]).long()
                n = torch.minimum(bud, n_valid)
                order = natural_order(mask) if mode == "natural" else random_order(mask, gen)
            preds = model.forward_order(b, order, n)
            out = preds.gather(0, n.view(1, -1, 1).expand(1, -1, preds.shape[-1])).squeeze(0)
            outs.append(out)
            reads.append(n)
            continue
        ro = Rollout(model, b)
        keys = model.item_keys(b, ro.items) if mode == "gain" else None
        final = None
        for t in range(ro.K + 1):
            pred = model.predictor(ro.summary())
            if mode == "policy":
                a = model.policy_logits(ro.summary(), ro.items, ro.avail).argmax(1)
            elif mode == "gain":
                g = model.gain_scores(ro.summary(), keys).masked_fill(ro.avail == 0, -math.inf)
                best, a = g.max(1)
                a = torch.where(best < lam, torch.full_like(a, ro.K), a)
            else:
                a = oracle_action(ro, b["y"], task, lam)
            prev_done = ro.done.clone()
            ro.apply(a)
            newly = ro.done & ~prev_done
            final = pred.clone() if final is None else torch.where(newly.unsqueeze(1), pred, final)
            if ro.done.all():
                break
        if not ro.done.all():
            pred = model.predictor(ro.summary())
            final = torch.where((~ro.done).unsqueeze(1), pred, final)
        outs.append(final)
        reads.append(ro.n_read)
    return torch.cat(outs), torch.cat(reads)
