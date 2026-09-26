"""Sparse autoencoders on neural record embeddings (plan §6.6c). Needs torch + neural.py outputs.

TopK SAE: z = TopK(W_enc (x − b_dec) + b_enc), x̂ = W_dec z + b_dec, unit-norm decoder columns,
MSE loss + auxiliary loss that lets dead latents reconstruct the residual. Diagnostic only."""
from __future__ import annotations

import html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import INK2, MUTED, NEUTRAL_BAR, SERIES, FigureSink


def _torch():
    import torch
    return torch


class TopKSAE:
    def __init__(self, d: int, n_latents: int, k: int, seed: int = 0, device: str | None = None):
        torch = _torch()
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(d, n_latents, generator=g)
        W = W / W.norm(dim=0, keepdim=True)
        self.W_dec = W.clone().T.contiguous().to(self.device).requires_grad_(True)   # (n, d)
        self.W_enc = W.clone().to(self.device).requires_grad_(True)                  # (d, n)
        self.b_enc = torch.zeros(n_latents, device=self.device, requires_grad=True)
        self.b_dec = torch.zeros(d, device=self.device, requires_grad=True)
        self.k, self.n = k, n_latents
        self.last_fired = torch.zeros(n_latents, device=self.device)

    def params(self):
        return [self.W_enc, self.W_dec, self.b_enc, self.b_dec]

    def encode_pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x, k=None):
        torch = self.torch
        pre = self.encode_pre(x)
        v, i = torch.topk(pre, k or self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, i, torch.relu(v))
        return z, pre

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def fit(self, X: np.ndarray, epochs: int = 10, batch: int = 4096, lr: float = 3e-4, aux_k: int = 256,
            dead_steps: int = 2000, log=print):
        torch = self.torch
        Xt = torch.from_numpy(np.asarray(X, np.float32))
        opt = torch.optim.Adam(self.params(), lr=lr)
        step = 0
        for ep in range(epochs):
            perm = torch.randperm(len(Xt))
            tot = 0.0
            for b in range(0, len(Xt), batch):
                x = Xt[perm[b:b + batch]].to(self.device)
                z, pre = self.encode(x)
                xh = self.decode(z)
                err = x - xh
                loss = (err ** 2).sum(-1).mean()
                fired = (z > 0).any(0)
                self.last_fired = torch.where(fired, torch.zeros_like(self.last_fired), self.last_fired + 1)
                dead = self.last_fired > dead_steps
                if dead.any():                                   # auxiliary loss on dead latents
                    pre_d = pre.masked_fill(~dead, -float("inf"))
                    kk = int(min(aux_k, dead.sum().item()))
                    v, i = torch.topk(pre_d, kk, dim=-1)
                    zd = torch.zeros_like(pre).scatter_(-1, i, torch.relu(v))
                    loss = loss + (1 / 32) * ((err.detach() - zd @ self.W_dec) ** 2).sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                with torch.no_grad():                            # keep decoder rows unit norm, remove parallel grad
                    W = self.W_dec
                    W.grad -= (W.grad * W).sum(-1, keepdim=True) * W
                opt.step()
                with torch.no_grad():
                    self.W_dec /= self.W_dec.norm(dim=-1, keepdim=True)
                tot += loss.item() * len(x)
                step += 1
            log(f"SAE n={self.n} k={self.k} epoch {ep + 1}/{epochs}: loss {tot / len(Xt):.5f}")
        return self

    def transform(self, X: np.ndarray, batch: int = 16384) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for b in range(0, len(X), batch):
                z, _ = self.encode(torch.from_numpy(np.asarray(X[b:b + batch], np.float32)).to(self.device))
                out.append(z.cpu().numpy().astype(np.float32))
        return np.concatenate(out)

    def reconstruct(self, Z: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            return self.decode(torch.from_numpy(Z).to(self.device)).cpu().numpy()


def fvu(X, Xh) -> float:
    return float(((X - Xh) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())


def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def run(sink: FigureSink, emb: np.ndarray, ids: np.ndarray, texts: pd.Series, pairs: pd.DataFrame,
        ks=(16, 32, 64), expansion: int = 8, epochs: int = 8, log=print):
    """emb: (n, d) float record embeddings; ids: entity ids aligned; texts: display strings by id;
    pairs: dev candidate pairs (s1_entity_id, cand_entity_id, label) for downstream checks."""
    X = np.asarray(emb, np.float32)
    mu, sd = X.mean(0), X.std() + 1e-6
    Xn = (X - mu) / sd
    d = X.shape[1]
    idx = pd.Index(ids)
    pr = pairs[pairs.s1_entity_id.isin(idx) & pairs.cand_entity_id.isin(idx)].sample(
        min(200_000, len(pairs)), random_state=0)
    a, b = idx.get_indexer(pr.s1_entity_id), idx.get_indexer(pr.cand_entity_id)
    y = pr.label.to_numpy()

    def auc(Xe):
        from sklearn.metrics import roc_auc_score
        E = _norm(Xe)
        return float(roc_auc_score(y, (E[a] * E[b]).sum(1))) if len(np.unique(y)) > 1 else np.nan

    base_auc = auc(X)
    rows, saes = [], {}
    for k in ks:
        sae = TopKSAE(d, expansion * d, k).fit(Xn, epochs=epochs, log=log)
        Z = sae.transform(Xn)
        Xh = sae.reconstruct(Z) * sd + mu
        freq = (Z > 0).mean(0)
        rows.append({"k": k, "n_latents": expansion * d, "fvu": fvu(X, Xh), "dead_frac": float((freq == 0).mean()),
                     "dense_frac": float((freq > 0.1).mean()), "pair_auc_original": base_auc, "pair_auc_recon": auc(Xh)})
        saes[k] = (sae, Z, Xh)
        log(f"SAE k={k}: {rows[-1]}")
    q = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(q.k, q.fvu, color=SERIES[0], marker="o")
    axes[0].set(xlabel="k (active latents per record)", ylabel="fraction of variance unexplained",
                title="Reconstruction vs sparsity")
    axes[1].plot(q.k, q.pair_auc_recon, color=SERIES[0], marker="o", label="with SAE reconstruction")
    axes[1].axhline(base_auc, color=SERIES[1], lw=1.5, ls="--", label="original embedding")
    axes[1].set(xlabel="k", ylabel="match-vs-candidate AUC (cosine)", title="Downstream fidelity")
    axes[1].legend(fontsize=8)
    sink.save(fig, "sae_quality", "SAE quality: FVU vs L0 and downstream fidelity", "SAE", q,
              note="Interpret only SAEs that lose little downstream AUC (plan: < 1 pp).")

    k_mid = ks[len(ks) // 2]
    sae, Z, _ = saes[k_mid]
    freq = (Z > 0).mean(0)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.hist(np.log10(freq[freq > 0]), bins=50, color=SERIES[0])
    ax.set(xlabel="log10 firing frequency", ylabel="latents", title=f"Latent density (k={k_mid}, dead = {int((freq == 0).sum())})")
    sink.save(fig, "sae_latent_density", f"Latent firing-frequency histogram (k={k_mid})", "SAE",
              pd.DataFrame({"latent": np.arange(len(freq)), "freq": freq}))

    # which latents differ between a record and its candidate on false vs true pairs?
    Za, Zb = Z[a], Z[b]
    diff = np.abs((Za > 0).astype(np.float32) - (Zb > 0).astype(np.float32))
    pos, neg = diff[y == 1].mean(0), diff[y == 0].mean(0)
    lo = np.log2((neg + 1e-4) / (pos + 1e-4))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(np.log10(freq + 1e-6), lo, s=6, color=NEUTRAL_BAR, linewidths=0)
    top = np.argsort(-lo)[:15]
    ax.scatter(np.log10(freq[top] + 1e-6), lo[top], s=16, color=SERIES[0])
    for t in top[:8]:
        ax.annotate(str(t), (np.log10(freq[t] + 1e-6), lo[t]), xytext=(3, 2), textcoords="offset points", fontsize=7, color=INK2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set(xlabel="log10 firing frequency", ylabel="log2 disagreement ratio (non-match / match)",
           title="Latents that separate matches from candidates")
    assoc = pd.DataFrame({"latent": np.arange(len(freq)), "freq": freq, "disagree_match": pos, "disagree_nonmatch": neg,
                          "log2_ratio": lo}).sort_values("log2_ratio", ascending=False)
    sink.save(fig, "sae_label_association", "Match-discriminating SAE latents (volcano-style)", "SAE", assoc)

    # causal check: zero the top discriminating latents and re-measure pair AUC
    abl = []
    for n_off in (0, 5, 20, 50):
        Zc = Z.copy()
        Zc[:, np.argsort(-lo)[:n_off]] = 0
        abl.append({"latents_zeroed": n_off, "pair_auc": auc(sae.reconstruct(Zc) * sd + mu)})
    abl = pd.DataFrame(abl)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(abl.latents_zeroed, abl.pair_auc, color=SERIES[0], marker="o")
    ax.set(xlabel="top discriminating latents zeroed", ylabel="pair AUC", title="Ablating SAE latents")
    sink.save(fig, "sae_ablation", "Effect of zeroing the most discriminating latents", "SAE", abl)

    # dashboards: top activating records per latent + simple concept tags
    tags = _concept_tags(texts.reindex(ids).fillna("").to_numpy())
    cards = []
    for t in assoc.latent.head(40):
        acts = Z[:, t]
        top_i = np.argsort(-acts)[:12]
        fired = acts > 0
        tag_rates = {k: float(v[fired].mean()) if fired.any() else 0.0 for k, v in tags.items()}
        base = {k: float(v.mean()) for k, v in tags.items()}
        lift = {k: tag_rates[k] / max(base[k], 1e-6) for k in tags}
        best_tag = max(lift, key=lift.get)
        items = "".join(f"<li><b>{acts[i]:.2f}</b> {html.escape(str(texts.get(ids[i], '')))}</li>" for i in top_i)
        cards.append(f"<figure><figcaption>latent {t} · freq {freq[t]:.4f} · "
                     f"log2 non-match/match {lo[t]:+.2f} · strongest tag: {best_tag} (×{lift[best_tag]:.1f})"
                     f"</figcaption><ol>{items}</ol></figure>")
    sink.html("sae_dashboard", f"SAE latent dashboard (k={k_mid})", "SAE",
              f"<h1>SAE latents (k={k_mid})</h1><p class='note'>Top-activating records for the 40 latents that most "
              f"separate non-matches from matches.</p>" + "".join(cards))

    # latent map: 2-D projection of decoder directions coloured by strongest concept tag
    from sklearn.decomposition import PCA
    Wd = sae.W_dec.detach().cpu().numpy()
    P = PCA(2, random_state=0).fit_transform(Wd)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(P[:, 0], P[:, 1], s=4, color=NEUTRAL_BAR, linewidths=0)
    ax.scatter(P[top, 0], P[top, 1], s=18, color=SERIES[0])
    ax.set(title="Decoder directions (PCA); blue = match-discriminating")
    ax.set_xticks([])
    ax.set_yticks([])
    sink.save(fig, "sae_latent_map", "Map of SAE decoder directions", "SAE",
              pd.DataFrame({"latent": np.arange(len(P)), "x": P[:, 0], "y": P[:, 1]}))
    return q


def _concept_tags(texts: np.ndarray) -> dict[str, np.ndarray]:
    s = pd.Series(texts).str.lower()
    return {
        "has digits": s.str.contains(r"\d").to_numpy(),
        "legal form": s.str.contains(r"\b(?:llc|inc|ltd|limited|private|pvt|corp|sarl|sas)\b").to_numpy(),
        "street word": s.str.contains(r"\b(?:street|road|avenue|lane|rue|nagar|marg)\b").to_numpy(),
        "india words": s.str.contains(r"\b(?:nagar|colony|sector|marg|chowk|pradesh)\b").to_numpy(),
        "french words": s.str.contains(r"\b(?:rue|avenue|boulevard|chemin|place|sarl|sas)\b").to_numpy(),
        "short name": (s.str.split("|").str[0].str.len() < 12).to_numpy(),
    }
