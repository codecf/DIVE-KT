"""
DIVE-KT — Model

Dual-state Integration of Video Evidence for Knowledge Tracing.

Architecture overview:
  • Text projection bridge (EMBED_MODULES):
      Maps frozen LLM embeddings → hidden space.

  • Stage-1 trunk — DST modules (STAGE1_MODULES):
      Dual-state KT core: mastery state m_t, readiness state r_t,
      I²FRU forget gates, problem predictor.

  • Stage-2 branch — VMR + VEC modules (STAGE2_MODULES):
      Video Memory Retrieval (VMR), Video Evidence Credibility (VEC),
      fusion gate for probability-space mixing.

Parameter-group control:
  Use configure_for_stage1() / configure_for_stage2() / configure_for_joint()
  / configure_for_full_joint() before each training phase to freeze / unfreeze
  the appropriate modules.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DIVEKT(nn.Module):
    """
    DIVE-KT model with explicit parameter-group control for staged training.

    Module groups and their stage membership:

    ┌─────────────────┬──────────────────────────────────────────────────────┐
    │ Group           │ Modules                                              │
    ├─────────────────┼──────────────────────────────────────────────────────┤
    │ EMBED_MODULES   │ text_proj                                            │
    │ STAGE1_MODULES  │ resp_emb, time_mlp, assess_proj, forget_m, forget_r  │
    │                 │ readiness_gru, problem_gate, mastery_delta,          │
    │                 │ base_predictor                                       │
    │ STAGE2_MODULES  │ video_beh_enc, video_mem_proj, dt_bucket_emb,        │
    │                 │ vmr_*, video_predictor, fusion_gate                  │
    └─────────────────┴──────────────────────────────────────────────────────┘
    """

    EMBED_MODULES: frozenset = frozenset({"text_proj"})
    STAGE1_MODULES: frozenset = frozenset({
        "resp_emb", "time_mlp", "assess_proj",
        "forget_m", "forget_r", "readiness_gru",
        "problem_gate", "mastery_delta", "base_predictor",
    })
    STAGE2_MODULES: frozenset = frozenset({
        "video_beh_enc", "video_mem_proj", "dt_bucket_emb",
        "vmr_q_proj", "vmr_k_proj", "vmr_v_proj", "vmr_out_proj",
        "vmr_q_ln", "vmr_mem_ln", "vmr_norm1", "vmr_norm2",
        "vmr_drop", "vmr_ffn", "video_predictor", "fusion_gate",
    })

    def __init__(self, config, dataset_stats=None):
        super().__init__()
        self.cfg = config
        d = self.d = int(config.HIDDEN_DIM)
        llm_dim = int(config.LLM_EMBED_DIM)
        p1 = float(getattr(config, "DROPOUT",   0.3))
        p2 = float(getattr(config, "DROPOUT_2", 0.0))

        # ── Embedding bridge ──────────────────────────────────────────────────
        self.text_proj = nn.Sequential(
            nn.Linear(llm_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p1),
        )

        # ── Stage-1: DST trunk ────────────────────────────────────────────────
        self.resp_emb = nn.Embedding(2, d // 2)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d // 2), nn.GELU(), nn.Dropout(p1),
            nn.Linear(d // 2, d // 2), nn.GELU(),
        )
        self.assess_proj = nn.Sequential(
            nn.Linear(d + d // 2 + d // 2, d), nn.LayerNorm(d),
            nn.GELU(), nn.Dropout(p1),
        )
        self.forget_m = nn.Sequential(nn.Linear(d + d // 2, d), nn.Sigmoid())
        self.forget_r = nn.Sequential(nn.Linear(d + d // 2, d), nn.Sigmoid())
        self.readiness_gru   = nn.GRUCell(d, d)
        self.problem_gate    = nn.Sequential(
            nn.Linear(d + d + 1, d // 2), nn.GELU(), nn.Dropout(p1),
            nn.Linear(d // 2, 1), nn.Sigmoid(),
        )
        self.mastery_delta   = nn.Sequential(nn.Linear(d + d, d), nn.Tanh())
        self.base_predictor  = nn.Sequential(
            nn.Linear(d * 3, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p1),
            nn.Linear(d, 1),
        )

        # I²FRU incremental influence factors
        self.eta_pos = float(getattr(config, "I2FRU_ETA_P_POS", 0.25))
        self.eta_neg = float(getattr(config, "I2FRU_ETA_P_NEG", 0.05))

        # ── Stage-2: VMR + VEC branch ─────────────────────────────────────────
        self._video_corr_enabled = False
        self.K     = int(getattr(config, "VIDEO_CORR_MAX_BUF", 150))
        self.tau_v = float(getattr(config, "VIDEO_CORR_TEMP", 0.2))

        self.video_beh_enc = nn.Sequential(
            nn.Linear(4, d // 2), nn.GELU(), nn.Dropout(p2),
            nn.Linear(d // 2, d // 2), nn.GELU(),
        )
        self.video_mem_proj = nn.Sequential(
            nn.Linear(d + d // 2, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p2),
        )

        n_bins = self.n_dt_bins   = int(getattr(config, "EVID_DT_BINS",      64))
        self.dt_log_scale         = float(getattr(config, "EVID_DT_LOG_SCALE", 6.0))
        self.dt_bucket_emb        = nn.Embedding(n_bins, d)

        self.vmr_q_proj   = nn.Linear(d, d, bias=False)
        self.vmr_k_proj   = nn.Linear(d, d, bias=False)
        self.vmr_v_proj   = nn.Linear(d, d, bias=False)
        self.vmr_out_proj = nn.Linear(d, d, bias=False)
        self.vmr_q_ln     = nn.LayerNorm(d)
        self.vmr_mem_ln   = nn.LayerNorm(d)
        self.vmr_norm1    = nn.LayerNorm(d)
        self.vmr_norm2    = nn.LayerNorm(d)
        self.vmr_drop     = nn.Dropout(p2)
        self.vmr_ffn      = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(p2),
            nn.Linear(d * 4, d), nn.Dropout(p2),
        )

        self.video_predictor = nn.Sequential(
            nn.Linear(d * 2, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p2),
            nn.Linear(d, 1),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(d * 4, d // 2), nn.GELU(), nn.Dropout(p1),
            nn.Linear(d // 2, 1),
        )

        self._init_weights()
        nn.init.normal_(self.dt_bucket_emb.weight, 0.0, 0.02)
        if self.fusion_gate[-1].bias is not None:
            nn.init.constant_(self.fusion_gate[-1].bias, 1.5)

    # ──────────────────────────────────────────────────────────────────────────
    # Parameter-group configuration
    # ──────────────────────────────────────────────────────────────────────────

    def set_trainable_groups(
        self,
        train_embed:  bool = True,
        train_stage1: bool = True,
        train_stage2: bool = False,
    ) -> None:
        """
        Freeze / unfreeze parameter groups by name.

        Parameters whose root module name is not in any enabled group are
        frozen (``requires_grad = False``).
        """
        allowed: set[str] = set()
        if train_embed:
            allowed |= self.EMBED_MODULES
        if train_stage1:
            allowed |= self.STAGE1_MODULES
        if train_stage2:
            allowed |= self.STAGE2_MODULES

        for name, param in self.named_parameters():
            root = name.split(".")[0]
            param.requires_grad = root in allowed

    def unfreeze_all(self) -> None:
        """Unfreeze every parameter."""
        for p in self.parameters():
            p.requires_grad = True

    # Convenience configuration methods for each training phase:

    def configure_for_stage1(self, train_embed: bool = True) -> None:
        """Stage-1: train embedding bridge + DST trunk; disable video branch."""
        self.disable_video_corr()
        self.set_trainable_groups(train_embed=train_embed, train_stage1=True, train_stage2=False)

    def configure_for_stage2(self, train_embed: bool = True) -> None:
        """Stage-2 (default): train embedding + VMR/VEC; freeze DST trunk."""
        self.enable_video_corr()
        self.set_trainable_groups(train_embed=train_embed, train_stage1=False, train_stage2=True)

    def configure_for_stage2_joint(self, train_embed: bool = True) -> None:
        """Stage-2 joint: train embedding + both S1 trunk and S2 branch together."""
        self.enable_video_corr()
        self.set_trainable_groups(train_embed=train_embed, train_stage1=True, train_stage2=True)

    def configure_for_stage3(self, train_embed: bool = True) -> None:
        """Stage-3 fine-tune: same as joint — all modules trainable."""
        self.enable_video_corr()
        self.set_trainable_groups(train_embed=train_embed, train_stage1=True, train_stage2=True)

    def configure_for_full_joint(self, train_embed: bool = True) -> None:
        """
        Full-joint (single-stage): train ALL modules from scratch with video
        correlation enabled from epoch 1.  No staged checkpoint loading.
        """
        self.enable_video_corr()
        self.set_trainable_groups(train_embed=train_embed, train_stage1=True, train_stage2=True)

    # Legacy alias kept for backward compatibility with any saved scripts.
    def configure_for_joint(self, train_embed: bool = True) -> None:
        """Alias for :meth:`configure_for_stage3`."""
        self.configure_for_stage3(train_embed=train_embed)

    def freeze_trunk(self) -> None:
        """Freeze Stage-1 trunk; keep embedding + Stage-2 trainable."""
        self.set_trainable_groups(train_embed=True, train_stage1=False, train_stage2=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Video-correlation switch
    # ──────────────────────────────────────────────────────────────────────────

    def enable_video_corr(self, enable: bool = True) -> None:
        self._video_corr_enabled = bool(enable)

    def disable_video_corr(self) -> None:
        self._video_corr_enabled = False

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_summary(self) -> str:
        trainable = self.count_trainable_params()
        total     = sum(p.numel() for p in self.parameters())
        return f"{trainable:,}/{total:,} ({100.0 * trainable / max(total, 1):.1f}% trainable)"

    # ──────────────────────────────────────────────────────────────────────────
    # Weight initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _time_bucket(self, dt_sec: torch.Tensor) -> torch.Tensor:
        dt_min = dt_sec.float() / 60.0
        b = (torch.log1p(dt_min) * self.dt_log_scale).long()
        return b.clamp_(0, self.n_dt_bins - 1)

    @staticmethod
    def _masked_softmax(
        logits: torch.Tensor,
        mask:   torch.Tensor,
        temp:   float,
    ) -> torch.Tensor:
        logits = logits / max(temp, 1e-6)
        logits = logits.masked_fill(~mask, -1e9)
        w = torch.softmax(logits.float(), dim=-1) * mask.float()
        return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _vmr_retrieve(
        self,
        s_t:     torch.Tensor,   # (B', d)
        ts_t:    torch.Tensor,   # (B',)
        v_buf:   torch.Tensor,   # (B', K, d)
        v_ts:    torch.Tensor,   # (B', K)
        v_valid: torch.Tensor,   # (B', K)  bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend over the video memory buffer to produce a context vector."""
        dt  = torch.clamp(ts_t.unsqueeze(1) - v_ts, min=0)
        mem = self.vmr_mem_ln(v_buf + self.dt_bucket_emb(self._time_bucket(dt)))

        q = self.vmr_q_ln(self.vmr_q_proj(s_t))
        k = self.vmr_k_proj(mem)
        v = self.vmr_v_proj(mem)

        attn   = (k * q.unsqueeze(1)).sum(-1) / math.sqrt(self.d)
        w      = self._masked_softmax(attn, v_valid, self.tau_v)
        ctx    = (w.unsqueeze(-1) * v).sum(dim=1)

        q1     = self.vmr_norm1(q + self.vmr_drop(self.vmr_out_proj(ctx)))
        c_t    = self.vmr_norm2(q1 + self.vmr_ffn(q1))
        return c_t, w

    @staticmethod
    def _prob_space_fusion(
        logit_base: torch.Tensor,
        logit_vid:  torch.Tensor,
        gate_logit: torch.Tensor,
    ) -> torch.Tensor:
        """
        Probability-space mixture of two predictors via a learned gate λ.

        p_mix = (1 - λ) * p_base + λ * p_vid   (computed in log-space for stability)
        """
        log_lam   = F.logsigmoid(gate_logit)
        log_1mlam = F.logsigmoid(-gate_logit)
        log_pb, log_qb = F.logsigmoid(logit_base), F.logsigmoid(-logit_base)
        log_pv, log_qv = F.logsigmoid(logit_vid),  F.logsigmoid(-logit_vid)

        log_p = torch.logaddexp(log_1mlam + log_pb, log_lam + log_pv)
        log_q = torch.logaddexp(log_1mlam + log_qb, log_lam + log_qv)
        return log_p - log_q

    # ──────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        item_ids:     torch.Tensor,   # (B, L)
        text_embeds:  torch.Tensor,   # (B, L, llm_dim)
        beh_features: torch.Tensor,   # (B, L, 6)
        timestamps:   torch.Tensor,   # (B, L)
        return_trace: bool = False,
    ) -> tuple:
        """
        Returns:
            If ``return_trace`` is False:
                logits, aux_loss, cl_loss
            If ``return_trace`` is True:
                logits, aux_loss, cl_loss, trace

            ``trace`` contains per-position diagnostic tensors used for
            case-study mining, including base/video predictions, fusion
            gates, retrieval weights, and retrieved video item ids.
        """
        device = item_ids.device
        B, L = item_ids.shape
        d, K = self.d, self.K

        beh = beh_features.float()
        ts  = timestamps.long() if timestamps.dtype != torch.long else timestamps
        s_all = self.text_proj(text_embeds.float())   # (B, L, d)

        valid   = (item_ids != 0)
        is_prob = (beh[..., 0] >= 0.5)
        is_vid  = valid & ~is_prob

        # ── Running states ────────────────────────────────────────────────────
        m         = torch.zeros(B, d, device=device)
        r         = torch.zeros(B, d, device=device)
        last_ts   = torch.zeros(B, dtype=torch.long, device=device)
        seen_prob = torch.zeros(B, dtype=torch.bool, device=device)

        # ── Video memory buffer ───────────────────────────────────────────────
        v_buf      = torch.zeros(B, K, d, device=device)
        v_ts       = torch.zeros(B, K, dtype=torch.long, device=device)
        v_valid    = torch.zeros(B, K, dtype=torch.bool, device=device)
        v_size     = torch.zeros(B, dtype=torch.long, device=device)
        v_item_ids = torch.zeros(B, K, dtype=torch.long, device=device)

        logits_out = []

        if return_trace:
            trace = {
                "z_base": [],
                "p_base": [],
                "z_vid": [],
                "p_vid": [],
                "gate_logit": [],
                "lambda_gate": [],
                "z_mix": [],
                "p_mix": [],
                "attn_weights": [],
                "retrieved_video_ids": [],
                "has_video": [],
                "is_problem": [],
            }
        else:
            trace = None

        for t in range(L):
            ts_t = ts[:, t]
            s_t  = s_all[:, t]

            # ── Update video memory buffer ────────────────────────────────────
            if self._video_corr_enabled:
                vid_mask = is_vid[:, t]
                if vid_mask.any():
                    v_buf, v_ts, v_valid, v_size, v_item_ids = (
                        v_buf.clone(), v_ts.clone(),
                        v_valid.clone(), v_size.clone(), v_item_ids.clone(),
                    )
                    u_v = self.video_beh_enc(beh[:, t, 2:6])
                    h_v = self.video_mem_proj(torch.cat([s_t, u_v], dim=-1))

                    for i in torch.where(vid_mask)[0].tolist():
                        cur = int(v_size[i].item())
                        if cur < K:
                            pos = cur
                            v_size[i] = cur + 1
                        else:
                            # Shift oldest out (FIFO).
                            v_buf[i, :-1]      = v_buf[i, 1:].clone()
                            v_ts[i, :-1]       = v_ts[i, 1:].clone()
                            v_valid[i, :-1]    = v_valid[i, 1:].clone()
                            v_item_ids[i, :-1] = v_item_ids[i, 1:].clone()
                            pos = K - 1
                        v_buf[i, pos]      = h_v[i]
                        v_ts[i, pos]       = ts_t[i]
                        v_valid[i, pos]    = True
                        v_item_ids[i, pos] = item_ids[i, t]

            # ── Problem step ──────────────────────────────────────────────────
            active = valid[:, t] & is_prob[:, t]

            delta = torch.zeros(B, dtype=torch.long, device=device)
            if active.any():
                raw_dt = torch.clamp(ts_t - last_ts, min=0)
                delta  = torch.where(seen_prob, raw_dt, delta)
            last_ts   = torch.where(active, ts_t, last_ts)
            seen_prob = seen_prob | active

            e_dt  = self.time_mlp(torch.log1p(delta.float()).unsqueeze(-1))
            fm    = self.forget_m(torch.cat([m, e_dt], dim=-1))
            fr    = self.forget_r(torch.cat([r, e_dt], dim=-1))
            m_tilde = torch.where(active.unsqueeze(-1), fm * m, m)
            r_tilde = torch.where(active.unsqueeze(-1), fr * r, r)

            z_base = self.base_predictor(
                torch.cat([m_tilde, r_tilde, s_t], dim=-1)
            ).squeeze(-1)
            z_base  = torch.where(active, z_base, torch.zeros_like(z_base))
            logit_t = z_base

            if return_trace:
                z_vid_t       = torch.full_like(z_base, float("nan"))
                p_vid_t       = torch.full_like(z_base, float("nan"))
                gate_logit_t  = torch.full_like(z_base, float("nan"))
                lambda_t      = torch.full_like(z_base, float("nan"))
                attn_t        = torch.zeros(B, K, device=device)
                retrieved_t   = v_item_ids.clone()
                has_video_t   = active & v_valid.any(dim=-1)

            # ── VMR-based evidence fusion ─────────────────────────────────────
            if self._video_corr_enabled and active.any() and v_valid.any():
                has_evidence = active & v_valid.any(dim=-1)
                rows = torch.where(has_evidence)[0]
                if rows.numel() > 0:
                    logit_t = z_base.clone()
                    c_t, w = self._vmr_retrieve(
                        s_t[rows], ts_t[rows],
                        v_buf[rows], v_ts[rows], v_valid[rows],
                    )
                    z_vid  = self.video_predictor(
                        torch.cat([c_t, s_t[rows]], dim=-1)
                    ).squeeze(-1)
                    gamma  = self.fusion_gate(
                        torch.cat([m_tilde[rows], r_tilde[rows], c_t, s_t[rows]], dim=-1)
                    ).squeeze(-1)
                    z_mix  = self._prob_space_fusion(
                        z_base[rows].float(), z_vid.float(), gamma.float()
                    )
                    logit_t[rows] = z_mix.to(logit_t.dtype)
                    logit_t = torch.where(active, logit_t, torch.zeros_like(logit_t))

                    if return_trace:
                        z_vid_t[rows]      = z_vid.detach()
                        p_vid_t[rows]      = torch.sigmoid(z_vid.detach())
                        gate_logit_t[rows] = gamma.detach()
                        lambda_t[rows]     = torch.sigmoid(gamma.detach())
                        attn_t[rows]       = w.detach()
                        retrieved_t[rows]  = v_item_ids[rows].detach()

            logits_out.append(logit_t)

            if return_trace:
                trace["z_base"].append(z_base.detach())
                trace["p_base"].append(torch.sigmoid(z_base.detach()))
                trace["z_vid"].append(z_vid_t.detach())
                trace["p_vid"].append(p_vid_t.detach())
                trace["gate_logit"].append(gate_logit_t.detach())
                trace["lambda_gate"].append(lambda_t.detach())
                trace["z_mix"].append(logit_t.detach())
                trace["p_mix"].append(torch.sigmoid(logit_t.detach()))
                trace["attn_weights"].append(attn_t.detach())
                trace["retrieved_video_ids"].append(retrieved_t.detach())
                trace["has_video"].append(has_video_t.detach())
                trace["is_problem"].append(active.detach())

            # ── State update (problem events only) ───────────────────────────
            if active.any():
                y_t   = torch.clamp(beh[:, t, 1].round().long(), 0, 1)
                a_emb = self.resp_emb(y_t)
                x_A   = self.assess_proj(torch.cat([s_t, a_emb, e_dt], dim=-1))

                p_base_sg = torch.sigmoid(z_base.detach()).unsqueeze(-1)
                g_p = self.problem_gate(
                    torch.cat([r_tilde, x_A, p_base_sg], dim=-1)
                ).squeeze(-1)

                r_cand = self.readiness_gru(x_A, r_tilde)
                r_new  = r_tilde + g_p.unsqueeze(-1) * (r_cand - r_tilde)

                eta  = self.eta_neg + (self.eta_pos - self.eta_neg) * y_t.float()
                dm   = self.mastery_delta(torch.cat([m_tilde, x_A], dim=-1))
                m_new = m_tilde + (eta * g_p).unsqueeze(-1) * dm

                m = torch.where(active.unsqueeze(-1), m_new, m)
                r = torch.where(active.unsqueeze(-1), r_new, r)

        logits = torch.stack(logits_out, dim=1)
        zero   = torch.zeros((), device=device)

        if return_trace:
            trace = {k: torch.stack(v, dim=1) for k, v in trace.items()}
            self.last_trace = trace
            return logits, zero, zero, trace

        self.last_trace = {}
        return logits, zero, zero

    @torch.no_grad()
    def forward_with_trace(
        self,
        item_ids:     torch.Tensor,
        text_embeds:  torch.Tensor,
        beh_features: torch.Tensor,
        timestamps:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Evaluation-only forward pass that returns diagnostic traces for
        case-study mining. Training code should continue to call ``forward``
        without ``return_trace``.
        """
        return self.forward(
            item_ids=item_ids,
            text_embeds=text_embeds,
            beh_features=beh_features,
            timestamps=timestamps,
            return_trace=True,
        )
