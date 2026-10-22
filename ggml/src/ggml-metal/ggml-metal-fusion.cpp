#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-fusion.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-device.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <set>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
ਸਥਿਰ std::vector<ggml_op> ggml_metal_fusion_filter_ops(ਅਚਲ std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    ਲਈ (ggml_op op : ops_all) {
        ਜੇ (!ggml_op_is_ਖਾਲੀ_ਥਾਵਾਂ(op)) {
            ops.push_back(op);
        }
    }

    ਵਾਪਸ ops;
}

ਬਣਤਰ ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<ਪੂਰਨ>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    ਤਾਰਕਿਕ unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    ਤਾਰਕਿਕ (*check)(ਅਚਲ ਬਣਤਰ ggml_metal_fusion   * fusion,
                  ਅਚਲ ਬਣਤਰ ggml_tensor * ਅਚਲ * nodes,
                  ਅਚਲ ਬਣਤਰ ggml_cgraph         * gf,
                  ਅਚਲ ਪੂਰਨ                        * ਗੰਢ_idxs,
                        ਪੂਰਨ                          idx,
                        ggml_metal_fusion_ਬਹੁਲਕ       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            ਅਚਲ std::vector<ggml_op> & ops_all,
            ਅਚਲ std::vector<ਪੂਰਨ> & outs,
            ਤਾਰਕਿਕ unsafe,
            ਤਾਰਕਿਕ (*check)(ਅਚਲ ਬਣਤਰ ggml_metal_fusion   * fusion,
                          ਅਚਲ ਬਣਤਰ ggml_tensor * ਅਚਲ * nodes,
                          ਅਚਲ ਬਣਤਰ ggml_cgraph         * gf,
                          ਅਚਲ ਪੂਰਨ                        * ਗੰਢ_idxs,
                                ਪੂਰਨ                          idx,
                                ggml_metal_fusion_ਬਹੁਲਕ       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(ਅਚਲ ggml_metal_fusion * fusion) {
    ਵਾਪਸ fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_fusion_same_ਬਫਰ(ਅਚਲ ggml_tensor * a, ਅਚਲ ggml_tensor * b) {
    ਜੇ (!a || !b) {
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_backend_ਬਫਰ_t ba = a->view_src ? a->view_src->ਬਫਰ : a->ਬਫਰ;
    ggml_backend_ਬਫਰ_t bb = b->view_src ? b->view_src->ਬਫਰ : b->ਬਫਰ;

    ggml_metal_ਬਫਰ_t ca = (ggml_metal_ਬਫਰ_t) ba->context;
    ggml_metal_ਬਫਰ_t cb = (ggml_metal_ਬਫਰ_t) bb->context;

    ਵਾਪਸ ggml_metal_ਬਫਰ_get_id(ca, a).metal == ggml_metal_ਬਫਰ_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_fusion_overlap(ਅਚਲ ggml_tensor * a, ਅਚਲ ggml_tensor * b) {
    ggml_backend_ਬਫਰ_t ba = a->view_src ? a->view_src->ਬਫਰ : a->ਬਫਰ;
    ggml_backend_ਬਫਰ_t bb = b->view_src ? b->view_src->ਬਫਰ : b->ਬਫਰ;

    ਅਚਲ ggml_metal_ਬਫਰ_id bid_a = ggml_metal_ਬਫਰ_get_id((ggml_metal_ਬਫਰ_t) ba->context, a);
    ਅਚਲ ggml_metal_ਬਫਰ_id bid_b = ggml_metal_ਬਫਰ_get_id((ggml_metal_ਬਫਰ_t) bb->context, b);

    ਜੇ (bid_a.metal == ਖਾਲੀ_ਸੂਚਕ || bid_a.metal != bid_b.metal) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_fusion_check_norm(
        ਅਚਲ ggml_metal_fusion       * fusion,
        ਅਚਲ ggml_tensor * ਅਚਲ     * nodes,
        ਅਚਲ ggml_cgraph             * gf,
        ਅਚਲ ਪੂਰਨ                     * ਗੰਢ_idxs,
              ਪੂਰਨ                       idx,
              ggml_metal_fusion_ਬਹੁਲਕ    mode) {
    GGML_ਅਣਵਰਤਿਆ(mode);
    GGML_ਅਣਵਰਤਿਆ(gf);
    GGML_ਅਣਵਰਤਿਆ(ਗੰਢ_idxs);
    GGML_ਅਣਵਰਤਿਆ(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    ਜੇ (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        ਅਚਲ ggml_tensor * scale = nodes[1];
        ਜੇ (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਵਾਪਸ ਸੱਚ;
    }

    ਲਈ (ਪੂਰਨ j = 1; j < (ਪੂਰਨ) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        ਜੇ (nodes[j]->src[0] != nodes[j - 1]) {
            ਵਾਪਸ ਝੂਠ;
        }

        // the weight/bias must have the same row width as the norm input
        ਜੇ (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (!ggml_is_contiguous_ਪੰਕਤੀਆਂ(nodes[j]->src[1])) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (nodes[j]->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਵਾਪਸ ਸੱਚ;
}

// SSM_CONV + UNARY (silu)
ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_fusion_check_ssm_conv_silu(
        ਅਚਲ ggml_metal_fusion       * fusion,
        ਅਚਲ ggml_tensor * ਅਚਲ     * nodes,
        ਅਚਲ ggml_cgraph             * gf,
        ਅਚਲ ਪੂਰਨ                     * ਗੰਢ_idxs,
              ਪੂਰਨ                       idx,
              ggml_metal_fusion_ਬਹੁਲਕ    mode) {
    GGML_ਅਣਵਰਤਿਆ(fusion);
    GGML_ਅਣਵਰਤਿਆ(gf);
    GGML_ਅਣਵਰਤਿਆ(ਗੰਢ_idxs);
    GGML_ਅਣਵਰਤਿਆ(idx);
    GGML_ਅਣਵਰਤਿਆ(mode);

    ਅਚਲ ggml_tensor * conv = nodes[0];
    ਅਚਲ ggml_tensor * un   = nodes[1];

    ਜੇ (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_ਪੰਕਤੀਆਂ(un)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_fusion_check_ਜੋੜੋ_chain(
        ਅਚਲ ggml_metal_fusion       * fusion,
        ਅਚਲ ggml_tensor * ਅਚਲ     * nodes,
        ਅਚਲ ggml_cgraph             * gf,
        ਅਚਲ ਪੂਰਨ                     * ਗੰਢ_idxs,
              ਪੂਰਨ                       idx,
              ggml_metal_fusion_ਬਹੁਲਕ    mode) {
    GGML_ਅਣਵਰਤਿਆ(gf);
    GGML_ਅਣਵਰਤਿਆ(ਗੰਢ_idxs);
    GGML_ਅਣਵਰਤਿਆ(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    ਲਈ (ਪੂਰਨ j = 1; j < (ਪੂਰਨ) fusion->ops.size(); j++) {
        ਜੇ (nodes[j]->src[0] != nodes[j - 1]) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (mode == GGML_METAL_FUSION_ਭਰਿਆ) {
            ਜੇ (!ggml_metal_fusion_same_ਬਫਰ(nodes[j]->src[1], nodes[0]->src[1])) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
    }

    ਵਾਪਸ ਸੱਚ;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_fusion_check_gdn_cache(
        ਅਚਲ ggml_metal_fusion       * fusion,
        ਅਚਲ ggml_tensor * ਅਚਲ     * nodes,
        ਅਚਲ ggml_cgraph             * gf,
        ਅਚਲ ਪੂਰਨ                     * ਗੰਢ_idxs,
              ਪੂਰਨ                       idx,
              ggml_metal_fusion_ਬਹੁਲਕ    mode) {
    GGML_ਅਣਵਰਤਿਆ(fusion);
    GGML_ਅਣਵਰਤਿਆ(gf);
    GGML_ਅਣਵਰਤਿਆ(ਗੰਢ_idxs);
    GGML_ਅਣਵਰਤਿਆ(idx);

    ਅਚਲ ggml_tensor * gdn = nodes[0];
    ਅਚਲ ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    ਜੇ (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int64_t S_v      = gdn->src[2]->ne[0];
    ਅਚਲ int64_t H        = gdn->src[2]->ne[1];
    ਅਚਲ int64_t n_ਟੋਕਨ = gdn->src[2]->ne[2];
    ਅਚਲ int64_t n_seqs   = gdn->src[2]->ne[3];
    ਅਚਲ int64_t K        = ggml_get_op_params_i32(gdn, 0);
    ਅਚਲ size_t  ਪੂਛ_off = ggml_ਪੰਕਤੀ_ਮਾਪ(GGML_TYPE_F32, S_v * H * n_ਟੋਕਨ * n_seqs);

    ਅਚਲ int64_t D         = S_v * S_v * H;
    ਅਚਲ int64_t n_written = std::min<int64_t>(n_ਟੋਕਨ, K);

    ਅਚਲ ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    ਅਚਲ ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    ਜੇ (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != ਪੂਛ_off || !ggml_is_contiguous(src)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    ਜੇ (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_ਮਾਪ(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_ਪੰਕਤੀ_ਮਾਪ(GGML_TYPE_F32, D)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (mode == GGML_METAL_FUSION_ਭਰਿਆ) {
        // the cache must be allocated so the kernel can write straight to its buffer
        ਜੇ (dst->ਅੰਕੜੇ == ਖਾਲੀ_ਸੂਚਕ) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਵਾਪਸ ਸੱਚ;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_fusion_check_snake(
        ਅਚਲ ggml_metal_fusion       * fusion,
        ਅਚਲ ggml_tensor * ਅਚਲ     * nodes,
        ਅਚਲ ggml_cgraph             * gf,
        ਅਚਲ ਪੂਰਨ                     * ਗੰਢ_idxs,
              ਪੂਰਨ                       idx,
              ggml_metal_fusion_ਬਹੁਲਕ    mode) {
    GGML_ਅਣਵਰਤਿਆ(fusion);
    GGML_ਅਣਵਰਤਿਆ(mode);
    GGML_ਅਣਵਰਤਿਆ(gf);
    GGML_ਅਣਵਰਤਿਆ(ਗੰਢ_idxs);
    GGML_ਅਣਵਰਤਿਆ(idx);

    ਅਚਲ ggml_tensor * mul0     = nodes[0];
    ਅਚਲ ggml_tensor * sin_ਗੰਢ = nodes[1];
    ਅਚਲ ggml_tensor * sqr      = nodes[2];
    ਅਚਲ ggml_tensor * mul1     = nodes[3];
    ਅਚਲ ggml_tensor * ਜੋੜੋ_2      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    ਅਚਲ ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ਅਚਲ ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ਅਚਲ ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    ਅਚਲ ggml_tensor * x_ਵਿੱਚ_ਜੋੜੋ = (ਜੋੜੋ_2->src[0] == mul1) ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    ਅਚਲ ਤਾਰਕਿਕ types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_ਗੰਢ->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (ਜੋੜੋ_2->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    ਅਚਲ ਤਾਰਕਿਕ shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    ਅਚਲ ਤਾਰਕਿਕ dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (ਜੋੜੋ_2->ne[2]   == 1) && (ਜੋੜੋ_2->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    ਅਚਲ ਤਾਰਕਿਕ contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(ਜੋੜੋ_2) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    ਵਾਪਸ types_ok && shape_ok && dim_ok && contig_ok && x_ਵਿੱਚ_ਜੋੜੋ == x;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ਪੰਕਤੀਆਂ
};
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ਪੰਕਤੀਆਂ, GGML_OP_SCALE
};
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ਪੰਕਤੀਆਂ,
    GGML_OP_RESHAPE, GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ਪੰਕਤੀਆਂ,
    GGML_OP_RESHAPE, GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_fusion_check_topk_moe(
        ਅਚਲ ggml_metal_fusion       * fusion,
        ਅਚਲ ggml_tensor * ਅਚਲ     * nodes,
        ਅਚਲ ggml_cgraph             * gf,
        ਅਚਲ ਪੂਰਨ                     * ਗੰਢ_idxs,
              ਪੂਰਨ                       idx,
              ggml_metal_fusion_ਬਹੁਲਕ    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_ਅਣਵਰਤਿਆ(nodes);

    ਅਚਲ ਪੂਰਨ n_ops = (ਪੂਰਨ) fusion->ops.size();

    ਅਚਲ ਤਾਰਕਿਕ with_norm  = n_ops >= 6;
    ਅਚਲ ਤਾਰਕਿਕ with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    ਅਚਲ std::vector<ggml_op> & ops_all = fusion->ops_all;

    ਅਚਲ ਪੂਰਨ raw_ਸ਼ੁਰੂ = ਗੰਢ_idxs[idx];
    ਪੂਰਨ       raw_ਸਿਰਾ   = ਗੰਢ_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    ਜੇ (with_norm && !with_scale) {
        ਜੇ (raw_ਸਿਰਾ + 1 >= gf->n_nodes) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ ggml_tensor * trailing_reshape = gf->nodes[raw_ਸਿਰਾ + 1];
        ਜੇ (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_ਸਿਰਾ]) {
            ਵਾਪਸ ਝੂਠ;
        }
        raw_ਸਿਰਾ++;
    }

    ਅਚਲ ਪੂਰਨ raw_ਗਿਣਤੀ = raw_ਸਿਰਾ - raw_ਸ਼ੁਰੂ + 1;
    ਜੇ (raw_ਗਿਣਤੀ != (ਪੂਰਨ) ops_all.size()) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਪੂਰਨ raw_idxs[GGML_METAL_FUSION_MAX];
    ਲਈ (ਪੂਰਨ i = 0; i < raw_ਗਿਣਤੀ; ++i) {
        raw_idxs[i] = raw_ਸ਼ੁਰੂ + i;
        ਜੇ (gf->nodes[raw_ਸ਼ੁਰੂ + i]->op != ops_all[i]) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਅਚਲ ggml_tensor * softmax        = gf->nodes[raw_ਸ਼ੁਰੂ];
    ਅਚਲ ggml_tensor * probs_reshaped = gf->nodes[raw_ਸ਼ੁਰੂ + 1];
    ਅਚਲ ggml_tensor * argsort        = gf->nodes[raw_ਸ਼ੁਰੂ + 2];
    ਅਚਲ ggml_tensor * ids            = gf->nodes[raw_ਸ਼ੁਰੂ + 3];
    ਅਚਲ ggml_tensor * get_ਪੰਕਤੀਆਂ       = gf->nodes[raw_ਸ਼ੁਰੂ + 4];
    ਅਚਲ ggml_tensor * ਬਾਹਰਲਾ            = gf->nodes[raw_ਸਿਰਾ];
    ਅਚਲ ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    ਦਸ਼ਮਲਵ scale   = 1.0f;
    ਦਸ਼ਮਲਵ max_bias = 0.0f;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale,    ((ਅਚਲ int32_t *) softmax->op_params) + 0, ਆਕਾਰ(scale));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, ((ਅਚਲ int32_t *) softmax->op_params) + 1, ਆਕਾਰ(max_bias));
    ਜੇ (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        ਬਾਹਰਲਾ->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int64_t n_expert      = logits->ne[0];
    ਅਚਲ int64_t n_ਟੋਕਨ      = logits->ne[1];
    ਅਚਲ int64_t n_expert_ਵਰਤਿਆ = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    ਜੇ (n_expert <= 0 || n_expert_ਵਰਤਿਆ <= 0 || n_expert_ਵਰਤਿਆ > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_ਵਰਤਿਆ > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_ਟੋਕਨ || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        ਬਾਹਰਲਾ->ne[0] != 1 || ਬਾਹਰਲਾ->ne[1] != n_expert_ਵਰਤਿਆ || ਬਾਹਰਲਾ->ne[2] != n_ਟੋਕਨ || ਬਾਹਰਲਾ->ne[3] != 1) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(logits) || !ggml_is_contiguous(ਬਾਹਰਲਾ) ||
        ids->nb[0] != ggml_type_ਮਾਪ(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_ਮਾਪ(GGML_TYPE_I32) * n_expert) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_ਪੰਕਤੀਆਂ->src[0] != probs_reshaped || get_ਪੰਕਤੀਆਂ->src[1] != ids) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (with_norm) {
        ਅਚਲ ggml_tensor * weights_reshaped = gf->nodes[raw_ਸ਼ੁਰੂ + 5];
        ਅਚਲ ggml_tensor * ਜੋੜ_ਪੰਕਤੀਆਂ         = gf->nodes[raw_ਸ਼ੁਰੂ + 6];
        ਅਚਲ ggml_tensor * clamp            = gf->nodes[raw_ਸ਼ੁਰੂ + 7];
        ਅਚਲ ggml_tensor * div              = gf->nodes[raw_ਸ਼ੁਰੂ + 8];
        ਅਚਲ ggml_tensor * ਬਾਹਰਲਾ_reshaped     = gf->nodes[raw_ਸ਼ੁਰੂ + 9];

        ਜੇ (weights_reshaped->src[0] != get_ਪੰਕਤੀਆਂ || ਜੋੜ_ਪੰਕਤੀਆਂ->src[0] != weights_reshaped ||
            clamp->src[0] != ਜੋੜ_ਪੰਕਤੀਆਂ || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            ਬਾਹਰਲਾ_reshaped->src[0] != div) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (with_scale) {
            ਅਚਲ ggml_tensor * scale_ਗੰਢ = gf->nodes[raw_ਸ਼ੁਰੂ + 10];
            ਜੇ (scale_ਗੰਢ->src[0] != ਬਾਹਰਲਾ_reshaped) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (with_scale) {
        ਅਚਲ ggml_tensor * scale_ਗੰਢ = gf->nodes[raw_ਸ਼ੁਰੂ + 5];
        ਜੇ (scale_ਗੰਢ->src[0] != get_ਪੰਕਤੀਆਂ) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਅਚਲ ਪੂਰਨ outputs[2] = { raw_ਸ਼ੁਰੂ + 3, raw_ਸਿਰਾ };
    ਜੇ (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_ਗਿਣਤੀ, ops_all.ਅੰਕੜੇ(), outputs, 2)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (mode == GGML_METAL_FUSION_ਭਰਿਆ) {
        ਜੇ (!logits->ਅੰਕੜੇ || !ਬਾਹਰਲਾ->ਅੰਕੜੇ || !ids->ਅੰਕੜੇ) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਵਾਪਸ ਸੱਚ;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

ਬਣਤਰ ggml_metal_moe_reduce_ਮੇਲ {
    ਅਚਲ ggml_tensor * experts;
    ਅਚਲ ggml_tensor * weights;
    ਅਚਲ ggml_tensor * dst;
    ਪੂਰਨ ਗੰਢ_ਗਿਣਤੀ;
};

ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_fusion_ਮੇਲ_moe_reduce(
        ਅਚਲ ggml_cgraph * gf, ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_ਮੇਲ * ਮੇਲ) {
    ਜੇ (ਮੇਲ == ਖਾਲੀ_ਸੂਚਕ || ਗੰਢ_idx < 0 || ਗੰਢ_idx + (ਪੂਰਨ) ops_all.size() > gf->n_nodes) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ggml_tensor * mul = gf->nodes[ਗੰਢ_idx];
    ਜੇ (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    ਅਚਲ ਪੂਰਨ raw_ਗਿਣਤੀ     = (ਪੂਰਨ) ops_all.size();
    ਅਚਲ ਪੂਰਨ n_expert_ਵਰਤਿਆ = raw_ਗਿਣਤੀ / 2;

    ਜੇ (n_expert_ਵਰਤਿਆ < 2 || n_expert_ਵਰਤਿਆ > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_ਗਿਣਤੀ != 2 * n_expert_ਵਰਤਿਆ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਪੂਰਨ n_views = 0;
    ਜਦੋਂ (ਗੰਢ_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[ਗੰਢ_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    ਜੇ (n_views != n_expert_ਵਰਤਿਆ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਲਈ (ਪੂਰਨ i = n_expert_ਵਰਤਿਆ + 1; i < raw_ਗਿਣਤੀ; ++i) {
        ਜੇ (gf->nodes[ਗੰਢ_idx + i]->op != GGML_OP_ਜੋੜੋ) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਪੂਰਨ raw_idxs[GGML_METAL_FUSION_MAX];
    ਲਈ (ਪੂਰਨ i = 0; i < raw_ਗਿਣਤੀ; ++i) {
        raw_idxs[i] = ਗੰਢ_idx + i;
        ਜੇ (gf->nodes[ਗੰਢ_idx + i]->op != ops_all[i]) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਅਚਲ ggml_tensor * experts = mul->src[0];
    ਅਚਲ ggml_tensor * weights = mul->src[1];
    ਅਚਲ ggml_tensor * dst     = gf->nodes[ਗੰਢ_idx + raw_ਗਿਣਤੀ - 1];

    ਜੇ (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int64_t n_embd   = experts->ne[0];
    ਅਚਲ int64_t n_ਟੋਕਨ = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    ਜੇ (n_embd <= 0 || experts->ne[1] != n_expert_ਵਰਤਿਆ || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_ਵਰਤਿਆ || weights->ne[2] != n_ਟੋਕਨ || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_ਟੋਕਨ || dst->ne[2] != 1 || dst->ne[3] != 1) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਲਈ (ਪੂਰਨ i = 1; i <= n_expert_ਵਰਤਿਆ; ++i) {
        ਅਚਲ ggml_tensor * view = gf->nodes[ਗੰਢ_idx + i];
        ਜੇ (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_ਟੋਕਨ ||
            view->nb[1] != mul->nb[2]) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਅਚਲ ggml_tensor * prev_ਜੋੜੋ = ਖਾਲੀ_ਸੂਚਕ;
    ਲਈ (ਪੂਰਨ j = 1; j < n_expert_ਵਰਤਿਆ; ++j) {
        ਅਚਲ ggml_tensor * ਜੋੜੋ_2 = gf->nodes[ਗੰਢ_idx + n_expert_ਵਰਤਿਆ + j];
        ਅਚਲ ggml_tensor * rhs = gf->nodes[ਗੰਢ_idx + j + 1];
        ਅਚਲ ggml_tensor * lhs = j == 1 ? gf->nodes[ਗੰਢ_idx + 1] : prev_ਜੋੜੋ;
        ਜੇ (ਜੋੜੋ_2->src[0] != lhs || ਜੋੜੋ_2->src[1] != rhs) {
            ਵਾਪਸ ਝੂਠ;
        }
        prev_ਜੋੜੋ = ਜੋੜੋ_2;
    }

    ਅਚਲ ਪੂਰਨ outputs[1] = { ਗੰਢ_idx + raw_ਗਿਣਤੀ - 1 };
    ਜੇ (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_ਗਿਣਤੀ, ops_all.ਅੰਕੜੇ(), outputs, 1)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਮੇਲ->experts    = experts;
    ਮੇਲ->weights    = weights;
    ਮੇਲ->dst        = dst;
    ਮੇਲ->ਗੰਢ_ਗਿਣਤੀ = raw_ਗਿਣਤੀ;
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_fusion_check_moe_reduce(
        ਅਚਲ ggml_metal_fusion       * fusion,
        ਅਚਲ ggml_tensor * ਅਚਲ     * nodes,
        ਅਚਲ ggml_cgraph             * gf,
        ਅਚਲ ਪੂਰਨ                     * ਗੰਢ_idxs,
              ਪੂਰਨ                       idx,
              ggml_metal_fusion_ਬਹੁਲਕ    mode) {
    GGML_ਅਣਵਰਤਿਆ(nodes);

    ggml_metal_moe_reduce_ਮੇਲ ਮੇਲ;
    ਜੇ (!ggml_metal_fusion_ਮੇਲ_moe_reduce(gf, ਗੰਢ_idxs[idx], fusion->ops_all, &ਮੇਲ)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ ((ਪੂਰਨ) fusion->ops.size() != ਮੇਲ.experts->ne[1]) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ਪੂਰਨ raw_ਸਿਰਾ = ਗੰਢ_idxs[idx] + ਮੇਲ.ਗੰਢ_ਗਿਣਤੀ - 1;
    ਜੇ (ਗੰਢ_idxs[idx + (ਪੂਰਨ) fusion->ops.size() - 1] != raw_ਸਿਰਾ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (mode == GGML_METAL_FUSION_ਭਰਿਆ) {
        ਜੇ (!ਮੇਲ.experts->ਅੰਕੜੇ || !ਮੇਲ.weights->ਅੰਕੜੇ || !ਮੇਲ.dst->ਅੰਕੜੇ) {
            ਵਾਪਸ ਝੂਠ;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        ਜੇ (ggml_metal_fusion_overlap(ਮੇਲ.dst, ਮੇਲ.experts) || ggml_metal_fusion_overlap(ਮੇਲ.dst, ਮੇਲ.weights)) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਵਾਪਸ ਸੱਚ;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_tensor_is_ਭਾਰ(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਅਚਲ ggml_tensor * ਅਧਾਰ = t->view_src != ਨਿਰਾ ? t->view_src : t;

    ਵਾਪਸ ਅਧਾਰ->ਬਫਰ != ਨਿਰਾ && ggml_backend_ਬਫਰ_get_usage(ਅਧਾਰ->ਬਫਰ) == GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS;
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_tensor * ggml_metal_mul_mat_ਜੋੜੋ_ਸੰਕਾਰਜ(ਅਚਲ ਬਣਤਰ ggml_tensor * mm, ਅਚਲ ਬਣਤਰ ggml_tensor * ਜੋੜੋ_2) {
    ਜੇ (ਜੋੜੋ_2->op != GGML_OP_ਜੋੜੋ || (ਜੋੜੋ_2->src[0] == mm) == (ਜੋੜੋ_2->src[1] == mm)) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਅਚਲ ggml_tensor * other = ਜੋੜੋ_2->src[0] == mm ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];

    ਅਚਲ ਤਾਰਕਿਕ ok = other->type == GGML_TYPE_F32 && ਜੋੜੋ_2->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_ਭਾਰ(other);

    ਵਾਪਸ ok ? other : ਨਿਰਾ;
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_tensor * ggml_metal_mul_mat_ਜੋੜੋ_residual(ਅਚਲ ਬਣਤਰ ggml_tensor * mm, ਅਚਲ ਬਣਤਰ ggml_tensor * ਜੋੜੋ_2) {
    ਅਚਲ ggml_tensor * res = ggml_metal_mul_mat_ਜੋੜੋ_ਸੰਕਾਰਜ(mm, ਜੋੜੋ_2);

    ਅਚਲ ਤਾਰਕਿਕ ok = res != ਨਿਰਾ && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(ਜੋੜੋ_2);

    ਵਾਪਸ ok ? res : ਨਿਰਾ;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_fusion_check_mul_mat_ਜੋੜੋ(
        ਅਚਲ ggml_metal_fusion       * fusion,
        ਅਚਲ ggml_tensor * ਅਚਲ     * nodes,
        ਅਚਲ ggml_cgraph             * gf,
        ਅਚਲ ਪੂਰਨ                     * ਗੰਢ_idxs,
              ਪੂਰਨ                       idx,
              ggml_metal_fusion_ਬਹੁਲਕ    mode) {
    GGML_ਅਣਵਰਤਿਆ(gf);
    GGML_ਅਣਵਰਤਿਆ(ਗੰਢ_idxs);
    GGML_ਅਣਵਰਤਿਆ(idx);
    GGML_ਅਣਵਰਤਿਆ(fusion);

    ਅਚਲ ggml_tensor * mm  = nodes[0];
    ਅਚਲ ggml_tensor * ਜੋੜੋ_2 = nodes[1];

    ਜੇ (ggml_metal_mul_mat_ਜੋੜੋ_ਸੰਕਾਰਜ(mm, ਜੋੜੋ_2) == ਖਾਲੀ_ਸੂਚਕ ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (mode == GGML_METAL_FUSION_STRUCTURAL) {
        ਵਾਪਸ ਸੱਚ;
    }

    ਅਚਲ ggml_tensor * res = ggml_metal_mul_mat_ਜੋੜੋ_residual(mm, ਜੋੜੋ_2);

    ਜੇ (res == ਖਾਲੀ_ਸੂਚਕ || !ggml_metal_op_mul_mat_use_mma(mm)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ !ggml_metal_fusion_overlap(ਜੋੜੋ_2, mm->src[0]) && !ggml_metal_fusion_overlap(ਜੋੜੋ_2, mm->src[1]) &&
        (ਜੋੜੋ_2->ਅੰਕੜੇ == res->ਅੰਕੜੇ || !ggml_metal_fusion_overlap(ਜੋੜੋ_2, res));
}

// ---- patterns ------------------------------------------------------------

ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_norm_mul_ਜੋੜੋ     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ਜੋੜੋ };
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_rms_norm_mul_ਜੋੜੋ = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ਜੋੜੋ };
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_ਜੋੜੋ_2 = { GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ };
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_ਜੋੜੋ_3 = { GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ };
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_ਜੋੜੋ_4 = { GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ };
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_ਜੋੜੋ_5 = { GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ };
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_ਜੋੜੋ_6 = { GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ };
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_ਜੋੜੋ_7 = { GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ };
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ਜੋੜੋ };

ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ਜੋੜੋ
};
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ
};
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ
};
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ
};
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ
};
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ
};
ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ
};

ਸਥਿਰ ਅਚਲ std::vector<ggml_op> ops_mul_mat_ਜੋੜੋ = { GGML_OP_MUL_MAT, GGML_OP_ਜੋੜੋ };

ਸਥਿਰ ਅਚਲ std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     ਝੂਠ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ਜੋੜੋ,   ops_norm_mul_ਜੋੜੋ,           {},     ਝੂਠ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     ਝੂਠ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     ਝੂਠ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ਜੋੜੋ,   ops_rms_norm_mul_ਜੋੜੋ,       {},     ਝੂਠ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     ਝੂਠ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ਜੋੜੋ_CHAIN,      ops_ਜੋੜੋ_2,                  {},     ਝੂਠ, ggml_metal_fusion_check_ਜੋੜੋ_chain },
    { GGML_METAL_FUSION_ਜੋੜੋ_CHAIN,      ops_ਜੋੜੋ_3,                  {},     ਝੂਠ, ggml_metal_fusion_check_ਜੋੜੋ_chain },
    { GGML_METAL_FUSION_ਜੋੜੋ_CHAIN,      ops_ਜੋੜੋ_4,                  {},     ਝੂਠ, ggml_metal_fusion_check_ਜੋੜੋ_chain },
    { GGML_METAL_FUSION_ਜੋੜੋ_CHAIN,      ops_ਜੋੜੋ_5,                  {},     ਝੂਠ, ggml_metal_fusion_check_ਜੋੜੋ_chain },
    { GGML_METAL_FUSION_ਜੋੜੋ_CHAIN,      ops_ਜੋੜੋ_6,                  {},     ਝੂਠ, ggml_metal_fusion_check_ਜੋੜੋ_chain },
    { GGML_METAL_FUSION_ਜੋੜੋ_CHAIN,      ops_ਜੋੜੋ_7,                  {},     ਝੂਠ, ggml_metal_fusion_check_ਜੋੜੋ_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     ਝੂਠ, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     ਸੱਚ,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    ਸੱਚ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    ਸੱਚ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    ਸੱਚ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    ਸੱਚ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     ਸੱਚ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     ਸੱਚ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     ਸੱਚ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     ਸੱਚ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     ਸੱਚ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     ਸੱਚ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     ਸੱਚ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     ਝੂਠ, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_ਜੋੜੋ,    ops_mul_mat_ਜੋੜੋ,            {},     ਝੂਠ, ggml_metal_fusion_check_mul_mat_ਜੋੜੋ },
};

// ---- alloc deps -----------------------------------------------------------

ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_fusion_ਮੇਲ_raw_ਨਮੂਨਾ(
        ਅਚਲ ggml_cgraph * gf, ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ std::vector<ggml_op> & ops) {
    ਜੇ (ਗੰਢ_idx < 0 || ਗੰਢ_idx + (ਪੂਰਨ) ops.size() > gf->n_nodes) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < (ਪੂਰਨ) ops.size(); ++i) {
        ਜੇ (gf->nodes[ਗੰਢ_idx + i]->op != ops[i]) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਖਾਲੀ ggml_metal_fusion_ਜੋੜੋ_ਨਮੂਨਾ_alloc_deps(
        ਖਾਲੀ * user_ਅੰਕੜੇ,
        ਖਾਲੀ (*ਜੋੜੋ_alloc_dep)(ਖਾਲੀ *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        ਅਚਲ ggml_metal_fusion * fusion,
        ਪੂਰਨ ਗੰਢ_idx) {
    ਅਚਲ ਪੂਰਨ ਆਖਰੀ_ਗੰਢ = ਗੰਢ_idx + (ਪੂਰਨ) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> ਵੇਖਿਆ;
    ਲਈ (ਪੂਰਨ j = 0; j < (ਪੂਰਨ) fusion->ops_all.size(); ++j) {
        ggml_tensor * ਗੰਢ = gf->nodes[ਗੰਢ_idx + j];
        ਲਈ (ਪੂਰਨ s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = ਗੰਢ->src[s];
            ਜੇ (src && ਵੇਖਿਆ.insert(src).second) {
                ਜੋੜੋ_alloc_dep(user_ਅੰਕੜੇ, src, gf->nodes[ਆਖਰੀ_ਗੰਢ]);
            }
        }
        ਵੇਖਿਆ.insert(ਗੰਢ);
    }
}

ਖਾਲੀ ggml_metal_fusion_ਜੋੜੋ_alloc_deps(
        ਖਾਲੀ * user_ਅੰਕੜੇ,
        ਖਾਲੀ (*ਜੋੜੋ_alloc_dep)(ਖਾਲੀ *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    ਲਈ (ਪੂਰਨ i = 0; i < gf->n_nodes; ++i) {
        ਅਚਲ ggml_metal_fusion * ਉੱਤਮ = ਖਾਲੀ_ਸੂਚਕ;
        ਪੂਰਨ ਉੱਤਮ_raw = 0;

        ਲਈ (ਅਚਲ ggml_metal_fusion & fusion : ggml_metal_fusions) {
            ਜੇ ((ਪੂਰਨ) fusion.ops_all.size() <= ਉੱਤਮ_raw) {
                ਜਾਰੀ;
            }
            ਜੇ (ggml_metal_fusion_ਮੇਲ_raw_ਨਮੂਨਾ(gf, i, fusion.ops_all)) {
                ਉੱਤਮ = &fusion;
                ਉੱਤਮ_raw = (ਪੂਰਨ) fusion.ops_all.size();
            }
        }

        ਜੇ (ਉੱਤਮ) {
            ggml_metal_fusion_ਜੋੜੋ_ਨਮੂਨਾ_alloc_deps(user_ਅੰਕੜੇ, ਜੋੜੋ_alloc_dep, gf, ਉੱਤਮ, i);
            i += ਉੱਤਮ_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

ਸਥਿਰ std::string ggml_metal_fusion_label(ਅਚਲ ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != ਖਾਲੀ_ਸੂਚਕ);

    std::string label;
    ਲਈ (ਪੂਰਨ j = 0; j < (ਪੂਰਨ) fusion->ops.size(); j++) {
        ਜੇ (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    ਵਾਪਸ label;
}

ਬਣਤਰ ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    ਤਾਰਕਿਕ enabled;
    ਤਾਰਕਿਕ stats;
    ਤਾਰਕਿਕ labels_ਸੈੱਟ;
    ਪੂਰਨ  debug;
};

ਬਣਤਰ ggml_metal_fusion_info * ggml_metal_fusion_info_init(ਤਾਰਕਿਕ enabled, ਪੂਰਨ debug) {
    ggml_metal_fusion_info * finfo = ਨਵਾਂ ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_ਸੈੱਟ = ਝੂਠ;
    finfo->debug      = debug;

    ਜੇ (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    ਵਾਪਸ finfo;
}

ਖਾਲੀ ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    ਮਿਟਾਓ finfo;
}

ਤਾਰਕਿਕ ggml_metal_fusion_info_enabled(ਅਚਲ ggml_metal_fusion_info * finfo) {
    ਵਾਪਸ finfo->enabled;
}

ਤਾਰਕਿਕ ggml_metal_fusion_info_stats(ਅਚਲ ggml_metal_fusion_info * finfo) {
    ਵਾਪਸ finfo->stats;
}

ਪੂਰਨ ggml_metal_fusion_info_debug(ਅਚਲ ggml_metal_fusion_info * finfo) {
    ਵਾਪਸ finfo->debug;
}

ਪੂਰਨ ggml_metal_fusion_info_n_fusions(ਅਚਲ ggml_metal_fusion_info * finfo) {
    ਵਾਪਸ (ਪੂਰਨ) finfo->labels.size();
}

ਅਚਲ ਅੱਖਰ * ggml_metal_fusion_info_label(ਅਚਲ ggml_metal_fusion_info * finfo, ਪੂਰਨ idx) {
    GGML_ASSERT(idx >= 0 && idx < (ਪੂਰਨ) finfo->labels.size());
    ਵਾਪਸ finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_ਗਿਣਤੀ(ਅਚਲ ggml_metal_fusion_info * finfo, ਪੂਰਨ idx) {
    GGML_ASSERT(idx >= 0 && idx < (ਪੂਰਨ) finfo->counts.size());
    ਵਾਪਸ finfo->counts[idx];
}

ਖਾਲੀ ggml_metal_fusion_info_ਗਿਣਤੀ_fusion(ggml_metal_fusion_info * finfo, ਅਚਲ ggml_metal_fusion * fusion) {
    ਜੇ (!finfo->stats || fusion == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }

    ਅਚਲ ptrdiff_t idx = fusion - ggml_metal_fusions.ਅੰਕੜੇ();
    ਜੇ (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

ਖਾਲੀ ggml_metal_fusion_info_ਸੈੱਟ_enabled(ggml_metal_fusion_info * finfo, ਤਾਰਕਿਕ enabled) {
    finfo->enabled = enabled;
}

ਖਾਲੀ ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    ਜੇ (finfo->labels_ਸੈੱਟ) {
        ਵਾਪਸ;
    }

    finfo->labels.clear();
    finfo->counts.ਸੌਂਪੋ(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    ਲਈ (ਅਚਲ ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_ਸੈੱਟ = ਸੱਚ;
}

ਖਾਲੀ ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = ਸੱਚ;
    ggml_metal_fusion_info_labels_init(finfo);
}

ਖਾਲੀ ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

ਪੂਰਨ ggml_metal_fusion_info_stats_get(ਅਚਲ ggml_metal_fusion_info * finfo, ਅਚਲ ਅੱਖਰ ** labels, uint64_t * counts, ਪੂਰਨ n) {
    ਅਚਲ ਪੂਰਨ n_fusions = (ਪੂਰਨ) finfo->labels.size();

    ਜੇ (labels == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ n_fusions;
    }

    ਅਚਲ ਪੂਰਨ n_fill = std::min(n, n_fusions);
    ਲਈ (ਪੂਰਨ i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        ਜੇ (counts != ਖਾਲੀ_ਸੂਚਕ) {
            counts[i] = finfo->counts[i];
        }
    }

    ਵਾਪਸ n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
ਸਥਿਰ ਤਾਰਕਿਕ ggml_metal_fusion_check_ਸਮ੍ਰਿਤੀ_ranges(
        ਅਚਲ ggml_metal_fusion      * fusion,
        ਅਚਲ ggml_tensor * ਅਚਲ    * nodes,
        ਪੂਰਨ                            ਗੰਢ_ਗਿਣਤੀ) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    //ਸਰੋਤ
    ਸਵੈਚਾਲਿਤ same_ਸਮ੍ਰਿਤੀ = [](ਅਚਲ ggml_tensor * a, ਅਚਲ ggml_tensor * b) {
        ਜੇ (a->ਅੰਕੜੇ && b->ਅੰਕੜੇ && a->ਅੰਕੜੇ == b->ਅੰਕੜੇ) {
            ਵਾਪਸ ਸੱਚ;
        }
        ਲਈ (ਅਚਲ ggml_tensor * v = a; v; v = v->view_src) {
            ਜੇ (v == b) {
                ਵਾਪਸ ਸੱਚ;
            }
        }
        ਲਈ (ਅਚਲ ggml_tensor * v = b; v; v = v->view_src) {
            ਜੇ (v == a) {
                ਵਾਪਸ ਸੱਚ;
            }
        }
        ਵਾਪਸ ਝੂਠ;
    };

    ਸਵੈਚਾਲਿਤ nodes_overlap = [](ਅਚਲ ggml_tensor * a, ਅਚਲ ggml_tensor * b) {
        ਜੇ (!a || !b || !a->ਅੰਕੜੇ || !b->ਅੰਕੜੇ || !a->ਬਫਰ || !b->ਬਫਰ) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (a->ਬਫਰ != b->ਬਫਰ) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ int64_t a_ਸ਼ੁਰੂ = (int64_t) a->ਅੰਕੜੇ;
        ਅਚਲ int64_t a_ਸਿਰਾ   = a_ਸ਼ੁਰੂ + ggml_backend_buft_get_alloc_ਮਾਪ(a->ਬਫਰ->buft, a);
        ਅਚਲ int64_t b_ਸ਼ੁਰੂ = (int64_t) b->ਅੰਕੜੇ;
        ਅਚਲ int64_t b_ਸਿਰਾ   = b_ਸ਼ੁਰੂ + ggml_backend_buft_get_alloc_ਮਾਪ(b->ਬਫਰ->buft, b);

        ਵਾਪਸ (b_ਸ਼ੁਰੂ <= a_ਸ਼ੁਰੂ && a_ਸ਼ੁਰੂ < b_ਸਿਰਾ) ||
               (a_ਸ਼ੁਰੂ <= b_ਸ਼ੁਰੂ && b_ਸ਼ੁਰੂ < a_ਸਿਰਾ);
    };

    ਸਵੈਚਾਲਿਤ is_intermediate = [](ਅਚਲ ggml_tensor * src, ਅਚਲ ggml_tensor * ਅਚਲ * nodes, ਪੂਰਨ j) {
        ਲਈ (ਪੂਰਨ k = 0; k < j; ++k) {
            ਜੇ (src == nodes[k]) {
                ਵਾਪਸ ਸੱਚ;
            }
            ਲਈ (ਅਚਲ ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                ਜੇ (view_src == nodes[k]) {
                    ਵਾਪਸ ਸੱਚ;
                }
            }
        }
        ਵਾਪਸ ਝੂਠ;
    };

    ਸਵੈਚਾਲਿਤ check_dst = [&](ਅਚਲ ggml_tensor * dst) {
        ਲਈ (ਪੂਰਨ j = 0; j < ਗੰਢ_ਗਿਣਤੀ; ++j) {
            ਲਈ (ਪੂਰਨ s = 0; s < GGML_MAX_SRC; ++s) {
                ਅਚਲ ggml_tensor * src = nodes[j]->src[s];
                ਜੇ (!src || src->op == GGML_OP_NONE || same_ਸਮ੍ਰਿਤੀ(src, dst)) {
                    ਜਾਰੀ;
                }

                ਜੇ (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    ਵਾਪਸ ਝੂਠ;
                }
            }
        }
        ਵਾਪਸ ਸੱਚ;
    };

    ਜੇ (!check_dst(nodes[ਗੰਢ_ਗਿਣਤੀ - 1])) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਲਈ (ਪੂਰਨ ਵਿੱਥ : fusion->outs) {
        GGML_ASSERT(ਵਿੱਥ >= 0 && ਵਿੱਥ < ਗੰਢ_ਗਿਣਤੀ);
        ਜੇ (!check_dst(nodes[ਵਿੱਥ])) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਵਾਪਸ ਸੱਚ;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
ਅਚਲ ggml_metal_fusion * ggml_metal_fusion_ਅਗਲਾ(
        ਅਚਲ ggml_cgraph * gf,
        ਅਚਲ ਪੂਰਨ * ਗੰਢ_idxs,
        ਪੂਰਨ n_idxs,
        ਪੂਰਨ idx,
        ggml_metal_fusion_ਬਹੁਲਕ mode,
        ਪੂਰਨ * n_ਬਾਹਰਲਾ) {
    ਅਚਲ ggml_metal_fusion * res = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ ਉੱਤਮ = 1;

    ਲਈ (ਅਚਲ ggml_metal_fusion & fusion : ggml_metal_fusions) {
        ਅਚਲ ਪੂਰਨ n_ops = (ਪੂਰਨ) fusion.ops.size();

        // only look for a longer match than the current best
        ਜੇ (n_ops <= ਉੱਤਮ) {
            ਜਾਰੀ;
        }
        ਜੇ (idx + n_ops > n_idxs) {
            ਜਾਰੀ;
        }

        ਅਚਲ ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        ਤਾਰਕਿਕ ok = ਸੱਚ;
        ਲਈ (ਪੂਰਨ j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[ਗੰਢ_idxs[idx + j]];
            ਜੇ (nodes[j]->op != fusion.ops[j]) {
                ok = ਝੂਠ;
                ਰੋਕੋ;
            }
        }
        ਜੇ (!ok) {
            ਜਾਰੀ;
        }

        ਜੇ (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            ਲਈ (ਪੂਰਨ j = 1; j < n_ops && ok; j++) {
                ਜੇ (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = ਝੂਠ;
                    ਰੋਕੋ;
                }
                ਜੇ (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = ਝੂਠ;
                    ਰੋਕੋ;
                }
            }
            ਜੇ (!ok) {
                ਜਾਰੀ;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            ਪੂਰਨ outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = ਗੰਢ_idxs[idx + n_ops - 1];
            ਲਈ (size_t i = 0; i < fusion.outs.size(); ++i) {
                ਅਚਲ ਪੂਰਨ ਬਾਹਰਲਾ_ਵਿੱਥ = fusion.outs[i];
                GGML_ASSERT(ਬਾਹਰਲਾ_ਵਿੱਥ >= 0 && ਬਾਹਰਲਾ_ਵਿੱਥ < n_ops);
                outputs_buf[i + 1] = ਗੰਢ_idxs[idx + ਬਾਹਰਲਾ_ਵਿੱਥ];
            }

            ਅਚਲ ਪੂਰਨ n_outputs = 1 + (ਪੂਰਨ) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            ਜੇ (!ggml_can_fuse_subgraph_ext(gf, ਗੰਢ_idxs + idx, n_ops, fusion.ops.ਅੰਕੜੇ(), outputs_buf, n_outputs)) {
                ਜਾਰੀ;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        ਜੇ (fusion.check && !fusion.check(&fusion, nodes, gf, ਗੰਢ_idxs, idx, mode)) {
            ਜਾਰੀ;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        ਜੇ (mode == GGML_METAL_FUSION_ਭਰਿਆ &&
                !ggml_metal_fusion_check_ਸਮ੍ਰਿਤੀ_ranges(&fusion, nodes, n_ops)) {
            ਜਾਰੀ;
        }

        ਉੱਤਮ = n_ops;
        res = &fusion;
    }

    *n_ਬਾਹਰਲਾ = ਉੱਤਮ;

    ਵਾਪਸ res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
ਪੂਰਨ ggml_metal_fusion_max(ਅਚਲ ggml_cgraph * gf, ਪੂਰਨ idx) {
    // a view node cannot start a pattern - pack it alone
    ਜੇ (ggml_op_is_ਖਾਲੀ_ਥਾਵਾਂ(gf->nodes[idx]->op)) {
        ਵਾਪਸ 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    ਪੂਰਨ idxs[GGML_METAL_FUSION_MAX];
    ਪੂਰਨ n_idxs = 0;
    ਲਈ (ਪੂਰਨ i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        ਜੇ (!ggml_op_is_ਖਾਲੀ_ਥਾਵਾਂ(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    ਜੇ (n_idxs == 0) {
        ਵਾਪਸ 1;
    }

    ਪੂਰਨ ਕੁੱਲ = 0;
    ਪੂਰਨ i_f = 0;

    ਜਦੋਂ (i_f < n_idxs && ਕੁੱਲ < GGML_METAL_FUSION_MAX) {
        ਪੂਰਨ len = 1;
        ਅਚਲ ggml_metal_fusion * fusion = ggml_metal_fusion_ਅਗਲਾ(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        ਜੇ (!fusion || ਕੁੱਲ + len > GGML_METAL_FUSION_MAX) {
            ਰੋਕੋ;
        }

        ਕੁੱਲ += len;
        i_f += len;
    }

    ਜੇ (i_f == 0) {
        ਵਾਪਸ 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    ਵਾਪਸ std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
