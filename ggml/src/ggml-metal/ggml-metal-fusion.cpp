#ಅನ್ತರ್ಭಾವ "ggml-metal-fusion.h"

#ಅನ್ತರ್ಭಾವ "ggml-metal-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-metal-device.h"

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <set>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
ಸ್ಥಿರ std::vector<ggml_op> ggml_metal_fusion_filter_ops(ಧ್ರುವ std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    ಪ್ರತಿ (ggml_op op : ops_all) {
        ಯದಿ (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    ಪ್ರತ್ಯಾಗಮ ops;
}

ಸಂರಚನಾ ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<ಪೂರ್ಣಾಙ್ಕ>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    ತಾರ್ಕಿಕ unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    ತಾರ್ಕಿಕ (*check)(ಧ್ರುವ ಸಂರಚನಾ ggml_metal_fusion   * fusion,
                  ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * ಧ್ರುವ * nodes,
                  ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph         * gf,
                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ                        * ಗಂಟು_idxs,
                        ಪೂರ್ಣಾಙ್ಕ                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            ಧ್ರುವ std::vector<ggml_op> & ops_all,
            ಧ್ರುವ std::vector<ಪೂರ್ಣಾಙ್ಕ> & outs,
            ತಾರ್ಕಿಕ unsafe,
            ತಾರ್ಕಿಕ (*check)(ಧ್ರುವ ಸಂರಚನಾ ggml_metal_fusion   * fusion,
                          ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * ಧ್ರುವ * nodes,
                          ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph         * gf,
                          ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ                        * ಗಂಟು_idxs,
                                ಪೂರ್ಣಾಙ್ಕ                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(ಧ್ರುವ ggml_metal_fusion * fusion) {
    ಪ್ರತ್ಯಾಗಮ fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_same_buffer(ಧ್ರುವ ggml_tensor * a, ಧ್ರುವ ggml_tensor * b) {
    ಯದಿ (!a || !b) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    ಪ್ರತ್ಯಾಗಮ ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_overlap(ಧ್ರುವ ggml_tensor * a, ಧ್ರುವ ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ಧ್ರುವ ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    ಧ್ರುವ ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    ಯದಿ (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_norm(
        ಧ್ರುವ ggml_metal_fusion       * fusion,
        ಧ್ರುವ ggml_tensor * ಧ್ರುವ     * nodes,
        ಧ್ರುವ ggml_cgraph             * gf,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಙ್ಕ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(ಗಂಟು_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    ಯದಿ (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        ಧ್ರುವ ggml_tensor * scale = nodes[1];
        ಯದಿ (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 1; j < (ಪೂರ್ಣಾಙ್ಕ) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        ಯದಿ (nodes[j]->src[0] != nodes[j - 1]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // the weight/bias must have the same row width as the norm input
        ಯದಿ (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (nodes[j]->type != GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// SSM_CONV + UNARY (silu)
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_ssm_conv_silu(
        ಧ್ರುವ ggml_metal_fusion       * fusion,
        ಧ್ರುವ ggml_tensor * ಧ್ರುವ     * nodes,
        ಧ್ರುವ ggml_cgraph             * gf,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಙ್ಕ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(ಗಂಟು_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    ಧ್ರುವ ggml_tensor * conv = nodes[0];
    ಧ್ರುವ ggml_tensor * un   = nodes[1];

    ಯದಿ (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_add_chain(
        ಧ್ರುವ ggml_metal_fusion       * fusion,
        ಧ್ರುವ ggml_tensor * ಧ್ರುವ     * nodes,
        ಧ್ರುವ ggml_cgraph             * gf,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಙ್ಕ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(ಗಂಟು_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 1; j < (ಪೂರ್ಣಾಙ್ಕ) fusion->ops.size(); j++) {
        ಯದಿ (nodes[j]->src[0] != nodes[j - 1]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (mode == GGML_METAL_FUSION_FULL) {
            ಯದಿ (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_gdn_cache(
        ಧ್ರುವ ggml_metal_fusion       * fusion,
        ಧ್ರುವ ggml_tensor * ಧ್ರುವ     * nodes,
        ಧ್ರುವ ggml_cgraph             * gf,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಙ್ಕ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(ಗಂಟು_idxs);
    GGML_UNUSED(idx);

    ಧ್ರುವ ggml_tensor * gdn = nodes[0];
    ಧ್ರುವ ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    ಯದಿ (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int64_t S_v      = gdn->src[2]->ne[0];
    ಧ್ರುವ int64_t H        = gdn->src[2]->ne[1];
    ಧ್ರುವ int64_t n_tokens = gdn->src[2]->ne[2];
    ಧ್ರುವ int64_t n_seqs   = gdn->src[2]->ne[3];
    ಧ್ರುವ int64_t K        = ggml_get_op_params_i32(gdn, 0);
    ಧ್ರುವ size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    ಧ್ರುವ int64_t D         = S_v * S_v * H;
    ಧ್ರುವ int64_t n_written = std::min<int64_t>(n_tokens, K);

    ಧ್ರುವ ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    ಧ್ರುವ ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    ಯದಿ (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    ಯದಿ (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        ಯದಿ (dst->data == nullptr) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_snake(
        ಧ್ರುವ ggml_metal_fusion       * fusion,
        ಧ್ರುವ ggml_tensor * ಧ್ರುವ     * nodes,
        ಧ್ರುವ ggml_cgraph             * gf,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಙ್ಕ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(ಗಂಟು_idxs);
    GGML_UNUSED(idx);

    ಧ್ರುವ ggml_tensor * mul0     = nodes[0];
    ಧ್ರುವ ggml_tensor * sin_ಗಂಟು = nodes[1];
    ಧ್ರುವ ggml_tensor * sqr      = nodes[2];
    ಧ್ರುವ ggml_tensor * mul1     = nodes[3];
    ಧ್ರುವ ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    ಧ್ರುವ ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ಧ್ರುವ ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ಧ್ರುವ ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    ಧ್ರುವ ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    ಧ್ರುವ ತಾರ್ಕಿಕ types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_ಗಂಟು->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    ಧ್ರುವ ತಾರ್ಕಿಕ shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    ಧ್ರುವ ತಾರ್ಕಿಕ dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    ಧ್ರುವ ತಾರ್ಕಿಕ contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    ಪ್ರತ್ಯಾಗಮ types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_topk_moe(
        ಧ್ರುವ ggml_metal_fusion       * fusion,
        ಧ್ರುವ ggml_tensor * ಧ್ರುವ     * nodes,
        ಧ್ರುವ ggml_cgraph             * gf,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಙ್ಕ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_ops = (ಪೂರ್ಣಾಙ್ಕ) fusion->ops.size();

    ಧ್ರುವ ತಾರ್ಕಿಕ with_norm  = n_ops >= 6;
    ಧ್ರುವ ತಾರ್ಕಿಕ with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    ಧ್ರುವ std::vector<ggml_op> & ops_all = fusion->ops_all;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ raw_start = ಗಂಟು_idxs[idx];
    ಪೂರ್ಣಾಙ್ಕ       raw_end   = ಗಂಟು_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    ಯದಿ (with_norm && !with_scale) {
        ಯದಿ (raw_end + 1 >= gf->n_nodes) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಧ್ರುವ ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        ಯದಿ (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        raw_end++;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ raw_ಎಣಿಕೆ = raw_end - raw_start + 1;
    ಯದಿ (raw_ಎಣಿಕೆ != (ಪೂರ್ಣಾಙ್ಕ) ops_all.size()) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪೂರ್ಣಾಙ್ಕ raw_idxs[GGML_METAL_FUSION_MAX];
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < raw_ಎಣಿಕೆ; ++i) {
        raw_idxs[i] = raw_start + i;
        ಯದಿ (gf->nodes[raw_start + i]->op != ops_all[i]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಧ್ರುವ ggml_tensor * softmax        = gf->nodes[raw_start];
    ಧ್ರುವ ggml_tensor * probs_reshaped = gf->nodes[raw_start + 1];
    ಧ್ರುವ ggml_tensor * argsort        = gf->nodes[raw_start + 2];
    ಧ್ರುವ ggml_tensor * ids            = gf->nodes[raw_start + 3];
    ಧ್ರುವ ggml_tensor * get_rows       = gf->nodes[raw_start + 4];
    ಧ್ರುವ ggml_tensor * out            = gf->nodes[raw_end];
    ಧ್ರುವ ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    ಭಿನ್ನ scale   = 1.0f;
    ಭಿನ್ನ max_bias = 0.0f;
    memcpy(&scale,    ((ಧ್ರುವ int32_t *) softmax->op_params) + 0, ಪರಿಮಾಣ(scale));
    memcpy(&max_bias, ((ಧ್ರುವ int32_t *) softmax->op_params) + 1, ಪರಿಮಾಣ(max_bias));
    ಯದಿ (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int64_t n_expert      = logits->ne[0];
    ಧ್ರುವ int64_t n_tokens      = logits->ne[1];
    ಧ್ರುವ int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    ಯದಿ (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (with_norm) {
        ಧ್ರುವ ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
        ಧ್ರುವ ggml_tensor * sum_rows         = gf->nodes[raw_start + 6];
        ಧ್ರುವ ggml_tensor * clamp            = gf->nodes[raw_start + 7];
        ಧ್ರುವ ggml_tensor * div              = gf->nodes[raw_start + 8];
        ಧ್ರುವ ggml_tensor * out_reshaped     = gf->nodes[raw_start + 9];

        ಯದಿ (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            clamp->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            out_reshaped->src[0] != div) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (with_scale) {
            ಧ್ರುವ ggml_tensor * scale_ಗಂಟು = gf->nodes[raw_start + 10];
            ಯದಿ (scale_ಗಂಟು->src[0] != out_reshaped) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
    } ಅನ್ಯಥಾ ಯದಿ (with_scale) {
        ಧ್ರುವ ggml_tensor * scale_ಗಂಟು = gf->nodes[raw_start + 5];
        ಯದಿ (scale_ಗಂಟು->src[0] != get_rows) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ outputs[2] = { raw_start + 3, raw_end };
    ಯದಿ (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_ಎಣಿಕೆ, ops_all.data(), outputs, 2)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (mode == GGML_METAL_FUSION_FULL) {
        ಯದಿ (!logits->data || !out->data || !ids->data) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

ಸಂರಚನಾ ggml_metal_moe_reduce_match {
    ಧ್ರುವ ggml_tensor * experts;
    ಧ್ರುವ ggml_tensor * weights;
    ಧ್ರುವ ggml_tensor * dst;
    ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_ಎಣಿಕೆ;
};

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_match_moe_reduce(
        ಧ್ರುವ ggml_cgraph * gf, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    ಯದಿ (match == nullptr || ಗಂಟು_idx < 0 || ಗಂಟು_idx + (ಪೂರ್ಣಾಙ್ಕ) ops_all.size() > gf->n_nodes) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ggml_tensor * mul = gf->nodes[ಗಂಟು_idx];
    ಯದಿ (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ raw_ಎಣಿಕೆ     = (ಪೂರ್ಣಾಙ್ಕ) ops_all.size();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_expert_used = raw_ಎಣಿಕೆ / 2;

    ಯದಿ (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_ಎಣಿಕೆ != 2 * n_expert_used) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪೂರ್ಣಾಙ್ಕ n_views = 0;
    ಯಾವತ್ (ಗಂಟು_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[ಗಂಟು_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    ಯದಿ (n_views != n_expert_used) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = n_expert_used + 1; i < raw_ಎಣಿಕೆ; ++i) {
        ಯದಿ (gf->nodes[ಗಂಟು_idx + i]->op != GGML_OP_ADD) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪೂರ್ಣಾಙ್ಕ raw_idxs[GGML_METAL_FUSION_MAX];
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < raw_ಎಣಿಕೆ; ++i) {
        raw_idxs[i] = ಗಂಟು_idx + i;
        ಯದಿ (gf->nodes[ಗಂಟು_idx + i]->op != ops_all[i]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಧ್ರುವ ggml_tensor * experts = mul->src[0];
    ಧ್ರುವ ggml_tensor * weights = mul->src[1];
    ಧ್ರುವ ggml_tensor * dst     = gf->nodes[ಗಂಟು_idx + raw_ಎಣಿಕೆ - 1];

    ಯದಿ (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ int64_t n_embd   = experts->ne[0];
    ಧ್ರುವ int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    ಯದಿ (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 1; i <= n_expert_used; ++i) {
        ಧ್ರುವ ggml_tensor * view = gf->nodes[ಗಂಟು_idx + i];
        ಯದಿ (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಧ್ರುವ ggml_tensor * prev_add = nullptr;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 1; j < n_expert_used; ++j) {
        ಧ್ರುವ ggml_tensor * add = gf->nodes[ಗಂಟು_idx + n_expert_used + j];
        ಧ್ರುವ ggml_tensor * rhs = gf->nodes[ಗಂಟು_idx + j + 1];
        ಧ್ರುವ ggml_tensor * lhs = j == 1 ? gf->nodes[ಗಂಟು_idx + 1] : prev_add;
        ಯದಿ (add->src[0] != lhs || add->src[1] != rhs) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        prev_add = add;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ outputs[1] = { ಗಂಟು_idx + raw_ಎಣಿಕೆ - 1 };
    ಯದಿ (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_ಎಣಿಕೆ, ops_all.data(), outputs, 1)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->ಗಂಟು_ಎಣಿಕೆ = raw_ಎಣಿಕೆ;
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_moe_reduce(
        ಧ್ರುವ ggml_metal_fusion       * fusion,
        ಧ್ರುವ ggml_tensor * ಧ್ರುವ     * nodes,
        ಧ್ರುವ ggml_cgraph             * gf,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಙ್ಕ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    ಯದಿ (!ggml_metal_fusion_match_moe_reduce(gf, ಗಂಟು_idxs[idx], fusion->ops_all, &match)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ ((ಪೂರ್ಣಾಙ್ಕ) fusion->ops.size() != match.experts->ne[1]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ raw_end = ಗಂಟು_idxs[idx] + match.ಗಂಟು_ಎಣಿಕೆ - 1;
    ಯದಿ (ಗಂಟು_idxs[idx + (ಪೂರ್ಣಾಙ್ಕ) fusion->ops.size() - 1] != raw_end) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (mode == GGML_METAL_FUSION_FULL) {
        ಯದಿ (!match.experts->data || !match.weights->data || !match.dst->data) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        ಯದಿ (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_tensor_is_weight(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಧ್ರುವ ggml_tensor * base = t->view_src != ರಿಕ್ತ ? t->view_src : t;

    ಪ್ರತ್ಯಾಗಮ base->buffer != ರಿಕ್ತ && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * ggml_metal_mul_mat_add_operand(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * mm, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * add) {
    ಯದಿ (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಧ್ರುವ ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    ಧ್ರುವ ತಾರ್ಕಿಕ ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    ಪ್ರತ್ಯಾಗಮ ok ? other : ರಿಕ್ತ;
}

ಸ್ಥಿರ ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * ggml_metal_mul_mat_add_residual(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * mm, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * add) {
    ಧ್ರುವ ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    ಧ್ರುವ ತಾರ್ಕಿಕ ok = res != ರಿಕ್ತ && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    ಪ್ರತ್ಯಾಗಮ ok ? res : ರಿಕ್ತ;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_mul_mat_add(
        ಧ್ರುವ ggml_metal_fusion       * fusion,
        ಧ್ರುವ ggml_tensor * ಧ್ರುವ     * nodes,
        ಧ್ರುವ ggml_cgraph             * gf,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಙ್ಕ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(ಗಂಟು_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    ಧ್ರುವ ggml_tensor * mm  = nodes[0];
    ಧ್ರುವ ggml_tensor * add = nodes[1];

    ಯದಿ (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (mode == GGML_METAL_FUSION_STRUCTURAL) {
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಧ್ರುವ ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    ಯದಿ (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };

ಸ್ಥಿರ ಧ್ರುವ std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     ಅಸತ್ಯ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_norm_mul_add,           {},     ಅಸತ್ಯ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     ಅಸತ್ಯ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     ಅಸತ್ಯ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_rms_norm_mul_add,       {},     ಅಸತ್ಯ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     ಅಸತ್ಯ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_2,                  {},     ಅಸತ್ಯ, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_3,                  {},     ಅಸತ್ಯ, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_4,                  {},     ಅಸತ್ಯ, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_5,                  {},     ಅಸತ್ಯ, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_6,                  {},     ಅಸತ್ಯ, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_7,                  {},     ಅಸತ್ಯ, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     ಅಸತ್ಯ, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     ಸತ್ಯ,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    ಸತ್ಯ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    ಸತ್ಯ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    ಸತ್ಯ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    ಸತ್ಯ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     ಸತ್ಯ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     ಸತ್ಯ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     ಸತ್ಯ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     ಸತ್ಯ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     ಸತ್ಯ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     ಸತ್ಯ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     ಸತ್ಯ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     ಅಸತ್ಯ, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_ADD,    ops_mul_mat_add,            {},     ಅಸತ್ಯ, ggml_metal_fusion_check_mul_mat_add },
};

// ---- alloc deps -----------------------------------------------------------

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_match_raw_pattern(
        ಧ್ರುವ ggml_cgraph * gf, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::vector<ggml_op> & ops) {
    ಯದಿ (ಗಂಟು_idx < 0 || ಗಂಟು_idx + (ಪೂರ್ಣಾಙ್ಕ) ops.size() > gf->n_nodes) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < (ಪೂರ್ಣಾಙ್ಕ) ops.size(); ++i) {
        ಯದಿ (gf->nodes[ಗಂಟು_idx + i]->op != ops[i]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_metal_fusion_add_pattern_alloc_deps(
        ಶೂನ್ಯ * user_data,
        ಶೂನ್ಯ (*add_alloc_dep)(ಶೂನ್ಯ *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        ಧ್ರುವ ggml_metal_fusion * fusion,
        ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ last_ಗಂಟು = ಗಂಟು_idx + (ಪೂರ್ಣಾಙ್ಕ) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < (ಪೂರ್ಣಾಙ್ಕ) fusion->ops_all.size(); ++j) {
        ggml_tensor * ಗಂಟು = gf->nodes[ಗಂಟು_idx + j];
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = ಗಂಟು->src[s];
            ಯದಿ (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_ಗಂಟು]);
            }
        }
        seen.insert(ಗಂಟು);
    }
}

ಶೂನ್ಯ ggml_metal_fusion_add_alloc_deps(
        ಶೂನ್ಯ * user_data,
        ಶೂನ್ಯ (*add_alloc_dep)(ಶೂನ್ಯ *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < gf->n_nodes; ++i) {
        ಧ್ರುವ ggml_metal_fusion * best = nullptr;
        ಪೂರ್ಣಾಙ್ಕ best_raw = 0;

        ಪ್ರತಿ (ಧ್ರುವ ggml_metal_fusion & fusion : ggml_metal_fusions) {
            ಯದಿ ((ಪೂರ್ಣಾಙ್ಕ) fusion.ops_all.size() <= best_raw) {
                ಅನುವರ್ತ;
            }
            ಯದಿ (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (ಪೂರ್ಣಾಙ್ಕ) fusion.ops_all.size();
            }
        }

        ಯದಿ (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

ಸ್ಥಿರ std::string ggml_metal_fusion_label(ಧ್ರುವ ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < (ಪೂರ್ಣಾಙ್ಕ) fusion->ops.size(); j++) {
        ಯದಿ (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    ಪ್ರತ್ಯಾಗಮ label;
}

ಸಂರಚನಾ ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    ತಾರ್ಕಿಕ enabled;
    ತಾರ್ಕಿಕ stats;
    ತಾರ್ಕಿಕ labels_set;
    ಪೂರ್ಣಾಙ್ಕ  debug;
};

ಸಂರಚನಾ ggml_metal_fusion_info * ggml_metal_fusion_info_init(ತಾರ್ಕಿಕ enabled, ಪೂರ್ಣಾಙ್ಕ debug) {
    ggml_metal_fusion_info * finfo = ನೂತನ ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = ಅಸತ್ಯ;
    finfo->debug      = debug;

    ಯದಿ (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    ಪ್ರತ್ಯಾಗಮ finfo;
}

ಶೂನ್ಯ ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

ತಾರ್ಕಿಕ ggml_metal_fusion_info_enabled(ಧ್ರುವ ggml_metal_fusion_info * finfo) {
    ಪ್ರತ್ಯಾಗಮ finfo->enabled;
}

ತಾರ್ಕಿಕ ggml_metal_fusion_info_stats(ಧ್ರುವ ggml_metal_fusion_info * finfo) {
    ಪ್ರತ್ಯಾಗಮ finfo->stats;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_fusion_info_debug(ಧ್ರುವ ggml_metal_fusion_info * finfo) {
    ಪ್ರತ್ಯಾಗಮ finfo->debug;
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_fusion_info_n_fusions(ಧ್ರುವ ggml_metal_fusion_info * finfo) {
    ಪ್ರತ್ಯಾಗಮ (ಪೂರ್ಣಾಙ್ಕ) finfo->labels.size();
}

ಧ್ರುವ ವರ್ಣ * ggml_metal_fusion_info_label(ಧ್ರುವ ggml_metal_fusion_info * finfo, ಪೂರ್ಣಾಙ್ಕ idx) {
    GGML_ASSERT(idx >= 0 && idx < (ಪೂರ್ಣಾಙ್ಕ) finfo->labels.size());
    ಪ್ರತ್ಯಾಗಮ finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_ಎಣಿಕೆ(ಧ್ರುವ ggml_metal_fusion_info * finfo, ಪೂರ್ಣಾಙ್ಕ idx) {
    GGML_ASSERT(idx >= 0 && idx < (ಪೂರ್ಣಾಙ್ಕ) finfo->counts.size());
    ಪ್ರತ್ಯಾಗಮ finfo->counts[idx];
}

ಶೂನ್ಯ ggml_metal_fusion_info_ಎಣಿಕೆ_fusion(ggml_metal_fusion_info * finfo, ಧ್ರುವ ggml_metal_fusion * fusion) {
    ಯದಿ (!finfo->stats || fusion == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ptrdiff_t idx = fusion - ggml_metal_fusions.data();
    ಯದಿ (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

ಶೂನ್ಯ ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, ತಾರ್ಕಿಕ enabled) {
    finfo->enabled = enabled;
}

ಶೂನ್ಯ ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    ಯದಿ (finfo->labels_set) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    ಪ್ರತಿ (ಧ್ರುವ ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = ಸತ್ಯ;
}

ಶೂನ್ಯ ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = ಸತ್ಯ;
    ggml_metal_fusion_info_labels_init(finfo);
}

ಶೂನ್ಯ ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

ಪೂರ್ಣಾಙ್ಕ ggml_metal_fusion_info_stats_get(ಧ್ರುವ ggml_metal_fusion_info * finfo, ಧ್ರುವ ವರ್ಣ ** labels, uint64_t * counts, ಪೂರ್ಣಾಙ್ಕ n) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_fusions = (ಪೂರ್ಣಾಙ್ಕ) finfo->labels.size();

    ಯದಿ (labels == nullptr) {
        ಪ್ರತ್ಯಾಗಮ n_fusions;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_fill = std::min(n, n_fusions);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        ಯದಿ (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    ಪ್ರತ್ಯಾಗಮ n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_memory_ranges(
        ಧ್ರುವ ggml_metal_fusion      * fusion,
        ಧ್ರುವ ggml_tensor * ಧ್ರುವ    * nodes,
        ಪೂರ್ಣಾಙ್ಕ                            ಗಂಟು_ಎಣಿಕೆ) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](ಧ್ರುವ ggml_tensor * a, ಧ್ರುವ ggml_tensor * b) {
        ಯದಿ (a->data && b->data && a->data == b->data) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಪ್ರತಿ (ಧ್ರುವ ggml_tensor * v = a; v; v = v->view_src) {
            ಯದಿ (v == b) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        }
        ಪ್ರತಿ (ಧ್ರುವ ggml_tensor * v = b; v; v = v->view_src) {
            ಯದಿ (v == a) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    };

    auto nodes_overlap = [](ಧ್ರುವ ggml_tensor * a, ಧ್ರುವ ggml_tensor * b) {
        ಯದಿ (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (a->buffer != b->buffer) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ int64_t a_start = (int64_t) a->data;
        ಧ್ರುವ int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        ಧ್ರುವ int64_t b_start = (int64_t) b->data;
        ಧ್ರುವ int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        ಪ್ರತ್ಯಾಗಮ (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](ಧ್ರುವ ggml_tensor * src, ಧ್ರುವ ggml_tensor * ಧ್ರುವ * nodes, ಪೂರ್ಣಾಙ್ಕ j) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ k = 0; k < j; ++k) {
            ಯದಿ (src == nodes[k]) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
            ಪ್ರತಿ (ಧ್ರುವ ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                ಯದಿ (view_src == nodes[k]) {
                    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
            }
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    };

    auto check_dst = [&](ಧ್ರುವ ggml_tensor * dst) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < ಗಂಟು_ಎಣಿಕೆ; ++j) {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 0; s < GGML_MAX_SRC; ++s) {
                ಧ್ರುವ ggml_tensor * src = nodes[j]->src[s];
                ಯದಿ (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    ಅನುವರ್ತ;
                }

                ಯದಿ (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
            }
        }
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    };

    ಯದಿ (!check_dst(nodes[ಗಂಟು_ಎಣಿಕೆ - 1])) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < ಗಂಟು_ಎಣಿಕೆ);
        ಯದಿ (!check_dst(nodes[offset])) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
ಧ್ರುವ ggml_metal_fusion * ggml_metal_fusion_next(
        ಧ್ರುವ ggml_cgraph * gf,
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_idxs,
        ಪೂರ್ಣಾಙ್ಕ n_idxs,
        ಪೂರ್ಣಾಙ್ಕ idx,
        ggml_metal_fusion_mode mode,
        ಪೂರ್ಣಾಙ್ಕ * n_out) {
    ಧ್ರುವ ggml_metal_fusion * res = nullptr;
    ಪೂರ್ಣಾಙ್ಕ best = 1;

    ಪ್ರತಿ (ಧ್ರುವ ggml_metal_fusion & fusion : ggml_metal_fusions) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_ops = (ಪೂರ್ಣಾಙ್ಕ) fusion.ops.size();

        // only look for a longer match than the current best
        ಯದಿ (n_ops <= best) {
            ಅನುವರ್ತ;
        }
        ಯದಿ (idx + n_ops > n_idxs) {
            ಅನುವರ್ತ;
        }

        ಧ್ರುವ ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        ತಾರ್ಕಿಕ ok = ಸತ್ಯ;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[ಗಂಟು_idxs[idx + j]];
            ಯದಿ (nodes[j]->op != fusion.ops[j]) {
                ok = ಅಸತ್ಯ;
                ಭಙ್ಗ;
            }
        }
        ಯದಿ (!ok) {
            ಅನುವರ್ತ;
        }

        ಯದಿ (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 1; j < n_ops && ok; j++) {
                ಯದಿ (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = ಅಸತ್ಯ;
                    ಭಙ್ಗ;
                }
                ಯದಿ (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = ಅಸತ್ಯ;
                    ಭಙ್ಗ;
                }
            }
            ಯದಿ (!ok) {
                ಅನುವರ್ತ;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            ಪೂರ್ಣಾಙ್ಕ outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = ಗಂಟು_idxs[idx + n_ops - 1];
            ಪ್ರತಿ (size_t i = 0; i < fusion.outs.size(); ++i) {
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = ಗಂಟು_idxs[idx + out_offset];
            }

            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_outputs = 1 + (ಪೂರ್ಣಾಙ್ಕ) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            ಯದಿ (!ggml_can_fuse_subgraph_ext(gf, ಗಂಟು_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                ಅನುವರ್ತ;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        ಯದಿ (fusion.check && !fusion.check(&fusion, nodes, gf, ಗಂಟು_idxs, idx, mode)) {
            ಅನುವರ್ತ;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        ಯದಿ (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            ಅನುವರ್ತ;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    ಪ್ರತ್ಯಾಗಮ res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
ಪೂರ್ಣಾಙ್ಕ ggml_metal_fusion_max(ಧ್ರುವ ggml_cgraph * gf, ಪೂರ್ಣಾಙ್ಕ idx) {
    // a view node cannot start a pattern - pack it alone
    ಯದಿ (ggml_op_is_empty(gf->nodes[idx]->op)) {
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    ಪೂರ್ಣಾಙ್ಕ idxs[GGML_METAL_FUSION_MAX];
    ಪೂರ್ಣಾಙ್ಕ n_idxs = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        ಯದಿ (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    ಯದಿ (n_idxs == 0) {
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    ಪೂರ್ಣಾಙ್ಕ total = 0;
    ಪೂರ್ಣಾಙ್ಕ i_f = 0;

    ಯಾವತ್ (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        ಪೂರ್ಣಾಙ್ಕ len = 1;
        ಧ್ರುವ ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        ಯದಿ (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            ಭಙ್ಗ;
        }

        total += len;
        i_f += len;
    }

    ಯದಿ (i_f == 0) {
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    ಪ್ರತ್ಯಾಗಮ std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
