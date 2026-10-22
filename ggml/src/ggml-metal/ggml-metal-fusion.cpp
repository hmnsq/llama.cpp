#অন্তর্ভাব "ggml-metal-fusion.h"

#অন্তর্ভাব "ggml-metal-common.h"
#অন্তর্ভাব "ggml-metal-device.h"

#অন্তর্ভাব "ggml-backend-impl.h"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <set>
#অন্তর্ভাব <string>
#অন্তর্ভাব <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
স্থির std::vector<ggml_op> ggml_metal_fusion_filter_ops(ধ্রুবক std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    জন্য (ggml_op op : ops_all) {
        যদি (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    ফেরত ops;
}

কাঠামো ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<পূর্ণসংখ্যা>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    যৌক্তিক unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    যৌক্তিক (*check)(ধ্রুবক কাঠামো ggml_metal_fusion   * fusion,
                  ধ্রুবক কাঠামো ggml_tensor * ধ্রুবক * nodes,
                  ধ্রুবক কাঠামো ggml_cgraph         * gf,
                  ধ্রুবক পূর্ণসংখ্যা                        * গ্রন্থি_idxs,
                        পূর্ণসংখ্যা                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            ধ্রুবক std::vector<ggml_op> & ops_all,
            ধ্রুবক std::vector<পূর্ণসংখ্যা> & outs,
            যৌক্তিক unsafe,
            যৌক্তিক (*check)(ধ্রুবক কাঠামো ggml_metal_fusion   * fusion,
                          ধ্রুবক কাঠামো ggml_tensor * ধ্রুবক * nodes,
                          ধ্রুবক কাঠামো ggml_cgraph         * gf,
                          ধ্রুবক পূর্ণসংখ্যা                        * গ্রন্থি_idxs,
                                পূর্ণসংখ্যা                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(ধ্রুবক ggml_metal_fusion * fusion) {
    ফেরত fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
স্থির যৌক্তিক ggml_metal_fusion_same_বাফার(ধ্রুবক ggml_tensor * a, ধ্রুবক ggml_tensor * b) {
    যদি (!a || !b) {
        ফেরত মিথ্যা;
    }

    ggml_backend_বাফার_t ba = a->view_src ? a->view_src->বাফার : a->বাফার;
    ggml_backend_বাফার_t bb = b->view_src ? b->view_src->বাফার : b->বাফার;

    ggml_metal_বাফার_t ca = (ggml_metal_বাফার_t) ba->context;
    ggml_metal_বাফার_t cb = (ggml_metal_বাফার_t) bb->context;

    ফেরত ggml_metal_বাফার_get_id(ca, a).metal == ggml_metal_বাফার_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
স্থির যৌক্তিক ggml_metal_fusion_overlap(ধ্রুবক ggml_tensor * a, ধ্রুবক ggml_tensor * b) {
    ggml_backend_বাফার_t ba = a->view_src ? a->view_src->বাফার : a->বাফার;
    ggml_backend_বাফার_t bb = b->view_src ? b->view_src->বাফার : b->বাফার;

    ধ্রুবক ggml_metal_বাফার_id bid_a = ggml_metal_বাফার_get_id((ggml_metal_বাফার_t) ba->context, a);
    ধ্রুবক ggml_metal_বাফার_id bid_b = ggml_metal_বাফার_get_id((ggml_metal_বাফার_t) bb->context, b);

    যদি (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        ফেরত মিথ্যা;
    }

    ফেরত bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
স্থির যৌক্তিক ggml_metal_fusion_check_norm(
        ধ্রুবক ggml_metal_fusion       * fusion,
        ধ্রুবক ggml_tensor * ধ্রুবক     * nodes,
        ধ্রুবক ggml_cgraph             * gf,
        ধ্রুবক পূর্ণসংখ্যা                     * গ্রন্থি_idxs,
              পূর্ণসংখ্যা                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(গ্রন্থি_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    যদি (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        ধ্রুবক ggml_tensor * scale = nodes[1];
        যদি (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            ফেরত মিথ্যা;
        }

        ফেরত সত্য;
    }

    জন্য (পূর্ণসংখ্যা j = 1; j < (পূর্ণসংখ্যা) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        যদি (nodes[j]->src[0] != nodes[j - 1]) {
            ফেরত মিথ্যা;
        }

        // the weight/bias must have the same row width as the norm input
        যদি (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            ফেরত মিথ্যা;
        }

        যদি (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            ফেরত মিথ্যা;
        }

        যদি (nodes[j]->type != GGML_TYPE_F32) {
            ফেরত মিথ্যা;
        }
    }

    ফেরত সত্য;
}

// SSM_CONV + UNARY (silu)
স্থির যৌক্তিক ggml_metal_fusion_check_ssm_conv_silu(
        ধ্রুবক ggml_metal_fusion       * fusion,
        ধ্রুবক ggml_tensor * ধ্রুবক     * nodes,
        ধ্রুবক ggml_cgraph             * gf,
        ধ্রুবক পূর্ণসংখ্যা                     * গ্রন্থি_idxs,
              পূর্ণসংখ্যা                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(গ্রন্থি_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    ধ্রুবক ggml_tensor * conv = nodes[0];
    ধ্রুবক ggml_tensor * un   = nodes[1];

    যদি (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        ফেরত মিথ্যা;
    }

    যদি (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        ফেরত মিথ্যা;
    }

    যদি (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        ফেরত মিথ্যা;
    }

    ফেরত সত্য;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
স্থির যৌক্তিক ggml_metal_fusion_check_add_chain(
        ধ্রুবক ggml_metal_fusion       * fusion,
        ধ্রুবক ggml_tensor * ধ্রুবক     * nodes,
        ধ্রুবক ggml_cgraph             * gf,
        ধ্রুবক পূর্ণসংখ্যা                     * গ্রন্থি_idxs,
              পূর্ণসংখ্যা                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(গ্রন্থি_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    জন্য (পূর্ণসংখ্যা j = 1; j < (পূর্ণসংখ্যা) fusion->ops.size(); j++) {
        যদি (nodes[j]->src[0] != nodes[j - 1]) {
            ফেরত মিথ্যা;
        }

        যদি (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            ফেরত মিথ্যা;
        }

        যদি (mode == GGML_METAL_FUSION_FULL) {
            যদি (!ggml_metal_fusion_same_বাফার(nodes[j]->src[1], nodes[0]->src[1])) {
                ফেরত মিথ্যা;
            }
        }
    }

    ফেরত সত্য;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
স্থির যৌক্তিক ggml_metal_fusion_check_gdn_cache(
        ধ্রুবক ggml_metal_fusion       * fusion,
        ধ্রুবক ggml_tensor * ধ্রুবক     * nodes,
        ধ্রুবক ggml_cgraph             * gf,
        ধ্রুবক পূর্ণসংখ্যা                     * গ্রন্থি_idxs,
              পূর্ণসংখ্যা                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(গ্রন্থি_idxs);
    GGML_UNUSED(idx);

    ধ্রুবক ggml_tensor * gdn = nodes[0];
    ধ্রুবক ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    যদি (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ফেরত মিথ্যা;
    }

    যদি (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ফেরত মিথ্যা;
    }

    ধ্রুবক int64_t S_v      = gdn->src[2]->ne[0];
    ধ্রুবক int64_t H        = gdn->src[2]->ne[1];
    ধ্রুবক int64_t n_tokens = gdn->src[2]->ne[2];
    ধ্রুবক int64_t n_seqs   = gdn->src[2]->ne[3];
    ধ্রুবক int64_t K        = ggml_get_op_params_i32(gdn, 0);
    ধ্রুবক size_t  লেজ_off = ggml_row_আকার(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    ধ্রুবক int64_t D         = S_v * S_v * H;
    ধ্রুবক int64_t n_written = std::min<int64_t>(n_tokens, K);

    ধ্রুবক ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    ধ্রুবক ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    যদি (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != লেজ_off || !ggml_is_contiguous(src)) {
        ফেরত মিথ্যা;
    }

    ধ্রুবক int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    যদি (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_আকার(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_আকার(GGML_TYPE_F32, D)) {
        ফেরত মিথ্যা;
    }

    যদি (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        যদি (dst->তথ্য == nullptr) {
            ফেরত মিথ্যা;
        }
    }

    ফেরত সত্য;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
স্থির যৌক্তিক ggml_metal_fusion_check_snake(
        ধ্রুবক ggml_metal_fusion       * fusion,
        ধ্রুবক ggml_tensor * ধ্রুবক     * nodes,
        ধ্রুবক ggml_cgraph             * gf,
        ধ্রুবক পূর্ণসংখ্যা                     * গ্রন্থি_idxs,
              পূর্ণসংখ্যা                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(গ্রন্থি_idxs);
    GGML_UNUSED(idx);

    ধ্রুবক ggml_tensor * mul0     = nodes[0];
    ধ্রুবক ggml_tensor * sin_গ্রন্থি = nodes[1];
    ধ্রুবক ggml_tensor * sqr      = nodes[2];
    ধ্রুবক ggml_tensor * mul1     = nodes[3];
    ধ্রুবক ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    ধ্রুবক ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ধ্রুবক ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ধ্রুবক ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    ধ্রুবক ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    ধ্রুবক যৌক্তিক types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_গ্রন্থি->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    ধ্রুবক যৌক্তিক shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    ধ্রুবক যৌক্তিক dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    ধ্রুবক যৌক্তিক contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    ফেরত types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
স্থির ধ্রুবক std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
স্থির ধ্রুবক std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
স্থির ধ্রুবক std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_যোগ_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
স্থির ধ্রুবক std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_যোগ_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

স্থির যৌক্তিক ggml_metal_fusion_check_topk_moe(
        ধ্রুবক ggml_metal_fusion       * fusion,
        ধ্রুবক ggml_tensor * ধ্রুবক     * nodes,
        ধ্রুবক ggml_cgraph             * gf,
        ধ্রুবক পূর্ণসংখ্যা                     * গ্রন্থি_idxs,
              পূর্ণসংখ্যা                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    ধ্রুবক পূর্ণসংখ্যা n_ops = (পূর্ণসংখ্যা) fusion->ops.size();

    ধ্রুবক যৌক্তিক with_norm  = n_ops >= 6;
    ধ্রুবক যৌক্তিক with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    ধ্রুবক std::vector<ggml_op> & ops_all = fusion->ops_all;

    ধ্রুবক পূর্ণসংখ্যা raw_start = গ্রন্থি_idxs[idx];
    পূর্ণসংখ্যা       raw_end   = গ্রন্থি_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    যদি (with_norm && !with_scale) {
        যদি (raw_end + 1 >= gf->n_nodes) {
            ফেরত মিথ্যা;
        }
        ধ্রুবক ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        যদি (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            ফেরত মিথ্যা;
        }
        raw_end++;
    }

    ধ্রুবক পূর্ণসংখ্যা raw_গুনতি = raw_end - raw_start + 1;
    যদি (raw_গুনতি != (পূর্ণসংখ্যা) ops_all.size()) {
        ফেরত মিথ্যা;
    }

    পূর্ণসংখ্যা raw_idxs[GGML_METAL_FUSION_MAX];
    জন্য (পূর্ণসংখ্যা i = 0; i < raw_গুনতি; ++i) {
        raw_idxs[i] = raw_start + i;
        যদি (gf->nodes[raw_start + i]->op != ops_all[i]) {
            ফেরত মিথ্যা;
        }
    }

    ধ্রুবক ggml_tensor * softmax        = gf->nodes[raw_start];
    ধ্রুবক ggml_tensor * probs_reshaped = gf->nodes[raw_start + 1];
    ধ্রুবক ggml_tensor * argsort        = gf->nodes[raw_start + 2];
    ধ্রুবক ggml_tensor * ids            = gf->nodes[raw_start + 3];
    ধ্রুবক ggml_tensor * get_rows       = gf->nodes[raw_start + 4];
    ধ্রুবক ggml_tensor * out            = gf->nodes[raw_end];
    ধ্রুবক ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    দশমিক scale   = 1.0f;
    দশমিক max_bias = 0.0f;
    memcpy(&scale,    ((ধ্রুবক int32_t *) softmax->op_params) + 0, আকার(scale));
    memcpy(&max_bias, ((ধ্রুবক int32_t *) softmax->op_params) + 1, আকার(max_bias));
    যদি (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        ফেরত মিথ্যা;
    }

    যদি (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        ফেরত মিথ্যা;
    }

    ধ্রুবক int64_t n_expert      = logits->ne[0];
    ধ্রুবক int64_t n_tokens      = logits->ne[1];
    ধ্রুবক int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    যদি (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        ফেরত মিথ্যা;
    }

    যদি (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        ফেরত মিথ্যা;
    }

    যদি (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_আকার(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_আকার(GGML_TYPE_I32) * n_expert) {
        ফেরত মিথ্যা;
    }

    যদি (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        ফেরত মিথ্যা;
    }

    যদি (with_norm) {
        ধ্রুবক ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
        ধ্রুবক ggml_tensor * যোগ_rows_2         = gf->nodes[raw_start + 6];
        ধ্রুবক ggml_tensor * clamp            = gf->nodes[raw_start + 7];
        ধ্রুবক ggml_tensor * div              = gf->nodes[raw_start + 8];
        ধ্রুবক ggml_tensor * out_reshaped     = gf->nodes[raw_start + 9];

        যদি (weights_reshaped->src[0] != get_rows || যোগ_rows_2->src[0] != weights_reshaped ||
            clamp->src[0] != যোগ_rows_2 || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            out_reshaped->src[0] != div) {
            ফেরত মিথ্যা;
        }

        যদি (with_scale) {
            ধ্রুবক ggml_tensor * scale_গ্রন্থি = gf->nodes[raw_start + 10];
            যদি (scale_গ্রন্থি->src[0] != out_reshaped) {
                ফেরত মিথ্যা;
            }
        }
    } অন্যথা যদি (with_scale) {
        ধ্রুবক ggml_tensor * scale_গ্রন্থি = gf->nodes[raw_start + 5];
        যদি (scale_গ্রন্থি->src[0] != get_rows) {
            ফেরত মিথ্যা;
        }
    }

    ধ্রুবক পূর্ণসংখ্যা outputs[2] = { raw_start + 3, raw_end };
    যদি (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_গুনতি, ops_all.তথ্য(), outputs, 2)) {
        ফেরত মিথ্যা;
    }

    যদি (mode == GGML_METAL_FUSION_FULL) {
        যদি (!logits->তথ্য || !out->তথ্য || !ids->তথ্য) {
            ফেরত মিথ্যা;
        }
    }

    ফেরত সত্য;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

কাঠামো ggml_metal_moe_reduce_match {
    ধ্রুবক ggml_tensor * experts;
    ধ্রুবক ggml_tensor * weights;
    ধ্রুবক ggml_tensor * dst;
    পূর্ণসংখ্যা গ্রন্থি_গুনতি;
};

স্থির যৌক্তিক ggml_metal_fusion_match_moe_reduce(
        ধ্রুবক ggml_cgraph * gf, পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    যদি (match == nullptr || গ্রন্থি_idx < 0 || গ্রন্থি_idx + (পূর্ণসংখ্যা) ops_all.size() > gf->n_nodes) {
        ফেরত মিথ্যা;
    }

    ধ্রুবক ggml_tensor * mul = gf->nodes[গ্রন্থি_idx];
    যদি (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        ফেরত মিথ্যা;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    ধ্রুবক পূর্ণসংখ্যা raw_গুনতি     = (পূর্ণসংখ্যা) ops_all.size();
    ধ্রুবক পূর্ণসংখ্যা n_expert_used = raw_গুনতি / 2;

    যদি (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_গুনতি != 2 * n_expert_used) {
        ফেরত মিথ্যা;
    }

    পূর্ণসংখ্যা n_views = 0;
    যখন (গ্রন্থি_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[গ্রন্থি_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    যদি (n_views != n_expert_used) {
        ফেরত মিথ্যা;
    }

    জন্য (পূর্ণসংখ্যা i = n_expert_used + 1; i < raw_গুনতি; ++i) {
        যদি (gf->nodes[গ্রন্থি_idx + i]->op != GGML_OP_ADD) {
            ফেরত মিথ্যা;
        }
    }

    পূর্ণসংখ্যা raw_idxs[GGML_METAL_FUSION_MAX];
    জন্য (পূর্ণসংখ্যা i = 0; i < raw_গুনতি; ++i) {
        raw_idxs[i] = গ্রন্থি_idx + i;
        যদি (gf->nodes[গ্রন্থি_idx + i]->op != ops_all[i]) {
            ফেরত মিথ্যা;
        }
    }

    ধ্রুবক ggml_tensor * experts = mul->src[0];
    ধ্রুবক ggml_tensor * weights = mul->src[1];
    ধ্রুবক ggml_tensor * dst     = gf->nodes[গ্রন্থি_idx + raw_গুনতি - 1];

    যদি (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ফেরত মিথ্যা;
    }

    ধ্রুবক int64_t n_embd   = experts->ne[0];
    ধ্রুবক int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    যদি (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        ফেরত মিথ্যা;
    }

    যদি (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        ফেরত মিথ্যা;
    }

    জন্য (পূর্ণসংখ্যা i = 1; i <= n_expert_used; ++i) {
        ধ্রুবক ggml_tensor * view = gf->nodes[গ্রন্থি_idx + i];
        যদি (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            ফেরত মিথ্যা;
        }
    }

    ধ্রুবক ggml_tensor * পূর্ব_add = nullptr;
    জন্য (পূর্ণসংখ্যা j = 1; j < n_expert_used; ++j) {
        ধ্রুবক ggml_tensor * add = gf->nodes[গ্রন্থি_idx + n_expert_used + j];
        ধ্রুবক ggml_tensor * rhs = gf->nodes[গ্রন্থি_idx + j + 1];
        ধ্রুবক ggml_tensor * lhs = j == 1 ? gf->nodes[গ্রন্থি_idx + 1] : পূর্ব_add;
        যদি (add->src[0] != lhs || add->src[1] != rhs) {
            ফেরত মিথ্যা;
        }
        পূর্ব_add = add;
    }

    ধ্রুবক পূর্ণসংখ্যা outputs[1] = { গ্রন্থি_idx + raw_গুনতি - 1 };
    যদি (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_গুনতি, ops_all.তথ্য(), outputs, 1)) {
        ফেরত মিথ্যা;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->গ্রন্থি_গুনতি = raw_গুনতি;
    ফেরত সত্য;
}

স্থির যৌক্তিক ggml_metal_fusion_check_moe_reduce(
        ধ্রুবক ggml_metal_fusion       * fusion,
        ধ্রুবক ggml_tensor * ধ্রুবক     * nodes,
        ধ্রুবক ggml_cgraph             * gf,
        ধ্রুবক পূর্ণসংখ্যা                     * গ্রন্থি_idxs,
              পূর্ণসংখ্যা                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    যদি (!ggml_metal_fusion_match_moe_reduce(gf, গ্রন্থি_idxs[idx], fusion->ops_all, &match)) {
        ফেরত মিথ্যা;
    }

    যদি ((পূর্ণসংখ্যা) fusion->ops.size() != match.experts->ne[1]) {
        ফেরত মিথ্যা;
    }

    ধ্রুবক পূর্ণসংখ্যা raw_end = গ্রন্থি_idxs[idx] + match.গ্রন্থি_গুনতি - 1;
    যদি (গ্রন্থি_idxs[idx + (পূর্ণসংখ্যা) fusion->ops.size() - 1] != raw_end) {
        ফেরত মিথ্যা;
    }

    যদি (mode == GGML_METAL_FUSION_FULL) {
        যদি (!match.experts->তথ্য || !match.weights->তথ্য || !match.dst->তথ্য) {
            ফেরত মিথ্যা;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        যদি (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            ফেরত মিথ্যা;
        }
    }

    ফেরত সত্য;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
স্থির যৌক্তিক ggml_metal_tensor_is_weight(ধ্রুবক কাঠামো ggml_tensor * t) {
    ধ্রুবক ggml_tensor * base = t->view_src != ফাঁকা ? t->view_src : t;

    ফেরত base->বাফার != ফাঁকা && ggml_backend_বাফার_get_usage(base->বাফার) == GGML_BACKEND_বাফার_USAGE_WEIGHTS;
}

স্থির ধ্রুবক কাঠামো ggml_tensor * ggml_metal_mul_mat_add_operand(ধ্রুবক কাঠামো ggml_tensor * mm, ধ্রুবক কাঠামো ggml_tensor * add) {
    যদি (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        ফেরত ফাঁকা;
    }

    ধ্রুবক ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    ধ্রুবক যৌক্তিক ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    ফেরত ok ? other : ফাঁকা;
}

স্থির ধ্রুবক কাঠামো ggml_tensor * ggml_metal_mul_mat_add_residual(ধ্রুবক কাঠামো ggml_tensor * mm, ধ্রুবক কাঠামো ggml_tensor * add) {
    ধ্রুবক ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    ধ্রুবক যৌক্তিক ok = res != ফাঁকা && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    ফেরত ok ? res : ফাঁকা;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
স্থির যৌক্তিক ggml_metal_fusion_check_mul_mat_add(
        ধ্রুবক ggml_metal_fusion       * fusion,
        ধ্রুবক ggml_tensor * ধ্রুবক     * nodes,
        ধ্রুবক ggml_cgraph             * gf,
        ধ্রুবক পূর্ণসংখ্যা                     * গ্রন্থি_idxs,
              পূর্ণসংখ্যা                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(গ্রন্থি_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    ধ্রুবক ggml_tensor * mm  = nodes[0];
    ধ্রুবক ggml_tensor * add = nodes[1];

    যদি (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        ফেরত মিথ্যা;
    }

    যদি (mode == GGML_METAL_FUSION_STRUCTURAL) {
        ফেরত সত্য;
    }

    ধ্রুবক ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    যদি (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        ফেরত মিথ্যা;
    }

    ফেরত !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->তথ্য == res->তথ্য || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

স্থির ধ্রুবক std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
স্থির ধ্রুবক std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
স্থির ধ্রুবক std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
স্থির ধ্রুবক std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
স্থির ধ্রুবক std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
স্থির ধ্রুবক std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

স্থির ধ্রুবক std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
স্থির ধ্রুবক std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
স্থির ধ্রুবক std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
স্থির ধ্রুবক std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
স্থির ধ্রুবক std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
স্থির ধ্রুবক std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
স্থির ধ্রুবক std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

স্থির ধ্রুবক std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

স্থির ধ্রুবক std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

স্থির ধ্রুবক std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
স্থির ধ্রুবক std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
স্থির ধ্রুবক std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
স্থির ধ্রুবক std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
স্থির ধ্রুবক std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
স্থির ধ্রুবক std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
স্থির ধ্রুবক std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

স্থির ধ্রুবক std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };

স্থির ধ্রুবক std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     মিথ্যা, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_norm_mul_add,           {},     মিথ্যা, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     মিথ্যা, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     মিথ্যা, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_rms_norm_mul_add,       {},     মিথ্যা, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     মিথ্যা, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_2,                  {},     মিথ্যা, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_3,                  {},     মিথ্যা, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_4,                  {},     মিথ্যা, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_5,                  {},     মিথ্যা, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_6,                  {},     মিথ্যা, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_7,                  {},     মিথ্যা, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     মিথ্যা, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     সত্য,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    সত্য,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    সত্য,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    সত্য,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    সত্য,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     সত্য,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     সত্য,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     সত্য,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     সত্য,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     সত্য,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     সত্য,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     সত্য,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     মিথ্যা, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_ADD,    ops_mul_mat_add,            {},     মিথ্যা, ggml_metal_fusion_check_mul_mat_add },
};

// ---- alloc deps -----------------------------------------------------------

স্থির যৌক্তিক ggml_metal_fusion_match_raw_pattern(
        ধ্রুবক ggml_cgraph * gf, পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::vector<ggml_op> & ops) {
    যদি (গ্রন্থি_idx < 0 || গ্রন্থি_idx + (পূর্ণসংখ্যা) ops.size() > gf->n_nodes) {
        ফেরত মিথ্যা;
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < (পূর্ণসংখ্যা) ops.size(); ++i) {
        যদি (gf->nodes[গ্রন্থি_idx + i]->op != ops[i]) {
            ফেরত মিথ্যা;
        }
    }

    ফেরত সত্য;
}

স্থির শূন্য ggml_metal_fusion_add_pattern_alloc_deps(
        শূন্য * user_তথ্য,
        শূন্য (*add_alloc_dep)(শূন্য *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        ধ্রুবক ggml_metal_fusion * fusion,
        পূর্ণসংখ্যা গ্রন্থি_idx) {
    ধ্রুবক পূর্ণসংখ্যা last_গ্রন্থি = গ্রন্থি_idx + (পূর্ণসংখ্যা) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    জন্য (পূর্ণসংখ্যা j = 0; j < (পূর্ণসংখ্যা) fusion->ops_all.size(); ++j) {
        ggml_tensor * গ্রন্থি = gf->nodes[গ্রন্থি_idx + j];
        জন্য (পূর্ণসংখ্যা s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = গ্রন্থি->src[s];
            যদি (src && seen.insert(src).second) {
                add_alloc_dep(user_তথ্য, src, gf->nodes[last_গ্রন্থি]);
            }
        }
        seen.insert(গ্রন্থি);
    }
}

শূন্য ggml_metal_fusion_add_alloc_deps(
        শূন্য * user_তথ্য,
        শূন্য (*add_alloc_dep)(শূন্য *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    জন্য (পূর্ণসংখ্যা i = 0; i < gf->n_nodes; ++i) {
        ধ্রুবক ggml_metal_fusion * best = nullptr;
        পূর্ণসংখ্যা best_raw = 0;

        জন্য (ধ্রুবক ggml_metal_fusion & fusion : ggml_metal_fusions) {
            যদি ((পূর্ণসংখ্যা) fusion.ops_all.size() <= best_raw) {
                চালু;
            }
            যদি (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (পূর্ণসংখ্যা) fusion.ops_all.size();
            }
        }

        যদি (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_তথ্য, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

স্থির std::string ggml_metal_fusion_label(ধ্রুবক ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    জন্য (পূর্ণসংখ্যা j = 0; j < (পূর্ণসংখ্যা) fusion->ops.size(); j++) {
        যদি (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    ফেরত label;
}

কাঠামো ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    যৌক্তিক enabled;
    যৌক্তিক stats;
    যৌক্তিক labels_set;
    পূর্ণসংখ্যা  debug;
};

কাঠামো ggml_metal_fusion_info * ggml_metal_fusion_info_init(যৌক্তিক enabled, পূর্ণসংখ্যা debug) {
    ggml_metal_fusion_info * finfo = নতুন ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = মিথ্যা;
    finfo->debug      = debug;

    যদি (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    ফেরত finfo;
}

শূন্য ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

যৌক্তিক ggml_metal_fusion_info_enabled(ধ্রুবক ggml_metal_fusion_info * finfo) {
    ফেরত finfo->enabled;
}

যৌক্তিক ggml_metal_fusion_info_stats(ধ্রুবক ggml_metal_fusion_info * finfo) {
    ফেরত finfo->stats;
}

পূর্ণসংখ্যা ggml_metal_fusion_info_debug(ধ্রুবক ggml_metal_fusion_info * finfo) {
    ফেরত finfo->debug;
}

পূর্ণসংখ্যা ggml_metal_fusion_info_n_fusions(ধ্রুবক ggml_metal_fusion_info * finfo) {
    ফেরত (পূর্ণসংখ্যা) finfo->labels.size();
}

ধ্রুবক অক্ষর * ggml_metal_fusion_info_label(ধ্রুবক ggml_metal_fusion_info * finfo, পূর্ণসংখ্যা idx) {
    GGML_ASSERT(idx >= 0 && idx < (পূর্ণসংখ্যা) finfo->labels.size());
    ফেরত finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_গুনতি(ধ্রুবক ggml_metal_fusion_info * finfo, পূর্ণসংখ্যা idx) {
    GGML_ASSERT(idx >= 0 && idx < (পূর্ণসংখ্যা) finfo->counts.size());
    ফেরত finfo->counts[idx];
}

শূন্য ggml_metal_fusion_info_গুনতি_fusion(ggml_metal_fusion_info * finfo, ধ্রুবক ggml_metal_fusion * fusion) {
    যদি (!finfo->stats || fusion == nullptr) {
        ফেরত;
    }

    ধ্রুবক ptrdiff_t idx = fusion - ggml_metal_fusions.তথ্য();
    যদি (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

শূন্য ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, যৌক্তিক enabled) {
    finfo->enabled = enabled;
}

শূন্য ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    যদি (finfo->labels_set) {
        ফেরত;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    জন্য (ধ্রুবক ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = সত্য;
}

শূন্য ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = সত্য;
    ggml_metal_fusion_info_labels_init(finfo);
}

শূন্য ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

পূর্ণসংখ্যা ggml_metal_fusion_info_stats_get(ধ্রুবক ggml_metal_fusion_info * finfo, ধ্রুবক অক্ষর ** labels, uint64_t * counts, পূর্ণসংখ্যা n) {
    ধ্রুবক পূর্ণসংখ্যা n_fusions = (পূর্ণসংখ্যা) finfo->labels.size();

    যদি (labels == nullptr) {
        ফেরত n_fusions;
    }

    ধ্রুবক পূর্ণসংখ্যা n_fill = std::min(n, n_fusions);
    জন্য (পূর্ণসংখ্যা i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        যদি (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    ফেরত n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
স্থির যৌক্তিক ggml_metal_fusion_check_memory_ranges(
        ধ্রুবক ggml_metal_fusion      * fusion,
        ধ্রুবক ggml_tensor * ধ্রুবক    * nodes,
        পূর্ণসংখ্যা                            গ্রন্থি_গুনতি) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](ধ্রুবক ggml_tensor * a, ধ্রুবক ggml_tensor * b) {
        যদি (a->তথ্য && b->তথ্য && a->তথ্য == b->তথ্য) {
            ফেরত সত্য;
        }
        জন্য (ধ্রুবক ggml_tensor * v = a; v; v = v->view_src) {
            যদি (v == b) {
                ফেরত সত্য;
            }
        }
        জন্য (ধ্রুবক ggml_tensor * v = b; v; v = v->view_src) {
            যদি (v == a) {
                ফেরত সত্য;
            }
        }
        ফেরত মিথ্যা;
    };

    auto nodes_overlap = [](ধ্রুবক ggml_tensor * a, ধ্রুবক ggml_tensor * b) {
        যদি (!a || !b || !a->তথ্য || !b->তথ্য || !a->বাফার || !b->বাফার) {
            ফেরত মিথ্যা;
        }

        যদি (a->বাফার != b->বাফার) {
            ফেরত মিথ্যা;
        }

        ধ্রুবক int64_t a_start = (int64_t) a->তথ্য;
        ধ্রুবক int64_t a_end   = a_start + ggml_backend_buft_get_alloc_আকার(a->বাফার->buft, a);
        ধ্রুবক int64_t b_start = (int64_t) b->তথ্য;
        ধ্রুবক int64_t b_end   = b_start + ggml_backend_buft_get_alloc_আকার(b->বাফার->buft, b);

        ফেরত (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](ধ্রুবক ggml_tensor * src, ধ্রুবক ggml_tensor * ধ্রুবক * nodes, পূর্ণসংখ্যা j) {
        জন্য (পূর্ণসংখ্যা k = 0; k < j; ++k) {
            যদি (src == nodes[k]) {
                ফেরত সত্য;
            }
            জন্য (ধ্রুবক ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                যদি (view_src == nodes[k]) {
                    ফেরত সত্য;
                }
            }
        }
        ফেরত মিথ্যা;
    };

    auto check_dst = [&](ধ্রুবক ggml_tensor * dst) {
        জন্য (পূর্ণসংখ্যা j = 0; j < গ্রন্থি_গুনতি; ++j) {
            জন্য (পূর্ণসংখ্যা s = 0; s < GGML_MAX_SRC; ++s) {
                ধ্রুবক ggml_tensor * src = nodes[j]->src[s];
                যদি (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    চালু;
                }

                যদি (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    ফেরত মিথ্যা;
                }
            }
        }
        ফেরত সত্য;
    };

    যদি (!check_dst(nodes[গ্রন্থি_গুনতি - 1])) {
        ফেরত মিথ্যা;
    }

    জন্য (পূর্ণসংখ্যা offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < গ্রন্থি_গুনতি);
        যদি (!check_dst(nodes[offset])) {
            ফেরত মিথ্যা;
        }
    }

    ফেরত সত্য;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
ধ্রুবক ggml_metal_fusion * ggml_metal_fusion_পরবর্তী(
        ধ্রুবক ggml_cgraph * gf,
        ধ্রুবক পূর্ণসংখ্যা * গ্রন্থি_idxs,
        পূর্ণসংখ্যা n_idxs,
        পূর্ণসংখ্যা idx,
        ggml_metal_fusion_mode mode,
        পূর্ণসংখ্যা * n_out) {
    ধ্রুবক ggml_metal_fusion * res = nullptr;
    পূর্ণসংখ্যা best = 1;

    জন্য (ধ্রুবক ggml_metal_fusion & fusion : ggml_metal_fusions) {
        ধ্রুবক পূর্ণসংখ্যা n_ops = (পূর্ণসংখ্যা) fusion.ops.size();

        // only look for a longer match than the current best
        যদি (n_ops <= best) {
            চালু;
        }
        যদি (idx + n_ops > n_idxs) {
            চালু;
        }

        ধ্রুবক ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        যৌক্তিক ok = সত্য;
        জন্য (পূর্ণসংখ্যা j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[গ্রন্থি_idxs[idx + j]];
            যদি (nodes[j]->op != fusion.ops[j]) {
                ok = মিথ্যা;
                বিরতি;
            }
        }
        যদি (!ok) {
            চালু;
        }

        যদি (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            জন্য (পূর্ণসংখ্যা j = 1; j < n_ops && ok; j++) {
                যদি (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = মিথ্যা;
                    বিরতি;
                }
                যদি (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = মিথ্যা;
                    বিরতি;
                }
            }
            যদি (!ok) {
                চালু;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            পূর্ণসংখ্যা outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = গ্রন্থি_idxs[idx + n_ops - 1];
            জন্য (size_t i = 0; i < fusion.outs.size(); ++i) {
                ধ্রুবক পূর্ণসংখ্যা out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = গ্রন্থি_idxs[idx + out_offset];
            }

            ধ্রুবক পূর্ণসংখ্যা n_outputs = 1 + (পূর্ণসংখ্যা) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            যদি (!ggml_can_fuse_subgraph_ext(gf, গ্রন্থি_idxs + idx, n_ops, fusion.ops.তথ্য(), outputs_buf, n_outputs)) {
                চালু;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        যদি (fusion.check && !fusion.check(&fusion, nodes, gf, গ্রন্থি_idxs, idx, mode)) {
            চালু;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        যদি (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            চালু;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    ফেরত res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
পূর্ণসংখ্যা ggml_metal_fusion_max(ধ্রুবক ggml_cgraph * gf, পূর্ণসংখ্যা idx) {
    // a view node cannot start a pattern - pack it alone
    যদি (ggml_op_is_empty(gf->nodes[idx]->op)) {
        ফেরত 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    পূর্ণসংখ্যা idxs[GGML_METAL_FUSION_MAX];
    পূর্ণসংখ্যা n_idxs = 0;
    জন্য (পূর্ণসংখ্যা i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        যদি (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    যদি (n_idxs == 0) {
        ফেরত 1;
    }

    পূর্ণসংখ্যা যোগ = 0;
    পূর্ণসংখ্যা i_f = 0;

    যখন (i_f < n_idxs && যোগ < GGML_METAL_FUSION_MAX) {
        পূর্ণসংখ্যা len = 1;
        ধ্রুবক ggml_metal_fusion * fusion = ggml_metal_fusion_পরবর্তী(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        যদি (!fusion || যোগ + len > GGML_METAL_FUSION_MAX) {
            বিরতি;
        }

        যোগ += len;
        i_f += len;
    }

    যদি (i_f == 0) {
        ফেরত 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    ফেরত std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
