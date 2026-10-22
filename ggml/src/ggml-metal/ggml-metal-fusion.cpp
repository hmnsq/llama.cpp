#অন্তর্ভাৱ "ggml-metal-fusion.h"

#অন্তর্ভাৱ "ggml-metal-common.h"
#অন্তর্ভাৱ "ggml-metal-device.h"

#অন্তর্ভাৱ "ggml-backend-impl.h"

#অন্তর্ভাৱ <algorithm>
#অন্তর্ভাৱ <cstddef>
#অন্তর্ভাৱ <cstring>
#অন্তর্ভাৱ <set>
#অন্তর্ভাৱ <string>
#অন্তর্ভাৱ <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
স্থির std::vector<ggml_op> ggml_metal_fusion_filter_ops(ধ্রুৱ std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    বাবে (ggml_op op : ops_all) {
        যদি (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    ঘূৰাই ops;
}

সংরচনা ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<পূর্ণাঙ্ক>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    তার্কিক unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    তার্কিক (*check)(ধ্রুৱ সংরচনা ggml_metal_fusion   * fusion,
                  ধ্রুৱ সংরচনা ggml_tensor * ধ্রুৱ * nodes,
                  ধ্রুৱ সংরচনা ggml_cgraph         * gf,
                  ধ্রুৱ পূর্ণাঙ্ক                        * গ্ৰন্থি_idxs,
                        পূর্ণাঙ্ক                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            ধ্রুৱ std::vector<ggml_op> & ops_all,
            ধ্রুৱ std::vector<পূর্ণাঙ্ক> & outs,
            তার্কিক unsafe,
            তার্কিক (*check)(ধ্রুৱ সংরচনা ggml_metal_fusion   * fusion,
                          ধ্রুৱ সংরচনা ggml_tensor * ধ্রুৱ * nodes,
                          ধ্রুৱ সংরচনা ggml_cgraph         * gf,
                          ধ্রুৱ পূর্ণাঙ্ক                        * গ্ৰন্থি_idxs,
                                পূর্ণাঙ্ক                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(ধ্রুৱ ggml_metal_fusion * fusion) {
    ঘূৰাই fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
স্থির তার্কিক ggml_metal_fusion_same_buffer(ধ্রুৱ ggml_tensor * a, ধ্রুৱ ggml_tensor * b) {
    যদি (!a || !b) {
        ঘূৰাই মিছা;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    ঘূৰাই ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
স্থির তার্কিক ggml_metal_fusion_overlap(ধ্রুৱ ggml_tensor * a, ধ্রুৱ ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ধ্রুৱ ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    ধ্রুৱ ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    যদি (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        ঘূৰাই মিছা;
    }

    ঘূৰাই bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
স্থির তার্কিক ggml_metal_fusion_check_norm(
        ধ্রুৱ ggml_metal_fusion       * fusion,
        ধ্রুৱ ggml_tensor * ধ্রুৱ     * nodes,
        ধ্রুৱ ggml_cgraph             * gf,
        ধ্রুৱ পূর্ণাঙ্ক                     * গ্ৰন্থি_idxs,
              পূর্ণাঙ্ক                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(গ্ৰন্থি_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    যদি (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        ধ্রুৱ ggml_tensor * scale = nodes[1];
        যদি (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            ঘূৰাই মিছা;
        }

        ঘূৰাই সঁচা;
    }

    বাবে (পূর্ণাঙ্ক j = 1; j < (পূর্ণাঙ্ক) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        যদি (nodes[j]->src[0] != nodes[j - 1]) {
            ঘূৰাই মিছা;
        }

        // the weight/bias must have the same row width as the norm input
        যদি (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            ঘূৰাই মিছা;
        }

        যদি (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            ঘূৰাই মিছা;
        }

        যদি (nodes[j]->type != GGML_TYPE_F32) {
            ঘূৰাই মিছা;
        }
    }

    ঘূৰাই সঁচা;
}

// SSM_CONV + UNARY (silu)
স্থির তার্কিক ggml_metal_fusion_check_ssm_conv_silu(
        ধ্রুৱ ggml_metal_fusion       * fusion,
        ধ্রুৱ ggml_tensor * ধ্রুৱ     * nodes,
        ধ্রুৱ ggml_cgraph             * gf,
        ধ্রুৱ পূর্ণাঙ্ক                     * গ্ৰন্থি_idxs,
              পূর্ণাঙ্ক                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(গ্ৰন্থি_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    ধ্রুৱ ggml_tensor * conv = nodes[0];
    ধ্রুৱ ggml_tensor * un   = nodes[1];

    যদি (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        ঘূৰাই মিছা;
    }

    যদি (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        ঘূৰাই মিছা;
    }

    যদি (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        ঘূৰাই মিছা;
    }

    ঘূৰাই সঁচা;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
স্থির তার্কিক ggml_metal_fusion_check_add_chain(
        ধ্রুৱ ggml_metal_fusion       * fusion,
        ধ্রুৱ ggml_tensor * ধ্রুৱ     * nodes,
        ধ্রুৱ ggml_cgraph             * gf,
        ধ্রুৱ পূর্ণাঙ্ক                     * গ্ৰন্থি_idxs,
              পূর্ণাঙ্ক                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(গ্ৰন্থি_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    বাবে (পূর্ণাঙ্ক j = 1; j < (পূর্ণাঙ্ক) fusion->ops.size(); j++) {
        যদি (nodes[j]->src[0] != nodes[j - 1]) {
            ঘূৰাই মিছা;
        }

        যদি (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            ঘূৰাই মিছা;
        }

        যদি (mode == GGML_METAL_FUSION_FULL) {
            যদি (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                ঘূৰাই মিছা;
            }
        }
    }

    ঘূৰাই সঁচা;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
স্থির তার্কিক ggml_metal_fusion_check_gdn_cache(
        ধ্রুৱ ggml_metal_fusion       * fusion,
        ধ্রুৱ ggml_tensor * ধ্রুৱ     * nodes,
        ধ্রুৱ ggml_cgraph             * gf,
        ধ্রুৱ পূর্ণাঙ্ক                     * গ্ৰন্থি_idxs,
              পূর্ণাঙ্ক                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(গ্ৰন্থি_idxs);
    GGML_UNUSED(idx);

    ধ্রুৱ ggml_tensor * gdn = nodes[0];
    ধ্রুৱ ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    যদি (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ঘূৰাই মিছা;
    }

    যদি (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ঘূৰাই মিছা;
    }

    ধ্রুৱ int64_t S_v      = gdn->src[2]->ne[0];
    ধ্রুৱ int64_t H        = gdn->src[2]->ne[1];
    ধ্রুৱ int64_t n_tokens = gdn->src[2]->ne[2];
    ধ্রুৱ int64_t n_seqs   = gdn->src[2]->ne[3];
    ধ্রুৱ int64_t K        = ggml_get_op_params_i32(gdn, 0);
    ধ্রুৱ size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    ধ্রুৱ int64_t D         = S_v * S_v * H;
    ধ্রুৱ int64_t n_written = std::min<int64_t>(n_tokens, K);

    ধ্রুৱ ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    ধ্রুৱ ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    যদি (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        ঘূৰাই মিছা;
    }

    ধ্রুৱ int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    যদি (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        ঘূৰাই মিছা;
    }

    যদি (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        যদি (dst->data == nullptr) {
            ঘূৰাই মিছা;
        }
    }

    ঘূৰাই সঁচা;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
স্থির তার্কিক ggml_metal_fusion_check_snake(
        ধ্রুৱ ggml_metal_fusion       * fusion,
        ধ্রুৱ ggml_tensor * ধ্রুৱ     * nodes,
        ধ্রুৱ ggml_cgraph             * gf,
        ধ্রুৱ পূর্ণাঙ্ক                     * গ্ৰন্থি_idxs,
              পূর্ণাঙ্ক                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(গ্ৰন্থি_idxs);
    GGML_UNUSED(idx);

    ধ্রুৱ ggml_tensor * mul0     = nodes[0];
    ধ্রুৱ ggml_tensor * sin_গ্ৰন্থি = nodes[1];
    ধ্রুৱ ggml_tensor * sqr      = nodes[2];
    ধ্রুৱ ggml_tensor * mul1     = nodes[3];
    ধ্রুৱ ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    ধ্রুৱ ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ধ্রুৱ ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ধ্রুৱ ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    ধ্রুৱ ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    ধ্রুৱ তার্কিক types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_গ্ৰন্থি->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    ধ্রুৱ তার্কিক shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    ধ্রুৱ তার্কিক dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    ধ্রুৱ তার্কিক contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    ঘূৰাই types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
স্থির ধ্রুৱ std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
স্থির ধ্রুৱ std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
স্থির ধ্রুৱ std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
স্থির ধ্রুৱ std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

স্থির তার্কিক ggml_metal_fusion_check_topk_moe(
        ধ্রুৱ ggml_metal_fusion       * fusion,
        ধ্রুৱ ggml_tensor * ধ্রুৱ     * nodes,
        ধ্রুৱ ggml_cgraph             * gf,
        ধ্রুৱ পূর্ণাঙ্ক                     * গ্ৰন্থি_idxs,
              পূর্ণাঙ্ক                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    ধ্রুৱ পূর্ণাঙ্ক n_ops = (পূর্ণাঙ্ক) fusion->ops.size();

    ধ্রুৱ তার্কিক with_norm  = n_ops >= 6;
    ধ্রুৱ তার্কিক with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    ধ্রুৱ std::vector<ggml_op> & ops_all = fusion->ops_all;

    ধ্রুৱ পূর্ণাঙ্ক raw_start = গ্ৰন্থি_idxs[idx];
    পূর্ণাঙ্ক       raw_end   = গ্ৰন্থি_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    যদি (with_norm && !with_scale) {
        যদি (raw_end + 1 >= gf->n_nodes) {
            ঘূৰাই মিছা;
        }
        ধ্রুৱ ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        যদি (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            ঘূৰাই মিছা;
        }
        raw_end++;
    }

    ধ্রুৱ পূর্ণাঙ্ক raw_গুনতি = raw_end - raw_start + 1;
    যদি (raw_গুনতি != (পূর্ণাঙ্ক) ops_all.size()) {
        ঘূৰাই মিছা;
    }

    পূর্ণাঙ্ক raw_idxs[GGML_METAL_FUSION_MAX];
    বাবে (পূর্ণাঙ্ক i = 0; i < raw_গুনতি; ++i) {
        raw_idxs[i] = raw_start + i;
        যদি (gf->nodes[raw_start + i]->op != ops_all[i]) {
            ঘূৰাই মিছা;
        }
    }

    ধ্রুৱ ggml_tensor * softmax        = gf->nodes[raw_start];
    ধ্রুৱ ggml_tensor * probs_reshaped = gf->nodes[raw_start + 1];
    ধ্রুৱ ggml_tensor * argsort        = gf->nodes[raw_start + 2];
    ধ্রুৱ ggml_tensor * ids            = gf->nodes[raw_start + 3];
    ধ্রুৱ ggml_tensor * get_rows       = gf->nodes[raw_start + 4];
    ধ্রুৱ ggml_tensor * out            = gf->nodes[raw_end];
    ধ্রুৱ ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    ভিন্ন scale   = 1.0f;
    ভিন্ন max_bias = 0.0f;
    memcpy(&scale,    ((ধ্রুৱ int32_t *) softmax->op_params) + 0, পরিমাণ(scale));
    memcpy(&max_bias, ((ধ্রুৱ int32_t *) softmax->op_params) + 1, পরিমাণ(max_bias));
    যদি (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        ঘূৰাই মিছা;
    }

    যদি (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        ঘূৰাই মিছা;
    }

    ধ্রুৱ int64_t n_expert      = logits->ne[0];
    ধ্রুৱ int64_t n_tokens      = logits->ne[1];
    ধ্রুৱ int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    যদি (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        ঘূৰাই মিছা;
    }

    যদি (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        ঘূৰাই মিছা;
    }

    যদি (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        ঘূৰাই মিছা;
    }

    যদি (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        ঘূৰাই মিছা;
    }

    যদি (with_norm) {
        ধ্রুৱ ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
        ধ্রুৱ ggml_tensor * sum_rows         = gf->nodes[raw_start + 6];
        ধ্রুৱ ggml_tensor * clamp            = gf->nodes[raw_start + 7];
        ধ্রুৱ ggml_tensor * div              = gf->nodes[raw_start + 8];
        ধ্রুৱ ggml_tensor * out_reshaped     = gf->nodes[raw_start + 9];

        যদি (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            clamp->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            out_reshaped->src[0] != div) {
            ঘূৰাই মিছা;
        }

        যদি (with_scale) {
            ধ্রুৱ ggml_tensor * scale_গ্ৰন্থি = gf->nodes[raw_start + 10];
            যদি (scale_গ্ৰন্থি->src[0] != out_reshaped) {
                ঘূৰাই মিছা;
            }
        }
    } অন্যথা যদি (with_scale) {
        ধ্রুৱ ggml_tensor * scale_গ্ৰন্থি = gf->nodes[raw_start + 5];
        যদি (scale_গ্ৰন্থি->src[0] != get_rows) {
            ঘূৰাই মিছা;
        }
    }

    ধ্রুৱ পূর্ণাঙ্ক outputs[2] = { raw_start + 3, raw_end };
    যদি (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_গুনতি, ops_all.data(), outputs, 2)) {
        ঘূৰাই মিছা;
    }

    যদি (mode == GGML_METAL_FUSION_FULL) {
        যদি (!logits->data || !out->data || !ids->data) {
            ঘূৰাই মিছা;
        }
    }

    ঘূৰাই সঁচা;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

সংরচনা ggml_metal_moe_reduce_match {
    ধ্রুৱ ggml_tensor * experts;
    ধ্রুৱ ggml_tensor * weights;
    ধ্রুৱ ggml_tensor * dst;
    পূর্ণাঙ্ক গ্ৰন্থি_গুনতি;
};

স্থির তার্কিক ggml_metal_fusion_match_moe_reduce(
        ধ্রুৱ ggml_cgraph * gf, পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    যদি (match == nullptr || গ্ৰন্থি_idx < 0 || গ্ৰন্থি_idx + (পূর্ণাঙ্ক) ops_all.size() > gf->n_nodes) {
        ঘূৰাই মিছা;
    }

    ধ্রুৱ ggml_tensor * mul = gf->nodes[গ্ৰন্থি_idx];
    যদি (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        ঘূৰাই মিছা;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    ধ্রুৱ পূর্ণাঙ্ক raw_গুনতি     = (পূর্ণাঙ্ক) ops_all.size();
    ধ্রুৱ পূর্ণাঙ্ক n_expert_used = raw_গুনতি / 2;

    যদি (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_গুনতি != 2 * n_expert_used) {
        ঘূৰাই মিছা;
    }

    পূর্ণাঙ্ক n_views = 0;
    যেতিয়ালৈ (গ্ৰন্থি_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[গ্ৰন্থি_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    যদি (n_views != n_expert_used) {
        ঘূৰাই মিছা;
    }

    বাবে (পূর্ণাঙ্ক i = n_expert_used + 1; i < raw_গুনতি; ++i) {
        যদি (gf->nodes[গ্ৰন্থি_idx + i]->op != GGML_OP_ADD) {
            ঘূৰাই মিছা;
        }
    }

    পূর্ণাঙ্ক raw_idxs[GGML_METAL_FUSION_MAX];
    বাবে (পূর্ণাঙ্ক i = 0; i < raw_গুনতি; ++i) {
        raw_idxs[i] = গ্ৰন্থি_idx + i;
        যদি (gf->nodes[গ্ৰন্থি_idx + i]->op != ops_all[i]) {
            ঘূৰাই মিছা;
        }
    }

    ধ্রুৱ ggml_tensor * experts = mul->src[0];
    ধ্রুৱ ggml_tensor * weights = mul->src[1];
    ধ্রুৱ ggml_tensor * dst     = gf->nodes[গ্ৰন্থি_idx + raw_গুনতি - 1];

    যদি (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ঘূৰাই মিছা;
    }

    ধ্রুৱ int64_t n_embd   = experts->ne[0];
    ধ্রুৱ int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    যদি (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        ঘূৰাই মিছা;
    }

    যদি (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        ঘূৰাই মিছা;
    }

    বাবে (পূর্ণাঙ্ক i = 1; i <= n_expert_used; ++i) {
        ধ্রুৱ ggml_tensor * view = gf->nodes[গ্ৰন্থি_idx + i];
        যদি (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            ঘূৰাই মিছা;
        }
    }

    ধ্রুৱ ggml_tensor * prev_add = nullptr;
    বাবে (পূর্ণাঙ্ক j = 1; j < n_expert_used; ++j) {
        ধ্রুৱ ggml_tensor * add = gf->nodes[গ্ৰন্থি_idx + n_expert_used + j];
        ধ্রুৱ ggml_tensor * rhs = gf->nodes[গ্ৰন্থি_idx + j + 1];
        ধ্রুৱ ggml_tensor * lhs = j == 1 ? gf->nodes[গ্ৰন্থি_idx + 1] : prev_add;
        যদি (add->src[0] != lhs || add->src[1] != rhs) {
            ঘূৰাই মিছা;
        }
        prev_add = add;
    }

    ধ্রুৱ পূর্ণাঙ্ক outputs[1] = { গ্ৰন্থি_idx + raw_গুনতি - 1 };
    যদি (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_গুনতি, ops_all.data(), outputs, 1)) {
        ঘূৰাই মিছা;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->গ্ৰন্থি_গুনতি = raw_গুনতি;
    ঘূৰাই সঁচা;
}

স্থির তার্কিক ggml_metal_fusion_check_moe_reduce(
        ধ্রুৱ ggml_metal_fusion       * fusion,
        ধ্রুৱ ggml_tensor * ধ্রুৱ     * nodes,
        ধ্রুৱ ggml_cgraph             * gf,
        ধ্রুৱ পূর্ণাঙ্ক                     * গ্ৰন্থি_idxs,
              পূর্ণাঙ্ক                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    যদি (!ggml_metal_fusion_match_moe_reduce(gf, গ্ৰন্থি_idxs[idx], fusion->ops_all, &match)) {
        ঘূৰাই মিছা;
    }

    যদি ((পূর্ণাঙ্ক) fusion->ops.size() != match.experts->ne[1]) {
        ঘূৰাই মিছা;
    }

    ধ্রুৱ পূর্ণাঙ্ক raw_end = গ্ৰন্থি_idxs[idx] + match.গ্ৰন্থি_গুনতি - 1;
    যদি (গ্ৰন্থি_idxs[idx + (পূর্ণাঙ্ক) fusion->ops.size() - 1] != raw_end) {
        ঘূৰাই মিছা;
    }

    যদি (mode == GGML_METAL_FUSION_FULL) {
        যদি (!match.experts->data || !match.weights->data || !match.dst->data) {
            ঘূৰাই মিছা;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        যদি (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            ঘূৰাই মিছা;
        }
    }

    ঘূৰাই সঁচা;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
স্থির তার্কিক ggml_metal_tensor_is_weight(ধ্রুৱ সংরচনা ggml_tensor * t) {
    ধ্রুৱ ggml_tensor * base = t->view_src != রিক্ত ? t->view_src : t;

    ঘূৰাই base->buffer != রিক্ত && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

স্থির ধ্রুৱ সংরচনা ggml_tensor * ggml_metal_mul_mat_add_operand(ধ্রুৱ সংরচনা ggml_tensor * mm, ধ্রুৱ সংরচনা ggml_tensor * add) {
    যদি (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        ঘূৰাই রিক্ত;
    }

    ধ্রুৱ ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    ধ্রুৱ তার্কিক ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    ঘূৰাই ok ? other : রিক্ত;
}

স্থির ধ্রুৱ সংরচনা ggml_tensor * ggml_metal_mul_mat_add_residual(ধ্রুৱ সংরচনা ggml_tensor * mm, ধ্রুৱ সংরচনা ggml_tensor * add) {
    ধ্রুৱ ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    ধ্রুৱ তার্কিক ok = res != রিক্ত && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    ঘূৰাই ok ? res : রিক্ত;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
স্থির তার্কিক ggml_metal_fusion_check_mul_mat_add(
        ধ্রুৱ ggml_metal_fusion       * fusion,
        ধ্রুৱ ggml_tensor * ধ্রুৱ     * nodes,
        ধ্রুৱ ggml_cgraph             * gf,
        ধ্রুৱ পূর্ণাঙ্ক                     * গ্ৰন্থি_idxs,
              পূর্ণাঙ্ক                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(গ্ৰন্থি_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    ধ্রুৱ ggml_tensor * mm  = nodes[0];
    ধ্রুৱ ggml_tensor * add = nodes[1];

    যদি (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        ঘূৰাই মিছা;
    }

    যদি (mode == GGML_METAL_FUSION_STRUCTURAL) {
        ঘূৰাই সঁচা;
    }

    ধ্রুৱ ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    যদি (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        ঘূৰাই মিছা;
    }

    ঘূৰাই !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

স্থির ধ্রুৱ std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
স্থির ধ্রুৱ std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
স্থির ধ্রুৱ std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
স্থির ধ্রুৱ std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
স্থির ধ্রুৱ std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
স্থির ধ্রুৱ std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

স্থির ধ্রুৱ std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
স্থির ধ্রুৱ std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
স্থির ধ্রুৱ std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
স্থির ধ্রুৱ std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
স্থির ধ্রুৱ std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
স্থির ধ্রুৱ std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
স্থির ধ্রুৱ std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

স্থির ধ্রুৱ std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

স্থির ধ্রুৱ std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

স্থির ধ্রুৱ std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
স্থির ধ্রুৱ std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
স্থির ধ্রুৱ std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
স্থির ধ্রুৱ std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
স্থির ধ্রুৱ std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
স্থির ধ্রুৱ std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
স্থির ধ্রুৱ std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

স্থির ধ্রুৱ std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };

স্থির ধ্রুৱ std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     মিছা, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_norm_mul_add,           {},     মিছা, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     মিছা, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     মিছা, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_rms_norm_mul_add,       {},     মিছা, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     মিছা, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_2,                  {},     মিছা, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_3,                  {},     মিছা, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_4,                  {},     মিছা, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_5,                  {},     মিছা, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_6,                  {},     মিছা, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_7,                  {},     মিছা, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     মিছা, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     সঁচা,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    সঁচা,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    সঁচা,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    সঁচা,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    সঁচা,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     সঁচা,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     সঁচা,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     সঁচা,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     সঁচা,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     সঁচা,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     সঁচা,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     সঁচা,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     মিছা, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_ADD,    ops_mul_mat_add,            {},     মিছা, ggml_metal_fusion_check_mul_mat_add },
};

// ---- alloc deps -----------------------------------------------------------

স্থির তার্কিক ggml_metal_fusion_match_raw_pattern(
        ধ্রুৱ ggml_cgraph * gf, পূর্ণাঙ্ক গ্ৰন্থি_idx, ধ্রুৱ std::vector<ggml_op> & ops) {
    যদি (গ্ৰন্থি_idx < 0 || গ্ৰন্থি_idx + (পূর্ণাঙ্ক) ops.size() > gf->n_nodes) {
        ঘূৰাই মিছা;
    }

    বাবে (পূর্ণাঙ্ক i = 0; i < (পূর্ণাঙ্ক) ops.size(); ++i) {
        যদি (gf->nodes[গ্ৰন্থি_idx + i]->op != ops[i]) {
            ঘূৰাই মিছা;
        }
    }

    ঘূৰাই সঁচা;
}

স্থির শূন্য ggml_metal_fusion_add_pattern_alloc_deps(
        শূন্য * user_data,
        শূন্য (*add_alloc_dep)(শূন্য *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        ধ্রুৱ ggml_metal_fusion * fusion,
        পূর্ণাঙ্ক গ্ৰন্থি_idx) {
    ধ্রুৱ পূর্ণাঙ্ক last_গ্ৰন্থি = গ্ৰন্থি_idx + (পূর্ণাঙ্ক) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    বাবে (পূর্ণাঙ্ক j = 0; j < (পূর্ণাঙ্ক) fusion->ops_all.size(); ++j) {
        ggml_tensor * গ্ৰন্থি = gf->nodes[গ্ৰন্থি_idx + j];
        বাবে (পূর্ণাঙ্ক s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = গ্ৰন্থি->src[s];
            যদি (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_গ্ৰন্থি]);
            }
        }
        seen.insert(গ্ৰন্থি);
    }
}

শূন্য ggml_metal_fusion_add_alloc_deps(
        শূন্য * user_data,
        শূন্য (*add_alloc_dep)(শূন্য *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    বাবে (পূর্ণাঙ্ক i = 0; i < gf->n_nodes; ++i) {
        ধ্রুৱ ggml_metal_fusion * best = nullptr;
        পূর্ণাঙ্ক best_raw = 0;

        বাবে (ধ্রুৱ ggml_metal_fusion & fusion : ggml_metal_fusions) {
            যদি ((পূর্ণাঙ্ক) fusion.ops_all.size() <= best_raw) {
                অনুৱর্ত;
            }
            যদি (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (পূর্ণাঙ্ক) fusion.ops_all.size();
            }
        }

        যদি (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

স্থির std::string ggml_metal_fusion_label(ধ্রুৱ ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    বাবে (পূর্ণাঙ্ক j = 0; j < (পূর্ণাঙ্ক) fusion->ops.size(); j++) {
        যদি (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    ঘূৰাই label;
}

সংরচনা ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    তার্কিক enabled;
    তার্কিক stats;
    তার্কিক labels_set;
    পূর্ণাঙ্ক  debug;
};

সংরচনা ggml_metal_fusion_info * ggml_metal_fusion_info_init(তার্কিক enabled, পূর্ণাঙ্ক debug) {
    ggml_metal_fusion_info * finfo = নূতন ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = মিছা;
    finfo->debug      = debug;

    যদি (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    ঘূৰাই finfo;
}

শূন্য ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

তার্কিক ggml_metal_fusion_info_enabled(ধ্রুৱ ggml_metal_fusion_info * finfo) {
    ঘূৰাই finfo->enabled;
}

তার্কিক ggml_metal_fusion_info_stats(ধ্রুৱ ggml_metal_fusion_info * finfo) {
    ঘূৰাই finfo->stats;
}

পূর্ণাঙ্ক ggml_metal_fusion_info_debug(ধ্রুৱ ggml_metal_fusion_info * finfo) {
    ঘূৰাই finfo->debug;
}

পূর্ণাঙ্ক ggml_metal_fusion_info_n_fusions(ধ্রুৱ ggml_metal_fusion_info * finfo) {
    ঘূৰাই (পূর্ণাঙ্ক) finfo->labels.size();
}

ধ্রুৱ ৱর্ণ * ggml_metal_fusion_info_label(ধ্রুৱ ggml_metal_fusion_info * finfo, পূর্ণাঙ্ক idx) {
    GGML_ASSERT(idx >= 0 && idx < (পূর্ণাঙ্ক) finfo->labels.size());
    ঘূৰাই finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_গুনতি(ধ্রুৱ ggml_metal_fusion_info * finfo, পূর্ণাঙ্ক idx) {
    GGML_ASSERT(idx >= 0 && idx < (পূর্ণাঙ্ক) finfo->counts.size());
    ঘূৰাই finfo->counts[idx];
}

শূন্য ggml_metal_fusion_info_গুনতি_fusion(ggml_metal_fusion_info * finfo, ধ্রুৱ ggml_metal_fusion * fusion) {
    যদি (!finfo->stats || fusion == nullptr) {
        ঘূৰাই;
    }

    ধ্রুৱ ptrdiff_t idx = fusion - ggml_metal_fusions.data();
    যদি (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

শূন্য ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, তার্কিক enabled) {
    finfo->enabled = enabled;
}

শূন্য ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    যদি (finfo->labels_set) {
        ঘূৰাই;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    বাবে (ধ্রুৱ ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = সঁচা;
}

শূন্য ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = সঁচা;
    ggml_metal_fusion_info_labels_init(finfo);
}

শূন্য ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

পূর্ণাঙ্ক ggml_metal_fusion_info_stats_get(ধ্রুৱ ggml_metal_fusion_info * finfo, ধ্রুৱ ৱর্ণ ** labels, uint64_t * counts, পূর্ণাঙ্ক n) {
    ধ্রুৱ পূর্ণাঙ্ক n_fusions = (পূর্ণাঙ্ক) finfo->labels.size();

    যদি (labels == nullptr) {
        ঘূৰাই n_fusions;
    }

    ধ্রুৱ পূর্ণাঙ্ক n_fill = std::min(n, n_fusions);
    বাবে (পূর্ণাঙ্ক i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        যদি (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    ঘূৰাই n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
স্থির তার্কিক ggml_metal_fusion_check_memory_ranges(
        ধ্রুৱ ggml_metal_fusion      * fusion,
        ধ্রুৱ ggml_tensor * ধ্রুৱ    * nodes,
        পূর্ণাঙ্ক                            গ্ৰন্থি_গুনতি) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](ধ্রুৱ ggml_tensor * a, ধ্রুৱ ggml_tensor * b) {
        যদি (a->data && b->data && a->data == b->data) {
            ঘূৰাই সঁচা;
        }
        বাবে (ধ্রুৱ ggml_tensor * v = a; v; v = v->view_src) {
            যদি (v == b) {
                ঘূৰাই সঁচা;
            }
        }
        বাবে (ধ্রুৱ ggml_tensor * v = b; v; v = v->view_src) {
            যদি (v == a) {
                ঘূৰাই সঁচা;
            }
        }
        ঘূৰাই মিছা;
    };

    auto nodes_overlap = [](ধ্রুৱ ggml_tensor * a, ধ্রুৱ ggml_tensor * b) {
        যদি (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            ঘূৰাই মিছা;
        }

        যদি (a->buffer != b->buffer) {
            ঘূৰাই মিছা;
        }

        ধ্রুৱ int64_t a_start = (int64_t) a->data;
        ধ্রুৱ int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        ধ্রুৱ int64_t b_start = (int64_t) b->data;
        ধ্রুৱ int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        ঘূৰাই (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](ধ্রুৱ ggml_tensor * src, ধ্রুৱ ggml_tensor * ধ্রুৱ * nodes, পূর্ণাঙ্ক j) {
        বাবে (পূর্ণাঙ্ক k = 0; k < j; ++k) {
            যদি (src == nodes[k]) {
                ঘূৰাই সঁচা;
            }
            বাবে (ধ্রুৱ ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                যদি (view_src == nodes[k]) {
                    ঘূৰাই সঁচা;
                }
            }
        }
        ঘূৰাই মিছা;
    };

    auto check_dst = [&](ধ্রুৱ ggml_tensor * dst) {
        বাবে (পূর্ণাঙ্ক j = 0; j < গ্ৰন্থি_গুনতি; ++j) {
            বাবে (পূর্ণাঙ্ক s = 0; s < GGML_MAX_SRC; ++s) {
                ধ্রুৱ ggml_tensor * src = nodes[j]->src[s];
                যদি (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    অনুৱর্ত;
                }

                যদি (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    ঘূৰাই মিছা;
                }
            }
        }
        ঘূৰাই সঁচা;
    };

    যদি (!check_dst(nodes[গ্ৰন্থি_গুনতি - 1])) {
        ঘূৰাই মিছা;
    }

    বাবে (পূর্ণাঙ্ক offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < গ্ৰন্থি_গুনতি);
        যদি (!check_dst(nodes[offset])) {
            ঘূৰাই মিছা;
        }
    }

    ঘূৰাই সঁচা;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
ধ্রুৱ ggml_metal_fusion * ggml_metal_fusion_next(
        ধ্রুৱ ggml_cgraph * gf,
        ধ্রুৱ পূর্ণাঙ্ক * গ্ৰন্থি_idxs,
        পূর্ণাঙ্ক n_idxs,
        পূর্ণাঙ্ক idx,
        ggml_metal_fusion_mode mode,
        পূর্ণাঙ্ক * n_out) {
    ধ্রুৱ ggml_metal_fusion * res = nullptr;
    পূর্ণাঙ্ক best = 1;

    বাবে (ধ্রুৱ ggml_metal_fusion & fusion : ggml_metal_fusions) {
        ধ্রুৱ পূর্ণাঙ্ক n_ops = (পূর্ণাঙ্ক) fusion.ops.size();

        // only look for a longer match than the current best
        যদি (n_ops <= best) {
            অনুৱর্ত;
        }
        যদি (idx + n_ops > n_idxs) {
            অনুৱর্ত;
        }

        ধ্রুৱ ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        তার্কিক ok = সঁচা;
        বাবে (পূর্ণাঙ্ক j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[গ্ৰন্থি_idxs[idx + j]];
            যদি (nodes[j]->op != fusion.ops[j]) {
                ok = মিছা;
                ভঙ্গ;
            }
        }
        যদি (!ok) {
            অনুৱর্ত;
        }

        যদি (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            বাবে (পূর্ণাঙ্ক j = 1; j < n_ops && ok; j++) {
                যদি (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = মিছা;
                    ভঙ্গ;
                }
                যদি (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = মিছা;
                    ভঙ্গ;
                }
            }
            যদি (!ok) {
                অনুৱর্ত;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            পূর্ণাঙ্ক outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = গ্ৰন্থি_idxs[idx + n_ops - 1];
            বাবে (size_t i = 0; i < fusion.outs.size(); ++i) {
                ধ্রুৱ পূর্ণাঙ্ক out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = গ্ৰন্থি_idxs[idx + out_offset];
            }

            ধ্রুৱ পূর্ণাঙ্ক n_outputs = 1 + (পূর্ণাঙ্ক) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            যদি (!ggml_can_fuse_subgraph_ext(gf, গ্ৰন্থি_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                অনুৱর্ত;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        যদি (fusion.check && !fusion.check(&fusion, nodes, gf, গ্ৰন্থি_idxs, idx, mode)) {
            অনুৱর্ত;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        যদি (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            অনুৱর্ত;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    ঘূৰাই res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
পূর্ণাঙ্ক ggml_metal_fusion_max(ধ্রুৱ ggml_cgraph * gf, পূর্ণাঙ্ক idx) {
    // a view node cannot start a pattern - pack it alone
    যদি (ggml_op_is_empty(gf->nodes[idx]->op)) {
        ঘূৰাই 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    পূর্ণাঙ্ক idxs[GGML_METAL_FUSION_MAX];
    পূর্ণাঙ্ক n_idxs = 0;
    বাবে (পূর্ণাঙ্ক i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        যদি (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    যদি (n_idxs == 0) {
        ঘূৰাই 1;
    }

    পূর্ণাঙ্ক total = 0;
    পূর্ণাঙ্ক i_f = 0;

    যেতিয়ালৈ (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        পূর্ণাঙ্ক len = 1;
        ধ্রুৱ ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        যদি (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            ভঙ্গ;
        }

        total += len;
        i_f += len;
    }

    যদি (i_f == 0) {
        ঘূৰাই 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    ঘূৰাই std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
