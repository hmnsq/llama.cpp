#ଅନ୍ତର୍ଭାଵ "ggml-metal-fusion.h"

#ଅନ୍ତର୍ଭାଵ "ggml-metal-common.h"
#ଅନ୍ତର୍ଭାଵ "ggml-metal-device.h"

#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstring>
#ଅନ୍ତର୍ଭାଵ <set>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
ସ୍ଥିର std::vector<ggml_op> ggml_metal_fusion_filter_ops(ଅଚଳ std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    ପାଇଁ (ggml_op op : ops_all) {
        ଯଦି (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    ଫେରନ୍ତୁ ops;
}

ସଂରଚନା ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    ତାର୍କିକ unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    ତାର୍କିକ (*check)(ଅଚଳ ସଂରଚନା ggml_metal_fusion   * fusion,
                  ଅଚଳ ସଂରଚନା ggml_tensor * ଅଚଳ * nodes,
                  ଅଚଳ ସଂରଚନା ggml_cgraph         * gf,
                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        * ଗଣ୍ଠି_idxs,
                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            ଅଚଳ std::vector<ggml_op> & ops_all,
            ଅଚଳ std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> & outs,
            ତାର୍କିକ unsafe,
            ତାର୍କିକ (*check)(ଅଚଳ ସଂରଚନା ggml_metal_fusion   * fusion,
                          ଅଚଳ ସଂରଚନା ggml_tensor * ଅଚଳ * nodes,
                          ଅଚଳ ସଂରଚନା ggml_cgraph         * gf,
                          ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        * ଗଣ୍ଠି_idxs,
                                ପୂର୍ଣ୍ଣସଂଖ୍ୟା                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(ଅଚଳ ggml_metal_fusion * fusion) {
    ଫେରନ୍ତୁ fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
ସ୍ଥିର ତାର୍କିକ ggml_metal_fusion_same_buffer(ଅଚଳ ggml_tensor * a, ଅଚଳ ggml_tensor * b) {
    ଯଦି (!a || !b) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    ଫେରନ୍ତୁ ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
ସ୍ଥିର ତାର୍କିକ ggml_metal_fusion_overlap(ଅଚଳ ggml_tensor * a, ଅଚଳ ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ଅଚଳ ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    ଅଚଳ ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    ଯଦି (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
ସ୍ଥିର ତାର୍କିକ ggml_metal_fusion_check_norm(
        ଅଚଳ ggml_metal_fusion       * fusion,
        ଅଚଳ ggml_tensor * ଅଚଳ     * nodes,
        ଅଚଳ ggml_cgraph             * gf,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                     * ଗଣ୍ଠି_idxs,
              ପୂର୍ଣ୍ଣସଂଖ୍ୟା                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(ଗଣ୍ଠି_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    ଯଦି (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        ଅଚଳ ggml_tensor * scale = nodes[1];
        ଯଦି (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 1; j < (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        ଯଦି (nodes[j]->src[0] != nodes[j - 1]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        // the weight/bias must have the same row width as the norm input
        ଯଦି (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (nodes[j]->type != GGML_TYPE_F32) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// SSM_CONV + UNARY (silu)
ସ୍ଥିର ତାର୍କିକ ggml_metal_fusion_check_ssm_conv_silu(
        ଅଚଳ ggml_metal_fusion       * fusion,
        ଅଚଳ ggml_tensor * ଅଚଳ     * nodes,
        ଅଚଳ ggml_cgraph             * gf,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                     * ଗଣ୍ଠି_idxs,
              ପୂର୍ଣ୍ଣସଂଖ୍ୟା                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(ଗଣ୍ଠି_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    ଅଚଳ ggml_tensor * conv = nodes[0];
    ଅଚଳ ggml_tensor * un   = nodes[1];

    ଯଦି (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
ସ୍ଥିର ତାର୍କିକ ggml_metal_fusion_check_add_chain(
        ଅଚଳ ggml_metal_fusion       * fusion,
        ଅଚଳ ggml_tensor * ଅଚଳ     * nodes,
        ଅଚଳ ggml_cgraph             * gf,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                     * ଗଣ୍ଠି_idxs,
              ପୂର୍ଣ୍ଣସଂଖ୍ୟା                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(ଗଣ୍ଠି_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 1; j < (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) fusion->ops.size(); j++) {
        ଯଦି (nodes[j]->src[0] != nodes[j - 1]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (mode == GGML_METAL_FUSION_FULL) {
            ଯଦି (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
ସ୍ଥିର ତାର୍କିକ ggml_metal_fusion_check_gdn_cache(
        ଅଚଳ ggml_metal_fusion       * fusion,
        ଅଚଳ ggml_tensor * ଅଚଳ     * nodes,
        ଅଚଳ ggml_cgraph             * gf,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                     * ଗଣ୍ଠି_idxs,
              ପୂର୍ଣ୍ଣସଂଖ୍ୟା                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(ଗଣ୍ଠି_idxs);
    GGML_UNUSED(idx);

    ଅଚଳ ggml_tensor * gdn = nodes[0];
    ଅଚଳ ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    ଯଦି (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int64_t S_v      = gdn->src[2]->ne[0];
    ଅଚଳ int64_t H        = gdn->src[2]->ne[1];
    ଅଚଳ int64_t n_tokens = gdn->src[2]->ne[2];
    ଅଚଳ int64_t n_seqs   = gdn->src[2]->ne[3];
    ଅଚଳ int64_t K        = ggml_get_op_params_i32(gdn, 0);
    ଅଚଳ size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    ଅଚଳ int64_t D         = S_v * S_v * H;
    ଅଚଳ int64_t n_written = std::min<int64_t>(n_tokens, K);

    ଅଚଳ ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    ଅଚଳ ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    ଯଦି (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    ଯଦି (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        ଯଦି (dst->data == nullptr) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
ସ୍ଥିର ତାର୍କିକ ggml_metal_fusion_check_snake(
        ଅଚଳ ggml_metal_fusion       * fusion,
        ଅଚଳ ggml_tensor * ଅଚଳ     * nodes,
        ଅଚଳ ggml_cgraph             * gf,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                     * ଗଣ୍ଠି_idxs,
              ପୂର୍ଣ୍ଣସଂଖ୍ୟା                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(ଗଣ୍ଠି_idxs);
    GGML_UNUSED(idx);

    ଅଚଳ ggml_tensor * mul0     = nodes[0];
    ଅଚଳ ggml_tensor * sin_ଗଣ୍ଠି = nodes[1];
    ଅଚଳ ggml_tensor * sqr      = nodes[2];
    ଅଚଳ ggml_tensor * mul1     = nodes[3];
    ଅଚଳ ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    ଅଚଳ ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ଅଚଳ ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ଅଚଳ ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    ଅଚଳ ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    ଅଚଳ ତାର୍କିକ types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_ଗଣ୍ଠି->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    ଅଚଳ ତାର୍କିକ shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    ଅଚଳ ତାର୍କିକ dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    ଅଚଳ ତାର୍କିକ contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    ଫେରନ୍ତୁ types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

ସ୍ଥିର ତାର୍କିକ ggml_metal_fusion_check_topk_moe(
        ଅଚଳ ggml_metal_fusion       * fusion,
        ଅଚଳ ggml_tensor * ଅଚଳ     * nodes,
        ଅଚଳ ggml_cgraph             * gf,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                     * ଗଣ୍ଠି_idxs,
              ପୂର୍ଣ୍ଣସଂଖ୍ୟା                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ops = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) fusion->ops.size();

    ଅଚଳ ତାର୍କିକ with_norm  = n_ops >= 6;
    ଅଚଳ ତାର୍କିକ with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    ଅଚଳ std::vector<ggml_op> & ops_all = fusion->ops_all;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା raw_start = ଗଣ୍ଠି_idxs[idx];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା       raw_end   = ଗଣ୍ଠି_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    ଯଦି (with_norm && !with_scale) {
        ଯଦି (raw_end + 1 >= gf->n_nodes) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଅଚଳ ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        ଯଦି (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        raw_end++;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା raw_ଗଣନା = raw_end - raw_start + 1;
    ଯଦି (raw_ଗଣନା != (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ops_all.size()) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା raw_idxs[GGML_METAL_FUSION_MAX];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < raw_ଗଣନା; ++i) {
        raw_idxs[i] = raw_start + i;
        ଯଦି (gf->nodes[raw_start + i]->op != ops_all[i]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଅଚଳ ggml_tensor * softmax        = gf->nodes[raw_start];
    ଅଚଳ ggml_tensor * probs_reshaped = gf->nodes[raw_start + 1];
    ଅଚଳ ggml_tensor * argsort        = gf->nodes[raw_start + 2];
    ଅଚଳ ggml_tensor * ids            = gf->nodes[raw_start + 3];
    ଅଚଳ ggml_tensor * get_rows       = gf->nodes[raw_start + 4];
    ଅଚଳ ggml_tensor * out            = gf->nodes[raw_end];
    ଅଚଳ ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    ଦଶମିକ scale   = 1.0f;
    ଦଶମିକ max_bias = 0.0f;
    memcpy(&scale,    ((ଅଚଳ int32_t *) softmax->op_params) + 0, ଆକାର(scale));
    memcpy(&max_bias, ((ଅଚଳ int32_t *) softmax->op_params) + 1, ଆକାର(max_bias));
    ଯଦି (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int64_t n_expert      = logits->ne[0];
    ଅଚଳ int64_t n_tokens      = logits->ne[1];
    ଅଚଳ int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    ଯଦି (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (with_norm) {
        ଅଚଳ ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
        ଅଚଳ ggml_tensor * sum_rows         = gf->nodes[raw_start + 6];
        ଅଚଳ ggml_tensor * clamp            = gf->nodes[raw_start + 7];
        ଅଚଳ ggml_tensor * div              = gf->nodes[raw_start + 8];
        ଅଚଳ ggml_tensor * out_reshaped     = gf->nodes[raw_start + 9];

        ଯଦି (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            clamp->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            out_reshaped->src[0] != div) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (with_scale) {
            ଅଚଳ ggml_tensor * scale_ଗଣ୍ଠି = gf->nodes[raw_start + 10];
            ଯଦି (scale_ଗଣ୍ଠି->src[0] != out_reshaped) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }
    } ଅନ୍ୟଥା ଯଦି (with_scale) {
        ଅଚଳ ggml_tensor * scale_ଗଣ୍ଠି = gf->nodes[raw_start + 5];
        ଯଦି (scale_ଗଣ୍ଠି->src[0] != get_rows) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା outputs[2] = { raw_start + 3, raw_end };
    ଯଦି (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_ଗଣନା, ops_all.data(), outputs, 2)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (mode == GGML_METAL_FUSION_FULL) {
        ଯଦି (!logits->data || !out->data || !ids->data) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

ସଂରଚନା ggml_metal_moe_reduce_match {
    ଅଚଳ ggml_tensor * experts;
    ଅଚଳ ggml_tensor * weights;
    ଅଚଳ ggml_tensor * dst;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_ଗଣନା;
};

ସ୍ଥିର ତାର୍କିକ ggml_metal_fusion_match_moe_reduce(
        ଅଚଳ ggml_cgraph * gf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    ଯଦି (match == nullptr || ଗଣ୍ଠି_idx < 0 || ଗଣ୍ଠି_idx + (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ops_all.size() > gf->n_nodes) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ggml_tensor * mul = gf->nodes[ଗଣ୍ଠି_idx];
    ଯଦି (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା raw_ଗଣନା     = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ops_all.size();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_expert_used = raw_ଗଣନା / 2;

    ଯଦି (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_ଗଣନା != 2 * n_expert_used) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_views = 0;
    ଯେତେବେଳେ (ଗଣ୍ଠି_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[ଗଣ୍ଠି_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    ଯଦି (n_views != n_expert_used) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = n_expert_used + 1; i < raw_ଗଣନା; ++i) {
        ଯଦି (gf->nodes[ଗଣ୍ଠି_idx + i]->op != GGML_OP_ADD) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା raw_idxs[GGML_METAL_FUSION_MAX];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < raw_ଗଣନା; ++i) {
        raw_idxs[i] = ଗଣ୍ଠି_idx + i;
        ଯଦି (gf->nodes[ଗଣ୍ଠି_idx + i]->op != ops_all[i]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଅଚଳ ggml_tensor * experts = mul->src[0];
    ଅଚଳ ggml_tensor * weights = mul->src[1];
    ଅଚଳ ggml_tensor * dst     = gf->nodes[ଗଣ୍ଠି_idx + raw_ଗଣନା - 1];

    ଯଦି (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int64_t n_embd   = experts->ne[0];
    ଅଚଳ int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    ଯଦି (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 1; i <= n_expert_used; ++i) {
        ଅଚଳ ggml_tensor * view = gf->nodes[ଗଣ୍ଠି_idx + i];
        ଯଦି (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଅଚଳ ggml_tensor * prev_add = nullptr;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 1; j < n_expert_used; ++j) {
        ଅଚଳ ggml_tensor * add = gf->nodes[ଗଣ୍ଠି_idx + n_expert_used + j];
        ଅଚଳ ggml_tensor * rhs = gf->nodes[ଗଣ୍ଠି_idx + j + 1];
        ଅଚଳ ggml_tensor * lhs = j == 1 ? gf->nodes[ଗଣ୍ଠି_idx + 1] : prev_add;
        ଯଦି (add->src[0] != lhs || add->src[1] != rhs) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        prev_add = add;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା outputs[1] = { ଗଣ୍ଠି_idx + raw_ଗଣନା - 1 };
    ଯଦି (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_ଗଣନା, ops_all.data(), outputs, 1)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->ଗଣ୍ଠି_ଗଣନା = raw_ଗଣନା;
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ ggml_metal_fusion_check_moe_reduce(
        ଅଚଳ ggml_metal_fusion       * fusion,
        ଅଚଳ ggml_tensor * ଅଚଳ     * nodes,
        ଅଚଳ ggml_cgraph             * gf,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                     * ଗଣ୍ଠି_idxs,
              ପୂର୍ଣ୍ଣସଂଖ୍ୟା                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    ଯଦି (!ggml_metal_fusion_match_moe_reduce(gf, ଗଣ୍ଠି_idxs[idx], fusion->ops_all, &match)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି ((ପୂର୍ଣ୍ଣସଂଖ୍ୟା) fusion->ops.size() != match.experts->ne[1]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା raw_end = ଗଣ୍ଠି_idxs[idx] + match.ଗଣ୍ଠି_ଗଣନା - 1;
    ଯଦି (ଗଣ୍ଠି_idxs[idx + (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) fusion->ops.size() - 1] != raw_end) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (mode == GGML_METAL_FUSION_FULL) {
        ଯଦି (!match.experts->data || !match.weights->data || !match.dst->data) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        ଯଦି (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
ସ୍ଥିର ତାର୍କିକ ggml_metal_tensor_is_weight(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଅଚଳ ggml_tensor * base = t->view_src != ଖାଲି ? t->view_src : t;

    ଫେରନ୍ତୁ base->buffer != ଖାଲି && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_tensor * ggml_metal_mul_mat_add_operand(ଅଚଳ ସଂରଚନା ggml_tensor * mm, ଅଚଳ ସଂରଚନା ggml_tensor * add) {
    ଯଦି (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        ଫେରନ୍ତୁ ଖାଲି;
    }

    ଅଚଳ ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    ଅଚଳ ତାର୍କିକ ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    ଫେରନ୍ତୁ ok ? other : ଖାଲି;
}

ସ୍ଥିର ଅଚଳ ସଂରଚନା ggml_tensor * ggml_metal_mul_mat_add_residual(ଅଚଳ ସଂରଚନା ggml_tensor * mm, ଅଚଳ ସଂରଚନା ggml_tensor * add) {
    ଅଚଳ ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    ଅଚଳ ତାର୍କିକ ok = res != ଖାଲି && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    ଫେରନ୍ତୁ ok ? res : ଖାଲି;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
ସ୍ଥିର ତାର୍କିକ ggml_metal_fusion_check_mul_mat_add(
        ଅଚଳ ggml_metal_fusion       * fusion,
        ଅଚଳ ggml_tensor * ଅଚଳ     * nodes,
        ଅଚଳ ggml_cgraph             * gf,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                     * ଗଣ୍ଠି_idxs,
              ପୂର୍ଣ୍ଣସଂଖ୍ୟା                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(ଗଣ୍ଠି_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    ଅଚଳ ggml_tensor * mm  = nodes[0];
    ଅଚଳ ggml_tensor * add = nodes[1];

    ଯଦି (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (mode == GGML_METAL_FUSION_STRUCTURAL) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଅଚଳ ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    ଯଦି (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

ସ୍ଥିର ଅଚଳ std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };

ସ୍ଥିର ଅଚଳ std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     ମିଥ୍ୟା, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_norm_mul_add,           {},     ମିଥ୍ୟା, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     ମିଥ୍ୟା, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     ମିଥ୍ୟା, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_rms_norm_mul_add,       {},     ମିଥ୍ୟା, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     ମିଥ୍ୟା, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_2,                  {},     ମିଥ୍ୟା, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_3,                  {},     ମିଥ୍ୟା, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_4,                  {},     ମିଥ୍ୟା, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_5,                  {},     ମିଥ୍ୟା, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_6,                  {},     ମିଥ୍ୟା, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_7,                  {},     ମିଥ୍ୟା, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     ମିଥ୍ୟା, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     ସତ୍ୟ,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    ସତ୍ୟ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    ସତ୍ୟ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    ସତ୍ୟ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    ସତ୍ୟ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     ସତ୍ୟ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     ସତ୍ୟ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     ସତ୍ୟ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     ସତ୍ୟ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     ସତ୍ୟ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     ସତ୍ୟ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     ସତ୍ୟ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     ମିଥ୍ୟା, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_ADD,    ops_mul_mat_add,            {},     ମିଥ୍ୟା, ggml_metal_fusion_check_mul_mat_add },
};

// ---- alloc deps -----------------------------------------------------------

ସ୍ଥିର ତାର୍କିକ ggml_metal_fusion_match_raw_pattern(
        ଅଚଳ ggml_cgraph * gf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::vector<ggml_op> & ops) {
    ଯଦି (ଗଣ୍ଠି_idx < 0 || ଗଣ୍ଠି_idx + (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ops.size() > gf->n_nodes) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ops.size(); ++i) {
        ଯଦି (gf->nodes[ଗଣ୍ଠି_idx + i]->op != ops[i]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_metal_fusion_add_pattern_alloc_deps(
        ଶୂନ୍ୟ * user_data,
        ଶୂନ୍ୟ (*add_alloc_dep)(ଶୂନ୍ୟ *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        ଅଚଳ ggml_metal_fusion * fusion,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା last_ଗଣ୍ଠି = ଗଣ୍ଠି_idx + (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) fusion->ops_all.size(); ++j) {
        ggml_tensor * ଗଣ୍ଠି = gf->nodes[ଗଣ୍ଠି_idx + j];
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = ଗଣ୍ଠି->src[s];
            ଯଦି (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_ଗଣ୍ଠି]);
            }
        }
        seen.insert(ଗଣ୍ଠି);
    }
}

ଶୂନ୍ୟ ggml_metal_fusion_add_alloc_deps(
        ଶୂନ୍ୟ * user_data,
        ଶୂନ୍ୟ (*add_alloc_dep)(ଶୂନ୍ୟ *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < gf->n_nodes; ++i) {
        ଅଚଳ ggml_metal_fusion * best = nullptr;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା best_raw = 0;

        ପାଇଁ (ଅଚଳ ggml_metal_fusion & fusion : ggml_metal_fusions) {
            ଯଦି ((ପୂର୍ଣ୍ଣସଂଖ୍ୟା) fusion.ops_all.size() <= best_raw) {
                ଜାରି;
            }
            ଯଦି (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) fusion.ops_all.size();
            }
        }

        ଯଦି (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

ସ୍ଥିର std::string ggml_metal_fusion_label(ଅଚଳ ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) fusion->ops.size(); j++) {
        ଯଦି (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    ଫେରନ୍ତୁ label;
}

ସଂରଚନା ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    ତାର୍କିକ enabled;
    ତାର୍କିକ stats;
    ତାର୍କିକ labels_set;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା  debug;
};

ସଂରଚନା ggml_metal_fusion_info * ggml_metal_fusion_info_init(ତାର୍କିକ enabled, ପୂର୍ଣ୍ଣସଂଖ୍ୟା debug) {
    ggml_metal_fusion_info * finfo = ନୂତନ ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = ମିଥ୍ୟା;
    finfo->debug      = debug;

    ଯଦି (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    ଫେରନ୍ତୁ finfo;
}

ଶୂନ୍ୟ ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

ତାର୍କିକ ggml_metal_fusion_info_enabled(ଅଚଳ ggml_metal_fusion_info * finfo) {
    ଫେରନ୍ତୁ finfo->enabled;
}

ତାର୍କିକ ggml_metal_fusion_info_stats(ଅଚଳ ggml_metal_fusion_info * finfo) {
    ଫେରନ୍ତୁ finfo->stats;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_metal_fusion_info_debug(ଅଚଳ ggml_metal_fusion_info * finfo) {
    ଫେରନ୍ତୁ finfo->debug;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_metal_fusion_info_n_fusions(ଅଚଳ ggml_metal_fusion_info * finfo) {
    ଫେରନ୍ତୁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) finfo->labels.size();
}

ଅଚଳ ଅକ୍ଷର * ggml_metal_fusion_info_label(ଅଚଳ ggml_metal_fusion_info * finfo, ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx) {
    GGML_ASSERT(idx >= 0 && idx < (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) finfo->labels.size());
    ଫେରନ୍ତୁ finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_ଗଣନା(ଅଚଳ ggml_metal_fusion_info * finfo, ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx) {
    GGML_ASSERT(idx >= 0 && idx < (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) finfo->counts.size());
    ଫେରନ୍ତୁ finfo->counts[idx];
}

ଶୂନ୍ୟ ggml_metal_fusion_info_ଗଣନା_fusion(ggml_metal_fusion_info * finfo, ଅଚଳ ggml_metal_fusion * fusion) {
    ଯଦି (!finfo->stats || fusion == nullptr) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ptrdiff_t idx = fusion - ggml_metal_fusions.data();
    ଯଦି (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

ଶୂନ୍ୟ ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, ତାର୍କିକ enabled) {
    finfo->enabled = enabled;
}

ଶୂନ୍ୟ ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    ଯଦି (finfo->labels_set) {
        ଫେରନ୍ତୁ;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    ପାଇଁ (ଅଚଳ ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = ସତ୍ୟ;
}

ଶୂନ୍ୟ ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = ସତ୍ୟ;
    ggml_metal_fusion_info_labels_init(finfo);
}

ଶୂନ୍ୟ ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_metal_fusion_info_stats_get(ଅଚଳ ggml_metal_fusion_info * finfo, ଅଚଳ ଅକ୍ଷର ** labels, uint64_t * counts, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_fusions = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) finfo->labels.size();

    ଯଦି (labels == nullptr) {
        ଫେରନ୍ତୁ n_fusions;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_fill = std::min(n, n_fusions);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        ଯଦି (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    ଫେରନ୍ତୁ n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
ସ୍ଥିର ତାର୍କିକ ggml_metal_fusion_check_memory_ranges(
        ଅଚଳ ggml_metal_fusion      * fusion,
        ଅଚଳ ggml_tensor * ଅଚଳ    * nodes,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା                            ଗଣ୍ଠି_ଗଣନା) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](ଅଚଳ ggml_tensor * a, ଅଚଳ ggml_tensor * b) {
        ଯଦି (a->data && b->data && a->data == b->data) {
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        ପାଇଁ (ଅଚଳ ggml_tensor * v = a; v; v = v->view_src) {
            ଯଦି (v == b) {
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
        }
        ପାଇଁ (ଅଚଳ ggml_tensor * v = b; v; v = v->view_src) {
            ଯଦି (v == a) {
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    };

    auto nodes_overlap = [](ଅଚଳ ggml_tensor * a, ଅଚଳ ggml_tensor * b) {
        ଯଦି (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (a->buffer != b->buffer) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ int64_t a_start = (int64_t) a->data;
        ଅଚଳ int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        ଅଚଳ int64_t b_start = (int64_t) b->data;
        ଅଚଳ int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        ଫେରନ୍ତୁ (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](ଅଚଳ ggml_tensor * src, ଅଚଳ ggml_tensor * ଅଚଳ * nodes, ପୂର୍ଣ୍ଣସଂଖ୍ୟା j) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < j; ++k) {
            ଯଦି (src == nodes[k]) {
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
            ପାଇଁ (ଅଚଳ ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                ଯଦି (view_src == nodes[k]) {
                    ଫେରନ୍ତୁ ସତ୍ୟ;
                }
            }
        }
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    };

    auto check_dst = [&](ଅଚଳ ggml_tensor * dst) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < ଗଣ୍ଠି_ଗଣନା; ++j) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < GGML_MAX_SRC; ++s) {
                ଅଚଳ ggml_tensor * src = nodes[j]->src[s];
                ଯଦି (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    ଜାରି;
                }

                ଯଦି (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
            }
        }
        ଫେରନ୍ତୁ ସତ୍ୟ;
    };

    ଯଦି (!check_dst(nodes[ଗଣ୍ଠି_ଗଣନା - 1])) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < ଗଣ୍ଠି_ଗଣନା);
        ଯଦି (!check_dst(nodes[offset])) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
ଅଚଳ ggml_metal_fusion * ggml_metal_fusion_next(
        ଅଚଳ ggml_cgraph * gf,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * ଗଣ୍ଠି_idxs,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_idxs,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx,
        ggml_metal_fusion_mode mode,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା * n_out) {
    ଅଚଳ ggml_metal_fusion * res = nullptr;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା best = 1;

    ପାଇଁ (ଅଚଳ ggml_metal_fusion & fusion : ggml_metal_fusions) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ops = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) fusion.ops.size();

        // only look for a longer match than the current best
        ଯଦି (n_ops <= best) {
            ଜାରି;
        }
        ଯଦି (idx + n_ops > n_idxs) {
            ଜାରି;
        }

        ଅଚଳ ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        ତାର୍କିକ ok = ସତ୍ୟ;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[ଗଣ୍ଠି_idxs[idx + j]];
            ଯଦି (nodes[j]->op != fusion.ops[j]) {
                ok = ମିଥ୍ୟା;
                ଭାଙ୍ଗ;
            }
        }
        ଯଦି (!ok) {
            ଜାରି;
        }

        ଯଦି (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 1; j < n_ops && ok; j++) {
                ଯଦି (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = ମିଥ୍ୟା;
                    ଭାଙ୍ଗ;
                }
                ଯଦି (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = ମିଥ୍ୟା;
                    ଭାଙ୍ଗ;
                }
            }
            ଯଦି (!ok) {
                ଜାରି;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = ଗଣ୍ଠି_idxs[idx + n_ops - 1];
            ପାଇଁ (size_t i = 0; i < fusion.outs.size(); ++i) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = ଗଣ୍ଠି_idxs[idx + out_offset];
            }

            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_outputs = 1 + (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            ଯଦି (!ggml_can_fuse_subgraph_ext(gf, ଗଣ୍ଠି_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                ଜାରି;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        ଯଦି (fusion.check && !fusion.check(&fusion, nodes, gf, ଗଣ୍ଠି_idxs, idx, mode)) {
            ଜାରି;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        ଯଦି (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            ଜାରି;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    ଫେରନ୍ତୁ res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_metal_fusion_max(ଅଚଳ ggml_cgraph * gf, ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx) {
    // a view node cannot start a pattern - pack it alone
    ଯଦି (ggml_op_is_empty(gf->nodes[idx]->op)) {
        ଫେରନ୍ତୁ 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା idxs[GGML_METAL_FUSION_MAX];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_idxs = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        ଯଦି (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    ଯଦି (n_idxs == 0) {
        ଫେରନ୍ତୁ 1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା total = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i_f = 0;

    ଯେତେବେଳେ (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା len = 1;
        ଅଚଳ ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        ଯଦି (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            ଭାଙ୍ଗ;
        }

        total += len;
        i_f += len;
    }

    ଯଦି (i_f == 0) {
        ଫେରନ୍ତୁ 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    ଫେରନ୍ତୁ std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
