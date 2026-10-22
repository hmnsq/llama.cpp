#அந்தர்பாவ "ggml-metal-fusion.h"

#அந்தர்பாவ "ggml-metal-common.h"
#அந்தர்பாவ "ggml-metal-device.h"

#அந்தர்பாவ "ggml-backend-impl.h"

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <set>
#அந்தர்பாவ <string>
#அந்தர்பாவ <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
நிலையான std::vector<ggml_op> ggml_metal_fusion_filter_ops(மாறிலி std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    க்கு (ggml_op op : ops_all) {
        எனில் (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    திரும்பு ops;
}

கட்டமைப்பு ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<முழு>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    தருக்க unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    தருக்க (*check)(மாறிலி கட்டமைப்பு ggml_metal_fusion   * fusion,
                  மாறிலி கட்டமைப்பு ggml_tensor * மாறிலி * nodes,
                  மாறிலி கட்டமைப்பு ggml_cgraph         * gf,
                  மாறிலி முழு                        * முனை_idxs,
                        முழு                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            மாறிலி std::vector<ggml_op> & ops_all,
            மாறிலி std::vector<முழு> & outs,
            தருக்க unsafe,
            தருக்க (*check)(மாறிலி கட்டமைப்பு ggml_metal_fusion   * fusion,
                          மாறிலி கட்டமைப்பு ggml_tensor * மாறிலி * nodes,
                          மாறிலி கட்டமைப்பு ggml_cgraph         * gf,
                          மாறிலி முழு                        * முனை_idxs,
                                முழு                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(மாறிலி ggml_metal_fusion * fusion) {
    திரும்பு fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
நிலையான தருக்க ggml_metal_fusion_same_buffer(மாறிலி ggml_tensor * a, மாறிலி ggml_tensor * b) {
    எனில் (!a || !b) {
        திரும்பு பொய்;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    திரும்பு ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
நிலையான தருக்க ggml_metal_fusion_overlap(மாறிலி ggml_tensor * a, மாறிலி ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    மாறிலி ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    மாறிலி ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    எனில் (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        திரும்பு பொய்;
    }

    திரும்பு bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
நிலையான தருக்க ggml_metal_fusion_check_norm(
        மாறிலி ggml_metal_fusion       * fusion,
        மாறிலி ggml_tensor * மாறிலி     * nodes,
        மாறிலி ggml_cgraph             * gf,
        மாறிலி முழு                     * முனை_idxs,
              முழு                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(முனை_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    எனில் (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        மாறிலி ggml_tensor * scale = nodes[1];
        எனில் (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            திரும்பு பொய்;
        }

        திரும்பு உண்மை;
    }

    க்கு (முழு j = 1; j < (முழு) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        எனில் (nodes[j]->src[0] != nodes[j - 1]) {
            திரும்பு பொய்;
        }

        // the weight/bias must have the same row width as the norm input
        எனில் (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            திரும்பு பொய்;
        }

        எனில் (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            திரும்பு பொய்;
        }

        எனில் (nodes[j]->type != GGML_TYPE_F32) {
            திரும்பு பொய்;
        }
    }

    திரும்பு உண்மை;
}

// SSM_CONV + UNARY (silu)
நிலையான தருக்க ggml_metal_fusion_check_ssm_conv_silu(
        மாறிலி ggml_metal_fusion       * fusion,
        மாறிலி ggml_tensor * மாறிலி     * nodes,
        மாறிலி ggml_cgraph             * gf,
        மாறிலி முழு                     * முனை_idxs,
              முழு                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(முனை_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    மாறிலி ggml_tensor * conv = nodes[0];
    மாறிலி ggml_tensor * un   = nodes[1];

    எனில் (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        திரும்பு பொய்;
    }

    எனில் (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        திரும்பு பொய்;
    }

    எனில் (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        திரும்பு பொய்;
    }

    திரும்பு உண்மை;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
நிலையான தருக்க ggml_metal_fusion_check_add_chain(
        மாறிலி ggml_metal_fusion       * fusion,
        மாறிலி ggml_tensor * மாறிலி     * nodes,
        மாறிலி ggml_cgraph             * gf,
        மாறிலி முழு                     * முனை_idxs,
              முழு                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(முனை_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    க்கு (முழு j = 1; j < (முழு) fusion->ops.size(); j++) {
        எனில் (nodes[j]->src[0] != nodes[j - 1]) {
            திரும்பு பொய்;
        }

        எனில் (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            திரும்பு பொய்;
        }

        எனில் (mode == GGML_METAL_FUSION_FULL) {
            எனில் (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                திரும்பு பொய்;
            }
        }
    }

    திரும்பு உண்மை;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
நிலையான தருக்க ggml_metal_fusion_check_gdn_cache(
        மாறிலி ggml_metal_fusion       * fusion,
        மாறிலி ggml_tensor * மாறிலி     * nodes,
        மாறிலி ggml_cgraph             * gf,
        மாறிலி முழு                     * முனை_idxs,
              முழு                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(முனை_idxs);
    GGML_UNUSED(idx);

    மாறிலி ggml_tensor * gdn = nodes[0];
    மாறிலி ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    எனில் (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        திரும்பு பொய்;
    }

    எனில் (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        திரும்பு பொய்;
    }

    மாறிலி int64_t S_v      = gdn->src[2]->ne[0];
    மாறிலி int64_t H        = gdn->src[2]->ne[1];
    மாறிலி int64_t n_tokens = gdn->src[2]->ne[2];
    மாறிலி int64_t n_seqs   = gdn->src[2]->ne[3];
    மாறிலி int64_t K        = ggml_get_op_params_i32(gdn, 0);
    மாறிலி size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    மாறிலி int64_t D         = S_v * S_v * H;
    மாறிலி int64_t n_written = std::min<int64_t>(n_tokens, K);

    மாறிலி ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    மாறிலி ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    எனில் (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        திரும்பு பொய்;
    }

    மாறிலி int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    எனில் (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        திரும்பு பொய்;
    }

    எனில் (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        எனில் (dst->data == nullptr) {
            திரும்பு பொய்;
        }
    }

    திரும்பு உண்மை;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
நிலையான தருக்க ggml_metal_fusion_check_snake(
        மாறிலி ggml_metal_fusion       * fusion,
        மாறிலி ggml_tensor * மாறிலி     * nodes,
        மாறிலி ggml_cgraph             * gf,
        மாறிலி முழு                     * முனை_idxs,
              முழு                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(முனை_idxs);
    GGML_UNUSED(idx);

    மாறிலி ggml_tensor * mul0     = nodes[0];
    மாறிலி ggml_tensor * sin_முனை = nodes[1];
    மாறிலி ggml_tensor * sqr      = nodes[2];
    மாறிலி ggml_tensor * mul1     = nodes[3];
    மாறிலி ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    மாறிலி ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    மாறிலி ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    மாறிலி ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    மாறிலி ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    மாறிலி தருக்க types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_முனை->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    மாறிலி தருக்க shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    மாறிலி தருக்க dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    மாறிலி தருக்க contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    திரும்பு types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
நிலையான மாறிலி std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
நிலையான மாறிலி std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
நிலையான மாறிலி std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
நிலையான மாறிலி std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

நிலையான தருக்க ggml_metal_fusion_check_topk_moe(
        மாறிலி ggml_metal_fusion       * fusion,
        மாறிலி ggml_tensor * மாறிலி     * nodes,
        மாறிலி ggml_cgraph             * gf,
        மாறிலி முழு                     * முனை_idxs,
              முழு                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    மாறிலி முழு n_ops = (முழு) fusion->ops.size();

    மாறிலி தருக்க with_norm  = n_ops >= 6;
    மாறிலி தருக்க with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    மாறிலி std::vector<ggml_op> & ops_all = fusion->ops_all;

    மாறிலி முழு raw_start = முனை_idxs[idx];
    முழு       raw_end   = முனை_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    எனில் (with_norm && !with_scale) {
        எனில் (raw_end + 1 >= gf->n_nodes) {
            திரும்பு பொய்;
        }
        மாறிலி ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        எனில் (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            திரும்பு பொய்;
        }
        raw_end++;
    }

    மாறிலி முழு raw_மொத்தம் = raw_end - raw_start + 1;
    எனில் (raw_மொத்தம் != (முழு) ops_all.size()) {
        திரும்பு பொய்;
    }

    முழு raw_idxs[GGML_METAL_FUSION_MAX];
    க்கு (முழு i = 0; i < raw_மொத்தம்; ++i) {
        raw_idxs[i] = raw_start + i;
        எனில் (gf->nodes[raw_start + i]->op != ops_all[i]) {
            திரும்பு பொய்;
        }
    }

    மாறிலி ggml_tensor * softmax        = gf->nodes[raw_start];
    மாறிலி ggml_tensor * probs_reshaped = gf->nodes[raw_start + 1];
    மாறிலி ggml_tensor * argsort        = gf->nodes[raw_start + 2];
    மாறிலி ggml_tensor * ids            = gf->nodes[raw_start + 3];
    மாறிலி ggml_tensor * get_rows       = gf->nodes[raw_start + 4];
    மாறிலி ggml_tensor * out            = gf->nodes[raw_end];
    மாறிலி ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    தசம scale   = 1.0f;
    தசம max_bias = 0.0f;
    memcpy(&scale,    ((மாறிலி int32_t *) softmax->op_params) + 0, அளவு(scale));
    memcpy(&max_bias, ((மாறிலி int32_t *) softmax->op_params) + 1, அளவு(max_bias));
    எனில் (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        திரும்பு பொய்;
    }

    எனில் (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        திரும்பு பொய்;
    }

    மாறிலி int64_t n_expert      = logits->ne[0];
    மாறிலி int64_t n_tokens      = logits->ne[1];
    மாறிலி int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    எனில் (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        திரும்பு பொய்;
    }

    எனில் (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        திரும்பு பொய்;
    }

    எனில் (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        திரும்பு பொய்;
    }

    எனில் (with_norm) {
        மாறிலி ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
        மாறிலி ggml_tensor * sum_rows         = gf->nodes[raw_start + 6];
        மாறிலி ggml_tensor * clamp            = gf->nodes[raw_start + 7];
        மாறிலி ggml_tensor * div              = gf->nodes[raw_start + 8];
        மாறிலி ggml_tensor * out_reshaped     = gf->nodes[raw_start + 9];

        எனில் (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            clamp->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            out_reshaped->src[0] != div) {
            திரும்பு பொய்;
        }

        எனில் (with_scale) {
            மாறிலி ggml_tensor * scale_முனை = gf->nodes[raw_start + 10];
            எனில் (scale_முனை->src[0] != out_reshaped) {
                திரும்பு பொய்;
            }
        }
    } இல்லையெனில் எனில் (with_scale) {
        மாறிலி ggml_tensor * scale_முனை = gf->nodes[raw_start + 5];
        எனில் (scale_முனை->src[0] != get_rows) {
            திரும்பு பொய்;
        }
    }

    மாறிலி முழு outputs[2] = { raw_start + 3, raw_end };
    எனில் (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_மொத்தம், ops_all.data(), outputs, 2)) {
        திரும்பு பொய்;
    }

    எனில் (mode == GGML_METAL_FUSION_FULL) {
        எனில் (!logits->data || !out->data || !ids->data) {
            திரும்பு பொய்;
        }
    }

    திரும்பு உண்மை;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

கட்டமைப்பு ggml_metal_moe_reduce_match {
    மாறிலி ggml_tensor * experts;
    மாறிலி ggml_tensor * weights;
    மாறிலி ggml_tensor * dst;
    முழு முனை_மொத்தம்;
};

நிலையான தருக்க ggml_metal_fusion_match_moe_reduce(
        மாறிலி ggml_cgraph * gf, முழு முனை_idx, மாறிலி std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    எனில் (match == nullptr || முனை_idx < 0 || முனை_idx + (முழு) ops_all.size() > gf->n_nodes) {
        திரும்பு பொய்;
    }

    மாறிலி ggml_tensor * mul = gf->nodes[முனை_idx];
    எனில் (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    மாறிலி முழு raw_மொத்தம்     = (முழு) ops_all.size();
    மாறிலி முழு n_expert_used = raw_மொத்தம் / 2;

    எனில் (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_மொத்தம் != 2 * n_expert_used) {
        திரும்பு பொய்;
    }

    முழு n_views = 0;
    போது (முனை_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[முனை_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    எனில் (n_views != n_expert_used) {
        திரும்பு பொய்;
    }

    க்கு (முழு i = n_expert_used + 1; i < raw_மொத்தம்; ++i) {
        எனில் (gf->nodes[முனை_idx + i]->op != GGML_OP_ADD) {
            திரும்பு பொய்;
        }
    }

    முழு raw_idxs[GGML_METAL_FUSION_MAX];
    க்கு (முழு i = 0; i < raw_மொத்தம்; ++i) {
        raw_idxs[i] = முனை_idx + i;
        எனில் (gf->nodes[முனை_idx + i]->op != ops_all[i]) {
            திரும்பு பொய்;
        }
    }

    மாறிலி ggml_tensor * experts = mul->src[0];
    மாறிலி ggml_tensor * weights = mul->src[1];
    மாறிலி ggml_tensor * dst     = gf->nodes[முனை_idx + raw_மொத்தம் - 1];

    எனில் (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு பொய்;
    }

    மாறிலி int64_t n_embd   = experts->ne[0];
    மாறிலி int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    எனில் (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        திரும்பு பொய்;
    }

    க்கு (முழு i = 1; i <= n_expert_used; ++i) {
        மாறிலி ggml_tensor * view = gf->nodes[முனை_idx + i];
        எனில் (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            திரும்பு பொய்;
        }
    }

    மாறிலி ggml_tensor * prev_add = nullptr;
    க்கு (முழு j = 1; j < n_expert_used; ++j) {
        மாறிலி ggml_tensor * add = gf->nodes[முனை_idx + n_expert_used + j];
        மாறிலி ggml_tensor * rhs = gf->nodes[முனை_idx + j + 1];
        மாறிலி ggml_tensor * lhs = j == 1 ? gf->nodes[முனை_idx + 1] : prev_add;
        எனில் (add->src[0] != lhs || add->src[1] != rhs) {
            திரும்பு பொய்;
        }
        prev_add = add;
    }

    மாறிலி முழு outputs[1] = { முனை_idx + raw_மொத்தம் - 1 };
    எனில் (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_மொத்தம், ops_all.data(), outputs, 1)) {
        திரும்பு பொய்;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->முனை_மொத்தம் = raw_மொத்தம்;
    திரும்பு உண்மை;
}

நிலையான தருக்க ggml_metal_fusion_check_moe_reduce(
        மாறிலி ggml_metal_fusion       * fusion,
        மாறிலி ggml_tensor * மாறிலி     * nodes,
        மாறிலி ggml_cgraph             * gf,
        மாறிலி முழு                     * முனை_idxs,
              முழு                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    எனில் (!ggml_metal_fusion_match_moe_reduce(gf, முனை_idxs[idx], fusion->ops_all, &match)) {
        திரும்பு பொய்;
    }

    எனில் ((முழு) fusion->ops.size() != match.experts->ne[1]) {
        திரும்பு பொய்;
    }

    மாறிலி முழு raw_end = முனை_idxs[idx] + match.முனை_மொத்தம் - 1;
    எனில் (முனை_idxs[idx + (முழு) fusion->ops.size() - 1] != raw_end) {
        திரும்பு பொய்;
    }

    எனில் (mode == GGML_METAL_FUSION_FULL) {
        எனில் (!match.experts->data || !match.weights->data || !match.dst->data) {
            திரும்பு பொய்;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        எனில் (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            திரும்பு பொய்;
        }
    }

    திரும்பு உண்மை;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
நிலையான தருக்க ggml_metal_tensor_is_weight(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    மாறிலி ggml_tensor * base = t->view_src != வெற்று ? t->view_src : t;

    திரும்பு base->buffer != வெற்று && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

நிலையான மாறிலி கட்டமைப்பு ggml_tensor * ggml_metal_mul_mat_add_operand(மாறிலி கட்டமைப்பு ggml_tensor * mm, மாறிலி கட்டமைப்பு ggml_tensor * add) {
    எனில் (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        திரும்பு வெற்று;
    }

    மாறிலி ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    மாறிலி தருக்க ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    திரும்பு ok ? other : வெற்று;
}

நிலையான மாறிலி கட்டமைப்பு ggml_tensor * ggml_metal_mul_mat_add_residual(மாறிலி கட்டமைப்பு ggml_tensor * mm, மாறிலி கட்டமைப்பு ggml_tensor * add) {
    மாறிலி ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    மாறிலி தருக்க ok = res != வெற்று && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    திரும்பு ok ? res : வெற்று;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
நிலையான தருக்க ggml_metal_fusion_check_mul_mat_add(
        மாறிலி ggml_metal_fusion       * fusion,
        மாறிலி ggml_tensor * மாறிலி     * nodes,
        மாறிலி ggml_cgraph             * gf,
        மாறிலி முழு                     * முனை_idxs,
              முழு                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(முனை_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    மாறிலி ggml_tensor * mm  = nodes[0];
    மாறிலி ggml_tensor * add = nodes[1];

    எனில் (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        திரும்பு பொய்;
    }

    எனில் (mode == GGML_METAL_FUSION_STRUCTURAL) {
        திரும்பு உண்மை;
    }

    மாறிலி ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    எனில் (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        திரும்பு பொய்;
    }

    திரும்பு !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

நிலையான மாறிலி std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
நிலையான மாறிலி std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
நிலையான மாறிலி std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
நிலையான மாறிலி std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
நிலையான மாறிலி std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
நிலையான மாறிலி std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

நிலையான மாறிலி std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
நிலையான மாறிலி std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
நிலையான மாறிலி std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
நிலையான மாறிலி std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
நிலையான மாறிலி std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
நிலையான மாறிலி std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
நிலையான மாறிலி std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

நிலையான மாறிலி std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

நிலையான மாறிலி std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

நிலையான மாறிலி std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
நிலையான மாறிலி std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
நிலையான மாறிலி std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
நிலையான மாறிலி std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
நிலையான மாறிலி std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
நிலையான மாறிலி std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
நிலையான மாறிலி std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

நிலையான மாறிலி std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };

நிலையான மாறிலி std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     பொய், ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_norm_mul_add,           {},     பொய், ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     பொய், ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     பொய், ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_rms_norm_mul_add,       {},     பொய், ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     பொய், ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_2,                  {},     பொய், ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_3,                  {},     பொய், ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_4,                  {},     பொய், ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_5,                  {},     பொய், ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_6,                  {},     பொய், ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_7,                  {},     பொய், ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     பொய், ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     உண்மை,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    உண்மை,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    உண்மை,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    உண்மை,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    உண்மை,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     உண்மை,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     உண்மை,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     உண்மை,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     உண்மை,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     உண்மை,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     உண்மை,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     உண்மை,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     பொய், ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_ADD,    ops_mul_mat_add,            {},     பொய், ggml_metal_fusion_check_mul_mat_add },
};

// ---- alloc deps -----------------------------------------------------------

நிலையான தருக்க ggml_metal_fusion_match_raw_pattern(
        மாறிலி ggml_cgraph * gf, முழு முனை_idx, மாறிலி std::vector<ggml_op> & ops) {
    எனில் (முனை_idx < 0 || முனை_idx + (முழு) ops.size() > gf->n_nodes) {
        திரும்பு பொய்;
    }

    க்கு (முழு i = 0; i < (முழு) ops.size(); ++i) {
        எனில் (gf->nodes[முனை_idx + i]->op != ops[i]) {
            திரும்பு பொய்;
        }
    }

    திரும்பு உண்மை;
}

நிலையான வெறுமை ggml_metal_fusion_add_pattern_alloc_deps(
        வெறுமை * user_data,
        வெறுமை (*add_alloc_dep)(வெறுமை *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        மாறிலி ggml_metal_fusion * fusion,
        முழு முனை_idx) {
    மாறிலி முழு last_முனை = முனை_idx + (முழு) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    க்கு (முழு j = 0; j < (முழு) fusion->ops_all.size(); ++j) {
        ggml_tensor * முனை = gf->nodes[முனை_idx + j];
        க்கு (முழு s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = முனை->src[s];
            எனில் (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_முனை]);
            }
        }
        seen.insert(முனை);
    }
}

வெறுமை ggml_metal_fusion_add_alloc_deps(
        வெறுமை * user_data,
        வெறுமை (*add_alloc_dep)(வெறுமை *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    க்கு (முழு i = 0; i < gf->n_nodes; ++i) {
        மாறிலி ggml_metal_fusion * best = nullptr;
        முழு best_raw = 0;

        க்கு (மாறிலி ggml_metal_fusion & fusion : ggml_metal_fusions) {
            எனில் ((முழு) fusion.ops_all.size() <= best_raw) {
                தொடர்;
            }
            எனில் (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (முழு) fusion.ops_all.size();
            }
        }

        எனில் (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

நிலையான std::string ggml_metal_fusion_label(மாறிலி ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    க்கு (முழு j = 0; j < (முழு) fusion->ops.size(); j++) {
        எனில் (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    திரும்பு label;
}

கட்டமைப்பு ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    தருக்க enabled;
    தருக்க stats;
    தருக்க labels_set;
    முழு  debug;
};

கட்டமைப்பு ggml_metal_fusion_info * ggml_metal_fusion_info_init(தருக்க enabled, முழு debug) {
    ggml_metal_fusion_info * finfo = புதிய ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = பொய்;
    finfo->debug      = debug;

    எனில் (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    திரும்பு finfo;
}

வெறுமை ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

தருக்க ggml_metal_fusion_info_enabled(மாறிலி ggml_metal_fusion_info * finfo) {
    திரும்பு finfo->enabled;
}

தருக்க ggml_metal_fusion_info_stats(மாறிலி ggml_metal_fusion_info * finfo) {
    திரும்பு finfo->stats;
}

முழு ggml_metal_fusion_info_debug(மாறிலி ggml_metal_fusion_info * finfo) {
    திரும்பு finfo->debug;
}

முழு ggml_metal_fusion_info_n_fusions(மாறிலி ggml_metal_fusion_info * finfo) {
    திரும்பு (முழு) finfo->labels.size();
}

மாறிலி எழுத்து * ggml_metal_fusion_info_label(மாறிலி ggml_metal_fusion_info * finfo, முழு idx) {
    GGML_ASSERT(idx >= 0 && idx < (முழு) finfo->labels.size());
    திரும்பு finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_மொத்தம்(மாறிலி ggml_metal_fusion_info * finfo, முழு idx) {
    GGML_ASSERT(idx >= 0 && idx < (முழு) finfo->counts.size());
    திரும்பு finfo->counts[idx];
}

வெறுமை ggml_metal_fusion_info_மொத்தம்_fusion(ggml_metal_fusion_info * finfo, மாறிலி ggml_metal_fusion * fusion) {
    எனில் (!finfo->stats || fusion == nullptr) {
        திரும்பு;
    }

    மாறிலி ptrdiff_t idx = fusion - ggml_metal_fusions.data();
    எனில் (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

வெறுமை ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, தருக்க enabled) {
    finfo->enabled = enabled;
}

வெறுமை ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    எனில் (finfo->labels_set) {
        திரும்பு;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    க்கு (மாறிலி ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = உண்மை;
}

வெறுமை ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = உண்மை;
    ggml_metal_fusion_info_labels_init(finfo);
}

வெறுமை ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

முழு ggml_metal_fusion_info_stats_get(மாறிலி ggml_metal_fusion_info * finfo, மாறிலி எழுத்து ** labels, uint64_t * counts, முழு n) {
    மாறிலி முழு n_fusions = (முழு) finfo->labels.size();

    எனில் (labels == nullptr) {
        திரும்பு n_fusions;
    }

    மாறிலி முழு n_fill = std::min(n, n_fusions);
    க்கு (முழு i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        எனில் (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    திரும்பு n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
நிலையான தருக்க ggml_metal_fusion_check_memory_ranges(
        மாறிலி ggml_metal_fusion      * fusion,
        மாறிலி ggml_tensor * மாறிலி    * nodes,
        முழு                            முனை_மொத்தம்) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](மாறிலி ggml_tensor * a, மாறிலி ggml_tensor * b) {
        எனில் (a->data && b->data && a->data == b->data) {
            திரும்பு உண்மை;
        }
        க்கு (மாறிலி ggml_tensor * v = a; v; v = v->view_src) {
            எனில் (v == b) {
                திரும்பு உண்மை;
            }
        }
        க்கு (மாறிலி ggml_tensor * v = b; v; v = v->view_src) {
            எனில் (v == a) {
                திரும்பு உண்மை;
            }
        }
        திரும்பு பொய்;
    };

    auto nodes_overlap = [](மாறிலி ggml_tensor * a, மாறிலி ggml_tensor * b) {
        எனில் (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            திரும்பு பொய்;
        }

        எனில் (a->buffer != b->buffer) {
            திரும்பு பொய்;
        }

        மாறிலி int64_t a_start = (int64_t) a->data;
        மாறிலி int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        மாறிலி int64_t b_start = (int64_t) b->data;
        மாறிலி int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        திரும்பு (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](மாறிலி ggml_tensor * src, மாறிலி ggml_tensor * மாறிலி * nodes, முழு j) {
        க்கு (முழு k = 0; k < j; ++k) {
            எனில் (src == nodes[k]) {
                திரும்பு உண்மை;
            }
            க்கு (மாறிலி ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                எனில் (view_src == nodes[k]) {
                    திரும்பு உண்மை;
                }
            }
        }
        திரும்பு பொய்;
    };

    auto check_dst = [&](மாறிலி ggml_tensor * dst) {
        க்கு (முழு j = 0; j < முனை_மொத்தம்; ++j) {
            க்கு (முழு s = 0; s < GGML_MAX_SRC; ++s) {
                மாறிலி ggml_tensor * src = nodes[j]->src[s];
                எனில் (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    தொடர்;
                }

                எனில் (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    திரும்பு பொய்;
                }
            }
        }
        திரும்பு உண்மை;
    };

    எனில் (!check_dst(nodes[முனை_மொத்தம் - 1])) {
        திரும்பு பொய்;
    }

    க்கு (முழு offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < முனை_மொத்தம்);
        எனில் (!check_dst(nodes[offset])) {
            திரும்பு பொய்;
        }
    }

    திரும்பு உண்மை;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
மாறிலி ggml_metal_fusion * ggml_metal_fusion_next(
        மாறிலி ggml_cgraph * gf,
        மாறிலி முழு * முனை_idxs,
        முழு n_idxs,
        முழு idx,
        ggml_metal_fusion_mode mode,
        முழு * n_out) {
    மாறிலி ggml_metal_fusion * res = nullptr;
    முழு best = 1;

    க்கு (மாறிலி ggml_metal_fusion & fusion : ggml_metal_fusions) {
        மாறிலி முழு n_ops = (முழு) fusion.ops.size();

        // only look for a longer match than the current best
        எனில் (n_ops <= best) {
            தொடர்;
        }
        எனில் (idx + n_ops > n_idxs) {
            தொடர்;
        }

        மாறிலி ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        தருக்க ok = உண்மை;
        க்கு (முழு j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[முனை_idxs[idx + j]];
            எனில் (nodes[j]->op != fusion.ops[j]) {
                ok = பொய்;
                நிறுத்து;
            }
        }
        எனில் (!ok) {
            தொடர்;
        }

        எனில் (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            க்கு (முழு j = 1; j < n_ops && ok; j++) {
                எனில் (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = பொய்;
                    நிறுத்து;
                }
                எனில் (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = பொய்;
                    நிறுத்து;
                }
            }
            எனில் (!ok) {
                தொடர்;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            முழு outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = முனை_idxs[idx + n_ops - 1];
            க்கு (size_t i = 0; i < fusion.outs.size(); ++i) {
                மாறிலி முழு out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = முனை_idxs[idx + out_offset];
            }

            மாறிலி முழு n_outputs = 1 + (முழு) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            எனில் (!ggml_can_fuse_subgraph_ext(gf, முனை_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                தொடர்;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        எனில் (fusion.check && !fusion.check(&fusion, nodes, gf, முனை_idxs, idx, mode)) {
            தொடர்;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        எனில் (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            தொடர்;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    திரும்பு res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
முழு ggml_metal_fusion_max(மாறிலி ggml_cgraph * gf, முழு idx) {
    // a view node cannot start a pattern - pack it alone
    எனில் (ggml_op_is_empty(gf->nodes[idx]->op)) {
        திரும்பு 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    முழு idxs[GGML_METAL_FUSION_MAX];
    முழு n_idxs = 0;
    க்கு (முழு i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        எனில் (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    எனில் (n_idxs == 0) {
        திரும்பு 1;
    }

    முழு total = 0;
    முழு i_f = 0;

    போது (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        முழு len = 1;
        மாறிலி ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        எனில் (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            நிறுத்து;
        }

        total += len;
        i_f += len;
    }

    எனில் (i_f == 0) {
        திரும்பு 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    திரும்பு std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
