#अन्तर्भाव "ggml-metal-fusion.h"

#अन्तर्भाव "ggml-metal-common.h"
#अन्तर्भाव "ggml-metal-device.h"

#अन्तर्भाव "ggml-backend-impl.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstring>
#अन्तर्भाव <set>
#अन्तर्भाव <string>
#अन्तर्भाव <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
स्थिर std::vector<ggml_op> ggml_metal_fusion_filter_ops(स्थिरांक std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    साठी (ggml_op op : ops_all) {
        जर (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    परत ops;
}

संरचना ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<पूर्णांक>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    तार्किक unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    तार्किक (*check)(स्थिरांक संरचना ggml_metal_fusion   * fusion,
                  स्थिरांक संरचना ggml_tensor * स्थिरांक * nodes,
                  स्थिरांक संरचना ggml_cgraph         * gf,
                  स्थिरांक पूर्णांक                        * गाठ_idxs,
                        पूर्णांक                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            स्थिरांक std::vector<ggml_op> & ops_all,
            स्थिरांक std::vector<पूर्णांक> & outs,
            तार्किक unsafe,
            तार्किक (*check)(स्थिरांक संरचना ggml_metal_fusion   * fusion,
                          स्थिरांक संरचना ggml_tensor * स्थिरांक * nodes,
                          स्थिरांक संरचना ggml_cgraph         * gf,
                          स्थिरांक पूर्णांक                        * गाठ_idxs,
                                पूर्णांक                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(स्थिरांक ggml_metal_fusion * fusion) {
    परत fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
स्थिर तार्किक ggml_metal_fusion_same_buffer(स्थिरांक ggml_tensor * a, स्थिरांक ggml_tensor * b) {
    जर (!a || !b) {
        परत असत्य;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    परत ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
स्थिर तार्किक ggml_metal_fusion_overlap(स्थिरांक ggml_tensor * a, स्थिरांक ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    स्थिरांक ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    स्थिरांक ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    जर (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        परत असत्य;
    }

    परत bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
स्थिर तार्किक ggml_metal_fusion_check_norm(
        स्थिरांक ggml_metal_fusion       * fusion,
        स्थिरांक ggml_tensor * स्थिरांक     * nodes,
        स्थिरांक ggml_cgraph             * gf,
        स्थिरांक पूर्णांक                     * गाठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(गाठ_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    जर (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        स्थिरांक ggml_tensor * scale = nodes[1];
        जर (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            परत असत्य;
        }

        परत सत्य;
    }

    साठी (पूर्णांक j = 1; j < (पूर्णांक) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        जर (nodes[j]->src[0] != nodes[j - 1]) {
            परत असत्य;
        }

        // the weight/bias must have the same row width as the norm input
        जर (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            परत असत्य;
        }

        जर (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            परत असत्य;
        }

        जर (nodes[j]->type != GGML_TYPE_F32) {
            परत असत्य;
        }
    }

    परत सत्य;
}

// SSM_CONV + UNARY (silu)
स्थिर तार्किक ggml_metal_fusion_check_ssm_conv_silu(
        स्थिरांक ggml_metal_fusion       * fusion,
        स्थिरांक ggml_tensor * स्थिरांक     * nodes,
        स्थिरांक ggml_cgraph             * gf,
        स्थिरांक पूर्णांक                     * गाठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(गाठ_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    स्थिरांक ggml_tensor * conv = nodes[0];
    स्थिरांक ggml_tensor * un   = nodes[1];

    जर (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        परत असत्य;
    }

    जर (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        परत असत्य;
    }

    जर (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        परत असत्य;
    }

    परत सत्य;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
स्थिर तार्किक ggml_metal_fusion_check_add_chain(
        स्थिरांक ggml_metal_fusion       * fusion,
        स्थिरांक ggml_tensor * स्थिरांक     * nodes,
        स्थिरांक ggml_cgraph             * gf,
        स्थिरांक पूर्णांक                     * गाठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(गाठ_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    साठी (पूर्णांक j = 1; j < (पूर्णांक) fusion->ops.size(); j++) {
        जर (nodes[j]->src[0] != nodes[j - 1]) {
            परत असत्य;
        }

        जर (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            परत असत्य;
        }

        जर (mode == GGML_METAL_FUSION_FULL) {
            जर (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                परत असत्य;
            }
        }
    }

    परत सत्य;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
स्थिर तार्किक ggml_metal_fusion_check_gdn_cache(
        स्थिरांक ggml_metal_fusion       * fusion,
        स्थिरांक ggml_tensor * स्थिरांक     * nodes,
        स्थिरांक ggml_cgraph             * gf,
        स्थिरांक पूर्णांक                     * गाठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(गाठ_idxs);
    GGML_UNUSED(idx);

    स्थिरांक ggml_tensor * gdn = nodes[0];
    स्थिरांक ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    जर (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        परत असत्य;
    }

    जर (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        परत असत्य;
    }

    स्थिरांक int64_t S_v      = gdn->src[2]->ne[0];
    स्थिरांक int64_t H        = gdn->src[2]->ne[1];
    स्थिरांक int64_t n_tokens = gdn->src[2]->ne[2];
    स्थिरांक int64_t n_seqs   = gdn->src[2]->ne[3];
    स्थिरांक int64_t K        = ggml_get_op_params_i32(gdn, 0);
    स्थिरांक माप_प्रकार  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    स्थिरांक int64_t D         = S_v * S_v * H;
    स्थिरांक int64_t n_written = std::min<int64_t>(n_tokens, K);

    स्थिरांक ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    स्थिरांक ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    जर (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        परत असत्य;
    }

    स्थिरांक int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    जर (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        परत असत्य;
    }

    जर (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        जर (dst->data == nullptr) {
            परत असत्य;
        }
    }

    परत सत्य;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
स्थिर तार्किक ggml_metal_fusion_check_snake(
        स्थिरांक ggml_metal_fusion       * fusion,
        स्थिरांक ggml_tensor * स्थिरांक     * nodes,
        स्थिरांक ggml_cgraph             * gf,
        स्थिरांक पूर्णांक                     * गाठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(गाठ_idxs);
    GGML_UNUSED(idx);

    स्थिरांक ggml_tensor * mul0     = nodes[0];
    स्थिरांक ggml_tensor * sin_गाठ = nodes[1];
    स्थिरांक ggml_tensor * sqr      = nodes[2];
    स्थिरांक ggml_tensor * mul1     = nodes[3];
    स्थिरांक ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    स्थिरांक ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    स्थिरांक ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    स्थिरांक ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    स्थिरांक ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    स्थिरांक तार्किक types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_गाठ->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    स्थिरांक तार्किक shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    स्थिरांक तार्किक dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    स्थिरांक तार्किक contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    परत types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
स्थिर स्थिरांक std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
स्थिर स्थिरांक std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
स्थिर स्थिरांक std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
स्थिर स्थिरांक std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

स्थिर तार्किक ggml_metal_fusion_check_topk_moe(
        स्थिरांक ggml_metal_fusion       * fusion,
        स्थिरांक ggml_tensor * स्थिरांक     * nodes,
        स्थिरांक ggml_cgraph             * gf,
        स्थिरांक पूर्णांक                     * गाठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    स्थिरांक पूर्णांक n_ops = (पूर्णांक) fusion->ops.size();

    स्थिरांक तार्किक with_norm  = n_ops >= 6;
    स्थिरांक तार्किक with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    स्थिरांक std::vector<ggml_op> & ops_all = fusion->ops_all;

    स्थिरांक पूर्णांक raw_start = गाठ_idxs[idx];
    पूर्णांक       raw_end   = गाठ_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    जर (with_norm && !with_scale) {
        जर (raw_end + 1 >= gf->n_nodes) {
            परत असत्य;
        }
        स्थिरांक ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        जर (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            परत असत्य;
        }
        raw_end++;
    }

    स्थिरांक पूर्णांक raw_मोजणी = raw_end - raw_start + 1;
    जर (raw_मोजणी != (पूर्णांक) ops_all.size()) {
        परत असत्य;
    }

    पूर्णांक raw_idxs[GGML_METAL_FUSION_MAX];
    साठी (पूर्णांक i = 0; i < raw_मोजणी; ++i) {
        raw_idxs[i] = raw_start + i;
        जर (gf->nodes[raw_start + i]->op != ops_all[i]) {
            परत असत्य;
        }
    }

    स्थिरांक ggml_tensor * softmax        = gf->nodes[raw_start];
    स्थिरांक ggml_tensor * probs_reshaped = gf->nodes[raw_start + 1];
    स्थिरांक ggml_tensor * argsort        = gf->nodes[raw_start + 2];
    स्थिरांक ggml_tensor * ids            = gf->nodes[raw_start + 3];
    स्थिरांक ggml_tensor * get_rows       = gf->nodes[raw_start + 4];
    स्थिरांक ggml_tensor * out            = gf->nodes[raw_end];
    स्थिरांक ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    दशांश scale   = 1.0f;
    दशांश max_bias = 0.0f;
    स_नकल(&scale,    ((स्थिरांक int32_t *) softmax->op_params) + 0, आकार(scale));
    स_नकल(&max_bias, ((स्थिरांक int32_t *) softmax->op_params) + 1, आकार(max_bias));
    जर (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        परत असत्य;
    }

    जर (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        परत असत्य;
    }

    स्थिरांक int64_t n_expert      = logits->ne[0];
    स्थिरांक int64_t n_tokens      = logits->ne[1];
    स्थिरांक int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    जर (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        परत असत्य;
    }

    जर (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        परत असत्य;
    }

    जर (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        परत असत्य;
    }

    जर (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        परत असत्य;
    }

    जर (with_norm) {
        स्थिरांक ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
        स्थिरांक ggml_tensor * sum_rows         = gf->nodes[raw_start + 6];
        स्थिरांक ggml_tensor * clamp            = gf->nodes[raw_start + 7];
        स्थिरांक ggml_tensor * भाग              = gf->nodes[raw_start + 8];
        स्थिरांक ggml_tensor * out_reshaped     = gf->nodes[raw_start + 9];

        जर (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            clamp->src[0] != sum_rows || भाग->src[0] != weights_reshaped || भाग->src[1] != clamp ||
            out_reshaped->src[0] != भाग) {
            परत असत्य;
        }

        जर (with_scale) {
            स्थिरांक ggml_tensor * scale_गाठ = gf->nodes[raw_start + 10];
            जर (scale_गाठ->src[0] != out_reshaped) {
                परत असत्य;
            }
        }
    } अन्यथा जर (with_scale) {
        स्थिरांक ggml_tensor * scale_गाठ = gf->nodes[raw_start + 5];
        जर (scale_गाठ->src[0] != get_rows) {
            परत असत्य;
        }
    }

    स्थिरांक पूर्णांक outputs[2] = { raw_start + 3, raw_end };
    जर (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_मोजणी, ops_all.data(), outputs, 2)) {
        परत असत्य;
    }

    जर (mode == GGML_METAL_FUSION_FULL) {
        जर (!logits->data || !out->data || !ids->data) {
            परत असत्य;
        }
    }

    परत सत्य;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

संरचना ggml_metal_moe_reduce_match {
    स्थिरांक ggml_tensor * experts;
    स्थिरांक ggml_tensor * weights;
    स्थिरांक ggml_tensor * dst;
    पूर्णांक गाठ_मोजणी;
};

स्थिर तार्किक ggml_metal_fusion_match_moe_reduce(
        स्थिरांक ggml_cgraph * gf, पूर्णांक गाठ_idx, स्थिरांक std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    जर (match == nullptr || गाठ_idx < 0 || गाठ_idx + (पूर्णांक) ops_all.size() > gf->n_nodes) {
        परत असत्य;
    }

    स्थिरांक ggml_tensor * mul = gf->nodes[गाठ_idx];
    जर (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    स्थिरांक पूर्णांक raw_मोजणी     = (पूर्णांक) ops_all.size();
    स्थिरांक पूर्णांक n_expert_used = raw_मोजणी / 2;

    जर (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_मोजणी != 2 * n_expert_used) {
        परत असत्य;
    }

    पूर्णांक n_views = 0;
    पर्यंत (गाठ_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[गाठ_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    जर (n_views != n_expert_used) {
        परत असत्य;
    }

    साठी (पूर्णांक i = n_expert_used + 1; i < raw_मोजणी; ++i) {
        जर (gf->nodes[गाठ_idx + i]->op != GGML_OP_ADD) {
            परत असत्य;
        }
    }

    पूर्णांक raw_idxs[GGML_METAL_FUSION_MAX];
    साठी (पूर्णांक i = 0; i < raw_मोजणी; ++i) {
        raw_idxs[i] = गाठ_idx + i;
        जर (gf->nodes[गाठ_idx + i]->op != ops_all[i]) {
            परत असत्य;
        }
    }

    स्थिरांक ggml_tensor * experts = mul->src[0];
    स्थिरांक ggml_tensor * weights = mul->src[1];
    स्थिरांक ggml_tensor * dst     = gf->nodes[गाठ_idx + raw_मोजणी - 1];

    जर (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत असत्य;
    }

    स्थिरांक int64_t n_embd   = experts->ne[0];
    स्थिरांक int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    जर (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        परत असत्य;
    }

    जर (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        परत असत्य;
    }

    साठी (पूर्णांक i = 1; i <= n_expert_used; ++i) {
        स्थिरांक ggml_tensor * view = gf->nodes[गाठ_idx + i];
        जर (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (माप_प्रकार) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            परत असत्य;
        }
    }

    स्थिरांक ggml_tensor * prev_add = nullptr;
    साठी (पूर्णांक j = 1; j < n_expert_used; ++j) {
        स्थिरांक ggml_tensor * add = gf->nodes[गाठ_idx + n_expert_used + j];
        स्थिरांक ggml_tensor * rhs = gf->nodes[गाठ_idx + j + 1];
        स्थिरांक ggml_tensor * lhs = j == 1 ? gf->nodes[गाठ_idx + 1] : prev_add;
        जर (add->src[0] != lhs || add->src[1] != rhs) {
            परत असत्य;
        }
        prev_add = add;
    }

    स्थिरांक पूर्णांक outputs[1] = { गाठ_idx + raw_मोजणी - 1 };
    जर (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_मोजणी, ops_all.data(), outputs, 1)) {
        परत असत्य;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->गाठ_मोजणी = raw_मोजणी;
    परत सत्य;
}

स्थिर तार्किक ggml_metal_fusion_check_moe_reduce(
        स्थिरांक ggml_metal_fusion       * fusion,
        स्थिरांक ggml_tensor * स्थिरांक     * nodes,
        स्थिरांक ggml_cgraph             * gf,
        स्थिरांक पूर्णांक                     * गाठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    जर (!ggml_metal_fusion_match_moe_reduce(gf, गाठ_idxs[idx], fusion->ops_all, &match)) {
        परत असत्य;
    }

    जर ((पूर्णांक) fusion->ops.size() != match.experts->ne[1]) {
        परत असत्य;
    }

    स्थिरांक पूर्णांक raw_end = गाठ_idxs[idx] + match.गाठ_मोजणी - 1;
    जर (गाठ_idxs[idx + (पूर्णांक) fusion->ops.size() - 1] != raw_end) {
        परत असत्य;
    }

    जर (mode == GGML_METAL_FUSION_FULL) {
        जर (!match.experts->data || !match.weights->data || !match.dst->data) {
            परत असत्य;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        जर (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            परत असत्य;
        }
    }

    परत सत्य;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
स्थिर तार्किक ggml_metal_tensor_is_weight(स्थिरांक संरचना ggml_tensor * t) {
    स्थिरांक ggml_tensor * base = t->view_src != रिक्त ? t->view_src : t;

    परत base->buffer != रिक्त && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

स्थिर स्थिरांक संरचना ggml_tensor * ggml_metal_mul_mat_add_operand(स्थिरांक संरचना ggml_tensor * mm, स्थिरांक संरचना ggml_tensor * add) {
    जर (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        परत रिक्त;
    }

    स्थिरांक ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    स्थिरांक तार्किक ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    परत ok ? other : रिक्त;
}

स्थिर स्थिरांक संरचना ggml_tensor * ggml_metal_mul_mat_add_residual(स्थिरांक संरचना ggml_tensor * mm, स्थिरांक संरचना ggml_tensor * add) {
    स्थिरांक ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    स्थिरांक तार्किक ok = res != रिक्त && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    परत ok ? res : रिक्त;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
स्थिर तार्किक ggml_metal_fusion_check_mul_mat_add(
        स्थिरांक ggml_metal_fusion       * fusion,
        स्थिरांक ggml_tensor * स्थिरांक     * nodes,
        स्थिरांक ggml_cgraph             * gf,
        स्थिरांक पूर्णांक                     * गाठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(गाठ_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    स्थिरांक ggml_tensor * mm  = nodes[0];
    स्थिरांक ggml_tensor * add = nodes[1];

    जर (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        परत असत्य;
    }

    जर (mode == GGML_METAL_FUSION_STRUCTURAL) {
        परत सत्य;
    }

    स्थिरांक ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    जर (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        परत असत्य;
    }

    परत !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

स्थिर स्थिरांक std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
स्थिर स्थिरांक std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
स्थिर स्थिरांक std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
स्थिर स्थिरांक std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
स्थिर स्थिरांक std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
स्थिर स्थिरांक std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

स्थिर स्थिरांक std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
स्थिर स्थिरांक std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
स्थिर स्थिरांक std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
स्थिर स्थिरांक std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
स्थिर स्थिरांक std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
स्थिर स्थिरांक std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
स्थिर स्थिरांक std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

स्थिर स्थिरांक std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

स्थिर स्थिरांक std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

स्थिर स्थिरांक std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
स्थिर स्थिरांक std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
स्थिर स्थिरांक std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
स्थिर स्थिरांक std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
स्थिर स्थिरांक std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
स्थिर स्थिरांक std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
स्थिर स्थिरांक std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

स्थिर स्थिरांक std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };

स्थिर स्थिरांक std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     असत्य, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_norm_mul_add,           {},     असत्य, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     असत्य, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     असत्य, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_rms_norm_mul_add,       {},     असत्य, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     असत्य, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_2,                  {},     असत्य, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_3,                  {},     असत्य, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_4,                  {},     असत्य, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_5,                  {},     असत्य, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_6,                  {},     असत्य, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_7,                  {},     असत्य, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     असत्य, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     सत्य,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    सत्य,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    सत्य,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    सत्य,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    सत्य,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     सत्य,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     सत्य,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     सत्य,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     सत्य,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     सत्य,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     सत्य,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     सत्य,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     असत्य, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_ADD,    ops_mul_mat_add,            {},     असत्य, ggml_metal_fusion_check_mul_mat_add },
};

// ---- alloc deps -----------------------------------------------------------

स्थिर तार्किक ggml_metal_fusion_match_raw_pattern(
        स्थिरांक ggml_cgraph * gf, पूर्णांक गाठ_idx, स्थिरांक std::vector<ggml_op> & ops) {
    जर (गाठ_idx < 0 || गाठ_idx + (पूर्णांक) ops.size() > gf->n_nodes) {
        परत असत्य;
    }

    साठी (पूर्णांक i = 0; i < (पूर्णांक) ops.size(); ++i) {
        जर (gf->nodes[गाठ_idx + i]->op != ops[i]) {
            परत असत्य;
        }
    }

    परत सत्य;
}

स्थिर शून्य ggml_metal_fusion_add_pattern_alloc_deps(
        शून्य * user_data,
        शून्य (*add_alloc_dep)(शून्य *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        स्थिरांक ggml_metal_fusion * fusion,
        पूर्णांक गाठ_idx) {
    स्थिरांक पूर्णांक last_गाठ = गाठ_idx + (पूर्णांक) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    साठी (पूर्णांक j = 0; j < (पूर्णांक) fusion->ops_all.size(); ++j) {
        ggml_tensor * गाठ = gf->nodes[गाठ_idx + j];
        साठी (पूर्णांक s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = गाठ->src[s];
            जर (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_गाठ]);
            }
        }
        seen.insert(गाठ);
    }
}

शून्य ggml_metal_fusion_add_alloc_deps(
        शून्य * user_data,
        शून्य (*add_alloc_dep)(शून्य *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    साठी (पूर्णांक i = 0; i < gf->n_nodes; ++i) {
        स्थिरांक ggml_metal_fusion * best = nullptr;
        पूर्णांक best_raw = 0;

        साठी (स्थिरांक ggml_metal_fusion & fusion : ggml_metal_fusions) {
            जर ((पूर्णांक) fusion.ops_all.size() <= best_raw) {
                सुरूठेवा;
            }
            जर (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (पूर्णांक) fusion.ops_all.size();
            }
        }

        जर (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

स्थिर std::string ggml_metal_fusion_label(स्थिरांक ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    साठी (पूर्णांक j = 0; j < (पूर्णांक) fusion->ops.size(); j++) {
        जर (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    परत label;
}

संरचना ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    तार्किक enabled;
    तार्किक stats;
    तार्किक labels_set;
    पूर्णांक  debug;
};

संरचना ggml_metal_fusion_info * ggml_metal_fusion_info_init(तार्किक enabled, पूर्णांक debug) {
    ggml_metal_fusion_info * finfo = नवीन ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = असत्य;
    finfo->debug      = debug;

    जर (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    परत finfo;
}

शून्य ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

तार्किक ggml_metal_fusion_info_enabled(स्थिरांक ggml_metal_fusion_info * finfo) {
    परत finfo->enabled;
}

तार्किक ggml_metal_fusion_info_stats(स्थिरांक ggml_metal_fusion_info * finfo) {
    परत finfo->stats;
}

पूर्णांक ggml_metal_fusion_info_debug(स्थिरांक ggml_metal_fusion_info * finfo) {
    परत finfo->debug;
}

पूर्णांक ggml_metal_fusion_info_n_fusions(स्थिरांक ggml_metal_fusion_info * finfo) {
    परत (पूर्णांक) finfo->labels.size();
}

स्थिरांक अक्षर * ggml_metal_fusion_info_label(स्थिरांक ggml_metal_fusion_info * finfo, पूर्णांक idx) {
    GGML_ASSERT(idx >= 0 && idx < (पूर्णांक) finfo->labels.size());
    परत finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_मोजणी(स्थिरांक ggml_metal_fusion_info * finfo, पूर्णांक idx) {
    GGML_ASSERT(idx >= 0 && idx < (पूर्णांक) finfo->counts.size());
    परत finfo->counts[idx];
}

शून्य ggml_metal_fusion_info_मोजणी_fusion(ggml_metal_fusion_info * finfo, स्थिरांक ggml_metal_fusion * fusion) {
    जर (!finfo->stats || fusion == nullptr) {
        परत;
    }

    स्थिरांक सूचक_भेद_प्रकार idx = fusion - ggml_metal_fusions.data();
    जर (idx >= 0 && idx < (सूचक_भेद_प्रकार) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

शून्य ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, तार्किक enabled) {
    finfo->enabled = enabled;
}

शून्य ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    जर (finfo->labels_set) {
        परत;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    साठी (स्थिरांक ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = सत्य;
}

शून्य ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = सत्य;
    ggml_metal_fusion_info_labels_init(finfo);
}

शून्य ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

पूर्णांक ggml_metal_fusion_info_stats_get(स्थिरांक ggml_metal_fusion_info * finfo, स्थिरांक अक्षर ** labels, uint64_t * counts, पूर्णांक n) {
    स्थिरांक पूर्णांक n_fusions = (पूर्णांक) finfo->labels.size();

    जर (labels == nullptr) {
        परत n_fusions;
    }

    स्थिरांक पूर्णांक n_fill = std::min(n, n_fusions);
    साठी (पूर्णांक i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        जर (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    परत n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
स्थिर तार्किक ggml_metal_fusion_check_memory_ranges(
        स्थिरांक ggml_metal_fusion      * fusion,
        स्थिरांक ggml_tensor * स्थिरांक    * nodes,
        पूर्णांक                            गाठ_मोजणी) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    स्वतः same_memory = [](स्थिरांक ggml_tensor * a, स्थिरांक ggml_tensor * b) {
        जर (a->data && b->data && a->data == b->data) {
            परत सत्य;
        }
        साठी (स्थिरांक ggml_tensor * v = a; v; v = v->view_src) {
            जर (v == b) {
                परत सत्य;
            }
        }
        साठी (स्थिरांक ggml_tensor * v = b; v; v = v->view_src) {
            जर (v == a) {
                परत सत्य;
            }
        }
        परत असत्य;
    };

    स्वतः nodes_overlap = [](स्थिरांक ggml_tensor * a, स्थिरांक ggml_tensor * b) {
        जर (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            परत असत्य;
        }

        जर (a->buffer != b->buffer) {
            परत असत्य;
        }

        स्थिरांक int64_t a_start = (int64_t) a->data;
        स्थिरांक int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        स्थिरांक int64_t b_start = (int64_t) b->data;
        स्थिरांक int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        परत (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    स्वतः is_intermediate = [](स्थिरांक ggml_tensor * src, स्थिरांक ggml_tensor * स्थिरांक * nodes, पूर्णांक j) {
        साठी (पूर्णांक k = 0; k < j; ++k) {
            जर (src == nodes[k]) {
                परत सत्य;
            }
            साठी (स्थिरांक ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                जर (view_src == nodes[k]) {
                    परत सत्य;
                }
            }
        }
        परत असत्य;
    };

    स्वतः check_dst = [&](स्थिरांक ggml_tensor * dst) {
        साठी (पूर्णांक j = 0; j < गाठ_मोजणी; ++j) {
            साठी (पूर्णांक s = 0; s < GGML_MAX_SRC; ++s) {
                स्थिरांक ggml_tensor * src = nodes[j]->src[s];
                जर (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    सुरूठेवा;
                }

                जर (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    परत असत्य;
                }
            }
        }
        परत सत्य;
    };

    जर (!check_dst(nodes[गाठ_मोजणी - 1])) {
        परत असत्य;
    }

    साठी (पूर्णांक offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < गाठ_मोजणी);
        जर (!check_dst(nodes[offset])) {
            परत असत्य;
        }
    }

    परत सत्य;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
स्थिरांक ggml_metal_fusion * ggml_metal_fusion_next(
        स्थिरांक ggml_cgraph * gf,
        स्थिरांक पूर्णांक * गाठ_idxs,
        पूर्णांक n_idxs,
        पूर्णांक idx,
        ggml_metal_fusion_mode mode,
        पूर्णांक * n_out) {
    स्थिरांक ggml_metal_fusion * res = nullptr;
    पूर्णांक best = 1;

    साठी (स्थिरांक ggml_metal_fusion & fusion : ggml_metal_fusions) {
        स्थिरांक पूर्णांक n_ops = (पूर्णांक) fusion.ops.size();

        // only look for a longer match than the current best
        जर (n_ops <= best) {
            सुरूठेवा;
        }
        जर (idx + n_ops > n_idxs) {
            सुरूठेवा;
        }

        स्थिरांक ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        तार्किक ok = सत्य;
        साठी (पूर्णांक j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[गाठ_idxs[idx + j]];
            जर (nodes[j]->op != fusion.ops[j]) {
                ok = असत्य;
                थांबा;
            }
        }
        जर (!ok) {
            सुरूठेवा;
        }

        जर (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            साठी (पूर्णांक j = 1; j < n_ops && ok; j++) {
                जर (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = असत्य;
                    थांबा;
                }
                जर (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = असत्य;
                    थांबा;
                }
            }
            जर (!ok) {
                सुरूठेवा;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            पूर्णांक outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = गाठ_idxs[idx + n_ops - 1];
            साठी (माप_प्रकार i = 0; i < fusion.outs.size(); ++i) {
                स्थिरांक पूर्णांक out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = गाठ_idxs[idx + out_offset];
            }

            स्थिरांक पूर्णांक n_outputs = 1 + (पूर्णांक) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            जर (!ggml_can_fuse_subgraph_ext(gf, गाठ_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                सुरूठेवा;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        जर (fusion.check && !fusion.check(&fusion, nodes, gf, गाठ_idxs, idx, mode)) {
            सुरूठेवा;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        जर (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            सुरूठेवा;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    परत res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
पूर्णांक ggml_metal_fusion_max(स्थिरांक ggml_cgraph * gf, पूर्णांक idx) {
    // a view node cannot start a pattern - pack it alone
    जर (ggml_op_is_empty(gf->nodes[idx]->op)) {
        परत 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    पूर्णांक idxs[GGML_METAL_FUSION_MAX];
    पूर्णांक n_idxs = 0;
    साठी (पूर्णांक i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        जर (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    जर (n_idxs == 0) {
        परत 1;
    }

    पूर्णांक total = 0;
    पूर्णांक i_f = 0;

    पर्यंत (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        पूर्णांक len = 1;
        स्थिरांक ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        जर (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            थांबा;
        }

        total += len;
        i_f += len;
    }

    जर (i_f == 0) {
        परत 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    परत std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
