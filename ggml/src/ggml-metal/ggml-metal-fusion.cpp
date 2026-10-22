#समावेश "ggml-metal-fusion.h"

#समावेश "ggml-metal-common.h"
#समावेश "ggml-metal-device.h"

#समावेश "ggml-backend-impl.h"

#समावेश <algorithm>
#समावेश <cstddef>
#समावेश <cstring>
#समावेश <set>
#समावेश <string>
#समावेश <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
अटल std::vector<ggml_op> ggml_metal_fusion_filter_ops(स्थिर std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    क्रम (ggml_op op : ops_all) {
        अगर (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    वापस ops;
}

काष्ठा ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<पूर्णांक>     म_दो;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    bool unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    bool (*check)(स्थिर काष्ठा ggml_metal_fusion   * fusion,
                  स्थिर काष्ठा ggml_tensor * स्थिर * nodes,
                  स्थिर काष्ठा ggml_cgraph         * gf,
                  स्थिर पूर्णांक                        * गाँठ_idxs,
                        पूर्णांक                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            स्थिर std::vector<ggml_op> & ops_all,
            स्थिर std::vector<पूर्णांक> & म_दो,
            bool unsafe,
            bool (*check)(स्थिर काष्ठा ggml_metal_fusion   * fusion,
                          स्थिर काष्ठा ggml_tensor * स्थिर * nodes,
                          स्थिर काष्ठा ggml_cgraph         * gf,
                          स्थिर पूर्णांक                        * गाँठ_idxs,
                                पूर्णांक                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          म_दो(म_दो),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(स्थिर ggml_metal_fusion * fusion) {
    वापस fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
अटल bool ggml_metal_fusion_same_बफर(स्थिर ggml_tensor * a, स्थिर ggml_tensor * b) {
    अगर (!a || !b) {
        वापस false;
    }

    ggml_backend_बफर_t ba = a->view_src ? a->view_src->बफर : a->बफर;
    ggml_backend_बफर_t bb = b->view_src ? b->view_src->बफर : b->बफर;

    ggml_metal_बफर_t ca = (ggml_metal_बफर_t) ba->context;
    ggml_metal_बफर_t cb = (ggml_metal_बफर_t) bb->context;

    वापस ggml_metal_बफर_get_id(ca, a).metal == ggml_metal_बफर_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
अटल bool ggml_metal_fusion_overlap(स्थिर ggml_tensor * a, स्थिर ggml_tensor * b) {
    ggml_backend_बफर_t ba = a->view_src ? a->view_src->बफर : a->बफर;
    ggml_backend_बफर_t bb = b->view_src ? b->view_src->बफर : b->बफर;

    स्थिर ggml_metal_बफर_id bid_a = ggml_metal_बफर_get_id((ggml_metal_बफर_t) ba->context, a);
    स्थिर ggml_metal_बफर_id bid_b = ggml_metal_बफर_get_id((ggml_metal_बफर_t) bb->context, b);

    अगर (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        वापस false;
    }

    वापस bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
अटल bool ggml_metal_fusion_check_norm(
        स्थिर ggml_metal_fusion       * fusion,
        स्थिर ggml_tensor * स्थिर     * nodes,
        स्थिर ggml_cgraph             * gf,
        स्थिर पूर्णांक                     * गाँठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(गाँठ_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    अगर (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        स्थिर ggml_tensor * scale = nodes[1];
        अगर (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            वापस false;
        }

        वापस true;
    }

    क्रम (पूर्णांक j = 1; j < (पूर्णांक) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        अगर (nodes[j]->src[0] != nodes[j - 1]) {
            वापस false;
        }

        // the weight/bias must have the same row width as the norm input
        अगर (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            वापस false;
        }

        अगर (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            वापस false;
        }

        अगर (nodes[j]->type != GGML_TYPE_F32) {
            वापस false;
        }
    }

    वापस true;
}

// SSM_CONV + UNARY (silu)
अटल bool ggml_metal_fusion_check_ssm_conv_silu(
        स्थिर ggml_metal_fusion       * fusion,
        स्थिर ggml_tensor * स्थिर     * nodes,
        स्थिर ggml_cgraph             * gf,
        स्थिर पूर्णांक                     * गाँठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(गाँठ_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    स्थिर ggml_tensor * conv = nodes[0];
    स्थिर ggml_tensor * un   = nodes[1];

    अगर (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        वापस false;
    }

    अगर (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        वापस false;
    }

    अगर (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        वापस false;
    }

    वापस true;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
अटल bool ggml_metal_fusion_check_add_chain(
        स्थिर ggml_metal_fusion       * fusion,
        स्थिर ggml_tensor * स्थिर     * nodes,
        स्थिर ggml_cgraph             * gf,
        स्थिर पूर्णांक                     * गाँठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(गाँठ_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    क्रम (पूर्णांक j = 1; j < (पूर्णांक) fusion->ops.size(); j++) {
        अगर (nodes[j]->src[0] != nodes[j - 1]) {
            वापस false;
        }

        अगर (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            वापस false;
        }

        अगर (mode == GGML_METAL_FUSION_FULL) {
            अगर (!ggml_metal_fusion_same_बफर(nodes[j]->src[1], nodes[0]->src[1])) {
                वापस false;
            }
        }
    }

    वापस true;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
अटल bool ggml_metal_fusion_check_gdn_cache(
        स्थिर ggml_metal_fusion       * fusion,
        स्थिर ggml_tensor * स्थिर     * nodes,
        स्थिर ggml_cgraph             * gf,
        स्थिर पूर्णांक                     * गाँठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(गाँठ_idxs);
    GGML_UNUSED(idx);

    स्थिर ggml_tensor * gdn = nodes[0];
    स्थिर ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    अगर (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        वापस false;
    }

    अगर (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        वापस false;
    }

    स्थिर int64_t S_v      = gdn->src[2]->ne[0];
    स्थिर int64_t H        = gdn->src[2]->ne[1];
    स्थिर int64_t n_tokens = gdn->src[2]->ne[2];
    स्थिर int64_t n_seqs   = gdn->src[2]->ne[3];
    स्थिर int64_t K        = ggml_get_op_params_i32(gdn, 0);
    स्थिर माप_प्रकार  पुच्छ_off = ggml_row_आकार(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    स्थिर int64_t D         = S_v * S_v * H;
    स्थिर int64_t n_written = std::min<int64_t>(n_tokens, K);

    स्थिर ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    स्थिर ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    अगर (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != पुच्छ_off || !ggml_is_contiguous(src)) {
        वापस false;
    }

    स्थिर int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    अगर (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_आकार(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_आकार(GGML_TYPE_F32, D)) {
        वापस false;
    }

    अगर (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        अगर (dst->तथ्य == nullptr) {
            वापस false;
        }
    }

    वापस true;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
अटल bool ggml_metal_fusion_check_snake(
        स्थिर ggml_metal_fusion       * fusion,
        स्थिर ggml_tensor * स्थिर     * nodes,
        स्थिर ggml_cgraph             * gf,
        स्थिर पूर्णांक                     * गाँठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(गाँठ_idxs);
    GGML_UNUSED(idx);

    स्थिर ggml_tensor * mul0     = nodes[0];
    स्थिर ggml_tensor * sin_गाँठ = nodes[1];
    स्थिर ggml_tensor * sqr      = nodes[2];
    स्थिर ggml_tensor * mul1     = nodes[3];
    स्थिर ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    स्थिर ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    स्थिर ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    स्थिर ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    स्थिर ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    स्थिर bool types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_गाँठ->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    स्थिर bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    स्थिर bool dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    स्थिर bool contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    वापस types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#घोषणा GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
अटल स्थिर std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
अटल स्थिर std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
अटल स्थिर std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_योग_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
अटल स्थिर std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_योग_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

अटल bool ggml_metal_fusion_check_topk_moe(
        स्थिर ggml_metal_fusion       * fusion,
        स्थिर ggml_tensor * स्थिर     * nodes,
        स्थिर ggml_cgraph             * gf,
        स्थिर पूर्णांक                     * गाँठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    स्थिर पूर्णांक n_ops = (पूर्णांक) fusion->ops.size();

    स्थिर bool with_norm  = n_ops >= 6;
    स्थिर bool with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    स्थिर std::vector<ggml_op> & ops_all = fusion->ops_all;

    स्थिर पूर्णांक raw_start = गाँठ_idxs[idx];
    पूर्णांक       raw_end   = गाँठ_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    अगर (with_norm && !with_scale) {
        अगर (raw_end + 1 >= gf->n_nodes) {
            वापस false;
        }
        स्थिर ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        अगर (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            वापस false;
        }
        raw_end++;
    }

    स्थिर पूर्णांक raw_गिनती = raw_end - raw_start + 1;
    अगर (raw_गिनती != (पूर्णांक) ops_all.size()) {
        वापस false;
    }

    पूर्णांक raw_idxs[GGML_METAL_FUSION_MAX];
    क्रम (पूर्णांक i = 0; i < raw_गिनती; ++i) {
        raw_idxs[i] = raw_start + i;
        अगर (gf->nodes[raw_start + i]->op != ops_all[i]) {
            वापस false;
        }
    }

    स्थिर ggml_tensor * softmax        = gf->nodes[raw_start];
    स्थिर ggml_tensor * probs_reshaped = gf->nodes[raw_start + 1];
    स्थिर ggml_tensor * argsort        = gf->nodes[raw_start + 2];
    स्थिर ggml_tensor * ids            = gf->nodes[raw_start + 3];
    स्थिर ggml_tensor * get_rows       = gf->nodes[raw_start + 4];
    स्थिर ggml_tensor * out            = gf->nodes[raw_end];
    स्थिर ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    भग्न scale   = 1.0f;
    भग्न max_bias = 0.0f;
    स_नकल(&scale,    ((स्थिर int32_t *) softmax->op_params) + 0, माप(scale));
    स_नकल(&max_bias, ((स्थिर int32_t *) softmax->op_params) + 1, माप(max_bias));
    अगर (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        वापस false;
    }

    अगर (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        वापस false;
    }

    स्थिर int64_t n_expert      = logits->ne[0];
    स्थिर int64_t n_tokens      = logits->ne[1];
    स्थिर int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    अगर (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        वापस false;
    }

    अगर (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        वापस false;
    }

    अगर (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_आकार(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_आकार(GGML_TYPE_I32) * n_expert) {
        वापस false;
    }

    अगर (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        वापस false;
    }

    अगर (with_norm) {
        स्थिर ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
        स्थिर ggml_tensor * योग_rows_2         = gf->nodes[raw_start + 6];
        स्थिर ggml_tensor * clamp            = gf->nodes[raw_start + 7];
        स्थिर ggml_tensor * भाग              = gf->nodes[raw_start + 8];
        स्थिर ggml_tensor * out_reshaped     = gf->nodes[raw_start + 9];

        अगर (weights_reshaped->src[0] != get_rows || योग_rows_2->src[0] != weights_reshaped ||
            clamp->src[0] != योग_rows_2 || भाग->src[0] != weights_reshaped || भाग->src[1] != clamp ||
            out_reshaped->src[0] != भाग) {
            वापस false;
        }

        अगर (with_scale) {
            स्थिर ggml_tensor * scale_गाँठ = gf->nodes[raw_start + 10];
            अगर (scale_गाँठ->src[0] != out_reshaped) {
                वापस false;
            }
        }
    } अन्यथा अगर (with_scale) {
        स्थिर ggml_tensor * scale_गाँठ = gf->nodes[raw_start + 5];
        अगर (scale_गाँठ->src[0] != get_rows) {
            वापस false;
        }
    }

    स्थिर पूर्णांक outputs[2] = { raw_start + 3, raw_end };
    अगर (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_गिनती, ops_all.तथ्य(), outputs, 2)) {
        वापस false;
    }

    अगर (mode == GGML_METAL_FUSION_FULL) {
        अगर (!logits->तथ्य || !out->तथ्य || !ids->तथ्य) {
            वापस false;
        }
    }

    वापस true;
}

#घोषणा GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

काष्ठा ggml_metal_moe_reduce_match {
    स्थिर ggml_tensor * experts;
    स्थिर ggml_tensor * weights;
    स्थिर ggml_tensor * dst;
    पूर्णांक गाँठ_गिनती;
};

अटल bool ggml_metal_fusion_match_moe_reduce(
        स्थिर ggml_cgraph * gf, पूर्णांक गाँठ_idx, स्थिर std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    अगर (match == nullptr || गाँठ_idx < 0 || गाँठ_idx + (पूर्णांक) ops_all.size() > gf->n_nodes) {
        वापस false;
    }

    स्थिर ggml_tensor * mul = gf->nodes[गाँठ_idx];
    अगर (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        वापस false;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    स्थिर पूर्णांक raw_गिनती     = (पूर्णांक) ops_all.size();
    स्थिर पूर्णांक n_expert_used = raw_गिनती / 2;

    अगर (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_गिनती != 2 * n_expert_used) {
        वापस false;
    }

    पूर्णांक n_views = 0;
    जबतक (गाँठ_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[गाँठ_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    अगर (n_views != n_expert_used) {
        वापस false;
    }

    क्रम (पूर्णांक i = n_expert_used + 1; i < raw_गिनती; ++i) {
        अगर (gf->nodes[गाँठ_idx + i]->op != GGML_OP_ADD) {
            वापस false;
        }
    }

    पूर्णांक raw_idxs[GGML_METAL_FUSION_MAX];
    क्रम (पूर्णांक i = 0; i < raw_गिनती; ++i) {
        raw_idxs[i] = गाँठ_idx + i;
        अगर (gf->nodes[गाँठ_idx + i]->op != ops_all[i]) {
            वापस false;
        }
    }

    स्थिर ggml_tensor * experts = mul->src[0];
    स्थिर ggml_tensor * weights = mul->src[1];
    स्थिर ggml_tensor * dst     = gf->nodes[गाँठ_idx + raw_गिनती - 1];

    अगर (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस false;
    }

    स्थिर int64_t n_embd   = experts->ne[0];
    स्थिर int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    अगर (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        वापस false;
    }

    अगर (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        वापस false;
    }

    क्रम (पूर्णांक i = 1; i <= n_expert_used; ++i) {
        स्थिर ggml_tensor * view = gf->nodes[गाँठ_idx + i];
        अगर (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (माप_प्रकार) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            वापस false;
        }
    }

    स्थिर ggml_tensor * पूर्व_add = nullptr;
    क्रम (पूर्णांक j = 1; j < n_expert_used; ++j) {
        स्थिर ggml_tensor * add = gf->nodes[गाँठ_idx + n_expert_used + j];
        स्थिर ggml_tensor * rhs = gf->nodes[गाँठ_idx + j + 1];
        स्थिर ggml_tensor * lhs = j == 1 ? gf->nodes[गाँठ_idx + 1] : पूर्व_add;
        अगर (add->src[0] != lhs || add->src[1] != rhs) {
            वापस false;
        }
        पूर्व_add = add;
    }

    स्थिर पूर्णांक outputs[1] = { गाँठ_idx + raw_गिनती - 1 };
    अगर (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_गिनती, ops_all.तथ्य(), outputs, 1)) {
        वापस false;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->गाँठ_गिनती = raw_गिनती;
    वापस true;
}

अटल bool ggml_metal_fusion_check_moe_reduce(
        स्थिर ggml_metal_fusion       * fusion,
        स्थिर ggml_tensor * स्थिर     * nodes,
        स्थिर ggml_cgraph             * gf,
        स्थिर पूर्णांक                     * गाँठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    अगर (!ggml_metal_fusion_match_moe_reduce(gf, गाँठ_idxs[idx], fusion->ops_all, &match)) {
        वापस false;
    }

    अगर ((पूर्णांक) fusion->ops.size() != match.experts->ne[1]) {
        वापस false;
    }

    स्थिर पूर्णांक raw_end = गाँठ_idxs[idx] + match.गाँठ_गिनती - 1;
    अगर (गाँठ_idxs[idx + (पूर्णांक) fusion->ops.size() - 1] != raw_end) {
        वापस false;
    }

    अगर (mode == GGML_METAL_FUSION_FULL) {
        अगर (!match.experts->तथ्य || !match.weights->तथ्य || !match.dst->तथ्य) {
            वापस false;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        अगर (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            वापस false;
        }
    }

    वापस true;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
अटल bool ggml_metal_tensor_is_weight(स्थिर काष्ठा ggml_tensor * t) {
    स्थिर ggml_tensor * base = t->view_src != शून्य ? t->view_src : t;

    वापस base->बफर != शून्य && ggml_backend_बफर_get_usage(base->बफर) == GGML_BACKEND_बफर_USAGE_WEIGHTS;
}

अटल स्थिर काष्ठा ggml_tensor * ggml_metal_mul_mat_add_operand(स्थिर काष्ठा ggml_tensor * mm, स्थिर काष्ठा ggml_tensor * add) {
    अगर (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        वापस शून्य;
    }

    स्थिर ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    स्थिर bool ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    वापस ok ? other : शून्य;
}

अटल स्थिर काष्ठा ggml_tensor * ggml_metal_mul_mat_add_residual(स्थिर काष्ठा ggml_tensor * mm, स्थिर काष्ठा ggml_tensor * add) {
    स्थिर ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    स्थिर bool ok = res != शून्य && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    वापस ok ? res : शून्य;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
अटल bool ggml_metal_fusion_check_mul_mat_add(
        स्थिर ggml_metal_fusion       * fusion,
        स्थिर ggml_tensor * स्थिर     * nodes,
        स्थिर ggml_cgraph             * gf,
        स्थिर पूर्णांक                     * गाँठ_idxs,
              पूर्णांक                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(गाँठ_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    स्थिर ggml_tensor * mm  = nodes[0];
    स्थिर ggml_tensor * add = nodes[1];

    अगर (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        वापस false;
    }

    अगर (mode == GGML_METAL_FUSION_STRUCTURAL) {
        वापस true;
    }

    स्थिर ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    अगर (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        वापस false;
    }

    वापस !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->तथ्य == res->तथ्य || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

अटल स्थिर std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
अटल स्थिर std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
अटल स्थिर std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
अटल स्थिर std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
अटल स्थिर std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
अटल स्थिर std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

अटल स्थिर std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
अटल स्थिर std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
अटल स्थिर std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
अटल स्थिर std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
अटल स्थिर std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
अटल स्थिर std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
अटल स्थिर std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

अटल स्थिर std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

अटल स्थिर std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

अटल स्थिर std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
अटल स्थिर std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
अटल स्थिर std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
अटल स्थिर std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
अटल स्थिर std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
अटल स्थिर std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
अटल स्थिर std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

अटल स्थिर std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };

अटल स्थिर std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_norm_mul_add,           {},     false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_rms_norm_mul_add,       {},     false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_2,                  {},     false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_3,                  {},     false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_4,                  {},     false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_5,                  {},     false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_6,                  {},     false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_7,                  {},     false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     false, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     true,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    true,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    true,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    true,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    true,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     true,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     true,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     true,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     true,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     true,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     true,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     true,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     false, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_ADD,    ops_mul_mat_add,            {},     false, ggml_metal_fusion_check_mul_mat_add },
};

// ---- alloc deps -----------------------------------------------------------

अटल bool ggml_metal_fusion_match_raw_pattern(
        स्थिर ggml_cgraph * gf, पूर्णांक गाँठ_idx, स्थिर std::vector<ggml_op> & ops) {
    अगर (गाँठ_idx < 0 || गाँठ_idx + (पूर्णांक) ops.size() > gf->n_nodes) {
        वापस false;
    }

    क्रम (पूर्णांक i = 0; i < (पूर्णांक) ops.size(); ++i) {
        अगर (gf->nodes[गाँठ_idx + i]->op != ops[i]) {
            वापस false;
        }
    }

    वापस true;
}

अटल व्योम ggml_metal_fusion_add_pattern_alloc_deps(
        व्योम * user_तथ्य,
        व्योम (*add_alloc_dep)(व्योम *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        स्थिर ggml_metal_fusion * fusion,
        पूर्णांक गाँठ_idx) {
    स्थिर पूर्णांक last_गाँठ = गाँठ_idx + (पूर्णांक) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    क्रम (पूर्णांक j = 0; j < (पूर्णांक) fusion->ops_all.size(); ++j) {
        ggml_tensor * गाँठ = gf->nodes[गाँठ_idx + j];
        क्रम (पूर्णांक s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = गाँठ->src[s];
            अगर (src && seen.insert(src).second) {
                add_alloc_dep(user_तथ्य, src, gf->nodes[last_गाँठ]);
            }
        }
        seen.insert(गाँठ);
    }
}

व्योम ggml_metal_fusion_add_alloc_deps(
        व्योम * user_तथ्य,
        व्योम (*add_alloc_dep)(व्योम *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    क्रम (पूर्णांक i = 0; i < gf->n_nodes; ++i) {
        स्थिर ggml_metal_fusion * best = nullptr;
        पूर्णांक best_raw = 0;

        क्रम (स्थिर ggml_metal_fusion & fusion : ggml_metal_fusions) {
            अगर ((पूर्णांक) fusion.ops_all.size() <= best_raw) {
                जारी;
            }
            अगर (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (पूर्णांक) fusion.ops_all.size();
            }
        }

        अगर (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_तथ्य, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

अटल std::string ggml_metal_fusion_label(स्थिर ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    क्रम (पूर्णांक j = 0; j < (पूर्णांक) fusion->ops.size(); j++) {
        अगर (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    वापस label;
}

काष्ठा ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    bool enabled;
    bool stats;
    bool labels_set;
    पूर्णांक  debug;
};

काष्ठा ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, पूर्णांक debug) {
    ggml_metal_fusion_info * finfo = नया ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = false;
    finfo->debug      = debug;

    अगर (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    वापस finfo;
}

व्योम ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    मिटाओ finfo;
}

bool ggml_metal_fusion_info_enabled(स्थिर ggml_metal_fusion_info * finfo) {
    वापस finfo->enabled;
}

bool ggml_metal_fusion_info_stats(स्थिर ggml_metal_fusion_info * finfo) {
    वापस finfo->stats;
}

पूर्णांक ggml_metal_fusion_info_debug(स्थिर ggml_metal_fusion_info * finfo) {
    वापस finfo->debug;
}

पूर्णांक ggml_metal_fusion_info_n_fusions(स्थिर ggml_metal_fusion_info * finfo) {
    वापस (पूर्णांक) finfo->labels.size();
}

स्थिर अक्षर * ggml_metal_fusion_info_label(स्थिर ggml_metal_fusion_info * finfo, पूर्णांक idx) {
    GGML_ASSERT(idx >= 0 && idx < (पूर्णांक) finfo->labels.size());
    वापस finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_गिनती(स्थिर ggml_metal_fusion_info * finfo, पूर्णांक idx) {
    GGML_ASSERT(idx >= 0 && idx < (पूर्णांक) finfo->counts.size());
    वापस finfo->counts[idx];
}

व्योम ggml_metal_fusion_info_गिनती_fusion(ggml_metal_fusion_info * finfo, स्थिर ggml_metal_fusion * fusion) {
    अगर (!finfo->stats || fusion == nullptr) {
        वापस;
    }

    स्थिर सूचक_भेद_प्रकार idx = fusion - ggml_metal_fusions.तथ्य();
    अगर (idx >= 0 && idx < (सूचक_भेद_प्रकार) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

व्योम ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, bool enabled) {
    finfo->enabled = enabled;
}

व्योम ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    अगर (finfo->labels_set) {
        वापस;
    }

    finfo->labels.सफा();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    क्रम (स्थिर ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = true;
}

व्योम ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = true;
    ggml_metal_fusion_info_labels_init(finfo);
}

व्योम ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

पूर्णांक ggml_metal_fusion_info_stats_get(स्थिर ggml_metal_fusion_info * finfo, स्थिर अक्षर ** labels, uint64_t * counts, पूर्णांक n) {
    स्थिर पूर्णांक n_fusions = (पूर्णांक) finfo->labels.size();

    अगर (labels == nullptr) {
        वापस n_fusions;
    }

    स्थिर पूर्णांक n_fill = std::min(n, n_fusions);
    क्रम (पूर्णांक i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        अगर (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    वापस n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
अटल bool ggml_metal_fusion_check_memory_ranges(
        स्थिर ggml_metal_fusion      * fusion,
        स्थिर ggml_tensor * स्थिर    * nodes,
        पूर्णांक                            गाँठ_गिनती) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    स्वतः same_memory = [](स्थिर ggml_tensor * a, स्थिर ggml_tensor * b) {
        अगर (a->तथ्य && b->तथ्य && a->तथ्य == b->तथ्य) {
            वापस true;
        }
        क्रम (स्थिर ggml_tensor * v = a; v; v = v->view_src) {
            अगर (v == b) {
                वापस true;
            }
        }
        क्रम (स्थिर ggml_tensor * v = b; v; v = v->view_src) {
            अगर (v == a) {
                वापस true;
            }
        }
        वापस false;
    };

    स्वतः nodes_overlap = [](स्थिर ggml_tensor * a, स्थिर ggml_tensor * b) {
        अगर (!a || !b || !a->तथ्य || !b->तथ्य || !a->बफर || !b->बफर) {
            वापस false;
        }

        अगर (a->बफर != b->बफर) {
            वापस false;
        }

        स्थिर int64_t a_start = (int64_t) a->तथ्य;
        स्थिर int64_t a_end   = a_start + ggml_backend_buft_get_alloc_आकार(a->बफर->buft, a);
        स्थिर int64_t b_start = (int64_t) b->तथ्य;
        स्थिर int64_t b_end   = b_start + ggml_backend_buft_get_alloc_आकार(b->बफर->buft, b);

        वापस (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    स्वतः is_intermediate = [](स्थिर ggml_tensor * src, स्थिर ggml_tensor * स्थिर * nodes, पूर्णांक j) {
        क्रम (पूर्णांक k = 0; k < j; ++k) {
            अगर (src == nodes[k]) {
                वापस true;
            }
            क्रम (स्थिर ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                अगर (view_src == nodes[k]) {
                    वापस true;
                }
            }
        }
        वापस false;
    };

    स्वतः check_dst = [&](स्थिर ggml_tensor * dst) {
        क्रम (पूर्णांक j = 0; j < गाँठ_गिनती; ++j) {
            क्रम (पूर्णांक s = 0; s < GGML_MAX_SRC; ++s) {
                स्थिर ggml_tensor * src = nodes[j]->src[s];
                अगर (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    जारी;
                }

                अगर (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    वापस false;
                }
            }
        }
        वापस true;
    };

    अगर (!check_dst(nodes[गाँठ_गिनती - 1])) {
        वापस false;
    }

    क्रम (पूर्णांक offset : fusion->म_दो) {
        GGML_ASSERT(offset >= 0 && offset < गाँठ_गिनती);
        अगर (!check_dst(nodes[offset])) {
            वापस false;
        }
    }

    वापस true;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
स्थिर ggml_metal_fusion * ggml_metal_fusion_अग्र(
        स्थिर ggml_cgraph * gf,
        स्थिर पूर्णांक * गाँठ_idxs,
        पूर्णांक n_idxs,
        पूर्णांक idx,
        ggml_metal_fusion_mode mode,
        पूर्णांक * n_out) {
    स्थिर ggml_metal_fusion * res = nullptr;
    पूर्णांक best = 1;

    क्रम (स्थिर ggml_metal_fusion & fusion : ggml_metal_fusions) {
        स्थिर पूर्णांक n_ops = (पूर्णांक) fusion.ops.size();

        // only look for a longer match than the current best
        अगर (n_ops <= best) {
            जारी;
        }
        अगर (idx + n_ops > n_idxs) {
            जारी;
        }

        स्थिर ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        bool ok = true;
        क्रम (पूर्णांक j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[गाँठ_idxs[idx + j]];
            अगर (nodes[j]->op != fusion.ops[j]) {
                ok = false;
                अवरोध;
            }
        }
        अगर (!ok) {
            जारी;
        }

        अगर (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            क्रम (पूर्णांक j = 1; j < n_ops && ok; j++) {
                अगर (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = false;
                    अवरोध;
                }
                अगर (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    अवरोध;
                }
            }
            अगर (!ok) {
                जारी;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            पूर्णांक outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = गाँठ_idxs[idx + n_ops - 1];
            क्रम (माप_प्रकार i = 0; i < fusion.म_दो.size(); ++i) {
                स्थिर पूर्णांक out_offset = fusion.म_दो[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = गाँठ_idxs[idx + out_offset];
            }

            स्थिर पूर्णांक n_outputs = 1 + (पूर्णांक) fusion.म_दो.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            अगर (!ggml_can_fuse_subgraph_ext(gf, गाँठ_idxs + idx, n_ops, fusion.ops.तथ्य(), outputs_buf, n_outputs)) {
                जारी;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        अगर (fusion.check && !fusion.check(&fusion, nodes, gf, गाँठ_idxs, idx, mode)) {
            जारी;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        अगर (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            जारी;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    वापस res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
पूर्णांक ggml_metal_fusion_max(स्थिर ggml_cgraph * gf, पूर्णांक idx) {
    // a view node cannot start a pattern - pack it alone
    अगर (ggml_op_is_empty(gf->nodes[idx]->op)) {
        वापस 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    पूर्णांक idxs[GGML_METAL_FUSION_MAX];
    पूर्णांक n_idxs = 0;
    क्रम (पूर्णांक i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        अगर (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    अगर (n_idxs == 0) {
        वापस 1;
    }

    पूर्णांक योग = 0;
    पूर्णांक i_f = 0;

    जबतक (i_f < n_idxs && योग < GGML_METAL_FUSION_MAX) {
        पूर्णांक len = 1;
        स्थिर ggml_metal_fusion * fusion = ggml_metal_fusion_अग्र(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        अगर (!fusion || योग + len > GGML_METAL_FUSION_MAX) {
            अवरोध;
        }

        योग += len;
        i_f += len;
    }

    अगर (i_f == 0) {
        वापस 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    वापस std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
