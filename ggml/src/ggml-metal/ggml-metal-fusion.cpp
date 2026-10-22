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
स्थिर std::vector<ggml_op> ggml_metal_fusion_filter_ops(ध्रुव std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    प्रति (ggml_op op : ops_all) {
        यदि (!ggml_op_is_रिक्तानि(op)) {
            ops.push_back(op);
        }
    }

    प्रत्यागम ops;
}

संरचना ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<पूर्णाङ्क>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    तार्किक unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    तार्किक (*check)(ध्रुव संरचना ggml_metal_fusion   * fusion,
                  ध्रुव संरचना ggml_tensor * ध्रुव * nodes,
                  ध्रुव संरचना ggml_cgraph         * gf,
                  ध्रुव पूर्णाङ्क                        * ग्रन्थिः_idxs,
                        पूर्णाङ्क                          idx,
                        ggml_metal_fusion_बहुलकम्       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            ध्रुव std::vector<ggml_op> & ops_all,
            ध्रुव std::vector<पूर्णाङ्क> & outs,
            तार्किक unsafe,
            तार्किक (*check)(ध्रुव संरचना ggml_metal_fusion   * fusion,
                          ध्रुव संरचना ggml_tensor * ध्रुव * nodes,
                          ध्रुव संरचना ggml_cgraph         * gf,
                          ध्रुव पूर्णाङ्क                        * ग्रन्थिः_idxs,
                                पूर्णाङ्क                          idx,
                                ggml_metal_fusion_बहुलकम्       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(ध्रुव ggml_metal_fusion * fusion) {
    प्रत्यागम fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
स्थिर तार्किक ggml_metal_fusion_same_बफर(ध्रुव ggml_tensor * a, ध्रुव ggml_tensor * b) {
    यदि (!a || !b) {
        प्रत्यागम असत्य;
    }

    ggml_backend_बफर_t ba = a->view_src ? a->view_src->बफर : a->बफर;
    ggml_backend_बफर_t bb = b->view_src ? b->view_src->बफर : b->बफर;

    ggml_metal_बफर_t ca = (ggml_metal_बफर_t) ba->context;
    ggml_metal_बफर_t cb = (ggml_metal_बफर_t) bb->context;

    प्रत्यागम ggml_metal_बफर_get_id(ca, a).metal == ggml_metal_बफर_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
स्थिर तार्किक ggml_metal_fusion_overlap(ध्रुव ggml_tensor * a, ध्रुव ggml_tensor * b) {
    ggml_backend_बफर_t ba = a->view_src ? a->view_src->बफर : a->बफर;
    ggml_backend_बफर_t bb = b->view_src ? b->view_src->बफर : b->बफर;

    ध्रुव ggml_metal_बफर_id bid_a = ggml_metal_बफर_get_id((ggml_metal_बफर_t) ba->context, a);
    ध्रुव ggml_metal_बफर_id bid_b = ggml_metal_बफर_get_id((ggml_metal_बफर_t) bb->context, b);

    यदि (bid_a.metal == शून्यसूचकः || bid_a.metal != bid_b.metal) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
स्थिर तार्किक ggml_metal_fusion_check_norm(
        ध्रुव ggml_metal_fusion       * fusion,
        ध्रुव ggml_tensor * ध्रुव     * nodes,
        ध्रुव ggml_cgraph             * gf,
        ध्रुव पूर्णाङ्क                     * ग्रन्थिः_idxs,
              पूर्णाङ्क                       idx,
              ggml_metal_fusion_बहुलकम्    mode) {
    GGML_अप्रयुक्तम्(mode);
    GGML_अप्रयुक्तम्(gf);
    GGML_अप्रयुक्तम्(ग्रन्थिः_idxs);
    GGML_अप्रयुक्तम्(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    यदि (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        ध्रुव ggml_tensor * scale = nodes[1];
        यदि (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            प्रत्यागम असत्य;
        }

        प्रत्यागम सत्य;
    }

    प्रति (पूर्णाङ्क j = 1; j < (पूर्णाङ्क) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        यदि (nodes[j]->src[0] != nodes[j - 1]) {
            प्रत्यागम असत्य;
        }

        // the weight/bias must have the same row width as the norm input
        यदि (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            प्रत्यागम असत्य;
        }

        यदि (!ggml_is_contiguous_पङ्क्तिकाः(nodes[j]->src[1])) {
            प्रत्यागम असत्य;
        }

        यदि (nodes[j]->type != GGML_TYPE_F32) {
            प्रत्यागम असत्य;
        }
    }

    प्रत्यागम सत्य;
}

// SSM_CONV + UNARY (silu)
स्थिर तार्किक ggml_metal_fusion_check_ssm_conv_silu(
        ध्रुव ggml_metal_fusion       * fusion,
        ध्रुव ggml_tensor * ध्रुव     * nodes,
        ध्रुव ggml_cgraph             * gf,
        ध्रुव पूर्णाङ्क                     * ग्रन्थिः_idxs,
              पूर्णाङ्क                       idx,
              ggml_metal_fusion_बहुलकम्    mode) {
    GGML_अप्रयुक्तम्(fusion);
    GGML_अप्रयुक्तम्(gf);
    GGML_अप्रयुक्तम्(ग्रन्थिः_idxs);
    GGML_अप्रयुक्तम्(idx);
    GGML_अप्रयुक्तम्(mode);

    ध्रुव ggml_tensor * conv = nodes[0];
    ध्रुव ggml_tensor * un   = nodes[1];

    यदि (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        प्रत्यागम असत्य;
    }

    यदि (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        प्रत्यागम असत्य;
    }

    यदि (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_पङ्क्तिकाः(un)) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
स्थिर तार्किक ggml_metal_fusion_check_सङ्कलय_chain(
        ध्रुव ggml_metal_fusion       * fusion,
        ध्रुव ggml_tensor * ध्रुव     * nodes,
        ध्रुव ggml_cgraph             * gf,
        ध्रुव पूर्णाङ्क                     * ग्रन्थिः_idxs,
              पूर्णाङ्क                       idx,
              ggml_metal_fusion_बहुलकम्    mode) {
    GGML_अप्रयुक्तम्(gf);
    GGML_अप्रयुक्तम्(ग्रन्थिः_idxs);
    GGML_अप्रयुक्तम्(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    प्रति (पूर्णाङ्क j = 1; j < (पूर्णाङ्क) fusion->ops.size(); j++) {
        यदि (nodes[j]->src[0] != nodes[j - 1]) {
            प्रत्यागम असत्य;
        }

        यदि (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            प्रत्यागम असत्य;
        }

        यदि (mode == GGML_METAL_FUSION_पूर्णम्) {
            यदि (!ggml_metal_fusion_same_बफर(nodes[j]->src[1], nodes[0]->src[1])) {
                प्रत्यागम असत्य;
            }
        }
    }

    प्रत्यागम सत्य;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
स्थिर तार्किक ggml_metal_fusion_check_gdn_cache(
        ध्रुव ggml_metal_fusion       * fusion,
        ध्रुव ggml_tensor * ध्रुव     * nodes,
        ध्रुव ggml_cgraph             * gf,
        ध्रुव पूर्णाङ्क                     * ग्रन्थिः_idxs,
              पूर्णाङ्क                       idx,
              ggml_metal_fusion_बहुलकम्    mode) {
    GGML_अप्रयुक्तम्(fusion);
    GGML_अप्रयुक्तम्(gf);
    GGML_अप्रयुक्तम्(ग्रन्थिः_idxs);
    GGML_अप्रयुक्तम्(idx);

    ध्रुव ggml_tensor * gdn = nodes[0];
    ध्रुव ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    यदि (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        प्रत्यागम असत्य;
    }

    यदि (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        प्रत्यागम असत्य;
    }

    ध्रुव int64_t S_v      = gdn->src[2]->ne[0];
    ध्रुव int64_t H        = gdn->src[2]->ne[1];
    ध्रुव int64_t n_tokens = gdn->src[2]->ne[2];
    ध्रुव int64_t n_seqs   = gdn->src[2]->ne[3];
    ध्रुव int64_t K        = ggml_get_op_params_i32(gdn, 0);
    ध्रुव size_t  पुच्छ_off = ggml_row_आकार(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    ध्रुव int64_t D         = S_v * S_v * H;
    ध्रुव int64_t n_written = std::min<int64_t>(n_tokens, K);

    ध्रुव ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    ध्रुव ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    यदि (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != पुच्छ_off || !ggml_is_contiguous(src)) {
        प्रत्यागम असत्य;
    }

    ध्रुव int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    यदि (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_आकार(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_आकार(GGML_TYPE_F32, D)) {
        प्रत्यागम असत्य;
    }

    यदि (mode == GGML_METAL_FUSION_पूर्णम्) {
        // the cache must be allocated so the kernel can write straight to its buffer
        यदि (dst->दत्त == शून्यसूचकः) {
            प्रत्यागम असत्य;
        }
    }

    प्रत्यागम सत्य;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
स्थिर तार्किक ggml_metal_fusion_check_snake(
        ध्रुव ggml_metal_fusion       * fusion,
        ध्रुव ggml_tensor * ध्रुव     * nodes,
        ध्रुव ggml_cgraph             * gf,
        ध्रुव पूर्णाङ्क                     * ग्रन्थिः_idxs,
              पूर्णाङ्क                       idx,
              ggml_metal_fusion_बहुलकम्    mode) {
    GGML_अप्रयुक्तम्(fusion);
    GGML_अप्रयुक्तम्(mode);
    GGML_अप्रयुक्तम्(gf);
    GGML_अप्रयुक्तम्(ग्रन्थिः_idxs);
    GGML_अप्रयुक्तम्(idx);

    ध्रुव ggml_tensor * mul0     = nodes[0];
    ध्रुव ggml_tensor * sin_ग्रन्थिः = nodes[1];
    ध्रुव ggml_tensor * sqr      = nodes[2];
    ध्रुव ggml_tensor * mul1     = nodes[3];
    ध्रुव ggml_tensor * सङ्कलय_2      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    ध्रुव ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ध्रुव ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ध्रुव ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    ध्रुव ggml_tensor * x_अन्तर्_सङ्कलय = (सङ्कलय_2->src[0] == mul1) ? सङ्कलय_2->src[1] : सङ्कलय_2->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    ध्रुव तार्किक types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_ग्रन्थिः->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (सङ्कलय_2->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    ध्रुव तार्किक shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    ध्रुव तार्किक dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (सङ्कलय_2->ne[2]   == 1) && (सङ्कलय_2->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    ध्रुव तार्किक contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(सङ्कलय_2) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    प्रत्यागम types_ok && shape_ok && dim_ok && contig_ok && x_अन्तर्_सङ्कलय == x;
}

#परिभाषय GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
स्थिर ध्रुव std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_पङ्क्तिकाः
};
स्थिर ध्रुव std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_पङ्क्तिकाः, GGML_OP_SCALE
};
स्थिर ध्रुव std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_पङ्क्तिकाः,
    GGML_OP_RESHAPE, GGML_OP_योग_पङ्क्तिकाः, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
स्थिर ध्रुव std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_पङ्क्तिकाः,
    GGML_OP_RESHAPE, GGML_OP_योग_पङ्क्तिकाः, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

स्थिर तार्किक ggml_metal_fusion_check_topk_moe(
        ध्रुव ggml_metal_fusion       * fusion,
        ध्रुव ggml_tensor * ध्रुव     * nodes,
        ध्रुव ggml_cgraph             * gf,
        ध्रुव पूर्णाङ्क                     * ग्रन्थिः_idxs,
              पूर्णाङ्क                       idx,
              ggml_metal_fusion_बहुलकम्    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_अप्रयुक्तम्(nodes);

    ध्रुव पूर्णाङ्क n_ops = (पूर्णाङ्क) fusion->ops.size();

    ध्रुव तार्किक with_norm  = n_ops >= 6;
    ध्रुव तार्किक with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    ध्रुव std::vector<ggml_op> & ops_all = fusion->ops_all;

    ध्रुव पूर्णाङ्क raw_start = ग्रन्थिः_idxs[idx];
    पूर्णाङ्क       raw_अन्तः   = ग्रन्थिः_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    यदि (with_norm && !with_scale) {
        यदि (raw_अन्तः + 1 >= gf->n_nodes) {
            प्रत्यागम असत्य;
        }
        ध्रुव ggml_tensor * trailing_reshape = gf->nodes[raw_अन्तः + 1];
        यदि (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_अन्तः]) {
            प्रत्यागम असत्य;
        }
        raw_अन्तः++;
    }

    ध्रुव पूर्णाङ्क raw_गणनम् = raw_अन्तः - raw_start + 1;
    यदि (raw_गणनम् != (पूर्णाङ्क) ops_all.size()) {
        प्रत्यागम असत्य;
    }

    पूर्णाङ्क raw_idxs[GGML_METAL_FUSION_MAX];
    प्रति (पूर्णाङ्क i = 0; i < raw_गणनम्; ++i) {
        raw_idxs[i] = raw_start + i;
        यदि (gf->nodes[raw_start + i]->op != ops_all[i]) {
            प्रत्यागम असत्य;
        }
    }

    ध्रुव ggml_tensor * softmax        = gf->nodes[raw_start];
    ध्रुव ggml_tensor * probs_reshaped = gf->nodes[raw_start + 1];
    ध्रुव ggml_tensor * argsort        = gf->nodes[raw_start + 2];
    ध्रुव ggml_tensor * ids            = gf->nodes[raw_start + 3];
    ध्रुव ggml_tensor * get_पङ्क्तिकाः       = gf->nodes[raw_start + 4];
    ध्रुव ggml_tensor * बहिर्गमनम्            = gf->nodes[raw_अन्तः];
    ध्रुव ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    भिन्न scale   = 1.0f;
    भिन्न max_bias = 0.0f;
    स्मृतिं_प्रतिलिख(&scale,    ((ध्रुव int32_t *) softmax->op_params) + 0, परिमाण(scale));
    स्मृतिं_प्रतिलिख(&max_bias, ((ध्रुव int32_t *) softmax->op_params) + 1, परिमाण(max_bias));
    यदि (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        प्रत्यागम असत्य;
    }

    यदि (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        बहिर्गमनम्->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        प्रत्यागम असत्य;
    }

    ध्रुव int64_t n_expert      = logits->ne[0];
    ध्रुव int64_t n_tokens      = logits->ne[1];
    ध्रुव int64_t n_expert_प्रयुक्तम् = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    यदि (n_expert <= 0 || n_expert_प्रयुक्तम् <= 0 || n_expert_प्रयुक्तम् > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_प्रयुक्तम् > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        प्रत्यागम असत्य;
    }

    यदि (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        बहिर्गमनम्->ne[0] != 1 || बहिर्गमनम्->ne[1] != n_expert_प्रयुक्तम् || बहिर्गमनम्->ne[2] != n_tokens || बहिर्गमनम्->ne[3] != 1) {
        प्रत्यागम असत्य;
    }

    यदि (!ggml_is_contiguous(logits) || !ggml_is_contiguous(बहिर्गमनम्) ||
        ids->nb[0] != ggml_type_आकार(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_आकार(GGML_TYPE_I32) * n_expert) {
        प्रत्यागम असत्य;
    }

    यदि (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_पङ्क्तिकाः->src[0] != probs_reshaped || get_पङ्क्तिकाः->src[1] != ids) {
        प्रत्यागम असत्य;
    }

    यदि (with_norm) {
        ध्रुव ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
        ध्रुव ggml_tensor * योग_पङ्क्तिकाः_2         = gf->nodes[raw_start + 6];
        ध्रुव ggml_tensor * clamp            = gf->nodes[raw_start + 7];
        ध्रुव ggml_tensor * भाजय              = gf->nodes[raw_start + 8];
        ध्रुव ggml_tensor * बहिर्गमनम्_reshaped     = gf->nodes[raw_start + 9];

        यदि (weights_reshaped->src[0] != get_पङ्क्तिकाः || योग_पङ्क्तिकाः_2->src[0] != weights_reshaped ||
            clamp->src[0] != योग_पङ्क्तिकाः_2 || भाजय->src[0] != weights_reshaped || भाजय->src[1] != clamp ||
            बहिर्गमनम्_reshaped->src[0] != भाजय) {
            प्रत्यागम असत्य;
        }

        यदि (with_scale) {
            ध्रुव ggml_tensor * scale_ग्रन्थिः = gf->nodes[raw_start + 10];
            यदि (scale_ग्रन्थिः->src[0] != बहिर्गमनम्_reshaped) {
                प्रत्यागम असत्य;
            }
        }
    } अन्यथा यदि (with_scale) {
        ध्रुव ggml_tensor * scale_ग्रन्थिः = gf->nodes[raw_start + 5];
        यदि (scale_ग्रन्थिः->src[0] != get_पङ्क्तिकाः) {
            प्रत्यागम असत्य;
        }
    }

    ध्रुव पूर्णाङ्क outputs[2] = { raw_start + 3, raw_अन्तः };
    यदि (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_गणनम्, ops_all.दत्त(), outputs, 2)) {
        प्रत्यागम असत्य;
    }

    यदि (mode == GGML_METAL_FUSION_पूर्णम्) {
        यदि (!logits->दत्त || !बहिर्गमनम्->दत्त || !ids->दत्त) {
            प्रत्यागम असत्य;
        }
    }

    प्रत्यागम सत्य;
}

#परिभाषय GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

संरचना ggml_metal_moe_reduce_मेलनम् {
    ध्रुव ggml_tensor * experts;
    ध्रुव ggml_tensor * weights;
    ध्रुव ggml_tensor * dst;
    पूर्णाङ्क ग्रन्थिः_गणनम्;
};

स्थिर तार्किक ggml_metal_fusion_मेलनम्_moe_reduce(
        ध्रुव ggml_cgraph * gf, पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_मेलनम् * मेलनम्) {
    यदि (मेलनम् == शून्यसूचकः || ग्रन्थिः_idx < 0 || ग्रन्थिः_idx + (पूर्णाङ्क) ops_all.size() > gf->n_nodes) {
        प्रत्यागम असत्य;
    }

    ध्रुव ggml_tensor * mul = gf->nodes[ग्रन्थिः_idx];
    यदि (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    ध्रुव पूर्णाङ्क raw_गणनम्     = (पूर्णाङ्क) ops_all.size();
    ध्रुव पूर्णाङ्क n_expert_प्रयुक्तम् = raw_गणनम् / 2;

    यदि (n_expert_प्रयुक्तम् < 2 || n_expert_प्रयुक्तम् > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_गणनम् != 2 * n_expert_प्रयुक्तम्) {
        प्रत्यागम असत्य;
    }

    पूर्णाङ्क n_views = 0;
    यावत् (ग्रन्थिः_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[ग्रन्थिः_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    यदि (n_views != n_expert_प्रयुक्तम्) {
        प्रत्यागम असत्य;
    }

    प्रति (पूर्णाङ्क i = n_expert_प्रयुक्तम् + 1; i < raw_गणनम्; ++i) {
        यदि (gf->nodes[ग्रन्थिः_idx + i]->op != GGML_OP_सङ्कलय) {
            प्रत्यागम असत्य;
        }
    }

    पूर्णाङ्क raw_idxs[GGML_METAL_FUSION_MAX];
    प्रति (पूर्णाङ्क i = 0; i < raw_गणनम्; ++i) {
        raw_idxs[i] = ग्रन्थिः_idx + i;
        यदि (gf->nodes[ग्रन्थिः_idx + i]->op != ops_all[i]) {
            प्रत्यागम असत्य;
        }
    }

    ध्रुव ggml_tensor * experts = mul->src[0];
    ध्रुव ggml_tensor * weights = mul->src[1];
    ध्रुव ggml_tensor * dst     = gf->nodes[ग्रन्थिः_idx + raw_गणनम् - 1];

    यदि (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम असत्य;
    }

    ध्रुव int64_t n_embd   = experts->ne[0];
    ध्रुव int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    यदि (n_embd <= 0 || experts->ne[1] != n_expert_प्रयुक्तम् || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_प्रयुक्तम् || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        प्रत्यागम असत्य;
    }

    यदि (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        प्रत्यागम असत्य;
    }

    प्रति (पूर्णाङ्क i = 1; i <= n_expert_प्रयुक्तम्; ++i) {
        ध्रुव ggml_tensor * view = gf->nodes[ग्रन्थिः_idx + i];
        यदि (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            प्रत्यागम असत्य;
        }
    }

    ध्रुव ggml_tensor * पूर्व_सङ्कलय = शून्यसूचकः;
    प्रति (पूर्णाङ्क j = 1; j < n_expert_प्रयुक्तम्; ++j) {
        ध्रुव ggml_tensor * सङ्कलय_2 = gf->nodes[ग्रन्थिः_idx + n_expert_प्रयुक्तम् + j];
        ध्रुव ggml_tensor * rhs = gf->nodes[ग्रन्थिः_idx + j + 1];
        ध्रुव ggml_tensor * lhs = j == 1 ? gf->nodes[ग्रन्थिः_idx + 1] : पूर्व_सङ्कलय;
        यदि (सङ्कलय_2->src[0] != lhs || सङ्कलय_2->src[1] != rhs) {
            प्रत्यागम असत्य;
        }
        पूर्व_सङ्कलय = सङ्कलय_2;
    }

    ध्रुव पूर्णाङ्क outputs[1] = { ग्रन्थिः_idx + raw_गणनम् - 1 };
    यदि (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_गणनम्, ops_all.दत्त(), outputs, 1)) {
        प्रत्यागम असत्य;
    }

    मेलनम्->experts    = experts;
    मेलनम्->weights    = weights;
    मेलनम्->dst        = dst;
    मेलनम्->ग्रन्थिः_गणनम् = raw_गणनम्;
    प्रत्यागम सत्य;
}

स्थिर तार्किक ggml_metal_fusion_check_moe_reduce(
        ध्रुव ggml_metal_fusion       * fusion,
        ध्रुव ggml_tensor * ध्रुव     * nodes,
        ध्रुव ggml_cgraph             * gf,
        ध्रुव पूर्णाङ्क                     * ग्रन्थिः_idxs,
              पूर्णाङ्क                       idx,
              ggml_metal_fusion_बहुलकम्    mode) {
    GGML_अप्रयुक्तम्(nodes);

    ggml_metal_moe_reduce_मेलनम् मेलनम्;
    यदि (!ggml_metal_fusion_मेलनम्_moe_reduce(gf, ग्रन्थिः_idxs[idx], fusion->ops_all, &मेलनम्)) {
        प्रत्यागम असत्य;
    }

    यदि ((पूर्णाङ्क) fusion->ops.size() != मेलनम्.experts->ne[1]) {
        प्रत्यागम असत्य;
    }

    ध्रुव पूर्णाङ्क raw_अन्तः = ग्रन्थिः_idxs[idx] + मेलनम्.ग्रन्थिः_गणनम् - 1;
    यदि (ग्रन्थिः_idxs[idx + (पूर्णाङ्क) fusion->ops.size() - 1] != raw_अन्तः) {
        प्रत्यागम असत्य;
    }

    यदि (mode == GGML_METAL_FUSION_पूर्णम्) {
        यदि (!मेलनम्.experts->दत्त || !मेलनम्.weights->दत्त || !मेलनम्.dst->दत्त) {
            प्रत्यागम असत्य;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        यदि (ggml_metal_fusion_overlap(मेलनम्.dst, मेलनम्.experts) || ggml_metal_fusion_overlap(मेलनम्.dst, मेलनम्.weights)) {
            प्रत्यागम असत्य;
        }
    }

    प्रत्यागम सत्य;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
स्थिर तार्किक ggml_metal_tensor_is_weight(ध्रुव संरचना ggml_tensor * t) {
    ध्रुव ggml_tensor * आधारः = t->view_src != रिक्त ? t->view_src : t;

    प्रत्यागम आधारः->बफर != रिक्त && ggml_backend_बफर_get_usage(आधारः->बफर) == GGML_BACKEND_बफर_USAGE_WEIGHTS;
}

स्थिर ध्रुव संरचना ggml_tensor * ggml_metal_mul_mat_सङ्कलय_सङ्कार्यम्(ध्रुव संरचना ggml_tensor * mm, ध्रुव संरचना ggml_tensor * सङ्कलय_2) {
    यदि (सङ्कलय_2->op != GGML_OP_सङ्कलय || (सङ्कलय_2->src[0] == mm) == (सङ्कलय_2->src[1] == mm)) {
        प्रत्यागम रिक्त;
    }

    ध्रुव ggml_tensor * other = सङ्कलय_2->src[0] == mm ? सङ्कलय_2->src[1] : सङ्कलय_2->src[0];

    ध्रुव तार्किक ok = other->type == GGML_TYPE_F32 && सङ्कलय_2->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    प्रत्यागम ok ? other : रिक्त;
}

स्थिर ध्रुव संरचना ggml_tensor * ggml_metal_mul_mat_सङ्कलय_residual(ध्रुव संरचना ggml_tensor * mm, ध्रुव संरचना ggml_tensor * सङ्कलय_2) {
    ध्रुव ggml_tensor * res = ggml_metal_mul_mat_सङ्कलय_सङ्कार्यम्(mm, सङ्कलय_2);

    ध्रुव तार्किक ok = res != रिक्त && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(सङ्कलय_2);

    प्रत्यागम ok ? res : रिक्त;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
स्थिर तार्किक ggml_metal_fusion_check_mul_mat_सङ्कलय(
        ध्रुव ggml_metal_fusion       * fusion,
        ध्रुव ggml_tensor * ध्रुव     * nodes,
        ध्रुव ggml_cgraph             * gf,
        ध्रुव पूर्णाङ्क                     * ग्रन्थिः_idxs,
              पूर्णाङ्क                       idx,
              ggml_metal_fusion_बहुलकम्    mode) {
    GGML_अप्रयुक्तम्(gf);
    GGML_अप्रयुक्तम्(ग्रन्थिः_idxs);
    GGML_अप्रयुक्तम्(idx);
    GGML_अप्रयुक्तम्(fusion);

    ध्रुव ggml_tensor * mm  = nodes[0];
    ध्रुव ggml_tensor * सङ्कलय_2 = nodes[1];

    यदि (ggml_metal_mul_mat_सङ्कलय_सङ्कार्यम्(mm, सङ्कलय_2) == शून्यसूचकः ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        प्रत्यागम असत्य;
    }

    यदि (mode == GGML_METAL_FUSION_STRUCTURAL) {
        प्रत्यागम सत्य;
    }

    ध्रुव ggml_tensor * res = ggml_metal_mul_mat_सङ्कलय_residual(mm, सङ्कलय_2);

    यदि (res == शून्यसूचकः || !ggml_metal_op_mul_mat_use_mma(mm)) {
        प्रत्यागम असत्य;
    }

    प्रत्यागम !ggml_metal_fusion_overlap(सङ्कलय_2, mm->src[0]) && !ggml_metal_fusion_overlap(सङ्कलय_2, mm->src[1]) &&
        (सङ्कलय_2->दत्त == res->दत्त || !ggml_metal_fusion_overlap(सङ्कलय_2, res));
}

// ---- patterns ------------------------------------------------------------

स्थिर ध्रुव std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
स्थिर ध्रुव std::vector<ggml_op> ops_norm_mul_सङ्कलय     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_सङ्कलय };
स्थिर ध्रुव std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
स्थिर ध्रुव std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
स्थिर ध्रुव std::vector<ggml_op> ops_rms_norm_mul_सङ्कलय = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_सङ्कलय };
स्थिर ध्रुव std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

स्थिर ध्रुव std::vector<ggml_op> ops_सङ्कलय_2 = { GGML_OP_सङ्कलय, GGML_OP_सङ्कलय };
स्थिर ध्रुव std::vector<ggml_op> ops_सङ्कलय_3 = { GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय };
स्थिर ध्रुव std::vector<ggml_op> ops_सङ्कलय_4 = { GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय };
स्थिर ध्रुव std::vector<ggml_op> ops_सङ्कलय_5 = { GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय };
स्थिर ध्रुव std::vector<ggml_op> ops_सङ्कलय_6 = { GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय };
स्थिर ध्रुव std::vector<ggml_op> ops_सङ्कलय_7 = { GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय };
स्थिर ध्रुव std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_सङ्कलय };

स्थिर ध्रुव std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

स्थिर ध्रुव std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

स्थिर ध्रुव std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_सङ्कलय
};
स्थिर ध्रुव std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय
};
स्थिर ध्रुव std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय
};
स्थिर ध्रुव std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय
};
स्थिर ध्रुव std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय
};
स्थिर ध्रुव std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय
};
स्थिर ध्रुव std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय, GGML_OP_सङ्कलय
};

स्थिर ध्रुव std::vector<ggml_op> ops_mul_mat_सङ्कलय = { GGML_OP_MUL_MAT, GGML_OP_सङ्कलय };

स्थिर ध्रुव std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     असत्य, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_सङ्कलय,   ops_norm_mul_सङ्कलय,           {},     असत्य, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     असत्य, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     असत्य, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_सङ्कलय,   ops_rms_norm_mul_सङ्कलय,       {},     असत्य, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     असत्य, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_सङ्कलय_CHAIN,      ops_सङ्कलय_2,                  {},     असत्य, ggml_metal_fusion_check_सङ्कलय_chain },
    { GGML_METAL_FUSION_सङ्कलय_CHAIN,      ops_सङ्कलय_3,                  {},     असत्य, ggml_metal_fusion_check_सङ्कलय_chain },
    { GGML_METAL_FUSION_सङ्कलय_CHAIN,      ops_सङ्कलय_4,                  {},     असत्य, ggml_metal_fusion_check_सङ्कलय_chain },
    { GGML_METAL_FUSION_सङ्कलय_CHAIN,      ops_सङ्कलय_5,                  {},     असत्य, ggml_metal_fusion_check_सङ्कलय_chain },
    { GGML_METAL_FUSION_सङ्कलय_CHAIN,      ops_सङ्कलय_6,                  {},     असत्य, ggml_metal_fusion_check_सङ्कलय_chain },
    { GGML_METAL_FUSION_सङ्कलय_CHAIN,      ops_सङ्कलय_7,                  {},     असत्य, ggml_metal_fusion_check_सङ्कलय_chain },
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
    { GGML_METAL_FUSION_MUL_MAT_सङ्कलय,    ops_mul_mat_सङ्कलय,            {},     असत्य, ggml_metal_fusion_check_mul_mat_सङ्कलय },
};

// ---- alloc deps -----------------------------------------------------------

स्थिर तार्किक ggml_metal_fusion_मेलनम्_raw_आदर्शः(
        ध्रुव ggml_cgraph * gf, पूर्णाङ्क ग्रन्थिः_idx, ध्रुव std::vector<ggml_op> & ops) {
    यदि (ग्रन्थिः_idx < 0 || ग्रन्थिः_idx + (पूर्णाङ्क) ops.size() > gf->n_nodes) {
        प्रत्यागम असत्य;
    }

    प्रति (पूर्णाङ्क i = 0; i < (पूर्णाङ्क) ops.size(); ++i) {
        यदि (gf->nodes[ग्रन्थिः_idx + i]->op != ops[i]) {
            प्रत्यागम असत्य;
        }
    }

    प्रत्यागम सत्य;
}

स्थिर शून्य ggml_metal_fusion_सङ्कलय_आदर्शः_alloc_deps(
        शून्य * user_दत्त,
        शून्य (*सङ्कलय_alloc_dep)(शून्य *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        ध्रुव ggml_metal_fusion * fusion,
        पूर्णाङ्क ग्रन्थिः_idx) {
    ध्रुव पूर्णाङ्क चरमः_ग्रन्थिः = ग्रन्थिः_idx + (पूर्णाङ्क) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> आलोकितम्;
    प्रति (पूर्णाङ्क j = 0; j < (पूर्णाङ्क) fusion->ops_all.size(); ++j) {
        ggml_tensor * ग्रन्थिः = gf->nodes[ग्रन्थिः_idx + j];
        प्रति (पूर्णाङ्क s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = ग्रन्थिः->src[s];
            यदि (src && आलोकितम्.insert(src).second) {
                सङ्कलय_alloc_dep(user_दत्त, src, gf->nodes[चरमः_ग्रन्थिः]);
            }
        }
        आलोकितम्.insert(ग्रन्थिः);
    }
}

शून्य ggml_metal_fusion_सङ्कलय_alloc_deps(
        शून्य * user_दत्त,
        शून्य (*सङ्कलय_alloc_dep)(शून्य *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    प्रति (पूर्णाङ्क i = 0; i < gf->n_nodes; ++i) {
        ध्रुव ggml_metal_fusion * best = शून्यसूचकः;
        पूर्णाङ्क best_raw = 0;

        प्रति (ध्रुव ggml_metal_fusion & fusion : ggml_metal_fusions) {
            यदि ((पूर्णाङ्क) fusion.ops_all.size() <= best_raw) {
                अनुवर्त;
            }
            यदि (ggml_metal_fusion_मेलनम्_raw_आदर्शः(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (पूर्णाङ्क) fusion.ops_all.size();
            }
        }

        यदि (best) {
            ggml_metal_fusion_सङ्कलय_आदर्शः_alloc_deps(user_दत्त, सङ्कलय_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

स्थिर std::string ggml_metal_fusion_label(ध्रुव ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != शून्यसूचकः);

    std::string label;
    प्रति (पूर्णाङ्क j = 0; j < (पूर्णाङ्क) fusion->ops.size(); j++) {
        यदि (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    प्रत्यागम label;
}

संरचना ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    तार्किक enabled;
    तार्किक stats;
    तार्किक labels_समुच्चयः;
    पूर्णाङ्क  debug;
};

संरचना ggml_metal_fusion_info * ggml_metal_fusion_info_init(तार्किक enabled, पूर्णाङ्क debug) {
    ggml_metal_fusion_info * finfo = नूतन ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_समुच्चयः = असत्य;
    finfo->debug      = debug;

    यदि (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    प्रत्यागम finfo;
}

शून्य ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    विलोपय finfo;
}

तार्किक ggml_metal_fusion_info_enabled(ध्रुव ggml_metal_fusion_info * finfo) {
    प्रत्यागम finfo->enabled;
}

तार्किक ggml_metal_fusion_info_stats(ध्रुव ggml_metal_fusion_info * finfo) {
    प्रत्यागम finfo->stats;
}

पूर्णाङ्क ggml_metal_fusion_info_debug(ध्रुव ggml_metal_fusion_info * finfo) {
    प्रत्यागम finfo->debug;
}

पूर्णाङ्क ggml_metal_fusion_info_n_fusions(ध्रुव ggml_metal_fusion_info * finfo) {
    प्रत्यागम (पूर्णाङ्क) finfo->labels.size();
}

ध्रुव वर्ण * ggml_metal_fusion_info_label(ध्रुव ggml_metal_fusion_info * finfo, पूर्णाङ्क idx) {
    GGML_ASSERT(idx >= 0 && idx < (पूर्णाङ्क) finfo->labels.size());
    प्रत्यागम finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_गणनम्(ध्रुव ggml_metal_fusion_info * finfo, पूर्णाङ्क idx) {
    GGML_ASSERT(idx >= 0 && idx < (पूर्णाङ्क) finfo->counts.size());
    प्रत्यागम finfo->counts[idx];
}

शून्य ggml_metal_fusion_info_गणनम्_fusion(ggml_metal_fusion_info * finfo, ध्रुव ggml_metal_fusion * fusion) {
    यदि (!finfo->stats || fusion == शून्यसूचकः) {
        प्रत्यागम;
    }

    ध्रुव ptrdiff_t idx = fusion - ggml_metal_fusions.दत्त();
    यदि (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

शून्य ggml_metal_fusion_info_समुच्चयः_enabled(ggml_metal_fusion_info * finfo, तार्किक enabled) {
    finfo->enabled = enabled;
}

शून्य ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    यदि (finfo->labels_समुच्चयः) {
        प्रत्यागम;
    }

    finfo->labels.clear();
    finfo->counts.नियोजय(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    प्रति (ध्रुव ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_समुच्चयः = सत्य;
}

शून्य ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = सत्य;
    ggml_metal_fusion_info_labels_init(finfo);
}

शून्य ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

पूर्णाङ्क ggml_metal_fusion_info_stats_get(ध्रुव ggml_metal_fusion_info * finfo, ध्रुव वर्ण ** labels, uint64_t * counts, पूर्णाङ्क n) {
    ध्रुव पूर्णाङ्क n_fusions = (पूर्णाङ्क) finfo->labels.size();

    यदि (labels == शून्यसूचकः) {
        प्रत्यागम n_fusions;
    }

    ध्रुव पूर्णाङ्क n_fill = std::min(n, n_fusions);
    प्रति (पूर्णाङ्क i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        यदि (counts != शून्यसूचकः) {
            counts[i] = finfo->counts[i];
        }
    }

    प्रत्यागम n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
स्थिर तार्किक ggml_metal_fusion_check_स्मृतिः_ranges(
        ध्रुव ggml_metal_fusion      * fusion,
        ध्रुव ggml_tensor * ध्रुव    * nodes,
        पूर्णाङ्क                            ग्रन्थिः_गणनम्) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    स्वचालित same_स्मृतिः = [](ध्रुव ggml_tensor * a, ध्रुव ggml_tensor * b) {
        यदि (a->दत्त && b->दत्त && a->दत्त == b->दत्त) {
            प्रत्यागम सत्य;
        }
        प्रति (ध्रुव ggml_tensor * v = a; v; v = v->view_src) {
            यदि (v == b) {
                प्रत्यागम सत्य;
            }
        }
        प्रति (ध्रुव ggml_tensor * v = b; v; v = v->view_src) {
            यदि (v == a) {
                प्रत्यागम सत्य;
            }
        }
        प्रत्यागम असत्य;
    };

    स्वचालित nodes_overlap = [](ध्रुव ggml_tensor * a, ध्रुव ggml_tensor * b) {
        यदि (!a || !b || !a->दत्त || !b->दत्त || !a->बफर || !b->बफर) {
            प्रत्यागम असत्य;
        }

        यदि (a->बफर != b->बफर) {
            प्रत्यागम असत्य;
        }

        ध्रुव int64_t a_start = (int64_t) a->दत्त;
        ध्रुव int64_t a_अन्तः   = a_start + ggml_backend_buft_get_alloc_आकार(a->बफर->buft, a);
        ध्रुव int64_t b_start = (int64_t) b->दत्त;
        ध्रुव int64_t b_अन्तः   = b_start + ggml_backend_buft_get_alloc_आकार(b->बफर->buft, b);

        प्रत्यागम (b_start <= a_start && a_start < b_अन्तः) ||
               (a_start <= b_start && b_start < a_अन्तः);
    };

    स्वचालित is_intermediate = [](ध्रुव ggml_tensor * src, ध्रुव ggml_tensor * ध्रुव * nodes, पूर्णाङ्क j) {
        प्रति (पूर्णाङ्क k = 0; k < j; ++k) {
            यदि (src == nodes[k]) {
                प्रत्यागम सत्य;
            }
            प्रति (ध्रुव ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                यदि (view_src == nodes[k]) {
                    प्रत्यागम सत्य;
                }
            }
        }
        प्रत्यागम असत्य;
    };

    स्वचालित check_dst = [&](ध्रुव ggml_tensor * dst) {
        प्रति (पूर्णाङ्क j = 0; j < ग्रन्थिः_गणनम्; ++j) {
            प्रति (पूर्णाङ्क s = 0; s < GGML_MAX_SRC; ++s) {
                ध्रुव ggml_tensor * src = nodes[j]->src[s];
                यदि (!src || src->op == GGML_OP_NONE || same_स्मृतिः(src, dst)) {
                    अनुवर्त;
                }

                यदि (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    प्रत्यागम असत्य;
                }
            }
        }
        प्रत्यागम सत्य;
    };

    यदि (!check_dst(nodes[ग्रन्थिः_गणनम् - 1])) {
        प्रत्यागम असत्य;
    }

    प्रति (पूर्णाङ्क विस्थापनम् : fusion->outs) {
        GGML_ASSERT(विस्थापनम् >= 0 && विस्थापनम् < ग्रन्थिः_गणनम्);
        यदि (!check_dst(nodes[विस्थापनम्])) {
            प्रत्यागम असत्य;
        }
    }

    प्रत्यागम सत्य;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
ध्रुव ggml_metal_fusion * ggml_metal_fusion_अग्र(
        ध्रुव ggml_cgraph * gf,
        ध्रुव पूर्णाङ्क * ग्रन्थिः_idxs,
        पूर्णाङ्क n_idxs,
        पूर्णाङ्क idx,
        ggml_metal_fusion_बहुलकम् mode,
        पूर्णाङ्क * n_बहिर्गमनम्) {
    ध्रुव ggml_metal_fusion * res = शून्यसूचकः;
    पूर्णाङ्क best = 1;

    प्रति (ध्रुव ggml_metal_fusion & fusion : ggml_metal_fusions) {
        ध्रुव पूर्णाङ्क n_ops = (पूर्णाङ्क) fusion.ops.size();

        // only look for a longer match than the current best
        यदि (n_ops <= best) {
            अनुवर्त;
        }
        यदि (idx + n_ops > n_idxs) {
            अनुवर्त;
        }

        ध्रुव ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        तार्किक ok = सत्य;
        प्रति (पूर्णाङ्क j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[ग्रन्थिः_idxs[idx + j]];
            यदि (nodes[j]->op != fusion.ops[j]) {
                ok = असत्य;
                भङ्ग;
            }
        }
        यदि (!ok) {
            अनुवर्त;
        }

        यदि (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            प्रति (पूर्णाङ्क j = 1; j < n_ops && ok; j++) {
                यदि (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = असत्य;
                    भङ्ग;
                }
                यदि (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = असत्य;
                    भङ्ग;
                }
            }
            यदि (!ok) {
                अनुवर्त;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            पूर्णाङ्क outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = ग्रन्थिः_idxs[idx + n_ops - 1];
            प्रति (size_t i = 0; i < fusion.outs.size(); ++i) {
                ध्रुव पूर्णाङ्क बहिर्गमनम्_विस्थापनम् = fusion.outs[i];
                GGML_ASSERT(बहिर्गमनम्_विस्थापनम् >= 0 && बहिर्गमनम्_विस्थापनम् < n_ops);
                outputs_buf[i + 1] = ग्रन्थिः_idxs[idx + बहिर्गमनम्_विस्थापनम्];
            }

            ध्रुव पूर्णाङ्क n_outputs = 1 + (पूर्णाङ्क) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            यदि (!ggml_can_fuse_subgraph_ext(gf, ग्रन्थिः_idxs + idx, n_ops, fusion.ops.दत्त(), outputs_buf, n_outputs)) {
                अनुवर्त;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        यदि (fusion.check && !fusion.check(&fusion, nodes, gf, ग्रन्थिः_idxs, idx, mode)) {
            अनुवर्त;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        यदि (mode == GGML_METAL_FUSION_पूर्णम् &&
                !ggml_metal_fusion_check_स्मृतिः_ranges(&fusion, nodes, n_ops)) {
            अनुवर्त;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_बहिर्गमनम् = best;

    प्रत्यागम res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
पूर्णाङ्क ggml_metal_fusion_max(ध्रुव ggml_cgraph * gf, पूर्णाङ्क idx) {
    // a view node cannot start a pattern - pack it alone
    यदि (ggml_op_is_रिक्तानि(gf->nodes[idx]->op)) {
        प्रत्यागम 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    पूर्णाङ्क idxs[GGML_METAL_FUSION_MAX];
    पूर्णाङ्क n_idxs = 0;
    प्रति (पूर्णाङ्क i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        यदि (!ggml_op_is_रिक्तानि(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    यदि (n_idxs == 0) {
        प्रत्यागम 1;
    }

    पूर्णाङ्क योग = 0;
    पूर्णाङ्क i_f = 0;

    यावत् (i_f < n_idxs && योग < GGML_METAL_FUSION_MAX) {
        पूर्णाङ्क len = 1;
        ध्रुव ggml_metal_fusion * fusion = ggml_metal_fusion_अग्र(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        यदि (!fusion || योग + len > GGML_METAL_FUSION_MAX) {
            भङ्ग;
        }

        योग += len;
        i_f += len;
    }

    यदि (i_f == 0) {
        प्रत्यागम 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    प्रत्यागम std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
