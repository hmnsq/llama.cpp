#అన్తర్భావ "ggml-metal-fusion.h"

#అన్తర్భావ "ggml-metal-common.h"
#అన్తర్భావ "ggml-metal-device.h"

#అన్తర్భావ "ggml-backend-impl.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstring>
#అన్తర్భావ <set>
#అన్తర్భావ <string>
#అన్తర్భావ <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
స్థిర std::vector<ggml_op> ggml_metal_fusion_filter_ops(స్థిరరాశి std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    కొరకు (ggml_op op : ops_all) {
        ఒకవేళ (!ggml_op_is_ఖాళీలు(op)) {
            ops.push_back(op);
        }
    }

    తిరిగి ops;
}

నిర్మాణం ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<పూర్ణాంకం>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    తార్కిక unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    తార్కిక (*check)(స్థిరరాశి నిర్మాణం ggml_metal_fusion   * fusion,
                  స్థిరరాశి నిర్మాణం ggml_tensor * స్థిరరాశి * nodes,
                  స్థిరరాశి నిర్మాణం ggml_cgraph         * gf,
                  స్థిరరాశి పూర్ణాంకం                        * కణుపు_idxs,
                        పూర్ణాంకం                          idx,
                        ggml_metal_fusion_బహుళకం       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            స్థిరరాశి std::vector<ggml_op> & ops_all,
            స్థిరరాశి std::vector<పూర్ణాంకం> & outs,
            తార్కిక unsafe,
            తార్కిక (*check)(స్థిరరాశి నిర్మాణం ggml_metal_fusion   * fusion,
                          స్థిరరాశి నిర్మాణం ggml_tensor * స్థిరరాశి * nodes,
                          స్థిరరాశి నిర్మాణం ggml_cgraph         * gf,
                          స్థిరరాశి పూర్ణాంకం                        * కణుపు_idxs,
                                పూర్ణాంకం                          idx,
                                ggml_metal_fusion_బహుళకం       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(స్థిరరాశి ggml_metal_fusion * fusion) {
    తిరిగి fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
స్థిర తార్కిక ggml_metal_fusion_same_బఫరు(స్థిరరాశి ggml_tensor * a, స్థిరరాశి ggml_tensor * b) {
    ఒకవేళ (!a || !b) {
        తిరిగి అబద్ధం;
    }

    ggml_backend_బఫరు_t ba = a->view_src ? a->view_src->బఫరు : a->బఫరు;
    ggml_backend_బఫరు_t bb = b->view_src ? b->view_src->బఫరు : b->బఫరు;

    ggml_metal_బఫరు_t ca = (ggml_metal_బఫరు_t) ba->context;
    ggml_metal_బఫరు_t cb = (ggml_metal_బఫరు_t) bb->context;

    తిరిగి ggml_metal_బఫరు_get_id(ca, a).metal == ggml_metal_బఫరు_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
స్థిర తార్కిక ggml_metal_fusion_overlap(స్థిరరాశి ggml_tensor * a, స్థిరరాశి ggml_tensor * b) {
    ggml_backend_బఫరు_t ba = a->view_src ? a->view_src->బఫరు : a->బఫరు;
    ggml_backend_బఫరు_t bb = b->view_src ? b->view_src->బఫరు : b->బఫరు;

    స్థిరరాశి ggml_metal_బఫరు_id bid_a = ggml_metal_బఫరు_get_id((ggml_metal_బఫరు_t) ba->context, a);
    స్థిరరాశి ggml_metal_బఫరు_id bid_b = ggml_metal_బఫరు_get_id((ggml_metal_బఫరు_t) bb->context, b);

    ఒకవేళ (bid_a.metal == శూన్యసూచిక || bid_a.metal != bid_b.metal) {
        తిరిగి అబద్ధం;
    }

    తిరిగి bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
స్థిర తార్కిక ggml_metal_fusion_check_norm(
        స్థిరరాశి ggml_metal_fusion       * fusion,
        స్థిరరాశి ggml_tensor * స్థిరరాశి     * nodes,
        స్థిరరాశి ggml_cgraph             * gf,
        స్థిరరాశి పూర్ణాంకం                     * కణుపు_idxs,
              పూర్ణాంకం                       idx,
              ggml_metal_fusion_బహుళకం    mode) {
    GGML_వాడని(mode);
    GGML_వాడని(gf);
    GGML_వాడని(కణుపు_idxs);
    GGML_వాడని(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    ఒకవేళ (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        స్థిరరాశి ggml_tensor * scale = nodes[1];
        ఒకవేళ (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }

        తిరిగి నిజం;
    }

    కొరకు (పూర్ణాంకం j = 1; j < (పూర్ణాంకం) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        ఒకవేళ (nodes[j]->src[0] != nodes[j - 1]) {
            తిరిగి అబద్ధం;
        }

        // the weight/bias must have the same row width as the norm input
        ఒకవేళ (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (!ggml_is_contiguous_అడ్డువరుసలు(nodes[j]->src[1])) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (nodes[j]->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }
    }

    తిరిగి నిజం;
}

// SSM_CONV + UNARY (silu)
స్థిర తార్కిక ggml_metal_fusion_check_ssm_conv_silu(
        స్థిరరాశి ggml_metal_fusion       * fusion,
        స్థిరరాశి ggml_tensor * స్థిరరాశి     * nodes,
        స్థిరరాశి ggml_cgraph             * gf,
        స్థిరరాశి పూర్ణాంకం                     * కణుపు_idxs,
              పూర్ణాంకం                       idx,
              ggml_metal_fusion_బహుళకం    mode) {
    GGML_వాడని(fusion);
    GGML_వాడని(gf);
    GGML_వాడని(కణుపు_idxs);
    GGML_వాడని(idx);
    GGML_వాడని(mode);

    స్థిరరాశి ggml_tensor * conv = nodes[0];
    స్థిరరాశి ggml_tensor * un   = nodes[1];

    ఒకవేళ (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_అడ్డువరుసలు(un)) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
స్థిర తార్కిక ggml_metal_fusion_check_కూడు_chain(
        స్థిరరాశి ggml_metal_fusion       * fusion,
        స్థిరరాశి ggml_tensor * స్థిరరాశి     * nodes,
        స్థిరరాశి ggml_cgraph             * gf,
        స్థిరరాశి పూర్ణాంకం                     * కణుపు_idxs,
              పూర్ణాంకం                       idx,
              ggml_metal_fusion_బహుళకం    mode) {
    GGML_వాడని(gf);
    GGML_వాడని(కణుపు_idxs);
    GGML_వాడని(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    కొరకు (పూర్ణాంకం j = 1; j < (పూర్ణాంకం) fusion->ops.size(); j++) {
        ఒకవేళ (nodes[j]->src[0] != nodes[j - 1]) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (mode == GGML_METAL_FUSION_నిండిన) {
            ఒకవేళ (!ggml_metal_fusion_same_బఫరు(nodes[j]->src[1], nodes[0]->src[1])) {
                తిరిగి అబద్ధం;
            }
        }
    }

    తిరిగి నిజం;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
స్థిర తార్కిక ggml_metal_fusion_check_gdn_cache(
        స్థిరరాశి ggml_metal_fusion       * fusion,
        స్థిరరాశి ggml_tensor * స్థిరరాశి     * nodes,
        స్థిరరాశి ggml_cgraph             * gf,
        స్థిరరాశి పూర్ణాంకం                     * కణుపు_idxs,
              పూర్ణాంకం                       idx,
              ggml_metal_fusion_బహుళకం    mode) {
    GGML_వాడని(fusion);
    GGML_వాడని(gf);
    GGML_వాడని(కణుపు_idxs);
    GGML_వాడని(idx);

    స్థిరరాశి ggml_tensor * gdn = nodes[0];
    స్థిరరాశి ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    ఒకవేళ (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int64_t S_v      = gdn->src[2]->ne[0];
    స్థిరరాశి int64_t H        = gdn->src[2]->ne[1];
    స్థిరరాశి int64_t n_టోకెన్లు = gdn->src[2]->ne[2];
    స్థిరరాశి int64_t n_seqs   = gdn->src[2]->ne[3];
    స్థిరరాశి int64_t K        = ggml_get_op_params_i32(gdn, 0);
    స్థిరరాశి size_t  తోక_off = ggml_అడ్డువరుస_కొలత(GGML_TYPE_F32, S_v * H * n_టోకెన్లు * n_seqs);

    స్థిరరాశి int64_t D         = S_v * S_v * H;
    స్థిరరాశి int64_t n_written = std::min<int64_t>(n_టోకెన్లు, K);

    స్థిరరాశి ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    స్థిరరాశి ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    ఒకవేళ (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != తోక_off || !ggml_is_contiguous(src)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    ఒకవేళ (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_కొలత(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_అడ్డువరుస_కొలత(GGML_TYPE_F32, D)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (mode == GGML_METAL_FUSION_నిండిన) {
        // the cache must be allocated so the kernel can write straight to its buffer
        ఒకవేళ (dst->దత్తాంశం == శూన్యసూచిక) {
            తిరిగి అబద్ధం;
        }
    }

    తిరిగి నిజం;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
స్థిర తార్కిక ggml_metal_fusion_check_snake(
        స్థిరరాశి ggml_metal_fusion       * fusion,
        స్థిరరాశి ggml_tensor * స్థిరరాశి     * nodes,
        స్థిరరాశి ggml_cgraph             * gf,
        స్థిరరాశి పూర్ణాంకం                     * కణుపు_idxs,
              పూర్ణాంకం                       idx,
              ggml_metal_fusion_బహుళకం    mode) {
    GGML_వాడని(fusion);
    GGML_వాడని(mode);
    GGML_వాడని(gf);
    GGML_వాడని(కణుపు_idxs);
    GGML_వాడని(idx);

    స్థిరరాశి ggml_tensor * mul0     = nodes[0];
    స్థిరరాశి ggml_tensor * sin_కణుపు = nodes[1];
    స్థిరరాశి ggml_tensor * sqr      = nodes[2];
    స్థిరరాశి ggml_tensor * mul1     = nodes[3];
    స్థిరరాశి ggml_tensor * కూడు_2      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    స్థిరరాశి ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    స్థిరరాశి ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    స్థిరరాశి ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    స్థిరరాశి ggml_tensor * x_లోపల_కూడు = (కూడు_2->src[0] == mul1) ? కూడు_2->src[1] : కూడు_2->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    స్థిరరాశి తార్కిక types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_కణుపు->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (కూడు_2->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    స్థిరరాశి తార్కిక shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    స్థిరరాశి తార్కిక dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (కూడు_2->ne[2]   == 1) && (కూడు_2->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    స్థిరరాశి తార్కిక contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(కూడు_2) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    తిరిగి types_ok && shape_ok && dim_ok && contig_ok && x_లోపల_కూడు == x;
}

#నిర్వచించు GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
స్థిర స్థిరరాశి std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_అడ్డువరుసలు
};
స్థిర స్థిరరాశి std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_అడ్డువరుసలు, GGML_OP_SCALE
};
స్థిర స్థిరరాశి std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_అడ్డువరుసలు,
    GGML_OP_RESHAPE, GGML_OP_కూడిక_అడ్డువరుసలు, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
స్థిర స్థిరరాశి std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_అడ్డువరుసలు,
    GGML_OP_RESHAPE, GGML_OP_కూడిక_అడ్డువరుసలు, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

స్థిర తార్కిక ggml_metal_fusion_check_topk_moe(
        స్థిరరాశి ggml_metal_fusion       * fusion,
        స్థిరరాశి ggml_tensor * స్థిరరాశి     * nodes,
        స్థిరరాశి ggml_cgraph             * gf,
        స్థిరరాశి పూర్ణాంకం                     * కణుపు_idxs,
              పూర్ణాంకం                       idx,
              ggml_metal_fusion_బహుళకం    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_వాడని(nodes);

    స్థిరరాశి పూర్ణాంకం n_ops = (పూర్ణాంకం) fusion->ops.size();

    స్థిరరాశి తార్కిక with_norm  = n_ops >= 6;
    స్థిరరాశి తార్కిక with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    స్థిరరాశి std::vector<ggml_op> & ops_all = fusion->ops_all;

    స్థిరరాశి పూర్ణాంకం raw_ప్రారంభం = కణుపు_idxs[idx];
    పూర్ణాంకం       raw_చివర   = కణుపు_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    ఒకవేళ (with_norm && !with_scale) {
        ఒకవేళ (raw_చివర + 1 >= gf->n_nodes) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి ggml_tensor * trailing_reshape = gf->nodes[raw_చివర + 1];
        ఒకవేళ (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_చివర]) {
            తిరిగి అబద్ధం;
        }
        raw_చివర++;
    }

    స్థిరరాశి పూర్ణాంకం raw_లెక్క = raw_చివర - raw_ప్రారంభం + 1;
    ఒకవేళ (raw_లెక్క != (పూర్ణాంకం) ops_all.size()) {
        తిరిగి అబద్ధం;
    }

    పూర్ణాంకం raw_idxs[GGML_METAL_FUSION_MAX];
    కొరకు (పూర్ణాంకం i = 0; i < raw_లెక్క; ++i) {
        raw_idxs[i] = raw_ప్రారంభం + i;
        ఒకవేళ (gf->nodes[raw_ప్రారంభం + i]->op != ops_all[i]) {
            తిరిగి అబద్ధం;
        }
    }

    స్థిరరాశి ggml_tensor * softmax        = gf->nodes[raw_ప్రారంభం];
    స్థిరరాశి ggml_tensor * probs_reshaped = gf->nodes[raw_ప్రారంభం + 1];
    స్థిరరాశి ggml_tensor * argsort        = gf->nodes[raw_ప్రారంభం + 2];
    స్థిరరాశి ggml_tensor * ids            = gf->nodes[raw_ప్రారంభం + 3];
    స్థిరరాశి ggml_tensor * get_అడ్డువరుసలు       = gf->nodes[raw_ప్రారంభం + 4];
    స్థిరరాశి ggml_tensor * బహిర్గతం            = gf->nodes[raw_చివర];
    స్థిరరాశి ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    దశాంశం scale   = 1.0f;
    దశాంశం max_bias = 0.0f;
    జ్ఞాపకం_నకలు(&scale,    ((స్థిరరాశి int32_t *) softmax->op_params) + 0, పరిమాణం(scale));
    జ్ఞాపకం_నకలు(&max_bias, ((స్థిరరాశి int32_t *) softmax->op_params) + 1, పరిమాణం(max_bias));
    ఒకవేళ (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        బహిర్గతం->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int64_t n_expert      = logits->ne[0];
    స్థిరరాశి int64_t n_టోకెన్లు      = logits->ne[1];
    స్థిరరాశి int64_t n_expert_వాడిన = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    ఒకవేళ (n_expert <= 0 || n_expert_వాడిన <= 0 || n_expert_వాడిన > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_వాడిన > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_టోకెన్లు || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        బహిర్గతం->ne[0] != 1 || బహిర్గతం->ne[1] != n_expert_వాడిన || బహిర్గతం->ne[2] != n_టోకెన్లు || బహిర్గతం->ne[3] != 1) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(logits) || !ggml_is_contiguous(బహిర్గతం) ||
        ids->nb[0] != ggml_type_కొలత(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_కొలత(GGML_TYPE_I32) * n_expert) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_అడ్డువరుసలు->src[0] != probs_reshaped || get_అడ్డువరుసలు->src[1] != ids) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (with_norm) {
        స్థిరరాశి ggml_tensor * weights_reshaped = gf->nodes[raw_ప్రారంభం + 5];
        స్థిరరాశి ggml_tensor * కూడిక_అడ్డువరుసలు         = gf->nodes[raw_ప్రారంభం + 6];
        స్థిరరాశి ggml_tensor * clamp            = gf->nodes[raw_ప్రారంభం + 7];
        స్థిరరాశి ggml_tensor * div              = gf->nodes[raw_ప్రారంభం + 8];
        స్థిరరాశి ggml_tensor * బహిర్గతం_reshaped     = gf->nodes[raw_ప్రారంభం + 9];

        ఒకవేళ (weights_reshaped->src[0] != get_అడ్డువరుసలు || కూడిక_అడ్డువరుసలు->src[0] != weights_reshaped ||
            clamp->src[0] != కూడిక_అడ్డువరుసలు || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            బహిర్గతం_reshaped->src[0] != div) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (with_scale) {
            స్థిరరాశి ggml_tensor * scale_కణుపు = gf->nodes[raw_ప్రారంభం + 10];
            ఒకవేళ (scale_కణుపు->src[0] != బహిర్గతం_reshaped) {
                తిరిగి అబద్ధం;
            }
        }
    } లేకపోతే ఒకవేళ (with_scale) {
        స్థిరరాశి ggml_tensor * scale_కణుపు = gf->nodes[raw_ప్రారంభం + 5];
        ఒకవేళ (scale_కణుపు->src[0] != get_అడ్డువరుసలు) {
            తిరిగి అబద్ధం;
        }
    }

    స్థిరరాశి పూర్ణాంకం outputs[2] = { raw_ప్రారంభం + 3, raw_చివర };
    ఒకవేళ (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_లెక్క, ops_all.దత్తాంశం(), outputs, 2)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (mode == GGML_METAL_FUSION_నిండిన) {
        ఒకవేళ (!logits->దత్తాంశం || !బహిర్గతం->దత్తాంశం || !ids->దత్తాంశం) {
            తిరిగి అబద్ధం;
        }
    }

    తిరిగి నిజం;
}

#నిర్వచించు GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

నిర్మాణం ggml_metal_moe_reduce_సరిపోలిక {
    స్థిరరాశి ggml_tensor * experts;
    స్థిరరాశి ggml_tensor * weights;
    స్థిరరాశి ggml_tensor * dst;
    పూర్ణాంకం కణుపు_లెక్క;
};

స్థిర తార్కిక ggml_metal_fusion_సరిపోలిక_moe_reduce(
        స్థిరరాశి ggml_cgraph * gf, పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_సరిపోలిక * సరిపోలిక) {
    ఒకవేళ (సరిపోలిక == శూన్యసూచిక || కణుపు_idx < 0 || కణుపు_idx + (పూర్ణాంకం) ops_all.size() > gf->n_nodes) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి ggml_tensor * mul = gf->nodes[కణుపు_idx];
    ఒకవేళ (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    స్థిరరాశి పూర్ణాంకం raw_లెక్క     = (పూర్ణాంకం) ops_all.size();
    స్థిరరాశి పూర్ణాంకం n_expert_వాడిన = raw_లెక్క / 2;

    ఒకవేళ (n_expert_వాడిన < 2 || n_expert_వాడిన > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_లెక్క != 2 * n_expert_వాడిన) {
        తిరిగి అబద్ధం;
    }

    పూర్ణాంకం n_views = 0;
    అయితే (కణుపు_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[కణుపు_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    ఒకవేళ (n_views != n_expert_వాడిన) {
        తిరిగి అబద్ధం;
    }

    కొరకు (పూర్ణాంకం i = n_expert_వాడిన + 1; i < raw_లెక్క; ++i) {
        ఒకవేళ (gf->nodes[కణుపు_idx + i]->op != GGML_OP_కూడు) {
            తిరిగి అబద్ధం;
        }
    }

    పూర్ణాంకం raw_idxs[GGML_METAL_FUSION_MAX];
    కొరకు (పూర్ణాంకం i = 0; i < raw_లెక్క; ++i) {
        raw_idxs[i] = కణుపు_idx + i;
        ఒకవేళ (gf->nodes[కణుపు_idx + i]->op != ops_all[i]) {
            తిరిగి అబద్ధం;
        }
    }

    స్థిరరాశి ggml_tensor * experts = mul->src[0];
    స్థిరరాశి ggml_tensor * weights = mul->src[1];
    స్థిరరాశి ggml_tensor * dst     = gf->nodes[కణుపు_idx + raw_లెక్క - 1];

    ఒకవేళ (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int64_t n_embd   = experts->ne[0];
    స్థిరరాశి int64_t n_టోకెన్లు = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    ఒకవేళ (n_embd <= 0 || experts->ne[1] != n_expert_వాడిన || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_వాడిన || weights->ne[2] != n_టోకెన్లు || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_టోకెన్లు || dst->ne[2] != 1 || dst->ne[3] != 1) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        తిరిగి అబద్ధం;
    }

    కొరకు (పూర్ణాంకం i = 1; i <= n_expert_వాడిన; ++i) {
        స్థిరరాశి ggml_tensor * view = gf->nodes[కణుపు_idx + i];
        ఒకవేళ (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_టోకెన్లు ||
            view->nb[1] != mul->nb[2]) {
            తిరిగి అబద్ధం;
        }
    }

    స్థిరరాశి ggml_tensor * prev_కూడు = శూన్యసూచిక;
    కొరకు (పూర్ణాంకం j = 1; j < n_expert_వాడిన; ++j) {
        స్థిరరాశి ggml_tensor * కూడు_2 = gf->nodes[కణుపు_idx + n_expert_వాడిన + j];
        స్థిరరాశి ggml_tensor * rhs = gf->nodes[కణుపు_idx + j + 1];
        స్థిరరాశి ggml_tensor * lhs = j == 1 ? gf->nodes[కణుపు_idx + 1] : prev_కూడు;
        ఒకవేళ (కూడు_2->src[0] != lhs || కూడు_2->src[1] != rhs) {
            తిరిగి అబద్ధం;
        }
        prev_కూడు = కూడు_2;
    }

    స్థిరరాశి పూర్ణాంకం outputs[1] = { కణుపు_idx + raw_లెక్క - 1 };
    ఒకవేళ (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_లెక్క, ops_all.దత్తాంశం(), outputs, 1)) {
        తిరిగి అబద్ధం;
    }

    సరిపోలిక->experts    = experts;
    సరిపోలిక->weights    = weights;
    సరిపోలిక->dst        = dst;
    సరిపోలిక->కణుపు_లెక్క = raw_లెక్క;
    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_metal_fusion_check_moe_reduce(
        స్థిరరాశి ggml_metal_fusion       * fusion,
        స్థిరరాశి ggml_tensor * స్థిరరాశి     * nodes,
        స్థిరరాశి ggml_cgraph             * gf,
        స్థిరరాశి పూర్ణాంకం                     * కణుపు_idxs,
              పూర్ణాంకం                       idx,
              ggml_metal_fusion_బహుళకం    mode) {
    GGML_వాడని(nodes);

    ggml_metal_moe_reduce_సరిపోలిక సరిపోలిక;
    ఒకవేళ (!ggml_metal_fusion_సరిపోలిక_moe_reduce(gf, కణుపు_idxs[idx], fusion->ops_all, &సరిపోలిక)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ ((పూర్ణాంకం) fusion->ops.size() != సరిపోలిక.experts->ne[1]) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి పూర్ణాంకం raw_చివర = కణుపు_idxs[idx] + సరిపోలిక.కణుపు_లెక్క - 1;
    ఒకవేళ (కణుపు_idxs[idx + (పూర్ణాంకం) fusion->ops.size() - 1] != raw_చివర) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (mode == GGML_METAL_FUSION_నిండిన) {
        ఒకవేళ (!సరిపోలిక.experts->దత్తాంశం || !సరిపోలిక.weights->దత్తాంశం || !సరిపోలిక.dst->దత్తాంశం) {
            తిరిగి అబద్ధం;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        ఒకవేళ (ggml_metal_fusion_overlap(సరిపోలిక.dst, సరిపోలిక.experts) || ggml_metal_fusion_overlap(సరిపోలిక.dst, సరిపోలిక.weights)) {
            తిరిగి అబద్ధం;
        }
    }

    తిరిగి నిజం;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
స్థిర తార్కిక ggml_metal_tensor_is_బరువు(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    స్థిరరాశి ggml_tensor * పునాది = t->view_src != ఖాళీ ? t->view_src : t;

    తిరిగి పునాది->బఫరు != ఖాళీ && ggml_backend_బఫరు_get_usage(పునాది->బఫరు) == GGML_BACKEND_బఫరు_USAGE_WEIGHTS;
}

స్థిర స్థిరరాశి నిర్మాణం ggml_tensor * ggml_metal_mul_mat_కూడు_నిర్వాహ్యం(స్థిరరాశి నిర్మాణం ggml_tensor * mm, స్థిరరాశి నిర్మాణం ggml_tensor * కూడు_2) {
    ఒకవేళ (కూడు_2->op != GGML_OP_కూడు || (కూడు_2->src[0] == mm) == (కూడు_2->src[1] == mm)) {
        తిరిగి ఖాళీ;
    }

    స్థిరరాశి ggml_tensor * other = కూడు_2->src[0] == mm ? కూడు_2->src[1] : కూడు_2->src[0];

    స్థిరరాశి తార్కిక ok = other->type == GGML_TYPE_F32 && కూడు_2->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_బరువు(other);

    తిరిగి ok ? other : ఖాళీ;
}

స్థిర స్థిరరాశి నిర్మాణం ggml_tensor * ggml_metal_mul_mat_కూడు_residual(స్థిరరాశి నిర్మాణం ggml_tensor * mm, స్థిరరాశి నిర్మాణం ggml_tensor * కూడు_2) {
    స్థిరరాశి ggml_tensor * res = ggml_metal_mul_mat_కూడు_నిర్వాహ్యం(mm, కూడు_2);

    స్థిరరాశి తార్కిక ok = res != ఖాళీ && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(కూడు_2);

    తిరిగి ok ? res : ఖాళీ;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
స్థిర తార్కిక ggml_metal_fusion_check_mul_mat_కూడు(
        స్థిరరాశి ggml_metal_fusion       * fusion,
        స్థిరరాశి ggml_tensor * స్థిరరాశి     * nodes,
        స్థిరరాశి ggml_cgraph             * gf,
        స్థిరరాశి పూర్ణాంకం                     * కణుపు_idxs,
              పూర్ణాంకం                       idx,
              ggml_metal_fusion_బహుళకం    mode) {
    GGML_వాడని(gf);
    GGML_వాడని(కణుపు_idxs);
    GGML_వాడని(idx);
    GGML_వాడని(fusion);

    స్థిరరాశి ggml_tensor * mm  = nodes[0];
    స్థిరరాశి ggml_tensor * కూడు_2 = nodes[1];

    ఒకవేళ (ggml_metal_mul_mat_కూడు_నిర్వాహ్యం(mm, కూడు_2) == శూన్యసూచిక ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (mode == GGML_METAL_FUSION_STRUCTURAL) {
        తిరిగి నిజం;
    }

    స్థిరరాశి ggml_tensor * res = ggml_metal_mul_mat_కూడు_residual(mm, కూడు_2);

    ఒకవేళ (res == శూన్యసూచిక || !ggml_metal_op_mul_mat_use_mma(mm)) {
        తిరిగి అబద్ధం;
    }

    తిరిగి !ggml_metal_fusion_overlap(కూడు_2, mm->src[0]) && !ggml_metal_fusion_overlap(కూడు_2, mm->src[1]) &&
        (కూడు_2->దత్తాంశం == res->దత్తాంశం || !ggml_metal_fusion_overlap(కూడు_2, res));
}

// ---- patterns ------------------------------------------------------------

స్థిర స్థిరరాశి std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
స్థిర స్థిరరాశి std::vector<ggml_op> ops_norm_mul_కూడు     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_కూడు };
స్థిర స్థిరరాశి std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
స్థిర స్థిరరాశి std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
స్థిర స్థిరరాశి std::vector<ggml_op> ops_rms_norm_mul_కూడు = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_కూడు };
స్థిర స్థిరరాశి std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

స్థిర స్థిరరాశి std::vector<ggml_op> ops_కూడు_2 = { GGML_OP_కూడు, GGML_OP_కూడు };
స్థిర స్థిరరాశి std::vector<ggml_op> ops_కూడు_3 = { GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు };
స్థిర స్థిరరాశి std::vector<ggml_op> ops_కూడు_4 = { GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు };
స్థిర స్థిరరాశి std::vector<ggml_op> ops_కూడు_5 = { GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు };
స్థిర స్థిరరాశి std::vector<ggml_op> ops_కూడు_6 = { GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు };
స్థిర స్థిరరాశి std::vector<ggml_op> ops_కూడు_7 = { GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు };
స్థిర స్థిరరాశి std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_కూడు };

స్థిర స్థిరరాశి std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

స్థిర స్థిరరాశి std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

స్థిర స్థిరరాశి std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_కూడు
};
స్థిర స్థిరరాశి std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_కూడు, GGML_OP_కూడు
};
స్థిర స్థిరరాశి std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు
};
స్థిర స్థిరరాశి std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు
};
స్థిర స్థిరరాశి std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు
};
స్థిర స్థిరరాశి std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు
};
స్థిర స్థిరరాశి std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు, GGML_OP_కూడు
};

స్థిర స్థిరరాశి std::vector<ggml_op> ops_mul_mat_కూడు = { GGML_OP_MUL_MAT, GGML_OP_కూడు };

స్థిర స్థిరరాశి std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     అబద్ధం, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_కూడు,   ops_norm_mul_కూడు,           {},     అబద్ధం, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     అబద్ధం, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     అబద్ధం, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_కూడు,   ops_rms_norm_mul_కూడు,       {},     అబద్ధం, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     అబద్ధం, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_కూడు_CHAIN,      ops_కూడు_2,                  {},     అబద్ధం, ggml_metal_fusion_check_కూడు_chain },
    { GGML_METAL_FUSION_కూడు_CHAIN,      ops_కూడు_3,                  {},     అబద్ధం, ggml_metal_fusion_check_కూడు_chain },
    { GGML_METAL_FUSION_కూడు_CHAIN,      ops_కూడు_4,                  {},     అబద్ధం, ggml_metal_fusion_check_కూడు_chain },
    { GGML_METAL_FUSION_కూడు_CHAIN,      ops_కూడు_5,                  {},     అబద్ధం, ggml_metal_fusion_check_కూడు_chain },
    { GGML_METAL_FUSION_కూడు_CHAIN,      ops_కూడు_6,                  {},     అబద్ధం, ggml_metal_fusion_check_కూడు_chain },
    { GGML_METAL_FUSION_కూడు_CHAIN,      ops_కూడు_7,                  {},     అబద్ధం, ggml_metal_fusion_check_కూడు_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     అబద్ధం, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     నిజం,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    నిజం,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    నిజం,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    నిజం,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    నిజం,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     నిజం,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     నిజం,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     నిజం,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     నిజం,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     నిజం,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     నిజం,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     నిజం,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     అబద్ధం, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_కూడు,    ops_mul_mat_కూడు,            {},     అబద్ధం, ggml_metal_fusion_check_mul_mat_కూడు },
};

// ---- alloc deps -----------------------------------------------------------

స్థిర తార్కిక ggml_metal_fusion_సరిపోలిక_raw_నమూనా(
        స్థిరరాశి ggml_cgraph * gf, పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::vector<ggml_op> & ops) {
    ఒకవేళ (కణుపు_idx < 0 || కణుపు_idx + (పూర్ణాంకం) ops.size() > gf->n_nodes) {
        తిరిగి అబద్ధం;
    }

    కొరకు (పూర్ణాంకం i = 0; i < (పూర్ణాంకం) ops.size(); ++i) {
        ఒకవేళ (gf->nodes[కణుపు_idx + i]->op != ops[i]) {
            తిరిగి అబద్ధం;
        }
    }

    తిరిగి నిజం;
}

స్థిర శూన్య ggml_metal_fusion_కూడు_నమూనా_alloc_deps(
        శూన్య * user_దత్తాంశం,
        శూన్య (*కూడు_alloc_dep)(శూన్య *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        స్థిరరాశి ggml_metal_fusion * fusion,
        పూర్ణాంకం కణుపు_idx) {
    స్థిరరాశి పూర్ణాంకం చివరి_కణుపు = కణుపు_idx + (పూర్ణాంకం) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> చూసిన;
    కొరకు (పూర్ణాంకం j = 0; j < (పూర్ణాంకం) fusion->ops_all.size(); ++j) {
        ggml_tensor * కణుపు = gf->nodes[కణుపు_idx + j];
        కొరకు (పూర్ణాంకం s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = కణుపు->src[s];
            ఒకవేళ (src && చూసిన.insert(src).second) {
                కూడు_alloc_dep(user_దత్తాంశం, src, gf->nodes[చివరి_కణుపు]);
            }
        }
        చూసిన.insert(కణుపు);
    }
}

శూన్య ggml_metal_fusion_కూడు_alloc_deps(
        శూన్య * user_దత్తాంశం,
        శూన్య (*కూడు_alloc_dep)(శూన్య *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    కొరకు (పూర్ణాంకం i = 0; i < gf->n_nodes; ++i) {
        స్థిరరాశి ggml_metal_fusion * ఉత్తమ = శూన్యసూచిక;
        పూర్ణాంకం ఉత్తమ_raw = 0;

        కొరకు (స్థిరరాశి ggml_metal_fusion & fusion : ggml_metal_fusions) {
            ఒకవేళ ((పూర్ణాంకం) fusion.ops_all.size() <= ఉత్తమ_raw) {
                కొనసాగు;
            }
            ఒకవేళ (ggml_metal_fusion_సరిపోలిక_raw_నమూనా(gf, i, fusion.ops_all)) {
                ఉత్తమ = &fusion;
                ఉత్తమ_raw = (పూర్ణాంకం) fusion.ops_all.size();
            }
        }

        ఒకవేళ (ఉత్తమ) {
            ggml_metal_fusion_కూడు_నమూనా_alloc_deps(user_దత్తాంశం, కూడు_alloc_dep, gf, ఉత్తమ, i);
            i += ఉత్తమ_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

స్థిర std::string ggml_metal_fusion_label(స్థిరరాశి ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != శూన్యసూచిక);

    std::string label;
    కొరకు (పూర్ణాంకం j = 0; j < (పూర్ణాంకం) fusion->ops.size(); j++) {
        ఒకవేళ (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    తిరిగి label;
}

నిర్మాణం ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    తార్కిక enabled;
    తార్కిక stats;
    తార్కిక labels_సమితి;
    పూర్ణాంకం  debug;
};

నిర్మాణం ggml_metal_fusion_info * ggml_metal_fusion_info_init(తార్కిక enabled, పూర్ణాంకం debug) {
    ggml_metal_fusion_info * finfo = కొత్త ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_సమితి = అబద్ధం;
    finfo->debug      = debug;

    ఒకవేళ (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    తిరిగి finfo;
}

శూన్య ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    తొలగించు finfo;
}

తార్కిక ggml_metal_fusion_info_enabled(స్థిరరాశి ggml_metal_fusion_info * finfo) {
    తిరిగి finfo->enabled;
}

తార్కిక ggml_metal_fusion_info_stats(స్థిరరాశి ggml_metal_fusion_info * finfo) {
    తిరిగి finfo->stats;
}

పూర్ణాంకం ggml_metal_fusion_info_debug(స్థిరరాశి ggml_metal_fusion_info * finfo) {
    తిరిగి finfo->debug;
}

పూర్ణాంకం ggml_metal_fusion_info_n_fusions(స్థిరరాశి ggml_metal_fusion_info * finfo) {
    తిరిగి (పూర్ణాంకం) finfo->labels.size();
}

స్థిరరాశి అక్షరం * ggml_metal_fusion_info_label(స్థిరరాశి ggml_metal_fusion_info * finfo, పూర్ణాంకం idx) {
    GGML_ASSERT(idx >= 0 && idx < (పూర్ణాంకం) finfo->labels.size());
    తిరిగి finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_లెక్క(స్థిరరాశి ggml_metal_fusion_info * finfo, పూర్ణాంకం idx) {
    GGML_ASSERT(idx >= 0 && idx < (పూర్ణాంకం) finfo->counts.size());
    తిరిగి finfo->counts[idx];
}

శూన్య ggml_metal_fusion_info_లెక్క_fusion(ggml_metal_fusion_info * finfo, స్థిరరాశి ggml_metal_fusion * fusion) {
    ఒకవేళ (!finfo->stats || fusion == శూన్యసూచిక) {
        తిరిగి;
    }

    స్థిరరాశి ptrdiff_t idx = fusion - ggml_metal_fusions.దత్తాంశం();
    ఒకవేళ (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

శూన్య ggml_metal_fusion_info_సమితి_enabled(ggml_metal_fusion_info * finfo, తార్కిక enabled) {
    finfo->enabled = enabled;
}

శూన్య ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    ఒకవేళ (finfo->labels_సమితి) {
        తిరిగి;
    }

    finfo->labels.clear();
    finfo->counts.కేటాయించు(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    కొరకు (స్థిరరాశి ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_సమితి = నిజం;
}

శూన్య ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = నిజం;
    ggml_metal_fusion_info_labels_init(finfo);
}

శూన్య ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

పూర్ణాంకం ggml_metal_fusion_info_stats_get(స్థిరరాశి ggml_metal_fusion_info * finfo, స్థిరరాశి అక్షరం ** labels, uint64_t * counts, పూర్ణాంకం n) {
    స్థిరరాశి పూర్ణాంకం n_fusions = (పూర్ణాంకం) finfo->labels.size();

    ఒకవేళ (labels == శూన్యసూచిక) {
        తిరిగి n_fusions;
    }

    స్థిరరాశి పూర్ణాంకం n_fill = std::min(n, n_fusions);
    కొరకు (పూర్ణాంకం i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        ఒకవేళ (counts != శూన్యసూచిక) {
            counts[i] = finfo->counts[i];
        }
    }

    తిరిగి n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
స్థిర తార్కిక ggml_metal_fusion_check_జ్ఞాపకం_ranges(
        స్థిరరాశి ggml_metal_fusion      * fusion,
        స్థిరరాశి ggml_tensor * స్థిరరాశి    * nodes,
        పూర్ణాంకం                            కణుపు_లెక్క) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    //ఆధారం
    స్వయంచాలక same_జ్ఞాపకం = [](స్థిరరాశి ggml_tensor * a, స్థిరరాశి ggml_tensor * b) {
        ఒకవేళ (a->దత్తాంశం && b->దత్తాంశం && a->దత్తాంశం == b->దత్తాంశం) {
            తిరిగి నిజం;
        }
        కొరకు (స్థిరరాశి ggml_tensor * v = a; v; v = v->view_src) {
            ఒకవేళ (v == b) {
                తిరిగి నిజం;
            }
        }
        కొరకు (స్థిరరాశి ggml_tensor * v = b; v; v = v->view_src) {
            ఒకవేళ (v == a) {
                తిరిగి నిజం;
            }
        }
        తిరిగి అబద్ధం;
    };

    స్వయంచాలక nodes_overlap = [](స్థిరరాశి ggml_tensor * a, స్థిరరాశి ggml_tensor * b) {
        ఒకవేళ (!a || !b || !a->దత్తాంశం || !b->దత్తాంశం || !a->బఫరు || !b->బఫరు) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (a->బఫరు != b->బఫరు) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి int64_t a_ప్రారంభం = (int64_t) a->దత్తాంశం;
        స్థిరరాశి int64_t a_చివర   = a_ప్రారంభం + ggml_backend_buft_get_alloc_కొలత(a->బఫరు->buft, a);
        స్థిరరాశి int64_t b_ప్రారంభం = (int64_t) b->దత్తాంశం;
        స్థిరరాశి int64_t b_చివర   = b_ప్రారంభం + ggml_backend_buft_get_alloc_కొలత(b->బఫరు->buft, b);

        తిరిగి (b_ప్రారంభం <= a_ప్రారంభం && a_ప్రారంభం < b_చివర) ||
               (a_ప్రారంభం <= b_ప్రారంభం && b_ప్రారంభం < a_చివర);
    };

    స్వయంచాలక is_intermediate = [](స్థిరరాశి ggml_tensor * src, స్థిరరాశి ggml_tensor * స్థిరరాశి * nodes, పూర్ణాంకం j) {
        కొరకు (పూర్ణాంకం k = 0; k < j; ++k) {
            ఒకవేళ (src == nodes[k]) {
                తిరిగి నిజం;
            }
            కొరకు (స్థిరరాశి ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                ఒకవేళ (view_src == nodes[k]) {
                    తిరిగి నిజం;
                }
            }
        }
        తిరిగి అబద్ధం;
    };

    స్వయంచాలక check_dst = [&](స్థిరరాశి ggml_tensor * dst) {
        కొరకు (పూర్ణాంకం j = 0; j < కణుపు_లెక్క; ++j) {
            కొరకు (పూర్ణాంకం s = 0; s < GGML_MAX_SRC; ++s) {
                స్థిరరాశి ggml_tensor * src = nodes[j]->src[s];
                ఒకవేళ (!src || src->op == GGML_OP_NONE || same_జ్ఞాపకం(src, dst)) {
                    కొనసాగు;
                }

                ఒకవేళ (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    తిరిగి అబద్ధం;
                }
            }
        }
        తిరిగి నిజం;
    };

    ఒకవేళ (!check_dst(nodes[కణుపు_లెక్క - 1])) {
        తిరిగి అబద్ధం;
    }

    కొరకు (పూర్ణాంకం స్థానభ్రంశం : fusion->outs) {
        GGML_ASSERT(స్థానభ్రంశం >= 0 && స్థానభ్రంశం < కణుపు_లెక్క);
        ఒకవేళ (!check_dst(nodes[స్థానభ్రంశం])) {
            తిరిగి అబద్ధం;
        }
    }

    తిరిగి నిజం;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
స్థిరరాశి ggml_metal_fusion * ggml_metal_fusion_తదుపరి(
        స్థిరరాశి ggml_cgraph * gf,
        స్థిరరాశి పూర్ణాంకం * కణుపు_idxs,
        పూర్ణాంకం n_idxs,
        పూర్ణాంకం idx,
        ggml_metal_fusion_బహుళకం mode,
        పూర్ణాంకం * n_బహిర్గతం) {
    స్థిరరాశి ggml_metal_fusion * res = శూన్యసూచిక;
    పూర్ణాంకం ఉత్తమ = 1;

    కొరకు (స్థిరరాశి ggml_metal_fusion & fusion : ggml_metal_fusions) {
        స్థిరరాశి పూర్ణాంకం n_ops = (పూర్ణాంకం) fusion.ops.size();

        // only look for a longer match than the current best
        ఒకవేళ (n_ops <= ఉత్తమ) {
            కొనసాగు;
        }
        ఒకవేళ (idx + n_ops > n_idxs) {
            కొనసాగు;
        }

        స్థిరరాశి ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        తార్కిక ok = నిజం;
        కొరకు (పూర్ణాంకం j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[కణుపు_idxs[idx + j]];
            ఒకవేళ (nodes[j]->op != fusion.ops[j]) {
                ok = అబద్ధం;
                ఆపు;
            }
        }
        ఒకవేళ (!ok) {
            కొనసాగు;
        }

        ఒకవేళ (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            కొరకు (పూర్ణాంకం j = 1; j < n_ops && ok; j++) {
                ఒకవేళ (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = అబద్ధం;
                    ఆపు;
                }
                ఒకవేళ (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = అబద్ధం;
                    ఆపు;
                }
            }
            ఒకవేళ (!ok) {
                కొనసాగు;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            పూర్ణాంకం outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = కణుపు_idxs[idx + n_ops - 1];
            కొరకు (size_t i = 0; i < fusion.outs.size(); ++i) {
                స్థిరరాశి పూర్ణాంకం బహిర్గతం_స్థానభ్రంశం = fusion.outs[i];
                GGML_ASSERT(బహిర్గతం_స్థానభ్రంశం >= 0 && బహిర్గతం_స్థానభ్రంశం < n_ops);
                outputs_buf[i + 1] = కణుపు_idxs[idx + బహిర్గతం_స్థానభ్రంశం];
            }

            స్థిరరాశి పూర్ణాంకం n_outputs = 1 + (పూర్ణాంకం) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            ఒకవేళ (!ggml_can_fuse_subgraph_ext(gf, కణుపు_idxs + idx, n_ops, fusion.ops.దత్తాంశం(), outputs_buf, n_outputs)) {
                కొనసాగు;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        ఒకవేళ (fusion.check && !fusion.check(&fusion, nodes, gf, కణుపు_idxs, idx, mode)) {
            కొనసాగు;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        ఒకవేళ (mode == GGML_METAL_FUSION_నిండిన &&
                !ggml_metal_fusion_check_జ్ఞాపకం_ranges(&fusion, nodes, n_ops)) {
            కొనసాగు;
        }

        ఉత్తమ = n_ops;
        res = &fusion;
    }

    *n_బహిర్గతం = ఉత్తమ;

    తిరిగి res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
పూర్ణాంకం ggml_metal_fusion_max(స్థిరరాశి ggml_cgraph * gf, పూర్ణాంకం idx) {
    // a view node cannot start a pattern - pack it alone
    ఒకవేళ (ggml_op_is_ఖాళీలు(gf->nodes[idx]->op)) {
        తిరిగి 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    పూర్ణాంకం idxs[GGML_METAL_FUSION_MAX];
    పూర్ణాంకం n_idxs = 0;
    కొరకు (పూర్ణాంకం i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        ఒకవేళ (!ggml_op_is_ఖాళీలు(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    ఒకవేళ (n_idxs == 0) {
        తిరిగి 1;
    }

    పూర్ణాంకం మొత్తం = 0;
    పూర్ణాంకం i_f = 0;

    అయితే (i_f < n_idxs && మొత్తం < GGML_METAL_FUSION_MAX) {
        పూర్ణాంకం len = 1;
        స్థిరరాశి ggml_metal_fusion * fusion = ggml_metal_fusion_తదుపరి(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        ఒకవేళ (!fusion || మొత్తం + len > GGML_METAL_FUSION_MAX) {
            ఆపు;
        }

        మొత్తం += len;
        i_f += len;
    }

    ఒకవేళ (i_f == 0) {
        తిరిగి 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    తిరిగి std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
