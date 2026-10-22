#അന്തര്ഭാവ "ggml-metal-fusion.h"

#അന്തര്ഭാവ "ggml-metal-common.h"
#അന്തര്ഭാവ "ggml-metal-device.h"

#അന്തര്ഭാവ "ggml-backend-impl.h"

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <set>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
സ്ഥിതം std::vector<ggml_op> ggml_metal_fusion_filter_ops(സ്ഥിരം std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    വേണ്ടി (ggml_op op : ops_all) {
        എങ്കിൽ (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    തിരികെ ops;
}

ഘടന ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<പൂർണ്ണസംഖ്യ>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    തർക്കം unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    തർക്കം (*check)(സ്ഥിരം ഘടന ggml_metal_fusion   * fusion,
                  സ്ഥിരം ഘടന ggml_tensor * സ്ഥിരം * nodes,
                  സ്ഥിരം ഘടന ggml_cgraph         * gf,
                  സ്ഥിരം പൂർണ്ണസംഖ്യ                        * കണ്ണി_idxs,
                        പൂർണ്ണസംഖ്യ                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            സ്ഥിരം std::vector<ggml_op> & ops_all,
            സ്ഥിരം std::vector<പൂർണ്ണസംഖ്യ> & outs,
            തർക്കം unsafe,
            തർക്കം (*check)(സ്ഥിരം ഘടന ggml_metal_fusion   * fusion,
                          സ്ഥിരം ഘടന ggml_tensor * സ്ഥിരം * nodes,
                          സ്ഥിരം ഘടന ggml_cgraph         * gf,
                          സ്ഥിരം പൂർണ്ണസംഖ്യ                        * കണ്ണി_idxs,
                                പൂർണ്ണസംഖ്യ                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(സ്ഥിരം ggml_metal_fusion * fusion) {
    തിരികെ fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
സ്ഥിതം തർക്കം ggml_metal_fusion_same_buffer(സ്ഥിരം ggml_tensor * a, സ്ഥിരം ggml_tensor * b) {
    എങ്കിൽ (!a || !b) {
        തിരികെ അസത്യം;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    തിരികെ ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
സ്ഥിതം തർക്കം ggml_metal_fusion_overlap(സ്ഥിരം ggml_tensor * a, സ്ഥിരം ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    സ്ഥിരം ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    സ്ഥിരം ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    എങ്കിൽ (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        തിരികെ അസത്യം;
    }

    തിരികെ bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
സ്ഥിതം തർക്കം ggml_metal_fusion_check_norm(
        സ്ഥിരം ggml_metal_fusion       * fusion,
        സ്ഥിരം ggml_tensor * സ്ഥിരം     * nodes,
        സ്ഥിരം ggml_cgraph             * gf,
        സ്ഥിരം പൂർണ്ണസംഖ്യ                     * കണ്ണി_idxs,
              പൂർണ്ണസംഖ്യ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(കണ്ണി_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    എങ്കിൽ (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        സ്ഥിരം ggml_tensor * scale = nodes[1];
        എങ്കിൽ (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            തിരികെ അസത്യം;
        }

        തിരികെ സത്യം;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 1; j < (പൂർണ്ണസംഖ്യ) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        എങ്കിൽ (nodes[j]->src[0] != nodes[j - 1]) {
            തിരികെ അസത്യം;
        }

        // the weight/bias must have the same row width as the norm input
        എങ്കിൽ (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (nodes[j]->type != GGML_TYPE_F32) {
            തിരികെ അസത്യം;
        }
    }

    തിരികെ സത്യം;
}

// SSM_CONV + UNARY (silu)
സ്ഥിതം തർക്കം ggml_metal_fusion_check_ssm_conv_silu(
        സ്ഥിരം ggml_metal_fusion       * fusion,
        സ്ഥിരം ggml_tensor * സ്ഥിരം     * nodes,
        സ്ഥിരം ggml_cgraph             * gf,
        സ്ഥിരം പൂർണ്ണസംഖ്യ                     * കണ്ണി_idxs,
              പൂർണ്ണസംഖ്യ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(കണ്ണി_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    സ്ഥിരം ggml_tensor * conv = nodes[0];
    സ്ഥിരം ggml_tensor * un   = nodes[1];

    എങ്കിൽ (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
സ്ഥിതം തർക്കം ggml_metal_fusion_check_add_chain(
        സ്ഥിരം ggml_metal_fusion       * fusion,
        സ്ഥിരം ggml_tensor * സ്ഥിരം     * nodes,
        സ്ഥിരം ggml_cgraph             * gf,
        സ്ഥിരം പൂർണ്ണസംഖ്യ                     * കണ്ണി_idxs,
              പൂർണ്ണസംഖ്യ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(കണ്ണി_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 1; j < (പൂർണ്ണസംഖ്യ) fusion->ops.size(); j++) {
        എങ്കിൽ (nodes[j]->src[0] != nodes[j - 1]) {
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (mode == GGML_METAL_FUSION_FULL) {
            എങ്കിൽ (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                തിരികെ അസത്യം;
            }
        }
    }

    തിരികെ സത്യം;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
സ്ഥിതം തർക്കം ggml_metal_fusion_check_gdn_cache(
        സ്ഥിരം ggml_metal_fusion       * fusion,
        സ്ഥിരം ggml_tensor * സ്ഥിരം     * nodes,
        സ്ഥിരം ggml_cgraph             * gf,
        സ്ഥിരം പൂർണ്ണസംഖ്യ                     * കണ്ണി_idxs,
              പൂർണ്ണസംഖ്യ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(കണ്ണി_idxs);
    GGML_UNUSED(idx);

    സ്ഥിരം ggml_tensor * gdn = nodes[0];
    സ്ഥിരം ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    എങ്കിൽ (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int64_t S_v      = gdn->src[2]->ne[0];
    സ്ഥിരം int64_t H        = gdn->src[2]->ne[1];
    സ്ഥിരം int64_t n_tokens = gdn->src[2]->ne[2];
    സ്ഥിരം int64_t n_seqs   = gdn->src[2]->ne[3];
    സ്ഥിരം int64_t K        = ggml_get_op_params_i32(gdn, 0);
    സ്ഥിരം size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    സ്ഥിരം int64_t D         = S_v * S_v * H;
    സ്ഥിരം int64_t n_written = std::min<int64_t>(n_tokens, K);

    സ്ഥിരം ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    സ്ഥിരം ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    എങ്കിൽ (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    എങ്കിൽ (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        എങ്കിൽ (dst->data == nullptr) {
            തിരികെ അസത്യം;
        }
    }

    തിരികെ സത്യം;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
സ്ഥിതം തർക്കം ggml_metal_fusion_check_snake(
        സ്ഥിരം ggml_metal_fusion       * fusion,
        സ്ഥിരം ggml_tensor * സ്ഥിരം     * nodes,
        സ്ഥിരം ggml_cgraph             * gf,
        സ്ഥിരം പൂർണ്ണസംഖ്യ                     * കണ്ണി_idxs,
              പൂർണ്ണസംഖ്യ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(കണ്ണി_idxs);
    GGML_UNUSED(idx);

    സ്ഥിരം ggml_tensor * mul0     = nodes[0];
    സ്ഥിരം ggml_tensor * sin_കണ്ണി = nodes[1];
    സ്ഥിരം ggml_tensor * sqr      = nodes[2];
    സ്ഥിരം ggml_tensor * mul1     = nodes[3];
    സ്ഥിരം ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    സ്ഥിരം ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    സ്ഥിരം ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    സ്ഥിരം ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    സ്ഥിരം ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    സ്ഥിരം തർക്കം types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_കണ്ണി->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    സ്ഥിരം തർക്കം shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    സ്ഥിരം തർക്കം dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    സ്ഥിരം തർക്കം contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    തിരികെ types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

സ്ഥിതം തർക്കം ggml_metal_fusion_check_topk_moe(
        സ്ഥിരം ggml_metal_fusion       * fusion,
        സ്ഥിരം ggml_tensor * സ്ഥിരം     * nodes,
        സ്ഥിരം ggml_cgraph             * gf,
        സ്ഥിരം പൂർണ്ണസംഖ്യ                     * കണ്ണി_idxs,
              പൂർണ്ണസംഖ്യ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    സ്ഥിരം പൂർണ്ണസംഖ്യ n_ops = (പൂർണ്ണസംഖ്യ) fusion->ops.size();

    സ്ഥിരം തർക്കം with_norm  = n_ops >= 6;
    സ്ഥിരം തർക്കം with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    സ്ഥിരം std::vector<ggml_op> & ops_all = fusion->ops_all;

    സ്ഥിരം പൂർണ്ണസംഖ്യ raw_start = കണ്ണി_idxs[idx];
    പൂർണ്ണസംഖ്യ       raw_end   = കണ്ണി_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    എങ്കിൽ (with_norm && !with_scale) {
        എങ്കിൽ (raw_end + 1 >= gf->n_nodes) {
            തിരികെ അസത്യം;
        }
        സ്ഥിരം ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        എങ്കിൽ (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            തിരികെ അസത്യം;
        }
        raw_end++;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ raw_എണ്ണം = raw_end - raw_start + 1;
    എങ്കിൽ (raw_എണ്ണം != (പൂർണ്ണസംഖ്യ) ops_all.size()) {
        തിരികെ അസത്യം;
    }

    പൂർണ്ണസംഖ്യ raw_idxs[GGML_METAL_FUSION_MAX];
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < raw_എണ്ണം; ++i) {
        raw_idxs[i] = raw_start + i;
        എങ്കിൽ (gf->nodes[raw_start + i]->op != ops_all[i]) {
            തിരികെ അസത്യം;
        }
    }

    സ്ഥിരം ggml_tensor * softmax        = gf->nodes[raw_start];
    സ്ഥിരം ggml_tensor * probs_reshaped = gf->nodes[raw_start + 1];
    സ്ഥിരം ggml_tensor * argsort        = gf->nodes[raw_start + 2];
    സ്ഥിരം ggml_tensor * ids            = gf->nodes[raw_start + 3];
    സ്ഥിരം ggml_tensor * get_rows       = gf->nodes[raw_start + 4];
    സ്ഥിരം ggml_tensor * out            = gf->nodes[raw_end];
    സ്ഥിരം ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    ദശാംശം scale   = 1.0f;
    ദശാംശം max_bias = 0.0f;
    memcpy(&scale,    ((സ്ഥിരം int32_t *) softmax->op_params) + 0, വലുപ്പം(scale));
    memcpy(&max_bias, ((സ്ഥിരം int32_t *) softmax->op_params) + 1, വലുപ്പം(max_bias));
    എങ്കിൽ (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int64_t n_expert      = logits->ne[0];
    സ്ഥിരം int64_t n_tokens      = logits->ne[1];
    സ്ഥിരം int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    എങ്കിൽ (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (with_norm) {
        സ്ഥിരം ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
        സ്ഥിരം ggml_tensor * sum_rows         = gf->nodes[raw_start + 6];
        സ്ഥിരം ggml_tensor * clamp            = gf->nodes[raw_start + 7];
        സ്ഥിരം ggml_tensor * div              = gf->nodes[raw_start + 8];
        സ്ഥിരം ggml_tensor * out_reshaped     = gf->nodes[raw_start + 9];

        എങ്കിൽ (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            clamp->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            out_reshaped->src[0] != div) {
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (with_scale) {
            സ്ഥിരം ggml_tensor * scale_കണ്ണി = gf->nodes[raw_start + 10];
            എങ്കിൽ (scale_കണ്ണി->src[0] != out_reshaped) {
                തിരികെ അസത്യം;
            }
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (with_scale) {
        സ്ഥിരം ggml_tensor * scale_കണ്ണി = gf->nodes[raw_start + 5];
        എങ്കിൽ (scale_കണ്ണി->src[0] != get_rows) {
            തിരികെ അസത്യം;
        }
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ outputs[2] = { raw_start + 3, raw_end };
    എങ്കിൽ (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_എണ്ണം, ops_all.data(), outputs, 2)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (mode == GGML_METAL_FUSION_FULL) {
        എങ്കിൽ (!logits->data || !out->data || !ids->data) {
            തിരികെ അസത്യം;
        }
    }

    തിരികെ സത്യം;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

ഘടന ggml_metal_moe_reduce_match {
    സ്ഥിരം ggml_tensor * experts;
    സ്ഥിരം ggml_tensor * weights;
    സ്ഥിരം ggml_tensor * dst;
    പൂർണ്ണസംഖ്യ കണ്ണി_എണ്ണം;
};

സ്ഥിതം തർക്കം ggml_metal_fusion_match_moe_reduce(
        സ്ഥിരം ggml_cgraph * gf, പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    എങ്കിൽ (match == nullptr || കണ്ണി_idx < 0 || കണ്ണി_idx + (പൂർണ്ണസംഖ്യ) ops_all.size() > gf->n_nodes) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം ggml_tensor * mul = gf->nodes[കണ്ണി_idx];
    എങ്കിൽ (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    സ്ഥിരം പൂർണ്ണസംഖ്യ raw_എണ്ണം     = (പൂർണ്ണസംഖ്യ) ops_all.size();
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_expert_used = raw_എണ്ണം / 2;

    എങ്കിൽ (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_എണ്ണം != 2 * n_expert_used) {
        തിരികെ അസത്യം;
    }

    പൂർണ്ണസംഖ്യ n_views = 0;
    ആകുമ്പോൾ (കണ്ണി_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[കണ്ണി_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    എങ്കിൽ (n_views != n_expert_used) {
        തിരികെ അസത്യം;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = n_expert_used + 1; i < raw_എണ്ണം; ++i) {
        എങ്കിൽ (gf->nodes[കണ്ണി_idx + i]->op != GGML_OP_ADD) {
            തിരികെ അസത്യം;
        }
    }

    പൂർണ്ണസംഖ്യ raw_idxs[GGML_METAL_FUSION_MAX];
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < raw_എണ്ണം; ++i) {
        raw_idxs[i] = കണ്ണി_idx + i;
        എങ്കിൽ (gf->nodes[കണ്ണി_idx + i]->op != ops_all[i]) {
            തിരികെ അസത്യം;
        }
    }

    സ്ഥിരം ggml_tensor * experts = mul->src[0];
    സ്ഥിരം ggml_tensor * weights = mul->src[1];
    സ്ഥിരം ggml_tensor * dst     = gf->nodes[കണ്ണി_idx + raw_എണ്ണം - 1];

    എങ്കിൽ (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int64_t n_embd   = experts->ne[0];
    സ്ഥിരം int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    എങ്കിൽ (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        തിരികെ അസത്യം;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 1; i <= n_expert_used; ++i) {
        സ്ഥിരം ggml_tensor * view = gf->nodes[കണ്ണി_idx + i];
        എങ്കിൽ (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            തിരികെ അസത്യം;
        }
    }

    സ്ഥിരം ggml_tensor * prev_add = nullptr;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 1; j < n_expert_used; ++j) {
        സ്ഥിരം ggml_tensor * add = gf->nodes[കണ്ണി_idx + n_expert_used + j];
        സ്ഥിരം ggml_tensor * rhs = gf->nodes[കണ്ണി_idx + j + 1];
        സ്ഥിരം ggml_tensor * lhs = j == 1 ? gf->nodes[കണ്ണി_idx + 1] : prev_add;
        എങ്കിൽ (add->src[0] != lhs || add->src[1] != rhs) {
            തിരികെ അസത്യം;
        }
        prev_add = add;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ outputs[1] = { കണ്ണി_idx + raw_എണ്ണം - 1 };
    എങ്കിൽ (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_എണ്ണം, ops_all.data(), outputs, 1)) {
        തിരികെ അസത്യം;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->കണ്ണി_എണ്ണം = raw_എണ്ണം;
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം ggml_metal_fusion_check_moe_reduce(
        സ്ഥിരം ggml_metal_fusion       * fusion,
        സ്ഥിരം ggml_tensor * സ്ഥിരം     * nodes,
        സ്ഥിരം ggml_cgraph             * gf,
        സ്ഥിരം പൂർണ്ണസംഖ്യ                     * കണ്ണി_idxs,
              പൂർണ്ണസംഖ്യ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    എങ്കിൽ (!ggml_metal_fusion_match_moe_reduce(gf, കണ്ണി_idxs[idx], fusion->ops_all, &match)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ ((പൂർണ്ണസംഖ്യ) fusion->ops.size() != match.experts->ne[1]) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ raw_end = കണ്ണി_idxs[idx] + match.കണ്ണി_എണ്ണം - 1;
    എങ്കിൽ (കണ്ണി_idxs[idx + (പൂർണ്ണസംഖ്യ) fusion->ops.size() - 1] != raw_end) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (mode == GGML_METAL_FUSION_FULL) {
        എങ്കിൽ (!match.experts->data || !match.weights->data || !match.dst->data) {
            തിരികെ അസത്യം;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        എങ്കിൽ (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            തിരികെ അസത്യം;
        }
    }

    തിരികെ സത്യം;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
സ്ഥിതം തർക്കം ggml_metal_tensor_is_weight(സ്ഥിരം ഘടന ggml_tensor * t) {
    സ്ഥിരം ggml_tensor * base = t->view_src != ശൂന്യ ? t->view_src : t;

    തിരികെ base->buffer != ശൂന്യ && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_tensor * ggml_metal_mul_mat_add_operand(സ്ഥിരം ഘടന ggml_tensor * mm, സ്ഥിരം ഘടന ggml_tensor * add) {
    എങ്കിൽ (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        തിരികെ ശൂന്യ;
    }

    സ്ഥിരം ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    സ്ഥിരം തർക്കം ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    തിരികെ ok ? other : ശൂന്യ;
}

സ്ഥിതം സ്ഥിരം ഘടന ggml_tensor * ggml_metal_mul_mat_add_residual(സ്ഥിരം ഘടന ggml_tensor * mm, സ്ഥിരം ഘടന ggml_tensor * add) {
    സ്ഥിരം ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    സ്ഥിരം തർക്കം ok = res != ശൂന്യ && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    തിരികെ ok ? res : ശൂന്യ;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
സ്ഥിതം തർക്കം ggml_metal_fusion_check_mul_mat_add(
        സ്ഥിരം ggml_metal_fusion       * fusion,
        സ്ഥിരം ggml_tensor * സ്ഥിരം     * nodes,
        സ്ഥിരം ggml_cgraph             * gf,
        സ്ഥിരം പൂർണ്ണസംഖ്യ                     * കണ്ണി_idxs,
              പൂർണ്ണസംഖ്യ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(കണ്ണി_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    സ്ഥിരം ggml_tensor * mm  = nodes[0];
    സ്ഥിരം ggml_tensor * add = nodes[1];

    എങ്കിൽ (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        തിരികെ അസത്യം;
    }

    എങ്കിൽ (mode == GGML_METAL_FUSION_STRUCTURAL) {
        തിരികെ സത്യം;
    }

    സ്ഥിരം ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    എങ്കിൽ (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        തിരികെ അസത്യം;
    }

    തിരികെ !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

സ്ഥിതം സ്ഥിരം std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };

സ്ഥിതം സ്ഥിരം std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     അസത്യം, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_norm_mul_add,           {},     അസത്യം, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     അസത്യം, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     അസത്യം, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_rms_norm_mul_add,       {},     അസത്യം, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     അസത്യം, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_2,                  {},     അസത്യം, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_3,                  {},     അസത്യം, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_4,                  {},     അസത്യം, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_5,                  {},     അസത്യം, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_6,                  {},     അസത്യം, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_7,                  {},     അസത്യം, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     അസത്യം, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     സത്യം,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    സത്യം,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    സത്യം,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    സത്യം,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    സത്യം,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     സത്യം,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     സത്യം,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     സത്യം,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     സത്യം,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     സത്യം,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     സത്യം,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     സത്യം,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     അസത്യം, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_ADD,    ops_mul_mat_add,            {},     അസത്യം, ggml_metal_fusion_check_mul_mat_add },
};

// ---- alloc deps -----------------------------------------------------------

സ്ഥിതം തർക്കം ggml_metal_fusion_match_raw_pattern(
        സ്ഥിരം ggml_cgraph * gf, പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::vector<ggml_op> & ops) {
    എങ്കിൽ (കണ്ണി_idx < 0 || കണ്ണി_idx + (പൂർണ്ണസംഖ്യ) ops.size() > gf->n_nodes) {
        തിരികെ അസത്യം;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < (പൂർണ്ണസംഖ്യ) ops.size(); ++i) {
        എങ്കിൽ (gf->nodes[കണ്ണി_idx + i]->op != ops[i]) {
            തിരികെ അസത്യം;
        }
    }

    തിരികെ സത്യം;
}

സ്ഥിതം ശൂന്യം ggml_metal_fusion_add_pattern_alloc_deps(
        ശൂന്യം * user_data,
        ശൂന്യം (*add_alloc_dep)(ശൂന്യം *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        സ്ഥിരം ggml_metal_fusion * fusion,
        പൂർണ്ണസംഖ്യ കണ്ണി_idx) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ last_കണ്ണി = കണ്ണി_idx + (പൂർണ്ണസംഖ്യ) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < (പൂർണ്ണസംഖ്യ) fusion->ops_all.size(); ++j) {
        ggml_tensor * കണ്ണി = gf->nodes[കണ്ണി_idx + j];
        വേണ്ടി (പൂർണ്ണസംഖ്യ s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = കണ്ണി->src[s];
            എങ്കിൽ (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_കണ്ണി]);
            }
        }
        seen.insert(കണ്ണി);
    }
}

ശൂന്യം ggml_metal_fusion_add_alloc_deps(
        ശൂന്യം * user_data,
        ശൂന്യം (*add_alloc_dep)(ശൂന്യം *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < gf->n_nodes; ++i) {
        സ്ഥിരം ggml_metal_fusion * best = nullptr;
        പൂർണ്ണസംഖ്യ best_raw = 0;

        വേണ്ടി (സ്ഥിരം ggml_metal_fusion & fusion : ggml_metal_fusions) {
            എങ്കിൽ ((പൂർണ്ണസംഖ്യ) fusion.ops_all.size() <= best_raw) {
                തുടരുക;
            }
            എങ്കിൽ (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (പൂർണ്ണസംഖ്യ) fusion.ops_all.size();
            }
        }

        എങ്കിൽ (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

സ്ഥിതം std::string ggml_metal_fusion_label(സ്ഥിരം ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < (പൂർണ്ണസംഖ്യ) fusion->ops.size(); j++) {
        എങ്കിൽ (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    തിരികെ label;
}

ഘടന ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    തർക്കം enabled;
    തർക്കം stats;
    തർക്കം labels_set;
    പൂർണ്ണസംഖ്യ  debug;
};

ഘടന ggml_metal_fusion_info * ggml_metal_fusion_info_init(തർക്കം enabled, പൂർണ്ണസംഖ്യ debug) {
    ggml_metal_fusion_info * finfo = പുതിയ ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = അസത്യം;
    finfo->debug      = debug;

    എങ്കിൽ (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    തിരികെ finfo;
}

ശൂന്യം ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

തർക്കം ggml_metal_fusion_info_enabled(സ്ഥിരം ggml_metal_fusion_info * finfo) {
    തിരികെ finfo->enabled;
}

തർക്കം ggml_metal_fusion_info_stats(സ്ഥിരം ggml_metal_fusion_info * finfo) {
    തിരികെ finfo->stats;
}

പൂർണ്ണസംഖ്യ ggml_metal_fusion_info_debug(സ്ഥിരം ggml_metal_fusion_info * finfo) {
    തിരികെ finfo->debug;
}

പൂർണ്ണസംഖ്യ ggml_metal_fusion_info_n_fusions(സ്ഥിരം ggml_metal_fusion_info * finfo) {
    തിരികെ (പൂർണ്ണസംഖ്യ) finfo->labels.size();
}

സ്ഥിരം അക്ഷരം * ggml_metal_fusion_info_label(സ്ഥിരം ggml_metal_fusion_info * finfo, പൂർണ്ണസംഖ്യ idx) {
    GGML_ASSERT(idx >= 0 && idx < (പൂർണ്ണസംഖ്യ) finfo->labels.size());
    തിരികെ finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_എണ്ണം(സ്ഥിരം ggml_metal_fusion_info * finfo, പൂർണ്ണസംഖ്യ idx) {
    GGML_ASSERT(idx >= 0 && idx < (പൂർണ്ണസംഖ്യ) finfo->counts.size());
    തിരികെ finfo->counts[idx];
}

ശൂന്യം ggml_metal_fusion_info_എണ്ണം_fusion(ggml_metal_fusion_info * finfo, സ്ഥിരം ggml_metal_fusion * fusion) {
    എങ്കിൽ (!finfo->stats || fusion == nullptr) {
        തിരികെ;
    }

    സ്ഥിരം ptrdiff_t idx = fusion - ggml_metal_fusions.data();
    എങ്കിൽ (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

ശൂന്യം ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, തർക്കം enabled) {
    finfo->enabled = enabled;
}

ശൂന്യം ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    എങ്കിൽ (finfo->labels_set) {
        തിരികെ;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    വേണ്ടി (സ്ഥിരം ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = സത്യം;
}

ശൂന്യം ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = സത്യം;
    ggml_metal_fusion_info_labels_init(finfo);
}

ശൂന്യം ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

പൂർണ്ണസംഖ്യ ggml_metal_fusion_info_stats_get(സ്ഥിരം ggml_metal_fusion_info * finfo, സ്ഥിരം അക്ഷരം ** labels, uint64_t * counts, പൂർണ്ണസംഖ്യ n) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_fusions = (പൂർണ്ണസംഖ്യ) finfo->labels.size();

    എങ്കിൽ (labels == nullptr) {
        തിരികെ n_fusions;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ n_fill = std::min(n, n_fusions);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        എങ്കിൽ (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    തിരികെ n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
സ്ഥിതം തർക്കം ggml_metal_fusion_check_memory_ranges(
        സ്ഥിരം ggml_metal_fusion      * fusion,
        സ്ഥിരം ggml_tensor * സ്ഥിരം    * nodes,
        പൂർണ്ണസംഖ്യ                            കണ്ണി_എണ്ണം) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](സ്ഥിരം ggml_tensor * a, സ്ഥിരം ggml_tensor * b) {
        എങ്കിൽ (a->data && b->data && a->data == b->data) {
            തിരികെ സത്യം;
        }
        വേണ്ടി (സ്ഥിരം ggml_tensor * v = a; v; v = v->view_src) {
            എങ്കിൽ (v == b) {
                തിരികെ സത്യം;
            }
        }
        വേണ്ടി (സ്ഥിരം ggml_tensor * v = b; v; v = v->view_src) {
            എങ്കിൽ (v == a) {
                തിരികെ സത്യം;
            }
        }
        തിരികെ അസത്യം;
    };

    auto nodes_overlap = [](സ്ഥിരം ggml_tensor * a, സ്ഥിരം ggml_tensor * b) {
        എങ്കിൽ (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            തിരികെ അസത്യം;
        }

        എങ്കിൽ (a->buffer != b->buffer) {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം int64_t a_start = (int64_t) a->data;
        സ്ഥിരം int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        സ്ഥിരം int64_t b_start = (int64_t) b->data;
        സ്ഥിരം int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        തിരികെ (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](സ്ഥിരം ggml_tensor * src, സ്ഥിരം ggml_tensor * സ്ഥിരം * nodes, പൂർണ്ണസംഖ്യ j) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < j; ++k) {
            എങ്കിൽ (src == nodes[k]) {
                തിരികെ സത്യം;
            }
            വേണ്ടി (സ്ഥിരം ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                എങ്കിൽ (view_src == nodes[k]) {
                    തിരികെ സത്യം;
                }
            }
        }
        തിരികെ അസത്യം;
    };

    auto check_dst = [&](സ്ഥിരം ggml_tensor * dst) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < കണ്ണി_എണ്ണം; ++j) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ s = 0; s < GGML_MAX_SRC; ++s) {
                സ്ഥിരം ggml_tensor * src = nodes[j]->src[s];
                എങ്കിൽ (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    തുടരുക;
                }

                എങ്കിൽ (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    തിരികെ അസത്യം;
                }
            }
        }
        തിരികെ സത്യം;
    };

    എങ്കിൽ (!check_dst(nodes[കണ്ണി_എണ്ണം - 1])) {
        തിരികെ അസത്യം;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < കണ്ണി_എണ്ണം);
        എങ്കിൽ (!check_dst(nodes[offset])) {
            തിരികെ അസത്യം;
        }
    }

    തിരികെ സത്യം;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
സ്ഥിരം ggml_metal_fusion * ggml_metal_fusion_next(
        സ്ഥിരം ggml_cgraph * gf,
        സ്ഥിരം പൂർണ്ണസംഖ്യ * കണ്ണി_idxs,
        പൂർണ്ണസംഖ്യ n_idxs,
        പൂർണ്ണസംഖ്യ idx,
        ggml_metal_fusion_mode mode,
        പൂർണ്ണസംഖ്യ * n_out) {
    സ്ഥിരം ggml_metal_fusion * res = nullptr;
    പൂർണ്ണസംഖ്യ best = 1;

    വേണ്ടി (സ്ഥിരം ggml_metal_fusion & fusion : ggml_metal_fusions) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ n_ops = (പൂർണ്ണസംഖ്യ) fusion.ops.size();

        // only look for a longer match than the current best
        എങ്കിൽ (n_ops <= best) {
            തുടരുക;
        }
        എങ്കിൽ (idx + n_ops > n_idxs) {
            തുടരുക;
        }

        സ്ഥിരം ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        തർക്കം ok = സത്യം;
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[കണ്ണി_idxs[idx + j]];
            എങ്കിൽ (nodes[j]->op != fusion.ops[j]) {
                ok = അസത്യം;
                നിർത്തുക;
            }
        }
        എങ്കിൽ (!ok) {
            തുടരുക;
        }

        എങ്കിൽ (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 1; j < n_ops && ok; j++) {
                എങ്കിൽ (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = അസത്യം;
                    നിർത്തുക;
                }
                എങ്കിൽ (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = അസത്യം;
                    നിർത്തുക;
                }
            }
            എങ്കിൽ (!ok) {
                തുടരുക;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            പൂർണ്ണസംഖ്യ outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = കണ്ണി_idxs[idx + n_ops - 1];
            വേണ്ടി (size_t i = 0; i < fusion.outs.size(); ++i) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = കണ്ണി_idxs[idx + out_offset];
            }

            സ്ഥിരം പൂർണ്ണസംഖ്യ n_outputs = 1 + (പൂർണ്ണസംഖ്യ) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            എങ്കിൽ (!ggml_can_fuse_subgraph_ext(gf, കണ്ണി_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                തുടരുക;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        എങ്കിൽ (fusion.check && !fusion.check(&fusion, nodes, gf, കണ്ണി_idxs, idx, mode)) {
            തുടരുക;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        എങ്കിൽ (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            തുടരുക;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    തിരികെ res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
പൂർണ്ണസംഖ്യ ggml_metal_fusion_max(സ്ഥിരം ggml_cgraph * gf, പൂർണ്ണസംഖ്യ idx) {
    // a view node cannot start a pattern - pack it alone
    എങ്കിൽ (ggml_op_is_empty(gf->nodes[idx]->op)) {
        തിരികെ 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    പൂർണ്ണസംഖ്യ idxs[GGML_METAL_FUSION_MAX];
    പൂർണ്ണസംഖ്യ n_idxs = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        എങ്കിൽ (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    എങ്കിൽ (n_idxs == 0) {
        തിരികെ 1;
    }

    പൂർണ്ണസംഖ്യ total = 0;
    പൂർണ്ണസംഖ്യ i_f = 0;

    ആകുമ്പോൾ (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        പൂർണ്ണസംഖ്യ len = 1;
        സ്ഥിരം ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        എങ്കിൽ (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            നിർത്തുക;
        }

        total += len;
        i_f += len;
    }

    എങ്കിൽ (i_f == 0) {
        തിരികെ 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    തിരികെ std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
