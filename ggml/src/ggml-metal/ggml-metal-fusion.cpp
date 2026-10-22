#ضمن "ggml-metal-fusion.h"

#ضمن "ggml-metal-common.h"
#ضمن "ggml-metal-device.h"

#ضمن "ggml-backend-impl.h"

#ضمن <algorithm>
#ضمن <cstddef>
#ضمن <cstring>
#ضمن <set>
#ضمن <string>
#ضمن <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
ساكن std::vector<ggml_op> ggml_metal_fusion_filter_ops(ثابت std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    עבור (ggml_op op : ops_all) {
        אם (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    החזר ops;
}

بنية ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<صحيح>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    منطقي unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    منطقي (*check)(ثابت بنية ggml_metal_fusion   * fusion,
                  ثابت بنية ggml_tensor * ثابت * nodes,
                  ثابت بنية ggml_cgraph         * gf,
                  ثابت صحيح                        * node_idxs,
                        صحيح                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            ثابت std::vector<ggml_op> & ops_all,
            ثابت std::vector<صحيح> & outs,
            منطقي unsafe,
            منطقي (*check)(ثابت بنية ggml_metal_fusion   * fusion,
                          ثابت بنية ggml_tensor * ثابت * nodes,
                          ثابت بنية ggml_cgraph         * gf,
                          ثابت صحيح                        * node_idxs,
                                صحيح                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(ثابت ggml_metal_fusion * fusion) {
    החזר fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
ساكن منطقي ggml_metal_fusion_same_buffer(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    אם (!a || !b) {
        החזר שקר;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    החזר ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
ساكن منطقي ggml_metal_fusion_overlap(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ثابت ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    ثابت ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    אם (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        החזר שקר;
    }

    החזר bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
ساكن منطقي ggml_metal_fusion_check_norm(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحيح                     * node_idxs,
              صحيح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    אם (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        ثابت ggml_tensor * scale = nodes[1];
        אם (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            החזר שקר;
        }

        החזר אמת;
    }

    עבור (صحيح j = 1; j < (صحيح) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        אם (nodes[j]->src[0] != nodes[j - 1]) {
            החזר שקר;
        }

        // the weight/bias must have the same row width as the norm input
        אם (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            החזר שקר;
        }

        אם (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            החזר שקר;
        }

        אם (nodes[j]->type != GGML_TYPE_F32) {
            החזר שקר;
        }
    }

    החזר אמת;
}

// SSM_CONV + UNARY (silu)
ساكن منطقي ggml_metal_fusion_check_ssm_conv_silu(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحيح                     * node_idxs,
              صحيح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    ثابت ggml_tensor * conv = nodes[0];
    ثابت ggml_tensor * un   = nodes[1];

    אם (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        החזר שקר;
    }

    אם (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        החזר שקר;
    }

    אם (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        החזר שקר;
    }

    החזר אמת;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
ساكن منطقي ggml_metal_fusion_check_add_chain(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحيح                     * node_idxs,
              صحيح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    עבור (صحيح j = 1; j < (صحيح) fusion->ops.size(); j++) {
        אם (nodes[j]->src[0] != nodes[j - 1]) {
            החזר שקר;
        }

        אם (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            החזר שקר;
        }

        אם (mode == GGML_METAL_FUSION_FULL) {
            אם (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                החזר שקר;
            }
        }
    }

    החזר אמת;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
ساكن منطقي ggml_metal_fusion_check_gdn_cache(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحيح                     * node_idxs,
              صحيح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    ثابت ggml_tensor * gdn = nodes[0];
    ثابت ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    אם (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        החזר שקר;
    }

    אם (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        החזר שקר;
    }

    ثابت int64_t S_v      = gdn->src[2]->ne[0];
    ثابت int64_t H        = gdn->src[2]->ne[1];
    ثابت int64_t n_tokens = gdn->src[2]->ne[2];
    ثابت int64_t n_seqs   = gdn->src[2]->ne[3];
    ثابت int64_t K        = ggml_get_op_params_i32(gdn, 0);
    ثابت size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    ثابت int64_t D         = S_v * S_v * H;
    ثابت int64_t n_written = std::min<int64_t>(n_tokens, K);

    ثابت ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    ثابت ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    אם (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        החזר שקר;
    }

    ثابت int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    אם (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        החזר שקר;
    }

    אם (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        אם (dst->data == nullptr) {
            החזר שקר;
        }
    }

    החזר אמת;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
ساكن منطقي ggml_metal_fusion_check_snake(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحيح                     * node_idxs,
              صحيح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    ثابت ggml_tensor * mul0     = nodes[0];
    ثابت ggml_tensor * sin_node = nodes[1];
    ثابت ggml_tensor * sqr      = nodes[2];
    ثابت ggml_tensor * mul1     = nodes[3];
    ثابت ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    ثابت ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ثابت ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ثابت ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    ثابت ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    ثابت منطقي types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_node->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    ثابت منطقي shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    ثابت منطقي dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    ثابت منطقي contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    החזר types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
ساكن ثابت std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
ساكن ثابت std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
ساكن ثابت std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
ساكن ثابت std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

ساكن منطقي ggml_metal_fusion_check_topk_moe(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحيح                     * node_idxs,
              صحيح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    ثابت صحيح n_ops = (صحيح) fusion->ops.size();

    ثابت منطقي with_norm  = n_ops >= 6;
    ثابت منطقي with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    ثابت std::vector<ggml_op> & ops_all = fusion->ops_all;

    ثابت صحيح raw_start = node_idxs[idx];
    صحيح       raw_end   = node_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    אם (with_norm && !with_scale) {
        אם (raw_end + 1 >= gf->n_nodes) {
            החזר שקר;
        }
        ثابت ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        אם (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            החזר שקר;
        }
        raw_end++;
    }

    ثابت صحيح raw_count = raw_end - raw_start + 1;
    אם (raw_count != (صحيح) ops_all.size()) {
        החזר שקר;
    }

    صحيح raw_idxs[GGML_METAL_FUSION_MAX];
    עבור (صحيح i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        אם (gf->nodes[raw_start + i]->op != ops_all[i]) {
            החזר שקר;
        }
    }

    ثابت ggml_tensor * softmax        = gf->nodes[raw_start];
    ثابت ggml_tensor * probs_reshaped = gf->nodes[raw_start + 1];
    ثابت ggml_tensor * argsort        = gf->nodes[raw_start + 2];
    ثابت ggml_tensor * ids            = gf->nodes[raw_start + 3];
    ثابت ggml_tensor * get_rows       = gf->nodes[raw_start + 4];
    ثابت ggml_tensor * out            = gf->nodes[raw_end];
    ثابت ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    عائم scale   = 1.0f;
    عائم max_bias = 0.0f;
    memcpy(&scale,    ((ثابت int32_t *) softmax->op_params) + 0, حجم(scale));
    memcpy(&max_bias, ((ثابت int32_t *) softmax->op_params) + 1, حجم(max_bias));
    אם (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        החזר שקר;
    }

    אם (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        החזר שקר;
    }

    ثابت int64_t n_expert      = logits->ne[0];
    ثابت int64_t n_tokens      = logits->ne[1];
    ثابت int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    אם (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        החזר שקר;
    }

    אם (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        החזר שקר;
    }

    אם (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        החזר שקר;
    }

    אם (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        החזר שקר;
    }

    אם (with_norm) {
        ثابت ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
        ثابت ggml_tensor * sum_rows         = gf->nodes[raw_start + 6];
        ثابت ggml_tensor * clamp            = gf->nodes[raw_start + 7];
        ثابت ggml_tensor * div              = gf->nodes[raw_start + 8];
        ثابت ggml_tensor * out_reshaped     = gf->nodes[raw_start + 9];

        אם (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            clamp->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            out_reshaped->src[0] != div) {
            החזר שקר;
        }

        אם (with_scale) {
            ثابت ggml_tensor * scale_node = gf->nodes[raw_start + 10];
            אם (scale_node->src[0] != out_reshaped) {
                החזר שקר;
            }
        }
    } אחרת אם (with_scale) {
        ثابت ggml_tensor * scale_node = gf->nodes[raw_start + 5];
        אם (scale_node->src[0] != get_rows) {
            החזר שקר;
        }
    }

    ثابت صحيح outputs[2] = { raw_start + 3, raw_end };
    אם (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 2)) {
        החזר שקר;
    }

    אם (mode == GGML_METAL_FUSION_FULL) {
        אם (!logits->data || !out->data || !ids->data) {
            החזר שקר;
        }
    }

    החזר אמת;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

بنية ggml_metal_moe_reduce_match {
    ثابت ggml_tensor * experts;
    ثابت ggml_tensor * weights;
    ثابت ggml_tensor * dst;
    صحيح node_count;
};

ساكن منطقي ggml_metal_fusion_match_moe_reduce(
        ثابت ggml_cgraph * gf, صحيح node_idx, ثابت std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    אם (match == nullptr || node_idx < 0 || node_idx + (صحيح) ops_all.size() > gf->n_nodes) {
        החזר שקר;
    }

    ثابت ggml_tensor * mul = gf->nodes[node_idx];
    אם (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        החזר שקר;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    ثابت صحيح raw_count     = (صحيح) ops_all.size();
    ثابت صحيح n_expert_used = raw_count / 2;

    אם (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_count != 2 * n_expert_used) {
        החזר שקר;
    }

    صحيح n_views = 0;
    כלעוד (node_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[node_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    אם (n_views != n_expert_used) {
        החזר שקר;
    }

    עבור (صحيح i = n_expert_used + 1; i < raw_count; ++i) {
        אם (gf->nodes[node_idx + i]->op != GGML_OP_ADD) {
            החזר שקר;
        }
    }

    صحيح raw_idxs[GGML_METAL_FUSION_MAX];
    עבור (صحيح i = 0; i < raw_count; ++i) {
        raw_idxs[i] = node_idx + i;
        אם (gf->nodes[node_idx + i]->op != ops_all[i]) {
            החזר שקר;
        }
    }

    ثابت ggml_tensor * experts = mul->src[0];
    ثابت ggml_tensor * weights = mul->src[1];
    ثابت ggml_tensor * dst     = gf->nodes[node_idx + raw_count - 1];

    אם (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        החזר שקר;
    }

    ثابت int64_t n_embd   = experts->ne[0];
    ثابت int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    אם (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        החזר שקר;
    }

    אם (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        החזר שקר;
    }

    עבור (صحيح i = 1; i <= n_expert_used; ++i) {
        ثابت ggml_tensor * view = gf->nodes[node_idx + i];
        אם (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            החזר שקר;
        }
    }

    ثابت ggml_tensor * prev_add = nullptr;
    עבור (صحيح j = 1; j < n_expert_used; ++j) {
        ثابت ggml_tensor * add = gf->nodes[node_idx + n_expert_used + j];
        ثابت ggml_tensor * rhs = gf->nodes[node_idx + j + 1];
        ثابت ggml_tensor * lhs = j == 1 ? gf->nodes[node_idx + 1] : prev_add;
        אם (add->src[0] != lhs || add->src[1] != rhs) {
            החזר שקר;
        }
        prev_add = add;
    }

    ثابت صحيح outputs[1] = { node_idx + raw_count - 1 };
    אם (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 1)) {
        החזר שקר;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->node_count = raw_count;
    החזר אמת;
}

ساكن منطقي ggml_metal_fusion_check_moe_reduce(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحيح                     * node_idxs,
              صحيح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    אם (!ggml_metal_fusion_match_moe_reduce(gf, node_idxs[idx], fusion->ops_all, &match)) {
        החזר שקר;
    }

    אם ((صحيح) fusion->ops.size() != match.experts->ne[1]) {
        החזר שקר;
    }

    ثابت صحيح raw_end = node_idxs[idx] + match.node_count - 1;
    אם (node_idxs[idx + (صحيح) fusion->ops.size() - 1] != raw_end) {
        החזר שקר;
    }

    אם (mode == GGML_METAL_FUSION_FULL) {
        אם (!match.experts->data || !match.weights->data || !match.dst->data) {
            החזר שקר;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        אם (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            החזר שקר;
        }
    }

    החזר אמת;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
ساكن منطقي ggml_metal_tensor_is_weight(ثابت بنية ggml_tensor * t) {
    ثابت ggml_tensor * base = t->view_src != عدم ? t->view_src : t;

    החזר base->buffer != عدم && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

ساكن ثابت بنية ggml_tensor * ggml_metal_mul_mat_add_operand(ثابت بنية ggml_tensor * mm, ثابت بنية ggml_tensor * add) {
    אם (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        החזר عدم;
    }

    ثابت ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    ثابت منطقي ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    החזר ok ? other : عدم;
}

ساكن ثابت بنية ggml_tensor * ggml_metal_mul_mat_add_residual(ثابت بنية ggml_tensor * mm, ثابت بنية ggml_tensor * add) {
    ثابت ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    ثابت منطقي ok = res != عدم && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    החזר ok ? res : عدم;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
ساكن منطقي ggml_metal_fusion_check_mul_mat_add(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحيح                     * node_idxs,
              صحيح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    ثابت ggml_tensor * mm  = nodes[0];
    ثابت ggml_tensor * add = nodes[1];

    אם (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        החזר שקר;
    }

    אם (mode == GGML_METAL_FUSION_STRUCTURAL) {
        החזר אמת;
    }

    ثابت ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    אם (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        החזר שקר;
    }

    החזר !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

ساكن ثابت std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
ساكن ثابت std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
ساكن ثابت std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
ساكن ثابت std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
ساكن ثابت std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
ساكن ثابت std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

ساكن ثابت std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
ساكن ثابت std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ساكن ثابت std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ساكن ثابت std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ساكن ثابت std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ساكن ثابت std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ساكن ثابت std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

ساكن ثابت std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

ساكن ثابت std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

ساكن ثابت std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
ساكن ثابت std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
ساكن ثابت std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ساكن ثابت std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ساكن ثابت std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ساكن ثابت std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ساكن ثابت std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

ساكن ثابت std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };

ساكن ثابت std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     שקר, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_norm_mul_add,           {},     שקר, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     שקר, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     שקר, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_rms_norm_mul_add,       {},     שקר, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     שקר, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_2,                  {},     שקר, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_3,                  {},     שקר, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_4,                  {},     שקר, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_5,                  {},     שקר, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_6,                  {},     שקר, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_7,                  {},     שקר, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     שקר, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     אמת,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    אמת,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    אמת,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    אמת,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    אמת,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     אמת,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     אמת,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     אמת,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     אמת,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     אמת,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     אמת,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     אמת,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     שקר, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_ADD,    ops_mul_mat_add,            {},     שקר, ggml_metal_fusion_check_mul_mat_add },
};

// ---- alloc deps -----------------------------------------------------------

ساكن منطقي ggml_metal_fusion_match_raw_pattern(
        ثابت ggml_cgraph * gf, صحيح node_idx, ثابت std::vector<ggml_op> & ops) {
    אם (node_idx < 0 || node_idx + (صحيح) ops.size() > gf->n_nodes) {
        החזר שקר;
    }

    עבור (صحيح i = 0; i < (صحيح) ops.size(); ++i) {
        אם (gf->nodes[node_idx + i]->op != ops[i]) {
            החזר שקר;
        }
    }

    החזר אמת;
}

ساكن فراغ ggml_metal_fusion_add_pattern_alloc_deps(
        فراغ * user_data,
        فراغ (*add_alloc_dep)(فراغ *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        ثابت ggml_metal_fusion * fusion,
        صحيح node_idx) {
    ثابت صحيح last_node = node_idx + (صحيح) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    עבור (صحيح j = 0; j < (صحيح) fusion->ops_all.size(); ++j) {
        ggml_tensor * node = gf->nodes[node_idx + j];
        עבור (صحيح s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = node->src[s];
            אם (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_node]);
            }
        }
        seen.insert(node);
    }
}

فراغ ggml_metal_fusion_add_alloc_deps(
        فراغ * user_data,
        فراغ (*add_alloc_dep)(فراغ *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    עבור (صحيح i = 0; i < gf->n_nodes; ++i) {
        ثابت ggml_metal_fusion * best = nullptr;
        صحيح best_raw = 0;

        עבור (ثابت ggml_metal_fusion & fusion : ggml_metal_fusions) {
            אם ((صحيح) fusion.ops_all.size() <= best_raw) {
                استمر;
            }
            אם (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (صحيح) fusion.ops_all.size();
            }
        }

        אם (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

ساكن std::string ggml_metal_fusion_label(ثابت ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    עבור (صحيح j = 0; j < (صحيح) fusion->ops.size(); j++) {
        אם (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    החזר label;
}

بنية ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    منطقي enabled;
    منطقي stats;
    منطقي labels_set;
    صحيح  debug;
};

بنية ggml_metal_fusion_info * ggml_metal_fusion_info_init(منطقي enabled, صحيح debug) {
    ggml_metal_fusion_info * finfo = جديد ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = שקר;
    finfo->debug      = debug;

    אם (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    החזר finfo;
}

فراغ ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

منطقي ggml_metal_fusion_info_enabled(ثابت ggml_metal_fusion_info * finfo) {
    החזר finfo->enabled;
}

منطقي ggml_metal_fusion_info_stats(ثابت ggml_metal_fusion_info * finfo) {
    החזר finfo->stats;
}

صحيح ggml_metal_fusion_info_debug(ثابت ggml_metal_fusion_info * finfo) {
    החזר finfo->debug;
}

صحيح ggml_metal_fusion_info_n_fusions(ثابت ggml_metal_fusion_info * finfo) {
    החזר (صحيح) finfo->labels.size();
}

ثابت حرف * ggml_metal_fusion_info_label(ثابت ggml_metal_fusion_info * finfo, صحيح idx) {
    GGML_ASSERT(idx >= 0 && idx < (صحيح) finfo->labels.size());
    החזר finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_count(ثابت ggml_metal_fusion_info * finfo, صحيح idx) {
    GGML_ASSERT(idx >= 0 && idx < (صحيح) finfo->counts.size());
    החזר finfo->counts[idx];
}

فراغ ggml_metal_fusion_info_count_fusion(ggml_metal_fusion_info * finfo, ثابت ggml_metal_fusion * fusion) {
    אם (!finfo->stats || fusion == nullptr) {
        החזר;
    }

    ثابت ptrdiff_t idx = fusion - ggml_metal_fusions.data();
    אם (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

فراغ ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, منطقي enabled) {
    finfo->enabled = enabled;
}

فراغ ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    אם (finfo->labels_set) {
        החזר;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    עבור (ثابت ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = אמת;
}

فراغ ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = אמת;
    ggml_metal_fusion_info_labels_init(finfo);
}

فراغ ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

صحيح ggml_metal_fusion_info_stats_get(ثابت ggml_metal_fusion_info * finfo, ثابت حرف ** labels, uint64_t * counts, صحيح n) {
    ثابت صحيح n_fusions = (صحيح) finfo->labels.size();

    אם (labels == nullptr) {
        החזר n_fusions;
    }

    ثابت صحيح n_fill = std::min(n, n_fusions);
    עבור (صحيح i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        אם (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    החזר n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
ساكن منطقي ggml_metal_fusion_check_memory_ranges(
        ثابت ggml_metal_fusion      * fusion,
        ثابت ggml_tensor * ثابت    * nodes,
        صحيح                            node_count) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        אם (a->data && b->data && a->data == b->data) {
            החזר אמת;
        }
        עבור (ثابت ggml_tensor * v = a; v; v = v->view_src) {
            אם (v == b) {
                החזר אמת;
            }
        }
        עבור (ثابت ggml_tensor * v = b; v; v = v->view_src) {
            אם (v == a) {
                החזר אמת;
            }
        }
        החזר שקר;
    };

    auto nodes_overlap = [](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        אם (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            החזר שקר;
        }

        אם (a->buffer != b->buffer) {
            החזר שקר;
        }

        ثابت int64_t a_start = (int64_t) a->data;
        ثابت int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        ثابت int64_t b_start = (int64_t) b->data;
        ثابت int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        החזר (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](ثابت ggml_tensor * src, ثابت ggml_tensor * ثابت * nodes, صحيح j) {
        עבור (صحيح k = 0; k < j; ++k) {
            אם (src == nodes[k]) {
                החזר אמת;
            }
            עבור (ثابت ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                אם (view_src == nodes[k]) {
                    החזר אמת;
                }
            }
        }
        החזר שקר;
    };

    auto check_dst = [&](ثابت ggml_tensor * dst) {
        עבור (صحيح j = 0; j < node_count; ++j) {
            עבור (صحيح s = 0; s < GGML_MAX_SRC; ++s) {
                ثابت ggml_tensor * src = nodes[j]->src[s];
                אם (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    استمر;
                }

                אם (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    החזר שקר;
                }
            }
        }
        החזר אמת;
    };

    אם (!check_dst(nodes[node_count - 1])) {
        החזר שקר;
    }

    עבור (صحيح offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < node_count);
        אם (!check_dst(nodes[offset])) {
            החזר שקר;
        }
    }

    החזר אמת;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
ثابت ggml_metal_fusion * ggml_metal_fusion_next(
        ثابت ggml_cgraph * gf,
        ثابت صحيح * node_idxs,
        صحيح n_idxs,
        صحيح idx,
        ggml_metal_fusion_mode mode,
        صحيح * n_out) {
    ثابت ggml_metal_fusion * res = nullptr;
    صحيح best = 1;

    עבור (ثابت ggml_metal_fusion & fusion : ggml_metal_fusions) {
        ثابت صحيح n_ops = (صحيح) fusion.ops.size();

        // only look for a longer match than the current best
        אם (n_ops <= best) {
            استمر;
        }
        אם (idx + n_ops > n_idxs) {
            استمر;
        }

        ثابت ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        منطقي ok = אמת;
        עבור (صحيح j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[node_idxs[idx + j]];
            אם (nodes[j]->op != fusion.ops[j]) {
                ok = שקר;
                اكسر;
            }
        }
        אם (!ok) {
            استمر;
        }

        אם (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            עבור (صحيح j = 1; j < n_ops && ok; j++) {
                אם (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = שקר;
                    اكسر;
                }
                אם (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = שקר;
                    اكسر;
                }
            }
            אם (!ok) {
                استمر;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            صحيح outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = node_idxs[idx + n_ops - 1];
            עבור (size_t i = 0; i < fusion.outs.size(); ++i) {
                ثابت صحيح out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = node_idxs[idx + out_offset];
            }

            ثابت صحيح n_outputs = 1 + (صحيح) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            אם (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                استمر;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        אם (fusion.check && !fusion.check(&fusion, nodes, gf, node_idxs, idx, mode)) {
            استمر;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        אם (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            استمر;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    החזר res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
صحيح ggml_metal_fusion_max(ثابت ggml_cgraph * gf, صحيح idx) {
    // a view node cannot start a pattern - pack it alone
    אם (ggml_op_is_empty(gf->nodes[idx]->op)) {
        החזר 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    صحيح idxs[GGML_METAL_FUSION_MAX];
    صحيح n_idxs = 0;
    עבור (صحيح i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        אם (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    אם (n_idxs == 0) {
        החזר 1;
    }

    صحيح total = 0;
    صحيح i_f = 0;

    כלעוד (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        صحيح len = 1;
        ثابت ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        אם (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            اكسر;
        }

        total += len;
        i_f += len;
    }

    אם (i_f == 0) {
        החזר 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    החזר std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
