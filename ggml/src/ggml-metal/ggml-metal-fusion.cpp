#شامل "ggml-metal-fusion.h"

#شامل "ggml-metal-common.h"
#شامل "ggml-metal-device.h"

#شامل "ggml-backend-impl.h"

#شامل <algorithm>
#شامل <cstddef>
#شامل <cstring>
#شامل <set>
#شامل <string>
#شامل <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
ایستا std::vector<ggml_op> ggml_metal_fusion_filter_ops(ثابت std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    برای (ggml_op op : ops_all) {
        اگر (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    بازگشت ops;
}

ساختار ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<صحیح>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    منطقی unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    منطقی (*check)(ثابت ساختار ggml_metal_fusion   * fusion,
                  ثابت ساختار ggml_tensor * ثابت * nodes,
                  ثابت ساختار ggml_cgraph         * gf,
                  ثابت صحیح                        * گره_idxs,
                        صحیح                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            ثابت std::vector<ggml_op> & ops_all,
            ثابت std::vector<صحیح> & outs,
            منطقی unsafe,
            منطقی (*check)(ثابت ساختار ggml_metal_fusion   * fusion,
                          ثابت ساختار ggml_tensor * ثابت * nodes,
                          ثابت ساختار ggml_cgraph         * gf,
                          ثابت صحیح                        * گره_idxs,
                                صحیح                          idx,
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
    بازگشت fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
ایستا منطقی ggml_metal_fusion_same_buffer(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    اگر (!a || !b) {
        بازگشت نادرست;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    بازگشت ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
ایستا منطقی ggml_metal_fusion_overlap(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ثابت ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    ثابت ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    اگر (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        بازگشت نادرست;
    }

    بازگشت bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
ایستا منطقی ggml_metal_fusion_check_norm(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گره_idxs,
              صحیح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(گره_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    اگر (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        ثابت ggml_tensor * scale = nodes[1];
        اگر (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            بازگشت نادرست;
        }

        بازگشت درست;
    }

    برای (صحیح j = 1; j < (صحیح) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        اگر (nodes[j]->src[0] != nodes[j - 1]) {
            بازگشت نادرست;
        }

        // the weight/bias must have the same row width as the norm input
        اگر (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            بازگشت نادرست;
        }

        اگر (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            بازگشت نادرست;
        }

        اگر (nodes[j]->type != GGML_TYPE_F32) {
            بازگشت نادرست;
        }
    }

    بازگشت درست;
}

// SSM_CONV + UNARY (silu)
ایستا منطقی ggml_metal_fusion_check_ssm_conv_silu(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گره_idxs,
              صحیح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(گره_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    ثابت ggml_tensor * conv = nodes[0];
    ثابت ggml_tensor * un   = nodes[1];

    اگر (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        بازگشت نادرست;
    }

    اگر (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        بازگشت نادرست;
    }

    اگر (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        بازگشت نادرست;
    }

    بازگشت درست;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
ایستا منطقی ggml_metal_fusion_check_add_chain(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گره_idxs,
              صحیح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(گره_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    برای (صحیح j = 1; j < (صحیح) fusion->ops.size(); j++) {
        اگر (nodes[j]->src[0] != nodes[j - 1]) {
            بازگشت نادرست;
        }

        اگر (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            بازگشت نادرست;
        }

        اگر (mode == GGML_METAL_FUSION_FULL) {
            اگر (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                بازگشت نادرست;
            }
        }
    }

    بازگشت درست;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
ایستا منطقی ggml_metal_fusion_check_gdn_cache(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گره_idxs,
              صحیح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(گره_idxs);
    GGML_UNUSED(idx);

    ثابت ggml_tensor * gdn = nodes[0];
    ثابت ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    اگر (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        بازگشت نادرست;
    }

    اگر (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        بازگشت نادرست;
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
    اگر (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        بازگشت نادرست;
    }

    ثابت int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    اگر (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        بازگشت نادرست;
    }

    اگر (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        اگر (dst->data == nullptr) {
            بازگشت نادرست;
        }
    }

    بازگشت درست;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
ایستا منطقی ggml_metal_fusion_check_snake(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گره_idxs,
              صحیح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(گره_idxs);
    GGML_UNUSED(idx);

    ثابت ggml_tensor * mul0     = nodes[0];
    ثابت ggml_tensor * sin_گره = nodes[1];
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
    ثابت منطقی types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_گره->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    ثابت منطقی shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    ثابت منطقی dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    ثابت منطقی contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    بازگشت types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
ایستا ثابت std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
ایستا ثابت std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
ایستا ثابت std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
ایستا ثابت std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

ایستا منطقی ggml_metal_fusion_check_topk_moe(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گره_idxs,
              صحیح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    ثابت صحیح n_ops = (صحیح) fusion->ops.size();

    ثابت منطقی with_norm  = n_ops >= 6;
    ثابت منطقی with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    ثابت std::vector<ggml_op> & ops_all = fusion->ops_all;

    ثابت صحیح raw_start = گره_idxs[idx];
    صحیح       raw_end   = گره_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    اگر (with_norm && !with_scale) {
        اگر (raw_end + 1 >= gf->n_nodes) {
            بازگشت نادرست;
        }
        ثابت ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        اگر (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            بازگشت نادرست;
        }
        raw_end++;
    }

    ثابت صحیح raw_شمار = raw_end - raw_start + 1;
    اگر (raw_شمار != (صحیح) ops_all.size()) {
        بازگشت نادرست;
    }

    صحیح raw_idxs[GGML_METAL_FUSION_MAX];
    برای (صحیح i = 0; i < raw_شمار; ++i) {
        raw_idxs[i] = raw_start + i;
        اگر (gf->nodes[raw_start + i]->op != ops_all[i]) {
            بازگشت نادرست;
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
    اعشاری scale   = 1.0f;
    اعشاری max_bias = 0.0f;
    memcpy(&scale,    ((ثابت int32_t *) softmax->op_params) + 0, اندازه(scale));
    memcpy(&max_bias, ((ثابت int32_t *) softmax->op_params) + 1, اندازه(max_bias));
    اگر (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        بازگشت نادرست;
    }

    اگر (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        بازگشت نادرست;
    }

    ثابت int64_t n_expert      = logits->ne[0];
    ثابت int64_t n_tokens      = logits->ne[1];
    ثابت int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    اگر (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        بازگشت نادرست;
    }

    اگر (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        بازگشت نادرست;
    }

    اگر (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        بازگشت نادرست;
    }

    اگر (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        بازگشت نادرست;
    }

    اگر (with_norm) {
        ثابت ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
        ثابت ggml_tensor * sum_rows         = gf->nodes[raw_start + 6];
        ثابت ggml_tensor * clamp            = gf->nodes[raw_start + 7];
        ثابت ggml_tensor * div              = gf->nodes[raw_start + 8];
        ثابت ggml_tensor * out_reshaped     = gf->nodes[raw_start + 9];

        اگر (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            clamp->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            out_reshaped->src[0] != div) {
            بازگشت نادرست;
        }

        اگر (with_scale) {
            ثابت ggml_tensor * scale_گره = gf->nodes[raw_start + 10];
            اگر (scale_گره->src[0] != out_reshaped) {
                بازگشت نادرست;
            }
        }
    } وگرنه اگر (with_scale) {
        ثابت ggml_tensor * scale_گره = gf->nodes[raw_start + 5];
        اگر (scale_گره->src[0] != get_rows) {
            بازگشت نادرست;
        }
    }

    ثابت صحیح outputs[2] = { raw_start + 3, raw_end };
    اگر (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_شمار, ops_all.data(), outputs, 2)) {
        بازگشت نادرست;
    }

    اگر (mode == GGML_METAL_FUSION_FULL) {
        اگر (!logits->data || !out->data || !ids->data) {
            بازگشت نادرست;
        }
    }

    بازگشت درست;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

ساختار ggml_metal_moe_reduce_match {
    ثابت ggml_tensor * experts;
    ثابت ggml_tensor * weights;
    ثابت ggml_tensor * dst;
    صحیح گره_شمار;
};

ایستا منطقی ggml_metal_fusion_match_moe_reduce(
        ثابت ggml_cgraph * gf, صحیح گره_idx, ثابت std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    اگر (match == nullptr || گره_idx < 0 || گره_idx + (صحیح) ops_all.size() > gf->n_nodes) {
        بازگشت نادرست;
    }

    ثابت ggml_tensor * mul = gf->nodes[گره_idx];
    اگر (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    ثابت صحیح raw_شمار     = (صحیح) ops_all.size();
    ثابت صحیح n_expert_used = raw_شمار / 2;

    اگر (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_شمار != 2 * n_expert_used) {
        بازگشت نادرست;
    }

    صحیح n_views = 0;
    وقتی (گره_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[گره_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    اگر (n_views != n_expert_used) {
        بازگشت نادرست;
    }

    برای (صحیح i = n_expert_used + 1; i < raw_شمار; ++i) {
        اگر (gf->nodes[گره_idx + i]->op != GGML_OP_ADD) {
            بازگشت نادرست;
        }
    }

    صحیح raw_idxs[GGML_METAL_FUSION_MAX];
    برای (صحیح i = 0; i < raw_شمار; ++i) {
        raw_idxs[i] = گره_idx + i;
        اگر (gf->nodes[گره_idx + i]->op != ops_all[i]) {
            بازگشت نادرست;
        }
    }

    ثابت ggml_tensor * experts = mul->src[0];
    ثابت ggml_tensor * weights = mul->src[1];
    ثابت ggml_tensor * dst     = gf->nodes[گره_idx + raw_شمار - 1];

    اگر (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    ثابت int64_t n_embd   = experts->ne[0];
    ثابت int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    اگر (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        بازگشت نادرست;
    }

    اگر (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        بازگشت نادرست;
    }

    برای (صحیح i = 1; i <= n_expert_used; ++i) {
        ثابت ggml_tensor * view = gf->nodes[گره_idx + i];
        اگر (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            بازگشت نادرست;
        }
    }

    ثابت ggml_tensor * prev_add = nullptr;
    برای (صحیح j = 1; j < n_expert_used; ++j) {
        ثابت ggml_tensor * add = gf->nodes[گره_idx + n_expert_used + j];
        ثابت ggml_tensor * rhs = gf->nodes[گره_idx + j + 1];
        ثابت ggml_tensor * lhs = j == 1 ? gf->nodes[گره_idx + 1] : prev_add;
        اگر (add->src[0] != lhs || add->src[1] != rhs) {
            بازگشت نادرست;
        }
        prev_add = add;
    }

    ثابت صحیح outputs[1] = { گره_idx + raw_شمار - 1 };
    اگر (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_شمار, ops_all.data(), outputs, 1)) {
        بازگشت نادرست;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->گره_شمار = raw_شمار;
    بازگشت درست;
}

ایستا منطقی ggml_metal_fusion_check_moe_reduce(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گره_idxs,
              صحیح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    اگر (!ggml_metal_fusion_match_moe_reduce(gf, گره_idxs[idx], fusion->ops_all, &match)) {
        بازگشت نادرست;
    }

    اگر ((صحیح) fusion->ops.size() != match.experts->ne[1]) {
        بازگشت نادرست;
    }

    ثابت صحیح raw_end = گره_idxs[idx] + match.گره_شمار - 1;
    اگر (گره_idxs[idx + (صحیح) fusion->ops.size() - 1] != raw_end) {
        بازگشت نادرست;
    }

    اگر (mode == GGML_METAL_FUSION_FULL) {
        اگر (!match.experts->data || !match.weights->data || !match.dst->data) {
            بازگشت نادرست;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        اگر (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            بازگشت نادرست;
        }
    }

    بازگشت درست;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
ایستا منطقی ggml_metal_tensor_is_weight(ثابت ساختار ggml_tensor * t) {
    ثابت ggml_tensor * base = t->view_src != پوچ ? t->view_src : t;

    بازگشت base->buffer != پوچ && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

ایستا ثابت ساختار ggml_tensor * ggml_metal_mul_mat_add_operand(ثابت ساختار ggml_tensor * mm, ثابت ساختار ggml_tensor * add) {
    اگر (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        بازگشت پوچ;
    }

    ثابت ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    ثابت منطقی ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    بازگشت ok ? other : پوچ;
}

ایستا ثابت ساختار ggml_tensor * ggml_metal_mul_mat_add_residual(ثابت ساختار ggml_tensor * mm, ثابت ساختار ggml_tensor * add) {
    ثابت ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    ثابت منطقی ok = res != پوچ && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    بازگشت ok ? res : پوچ;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
ایستا منطقی ggml_metal_fusion_check_mul_mat_add(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گره_idxs,
              صحیح                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(گره_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    ثابت ggml_tensor * mm  = nodes[0];
    ثابت ggml_tensor * add = nodes[1];

    اگر (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        بازگشت نادرست;
    }

    اگر (mode == GGML_METAL_FUSION_STRUCTURAL) {
        بازگشت درست;
    }

    ثابت ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    اگر (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        بازگشت نادرست;
    }

    بازگشت !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

ایستا ثابت std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
ایستا ثابت std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
ایستا ثابت std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
ایستا ثابت std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
ایستا ثابت std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
ایستا ثابت std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

ایستا ثابت std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
ایستا ثابت std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ایستا ثابت std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ایستا ثابت std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ایستا ثابت std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ایستا ثابت std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
ایستا ثابت std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

ایستا ثابت std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

ایستا ثابت std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

ایستا ثابت std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
ایستا ثابت std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
ایستا ثابت std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ایستا ثابت std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ایستا ثابت std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ایستا ثابت std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
ایستا ثابت std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

ایستا ثابت std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };

ایستا ثابت std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     نادرست, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_norm_mul_add,           {},     نادرست, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     نادرست, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     نادرست, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD,   ops_rms_norm_mul_add,       {},     نادرست, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     نادرست, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_2,                  {},     نادرست, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_3,                  {},     نادرست, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_4,                  {},     نادرست, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_5,                  {},     نادرست, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_6,                  {},     نادرست, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,      ops_add_7,                  {},     نادرست, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     نادرست, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     درست,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    درست,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    درست,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    درست,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    درست,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     درست,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     درست,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     درست,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     درست,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     درست,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     درست,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     درست,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     نادرست, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_ADD,    ops_mul_mat_add,            {},     نادرست, ggml_metal_fusion_check_mul_mat_add },
};

// ---- alloc deps -----------------------------------------------------------

ایستا منطقی ggml_metal_fusion_match_raw_pattern(
        ثابت ggml_cgraph * gf, صحیح گره_idx, ثابت std::vector<ggml_op> & ops) {
    اگر (گره_idx < 0 || گره_idx + (صحیح) ops.size() > gf->n_nodes) {
        بازگشت نادرست;
    }

    برای (صحیح i = 0; i < (صحیح) ops.size(); ++i) {
        اگر (gf->nodes[گره_idx + i]->op != ops[i]) {
            بازگشت نادرست;
        }
    }

    بازگشت درست;
}

ایستا تهی ggml_metal_fusion_add_pattern_alloc_deps(
        تهی * user_data,
        تهی (*add_alloc_dep)(تهی *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        ثابت ggml_metal_fusion * fusion,
        صحیح گره_idx) {
    ثابت صحیح last_گره = گره_idx + (صحیح) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    برای (صحیح j = 0; j < (صحیح) fusion->ops_all.size(); ++j) {
        ggml_tensor * گره = gf->nodes[گره_idx + j];
        برای (صحیح s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = گره->src[s];
            اگر (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_گره]);
            }
        }
        seen.insert(گره);
    }
}

تهی ggml_metal_fusion_add_alloc_deps(
        تهی * user_data,
        تهی (*add_alloc_dep)(تهی *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    برای (صحیح i = 0; i < gf->n_nodes; ++i) {
        ثابت ggml_metal_fusion * best = nullptr;
        صحیح best_raw = 0;

        برای (ثابت ggml_metal_fusion & fusion : ggml_metal_fusions) {
            اگر ((صحیح) fusion.ops_all.size() <= best_raw) {
                ادامه;
            }
            اگر (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (صحیح) fusion.ops_all.size();
            }
        }

        اگر (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

ایستا std::string ggml_metal_fusion_label(ثابت ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    برای (صحیح j = 0; j < (صحیح) fusion->ops.size(); j++) {
        اگر (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    بازگشت label;
}

ساختار ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    منطقی enabled;
    منطقی stats;
    منطقی labels_set;
    صحیح  debug;
};

ساختار ggml_metal_fusion_info * ggml_metal_fusion_info_init(منطقی enabled, صحیح debug) {
    ggml_metal_fusion_info * finfo = جدید ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = نادرست;
    finfo->debug      = debug;

    اگر (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    بازگشت finfo;
}

تهی ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

منطقی ggml_metal_fusion_info_enabled(ثابت ggml_metal_fusion_info * finfo) {
    بازگشت finfo->enabled;
}

منطقی ggml_metal_fusion_info_stats(ثابت ggml_metal_fusion_info * finfo) {
    بازگشت finfo->stats;
}

صحیح ggml_metal_fusion_info_debug(ثابت ggml_metal_fusion_info * finfo) {
    بازگشت finfo->debug;
}

صحیح ggml_metal_fusion_info_n_fusions(ثابت ggml_metal_fusion_info * finfo) {
    بازگشت (صحیح) finfo->labels.size();
}

ثابت نویسه * ggml_metal_fusion_info_label(ثابت ggml_metal_fusion_info * finfo, صحیح idx) {
    GGML_ASSERT(idx >= 0 && idx < (صحیح) finfo->labels.size());
    بازگشت finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_شمار(ثابت ggml_metal_fusion_info * finfo, صحیح idx) {
    GGML_ASSERT(idx >= 0 && idx < (صحیح) finfo->counts.size());
    بازگشت finfo->counts[idx];
}

تهی ggml_metal_fusion_info_شمار_fusion(ggml_metal_fusion_info * finfo, ثابت ggml_metal_fusion * fusion) {
    اگر (!finfo->stats || fusion == nullptr) {
        بازگشت;
    }

    ثابت ptrdiff_t idx = fusion - ggml_metal_fusions.data();
    اگر (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

تهی ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, منطقی enabled) {
    finfo->enabled = enabled;
}

تهی ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    اگر (finfo->labels_set) {
        بازگشت;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    برای (ثابت ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = درست;
}

تهی ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = درست;
    ggml_metal_fusion_info_labels_init(finfo);
}

تهی ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

صحیح ggml_metal_fusion_info_stats_get(ثابت ggml_metal_fusion_info * finfo, ثابت نویسه ** labels, uint64_t * counts, صحیح n) {
    ثابت صحیح n_fusions = (صحیح) finfo->labels.size();

    اگر (labels == nullptr) {
        بازگشت n_fusions;
    }

    ثابت صحیح n_fill = std::min(n, n_fusions);
    برای (صحیح i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        اگر (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    بازگشت n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
ایستا منطقی ggml_metal_fusion_check_memory_ranges(
        ثابت ggml_metal_fusion      * fusion,
        ثابت ggml_tensor * ثابت    * nodes,
        صحیح                            گره_شمار) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        اگر (a->data && b->data && a->data == b->data) {
            بازگشت درست;
        }
        برای (ثابت ggml_tensor * v = a; v; v = v->view_src) {
            اگر (v == b) {
                بازگشت درست;
            }
        }
        برای (ثابت ggml_tensor * v = b; v; v = v->view_src) {
            اگر (v == a) {
                بازگشت درست;
            }
        }
        بازگشت نادرست;
    };

    auto nodes_overlap = [](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        اگر (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            بازگشت نادرست;
        }

        اگر (a->buffer != b->buffer) {
            بازگشت نادرست;
        }

        ثابت int64_t a_start = (int64_t) a->data;
        ثابت int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        ثابت int64_t b_start = (int64_t) b->data;
        ثابت int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        بازگشت (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](ثابت ggml_tensor * src, ثابت ggml_tensor * ثابت * nodes, صحیح j) {
        برای (صحیح k = 0; k < j; ++k) {
            اگر (src == nodes[k]) {
                بازگشت درست;
            }
            برای (ثابت ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                اگر (view_src == nodes[k]) {
                    بازگشت درست;
                }
            }
        }
        بازگشت نادرست;
    };

    auto check_dst = [&](ثابت ggml_tensor * dst) {
        برای (صحیح j = 0; j < گره_شمار; ++j) {
            برای (صحیح s = 0; s < GGML_MAX_SRC; ++s) {
                ثابت ggml_tensor * src = nodes[j]->src[s];
                اگر (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    ادامه;
                }

                اگر (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    بازگشت نادرست;
                }
            }
        }
        بازگشت درست;
    };

    اگر (!check_dst(nodes[گره_شمار - 1])) {
        بازگشت نادرست;
    }

    برای (صحیح offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < گره_شمار);
        اگر (!check_dst(nodes[offset])) {
            بازگشت نادرست;
        }
    }

    بازگشت درست;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
ثابت ggml_metal_fusion * ggml_metal_fusion_next(
        ثابت ggml_cgraph * gf,
        ثابت صحیح * گره_idxs,
        صحیح n_idxs,
        صحیح idx,
        ggml_metal_fusion_mode mode,
        صحیح * n_out) {
    ثابت ggml_metal_fusion * res = nullptr;
    صحیح best = 1;

    برای (ثابت ggml_metal_fusion & fusion : ggml_metal_fusions) {
        ثابت صحیح n_ops = (صحیح) fusion.ops.size();

        // only look for a longer match than the current best
        اگر (n_ops <= best) {
            ادامه;
        }
        اگر (idx + n_ops > n_idxs) {
            ادامه;
        }

        ثابت ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        منطقی ok = درست;
        برای (صحیح j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[گره_idxs[idx + j]];
            اگر (nodes[j]->op != fusion.ops[j]) {
                ok = نادرست;
                بشکن;
            }
        }
        اگر (!ok) {
            ادامه;
        }

        اگر (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            برای (صحیح j = 1; j < n_ops && ok; j++) {
                اگر (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = نادرست;
                    بشکن;
                }
                اگر (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = نادرست;
                    بشکن;
                }
            }
            اگر (!ok) {
                ادامه;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            صحیح outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = گره_idxs[idx + n_ops - 1];
            برای (size_t i = 0; i < fusion.outs.size(); ++i) {
                ثابت صحیح out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = گره_idxs[idx + out_offset];
            }

            ثابت صحیح n_outputs = 1 + (صحیح) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            اگر (!ggml_can_fuse_subgraph_ext(gf, گره_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                ادامه;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        اگر (fusion.check && !fusion.check(&fusion, nodes, gf, گره_idxs, idx, mode)) {
            ادامه;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        اگر (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            ادامه;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    بازگشت res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
صحیح ggml_metal_fusion_max(ثابت ggml_cgraph * gf, صحیح idx) {
    // a view node cannot start a pattern - pack it alone
    اگر (ggml_op_is_empty(gf->nodes[idx]->op)) {
        بازگشت 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    صحیح idxs[GGML_METAL_FUSION_MAX];
    صحیح n_idxs = 0;
    برای (صحیح i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        اگر (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    اگر (n_idxs == 0) {
        بازگشت 1;
    }

    صحیح total = 0;
    صحیح i_f = 0;

    وقتی (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        صحیح len = 1;
        ثابت ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        اگر (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            بشکن;
        }

        total += len;
        i_f += len;
    }

    اگر (i_f == 0) {
        بازگشت 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    بازگشت std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
