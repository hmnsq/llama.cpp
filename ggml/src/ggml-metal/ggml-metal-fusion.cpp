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
ساکن std::vector<ggml_op> ggml_metal_fusion_filter_ops(ثابت std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    برائے (ggml_op op : ops_all) {
        اگر (!ggml_op_is_خالی_خانے(op)) {
            ops.push_back(op);
        }
    }

    واپس ops;
}

ساخت ggml_metal_fusion {
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
    منطقی (*check)(ثابت ساخت ggml_metal_fusion   * fusion,
                  ثابت ساخت ggml_tensor * ثابت * nodes,
                  ثابت ساخت ggml_cgraph         * gf,
                  ثابت صحیح                        * گرہ_idxs,
                        صحیح                          idx,
                        ggml_metal_fusion_منوال       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            ثابت std::vector<ggml_op> & ops_all,
            ثابت std::vector<صحیح> & outs,
            منطقی unsafe,
            منطقی (*check)(ثابت ساخت ggml_metal_fusion   * fusion,
                          ثابت ساخت ggml_tensor * ثابت * nodes,
                          ثابت ساخت ggml_cgraph         * gf,
                          ثابت صحیح                        * گرہ_idxs,
                                صحیح                          idx,
                                ggml_metal_fusion_منوال       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(ثابت ggml_metal_fusion * fusion) {
    واپس fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
ساکن منطقی ggml_metal_fusion_same_بفر(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    اگر (!a || !b) {
        واپس جھوٹ;
    }

    ggml_backend_بفر_t ba = a->view_src ? a->view_src->بفر : a->بفر;
    ggml_backend_بفر_t bb = b->view_src ? b->view_src->بفر : b->بفر;

    ggml_metal_بفر_t ca = (ggml_metal_بفر_t) ba->context;
    ggml_metal_بفر_t cb = (ggml_metal_بفر_t) bb->context;

    واپس ggml_metal_بفر_get_id(ca, a).metal == ggml_metal_بفر_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
ساکن منطقی ggml_metal_fusion_overlap(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    ggml_backend_بفر_t ba = a->view_src ? a->view_src->بفر : a->بفر;
    ggml_backend_بفر_t bb = b->view_src ? b->view_src->بفر : b->بفر;

    ثابت ggml_metal_بفر_id bid_a = ggml_metal_بفر_get_id((ggml_metal_بفر_t) ba->context, a);
    ثابت ggml_metal_بفر_id bid_b = ggml_metal_بفر_get_id((ggml_metal_بفر_t) bb->context, b);

    اگر (bid_a.metal == خالی_اشاریہ || bid_a.metal != bid_b.metal) {
        واپس جھوٹ;
    }

    واپس bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
ساکن منطقی ggml_metal_fusion_check_norm(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گرہ_idxs,
              صحیح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_غیر_مستعمل(mode);
    GGML_غیر_مستعمل(gf);
    GGML_غیر_مستعمل(گرہ_idxs);
    GGML_غیر_مستعمل(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    اگر (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        ثابت ggml_tensor * scale = nodes[1];
        اگر (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            واپس جھوٹ;
        }

        واپس سچ;
    }

    برائے (صحیح j = 1; j < (صحیح) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        اگر (nodes[j]->src[0] != nodes[j - 1]) {
            واپس جھوٹ;
        }

        // the weight/bias must have the same row width as the norm input
        اگر (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            واپس جھوٹ;
        }

        اگر (!ggml_is_contiguous_صفیں(nodes[j]->src[1])) {
            واپس جھوٹ;
        }

        اگر (nodes[j]->type != GGML_TYPE_F32) {
            واپس جھوٹ;
        }
    }

    واپس سچ;
}

// SSM_CONV + UNARY (silu)
ساکن منطقی ggml_metal_fusion_check_ssm_conv_silu(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گرہ_idxs,
              صحیح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_غیر_مستعمل(fusion);
    GGML_غیر_مستعمل(gf);
    GGML_غیر_مستعمل(گرہ_idxs);
    GGML_غیر_مستعمل(idx);
    GGML_غیر_مستعمل(mode);

    ثابت ggml_tensor * conv = nodes[0];
    ثابت ggml_tensor * un   = nodes[1];

    اگر (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        واپس جھوٹ;
    }

    اگر (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        واپس جھوٹ;
    }

    اگر (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_صفیں(un)) {
        واپس جھوٹ;
    }

    واپس سچ;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
ساکن منطقی ggml_metal_fusion_check_جوڑو_chain(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گرہ_idxs,
              صحیح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_غیر_مستعمل(gf);
    GGML_غیر_مستعمل(گرہ_idxs);
    GGML_غیر_مستعمل(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    برائے (صحیح j = 1; j < (صحیح) fusion->ops.size(); j++) {
        اگر (nodes[j]->src[0] != nodes[j - 1]) {
            واپس جھوٹ;
        }

        اگر (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            واپس جھوٹ;
        }

        اگر (mode == GGML_METAL_FUSION_بھرا) {
            اگر (!ggml_metal_fusion_same_بفر(nodes[j]->src[1], nodes[0]->src[1])) {
                واپس جھوٹ;
            }
        }
    }

    واپس سچ;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
ساکن منطقی ggml_metal_fusion_check_gdn_cache(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گرہ_idxs,
              صحیح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_غیر_مستعمل(fusion);
    GGML_غیر_مستعمل(gf);
    GGML_غیر_مستعمل(گرہ_idxs);
    GGML_غیر_مستعمل(idx);

    ثابت ggml_tensor * gdn = nodes[0];
    ثابت ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    اگر (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        واپس جھوٹ;
    }

    اگر (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        واپس جھوٹ;
    }

    ثابت int64_t S_v      = gdn->src[2]->ne[0];
    ثابت int64_t H        = gdn->src[2]->ne[1];
    ثابت int64_t n_علامات = gdn->src[2]->ne[2];
    ثابت int64_t n_seqs   = gdn->src[2]->ne[3];
    ثابت int64_t K        = ggml_get_op_params_i32(gdn, 0);
    ثابت size_t  دم_off = ggml_صف_سائز(GGML_TYPE_F32, S_v * H * n_علامات * n_seqs);

    ثابت int64_t D         = S_v * S_v * H;
    ثابت int64_t n_written = std::min<int64_t>(n_علامات, K);

    ثابت ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    ثابت ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    اگر (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != دم_off || !ggml_is_contiguous(src)) {
        واپس جھوٹ;
    }

    ثابت int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    اگر (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_سائز(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_صف_سائز(GGML_TYPE_F32, D)) {
        واپس جھوٹ;
    }

    اگر (mode == GGML_METAL_FUSION_بھرا) {
        // the cache must be allocated so the kernel can write straight to its buffer
        اگر (dst->ڈیٹا == خالی_اشاریہ) {
            واپس جھوٹ;
        }
    }

    واپس سچ;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
ساکن منطقی ggml_metal_fusion_check_snake(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گرہ_idxs,
              صحیح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_غیر_مستعمل(fusion);
    GGML_غیر_مستعمل(mode);
    GGML_غیر_مستعمل(gf);
    GGML_غیر_مستعمل(گرہ_idxs);
    GGML_غیر_مستعمل(idx);

    ثابت ggml_tensor * mul0     = nodes[0];
    ثابت ggml_tensor * sin_گرہ = nodes[1];
    ثابت ggml_tensor * sqr      = nodes[2];
    ثابت ggml_tensor * mul1     = nodes[3];
    ثابت ggml_tensor * جوڑو_2      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    ثابت ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ثابت ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ثابت ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    ثابت ggml_tensor * x_اندر_جوڑو = (جوڑو_2->src[0] == mul1) ? جوڑو_2->src[1] : جوڑو_2->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    ثابت منطقی types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_گرہ->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (جوڑو_2->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    ثابت منطقی shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    ثابت منطقی dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (جوڑو_2->ne[2]   == 1) && (جوڑو_2->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    ثابت منطقی contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(جوڑو_2) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    واپس types_ok && shape_ok && dim_ok && contig_ok && x_اندر_جوڑو == x;
}

#تعریف_کرو GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
ساکن ثابت std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_صفیں
};
ساکن ثابت std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_صفیں, GGML_OP_SCALE
};
ساکن ثابت std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_صفیں,
    GGML_OP_RESHAPE, GGML_OP_مجموع_صفیں, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
ساکن ثابت std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_صفیں,
    GGML_OP_RESHAPE, GGML_OP_مجموع_صفیں, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

ساکن منطقی ggml_metal_fusion_check_topk_moe(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گرہ_idxs,
              صحیح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_غیر_مستعمل(nodes);

    ثابت صحیح n_ops = (صحیح) fusion->ops.size();

    ثابت منطقی with_norm  = n_ops >= 6;
    ثابت منطقی with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    ثابت std::vector<ggml_op> & ops_all = fusion->ops_all;

    ثابت صحیح raw_آغاز = گرہ_idxs[idx];
    صحیح       raw_انتہا   = گرہ_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    اگر (with_norm && !with_scale) {
        اگر (raw_انتہا + 1 >= gf->n_nodes) {
            واپس جھوٹ;
        }
        ثابت ggml_tensor * trailing_reshape = gf->nodes[raw_انتہا + 1];
        اگر (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_انتہا]) {
            واپس جھوٹ;
        }
        raw_انتہا++;
    }

    ثابت صحیح raw_گنتی = raw_انتہا - raw_آغاز + 1;
    اگر (raw_گنتی != (صحیح) ops_all.size()) {
        واپس جھوٹ;
    }

    صحیح raw_idxs[GGML_METAL_FUSION_MAX];
    برائے (صحیح i = 0; i < raw_گنتی; ++i) {
        raw_idxs[i] = raw_آغاز + i;
        اگر (gf->nodes[raw_آغاز + i]->op != ops_all[i]) {
            واپس جھوٹ;
        }
    }

    ثابت ggml_tensor * softmax        = gf->nodes[raw_آغاز];
    ثابت ggml_tensor * probs_reshaped = gf->nodes[raw_آغاز + 1];
    ثابت ggml_tensor * argsort        = gf->nodes[raw_آغاز + 2];
    ثابت ggml_tensor * ids            = gf->nodes[raw_آغاز + 3];
    ثابت ggml_tensor * get_صفیں       = gf->nodes[raw_آغاز + 4];
    ثابت ggml_tensor * برآمد            = gf->nodes[raw_انتہا];
    ثابت ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    اعشاری scale   = 1.0f;
    اعشاری max_bias = 0.0f;
    حافظہ_نقل(&scale,    ((ثابت int32_t *) softmax->op_params) + 0, حجم(scale));
    حافظہ_نقل(&max_bias, ((ثابت int32_t *) softmax->op_params) + 1, حجم(max_bias));
    اگر (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        واپس جھوٹ;
    }

    اگر (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        برآمد->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        واپس جھوٹ;
    }

    ثابت int64_t n_expert      = logits->ne[0];
    ثابت int64_t n_علامات      = logits->ne[1];
    ثابت int64_t n_expert_مستعمل = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    اگر (n_expert <= 0 || n_expert_مستعمل <= 0 || n_expert_مستعمل > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_مستعمل > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        واپس جھوٹ;
    }

    اگر (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_علامات || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        برآمد->ne[0] != 1 || برآمد->ne[1] != n_expert_مستعمل || برآمد->ne[2] != n_علامات || برآمد->ne[3] != 1) {
        واپس جھوٹ;
    }

    اگر (!ggml_is_contiguous(logits) || !ggml_is_contiguous(برآمد) ||
        ids->nb[0] != ggml_type_سائز(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_سائز(GGML_TYPE_I32) * n_expert) {
        واپس جھوٹ;
    }

    اگر (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_صفیں->src[0] != probs_reshaped || get_صفیں->src[1] != ids) {
        واپس جھوٹ;
    }

    اگر (with_norm) {
        ثابت ggml_tensor * weights_reshaped = gf->nodes[raw_آغاز + 5];
        ثابت ggml_tensor * مجموع_صفیں         = gf->nodes[raw_آغاز + 6];
        ثابت ggml_tensor * clamp            = gf->nodes[raw_آغاز + 7];
        ثابت ggml_tensor * div              = gf->nodes[raw_آغاز + 8];
        ثابت ggml_tensor * برآمد_reshaped     = gf->nodes[raw_آغاز + 9];

        اگر (weights_reshaped->src[0] != get_صفیں || مجموع_صفیں->src[0] != weights_reshaped ||
            clamp->src[0] != مجموع_صفیں || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            برآمد_reshaped->src[0] != div) {
            واپس جھوٹ;
        }

        اگر (with_scale) {
            ثابت ggml_tensor * scale_گرہ = gf->nodes[raw_آغاز + 10];
            اگر (scale_گرہ->src[0] != برآمد_reshaped) {
                واپس جھوٹ;
            }
        }
    } ورنہ اگر (with_scale) {
        ثابت ggml_tensor * scale_گرہ = gf->nodes[raw_آغاز + 5];
        اگر (scale_گرہ->src[0] != get_صفیں) {
            واپس جھوٹ;
        }
    }

    ثابت صحیح outputs[2] = { raw_آغاز + 3, raw_انتہا };
    اگر (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_گنتی, ops_all.ڈیٹا(), outputs, 2)) {
        واپس جھوٹ;
    }

    اگر (mode == GGML_METAL_FUSION_بھرا) {
        اگر (!logits->ڈیٹا || !برآمد->ڈیٹا || !ids->ڈیٹا) {
            واپس جھوٹ;
        }
    }

    واپس سچ;
}

#تعریف_کرو GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

ساخت ggml_metal_moe_reduce_مطابقت {
    ثابت ggml_tensor * experts;
    ثابت ggml_tensor * weights;
    ثابت ggml_tensor * dst;
    صحیح گرہ_گنتی;
};

ساکن منطقی ggml_metal_fusion_مطابقت_moe_reduce(
        ثابت ggml_cgraph * gf, صحیح گرہ_idx, ثابت std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_مطابقت * مطابقت) {
    اگر (مطابقت == خالی_اشاریہ || گرہ_idx < 0 || گرہ_idx + (صحیح) ops_all.size() > gf->n_nodes) {
        واپس جھوٹ;
    }

    ثابت ggml_tensor * mul = gf->nodes[گرہ_idx];
    اگر (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    ثابت صحیح raw_گنتی     = (صحیح) ops_all.size();
    ثابت صحیح n_expert_مستعمل = raw_گنتی / 2;

    اگر (n_expert_مستعمل < 2 || n_expert_مستعمل > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_گنتی != 2 * n_expert_مستعمل) {
        واپس جھوٹ;
    }

    صحیح n_views = 0;
    جب (گرہ_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[گرہ_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    اگر (n_views != n_expert_مستعمل) {
        واپس جھوٹ;
    }

    برائے (صحیح i = n_expert_مستعمل + 1; i < raw_گنتی; ++i) {
        اگر (gf->nodes[گرہ_idx + i]->op != GGML_OP_جوڑو) {
            واپس جھوٹ;
        }
    }

    صحیح raw_idxs[GGML_METAL_FUSION_MAX];
    برائے (صحیح i = 0; i < raw_گنتی; ++i) {
        raw_idxs[i] = گرہ_idx + i;
        اگر (gf->nodes[گرہ_idx + i]->op != ops_all[i]) {
            واپس جھوٹ;
        }
    }

    ثابت ggml_tensor * experts = mul->src[0];
    ثابت ggml_tensor * weights = mul->src[1];
    ثابت ggml_tensor * dst     = gf->nodes[گرہ_idx + raw_گنتی - 1];

    اگر (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس جھوٹ;
    }

    ثابت int64_t n_embd   = experts->ne[0];
    ثابت int64_t n_علامات = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    اگر (n_embd <= 0 || experts->ne[1] != n_expert_مستعمل || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_مستعمل || weights->ne[2] != n_علامات || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_علامات || dst->ne[2] != 1 || dst->ne[3] != 1) {
        واپس جھوٹ;
    }

    اگر (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        واپس جھوٹ;
    }

    برائے (صحیح i = 1; i <= n_expert_مستعمل; ++i) {
        ثابت ggml_tensor * view = gf->nodes[گرہ_idx + i];
        اگر (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_علامات ||
            view->nb[1] != mul->nb[2]) {
            واپس جھوٹ;
        }
    }

    ثابت ggml_tensor * پچھلا_جوڑو = خالی_اشاریہ;
    برائے (صحیح j = 1; j < n_expert_مستعمل; ++j) {
        ثابت ggml_tensor * جوڑو_2 = gf->nodes[گرہ_idx + n_expert_مستعمل + j];
        ثابت ggml_tensor * rhs = gf->nodes[گرہ_idx + j + 1];
        ثابت ggml_tensor * lhs = j == 1 ? gf->nodes[گرہ_idx + 1] : پچھلا_جوڑو;
        اگر (جوڑو_2->src[0] != lhs || جوڑو_2->src[1] != rhs) {
            واپس جھوٹ;
        }
        پچھلا_جوڑو = جوڑو_2;
    }

    ثابت صحیح outputs[1] = { گرہ_idx + raw_گنتی - 1 };
    اگر (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_گنتی, ops_all.ڈیٹا(), outputs, 1)) {
        واپس جھوٹ;
    }

    مطابقت->experts    = experts;
    مطابقت->weights    = weights;
    مطابقت->dst        = dst;
    مطابقت->گرہ_گنتی = raw_گنتی;
    واپس سچ;
}

ساکن منطقی ggml_metal_fusion_check_moe_reduce(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گرہ_idxs,
              صحیح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_غیر_مستعمل(nodes);

    ggml_metal_moe_reduce_مطابقت مطابقت;
    اگر (!ggml_metal_fusion_مطابقت_moe_reduce(gf, گرہ_idxs[idx], fusion->ops_all, &مطابقت)) {
        واپس جھوٹ;
    }

    اگر ((صحیح) fusion->ops.size() != مطابقت.experts->ne[1]) {
        واپس جھوٹ;
    }

    ثابت صحیح raw_انتہا = گرہ_idxs[idx] + مطابقت.گرہ_گنتی - 1;
    اگر (گرہ_idxs[idx + (صحیح) fusion->ops.size() - 1] != raw_انتہا) {
        واپس جھوٹ;
    }

    اگر (mode == GGML_METAL_FUSION_بھرا) {
        اگر (!مطابقت.experts->ڈیٹا || !مطابقت.weights->ڈیٹا || !مطابقت.dst->ڈیٹا) {
            واپس جھوٹ;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        اگر (ggml_metal_fusion_overlap(مطابقت.dst, مطابقت.experts) || ggml_metal_fusion_overlap(مطابقت.dst, مطابقت.weights)) {
            واپس جھوٹ;
        }
    }

    واپس سچ;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
ساکن منطقی ggml_metal_tensor_is_وزن(ثابت ساخت ggml_tensor * t) {
    ثابت ggml_tensor * بنیاد = t->view_src != تہی ? t->view_src : t;

    واپس بنیاد->بفر != تہی && ggml_backend_بفر_get_usage(بنیاد->بفر) == GGML_BACKEND_بفر_USAGE_WEIGHTS;
}

ساکن ثابت ساخت ggml_tensor * ggml_metal_mul_mat_جوڑو_عملیہ(ثابت ساخت ggml_tensor * mm, ثابت ساخت ggml_tensor * جوڑو_2) {
    اگر (جوڑو_2->op != GGML_OP_جوڑو || (جوڑو_2->src[0] == mm) == (جوڑو_2->src[1] == mm)) {
        واپس تہی;
    }

    ثابت ggml_tensor * other = جوڑو_2->src[0] == mm ? جوڑو_2->src[1] : جوڑو_2->src[0];

    ثابت منطقی ok = other->type == GGML_TYPE_F32 && جوڑو_2->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_وزن(other);

    واپس ok ? other : تہی;
}

ساکن ثابت ساخت ggml_tensor * ggml_metal_mul_mat_جوڑو_residual(ثابت ساخت ggml_tensor * mm, ثابت ساخت ggml_tensor * جوڑو_2) {
    ثابت ggml_tensor * res = ggml_metal_mul_mat_جوڑو_عملیہ(mm, جوڑو_2);

    ثابت منطقی ok = res != تہی && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(جوڑو_2);

    واپس ok ? res : تہی;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
ساکن منطقی ggml_metal_fusion_check_mul_mat_جوڑو(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحیح                     * گرہ_idxs,
              صحیح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_غیر_مستعمل(gf);
    GGML_غیر_مستعمل(گرہ_idxs);
    GGML_غیر_مستعمل(idx);
    GGML_غیر_مستعمل(fusion);

    ثابت ggml_tensor * mm  = nodes[0];
    ثابت ggml_tensor * جوڑو_2 = nodes[1];

    اگر (ggml_metal_mul_mat_جوڑو_عملیہ(mm, جوڑو_2) == خالی_اشاریہ ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        واپس جھوٹ;
    }

    اگر (mode == GGML_METAL_FUSION_STRUCTURAL) {
        واپس سچ;
    }

    ثابت ggml_tensor * res = ggml_metal_mul_mat_جوڑو_residual(mm, جوڑو_2);

    اگر (res == خالی_اشاریہ || !ggml_metal_op_mul_mat_use_mma(mm)) {
        واپس جھوٹ;
    }

    واپس !ggml_metal_fusion_overlap(جوڑو_2, mm->src[0]) && !ggml_metal_fusion_overlap(جوڑو_2, mm->src[1]) &&
        (جوڑو_2->ڈیٹا == res->ڈیٹا || !ggml_metal_fusion_overlap(جوڑو_2, res));
}

// ---- patterns ------------------------------------------------------------

ساکن ثابت std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
ساکن ثابت std::vector<ggml_op> ops_norm_mul_جوڑو     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_جوڑو };
ساکن ثابت std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
ساکن ثابت std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
ساکن ثابت std::vector<ggml_op> ops_rms_norm_mul_جوڑو = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_جوڑو };
ساکن ثابت std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

ساکن ثابت std::vector<ggml_op> ops_جوڑو_2 = { GGML_OP_جوڑو, GGML_OP_جوڑو };
ساکن ثابت std::vector<ggml_op> ops_جوڑو_3 = { GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو };
ساکن ثابت std::vector<ggml_op> ops_جوڑو_4 = { GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو };
ساکن ثابت std::vector<ggml_op> ops_جوڑو_5 = { GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو };
ساکن ثابت std::vector<ggml_op> ops_جوڑو_6 = { GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو };
ساکن ثابت std::vector<ggml_op> ops_جوڑو_7 = { GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو };
ساکن ثابت std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_جوڑو };

ساکن ثابت std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

ساکن ثابت std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

ساکن ثابت std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_جوڑو
};
ساکن ثابت std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_جوڑو, GGML_OP_جوڑو
};
ساکن ثابت std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو
};
ساکن ثابت std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو
};
ساکن ثابت std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو
};
ساکن ثابت std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو
};
ساکن ثابت std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو
};

ساکن ثابت std::vector<ggml_op> ops_mul_mat_جوڑو = { GGML_OP_MUL_MAT, GGML_OP_جوڑو };

ساکن ثابت std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     جھوٹ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_جوڑو,   ops_norm_mul_جوڑو,           {},     جھوٹ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     جھوٹ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     جھوٹ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_جوڑو,   ops_rms_norm_mul_جوڑو,       {},     جھوٹ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     جھوٹ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_جوڑو_CHAIN,      ops_جوڑو_2,                  {},     جھوٹ, ggml_metal_fusion_check_جوڑو_chain },
    { GGML_METAL_FUSION_جوڑو_CHAIN,      ops_جوڑو_3,                  {},     جھوٹ, ggml_metal_fusion_check_جوڑو_chain },
    { GGML_METAL_FUSION_جوڑو_CHAIN,      ops_جوڑو_4,                  {},     جھوٹ, ggml_metal_fusion_check_جوڑو_chain },
    { GGML_METAL_FUSION_جوڑو_CHAIN,      ops_جوڑو_5,                  {},     جھوٹ, ggml_metal_fusion_check_جوڑو_chain },
    { GGML_METAL_FUSION_جوڑو_CHAIN,      ops_جوڑو_6,                  {},     جھوٹ, ggml_metal_fusion_check_جوڑو_chain },
    { GGML_METAL_FUSION_جوڑو_CHAIN,      ops_جوڑو_7,                  {},     جھوٹ, ggml_metal_fusion_check_جوڑو_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     جھوٹ, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     سچ,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    سچ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    سچ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    سچ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    سچ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     سچ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     سچ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     سچ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     سچ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     سچ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     سچ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     سچ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     جھوٹ, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_جوڑو,    ops_mul_mat_جوڑو,            {},     جھوٹ, ggml_metal_fusion_check_mul_mat_جوڑو },
};

// ---- alloc deps -----------------------------------------------------------

ساکن منطقی ggml_metal_fusion_مطابقت_raw_نمونہ(
        ثابت ggml_cgraph * gf, صحیح گرہ_idx, ثابت std::vector<ggml_op> & ops) {
    اگر (گرہ_idx < 0 || گرہ_idx + (صحیح) ops.size() > gf->n_nodes) {
        واپس جھوٹ;
    }

    برائے (صحیح i = 0; i < (صحیح) ops.size(); ++i) {
        اگر (gf->nodes[گرہ_idx + i]->op != ops[i]) {
            واپس جھوٹ;
        }
    }

    واپس سچ;
}

ساکن خالی ggml_metal_fusion_جوڑو_نمونہ_alloc_deps(
        خالی * user_ڈیٹا,
        خالی (*جوڑو_alloc_dep)(خالی *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        ثابت ggml_metal_fusion * fusion,
        صحیح گرہ_idx) {
    ثابت صحیح آخری_گرہ = گرہ_idx + (صحیح) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> نظر_آیا;
    برائے (صحیح j = 0; j < (صحیح) fusion->ops_all.size(); ++j) {
        ggml_tensor * گرہ = gf->nodes[گرہ_idx + j];
        برائے (صحیح s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = گرہ->src[s];
            اگر (src && نظر_آیا.insert(src).second) {
                جوڑو_alloc_dep(user_ڈیٹا, src, gf->nodes[آخری_گرہ]);
            }
        }
        نظر_آیا.insert(گرہ);
    }
}

خالی ggml_metal_fusion_جوڑو_alloc_deps(
        خالی * user_ڈیٹا,
        خالی (*جوڑو_alloc_dep)(خالی *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    برائے (صحیح i = 0; i < gf->n_nodes; ++i) {
        ثابت ggml_metal_fusion * بہترین = خالی_اشاریہ;
        صحیح بہترین_raw = 0;

        برائے (ثابت ggml_metal_fusion & fusion : ggml_metal_fusions) {
            اگر ((صحیح) fusion.ops_all.size() <= بہترین_raw) {
                جاری;
            }
            اگر (ggml_metal_fusion_مطابقت_raw_نمونہ(gf, i, fusion.ops_all)) {
                بہترین = &fusion;
                بہترین_raw = (صحیح) fusion.ops_all.size();
            }
        }

        اگر (بہترین) {
            ggml_metal_fusion_جوڑو_نمونہ_alloc_deps(user_ڈیٹا, جوڑو_alloc_dep, gf, بہترین, i);
            i += بہترین_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

ساکن std::string ggml_metal_fusion_label(ثابت ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != خالی_اشاریہ);

    std::string label;
    برائے (صحیح j = 0; j < (صحیح) fusion->ops.size(); j++) {
        اگر (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    واپس label;
}

ساخت ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    منطقی enabled;
    منطقی stats;
    منطقی labels_سیٹ;
    صحیح  debug;
};

ساخت ggml_metal_fusion_info * ggml_metal_fusion_info_init(منطقی enabled, صحیح debug) {
    ggml_metal_fusion_info * finfo = نیا ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_سیٹ = جھوٹ;
    finfo->debug      = debug;

    اگر (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    واپس finfo;
}

خالی ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    حذف finfo;
}

منطقی ggml_metal_fusion_info_enabled(ثابت ggml_metal_fusion_info * finfo) {
    واپس finfo->enabled;
}

منطقی ggml_metal_fusion_info_stats(ثابت ggml_metal_fusion_info * finfo) {
    واپس finfo->stats;
}

صحیح ggml_metal_fusion_info_debug(ثابت ggml_metal_fusion_info * finfo) {
    واپس finfo->debug;
}

صحیح ggml_metal_fusion_info_n_fusions(ثابت ggml_metal_fusion_info * finfo) {
    واپس (صحیح) finfo->labels.size();
}

ثابت حرف * ggml_metal_fusion_info_label(ثابت ggml_metal_fusion_info * finfo, صحیح idx) {
    GGML_ASSERT(idx >= 0 && idx < (صحیح) finfo->labels.size());
    واپس finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_گنتی(ثابت ggml_metal_fusion_info * finfo, صحیح idx) {
    GGML_ASSERT(idx >= 0 && idx < (صحیح) finfo->counts.size());
    واپس finfo->counts[idx];
}

خالی ggml_metal_fusion_info_گنتی_fusion(ggml_metal_fusion_info * finfo, ثابت ggml_metal_fusion * fusion) {
    اگر (!finfo->stats || fusion == خالی_اشاریہ) {
        واپس;
    }

    ثابت ptrdiff_t idx = fusion - ggml_metal_fusions.ڈیٹا();
    اگر (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

خالی ggml_metal_fusion_info_سیٹ_enabled(ggml_metal_fusion_info * finfo, منطقی enabled) {
    finfo->enabled = enabled;
}

خالی ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    اگر (finfo->labels_سیٹ) {
        واپس;
    }

    finfo->labels.clear();
    finfo->counts.تفویض(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    برائے (ثابت ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_سیٹ = سچ;
}

خالی ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = سچ;
    ggml_metal_fusion_info_labels_init(finfo);
}

خالی ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

صحیح ggml_metal_fusion_info_stats_get(ثابت ggml_metal_fusion_info * finfo, ثابت حرف ** labels, uint64_t * counts, صحیح n) {
    ثابت صحیح n_fusions = (صحیح) finfo->labels.size();

    اگر (labels == خالی_اشاریہ) {
        واپس n_fusions;
    }

    ثابت صحیح n_fill = std::min(n, n_fusions);
    برائے (صحیح i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        اگر (counts != خالی_اشاریہ) {
            counts[i] = finfo->counts[i];
        }
    }

    واپس n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
ساکن منطقی ggml_metal_fusion_check_حافظہ_ranges(
        ثابت ggml_metal_fusion      * fusion,
        ثابت ggml_tensor * ثابت    * nodes,
        صحیح                            گرہ_گنتی) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    //ماخذ
    خودکار same_حافظہ = [](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        اگر (a->ڈیٹا && b->ڈیٹا && a->ڈیٹا == b->ڈیٹا) {
            واپس سچ;
        }
        برائے (ثابت ggml_tensor * v = a; v; v = v->view_src) {
            اگر (v == b) {
                واپس سچ;
            }
        }
        برائے (ثابت ggml_tensor * v = b; v; v = v->view_src) {
            اگر (v == a) {
                واپس سچ;
            }
        }
        واپس جھوٹ;
    };

    خودکار nodes_overlap = [](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        اگر (!a || !b || !a->ڈیٹا || !b->ڈیٹا || !a->بفر || !b->بفر) {
            واپس جھوٹ;
        }

        اگر (a->بفر != b->بفر) {
            واپس جھوٹ;
        }

        ثابت int64_t a_آغاز = (int64_t) a->ڈیٹا;
        ثابت int64_t a_انتہا   = a_آغاز + ggml_backend_buft_get_alloc_سائز(a->بفر->buft, a);
        ثابت int64_t b_آغاز = (int64_t) b->ڈیٹا;
        ثابت int64_t b_انتہا   = b_آغاز + ggml_backend_buft_get_alloc_سائز(b->بفر->buft, b);

        واپس (b_آغاز <= a_آغاز && a_آغاز < b_انتہا) ||
               (a_آغاز <= b_آغاز && b_آغاز < a_انتہا);
    };

    خودکار is_intermediate = [](ثابت ggml_tensor * src, ثابت ggml_tensor * ثابت * nodes, صحیح j) {
        برائے (صحیح k = 0; k < j; ++k) {
            اگر (src == nodes[k]) {
                واپس سچ;
            }
            برائے (ثابت ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                اگر (view_src == nodes[k]) {
                    واپس سچ;
                }
            }
        }
        واپس جھوٹ;
    };

    خودکار check_dst = [&](ثابت ggml_tensor * dst) {
        برائے (صحیح j = 0; j < گرہ_گنتی; ++j) {
            برائے (صحیح s = 0; s < GGML_MAX_SRC; ++s) {
                ثابت ggml_tensor * src = nodes[j]->src[s];
                اگر (!src || src->op == GGML_OP_NONE || same_حافظہ(src, dst)) {
                    جاری;
                }

                اگر (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    واپس جھوٹ;
                }
            }
        }
        واپس سچ;
    };

    اگر (!check_dst(nodes[گرہ_گنتی - 1])) {
        واپس جھوٹ;
    }

    برائے (صحیح انحراف : fusion->outs) {
        GGML_ASSERT(انحراف >= 0 && انحراف < گرہ_گنتی);
        اگر (!check_dst(nodes[انحراف])) {
            واپس جھوٹ;
        }
    }

    واپس سچ;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
ثابت ggml_metal_fusion * ggml_metal_fusion_اگلا(
        ثابت ggml_cgraph * gf,
        ثابت صحیح * گرہ_idxs,
        صحیح n_idxs,
        صحیح idx,
        ggml_metal_fusion_منوال mode,
        صحیح * n_برآمد) {
    ثابت ggml_metal_fusion * res = خالی_اشاریہ;
    صحیح بہترین = 1;

    برائے (ثابت ggml_metal_fusion & fusion : ggml_metal_fusions) {
        ثابت صحیح n_ops = (صحیح) fusion.ops.size();

        // only look for a longer match than the current best
        اگر (n_ops <= بہترین) {
            جاری;
        }
        اگر (idx + n_ops > n_idxs) {
            جاری;
        }

        ثابت ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        منطقی ok = سچ;
        برائے (صحیح j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[گرہ_idxs[idx + j]];
            اگر (nodes[j]->op != fusion.ops[j]) {
                ok = جھوٹ;
                توڑو;
            }
        }
        اگر (!ok) {
            جاری;
        }

        اگر (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            برائے (صحیح j = 1; j < n_ops && ok; j++) {
                اگر (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = جھوٹ;
                    توڑو;
                }
                اگر (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = جھوٹ;
                    توڑو;
                }
            }
            اگر (!ok) {
                جاری;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            صحیح outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = گرہ_idxs[idx + n_ops - 1];
            برائے (size_t i = 0; i < fusion.outs.size(); ++i) {
                ثابت صحیح برآمد_انحراف = fusion.outs[i];
                GGML_ASSERT(برآمد_انحراف >= 0 && برآمد_انحراف < n_ops);
                outputs_buf[i + 1] = گرہ_idxs[idx + برآمد_انحراف];
            }

            ثابت صحیح n_outputs = 1 + (صحیح) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            اگر (!ggml_can_fuse_subgraph_ext(gf, گرہ_idxs + idx, n_ops, fusion.ops.ڈیٹا(), outputs_buf, n_outputs)) {
                جاری;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        اگر (fusion.check && !fusion.check(&fusion, nodes, gf, گرہ_idxs, idx, mode)) {
            جاری;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        اگر (mode == GGML_METAL_FUSION_بھرا &&
                !ggml_metal_fusion_check_حافظہ_ranges(&fusion, nodes, n_ops)) {
            جاری;
        }

        بہترین = n_ops;
        res = &fusion;
    }

    *n_برآمد = بہترین;

    واپس res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
صحیح ggml_metal_fusion_max(ثابت ggml_cgraph * gf, صحیح idx) {
    // a view node cannot start a pattern - pack it alone
    اگر (ggml_op_is_خالی_خانے(gf->nodes[idx]->op)) {
        واپس 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    صحیح idxs[GGML_METAL_FUSION_MAX];
    صحیح n_idxs = 0;
    برائے (صحیح i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        اگر (!ggml_op_is_خالی_خانے(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    اگر (n_idxs == 0) {
        واپس 1;
    }

    صحیح کل = 0;
    صحیح i_f = 0;

    جب (i_f < n_idxs && کل < GGML_METAL_FUSION_MAX) {
        صحیح len = 1;
        ثابت ggml_metal_fusion * fusion = ggml_metal_fusion_اگلا(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        اگر (!fusion || کل + len > GGML_METAL_FUSION_MAX) {
            توڑو;
        }

        کل += len;
        i_f += len;
    }

    اگر (i_f == 0) {
        واپس 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    واپس std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
