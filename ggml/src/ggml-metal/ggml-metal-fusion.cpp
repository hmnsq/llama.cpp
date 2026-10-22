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

    لكل (ggml_op op : ops_all) {
        إذا (!ggml_op_is_خالی_تھاواں(op)) {
            ops.push_back(op);
        }
    }

    أرجع ops;
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
                  ثابت صحيح                        * گرہ_idxs,
                        صحيح                          idx,
                        ggml_metal_fusion_منوال       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            ثابت std::vector<ggml_op> & ops_all,
            ثابت std::vector<صحيح> & outs,
            منطقي unsafe,
            منطقي (*check)(ثابت بنية ggml_metal_fusion   * fusion,
                          ثابت بنية ggml_tensor * ثابت * nodes,
                          ثابت بنية ggml_cgraph         * gf,
                          ثابت صحيح                        * گرہ_idxs,
                                صحيح                          idx,
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
    أرجع fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
ساكن منطقي ggml_metal_fusion_same_بفر(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    إذا (!a || !b) {
        أرجع خطأ;
    }

    ggml_backend_بفر_t ba = a->view_src ? a->view_src->بفر : a->بفر;
    ggml_backend_بفر_t bb = b->view_src ? b->view_src->بفر : b->بفر;

    ggml_metal_بفر_t ca = (ggml_metal_بفر_t) ba->context;
    ggml_metal_بفر_t cb = (ggml_metal_بفر_t) bb->context;

    أرجع ggml_metal_بفر_get_id(ca, a).metal == ggml_metal_بفر_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
ساكن منطقي ggml_metal_fusion_overlap(ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
    ggml_backend_بفر_t ba = a->view_src ? a->view_src->بفر : a->بفر;
    ggml_backend_بفر_t bb = b->view_src ? b->view_src->بفر : b->بفر;

    ثابت ggml_metal_بفر_id bid_a = ggml_metal_بفر_get_id((ggml_metal_بفر_t) ba->context, a);
    ثابت ggml_metal_بفر_id bid_b = ggml_metal_بفر_get_id((ggml_metal_بفر_t) bb->context, b);

    إذا (bid_a.metal == خالی_سوچک || bid_a.metal != bid_b.metal) {
        أرجع خطأ;
    }

    أرجع bid_a.offs <= bid_b.offs
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
        ثابت صحيح                     * گرہ_idxs,
              صحيح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_بن_ورتیا(mode);
    GGML_بن_ورتیا(gf);
    GGML_بن_ورتیا(گرہ_idxs);
    GGML_بن_ورتیا(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    إذا (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        ثابت ggml_tensor * scale = nodes[1];
        إذا (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }

        أرجع سچ;
    }

    لكل (صحيح j = 1; j < (صحيح) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        إذا (nodes[j]->src[0] != nodes[j - 1]) {
            أرجع خطأ;
        }

        // the weight/bias must have the same row width as the norm input
        إذا (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            أرجع خطأ;
        }

        إذا (!ggml_is_contiguous_صفاں(nodes[j]->src[1])) {
            أرجع خطأ;
        }

        إذا (nodes[j]->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }
    }

    أرجع سچ;
}

// SSM_CONV + UNARY (silu)
ساكن منطقي ggml_metal_fusion_check_ssm_conv_silu(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحيح                     * گرہ_idxs,
              صحيح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_بن_ورتیا(fusion);
    GGML_بن_ورتیا(gf);
    GGML_بن_ورتیا(گرہ_idxs);
    GGML_بن_ورتیا(idx);
    GGML_بن_ورتیا(mode);

    ثابت ggml_tensor * conv = nodes[0];
    ثابت ggml_tensor * un   = nodes[1];

    إذا (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        أرجع خطأ;
    }

    إذا (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        أرجع خطأ;
    }

    إذا (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_صفاں(un)) {
        أرجع خطأ;
    }

    أرجع سچ;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
ساكن منطقي ggml_metal_fusion_check_جوڑو_chain(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحيح                     * گرہ_idxs,
              صحيح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_بن_ورتیا(gf);
    GGML_بن_ورتیا(گرہ_idxs);
    GGML_بن_ورتیا(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    لكل (صحيح j = 1; j < (صحيح) fusion->ops.size(); j++) {
        إذا (nodes[j]->src[0] != nodes[j - 1]) {
            أرجع خطأ;
        }

        إذا (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            أرجع خطأ;
        }

        إذا (mode == GGML_METAL_FUSION_بھریا) {
            إذا (!ggml_metal_fusion_same_بفر(nodes[j]->src[1], nodes[0]->src[1])) {
                أرجع خطأ;
            }
        }
    }

    أرجع سچ;
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
        ثابت صحيح                     * گرہ_idxs,
              صحيح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_بن_ورتیا(fusion);
    GGML_بن_ورتیا(gf);
    GGML_بن_ورتیا(گرہ_idxs);
    GGML_بن_ورتیا(idx);

    ثابت ggml_tensor * gdn = nodes[0];
    ثابت ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    إذا (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        أرجع خطأ;
    }

    إذا (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        أرجع خطأ;
    }

    ثابت int64_t S_v      = gdn->src[2]->ne[0];
    ثابت int64_t H        = gdn->src[2]->ne[1];
    ثابت int64_t n_ٹوکن = gdn->src[2]->ne[2];
    ثابت int64_t n_seqs   = gdn->src[2]->ne[3];
    ثابت int64_t K        = ggml_get_op_params_i32(gdn, 0);
    ثابت size_t  پوچھ_off = ggml_صف_ناپ(GGML_TYPE_F32, S_v * H * n_ٹوکن * n_seqs);

    ثابت int64_t D         = S_v * S_v * H;
    ثابت int64_t n_written = std::min<int64_t>(n_ٹوکن, K);

    ثابت ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    ثابت ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    إذا (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != پوچھ_off || !ggml_is_contiguous(src)) {
        أرجع خطأ;
    }

    ثابت int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    إذا (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_ناپ(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_صف_ناپ(GGML_TYPE_F32, D)) {
        أرجع خطأ;
    }

    إذا (mode == GGML_METAL_FUSION_بھریا) {
        // the cache must be allocated so the kernel can write straight to its buffer
        إذا (dst->انکڑے == خالی_سوچک) {
            أرجع خطأ;
        }
    }

    أرجع سچ;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
ساكن منطقي ggml_metal_fusion_check_snake(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحيح                     * گرہ_idxs,
              صحيح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_بن_ورتیا(fusion);
    GGML_بن_ورتیا(mode);
    GGML_بن_ورتیا(gf);
    GGML_بن_ورتیا(گرہ_idxs);
    GGML_بن_ورتیا(idx);

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
    ثابت ggml_tensor * x_وچ_جوڑو = (جوڑو_2->src[0] == mul1) ? جوڑو_2->src[1] : جوڑو_2->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    ثابت منطقي types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_گرہ->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (جوڑو_2->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    ثابت منطقي shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    ثابت منطقي dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (جوڑو_2->ne[2]   == 1) && (جوڑو_2->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    ثابت منطقي contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(جوڑو_2) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    أرجع types_ok && shape_ok && dim_ok && contig_ok && x_وچ_جوڑو == x;
}

#تعریف_کرو GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
ساكن ثابت std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_صفاں
};
ساكن ثابت std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_صفاں, GGML_OP_SCALE
};
ساكن ثابت std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_صفاں,
    GGML_OP_RESHAPE, GGML_OP_جوڑ_صفاں, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
ساكن ثابت std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_صفاں,
    GGML_OP_RESHAPE, GGML_OP_جوڑ_صفاں, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

ساكن منطقي ggml_metal_fusion_check_topk_moe(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحيح                     * گرہ_idxs,
              صحيح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_بن_ورتیا(nodes);

    ثابت صحيح n_ops = (صحيح) fusion->ops.size();

    ثابت منطقي with_norm  = n_ops >= 6;
    ثابت منطقي with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    ثابت std::vector<ggml_op> & ops_all = fusion->ops_all;

    ثابت صحيح raw_شروع = گرہ_idxs[idx];
    صحيح       raw_سرا   = گرہ_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    إذا (with_norm && !with_scale) {
        إذا (raw_سرا + 1 >= gf->n_nodes) {
            أرجع خطأ;
        }
        ثابت ggml_tensor * trailing_reshape = gf->nodes[raw_سرا + 1];
        إذا (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_سرا]) {
            أرجع خطأ;
        }
        raw_سرا++;
    }

    ثابت صحيح raw_گنتی = raw_سرا - raw_شروع + 1;
    إذا (raw_گنتی != (صحيح) ops_all.size()) {
        أرجع خطأ;
    }

    صحيح raw_idxs[GGML_METAL_FUSION_MAX];
    لكل (صحيح i = 0; i < raw_گنتی; ++i) {
        raw_idxs[i] = raw_شروع + i;
        إذا (gf->nodes[raw_شروع + i]->op != ops_all[i]) {
            أرجع خطأ;
        }
    }

    ثابت ggml_tensor * softmax        = gf->nodes[raw_شروع];
    ثابت ggml_tensor * probs_reshaped = gf->nodes[raw_شروع + 1];
    ثابت ggml_tensor * argsort        = gf->nodes[raw_شروع + 2];
    ثابت ggml_tensor * ids            = gf->nodes[raw_شروع + 3];
    ثابت ggml_tensor * get_صفاں       = gf->nodes[raw_شروع + 4];
    ثابت ggml_tensor * باہرلا            = gf->nodes[raw_سرا];
    ثابت ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    عائم scale   = 1.0f;
    عائم max_bias = 0.0f;
    حافظہ_نقل(&scale,    ((ثابت int32_t *) softmax->op_params) + 0, حجم(scale));
    حافظہ_نقل(&max_bias, ((ثابت int32_t *) softmax->op_params) + 1, حجم(max_bias));
    إذا (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        أرجع خطأ;
    }

    إذا (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        باہرلا->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        أرجع خطأ;
    }

    ثابت int64_t n_expert      = logits->ne[0];
    ثابت int64_t n_ٹوکن      = logits->ne[1];
    ثابت int64_t n_expert_ورتیا = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    إذا (n_expert <= 0 || n_expert_ورتیا <= 0 || n_expert_ورتیا > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_ورتیا > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        أرجع خطأ;
    }

    إذا (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_ٹوکن || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        باہرلا->ne[0] != 1 || باہرلا->ne[1] != n_expert_ورتیا || باہرلا->ne[2] != n_ٹوکن || باہرلا->ne[3] != 1) {
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(logits) || !ggml_is_contiguous(باہرلا) ||
        ids->nb[0] != ggml_type_ناپ(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_ناپ(GGML_TYPE_I32) * n_expert) {
        أرجع خطأ;
    }

    إذا (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_صفاں->src[0] != probs_reshaped || get_صفاں->src[1] != ids) {
        أرجع خطأ;
    }

    إذا (with_norm) {
        ثابت ggml_tensor * weights_reshaped = gf->nodes[raw_شروع + 5];
        ثابت ggml_tensor * جوڑ_صفاں         = gf->nodes[raw_شروع + 6];
        ثابت ggml_tensor * clamp            = gf->nodes[raw_شروع + 7];
        ثابت ggml_tensor * div              = gf->nodes[raw_شروع + 8];
        ثابت ggml_tensor * باہرلا_reshaped     = gf->nodes[raw_شروع + 9];

        إذا (weights_reshaped->src[0] != get_صفاں || جوڑ_صفاں->src[0] != weights_reshaped ||
            clamp->src[0] != جوڑ_صفاں || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            باہرلا_reshaped->src[0] != div) {
            أرجع خطأ;
        }

        إذا (with_scale) {
            ثابت ggml_tensor * scale_گرہ = gf->nodes[raw_شروع + 10];
            إذا (scale_گرہ->src[0] != باہرلا_reshaped) {
                أرجع خطأ;
            }
        }
    } وإلا إذا (with_scale) {
        ثابت ggml_tensor * scale_گرہ = gf->nodes[raw_شروع + 5];
        إذا (scale_گرہ->src[0] != get_صفاں) {
            أرجع خطأ;
        }
    }

    ثابت صحيح outputs[2] = { raw_شروع + 3, raw_سرا };
    إذا (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_گنتی, ops_all.انکڑے(), outputs, 2)) {
        أرجع خطأ;
    }

    إذا (mode == GGML_METAL_FUSION_بھریا) {
        إذا (!logits->انکڑے || !باہرلا->انکڑے || !ids->انکڑے) {
            أرجع خطأ;
        }
    }

    أرجع سچ;
}

#تعریف_کرو GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

بنية ggml_metal_moe_reduce_میل {
    ثابت ggml_tensor * experts;
    ثابت ggml_tensor * weights;
    ثابت ggml_tensor * dst;
    صحيح گرہ_گنتی;
};

ساكن منطقي ggml_metal_fusion_میل_moe_reduce(
        ثابت ggml_cgraph * gf, صحيح گرہ_idx, ثابت std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_میل * میل) {
    إذا (میل == خالی_سوچک || گرہ_idx < 0 || گرہ_idx + (صحيح) ops_all.size() > gf->n_nodes) {
        أرجع خطأ;
    }

    ثابت ggml_tensor * mul = gf->nodes[گرہ_idx];
    إذا (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    ثابت صحيح raw_گنتی     = (صحيح) ops_all.size();
    ثابت صحيح n_expert_ورتیا = raw_گنتی / 2;

    إذا (n_expert_ورتیا < 2 || n_expert_ورتیا > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_گنتی != 2 * n_expert_ورتیا) {
        أرجع خطأ;
    }

    صحيح n_views = 0;
    طالما (گرہ_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[گرہ_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    إذا (n_views != n_expert_ورتیا) {
        أرجع خطأ;
    }

    لكل (صحيح i = n_expert_ورتیا + 1; i < raw_گنتی; ++i) {
        إذا (gf->nodes[گرہ_idx + i]->op != GGML_OP_جوڑو) {
            أرجع خطأ;
        }
    }

    صحيح raw_idxs[GGML_METAL_FUSION_MAX];
    لكل (صحيح i = 0; i < raw_گنتی; ++i) {
        raw_idxs[i] = گرہ_idx + i;
        إذا (gf->nodes[گرہ_idx + i]->op != ops_all[i]) {
            أرجع خطأ;
        }
    }

    ثابت ggml_tensor * experts = mul->src[0];
    ثابت ggml_tensor * weights = mul->src[1];
    ثابت ggml_tensor * dst     = gf->nodes[گرہ_idx + raw_گنتی - 1];

    إذا (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }

    ثابت int64_t n_embd   = experts->ne[0];
    ثابت int64_t n_ٹوکن = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    إذا (n_embd <= 0 || experts->ne[1] != n_expert_ورتیا || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_ورتیا || weights->ne[2] != n_ٹوکن || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_ٹوکن || dst->ne[2] != 1 || dst->ne[3] != 1) {
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        أرجع خطأ;
    }

    لكل (صحيح i = 1; i <= n_expert_ورتیا; ++i) {
        ثابت ggml_tensor * view = gf->nodes[گرہ_idx + i];
        إذا (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_ٹوکن ||
            view->nb[1] != mul->nb[2]) {
            أرجع خطأ;
        }
    }

    ثابت ggml_tensor * prev_جوڑو = خالی_سوچک;
    لكل (صحيح j = 1; j < n_expert_ورتیا; ++j) {
        ثابت ggml_tensor * جوڑو_2 = gf->nodes[گرہ_idx + n_expert_ورتیا + j];
        ثابت ggml_tensor * rhs = gf->nodes[گرہ_idx + j + 1];
        ثابت ggml_tensor * lhs = j == 1 ? gf->nodes[گرہ_idx + 1] : prev_جوڑو;
        إذا (جوڑو_2->src[0] != lhs || جوڑو_2->src[1] != rhs) {
            أرجع خطأ;
        }
        prev_جوڑو = جوڑو_2;
    }

    ثابت صحيح outputs[1] = { گرہ_idx + raw_گنتی - 1 };
    إذا (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_گنتی, ops_all.انکڑے(), outputs, 1)) {
        أرجع خطأ;
    }

    میل->experts    = experts;
    میل->weights    = weights;
    میل->dst        = dst;
    میل->گرہ_گنتی = raw_گنتی;
    أرجع سچ;
}

ساكن منطقي ggml_metal_fusion_check_moe_reduce(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحيح                     * گرہ_idxs,
              صحيح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_بن_ورتیا(nodes);

    ggml_metal_moe_reduce_میل میل;
    إذا (!ggml_metal_fusion_میل_moe_reduce(gf, گرہ_idxs[idx], fusion->ops_all, &میل)) {
        أرجع خطأ;
    }

    إذا ((صحيح) fusion->ops.size() != میل.experts->ne[1]) {
        أرجع خطأ;
    }

    ثابت صحيح raw_سرا = گرہ_idxs[idx] + میل.گرہ_گنتی - 1;
    إذا (گرہ_idxs[idx + (صحيح) fusion->ops.size() - 1] != raw_سرا) {
        أرجع خطأ;
    }

    إذا (mode == GGML_METAL_FUSION_بھریا) {
        إذا (!میل.experts->انکڑے || !میل.weights->انکڑے || !میل.dst->انکڑے) {
            أرجع خطأ;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        إذا (ggml_metal_fusion_overlap(میل.dst, میل.experts) || ggml_metal_fusion_overlap(میل.dst, میل.weights)) {
            أرجع خطأ;
        }
    }

    أرجع سچ;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
ساكن منطقي ggml_metal_tensor_is_بھار(ثابت بنية ggml_tensor * t) {
    ثابت ggml_tensor * ادھار = t->view_src != عدم ? t->view_src : t;

    أرجع ادھار->بفر != عدم && ggml_backend_بفر_get_usage(ادھار->بفر) == GGML_BACKEND_بفر_USAGE_WEIGHTS;
}

ساكن ثابت بنية ggml_tensor * ggml_metal_mul_mat_جوڑو_عملیہ(ثابت بنية ggml_tensor * mm, ثابت بنية ggml_tensor * جوڑو_2) {
    إذا (جوڑو_2->op != GGML_OP_جوڑو || (جوڑو_2->src[0] == mm) == (جوڑو_2->src[1] == mm)) {
        أرجع عدم;
    }

    ثابت ggml_tensor * other = جوڑو_2->src[0] == mm ? جوڑو_2->src[1] : جوڑو_2->src[0];

    ثابت منطقي ok = other->type == GGML_TYPE_F32 && جوڑو_2->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_بھار(other);

    أرجع ok ? other : عدم;
}

ساكن ثابت بنية ggml_tensor * ggml_metal_mul_mat_جوڑو_residual(ثابت بنية ggml_tensor * mm, ثابت بنية ggml_tensor * جوڑو_2) {
    ثابت ggml_tensor * res = ggml_metal_mul_mat_جوڑو_عملیہ(mm, جوڑو_2);

    ثابت منطقي ok = res != عدم && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(جوڑو_2);

    أرجع ok ? res : عدم;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
ساكن منطقي ggml_metal_fusion_check_mul_mat_جوڑو(
        ثابت ggml_metal_fusion       * fusion,
        ثابت ggml_tensor * ثابت     * nodes,
        ثابت ggml_cgraph             * gf,
        ثابت صحيح                     * گرہ_idxs,
              صحيح                       idx,
              ggml_metal_fusion_منوال    mode) {
    GGML_بن_ورتیا(gf);
    GGML_بن_ورتیا(گرہ_idxs);
    GGML_بن_ورتیا(idx);
    GGML_بن_ورتیا(fusion);

    ثابت ggml_tensor * mm  = nodes[0];
    ثابت ggml_tensor * جوڑو_2 = nodes[1];

    إذا (ggml_metal_mul_mat_جوڑو_عملیہ(mm, جوڑو_2) == خالی_سوچک ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        أرجع خطأ;
    }

    إذا (mode == GGML_METAL_FUSION_STRUCTURAL) {
        أرجع سچ;
    }

    ثابت ggml_tensor * res = ggml_metal_mul_mat_جوڑو_residual(mm, جوڑو_2);

    إذا (res == خالی_سوچک || !ggml_metal_op_mul_mat_use_mma(mm)) {
        أرجع خطأ;
    }

    أرجع !ggml_metal_fusion_overlap(جوڑو_2, mm->src[0]) && !ggml_metal_fusion_overlap(جوڑو_2, mm->src[1]) &&
        (جوڑو_2->انکڑے == res->انکڑے || !ggml_metal_fusion_overlap(جوڑو_2, res));
}

// ---- patterns ------------------------------------------------------------

ساكن ثابت std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
ساكن ثابت std::vector<ggml_op> ops_norm_mul_جوڑو     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_جوڑو };
ساكن ثابت std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
ساكن ثابت std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
ساكن ثابت std::vector<ggml_op> ops_rms_norm_mul_جوڑو = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_جوڑو };
ساكن ثابت std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

ساكن ثابت std::vector<ggml_op> ops_جوڑو_2 = { GGML_OP_جوڑو, GGML_OP_جوڑو };
ساكن ثابت std::vector<ggml_op> ops_جوڑو_3 = { GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو };
ساكن ثابت std::vector<ggml_op> ops_جوڑو_4 = { GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو };
ساكن ثابت std::vector<ggml_op> ops_جوڑو_5 = { GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو };
ساكن ثابت std::vector<ggml_op> ops_جوڑو_6 = { GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو };
ساكن ثابت std::vector<ggml_op> ops_جوڑو_7 = { GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو };
ساكن ثابت std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_جوڑو };

ساكن ثابت std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

ساكن ثابت std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

ساكن ثابت std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_جوڑو
};
ساكن ثابت std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_جوڑو, GGML_OP_جوڑو
};
ساكن ثابت std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو
};
ساكن ثابت std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو
};
ساكن ثابت std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو
};
ساكن ثابت std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو
};
ساكن ثابت std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو, GGML_OP_جوڑو
};

ساكن ثابت std::vector<ggml_op> ops_mul_mat_جوڑو = { GGML_OP_MUL_MAT, GGML_OP_جوڑو };

ساكن ثابت std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     خطأ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_جوڑو,   ops_norm_mul_جوڑو,           {},     خطأ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     خطأ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     خطأ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_جوڑو,   ops_rms_norm_mul_جوڑو,       {},     خطأ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     خطأ, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_جوڑو_CHAIN,      ops_جوڑو_2,                  {},     خطأ, ggml_metal_fusion_check_جوڑو_chain },
    { GGML_METAL_FUSION_جوڑو_CHAIN,      ops_جوڑو_3,                  {},     خطأ, ggml_metal_fusion_check_جوڑو_chain },
    { GGML_METAL_FUSION_جوڑو_CHAIN,      ops_جوڑو_4,                  {},     خطأ, ggml_metal_fusion_check_جوڑو_chain },
    { GGML_METAL_FUSION_جوڑو_CHAIN,      ops_جوڑو_5,                  {},     خطأ, ggml_metal_fusion_check_جوڑو_chain },
    { GGML_METAL_FUSION_جوڑو_CHAIN,      ops_جوڑو_6,                  {},     خطأ, ggml_metal_fusion_check_جوڑو_chain },
    { GGML_METAL_FUSION_جوڑو_CHAIN,      ops_جوڑو_7,                  {},     خطأ, ggml_metal_fusion_check_جوڑو_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     خطأ, ggml_metal_fusion_check_snake },
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
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     خطأ, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_جوڑو,    ops_mul_mat_جوڑو,            {},     خطأ, ggml_metal_fusion_check_mul_mat_جوڑو },
};

// ---- alloc deps -----------------------------------------------------------

ساكن منطقي ggml_metal_fusion_میل_raw_نمونہ(
        ثابت ggml_cgraph * gf, صحيح گرہ_idx, ثابت std::vector<ggml_op> & ops) {
    إذا (گرہ_idx < 0 || گرہ_idx + (صحيح) ops.size() > gf->n_nodes) {
        أرجع خطأ;
    }

    لكل (صحيح i = 0; i < (صحيح) ops.size(); ++i) {
        إذا (gf->nodes[گرہ_idx + i]->op != ops[i]) {
            أرجع خطأ;
        }
    }

    أرجع سچ;
}

ساكن فراغ ggml_metal_fusion_جوڑو_نمونہ_alloc_deps(
        فراغ * user_انکڑے,
        فراغ (*جوڑو_alloc_dep)(فراغ *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        ثابت ggml_metal_fusion * fusion,
        صحيح گرہ_idx) {
    ثابت صحيح آخری_گرہ = گرہ_idx + (صحيح) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> تکیا;
    لكل (صحيح j = 0; j < (صحيح) fusion->ops_all.size(); ++j) {
        ggml_tensor * گرہ = gf->nodes[گرہ_idx + j];
        لكل (صحيح s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = گرہ->src[s];
            إذا (src && تکیا.insert(src).second) {
                جوڑو_alloc_dep(user_انکڑے, src, gf->nodes[آخری_گرہ]);
            }
        }
        تکیا.insert(گرہ);
    }
}

فراغ ggml_metal_fusion_جوڑو_alloc_deps(
        فراغ * user_انکڑے,
        فراغ (*جوڑو_alloc_dep)(فراغ *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    لكل (صحيح i = 0; i < gf->n_nodes; ++i) {
        ثابت ggml_metal_fusion * بہترین = خالی_سوچک;
        صحيح بہترین_raw = 0;

        لكل (ثابت ggml_metal_fusion & fusion : ggml_metal_fusions) {
            إذا ((صحيح) fusion.ops_all.size() <= بہترین_raw) {
                استمر;
            }
            إذا (ggml_metal_fusion_میل_raw_نمونہ(gf, i, fusion.ops_all)) {
                بہترین = &fusion;
                بہترین_raw = (صحيح) fusion.ops_all.size();
            }
        }

        إذا (بہترین) {
            ggml_metal_fusion_جوڑو_نمونہ_alloc_deps(user_انکڑے, جوڑو_alloc_dep, gf, بہترین, i);
            i += بہترین_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

ساكن std::string ggml_metal_fusion_label(ثابت ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != خالی_سوچک);

    std::string label;
    لكل (صحيح j = 0; j < (صحيح) fusion->ops.size(); j++) {
        إذا (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    أرجع label;
}

بنية ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    منطقي enabled;
    منطقي stats;
    منطقي labels_سیٹ;
    صحيح  debug;
};

بنية ggml_metal_fusion_info * ggml_metal_fusion_info_init(منطقي enabled, صحيح debug) {
    ggml_metal_fusion_info * finfo = جديد ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_سیٹ = خطأ;
    finfo->debug      = debug;

    إذا (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    أرجع finfo;
}

فراغ ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    مٹاؤ finfo;
}

منطقي ggml_metal_fusion_info_enabled(ثابت ggml_metal_fusion_info * finfo) {
    أرجع finfo->enabled;
}

منطقي ggml_metal_fusion_info_stats(ثابت ggml_metal_fusion_info * finfo) {
    أرجع finfo->stats;
}

صحيح ggml_metal_fusion_info_debug(ثابت ggml_metal_fusion_info * finfo) {
    أرجع finfo->debug;
}

صحيح ggml_metal_fusion_info_n_fusions(ثابت ggml_metal_fusion_info * finfo) {
    أرجع (صحيح) finfo->labels.size();
}

ثابت حرف * ggml_metal_fusion_info_label(ثابت ggml_metal_fusion_info * finfo, صحيح idx) {
    GGML_ASSERT(idx >= 0 && idx < (صحيح) finfo->labels.size());
    أرجع finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_گنتی(ثابت ggml_metal_fusion_info * finfo, صحيح idx) {
    GGML_ASSERT(idx >= 0 && idx < (صحيح) finfo->counts.size());
    أرجع finfo->counts[idx];
}

فراغ ggml_metal_fusion_info_گنتی_fusion(ggml_metal_fusion_info * finfo, ثابت ggml_metal_fusion * fusion) {
    إذا (!finfo->stats || fusion == خالی_سوچک) {
        أرجع;
    }

    ثابت ptrdiff_t idx = fusion - ggml_metal_fusions.انکڑے();
    إذا (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

فراغ ggml_metal_fusion_info_سیٹ_enabled(ggml_metal_fusion_info * finfo, منطقي enabled) {
    finfo->enabled = enabled;
}

فراغ ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    إذا (finfo->labels_سیٹ) {
        أرجع;
    }

    finfo->labels.clear();
    finfo->counts.سونپو(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    لكل (ثابت ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_سیٹ = سچ;
}

فراغ ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = سچ;
    ggml_metal_fusion_info_labels_init(finfo);
}

فراغ ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

صحيح ggml_metal_fusion_info_stats_get(ثابت ggml_metal_fusion_info * finfo, ثابت حرف ** labels, uint64_t * counts, صحيح n) {
    ثابت صحيح n_fusions = (صحيح) finfo->labels.size();

    إذا (labels == خالی_سوچک) {
        أرجع n_fusions;
    }

    ثابت صحيح n_fill = std::min(n, n_fusions);
    لكل (صحيح i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        إذا (counts != خالی_سوچک) {
            counts[i] = finfo->counts[i];
        }
    }

    أرجع n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
ساكن منطقي ggml_metal_fusion_check_حافظہ_ranges(
        ثابت ggml_metal_fusion      * fusion,
        ثابت ggml_tensor * ثابت    * nodes,
        صحيح                            گرہ_گنتی) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    //سروت
    خودکار same_حافظہ = [](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        إذا (a->انکڑے && b->انکڑے && a->انکڑے == b->انکڑے) {
            أرجع سچ;
        }
        لكل (ثابت ggml_tensor * v = a; v; v = v->view_src) {
            إذا (v == b) {
                أرجع سچ;
            }
        }
        لكل (ثابت ggml_tensor * v = b; v; v = v->view_src) {
            إذا (v == a) {
                أرجع سچ;
            }
        }
        أرجع خطأ;
    };

    خودکار nodes_overlap = [](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        إذا (!a || !b || !a->انکڑے || !b->انکڑے || !a->بفر || !b->بفر) {
            أرجع خطأ;
        }

        إذا (a->بفر != b->بفر) {
            أرجع خطأ;
        }

        ثابت int64_t a_شروع = (int64_t) a->انکڑے;
        ثابت int64_t a_سرا   = a_شروع + ggml_backend_buft_get_alloc_ناپ(a->بفر->buft, a);
        ثابت int64_t b_شروع = (int64_t) b->انکڑے;
        ثابت int64_t b_سرا   = b_شروع + ggml_backend_buft_get_alloc_ناپ(b->بفر->buft, b);

        أرجع (b_شروع <= a_شروع && a_شروع < b_سرا) ||
               (a_شروع <= b_شروع && b_شروع < a_سرا);
    };

    خودکار is_intermediate = [](ثابت ggml_tensor * src, ثابت ggml_tensor * ثابت * nodes, صحيح j) {
        لكل (صحيح k = 0; k < j; ++k) {
            إذا (src == nodes[k]) {
                أرجع سچ;
            }
            لكل (ثابت ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                إذا (view_src == nodes[k]) {
                    أرجع سچ;
                }
            }
        }
        أرجع خطأ;
    };

    خودکار check_dst = [&](ثابت ggml_tensor * dst) {
        لكل (صحيح j = 0; j < گرہ_گنتی; ++j) {
            لكل (صحيح s = 0; s < GGML_MAX_SRC; ++s) {
                ثابت ggml_tensor * src = nodes[j]->src[s];
                إذا (!src || src->op == GGML_OP_NONE || same_حافظہ(src, dst)) {
                    استمر;
                }

                إذا (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    أرجع خطأ;
                }
            }
        }
        أرجع سچ;
    };

    إذا (!check_dst(nodes[گرہ_گنتی - 1])) {
        أرجع خطأ;
    }

    لكل (صحيح وتھ : fusion->outs) {
        GGML_ASSERT(وتھ >= 0 && وتھ < گرہ_گنتی);
        إذا (!check_dst(nodes[وتھ])) {
            أرجع خطأ;
        }
    }

    أرجع سچ;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
ثابت ggml_metal_fusion * ggml_metal_fusion_اگلا(
        ثابت ggml_cgraph * gf,
        ثابت صحيح * گرہ_idxs,
        صحيح n_idxs,
        صحيح idx,
        ggml_metal_fusion_منوال mode,
        صحيح * n_باہرلا) {
    ثابت ggml_metal_fusion * res = خالی_سوچک;
    صحيح بہترین = 1;

    لكل (ثابت ggml_metal_fusion & fusion : ggml_metal_fusions) {
        ثابت صحيح n_ops = (صحيح) fusion.ops.size();

        // only look for a longer match than the current best
        إذا (n_ops <= بہترین) {
            استمر;
        }
        إذا (idx + n_ops > n_idxs) {
            استمر;
        }

        ثابت ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        منطقي ok = سچ;
        لكل (صحيح j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[گرہ_idxs[idx + j]];
            إذا (nodes[j]->op != fusion.ops[j]) {
                ok = خطأ;
                اكسر;
            }
        }
        إذا (!ok) {
            استمر;
        }

        إذا (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            لكل (صحيح j = 1; j < n_ops && ok; j++) {
                إذا (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = خطأ;
                    اكسر;
                }
                إذا (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = خطأ;
                    اكسر;
                }
            }
            إذا (!ok) {
                استمر;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            صحيح outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = گرہ_idxs[idx + n_ops - 1];
            لكل (size_t i = 0; i < fusion.outs.size(); ++i) {
                ثابت صحيح باہرلا_وتھ = fusion.outs[i];
                GGML_ASSERT(باہرلا_وتھ >= 0 && باہرلا_وتھ < n_ops);
                outputs_buf[i + 1] = گرہ_idxs[idx + باہرلا_وتھ];
            }

            ثابت صحيح n_outputs = 1 + (صحيح) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            إذا (!ggml_can_fuse_subgraph_ext(gf, گرہ_idxs + idx, n_ops, fusion.ops.انکڑے(), outputs_buf, n_outputs)) {
                استمر;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        إذا (fusion.check && !fusion.check(&fusion, nodes, gf, گرہ_idxs, idx, mode)) {
            استمر;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        إذا (mode == GGML_METAL_FUSION_بھریا &&
                !ggml_metal_fusion_check_حافظہ_ranges(&fusion, nodes, n_ops)) {
            استمر;
        }

        بہترین = n_ops;
        res = &fusion;
    }

    *n_باہرلا = بہترین;

    أرجع res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
صحيح ggml_metal_fusion_max(ثابت ggml_cgraph * gf, صحيح idx) {
    // a view node cannot start a pattern - pack it alone
    إذا (ggml_op_is_خالی_تھاواں(gf->nodes[idx]->op)) {
        أرجع 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    صحيح idxs[GGML_METAL_FUSION_MAX];
    صحيح n_idxs = 0;
    لكل (صحيح i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        إذا (!ggml_op_is_خالی_تھاواں(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    إذا (n_idxs == 0) {
        أرجع 1;
    }

    صحيح کل = 0;
    صحيح i_f = 0;

    طالما (i_f < n_idxs && کل < GGML_METAL_FUSION_MAX) {
        صحيح len = 1;
        ثابت ggml_metal_fusion * fusion = ggml_metal_fusion_اگلا(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        إذا (!fusion || کل + len > GGML_METAL_FUSION_MAX) {
            اكسر;
        }

        کل += len;
        i_f += len;
    }

    إذا (i_f == 0) {
        أرجع 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    أرجع std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
