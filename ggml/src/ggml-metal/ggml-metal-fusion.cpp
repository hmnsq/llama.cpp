#ಅನ್ತರ್ಭಾವ "ggml-metal-fusion.h"

#ಅನ್ತರ್ಭಾವ "ggml-metal-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-metal-device.h"

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <set>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
ಸ್ಥಿರ std::vector<ggml_op> ggml_metal_fusion_filter_ops(ಸ್ಥಿರಾಂಕ std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    ಗಾಗಿ (ggml_op op : ops_all) {
        ಆದರೆ (!ggml_op_is_ಖಾಲಿಗಳು(op)) {
            ops.push_back(op);
        }
    }

    ಹಿಂತಿರುಗಿ ops;
}

ರಚನೆ ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<ಪೂರ್ಣಾಂಕ>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    ತಾರ್ಕಿಕ unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    ತಾರ್ಕಿಕ (*check)(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_metal_fusion   * fusion,
                  ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಸ್ಥಿರಾಂಕ * nodes,
                  ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph         * gf,
                  ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                        * ಗಂಟು_idxs,
                        ಪೂರ್ಣಾಂಕ                          idx,
                        ggml_metal_fusion_ಬಹುಲಕ       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            ಸ್ಥಿರಾಂಕ std::vector<ggml_op> & ops_all,
            ಸ್ಥಿರಾಂಕ std::vector<ಪೂರ್ಣಾಂಕ> & outs,
            ತಾರ್ಕಿಕ unsafe,
            ತಾರ್ಕಿಕ (*check)(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_metal_fusion   * fusion,
                          ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಸ್ಥಿರಾಂಕ * nodes,
                          ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph         * gf,
                          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                        * ಗಂಟು_idxs,
                                ಪೂರ್ಣಾಂಕ                          idx,
                                ggml_metal_fusion_ಬಹುಲಕ       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(ಸ್ಥಿರಾಂಕ ggml_metal_fusion * fusion) {
    ಹಿಂತಿರುಗಿ fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_same_ಬಫರ್(ಸ್ಥಿರಾಂಕ ggml_tensor * a, ಸ್ಥಿರಾಂಕ ggml_tensor * b) {
    ಆದರೆ (!a || !b) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_backend_ಬಫರ್_t ba = a->view_src ? a->view_src->ಬಫರ್ : a->ಬಫರ್;
    ggml_backend_ಬಫರ್_t bb = b->view_src ? b->view_src->ಬಫರ್ : b->ಬಫರ್;

    ggml_metal_ಬಫರ್_t ca = (ggml_metal_ಬಫರ್_t) ba->context;
    ggml_metal_ಬಫರ್_t cb = (ggml_metal_ಬಫರ್_t) bb->context;

    ಹಿಂತಿರುಗಿ ggml_metal_ಬಫರ್_get_id(ca, a).metal == ggml_metal_ಬಫರ್_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_overlap(ಸ್ಥಿರಾಂಕ ggml_tensor * a, ಸ್ಥಿರಾಂಕ ggml_tensor * b) {
    ggml_backend_ಬಫರ್_t ba = a->view_src ? a->view_src->ಬಫರ್ : a->ಬಫರ್;
    ggml_backend_ಬಫರ್_t bb = b->view_src ? b->view_src->ಬಫರ್ : b->ಬಫರ್;

    ಸ್ಥಿರಾಂಕ ggml_metal_ಬಫರ್_id bid_a = ggml_metal_ಬಫರ್_get_id((ggml_metal_ಬಫರ್_t) ba->context, a);
    ಸ್ಥಿರಾಂಕ ggml_metal_ಬಫರ್_id bid_b = ggml_metal_ಬಫರ್_get_id((ggml_metal_ಬಫರ್_t) bb->context, b);

    ಆದರೆ (bid_a.metal == ಶೂನ್ಯಸೂಚಕ || bid_a.metal != bid_b.metal) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_norm(
        ಸ್ಥಿರಾಂಕ ggml_metal_fusion       * fusion,
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಸ್ಥಿರಾಂಕ     * nodes,
        ಸ್ಥಿರಾಂಕ ggml_cgraph             * gf,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಂಕ                       idx,
              ggml_metal_fusion_ಬಹುಲಕ    mode) {
    GGML_ಬಳಸದ(mode);
    GGML_ಬಳಸದ(gf);
    GGML_ಬಳಸದ(ಗಂಟು_idxs);
    GGML_ಬಳಸದ(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    ಆದರೆ (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        ಸ್ಥಿರಾಂಕ ggml_tensor * scale = nodes[1];
        ಆದರೆ (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 1; j < (ಪೂರ್ಣಾಂಕ) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        ಆದರೆ (nodes[j]->src[0] != nodes[j - 1]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // the weight/bias must have the same row width as the norm input
        ಆದರೆ (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (!ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(nodes[j]->src[1])) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (nodes[j]->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// SSM_CONV + UNARY (silu)
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_ssm_conv_silu(
        ಸ್ಥಿರಾಂಕ ggml_metal_fusion       * fusion,
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಸ್ಥಿರಾಂಕ     * nodes,
        ಸ್ಥಿರಾಂಕ ggml_cgraph             * gf,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಂಕ                       idx,
              ggml_metal_fusion_ಬಹುಲಕ    mode) {
    GGML_ಬಳಸದ(fusion);
    GGML_ಬಳಸದ(gf);
    GGML_ಬಳಸದ(ಗಂಟು_idxs);
    GGML_ಬಳಸದ(idx);
    GGML_ಬಳಸದ(mode);

    ಸ್ಥಿರಾಂಕ ggml_tensor * conv = nodes[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * un   = nodes[1];

    ಆದರೆ (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(un)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_ಕೂಡಿಸು_chain(
        ಸ್ಥಿರಾಂಕ ggml_metal_fusion       * fusion,
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಸ್ಥಿರಾಂಕ     * nodes,
        ಸ್ಥಿರಾಂಕ ggml_cgraph             * gf,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಂಕ                       idx,
              ggml_metal_fusion_ಬಹುಲಕ    mode) {
    GGML_ಬಳಸದ(gf);
    GGML_ಬಳಸದ(ಗಂಟು_idxs);
    GGML_ಬಳಸದ(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 1; j < (ಪೂರ್ಣಾಂಕ) fusion->ops.size(); j++) {
        ಆದರೆ (nodes[j]->src[0] != nodes[j - 1]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (mode == GGML_METAL_FUSION_ತುಂಬಿದ) {
            ಆದರೆ (!ggml_metal_fusion_same_ಬಫರ್(nodes[j]->src[1], nodes[0]->src[1])) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_gdn_cache(
        ಸ್ಥಿರಾಂಕ ggml_metal_fusion       * fusion,
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಸ್ಥಿರಾಂಕ     * nodes,
        ಸ್ಥಿರಾಂಕ ggml_cgraph             * gf,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಂಕ                       idx,
              ggml_metal_fusion_ಬಹುಲಕ    mode) {
    GGML_ಬಳಸದ(fusion);
    GGML_ಬಳಸದ(gf);
    GGML_ಬಳಸದ(ಗಂಟು_idxs);
    GGML_ಬಳಸದ(idx);

    ಸ್ಥಿರಾಂಕ ggml_tensor * gdn = nodes[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    ಆದರೆ (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int64_t S_v      = gdn->src[2]->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t H        = gdn->src[2]->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = gdn->src[2]->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t n_seqs   = gdn->src[2]->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t K        = ggml_get_op_params_i32(gdn, 0);
    ಸ್ಥಿರಾಂಕ size_t  ಬಾಲ_off = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(GGML_TYPE_F32, S_v * H * n_ಟೋಕನುಗಳು * n_seqs);

    ಸ್ಥಿರಾಂಕ int64_t D         = S_v * S_v * H;
    ಸ್ಥಿರಾಂಕ int64_t n_written = std::min<int64_t>(n_ಟೋಕನುಗಳು, K);

    ಸ್ಥಿರಾಂಕ ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    ಸ್ಥಿರಾಂಕ ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    ಆದರೆ (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != ಬಾಲ_off || !ggml_is_contiguous(src)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    ಆದರೆ (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_ಗಾತ್ರ(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(GGML_TYPE_F32, D)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (mode == GGML_METAL_FUSION_ತುಂಬಿದ) {
        // the cache must be allocated so the kernel can write straight to its buffer
        ಆದರೆ (dst->ದತ್ತಾಂಶ == ಶೂನ್ಯಸೂಚಕ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_snake(
        ಸ್ಥಿರಾಂಕ ggml_metal_fusion       * fusion,
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಸ್ಥಿರಾಂಕ     * nodes,
        ಸ್ಥಿರಾಂಕ ggml_cgraph             * gf,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಂಕ                       idx,
              ggml_metal_fusion_ಬಹುಲಕ    mode) {
    GGML_ಬಳಸದ(fusion);
    GGML_ಬಳಸದ(mode);
    GGML_ಬಳಸದ(gf);
    GGML_ಬಳಸದ(ಗಂಟು_idxs);
    GGML_ಬಳಸದ(idx);

    ಸ್ಥಿರಾಂಕ ggml_tensor * mul0     = nodes[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sin_ಗಂಟು = nodes[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sqr      = nodes[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * mul1     = nodes[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    ಸ್ಥಿರಾಂಕ ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    ಸ್ಥಿರಾಂಕ ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    ಸ್ಥಿರಾಂಕ ggml_tensor * x_ಒಳ_ಕೂಡಿಸು = (ಕೂಡಿಸು_2->src[0] == mul1) ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_ಗಂಟು->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (ಕೂಡಿಸು_2->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (ಕೂಡಿಸು_2->ne[2]   == 1) && (ಕೂಡಿಸು_2->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(ಕೂಡಿಸು_2) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    ಹಿಂತಿರುಗಿ types_ok && shape_ok && dim_ok && contig_ok && x_ಒಳ_ಕೂಡಿಸು == x;
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು
};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು, GGML_OP_SCALE
};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು,
    GGML_OP_RESHAPE, GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು,
    GGML_OP_RESHAPE, GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_topk_moe(
        ಸ್ಥಿರಾಂಕ ggml_metal_fusion       * fusion,
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಸ್ಥಿರಾಂಕ     * nodes,
        ಸ್ಥಿರಾಂಕ ggml_cgraph             * gf,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಂಕ                       idx,
              ggml_metal_fusion_ಬಹುಲಕ    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_ಬಳಸದ(nodes);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ops = (ಪೂರ್ಣಾಂಕ) fusion->ops.size();

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ with_norm  = n_ops >= 6;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    ಸ್ಥಿರಾಂಕ std::vector<ggml_op> & ops_all = fusion->ops_all;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ raw_ಆರಂಭ = ಗಂಟು_idxs[idx];
    ಪೂರ್ಣಾಂಕ       raw_ಕೊನೆ   = ಗಂಟು_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    ಆದರೆ (with_norm && !with_scale) {
        ಆದರೆ (raw_ಕೊನೆ + 1 >= gf->n_nodes) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ ggml_tensor * trailing_reshape = gf->nodes[raw_ಕೊನೆ + 1];
        ಆದರೆ (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_ಕೊನೆ]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        raw_ಕೊನೆ++;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ raw_ಎಣಿಕೆ = raw_ಕೊನೆ - raw_ಆರಂಭ + 1;
    ಆದರೆ (raw_ಎಣಿಕೆ != (ಪೂರ್ಣಾಂಕ) ops_all.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಪೂರ್ಣಾಂಕ raw_idxs[GGML_METAL_FUSION_MAX];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < raw_ಎಣಿಕೆ; ++i) {
        raw_idxs[i] = raw_ಆರಂಭ + i;
        ಆದರೆ (gf->nodes[raw_ಆರಂಭ + i]->op != ops_all[i]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * softmax        = gf->nodes[raw_ಆರಂಭ];
    ಸ್ಥಿರಾಂಕ ggml_tensor * probs_reshaped = gf->nodes[raw_ಆರಂಭ + 1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * argsort        = gf->nodes[raw_ಆರಂಭ + 2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ids            = gf->nodes[raw_ಆರಂಭ + 3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * get_ಅಡ್ಡಸಾಲುಗಳು       = gf->nodes[raw_ಆರಂಭ + 4];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಹೊರಹೋಗು            = gf->nodes[raw_ಕೊನೆ];
    ಸ್ಥಿರಾಂಕ ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    ದಶಮಾಂಶ scale   = 1.0f;
    ದಶಮಾಂಶ max_bias = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&scale,    ((ಸ್ಥಿರಾಂಕ int32_t *) softmax->op_params) + 0, ಗಾತ್ರ(scale));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias, ((ಸ್ಥಿರಾಂಕ int32_t *) softmax->op_params) + 1, ಗಾತ್ರ(max_bias));
    ಆದರೆ (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        ಹೊರಹೋಗು->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int64_t n_expert      = logits->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು      = logits->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_expert_ಬಳಸಿದ = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    ಆದರೆ (n_expert <= 0 || n_expert_ಬಳಸಿದ <= 0 || n_expert_ಬಳಸಿದ > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_ಬಳಸಿದ > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_ಟೋಕನುಗಳು || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        ಹೊರಹೋಗು->ne[0] != 1 || ಹೊರಹೋಗು->ne[1] != n_expert_ಬಳಸಿದ || ಹೊರಹೋಗು->ne[2] != n_ಟೋಕನುಗಳು || ಹೊರಹೋಗು->ne[3] != 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(logits) || !ggml_is_contiguous(ಹೊರಹೋಗು) ||
        ids->nb[0] != ggml_type_ಗಾತ್ರ(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_ಗಾತ್ರ(GGML_TYPE_I32) * n_expert) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_ಅಡ್ಡಸಾಲುಗಳು->src[0] != probs_reshaped || get_ಅಡ್ಡಸಾಲುಗಳು->src[1] != ids) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (with_norm) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * weights_reshaped = gf->nodes[raw_ಆರಂಭ + 5];
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು         = gf->nodes[raw_ಆರಂಭ + 6];
        ಸ್ಥಿರಾಂಕ ggml_tensor * clamp            = gf->nodes[raw_ಆರಂಭ + 7];
        ಸ್ಥಿರಾಂಕ ggml_tensor * div              = gf->nodes[raw_ಆರಂಭ + 8];
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಹೊರಹೋಗು_reshaped     = gf->nodes[raw_ಆರಂಭ + 9];

        ಆದರೆ (weights_reshaped->src[0] != get_ಅಡ್ಡಸಾಲುಗಳು || ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು->src[0] != weights_reshaped ||
            clamp->src[0] != ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            ಹೊರಹೋಗು_reshaped->src[0] != div) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (with_scale) {
            ಸ್ಥಿರಾಂಕ ggml_tensor * scale_ಗಂಟು = gf->nodes[raw_ಆರಂಭ + 10];
            ಆದರೆ (scale_ಗಂಟು->src[0] != ಹೊರಹೋಗು_reshaped) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (with_scale) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * scale_ಗಂಟು = gf->nodes[raw_ಆರಂಭ + 5];
        ಆದರೆ (scale_ಗಂಟು->src[0] != get_ಅಡ್ಡಸಾಲುಗಳು) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ outputs[2] = { raw_ಆರಂಭ + 3, raw_ಕೊನೆ };
    ಆದರೆ (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_ಎಣಿಕೆ, ops_all.ದತ್ತಾಂಶ(), outputs, 2)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (mode == GGML_METAL_FUSION_ತುಂಬಿದ) {
        ಆದರೆ (!logits->ದತ್ತಾಂಶ || !ಹೊರಹೋಗು->ದತ್ತಾಂಶ || !ids->ದತ್ತಾಂಶ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

ರಚನೆ ggml_metal_moe_reduce_ಹೊಂದಿಕೆ {
    ಸ್ಥಿರಾಂಕ ggml_tensor * experts;
    ಸ್ಥಿರಾಂಕ ggml_tensor * weights;
    ಸ್ಥಿರಾಂಕ ggml_tensor * dst;
    ಪೂರ್ಣಾಂಕ ಗಂಟು_ಎಣಿಕೆ;
};

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_ಹೊಂದಿಕೆ_moe_reduce(
        ಸ್ಥಿರಾಂಕ ggml_cgraph * gf, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_ಹೊಂದಿಕೆ * ಹೊಂದಿಕೆ) {
    ಆದರೆ (ಹೊಂದಿಕೆ == ಶೂನ್ಯಸೂಚಕ || ಗಂಟು_idx < 0 || ಗಂಟು_idx + (ಪೂರ್ಣಾಂಕ) ops_all.size() > gf->n_nodes) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * mul = gf->nodes[ಗಂಟು_idx];
    ಆದರೆ (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ raw_ಎಣಿಕೆ     = (ಪೂರ್ಣಾಂಕ) ops_all.size();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_expert_ಬಳಸಿದ = raw_ಎಣಿಕೆ / 2;

    ಆದರೆ (n_expert_ಬಳಸಿದ < 2 || n_expert_ಬಳಸಿದ > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_ಎಣಿಕೆ != 2 * n_expert_ಬಳಸಿದ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಪೂರ್ಣಾಂಕ n_views = 0;
    ಆಗ (ಗಂಟು_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[ಗಂಟು_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    ಆದರೆ (n_views != n_expert_ಬಳಸಿದ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = n_expert_ಬಳಸಿದ + 1; i < raw_ಎಣಿಕೆ; ++i) {
        ಆದರೆ (gf->nodes[ಗಂಟು_idx + i]->op != GGML_OP_ಕೂಡಿಸು) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಪೂರ್ಣಾಂಕ raw_idxs[GGML_METAL_FUSION_MAX];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < raw_ಎಣಿಕೆ; ++i) {
        raw_idxs[i] = ಗಂಟು_idx + i;
        ಆದರೆ (gf->nodes[ಗಂಟು_idx + i]->op != ops_all[i]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * experts = mul->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * weights = mul->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * dst     = gf->nodes[ಗಂಟು_idx + raw_ಎಣಿಕೆ - 1];

    ಆದರೆ (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int64_t n_embd   = experts->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    ಆದರೆ (n_embd <= 0 || experts->ne[1] != n_expert_ಬಳಸಿದ || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_ಬಳಸಿದ || weights->ne[2] != n_ಟೋಕನುಗಳು || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_ಟೋಕನುಗಳು || dst->ne[2] != 1 || dst->ne[3] != 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i <= n_expert_ಬಳಸಿದ; ++i) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * view = gf->nodes[ಗಂಟು_idx + i];
        ಆದರೆ (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_ಟೋಕನುಗಳು ||
            view->nb[1] != mul->nb[2]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * prev_ಕೂಡಿಸು = ಶೂನ್ಯಸೂಚಕ;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 1; j < n_expert_ಬಳಸಿದ; ++j) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2 = gf->nodes[ಗಂಟು_idx + n_expert_ಬಳಸಿದ + j];
        ಸ್ಥಿರಾಂಕ ggml_tensor * rhs = gf->nodes[ಗಂಟು_idx + j + 1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * lhs = j == 1 ? gf->nodes[ಗಂಟು_idx + 1] : prev_ಕೂಡಿಸು;
        ಆದರೆ (ಕೂಡಿಸು_2->src[0] != lhs || ಕೂಡಿಸು_2->src[1] != rhs) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        prev_ಕೂಡಿಸು = ಕೂಡಿಸು_2;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ outputs[1] = { ಗಂಟು_idx + raw_ಎಣಿಕೆ - 1 };
    ಆದರೆ (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_ಎಣಿಕೆ, ops_all.ದತ್ತಾಂಶ(), outputs, 1)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹೊಂದಿಕೆ->experts    = experts;
    ಹೊಂದಿಕೆ->weights    = weights;
    ಹೊಂದಿಕೆ->dst        = dst;
    ಹೊಂದಿಕೆ->ಗಂಟು_ಎಣಿಕೆ = raw_ಎಣಿಕೆ;
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_moe_reduce(
        ಸ್ಥಿರಾಂಕ ggml_metal_fusion       * fusion,
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಸ್ಥಿರಾಂಕ     * nodes,
        ಸ್ಥಿರಾಂಕ ggml_cgraph             * gf,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಂಕ                       idx,
              ggml_metal_fusion_ಬಹುಲಕ    mode) {
    GGML_ಬಳಸದ(nodes);

    ggml_metal_moe_reduce_ಹೊಂದಿಕೆ ಹೊಂದಿಕೆ;
    ಆದರೆ (!ggml_metal_fusion_ಹೊಂದಿಕೆ_moe_reduce(gf, ಗಂಟು_idxs[idx], fusion->ops_all, &ಹೊಂದಿಕೆ)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ ((ಪೂರ್ಣಾಂಕ) fusion->ops.size() != ಹೊಂದಿಕೆ.experts->ne[1]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ raw_ಕೊನೆ = ಗಂಟು_idxs[idx] + ಹೊಂದಿಕೆ.ಗಂಟು_ಎಣಿಕೆ - 1;
    ಆದರೆ (ಗಂಟು_idxs[idx + (ಪೂರ್ಣಾಂಕ) fusion->ops.size() - 1] != raw_ಕೊನೆ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (mode == GGML_METAL_FUSION_ತುಂಬಿದ) {
        ಆದರೆ (!ಹೊಂದಿಕೆ.experts->ದತ್ತಾಂಶ || !ಹೊಂದಿಕೆ.weights->ದತ್ತಾಂಶ || !ಹೊಂದಿಕೆ.dst->ದತ್ತಾಂಶ) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        ಆದರೆ (ggml_metal_fusion_overlap(ಹೊಂದಿಕೆ.dst, ಹೊಂದಿಕೆ.experts) || ggml_metal_fusion_overlap(ಹೊಂದಿಕೆ.dst, ಹೊಂದಿಕೆ.weights)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_tensor_is_ತೂಕ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * ತಳ = t->view_src != ಖಾಲಿ ? t->view_src : t;

    ಹಿಂತಿರುಗಿ ತಳ->ಬಫರ್ != ಖಾಲಿ && ggml_backend_ಬಫರ್_get_usage(ತಳ->ಬಫರ್) == GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ggml_metal_mul_mat_ಕೂಡಿಸು_ಕಾರ್ಯಾಂಶ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * mm, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಕೂಡಿಸು_2) {
    ಆದರೆ (ಕೂಡಿಸು_2->op != GGML_OP_ಕೂಡಿಸು || (ಕೂಡಿಸು_2->src[0] == mm) == (ಕೂಡಿಸು_2->src[1] == mm)) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * other = ಕೂಡಿಸು_2->src[0] == mm ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ok = other->type == GGML_TYPE_F32 && ಕೂಡಿಸು_2->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_ತೂಕ(other);

    ಹಿಂತಿರುಗಿ ok ? other : ಖಾಲಿ;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ggml_metal_mul_mat_ಕೂಡಿಸು_residual(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * mm, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಕೂಡಿಸು_2) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * res = ggml_metal_mul_mat_ಕೂಡಿಸು_ಕಾರ್ಯಾಂಶ(mm, ಕೂಡಿಸು_2);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ok = res != ಖಾಲಿ && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(ಕೂಡಿಸು_2);

    ಹಿಂತಿರುಗಿ ok ? res : ಖಾಲಿ;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_mul_mat_ಕೂಡಿಸು(
        ಸ್ಥಿರಾಂಕ ggml_metal_fusion       * fusion,
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಸ್ಥಿರಾಂಕ     * nodes,
        ಸ್ಥಿರಾಂಕ ggml_cgraph             * gf,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                     * ಗಂಟು_idxs,
              ಪೂರ್ಣಾಂಕ                       idx,
              ggml_metal_fusion_ಬಹುಲಕ    mode) {
    GGML_ಬಳಸದ(gf);
    GGML_ಬಳಸದ(ಗಂಟು_idxs);
    GGML_ಬಳಸದ(idx);
    GGML_ಬಳಸದ(fusion);

    ಸ್ಥಿರಾಂಕ ggml_tensor * mm  = nodes[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2 = nodes[1];

    ಆದರೆ (ggml_metal_mul_mat_ಕೂಡಿಸು_ಕಾರ್ಯಾಂಶ(mm, ಕೂಡಿಸು_2) == ಶೂನ್ಯಸೂಚಕ ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (mode == GGML_METAL_FUSION_STRUCTURAL) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * res = ggml_metal_mul_mat_ಕೂಡಿಸು_residual(mm, ಕೂಡಿಸು_2);

    ಆದರೆ (res == ಶೂನ್ಯಸೂಚಕ || !ggml_metal_op_mul_mat_use_mma(mm)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ !ggml_metal_fusion_overlap(ಕೂಡಿಸು_2, mm->src[0]) && !ggml_metal_fusion_overlap(ಕೂಡಿಸು_2, mm->src[1]) &&
        (ಕೂಡಿಸು_2->ದತ್ತಾಂಶ == res->ದತ್ತಾಂಶ || !ggml_metal_fusion_overlap(ಕೂಡಿಸು_2, res));
}

// ---- patterns ------------------------------------------------------------

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_norm_mul_ಕೂಡಿಸು     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ಕೂಡಿಸು };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_rms_norm_mul_ಕೂಡಿಸು = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ಕೂಡಿಸು };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_ಕೂಡಿಸು_2 = { GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_ಕೂಡಿಸು_3 = { GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_ಕೂಡಿಸು_4 = { GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_ಕೂಡಿಸು_5 = { GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_ಕೂಡಿಸು_6 = { GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_ಕೂಡಿಸು_7 = { GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ಕೂಡಿಸು };

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ಕೂಡಿಸು
};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು
};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು
};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು
};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು
};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು
};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_op> ops_mul_mat_ಕೂಡಿಸು = { GGML_OP_MUL_MAT, GGML_OP_ಕೂಡಿಸು };

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,       ops_norm_mul,               {},     ಸುಳ್ಳು, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ಕೂಡಿಸು,   ops_norm_mul_ಕೂಡಿಸು,           {},     ಸುಳ್ಳು, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_norm_scale,             {},     ಸುಳ್ಳು, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,       ops_rms_norm_mul,           {},     ಸುಳ್ಳು, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ಕೂಡಿಸು,   ops_rms_norm_mul_ಕೂಡಿಸು,       {},     ಸುಳ್ಳು, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,     ops_rms_norm_scale,         {},     ಸುಳ್ಳು, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ಕೂಡಿಸು_CHAIN,      ops_ಕೂಡಿಸು_2,                  {},     ಸುಳ್ಳು, ggml_metal_fusion_check_ಕೂಡಿಸು_chain },
    { GGML_METAL_FUSION_ಕೂಡಿಸು_CHAIN,      ops_ಕೂಡಿಸು_3,                  {},     ಸುಳ್ಳು, ggml_metal_fusion_check_ಕೂಡಿಸು_chain },
    { GGML_METAL_FUSION_ಕೂಡಿಸು_CHAIN,      ops_ಕೂಡಿಸು_4,                  {},     ಸುಳ್ಳು, ggml_metal_fusion_check_ಕೂಡಿಸು_chain },
    { GGML_METAL_FUSION_ಕೂಡಿಸು_CHAIN,      ops_ಕೂಡಿಸು_5,                  {},     ಸುಳ್ಳು, ggml_metal_fusion_check_ಕೂಡಿಸು_chain },
    { GGML_METAL_FUSION_ಕೂಡಿಸು_CHAIN,      ops_ಕೂಡಿಸು_6,                  {},     ಸುಳ್ಳು, ggml_metal_fusion_check_ಕೂಡಿಸು_chain },
    { GGML_METAL_FUSION_ಕೂಡಿಸು_CHAIN,      ops_ಕೂಡಿಸು_7,                  {},     ಸುಳ್ಳು, ggml_metal_fusion_check_ಕೂಡಿಸು_chain },
    { GGML_METAL_FUSION_SNAKE,          ops_snake,                  {},     ಸುಳ್ಳು, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,      ops_gdn_cache,              {},     ನಿಜ,  ggml_metal_fusion_check_gdn_cache },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe,               {1},    ನಿಜ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_scale,         {1},    ನಿಜ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm,          {1},    ನಿಜ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE,       ops_topk_moe_norm_scale,    {1},    ನಿಜ,  ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_2,           {},     ನಿಜ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_3,           {},     ನಿಜ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_4,           {},     ನಿಜ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_5,           {},     ನಿಜ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_6,           {},     ನಿಜ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_7,           {},     ನಿಜ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE,     ops_moe_reduce_8,           {},     ನಿಜ,  ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_SSM_CONV_SILU,  ops_ssm_conv_silu,          {},     ಸುಳ್ಳು, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_MUL_MAT_ಕೂಡಿಸು,    ops_mul_mat_ಕೂಡಿಸು,            {},     ಸುಳ್ಳು, ggml_metal_fusion_check_mul_mat_ಕೂಡಿಸು },
};

// ---- alloc deps -----------------------------------------------------------

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_ಹೊಂದಿಕೆ_raw_ನಮೂನೆ(
        ಸ್ಥಿರಾಂಕ ggml_cgraph * gf, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::vector<ggml_op> & ops) {
    ಆದರೆ (ಗಂಟು_idx < 0 || ಗಂಟು_idx + (ಪೂರ್ಣಾಂಕ) ops.size() > gf->n_nodes) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < (ಪೂರ್ಣಾಂಕ) ops.size(); ++i) {
        ಆದರೆ (gf->nodes[ಗಂಟು_idx + i]->op != ops[i]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_metal_fusion_ಕೂಡಿಸು_ನಮೂನೆ_alloc_deps(
        ಶೂನ್ಯ * user_ದತ್ತಾಂಶ,
        ಶೂನ್ಯ (*ಕೂಡಿಸು_alloc_dep)(ಶೂನ್ಯ *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        ಸ್ಥಿರಾಂಕ ggml_metal_fusion * fusion,
        ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೊನೆಯ_ಗಂಟು = ಗಂಟು_idx + (ಪೂರ್ಣಾಂಕ) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> ನೋಡಿದ;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < (ಪೂರ್ಣಾಂಕ) fusion->ops_all.size(); ++j) {
        ggml_tensor * ಗಂಟು = gf->nodes[ಗಂಟು_idx + j];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = ಗಂಟು->src[s];
            ಆದರೆ (src && ನೋಡಿದ.insert(src).second) {
                ಕೂಡಿಸು_alloc_dep(user_ದತ್ತಾಂಶ, src, gf->nodes[ಕೊನೆಯ_ಗಂಟು]);
            }
        }
        ನೋಡಿದ.insert(ಗಂಟು);
    }
}

ಶೂನ್ಯ ggml_metal_fusion_ಕೂಡಿಸು_alloc_deps(
        ಶೂನ್ಯ * user_ದತ್ತಾಂಶ,
        ಶೂನ್ಯ (*ಕೂಡಿಸು_alloc_dep)(ಶೂನ್ಯ *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < gf->n_nodes; ++i) {
        ಸ್ಥಿರಾಂಕ ggml_metal_fusion * ಉತ್ತಮ = ಶೂನ್ಯಸೂಚಕ;
        ಪೂರ್ಣಾಂಕ ಉತ್ತಮ_raw = 0;

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_metal_fusion & fusion : ggml_metal_fusions) {
            ಆದರೆ ((ಪೂರ್ಣಾಂಕ) fusion.ops_all.size() <= ಉತ್ತಮ_raw) {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (ggml_metal_fusion_ಹೊಂದಿಕೆ_raw_ನಮೂನೆ(gf, i, fusion.ops_all)) {
                ಉತ್ತಮ = &fusion;
                ಉತ್ತಮ_raw = (ಪೂರ್ಣಾಂಕ) fusion.ops_all.size();
            }
        }

        ಆದರೆ (ಉತ್ತಮ) {
            ggml_metal_fusion_ಕೂಡಿಸು_ನಮೂನೆ_alloc_deps(user_ದತ್ತಾಂಶ, ಕೂಡಿಸು_alloc_dep, gf, ಉತ್ತಮ, i);
            i += ಉತ್ತಮ_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

ಸ್ಥಿರ std::string ggml_metal_fusion_label(ಸ್ಥಿರಾಂಕ ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != ಶೂನ್ಯಸೂಚಕ);

    std::string label;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < (ಪೂರ್ಣಾಂಕ) fusion->ops.size(); j++) {
        ಆದರೆ (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    ಹಿಂತಿರುಗಿ label;
}

ರಚನೆ ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    ತಾರ್ಕಿಕ enabled;
    ತಾರ್ಕಿಕ stats;
    ತಾರ್ಕಿಕ labels_ಗಣ;
    ಪೂರ್ಣಾಂಕ  debug;
};

ರಚನೆ ggml_metal_fusion_info * ggml_metal_fusion_info_init(ತಾರ್ಕಿಕ enabled, ಪೂರ್ಣಾಂಕ debug) {
    ggml_metal_fusion_info * finfo = ಹೊಸ ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_ಗಣ = ಸುಳ್ಳು;
    finfo->debug      = debug;

    ಆದರೆ (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    ಹಿಂತಿರುಗಿ finfo;
}

ಶೂನ್ಯ ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    ಅಳಿಸು finfo;
}

ತಾರ್ಕಿಕ ggml_metal_fusion_info_enabled(ಸ್ಥಿರಾಂಕ ggml_metal_fusion_info * finfo) {
    ಹಿಂತಿರುಗಿ finfo->enabled;
}

ತಾರ್ಕಿಕ ggml_metal_fusion_info_stats(ಸ್ಥಿರಾಂಕ ggml_metal_fusion_info * finfo) {
    ಹಿಂತಿರುಗಿ finfo->stats;
}

ಪೂರ್ಣಾಂಕ ggml_metal_fusion_info_debug(ಸ್ಥಿರಾಂಕ ggml_metal_fusion_info * finfo) {
    ಹಿಂತಿರುಗಿ finfo->debug;
}

ಪೂರ್ಣಾಂಕ ggml_metal_fusion_info_n_fusions(ಸ್ಥಿರಾಂಕ ggml_metal_fusion_info * finfo) {
    ಹಿಂತಿರುಗಿ (ಪೂರ್ಣಾಂಕ) finfo->labels.size();
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_metal_fusion_info_label(ಸ್ಥಿರಾಂಕ ggml_metal_fusion_info * finfo, ಪೂರ್ಣಾಂಕ idx) {
    GGML_ASSERT(idx >= 0 && idx < (ಪೂರ್ಣಾಂಕ) finfo->labels.size());
    ಹಿಂತಿರುಗಿ finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_ಎಣಿಕೆ(ಸ್ಥಿರಾಂಕ ggml_metal_fusion_info * finfo, ಪೂರ್ಣಾಂಕ idx) {
    GGML_ASSERT(idx >= 0 && idx < (ಪೂರ್ಣಾಂಕ) finfo->counts.size());
    ಹಿಂತಿರುಗಿ finfo->counts[idx];
}

ಶೂನ್ಯ ggml_metal_fusion_info_ಎಣಿಕೆ_fusion(ggml_metal_fusion_info * finfo, ಸ್ಥಿರಾಂಕ ggml_metal_fusion * fusion) {
    ಆದರೆ (!finfo->stats || fusion == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ptrdiff_t idx = fusion - ggml_metal_fusions.ದತ್ತಾಂಶ();
    ಆದರೆ (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

ಶೂನ್ಯ ggml_metal_fusion_info_ಗಣ_enabled(ggml_metal_fusion_info * finfo, ತಾರ್ಕಿಕ enabled) {
    finfo->enabled = enabled;
}

ಶೂನ್ಯ ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    ಆದರೆ (finfo->labels_ಗಣ) {
        ಹಿಂತಿರುಗಿ;
    }

    finfo->labels.clear();
    finfo->counts.ನಿಯೋಜಿಸು(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_ಗಣ = ನಿಜ;
}

ಶೂನ್ಯ ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = ನಿಜ;
    ggml_metal_fusion_info_labels_init(finfo);
}

ಶೂನ್ಯ ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

ಪೂರ್ಣಾಂಕ ggml_metal_fusion_info_stats_get(ಸ್ಥಿರಾಂಕ ggml_metal_fusion_info * finfo, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ ** labels, uint64_t * counts, ಪೂರ್ಣಾಂಕ n) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_fusions = (ಪೂರ್ಣಾಂಕ) finfo->labels.size();

    ಆದರೆ (labels == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ n_fusions;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_fill = std::min(n, n_fusions);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        ಆದರೆ (counts != ಶೂನ್ಯಸೂಚಕ) {
            counts[i] = finfo->counts[i];
        }
    }

    ಹಿಂತಿರುಗಿ n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_metal_fusion_check_ಸ್ಮರಣೆ_ranges(
        ಸ್ಥಿರಾಂಕ ggml_metal_fusion      * fusion,
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಸ್ಥಿರಾಂಕ    * nodes,
        ಪೂರ್ಣಾಂಕ                            ಗಂಟು_ಎಣಿಕೆ) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    //ಮೂಲ
    ಸ್ವಯಂಚಾಲಿತ same_ಸ್ಮರಣೆ = [](ಸ್ಥಿರಾಂಕ ggml_tensor * a, ಸ್ಥಿರಾಂಕ ggml_tensor * b) {
        ಆದರೆ (a->ದತ್ತಾಂಶ && b->ದತ್ತಾಂಶ && a->ದತ್ತಾಂಶ == b->ದತ್ತಾಂಶ) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_tensor * v = a; v; v = v->view_src) {
            ಆದರೆ (v == b) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_tensor * v = b; v; v = v->view_src) {
            ಆದರೆ (v == a) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    };

    ಸ್ವಯಂಚಾಲಿತ nodes_overlap = [](ಸ್ಥಿರಾಂಕ ggml_tensor * a, ಸ್ಥಿರಾಂಕ ggml_tensor * b) {
        ಆದರೆ (!a || !b || !a->ದತ್ತಾಂಶ || !b->ದತ್ತಾಂಶ || !a->ಬಫರ್ || !b->ಬಫರ್) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (a->ಬಫರ್ != b->ಬಫರ್) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ int64_t a_ಆರಂಭ = (int64_t) a->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ int64_t a_ಕೊನೆ   = a_ಆರಂಭ + ggml_backend_buft_get_alloc_ಗಾತ್ರ(a->ಬಫರ್->buft, a);
        ಸ್ಥಿರಾಂಕ int64_t b_ಆರಂಭ = (int64_t) b->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ int64_t b_ಕೊನೆ   = b_ಆರಂಭ + ggml_backend_buft_get_alloc_ಗಾತ್ರ(b->ಬಫರ್->buft, b);

        ಹಿಂತಿರುಗಿ (b_ಆರಂಭ <= a_ಆರಂಭ && a_ಆರಂಭ < b_ಕೊನೆ) ||
               (a_ಆರಂಭ <= b_ಆರಂಭ && b_ಆರಂಭ < a_ಕೊನೆ);
    };

    ಸ್ವಯಂಚಾಲಿತ is_intermediate = [](ಸ್ಥಿರಾಂಕ ggml_tensor * src, ಸ್ಥಿರಾಂಕ ggml_tensor * ಸ್ಥಿರಾಂಕ * nodes, ಪೂರ್ಣಾಂಕ j) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < j; ++k) {
            ಆದರೆ (src == nodes[k]) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                ಆದರೆ (view_src == nodes[k]) {
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }
            }
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    };

    ಸ್ವಯಂಚಾಲಿತ check_dst = [&](ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ಗಂಟು_ಎಣಿಕೆ; ++j) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < GGML_MAX_SRC; ++s) {
                ಸ್ಥಿರಾಂಕ ggml_tensor * src = nodes[j]->src[s];
                ಆದರೆ (!src || src->op == GGML_OP_NONE || same_ಸ್ಮರಣೆ(src, dst)) {
                    ಮುಂದುವರಿ;
                }

                ಆದರೆ (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
            }
        }
        ಹಿಂತಿರುಗಿ ನಿಜ;
    };

    ಆದರೆ (!check_dst(nodes[ಗಂಟು_ಎಣಿಕೆ - 1])) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ : fusion->outs) {
        GGML_ASSERT(ಸ್ಥಾನಾಂತರ >= 0 && ಸ್ಥಾನಾಂತರ < ಗಂಟು_ಎಣಿಕೆ);
        ಆದರೆ (!check_dst(nodes[ಸ್ಥಾನಾಂತರ])) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
ಸ್ಥಿರಾಂಕ ggml_metal_fusion * ggml_metal_fusion_ಮುಂದಿನ(
        ಸ್ಥಿರಾಂಕ ggml_cgraph * gf,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ಗಂಟು_idxs,
        ಪೂರ್ಣಾಂಕ n_idxs,
        ಪೂರ್ಣಾಂಕ idx,
        ggml_metal_fusion_ಬಹುಲಕ mode,
        ಪೂರ್ಣಾಂಕ * n_ಹೊರಹೋಗು) {
    ಸ್ಥಿರಾಂಕ ggml_metal_fusion * res = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ ಉತ್ತಮ = 1;

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_metal_fusion & fusion : ggml_metal_fusions) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ops = (ಪೂರ್ಣಾಂಕ) fusion.ops.size();

        // only look for a longer match than the current best
        ಆದರೆ (n_ops <= ಉತ್ತಮ) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (idx + n_ops > n_idxs) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        ತಾರ್ಕಿಕ ok = ನಿಜ;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[ಗಂಟು_idxs[idx + j]];
            ಆದರೆ (nodes[j]->op != fusion.ops[j]) {
                ok = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }
        }
        ಆದರೆ (!ok) {
            ಮುಂದುವರಿ;
        }

        ಆದರೆ (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 1; j < n_ops && ok; j++) {
                ಆದರೆ (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
                }
                ಆದರೆ (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
                }
            }
            ಆದರೆ (!ok) {
                ಮುಂದುವರಿ;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            ಪೂರ್ಣಾಂಕ outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = ಗಂಟು_idxs[idx + n_ops - 1];
            ಗಾಗಿ (size_t i = 0; i < fusion.outs.size(); ++i) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_ಸ್ಥಾನಾಂತರ = fusion.outs[i];
                GGML_ASSERT(ಹೊರಹೋಗು_ಸ್ಥಾನಾಂತರ >= 0 && ಹೊರಹೋಗು_ಸ್ಥಾನಾಂತರ < n_ops);
                outputs_buf[i + 1] = ಗಂಟು_idxs[idx + ಹೊರಹೋಗು_ಸ್ಥಾನಾಂತರ];
            }

            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_outputs = 1 + (ಪೂರ್ಣಾಂಕ) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            ಆದರೆ (!ggml_can_fuse_subgraph_ext(gf, ಗಂಟು_idxs + idx, n_ops, fusion.ops.ದತ್ತಾಂಶ(), outputs_buf, n_outputs)) {
                ಮುಂದುವರಿ;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        ಆದರೆ (fusion.check && !fusion.check(&fusion, nodes, gf, ಗಂಟು_idxs, idx, mode)) {
            ಮುಂದುವರಿ;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        ಆದರೆ (mode == GGML_METAL_FUSION_ತುಂಬಿದ &&
                !ggml_metal_fusion_check_ಸ್ಮರಣೆ_ranges(&fusion, nodes, n_ops)) {
            ಮುಂದುವರಿ;
        }

        ಉತ್ತಮ = n_ops;
        res = &fusion;
    }

    *n_ಹೊರಹೋಗು = ಉತ್ತಮ;

    ಹಿಂತಿರುಗಿ res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
ಪೂರ್ಣಾಂಕ ggml_metal_fusion_max(ಸ್ಥಿರಾಂಕ ggml_cgraph * gf, ಪೂರ್ಣಾಂಕ idx) {
    // a view node cannot start a pattern - pack it alone
    ಆದರೆ (ggml_op_is_ಖಾಲಿಗಳು(gf->nodes[idx]->op)) {
        ಹಿಂತಿರುಗಿ 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    ಪೂರ್ಣಾಂಕ idxs[GGML_METAL_FUSION_MAX];
    ಪೂರ್ಣಾಂಕ n_idxs = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        ಆದರೆ (!ggml_op_is_ಖಾಲಿಗಳು(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    ಆದರೆ (n_idxs == 0) {
        ಹಿಂತಿರುಗಿ 1;
    }

    ಪೂರ್ಣಾಂಕ ಒಟ್ಟು = 0;
    ಪೂರ್ಣಾಂಕ i_f = 0;

    ಆಗ (i_f < n_idxs && ಒಟ್ಟು < GGML_METAL_FUSION_MAX) {
        ಪೂರ್ಣಾಂಕ len = 1;
        ಸ್ಥಿರಾಂಕ ggml_metal_fusion * fusion = ggml_metal_fusion_ಮುಂದಿನ(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        ಆದರೆ (!fusion || ಒಟ್ಟು + len > GGML_METAL_FUSION_MAX) {
            ನಿಲ್ಲಿಸು;
        }

        ಒಟ್ಟು += len;
        i_f += len;
    }

    ಆದರೆ (i_f == 0) {
        ಹಿಂತಿರುಗಿ 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    ಹಿಂತಿರುಗಿ std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
