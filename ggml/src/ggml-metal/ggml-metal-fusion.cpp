#include "ggml-metal-fusion.h"

#include "ggml-metal-common.h"
#include "ggml-metal-device.h"

#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <set>
#include <string>
#include <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
static std::vector<ggml_op> ggml_metal_fusion_filter_ops(const std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    ཆེད་དུ (ggml_op op : ops_all) {
        གལ་ཏེ (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    ཕྱིར ops;
}

sa_nracanaa ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<ཧྲིལ>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    bool unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    bool (*check)(const sa_nracanaa ggml_metal_fusion   * fusion,
                  const sa_nracanaa ggml_tensor * const * nodes,
                  const sa_nracanaa ggml_cgraph         * gf,
                  const ཧྲིལ                        * node_idxs,
                        ཧྲིལ                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            const std::vector<ggml_op> & ops_all,
            const std::vector<ཧྲིལ> & outs,
            bool unsafe,
            bool (*check)(const sa_nracanaa ggml_metal_fusion   * fusion,
                          const sa_nracanaa ggml_tensor * const * nodes,
                          const sa_nracanaa ggml_cgraph         * gf,
                          const ཧྲིལ                        * node_idxs,
                                ཧྲིལ                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(const ggml_metal_fusion * fusion) {
    ཕྱིར fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
static bool ggml_metal_fusion_same_buffer(const ggml_tensor * a, const ggml_tensor * b) {
    གལ་ཏེ (!a || !b) {
        ཕྱིར false;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    ཕྱིར ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
static bool ggml_metal_fusion_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    const ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    const ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    གལ་ཏེ (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        ཕྱིར false;
    }

    ཕྱིར bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
static bool ggml_metal_fusion_check_norm(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const ཧྲིལ                     * node_idxs,
              ཧྲིལ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    གལ་ཏེ (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        const ggml_tensor * scale = nodes[1];
        གལ་ཏེ (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            ཕྱིར false;
        }

        ཕྱིར true;
    }

    ཆེད་དུ (ཧྲིལ j = 1; j < (ཧྲིལ) fusion->ops.size(); j++) {
        // the fused MUL/ADD must read the previous node as src0
        གལ་ཏེ (nodes[j]->src[0] != nodes[j - 1]) {
            ཕྱིར false;
        }

        // the weight/bias must have the same row width as the norm input
        གལ་ཏེ (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            ཕྱིར false;
        }

        གལ་ཏེ (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            ཕྱིར false;
        }

        གལ་ཏེ (nodes[j]->type != GGML_TYPE_F32) {
            ཕྱིར false;
        }
    }

    ཕྱིར true;
}

// SSM_CONV + UNARY (silu)
static bool ggml_metal_fusion_check_ssm_conv_silu(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const ཧྲིལ                     * node_idxs,
              ཧྲིལ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const ggml_tensor * conv = nodes[0];
    const ggml_tensor * un   = nodes[1];

    གལ་ཏེ (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        ཕྱིར false;
    }

    གལ་ཏེ (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        ཕྱིར false;
    }

    གལ་ཏེ (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        ཕྱིར false;
    }

    ཕྱིར true;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
static bool ggml_metal_fusion_check_add_chain(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const ཧྲིལ                     * node_idxs,
              ཧྲིལ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    ཆེད་དུ (ཧྲིལ j = 1; j < (ཧྲིལ) fusion->ops.size(); j++) {
        གལ་ཏེ (nodes[j]->src[0] != nodes[j - 1]) {
            ཕྱིར false;
        }

        གལ་ཏེ (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            ཕྱིར false;
        }

        གལ་ཏེ (mode == GGML_METAL_FUSION_FULL) {
            གལ་ཏེ (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                ཕྱིར false;
            }
        }
    }

    ཕྱིར true;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
static bool ggml_metal_fusion_check_gdn_cache(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const ཧྲིལ                     * node_idxs,
              ཧྲིལ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * gdn = nodes[0];
    const ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    གལ་ཏེ (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ཕྱིར false;
    }

    གལ་ཏེ (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ཕྱིར false;
    }

    const int64_t S_v      = gdn->src[2]->ne[0];
    const int64_t H        = gdn->src[2]->ne[1];
    const int64_t n_tokens = gdn->src[2]->ne[2];
    const int64_t n_seqs   = gdn->src[2]->ne[3];
    const int64_t K        = ggml_get_op_params_i32(gdn, 0);
    const size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    const int64_t D         = S_v * S_v * H;
    const int64_t n_written = std::min<int64_t>(n_tokens, K);

    const ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    const ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    གལ་ཏེ (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        ཕྱིར false;
    }

    const int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    གལ་ཏེ (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        ཕྱིར false;
    }

    གལ་ཏེ (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        གལ་ཏེ (dst->data == nullptr) {
            ཕྱིར false;
        }
    }

    ཕྱིར true;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
static bool ggml_metal_fusion_check_snake(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const ཧྲིལ                     * node_idxs,
              ཧྲིལ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * mul0     = nodes[0];
    const ggml_tensor * sin_node = nodes[1];
    const ggml_tensor * sqr      = nodes[2];
    const ggml_tensor * mul1     = nodes[3];
    const ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    const bool types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_node->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    const bool dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    const bool contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    ཕྱིར types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

// SOFT_MAX + ARGSORT + GET_ROWS (plus optional norm/scale) for MoE routing.
// This is a multi-output elision chain: the fused kernel writes both the selected
// expert ids and the gathered/normalized routing weights.
static const std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static bool ggml_metal_fusion_check_topk_moe(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const ཧྲིལ                     * node_idxs,
              ཧྲིལ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    const ཧྲིལ n_ops = (ཧྲིལ) fusion->ops.size();

    const bool with_norm  = n_ops >= 6;
    const bool with_scale = n_ops == 4 || n_ops == 7;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    const std::vector<ggml_op> & ops_all = fusion->ops_all;

    const ཧྲིལ raw_start = node_idxs[idx];
    ཧྲིལ       raw_end   = node_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    གལ་ཏེ (with_norm && !with_scale) {
        གལ་ཏེ (raw_end + 1 >= gf->n_nodes) {
            ཕྱིར false;
        }
        const ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        གལ་ཏེ (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            ཕྱིར false;
        }
        raw_end++;
    }

    const ཧྲིལ raw_count = raw_end - raw_start + 1;
    གལ་ཏེ (raw_count != (ཧྲིལ) ops_all.size()) {
        ཕྱིར false;
    }

    ཧྲིལ raw_idxs[GGML_METAL_FUSION_MAX];
    ཆེད་དུ (ཧྲིལ i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        གལ་ཏེ (gf->nodes[raw_start + i]->op != ops_all[i]) {
            ཕྱིར false;
        }
    }

    const ggml_tensor * softmax        = gf->nodes[raw_start];
    const ggml_tensor * probs_reshaped = gf->nodes[raw_start + 1];
    const ggml_tensor * argsort        = gf->nodes[raw_start + 2];
    const ggml_tensor * ids            = gf->nodes[raw_start + 3];
    const ggml_tensor * get_rows       = gf->nodes[raw_start + 4];
    const ggml_tensor * out            = gf->nodes[raw_end];
    const ggml_tensor * logits         = softmax->src[0];

    // the fused kernel implements plain softmax only
    float scale   = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    ((const int32_t *) softmax->op_params) + 0, parimaa_na(scale));
    memcpy(&max_bias, ((const int32_t *) softmax->op_params) + 1, parimaa_na(max_bias));
    གལ་ཏེ (scale != 1.0f || max_bias != 0.0f || softmax->src[1] || softmax->src[2]) {
        ཕྱིར false;
    }

    གལ་ཏེ (logits->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        ཕྱིར false;
    }

    const int64_t n_expert      = logits->ne[0];
    const int64_t n_tokens      = logits->ne[1];
    const int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    གལ་ཏེ (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        ཕྱིར false;
    }

    གལ་ཏེ (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        ཕྱིར false;
    }

    གལ་ཏེ (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        ཕྱིར false;
    }

    གལ་ཏེ (probs_reshaped->src[0] != softmax || argsort->src[0] != softmax ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        ཕྱིར false;
    }

    གལ་ཏེ (with_norm) {
        const ggml_tensor * weights_reshaped = gf->nodes[raw_start + 5];
        const ggml_tensor * sum_rows         = gf->nodes[raw_start + 6];
        const ggml_tensor * clamp            = gf->nodes[raw_start + 7];
        const ggml_tensor * div              = gf->nodes[raw_start + 8];
        const ggml_tensor * out_reshaped     = gf->nodes[raw_start + 9];

        གལ་ཏེ (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            clamp->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != clamp ||
            out_reshaped->src[0] != div) {
            ཕྱིར false;
        }

        གལ་ཏེ (with_scale) {
            const ggml_tensor * scale_node = gf->nodes[raw_start + 10];
            གལ་ཏེ (scale_node->src[0] != out_reshaped) {
                ཕྱིར false;
            }
        }
    } དེ་མིན གལ་ཏེ (with_scale) {
        const ggml_tensor * scale_node = gf->nodes[raw_start + 5];
        གལ་ཏེ (scale_node->src[0] != get_rows) {
            ཕྱིར false;
        }
    }

    const ཧྲིལ outputs[2] = { raw_start + 3, raw_end };
    གལ་ཏེ (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 2)) {
        ཕྱིར false;
    }

    གལ་ཏེ (mode == GGML_METAL_FUSION_FULL) {
        གལ་ཏེ (!logits->data || !out->data || !ids->data) {
            ཕྱིར false;
        }
    }

    ཕྱིར true;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

sa_nracanaa ggml_metal_moe_reduce_match {
    const ggml_tensor * experts;
    const ggml_tensor * weights;
    const ggml_tensor * dst;
    ཧྲིལ node_count;
};

static bool ggml_metal_fusion_match_moe_reduce(
        const ggml_cgraph * gf, ཧྲིལ node_idx, const std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    གལ་ཏེ (match == nullptr || node_idx < 0 || node_idx + (ཧྲིལ) ops_all.size() > gf->n_nodes) {
        ཕྱིར false;
    }

    const ggml_tensor * mul = gf->nodes[node_idx];
    གལ་ཏེ (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        ཕྱིར false;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    const ཧྲིལ raw_count     = (ཧྲིལ) ops_all.size();
    const ཧྲིལ n_expert_used = raw_count / 2;

    གལ་ཏེ (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_count != 2 * n_expert_used) {
        ཕྱིར false;
    }

    ཧྲིལ n_views = 0;
    སྐབས (node_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[node_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    གལ་ཏེ (n_views != n_expert_used) {
        ཕྱིར false;
    }

    ཆེད་དུ (ཧྲིལ i = n_expert_used + 1; i < raw_count; ++i) {
        གལ་ཏེ (gf->nodes[node_idx + i]->op != GGML_OP_ADD) {
            ཕྱིར false;
        }
    }

    ཧྲིལ raw_idxs[GGML_METAL_FUSION_MAX];
    ཆེད་དུ (ཧྲིལ i = 0; i < raw_count; ++i) {
        raw_idxs[i] = node_idx + i;
        གལ་ཏེ (gf->nodes[node_idx + i]->op != ops_all[i]) {
            ཕྱིར false;
        }
    }

    const ggml_tensor * experts = mul->src[0];
    const ggml_tensor * weights = mul->src[1];
    const ggml_tensor * dst     = gf->nodes[node_idx + raw_count - 1];

    གལ་ཏེ (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ཕྱིར false;
    }

    const int64_t n_embd   = experts->ne[0];
    const int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    གལ་ཏེ (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        ཕྱིར false;
    }

    གལ་ཏེ (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        ཕྱིར false;
    }

    ཆེད་དུ (ཧྲིལ i = 1; i <= n_expert_used; ++i) {
        const ggml_tensor * view = gf->nodes[node_idx + i];
        གལ་ཏེ (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            ཕྱིར false;
        }
    }

    const ggml_tensor * prev_add = nullptr;
    ཆེད་དུ (ཧྲིལ j = 1; j < n_expert_used; ++j) {
        const ggml_tensor * add = gf->nodes[node_idx + n_expert_used + j];
        const ggml_tensor * rhs = gf->nodes[node_idx + j + 1];
        const ggml_tensor * lhs = j == 1 ? gf->nodes[node_idx + 1] : prev_add;
        གལ་ཏེ (add->src[0] != lhs || add->src[1] != rhs) {
            ཕྱིར false;
        }
        prev_add = add;
    }

    const ཧྲིལ outputs[1] = { node_idx + raw_count - 1 };
    གལ་ཏེ (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 1)) {
        ཕྱིར false;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->node_count = raw_count;
    ཕྱིར true;
}

static bool ggml_metal_fusion_check_moe_reduce(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const ཧྲིལ                     * node_idxs,
              ཧྲིལ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    གལ་ཏེ (!ggml_metal_fusion_match_moe_reduce(gf, node_idxs[idx], fusion->ops_all, &match)) {
        ཕྱིར false;
    }

    གལ་ཏེ ((ཧྲིལ) fusion->ops.size() != match.experts->ne[1]) {
        ཕྱིར false;
    }

    const ཧྲིལ raw_end = node_idxs[idx] + match.node_count - 1;
    གལ་ཏེ (node_idxs[idx + (ཧྲིལ) fusion->ops.size() - 1] != raw_end) {
        ཕྱིར false;
    }

    གལ་ཏེ (mode == GGML_METAL_FUSION_FULL) {
        གལ་ཏེ (!match.experts->data || !match.weights->data || !match.dst->data) {
            ཕྱིར false;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        གལ་ཏེ (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            ཕྱིར false;
        }
    }

    ཕྱིར true;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
static bool ggml_metal_tensor_is_weight(const sa_nracanaa ggml_tensor * t) {
    const ggml_tensor * base = t->view_src != NULL ? t->view_src : t;

    ཕྱིར base->buffer != NULL && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static const sa_nracanaa ggml_tensor * ggml_metal_mul_mat_add_operand(const sa_nracanaa ggml_tensor * mm, const sa_nracanaa ggml_tensor * add) {
    གལ་ཏེ (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        ཕྱིར NULL;
    }

    const ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    const bool ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    ཕྱིར ok ? other : NULL;
}

static const sa_nracanaa ggml_tensor * ggml_metal_mul_mat_add_residual(const sa_nracanaa ggml_tensor * mm, const sa_nracanaa ggml_tensor * add) {
    const ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    const bool ok = res != NULL && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    ཕྱིར ok ? res : NULL;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
static bool ggml_metal_fusion_check_mul_mat_add(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const ཧྲིལ                     * node_idxs,
              ཧྲིལ                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    const ggml_tensor * mm  = nodes[0];
    const ggml_tensor * add = nodes[1];

    གལ་ཏེ (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        ཕྱིར false;
    }

    གལ་ཏེ (mode == GGML_METAL_FUSION_STRUCTURAL) {
        ཕྱིར true;
    }

    const ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    གལ་ཏེ (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        ཕྱིར false;
    }

    ཕྱིར !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

static const std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
static const std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
static const std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
static const std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

static const std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

static const std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

static const std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

static const std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

static const std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };

static const std::vector<ggml_metal_fusion> ggml_metal_fusions = {
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

static bool ggml_metal_fusion_match_raw_pattern(
        const ggml_cgraph * gf, ཧྲིལ node_idx, const std::vector<ggml_op> & ops) {
    གལ་ཏེ (node_idx < 0 || node_idx + (ཧྲིལ) ops.size() > gf->n_nodes) {
        ཕྱིར false;
    }

    ཆེད་དུ (ཧྲིལ i = 0; i < (ཧྲིལ) ops.size(); ++i) {
        གལ་ཏེ (gf->nodes[node_idx + i]->op != ops[i]) {
            ཕྱིར false;
        }
    }

    ཕྱིར true;
}

static void ggml_metal_fusion_add_pattern_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        const ggml_metal_fusion * fusion,
        ཧྲིལ node_idx) {
    const ཧྲིལ last_node = node_idx + (ཧྲིལ) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    ཆེད་དུ (ཧྲིལ j = 0; j < (ཧྲིལ) fusion->ops_all.size(); ++j) {
        ggml_tensor * node = gf->nodes[node_idx + j];
        ཆེད་དུ (ཧྲིལ s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = node->src[s];
            གལ་ཏེ (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_node]);
            }
        }
        seen.insert(node);
    }
}

void ggml_metal_fusion_add_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    ཆེད་དུ (ཧྲིལ i = 0; i < gf->n_nodes; ++i) {
        const ggml_metal_fusion * best = nullptr;
        ཧྲིལ best_raw = 0;

        ཆེད་དུ (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
            གལ་ཏེ ((ཧྲིལ) fusion.ops_all.size() <= best_raw) {
                continue;
            }
            གལ་ཏེ (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (ཧྲིལ) fusion.ops_all.size();
            }
        }

        གལ་ཏེ (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

static std::string ggml_metal_fusion_label(const ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    ཆེད་དུ (ཧྲིལ j = 0; j < (ཧྲིལ) fusion->ops.size(); j++) {
        གལ་ཏེ (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    ཕྱིར label;
}

sa_nracanaa ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    bool enabled;
    bool stats;
    bool labels_set;
    ཧྲིལ  debug;
};

sa_nracanaa ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, ཧྲིལ debug) {
    ggml_metal_fusion_info * finfo = new ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = false;
    finfo->debug      = debug;

    གལ་ཏེ (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    ཕྱིར finfo;
}

void ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

bool ggml_metal_fusion_info_enabled(const ggml_metal_fusion_info * finfo) {
    ཕྱིར finfo->enabled;
}

bool ggml_metal_fusion_info_stats(const ggml_metal_fusion_info * finfo) {
    ཕྱིར finfo->stats;
}

ཧྲིལ ggml_metal_fusion_info_debug(const ggml_metal_fusion_info * finfo) {
    ཕྱིར finfo->debug;
}

ཧྲིལ ggml_metal_fusion_info_n_fusions(const ggml_metal_fusion_info * finfo) {
    ཕྱིར (ཧྲིལ) finfo->labels.size();
}

const var_na * ggml_metal_fusion_info_label(const ggml_metal_fusion_info * finfo, ཧྲིལ idx) {
    GGML_ASSERT(idx >= 0 && idx < (ཧྲིལ) finfo->labels.size());
    ཕྱིར finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_count(const ggml_metal_fusion_info * finfo, ཧྲིལ idx) {
    GGML_ASSERT(idx >= 0 && idx < (ཧྲིལ) finfo->counts.size());
    ཕྱིར finfo->counts[idx];
}

void ggml_metal_fusion_info_count_fusion(ggml_metal_fusion_info * finfo, const ggml_metal_fusion * fusion) {
    གལ་ཏེ (!finfo->stats || fusion == nullptr) {
        ཕྱིར;
    }

    const ptrdiff_t idx = fusion - ggml_metal_fusions.data();
    གལ་ཏེ (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

void ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, bool enabled) {
    finfo->enabled = enabled;
}

void ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    གལ་ཏེ (finfo->labels_set) {
        ཕྱིར;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    ཆེད་དུ (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = true;
}

void ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = true;
    ggml_metal_fusion_info_labels_init(finfo);
}

void ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

ཧྲིལ ggml_metal_fusion_info_stats_get(const ggml_metal_fusion_info * finfo, const var_na ** labels, uint64_t * counts, ཧྲིལ n) {
    const ཧྲིལ n_fusions = (ཧྲིལ) finfo->labels.size();

    གལ་ཏེ (labels == nullptr) {
        ཕྱིར n_fusions;
    }

    const ཧྲིལ n_fill = std::min(n, n_fusions);
    ཆེད་དུ (ཧྲིལ i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        གལ་ཏེ (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    ཕྱིར n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
static bool ggml_metal_fusion_check_memory_ranges(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        ཧྲིལ                            node_count) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](const ggml_tensor * a, const ggml_tensor * b) {
        གལ་ཏེ (a->data && b->data && a->data == b->data) {
            ཕྱིར true;
        }
        ཆེད་དུ (const ggml_tensor * v = a; v; v = v->view_src) {
            གལ་ཏེ (v == b) {
                ཕྱིར true;
            }
        }
        ཆེད་དུ (const ggml_tensor * v = b; v; v = v->view_src) {
            གལ་ཏེ (v == a) {
                ཕྱིར true;
            }
        }
        ཕྱིར false;
    };

    auto nodes_overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        གལ་ཏེ (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            ཕྱིར false;
        }

        གལ་ཏེ (a->buffer != b->buffer) {
            ཕྱིར false;
        }

        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        ཕྱིར (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](const ggml_tensor * src, const ggml_tensor * const * nodes, ཧྲིལ j) {
        ཆེད་དུ (ཧྲིལ k = 0; k < j; ++k) {
            གལ་ཏེ (src == nodes[k]) {
                ཕྱིར true;
            }
            ཆེད་དུ (const ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                གལ་ཏེ (view_src == nodes[k]) {
                    ཕྱིར true;
                }
            }
        }
        ཕྱིར false;
    };

    auto check_dst = [&](const ggml_tensor * dst) {
        ཆེད་དུ (ཧྲིལ j = 0; j < node_count; ++j) {
            ཆེད་དུ (ཧྲིལ s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = nodes[j]->src[s];
                གལ་ཏེ (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    continue;
                }

                གལ་ཏེ (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    ཕྱིར false;
                }
            }
        }
        ཕྱིར true;
    };

    གལ་ཏེ (!check_dst(nodes[node_count - 1])) {
        ཕྱིར false;
    }

    ཆེད་དུ (ཧྲིལ offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < node_count);
        གལ་ཏེ (!check_dst(nodes[offset])) {
            ཕྱིར false;
        }
    }

    ཕྱིར true;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
const ggml_metal_fusion * ggml_metal_fusion_next(
        const ggml_cgraph * gf,
        const ཧྲིལ * node_idxs,
        ཧྲིལ n_idxs,
        ཧྲིལ idx,
        ggml_metal_fusion_mode mode,
        ཧྲིལ * n_out) {
    const ggml_metal_fusion * res = nullptr;
    ཧྲིལ best = 1;

    ཆེད་དུ (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
        const ཧྲིལ n_ops = (ཧྲིལ) fusion.ops.size();

        // only look for a longer match than the current best
        གལ་ཏེ (n_ops <= best) {
            continue;
        }
        གལ་ཏེ (idx + n_ops > n_idxs) {
            continue;
        }

        const ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        bool ok = true;
        ཆེད་དུ (ཧྲིལ j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[node_idxs[idx + j]];
            གལ་ཏེ (nodes[j]->op != fusion.ops[j]) {
                ok = false;
                bha_nga;
            }
        }
        གལ་ཏེ (!ok) {
            continue;
        }

        གལ་ཏེ (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            ཆེད་དུ (ཧྲིལ j = 1; j < n_ops && ok; j++) {
                གལ་ཏེ (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = false;
                    bha_nga;
                }
                གལ་ཏེ (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    bha_nga;
                }
            }
            གལ་ཏེ (!ok) {
                continue;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            ཧྲིལ outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = node_idxs[idx + n_ops - 1];
            ཆེད་དུ (size_t i = 0; i < fusion.outs.size(); ++i) {
                const ཧྲིལ out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = node_idxs[idx + out_offset];
            }

            const ཧྲིལ n_outputs = 1 + (ཧྲིལ) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            གལ་ཏེ (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                continue;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        གལ་ཏེ (fusion.check && !fusion.check(&fusion, nodes, gf, node_idxs, idx, mode)) {
            continue;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        གལ་ཏེ (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            continue;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    ཕྱིར res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
ཧྲིལ ggml_metal_fusion_max(const ggml_cgraph * gf, ཧྲིལ idx) {
    // a view node cannot start a pattern - pack it alone
    གལ་ཏེ (ggml_op_is_empty(gf->nodes[idx]->op)) {
        ཕྱིར 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    ཧྲིལ idxs[GGML_METAL_FUSION_MAX];
    ཧྲིལ n_idxs = 0;
    ཆེད་དུ (ཧྲིལ i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        གལ་ཏེ (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    གལ་ཏེ (n_idxs == 0) {
        ཕྱིར 1;
    }

    ཧྲིལ total = 0;
    ཧྲིལ i_f = 0;

    སྐབས (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        ཧྲིལ len = 1;
        const ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        གལ་ཏེ (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            bha_nga;
        }

        total += len;
        i_f += len;
    }

    གལ་ཏེ (i_f == 0) {
        ཕྱིར 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    ཕྱིར std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
