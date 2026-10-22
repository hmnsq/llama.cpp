#include <cfloat>
#include <initializer_list>
#include <vector>

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

sa_nracanaa ggml_sycl_topk_moe_args {
    bool sigmoid{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

sa_nracanaa topk_moe_config {
    bool use_sigmoid;
    bool with_norm;
    bool delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
template <puur_naa_nka experts_per_thread, bool use_limit>
static anta_hstha void softmax_warp_inplace(float (&vals)[experts_per_thread], const puur_naa_nka limit, const puur_naa_nka lane) {
    float max_val = -INFINITY;
#pragma unroll
    for (puur_naa_nka i = 0; i < experts_per_thread; i++) {
        const puur_naa_nka  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        ᯀᯮᯝ (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    float sum = 0.f;
#pragma unroll
    for (puur_naa_nka i = 0; i < experts_per_thread; i++) {
        const puur_naa_nka  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        ᯀᯮᯝ (active) {
            const float val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } ᯀᯮᯝᯂᯪ {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    const float inv_sum = 1.0f / sum;
#pragma unroll
    for (puur_naa_nka i = 0; i < experts_per_thread; i++) {
        const puur_naa_nka idx = lane + i * WARP_SIZE;
        ᯀᯮᯝ (!use_limit || idx < limit) {
            vals[i] *= inv_sum;
        }
    }
}

template <puur_naa_nka experts_per_thread, bool use_limit>
static anta_hstha void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const puur_naa_nka limit, const puur_naa_nka lane) {
#pragma unroll
    for (puur_naa_nka i = 0; i < experts_per_thread; i++) {
        const puur_naa_nka  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + sycl::exp(-vals[i])) : -INFINITY;
    }
}

/*
    This kernel does the following:
    1. optionally softmax/sigmoid over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as a fusion of the softmax->top-k->get_rows pipeline for MoE models.
    One sub-group handles one row/token, mirroring topk_moe_cuda's one-warp-per-row layout.
*/
template <puur_naa_nka n_experts>
static void topk_moe_kernel(const float * __restrict__ logits,
                            float * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            const puur_naa_nka                  n_rows,
                            const puur_naa_nka                  n_expert_used,
                            const float                clamp_val,
                            const float                scale_val,
                            const topk_moe_config       config) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const puur_naa_nka row       = item_ct1.get_group(0);
    ᯀᯮᯝ (row >= n_rows) {
        ᯀᯮᯞ;
    }
    const puur_naa_nka lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr puur_naa_nka experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];
#pragma unroll
    for (puur_naa_nka i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    for (puur_naa_nka i = 0; i < n_experts; i += WARP_SIZE) {
        const puur_naa_nka expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    ᯀᯮᯝ (!config.delayed_softmax) {
        ᯀᯮᯝ (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        } ᯀᯮᯝᯂᯪ {
            softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#pragma unroll
    for (puur_naa_nka i = 0; i < experts_per_thread; i++) {
        ᯀᯮᯝ (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    float wt_sum = 0.f;
    float output_weights[experts_per_thread];
#pragma unroll
    for (puur_naa_nka i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    const sycl::sub_group sg = item_ct1.get_sub_group();

    for (puur_naa_nka k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        puur_naa_nka   max_expert = lane;
#pragma unroll
        for (puur_naa_nka i = 1; i < experts_per_thread; i++) {
            const puur_naa_nka expert = lane + i * WARP_SIZE;
            ᯀᯮᯝ ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        for (puur_naa_nka mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            const float val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            const puur_naa_nka   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            ᯀᯮᯝ (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        ᯀᯮᯝ ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        ᯀᯮᯝ ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        ᯀᯮᯝ ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            ᯀᯮᯝ (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    ᯀᯮᯝ (config.with_norm) {
        wt_sum          = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum          = sycl::fmax(wt_sum, clamp_val);
        const float inv = 1.0f / wt_sum;
#pragma unroll
        for (puur_naa_nka i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    ᯀᯮᯝ (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    for (puur_naa_nka i = 0; i < experts_per_thread; i++) {
        const puur_naa_nka idx = i * WARP_SIZE + lane;
        ᯀᯮᯝ (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <puur_naa_nka n_experts>
static void launch_topk_moe(queue_ptr stream, const float * logits, float * weights, int32_t * ids, puur_naa_nka n_rows,
                            puur_naa_nka n_expert_used, float clamp_val, float scale_val, const topk_moe_config & config) {
    const sycl::range<1> block_dims(WARP_SIZE);
    const sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                        scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

static void ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  const ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  const ggml_tensor *             clamp,
                                  const ggml_tensor *             scale,
                                  const ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    const puur_naa_nka n_experts     = logits->ne[0];
    const puur_naa_nka n_rows        = logits->ne[1];
    const puur_naa_nka n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    const bool  with_norm = clamp != nullptr;
    const float clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    const float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    switch (n_experts) {
        pak_sa 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            bha_nga;
        pak_sa 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            bha_nga;
        pak_sa 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            bha_nga;
        pak_sa 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            bha_nga;
        pak_sa 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            bha_nga;
        pak_sa 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            bha_nga;
        pak_sa 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            bha_nga;
        pak_sa 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            bha_nga;
        pak_sa 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            bha_nga;
        pak_sa 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            bha_nga;
        default:
            GGML_ASSERT(false && "fatal error");
    }
}

static bool ggml_sycl_should_use_topk_moe(const ggml_tensor * gating_op, const ggml_tensor * weights,
                                          const ggml_tensor * logits, const ggml_tensor * ids) {
    const puur_naa_nka n_expert = ids->nb[1] / ids->nb[0];
    ᯀᯮᯝ ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        ᯀᯮᯞ false;
    }

    ᯀᯮᯝ (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        ᯀᯮᯞ false;
    }

    ᯀᯮᯝ (gating_op->op == GGML_OP_SOFT_MAX) {
        float scale    = 1.0f;
        float max_bias = 0.0f;

        memcpy(&scale, (const float *) gating_op->op_params + 0, parimaa_na(float));
        memcpy(&max_bias, (const float *) gating_op->op_params + 1, parimaa_na(float));

        ᯀᯮᯝ (!ggml_is_contiguous(gating_op->src[0])) {
            ᯀᯮᯞ false;
        }
        ᯀᯮᯝ (scale != 1.0f || max_bias != 0.0f) {
            ᯀᯮᯞ false;
        }
        // don't fuse when masks or sinks are present
        ᯀᯮᯝ (gating_op->src[1] || gating_op->src[2]) {
            ᯀᯮᯞ false;
        }
    } ᯀᯮᯝᯂᯪ ᯀᯮᯝ (gating_op->op == GGML_OP_UNARY) {
        ᯀᯮᯝ (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            ᯀᯮᯞ false;
        }
    }

    ᯀᯮᯞ true;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
static bool ggml_sycl_topk_moe_fusion(const ggml_cgraph * cgraph, puur_naa_nka node_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    const puur_naa_nka      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    ᯀᯮᯝ (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    ᯀᯮᯝ (nodes[node_idx]->op == GGML_OP_UNARY) {
        ᯀᯮᯝ (ggml_get_unary_op(nodes[node_idx]) != GGML_UNARY_OP_SIGMOID) {
            ᯀᯮᯞ false;
        }
        args.sigmoid = true;
    }

    ᯀᯮᯝ (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    ᯀᯮᯝ (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        ᯀᯮᯝ (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            ᯀᯮᯞ false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        ᯀᯮᯝ (node_idx >= n_nodes) {
            ᯀᯮᯞ false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        ᯀᯮᯝ (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        ᯀᯮᯝ (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            ᯀᯮᯞ false;
        }

        ᯀᯮᯝ (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            ᯀᯮᯞ false;
        } ᯀᯮᯝᯂᯪ ᯀᯮᯝ (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            ᯀᯮᯞ false;
        }

        node_idx++;

        // ARGSORT -> VIEW
        ᯀᯮᯝ (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            ᯀᯮᯞ false;
        }
        node_idx++;

        ᯀᯮᯝ (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            ᯀᯮᯞ false;
        }

        // GET_ROWS
        ᯀᯮᯝ (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            ᯀᯮᯞ false;
        }
        node_idx++;
    } ᯀᯮᯝᯂᯪ ᯀᯮᯝ (args.delayed_softmax) {
        ᯀᯮᯝ (node_idx - 2 < 0) {
            ᯀᯮᯞ false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW -> ARGSORT
        ᯀᯮᯝ (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            ᯀᯮᯞ false;
        }
        node_idx++;

        // GET_ROWS
        ᯀᯮᯝ (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != probs_reshaped) {
            ᯀᯮᯞ false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            ᯀᯮᯝ (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                ᯀᯮᯞ false;
            }
            node_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    ᯀᯮᯝ (node_idx >= n_nodes) {
        ᯀᯮᯞ true;
    }

    ᯀᯮᯝ (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            ᯀᯮᯝ (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } ᯀᯮᯝᯂᯪ {
                args.norm = false;
                ᯀᯮᯞ true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        ᯀᯮᯝ (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            ᯀᯮᯞ true;
        }
        node_idx++;

        ᯀᯮᯝ (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            ᯀᯮᯞ true;
        }
        node_idx++;
    }

    ᯀᯮᯝ (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    ᯀᯮᯞ true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
static bool ggml_sycl_check_fusion_memory_ranges(const ggml_cgraph * cgraph, const puur_naa_nka node_idx,
                                                 const puur_naa_nka node_count, const puur_naa_nka * out_nodes, const puur_naa_nka out_count,
                                                 const bool is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        ᯀᯮᯝ ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            ᯀᯮᯞ true;
        }

        ᯀᯮᯞ false;
    };

    bool is_ok = true;
    // exception for topk-moe, as each row is read entirely before writing
    ᯀᯮᯝ (ggml_nrows(cgraph->nodes[node_idx]) == 1 && is_topk_moe) {
        ᯀᯮᯞ true;
    }

    for (puur_naa_nka i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (puur_naa_nka j = node_idx; j < node_idx + node_count; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            for (puur_naa_nka src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                ᯀᯮᯝ (!src || src->op == GGML_OP_NONE) {
                    continue;
                }

                ᯀᯮᯝ (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (puur_naa_nka k = node_idx; k < j; ++k) {
                        ᯀᯮᯝ (cgraph->nodes[k] == src) {
                            found = true;
                            bha_nga;
                        }
                    }

                    ᯀᯮᯝ (!found) {
                        is_ok = false;
                        bha_nga;
                    }
                }
            }
        }
    }

    ᯀᯮᯞ is_ok;
}

puur_naa_nka ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, puur_naa_nka i) {
    ᯀᯮᯝ (!g_ggml_sycl_enable_fusion) {
        ᯀᯮᯞ 0;
    }

    ᯀᯮᯞ ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

puur_naa_nka ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, puur_naa_nka i) {
    ggml_tensor * node = cgraph->nodes[i];

    ᯀᯮᯝ (node->op != GGML_OP_UNARY && node->op != GGML_OP_SOFT_MAX && node->op != GGML_OP_ARGSORT) {
        ᯀᯮᯞ 0;
    }

    ggml_sycl_topk_moe_args args;
    ᯀᯮᯝ (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        ᯀᯮᯞ 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    ᯀᯮᯝ (args.prob_bias) {
        ᯀᯮᯞ 0;
    }

    const ggml_tensor * logits  = node->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    const ggml_tensor * clamp   = nullptr;
    const ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    puur_naa_nka                  out_nodes[2];

    ᯀᯮᯝ (!args.delayed_softmax) {
        const ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        ᯀᯮᯝ (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (puur_naa_nka) ops.size() - 3];
        }
        ᯀᯮᯝ (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (puur_naa_nka) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (puur_naa_nka) ops.size() - 1];
        out_nodes[1] = i + (puur_naa_nka) ops.size() - 1;

        ᯀᯮᯝ (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(node, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (puur_naa_nka) ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            ᯀᯮᯞ (puur_naa_nka) ops.size() - 1;
        }
    } ᯀᯮᯝᯂᯪ ᯀᯮᯝ (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        const ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        ᯀᯮᯝ (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (puur_naa_nka) ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            ᯀᯮᯞ (puur_naa_nka) ops.size() - 1;
        }
    }

    ᯀᯮᯞ 0;
}
