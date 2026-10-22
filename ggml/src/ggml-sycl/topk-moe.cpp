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

struct ggml_sycl_topk_moe_args {
    bool sigmoid{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

struct topk_moe_config {
    bool use_sigmoid;
    bool with_norm;
    bool delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
template <ціле experts_per_thread, bool use_limit>
static inline void softmax_warp_inplace(float (&vals)[experts_per_thread], const ціле limit, const ціле lane) {
    float max_val = -INFINITY;
#pragma unroll
    для (ціле i = 0; i < experts_per_thread; i++) {
        const ціле  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        якщо (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    float sum = 0.f;
#pragma unroll
    для (ціле i = 0; i < experts_per_thread; i++) {
        const ціле  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        якщо (active) {
            const float val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } інакше {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    const float inv_sum = 1.0f / sum;
#pragma unroll
    для (ціле i = 0; i < experts_per_thread; i++) {
        const ціле idx = lane + i * WARP_SIZE;
        якщо (!use_limit || idx < limit) {
            vals[i] *= inv_sum;
        }
    }
}

template <ціле experts_per_thread, bool use_limit>
static inline void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const ціле limit, const ціле lane) {
#pragma unroll
    для (ціле i = 0; i < experts_per_thread; i++) {
        const ціле  idx    = lane + i * WARP_SIZE;
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
template <ціле n_experts>
static void topk_moe_kernel(const float * __restrict__ logits,
                            float * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            const ціле                  n_rows,
                            const ціле                  n_expert_used,
                            const float                clamp_val,
                            const float                scale_val,
                            const topk_moe_config       config) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const ціле row       = item_ct1.get_group(0);
    якщо (row >= n_rows) {
        повернути;
    }
    const ціле lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr ціле experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];
#pragma unroll
    для (ціле i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    для (ціле i = 0; i < n_experts; i += WARP_SIZE) {
        const ціле expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    якщо (!config.delayed_softmax) {
        якщо (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        } інакше {
            softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#pragma unroll
    для (ціле i = 0; i < experts_per_thread; i++) {
        якщо (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    float wt_sum = 0.f;
    float output_weights[experts_per_thread];
#pragma unroll
    для (ціле i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    const sycl::sub_group sg = item_ct1.get_sub_group();

    для (ціле k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        ціле   max_expert = lane;
#pragma unroll
        для (ціле i = 1; i < experts_per_thread; i++) {
            const ціле expert = lane + i * WARP_SIZE;
            якщо ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        для (ціле mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            const float val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            const ціле   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            якщо (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        якщо ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        якщо ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        якщо ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            якщо (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    якщо (config.with_norm) {
        wt_sum          = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum          = sycl::fmax(wt_sum, clamp_val);
        const float inv = 1.0f / wt_sum;
#pragma unroll
        для (ціле i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    якщо (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    для (ціле i = 0; i < experts_per_thread; i++) {
        const ціле idx = i * WARP_SIZE + lane;
        якщо (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <ціле n_experts>
static void launch_topk_moe(queue_ptr stream, const float * logits, float * weights, int32_t * ids, ціле n_rows,
                            ціле n_expert_used, float clamp_val, float scale_val, const topk_moe_config & config) {
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

    const ціле n_experts     = logits->ne[0];
    const ціле n_rows        = logits->ne[1];
    const ціле n_expert_used = weights->ne[1];

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
        case 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            break;
        case 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            break;
        case 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            break;
        case 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            break;
        case 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            break;
        case 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            break;
        case 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            break;
        case 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            break;
        case 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            break;
        case 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
    }
}

static bool ggml_sycl_should_use_topk_moe(const ggml_tensor * gating_op, const ggml_tensor * weights,
                                          const ggml_tensor * logits, const ggml_tensor * ids) {
    const ціле n_expert = ids->nb[1] / ids->nb[0];
    якщо ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        повернути false;
    }

    якщо (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        повернути false;
    }

    якщо (gating_op->op == GGML_OP_SOFT_MAX) {
        float scale    = 1.0f;
        float max_bias = 0.0f;

        memcpy(&scale, (const float *) gating_op->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) gating_op->op_params + 1, sizeof(float));

        якщо (!ggml_is_contiguous(gating_op->src[0])) {
            повернути false;
        }
        якщо (scale != 1.0f || max_bias != 0.0f) {
            повернути false;
        }
        // don't fuse when masks or sinks are present
        якщо (gating_op->src[1] || gating_op->src[2]) {
            повернути false;
        }
    } інакше якщо (gating_op->op == GGML_OP_UNARY) {
        якщо (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            повернути false;
        }
    }

    повернути true;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
static bool ggml_sycl_topk_moe_fusion(const ggml_cgraph * cgraph, ціле node_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    const ціле      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    якщо (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    якщо (nodes[node_idx]->op == GGML_OP_UNARY) {
        якщо (ggml_get_unary_op(nodes[node_idx]) != GGML_UNARY_OP_SIGMOID) {
            повернути false;
        }
        args.sigmoid = true;
    }

    якщо (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    якщо (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        якщо (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            повернути false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        якщо (node_idx >= n_nodes) {
            повернути false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        якщо (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        якщо (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            повернути false;
        }

        якщо (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            повернути false;
        } інакше якщо (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            повернути false;
        }

        node_idx++;

        // ARGSORT -> VIEW
        якщо (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            повернути false;
        }
        node_idx++;

        якщо (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            повернути false;
        }

        // GET_ROWS
        якщо (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            повернути false;
        }
        node_idx++;
    } інакше якщо (args.delayed_softmax) {
        якщо (node_idx - 2 < 0) {
            повернути false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW -> ARGSORT
        якщо (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            повернути false;
        }
        node_idx++;

        // GET_ROWS
        якщо (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != probs_reshaped) {
            повернути false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        для (const ggml_op op : remaining_ops) {
            якщо (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                повернути false;
            }
            node_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    якщо (node_idx >= n_nodes) {
        повернути true;
    }

    якщо (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        для (const ggml_op op : norm_ops) {
            якщо (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } інакше {
                args.norm = false;
                повернути true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        якщо (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            повернути true;
        }
        node_idx++;

        якщо (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            повернути true;
        }
        node_idx++;
    }

    якщо (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    повернути true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
static bool ggml_sycl_check_fusion_memory_ranges(const ggml_cgraph * cgraph, const ціле node_idx,
                                                 const ціле node_count, const ціле * out_nodes, const ціле out_count,
                                                 const bool is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        якщо ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            повернути true;
        }

        повернути false;
    };

    bool is_ok = true;
    // exception for topk-moe, as each row is read entirely before writing
    якщо (ggml_nrows(cgraph->nodes[node_idx]) == 1 && is_topk_moe) {
        повернути true;
    }

    для (ціле i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        для (ціле j = node_idx; j < node_idx + node_count; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            для (ціле src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                якщо (!src || src->op == GGML_OP_NONE) {
                    continue;
                }

                якщо (nodes_overlap(dst, src)) {
                    bool found = false;

                    для (ціле k = node_idx; k < j; ++k) {
                        якщо (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    якщо (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    повернути is_ok;
}

ціле ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ціле i) {
    якщо (!g_ggml_sycl_enable_fusion) {
        повернути 0;
    }

    повернути ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

ціле ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ціле i) {
    ggml_tensor * node = cgraph->nodes[i];

    якщо (node->op != GGML_OP_UNARY && node->op != GGML_OP_SOFT_MAX && node->op != GGML_OP_ARGSORT) {
        повернути 0;
    }

    ggml_sycl_topk_moe_args args;
    якщо (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        повернути 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    якщо (args.prob_bias) {
        повернути 0;
    }

    const ggml_tensor * logits  = node->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    const ggml_tensor * clamp   = nullptr;
    const ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    ціле                  out_nodes[2];

    якщо (!args.delayed_softmax) {
        const ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        якщо (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (ціле) ops.size() - 3];
        }
        якщо (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (ціле) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (ціле) ops.size() - 1];
        out_nodes[1] = i + (ціле) ops.size() - 1;

        якщо (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(node, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (ціле) ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            повернути (ціле) ops.size() - 1;
        }
    } інакше якщо (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        const ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        якщо (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (ціле) ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            повернути (ціле) ops.size() - 1;
        }
    }

    повернути 0;
}
