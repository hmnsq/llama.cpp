#ضمن <cfloat>
#ضمن <initializer_list>
#ضمن <vector>

#ضمن "ggml.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-backend-impl.h"
#ضمن "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

بنية ggml_sycl_topk_moe_args {
    منطقي sigmoid{};
    منطقي softmax{};
    منطقي delayed_softmax{};
    منطقي prob_bias{};
    منطقي norm{};
    منطقي scale{};
};

بنية topk_moe_config {
    منطقي use_sigmoid;
    منطقي with_norm;
    منطقي delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
template <صحيح experts_per_thread, منطقي use_limit>
ساكن مضمن فراغ softmax_warp_inplace(عائم (&vals)[experts_per_thread], ثابت صحيح limit, ثابت صحيح lane) {
    عائم max_val = -INFINITY;
#pragma unroll
    لكل (صحيح i = 0; i < experts_per_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_SIZE;
        ثابت منطقي active = !use_limit || (idx < limit);
        إذا (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    عائم sum = 0.f;
#pragma unroll
    لكل (صحيح i = 0; i < experts_per_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_SIZE;
        ثابت منطقي active = !use_limit || (idx < limit);
        إذا (active) {
            ثابت عائم val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } وإلا {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    ثابت عائم inv_sum = 1.0f / sum;
#pragma unroll
    لكل (صحيح i = 0; i < experts_per_thread; i++) {
        ثابت صحيح idx = lane + i * WARP_SIZE;
        إذا (!use_limit || idx < limit) {
            vals[i] *= inv_sum;
        }
    }
}

template <صحيح experts_per_thread, منطقي use_limit>
ساكن مضمن فراغ sigmoid_warp_inplace(عائم (&vals)[experts_per_thread], ثابت صحيح limit, ثابت صحيح lane) {
#pragma unroll
    لكل (صحيح i = 0; i < experts_per_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_SIZE;
        ثابت منطقي active = !use_limit || (idx < limit);
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
template <صحيح n_experts>
ساكن فراغ topk_moe_kernel(ثابت عائم * __restrict__ logits,
                            عائم * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ثابت صحيح                  n_rows,
                            ثابت صحيح                  n_expert_used,
                            ثابت عائم                clamp_val,
                            ثابت عائم                scale_val,
                            ثابت topk_moe_config       config) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    ثابت صحيح row       = item_ct1.get_group(0);
    إذا (row >= n_rows) {
        أرجع;
    }
    ثابت صحيح lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr صحيح experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    عائم wt[experts_per_thread];
#pragma unroll
    لكل (صحيح i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    لكل (صحيح i = 0; i < n_experts; i += WARP_SIZE) {
        ثابت صحيح expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    إذا (!config.delayed_softmax) {
        إذا (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, خطأ>(wt, n_experts, lane);
        } وإلا {
            softmax_warp_inplace<experts_per_thread, خطأ>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#pragma unroll
    لكل (صحيح i = 0; i < experts_per_thread; i++) {
        إذا (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    عائم wt_sum = 0.f;
    عائم output_weights[experts_per_thread];
#pragma unroll
    لكل (صحيح i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ثابت sycl::sub_group sg = item_ct1.get_sub_group();

    لكل (صحيح k = 0; k < n_expert_used; k++) {
        عائم max_val    = wt[0];
        صحيح   max_expert = lane;
#pragma unroll
        لكل (صحيح i = 1; i < experts_per_thread; i++) {
            ثابت صحيح expert = lane + i * WARP_SIZE;
            إذا ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        لكل (صحيح mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            ثابت عائم val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            ثابت صحيح   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            إذا (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        إذا ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        إذا ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        إذا ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            إذا (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    إذا (config.with_norm) {
        wt_sum          = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum          = sycl::fmax(wt_sum, clamp_val);
        ثابت عائم inv = 1.0f / wt_sum;
#pragma unroll
        لكل (صحيح i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    إذا (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    لكل (صحيح i = 0; i < experts_per_thread; i++) {
        ثابت صحيح idx = i * WARP_SIZE + lane;
        إذا (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <صحيح n_experts>
ساكن فراغ launch_topk_moe(queue_ptr stream, ثابت عائم * logits, عائم * weights, int32_t * ids, صحيح n_rows,
                            صحيح n_expert_used, عائم clamp_val, عائم scale_val, ثابت topk_moe_config & config) {
    ثابت sycl::range<1> block_dims(WARP_SIZE);
    ثابت sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                        scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

ساكن فراغ ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ثابت ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ثابت ggml_tensor *             clamp,
                                  ثابت ggml_tensor *             scale,
                                  ثابت ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ثابت صحيح n_experts     = logits->ne[0];
    ثابت صحيح n_rows        = logits->ne[1];
    ثابت صحيح n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    ثابت عائم * logits_d  = (ثابت عائم *) logits->data;
    عائم *       weights_d = (عائم *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    ثابت منطقي  with_norm = clamp != nullptr;
    ثابت عائم clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ثابت عائم scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    بدل (n_experts) {
        حالة 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            اكسر;
        حالة 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            اكسر;
        حالة 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            اكسر;
        افتراضي:
            GGML_ASSERT(خطأ && "fatal error");
    }
}

ساكن منطقي ggml_sycl_should_use_topk_moe(ثابت ggml_tensor * gating_op, ثابت ggml_tensor * weights,
                                          ثابت ggml_tensor * logits, ثابت ggml_tensor * ids) {
    ثابت صحيح n_expert = ids->nb[1] / ids->nb[0];
    إذا ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        أرجع خطأ;
    }

    إذا (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        أرجع خطأ;
    }

    إذا (gating_op->op == GGML_OP_SOFT_MAX) {
        عائم scale    = 1.0f;
        عائم max_bias = 0.0f;

        memcpy(&scale, (ثابت عائم *) gating_op->op_params + 0, حجم(عائم));
        memcpy(&max_bias, (ثابت عائم *) gating_op->op_params + 1, حجم(عائم));

        إذا (!ggml_is_contiguous(gating_op->src[0])) {
            أرجع خطأ;
        }
        إذا (scale != 1.0f || max_bias != 0.0f) {
            أرجع خطأ;
        }
        // don't fuse when masks or sinks are present
        إذا (gating_op->src[1] || gating_op->src[2]) {
            أرجع خطأ;
        }
    } وإلا إذا (gating_op->op == GGML_OP_UNARY) {
        إذا (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            أرجع خطأ;
        }
    }

    أرجع true;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
ساكن منطقي ggml_sycl_topk_moe_fusion(ثابت ggml_cgraph * cgraph, صحيح نود_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    ثابت صحيح      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    إذا (nodes[نود_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    إذا (nodes[نود_idx]->op == GGML_OP_UNARY) {
        إذا (ggml_get_unary_op(nodes[نود_idx]) != GGML_UNARY_OP_SIGMOID) {
            أرجع خطأ;
        }
        args.sigmoid = true;
    }

    إذا (nodes[نود_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    نود_idx++;

    إذا (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        إذا (نود_idx >= n_nodes || nodes[نود_idx]->op != GGML_OP_RESHAPE ||
            nodes[نود_idx]->src[0] != nodes[نود_idx - 1]) {
            أرجع خطأ;
        }
        ggml_tensor * probs_reshaped = nodes[نود_idx];
        نود_idx++;

        إذا (نود_idx >= n_nodes) {
            أرجع خطأ;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        إذا (nodes[نود_idx]->op == GGML_OP_ADD && nodes[نود_idx]->src[0] == nodes[نود_idx - 2]) {
            args.prob_bias = true;
            نود_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        إذا (نود_idx >= n_nodes || nodes[نود_idx]->op != GGML_OP_ARGSORT) {
            أرجع خطأ;
        }

        إذا (args.prob_bias && nodes[نود_idx]->src[0] != nodes[نود_idx - 1]) {
            أرجع خطأ;
        } وإلا إذا (!args.prob_bias && nodes[نود_idx]->src[0] != nodes[نود_idx - 2]) {
            أرجع خطأ;
        }

        نود_idx++;

        // ARGSORT -> VIEW
        إذا (نود_idx >= n_nodes || nodes[نود_idx]->op != GGML_OP_VIEW ||
            nodes[نود_idx]->src[0] != nodes[نود_idx - 1]) {
            أرجع خطأ;
        }
        نود_idx++;

        إذا (نود_idx >= n_nodes || nodes[نود_idx]->op != GGML_OP_GET_ROWS) {
            أرجع خطأ;
        }

        // GET_ROWS
        إذا (nodes[نود_idx]->src[0] != probs_reshaped || nodes[نود_idx]->src[1] != nodes[نود_idx - 1]) {
            أرجع خطأ;
        }
        نود_idx++;
    } وإلا إذا (args.delayed_softmax) {
        إذا (نود_idx - 2 < 0) {
            أرجع خطأ;
        }
        ggml_tensor * probs_reshaped = nodes[نود_idx - 2];

        // VIEW -> ARGSORT
        إذا (نود_idx >= n_nodes || nodes[نود_idx]->op != GGML_OP_VIEW ||
            nodes[نود_idx]->src[0] != nodes[نود_idx - 1]) {
            أرجع خطأ;
        }
        نود_idx++;

        // GET_ROWS
        إذا (نود_idx >= n_nodes || nodes[نود_idx]->src[1] != nodes[نود_idx - 1] ||
            nodes[نود_idx]->src[0] != probs_reshaped) {
            أرجع خطأ;
        }
        نود_idx++;

        ساكن ثابت std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        لكل (ثابت ggml_op op : remaining_ops) {
            إذا (نود_idx >= n_nodes || nodes[نود_idx]->op != op || nodes[نود_idx]->src[0] != nodes[نود_idx - 1]) {
                أرجع خطأ;
            }
            نود_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    إذا (نود_idx >= n_nodes) {
        أرجع true;
    }

    إذا (nodes[نود_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        ساكن ثابت std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        لكل (ثابت ggml_op op : norm_ops) {
            إذا (nodes[نود_idx]->op == op && nodes[نود_idx]->src[0] == nodes[نود_idx - 1]) {
                نود_idx++;
            } وإلا {
                args.norm = خطأ;
                أرجع true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        إذا (nodes[نود_idx]->op != GGML_OP_DIV || nodes[نود_idx]->src[1] != nodes[نود_idx - 1] ||
            nodes[نود_idx]->src[0] != nodes[نود_idx - 3]) {
            args.norm = خطأ;
            أرجع true;
        }
        نود_idx++;

        إذا (nodes[نود_idx]->op != GGML_OP_RESHAPE || nodes[نود_idx]->src[0] != nodes[نود_idx - 1]) {
            args.norm = خطأ;
            أرجع true;
        }
        نود_idx++;
    }

    إذا (nodes[نود_idx]->op == GGML_OP_SCALE && nodes[نود_idx]->src[0] == nodes[نود_idx - 1]) {
        args.scale = true;
    }

    أرجع true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
ساكن منطقي ggml_sycl_check_fusion_memory_ranges(ثابت ggml_cgraph * cgraph, ثابت صحيح نود_idx,
                                                 ثابت صحيح نود_کيراءن, ثابت صحيح * out_nodes, ثابت صحيح out_کيراءن,
                                                 ثابت منطقي is_topk_moe = خطأ) {
    auto nodes_overlap = [&](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        ثابت int64_t a_start = (int64_t) a->data;
        ثابت int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        ثابت int64_t b_start = (int64_t) b->data;
        ثابت int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        إذا ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            أرجع true;
        }

        أرجع خطأ;
    };

    منطقي is_ok = true;
    // exception for topk-moe, as each row is read entirely before writing
    إذا (ggml_nrows(cgraph->nodes[نود_idx]) == 1 && is_topk_moe) {
        أرجع true;
    }

    لكل (صحيح i = 0; i < out_کيراءن; ++i) {
        ثابت ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        لكل (صحيح j = نود_idx; j < نود_idx + نود_کيراءن; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            لكل (صحيح src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ثابت ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                إذا (!src || src->op == GGML_OP_NONE) {
                    استمر;
                }

                إذا (nodes_overlap(dst, src)) {
                    منطقي found = خطأ;

                    لكل (صحيح k = نود_idx; k < j; ++k) {
                        إذا (cgraph->nodes[k] == src) {
                            found = true;
                            اكسر;
                        }
                    }

                    إذا (!found) {
                        is_ok = خطأ;
                        اكسر;
                    }
                }
            }
        }
    }

    أرجع is_ok;
}

صحيح ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحيح i) {
    إذا (!g_ggml_sycl_enable_fusion) {
        أرجع 0;
    }

    أرجع ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

صحيح ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحيح i) {
    ggml_tensor * نود = cgraph->nodes[i];

    إذا (نود->op != GGML_OP_UNARY && نود->op != GGML_OP_SOFT_MAX && نود->op != GGML_OP_ARGSORT) {
        أرجع 0;
    }

    ggml_sycl_topk_moe_args args;
    إذا (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        أرجع 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    إذا (args.prob_bias) {
        أرجع 0;
    }

    ثابت ggml_tensor * logits  = نود->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    ثابت ggml_tensor * clamp   = nullptr;
    ثابت ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    صحيح                  out_nodes[2];

    إذا (!args.delayed_softmax) {
        ثابت ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        إذا (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (صحيح) ops.size() - 3];
        }
        إذا (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (صحيح) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (صحيح) ops.size() - 1];
        out_nodes[1] = i + (صحيح) ops.size() - 1;

        إذا (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(نود, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (صحيح) ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            أرجع (صحيح) ops.size() - 1;
        }
    } وإلا إذا (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ثابت ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        إذا (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (صحيح) ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            أرجع (صحيح) ops.size() - 1;
        }
    }

    أرجع 0;
}
