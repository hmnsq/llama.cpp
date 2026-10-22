#شامل <cfloat>
#شامل <initializer_list>
#شامل <vector>

#شامل "ggml.h"
#شامل "ggml-impl.h"
#شامل "ggml-backend-impl.h"
#شامل "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

ساختار ggml_sycl_topk_moe_args {
    منطقی sigmoid{};
    منطقی softmax{};
    منطقی delayed_softmax{};
    منطقی prob_bias{};
    منطقی norm{};
    منطقی scale{};
};

ساختار topk_moe_config {
    منطقی use_sigmoid;
    منطقی with_norm;
    منطقی delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
template <صحیح experts_per_thread, منطقی use_limit>
ایستا درونخطی تهی softmax_warp_inplace(اعشاری (&vals)[experts_per_thread], ثابت صحیح limit, ثابت صحیح lane) {
    اعشاری max_val = -INFINITY;
#pragma unroll
    برای (صحیح i = 0; i < experts_per_thread; i++) {
        ثابت صحیح  idx    = lane + i * WARP_SIZE;
        ثابت منطقی active = !use_limit || (idx < limit);
        اگر (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    اعشاری sum = 0.f;
#pragma unroll
    برای (صحیح i = 0; i < experts_per_thread; i++) {
        ثابت صحیح  idx    = lane + i * WARP_SIZE;
        ثابت منطقی active = !use_limit || (idx < limit);
        اگر (active) {
            ثابت اعشاری val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } وگرنه {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    ثابت اعشاری inv_sum = 1.0f / sum;
#pragma unroll
    برای (صحیح i = 0; i < experts_per_thread; i++) {
        ثابت صحیح idx = lane + i * WARP_SIZE;
        اگر (!use_limit || idx < limit) {
            vals[i] *= inv_sum;
        }
    }
}

template <صحیح experts_per_thread, منطقی use_limit>
ایستا درونخطی تهی sigmoid_warp_inplace(اعشاری (&vals)[experts_per_thread], ثابت صحیح limit, ثابت صحیح lane) {
#pragma unroll
    برای (صحیح i = 0; i < experts_per_thread; i++) {
        ثابت صحیح  idx    = lane + i * WARP_SIZE;
        ثابت منطقی active = !use_limit || (idx < limit);
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
template <صحیح n_experts>
ایستا تهی topk_moe_kernel(ثابت اعشاری * __restrict__ logits,
                            اعشاری * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ثابت صحیح                  n_rows,
                            ثابت صحیح                  n_expert_used,
                            ثابت اعشاری                clamp_val,
                            ثابت اعشاری                scale_val,
                            ثابت topk_moe_config       config) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    ثابت صحیح row       = item_ct1.get_group(0);
    اگر (row >= n_rows) {
        بازگشت;
    }
    ثابت صحیح lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr صحیح experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    اعشاری wt[experts_per_thread];
#pragma unroll
    برای (صحیح i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    برای (صحیح i = 0; i < n_experts; i += WARP_SIZE) {
        ثابت صحیح expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    اگر (!config.delayed_softmax) {
        اگر (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, نادرست>(wt, n_experts, lane);
        } وگرنه {
            softmax_warp_inplace<experts_per_thread, نادرست>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#pragma unroll
    برای (صحیح i = 0; i < experts_per_thread; i++) {
        اگر (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    اعشاری wt_sum = 0.f;
    اعشاری output_weights[experts_per_thread];
#pragma unroll
    برای (صحیح i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ثابت sycl::sub_group sg = item_ct1.get_sub_group();

    برای (صحیح k = 0; k < n_expert_used; k++) {
        اعشاری max_val    = wt[0];
        صحیح   max_expert = lane;
#pragma unroll
        برای (صحیح i = 1; i < experts_per_thread; i++) {
            ثابت صحیح expert = lane + i * WARP_SIZE;
            اگر ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        برای (صحیح mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            ثابت اعشاری val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            ثابت صحیح   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            اگر (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        اگر ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        اگر ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        اگر ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            اگر (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    اگر (config.with_norm) {
        wt_sum          = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum          = sycl::fmax(wt_sum, clamp_val);
        ثابت اعشاری inv = 1.0f / wt_sum;
#pragma unroll
        برای (صحیح i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    اگر (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, درست>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    برای (صحیح i = 0; i < experts_per_thread; i++) {
        ثابت صحیح idx = i * WARP_SIZE + lane;
        اگر (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <صحیح n_experts>
ایستا تهی launch_topk_moe(queue_ptr stream, ثابت اعشاری * logits, اعشاری * weights, int32_t * ids, صحیح n_rows,
                            صحیح n_expert_used, اعشاری clamp_val, اعشاری scale_val, ثابت topk_moe_config & config) {
    ثابت sycl::range<1> block_dims(WARP_SIZE);
    ثابت sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                        scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

ایستا تهی ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ثابت ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ثابت ggml_tensor *             clamp,
                                  ثابت ggml_tensor *             scale,
                                  ثابت ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ثابت صحیح n_experts     = logits->ne[0];
    ثابت صحیح n_rows        = logits->ne[1];
    ثابت صحیح n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    ثابت اعشاری * logits_d  = (ثابت اعشاری *) logits->data;
    اعشاری *       weights_d = (اعشاری *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    ثابت منطقی  with_norm = clamp != nullptr;
    ثابت اعشاری clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ثابت اعشاری scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    انتخاب (n_experts) {
        حالت 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            بشکن;
        حالت 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            بشکن;
        حالت 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            بشکن;
        حالت 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            بشکن;
        حالت 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            بشکن;
        حالت 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            بشکن;
        حالت 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            بشکن;
        حالت 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            بشکن;
        حالت 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            بشکن;
        حالت 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            بشکن;
        پیشفرض:
            GGML_ASSERT(نادرست && "fatal error");
    }
}

ایستا منطقی ggml_sycl_should_use_topk_moe(ثابت ggml_tensor * gating_op, ثابت ggml_tensor * weights,
                                          ثابت ggml_tensor * logits, ثابت ggml_tensor * ids) {
    ثابت صحیح n_expert = ids->nb[1] / ids->nb[0];
    اگر ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        بازگشت نادرست;
    }

    اگر (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        بازگشت نادرست;
    }

    اگر (gating_op->op == GGML_OP_SOFT_MAX) {
        اعشاری scale    = 1.0f;
        اعشاری max_bias = 0.0f;

        memcpy(&scale, (ثابت اعشاری *) gating_op->op_params + 0, اندازه(اعشاری));
        memcpy(&max_bias, (ثابت اعشاری *) gating_op->op_params + 1, اندازه(اعشاری));

        اگر (!ggml_is_contiguous(gating_op->src[0])) {
            بازگشت نادرست;
        }
        اگر (scale != 1.0f || max_bias != 0.0f) {
            بازگشت نادرست;
        }
        // don't fuse when masks or sinks are present
        اگر (gating_op->src[1] || gating_op->src[2]) {
            بازگشت نادرست;
        }
    } وگرنه اگر (gating_op->op == GGML_OP_UNARY) {
        اگر (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            بازگشت نادرست;
        }
    }

    بازگشت درست;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
ایستا منطقی ggml_sycl_topk_moe_fusion(ثابت ggml_cgraph * cgraph, صحیح گره_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    ثابت صحیح      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    اگر (nodes[گره_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = درست;
    }

    اگر (nodes[گره_idx]->op == GGML_OP_UNARY) {
        اگر (ggml_get_unary_op(nodes[گره_idx]) != GGML_UNARY_OP_SIGMOID) {
            بازگشت نادرست;
        }
        args.sigmoid = درست;
    }

    اگر (nodes[گره_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = درست;
    }

    گره_idx++;

    اگر (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        اگر (گره_idx >= n_nodes || nodes[گره_idx]->op != GGML_OP_RESHAPE ||
            nodes[گره_idx]->src[0] != nodes[گره_idx - 1]) {
            بازگشت نادرست;
        }
        ggml_tensor * probs_reshaped = nodes[گره_idx];
        گره_idx++;

        اگر (گره_idx >= n_nodes) {
            بازگشت نادرست;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        اگر (nodes[گره_idx]->op == GGML_OP_ADD && nodes[گره_idx]->src[0] == nodes[گره_idx - 2]) {
            args.prob_bias = درست;
            گره_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        اگر (گره_idx >= n_nodes || nodes[گره_idx]->op != GGML_OP_ARGSORT) {
            بازگشت نادرست;
        }

        اگر (args.prob_bias && nodes[گره_idx]->src[0] != nodes[گره_idx - 1]) {
            بازگشت نادرست;
        } وگرنه اگر (!args.prob_bias && nodes[گره_idx]->src[0] != nodes[گره_idx - 2]) {
            بازگشت نادرست;
        }

        گره_idx++;

        // ARGSORT -> VIEW
        اگر (گره_idx >= n_nodes || nodes[گره_idx]->op != GGML_OP_VIEW ||
            nodes[گره_idx]->src[0] != nodes[گره_idx - 1]) {
            بازگشت نادرست;
        }
        گره_idx++;

        اگر (گره_idx >= n_nodes || nodes[گره_idx]->op != GGML_OP_GET_ROWS) {
            بازگشت نادرست;
        }

        // GET_ROWS
        اگر (nodes[گره_idx]->src[0] != probs_reshaped || nodes[گره_idx]->src[1] != nodes[گره_idx - 1]) {
            بازگشت نادرست;
        }
        گره_idx++;
    } وگرنه اگر (args.delayed_softmax) {
        اگر (گره_idx - 2 < 0) {
            بازگشت نادرست;
        }
        ggml_tensor * probs_reshaped = nodes[گره_idx - 2];

        // VIEW -> ARGSORT
        اگر (گره_idx >= n_nodes || nodes[گره_idx]->op != GGML_OP_VIEW ||
            nodes[گره_idx]->src[0] != nodes[گره_idx - 1]) {
            بازگشت نادرست;
        }
        گره_idx++;

        // GET_ROWS
        اگر (گره_idx >= n_nodes || nodes[گره_idx]->src[1] != nodes[گره_idx - 1] ||
            nodes[گره_idx]->src[0] != probs_reshaped) {
            بازگشت نادرست;
        }
        گره_idx++;

        ایستا ثابت std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        برای (ثابت ggml_op op : remaining_ops) {
            اگر (گره_idx >= n_nodes || nodes[گره_idx]->op != op || nodes[گره_idx]->src[0] != nodes[گره_idx - 1]) {
                بازگشت نادرست;
            }
            گره_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    اگر (گره_idx >= n_nodes) {
        بازگشت درست;
    }

    اگر (nodes[گره_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        ایستا ثابت std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = درست;
        برای (ثابت ggml_op op : norm_ops) {
            اگر (nodes[گره_idx]->op == op && nodes[گره_idx]->src[0] == nodes[گره_idx - 1]) {
                گره_idx++;
            } وگرنه {
                args.norm = نادرست;
                بازگشت درست;
            }
        }

        // DIV <- CLAMP, RESHAPE
        اگر (nodes[گره_idx]->op != GGML_OP_DIV || nodes[گره_idx]->src[1] != nodes[گره_idx - 1] ||
            nodes[گره_idx]->src[0] != nodes[گره_idx - 3]) {
            args.norm = نادرست;
            بازگشت درست;
        }
        گره_idx++;

        اگر (nodes[گره_idx]->op != GGML_OP_RESHAPE || nodes[گره_idx]->src[0] != nodes[گره_idx - 1]) {
            args.norm = نادرست;
            بازگشت درست;
        }
        گره_idx++;
    }

    اگر (nodes[گره_idx]->op == GGML_OP_SCALE && nodes[گره_idx]->src[0] == nodes[گره_idx - 1]) {
        args.scale = درست;
    }

    بازگشت درست;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
ایستا منطقی ggml_sycl_check_fusion_memory_ranges(ثابت ggml_cgraph * cgraph, ثابت صحیح گره_idx,
                                                 ثابت صحیح گره_شمار, ثابت صحیح * out_nodes, ثابت صحیح out_شمار,
                                                 ثابت منطقی is_topk_moe = نادرست) {
    auto nodes_overlap = [&](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        ثابت int64_t a_start = (int64_t) a->data;
        ثابت int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        ثابت int64_t b_start = (int64_t) b->data;
        ثابت int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        اگر ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            بازگشت درست;
        }

        بازگشت نادرست;
    };

    منطقی is_ok = درست;
    // exception for topk-moe, as each row is read entirely before writing
    اگر (ggml_nrows(cgraph->nodes[گره_idx]) == 1 && is_topk_moe) {
        بازگشت درست;
    }

    برای (صحیح i = 0; i < out_شمار; ++i) {
        ثابت ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        برای (صحیح j = گره_idx; j < گره_idx + گره_شمار; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            برای (صحیح src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ثابت ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                اگر (!src || src->op == GGML_OP_NONE) {
                    ادامه;
                }

                اگر (nodes_overlap(dst, src)) {
                    منطقی found = نادرست;

                    برای (صحیح k = گره_idx; k < j; ++k) {
                        اگر (cgraph->nodes[k] == src) {
                            found = درست;
                            بشکن;
                        }
                    }

                    اگر (!found) {
                        is_ok = نادرست;
                        بشکن;
                    }
                }
            }
        }
    }

    بازگشت is_ok;
}

صحیح ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحیح i) {
    اگر (!g_ggml_sycl_enable_fusion) {
        بازگشت 0;
    }

    بازگشت ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

صحیح ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحیح i) {
    ggml_tensor * گره = cgraph->nodes[i];

    اگر (گره->op != GGML_OP_UNARY && گره->op != GGML_OP_SOFT_MAX && گره->op != GGML_OP_ARGSORT) {
        بازگشت 0;
    }

    ggml_sycl_topk_moe_args args;
    اگر (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        بازگشت 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    اگر (args.prob_bias) {
        بازگشت 0;
    }

    ثابت ggml_tensor * logits  = گره->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    ثابت ggml_tensor * clamp   = nullptr;
    ثابت ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    صحیح                  out_nodes[2];

    اگر (!args.delayed_softmax) {
        ثابت ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        اگر (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (صحیح) ops.size() - 3];
        }
        اگر (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (صحیح) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (صحیح) ops.size() - 1];
        out_nodes[1] = i + (صحیح) ops.size() - 1;

        اگر (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(گره, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (صحیح) ops.size(), out_nodes, 2, /*is_topk_moe=*/درست)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            بازگشت (صحیح) ops.size() - 1;
        }
    } وگرنه اگر (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ثابت ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        اگر (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (صحیح) ops.size(), out_nodes, 2, /*is_topk_moe=*/درست)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            بازگشت (صحیح) ops.size() - 1;
        }
    }

    بازگشت 0;
}
