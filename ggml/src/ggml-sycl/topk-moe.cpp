#அந்தர்பாவ <cfloat>
#அந்தர்பாவ <initializer_list>
#அந்தர்பாவ <vector>

#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

கட்டமைப்பு ggml_sycl_topk_moe_args {
    தருக்க sigmoid{};
    தருக்க softmax{};
    தருக்க delayed_softmax{};
    தருக்க prob_bias{};
    தருக்க norm{};
    தருக்க scale{};
};

கட்டமைப்பு topk_moe_config {
    தருக்க use_sigmoid;
    தருக்க with_norm;
    தருக்க delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
template <முழு experts_per_thread, தருக்க use_limit>
நிலையான உள்வரி வெறுமை softmax_warp_inplace(தசம (&vals)[experts_per_thread], மாறிலி முழு limit, மாறிலி முழு lane) {
    தசம max_val = -INFINITY;
#pragma unroll
    க்கு (முழு i = 0; i < experts_per_thread; i++) {
        மாறிலி முழு  idx    = lane + i * WARP_SIZE;
        மாறிலி தருக்க active = !use_limit || (idx < limit);
        எனில் (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    தசம sum = 0.f;
#pragma unroll
    க்கு (முழு i = 0; i < experts_per_thread; i++) {
        மாறிலி முழு  idx    = lane + i * WARP_SIZE;
        மாறிலி தருக்க active = !use_limit || (idx < limit);
        எனில் (active) {
            மாறிலி தசம val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } இல்லையெனில் {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    மாறிலி தசம inv_sum = 1.0f / sum;
#pragma unroll
    க்கு (முழு i = 0; i < experts_per_thread; i++) {
        மாறிலி முழு idx = lane + i * WARP_SIZE;
        எனில் (!use_limit || idx < limit) {
            vals[i] *= inv_sum;
        }
    }
}

template <முழு experts_per_thread, தருக்க use_limit>
நிலையான உள்வரி வெறுமை sigmoid_warp_inplace(தசம (&vals)[experts_per_thread], மாறிலி முழு limit, மாறிலி முழு lane) {
#pragma unroll
    க்கு (முழு i = 0; i < experts_per_thread; i++) {
        மாறிலி முழு  idx    = lane + i * WARP_SIZE;
        மாறிலி தருக்க active = !use_limit || (idx < limit);
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
template <முழு n_experts>
நிலையான வெறுமை topk_moe_kernel(மாறிலி தசம * __restrict__ logits,
                            தசம * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            மாறிலி முழு                  n_rows,
                            மாறிலி முழு                  n_expert_used,
                            மாறிலி தசம                clamp_val,
                            மாறிலி தசம                scale_val,
                            மாறிலி topk_moe_config       config) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    மாறிலி முழு row       = item_ct1.get_group(0);
    எனில் (row >= n_rows) {
        திரும்பு;
    }
    மாறிலி முழு lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr முழு experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    தசம wt[experts_per_thread];
#pragma unroll
    க்கு (முழு i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    க்கு (முழு i = 0; i < n_experts; i += WARP_SIZE) {
        மாறிலி முழு expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    எனில் (!config.delayed_softmax) {
        எனில் (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, பொய்>(wt, n_experts, lane);
        } இல்லையெனில் {
            softmax_warp_inplace<experts_per_thread, பொய்>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#pragma unroll
    க்கு (முழு i = 0; i < experts_per_thread; i++) {
        எனில் (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    தசம wt_sum = 0.f;
    தசம output_weights[experts_per_thread];
#pragma unroll
    க்கு (முழு i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    மாறிலி sycl::sub_group sg = item_ct1.get_sub_group();

    க்கு (முழு k = 0; k < n_expert_used; k++) {
        தசம max_val    = wt[0];
        முழு   max_expert = lane;
#pragma unroll
        க்கு (முழு i = 1; i < experts_per_thread; i++) {
            மாறிலி முழு expert = lane + i * WARP_SIZE;
            எனில் ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        க்கு (முழு mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            மாறிலி தசம val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            மாறிலி முழு   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            எனில் (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        எனில் ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        எனில் ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        எனில் ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            எனில் (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    எனில் (config.with_norm) {
        wt_sum          = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum          = sycl::fmax(wt_sum, clamp_val);
        மாறிலி தசம inv = 1.0f / wt_sum;
#pragma unroll
        க்கு (முழு i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    எனில் (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, உண்மை>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    க்கு (முழு i = 0; i < experts_per_thread; i++) {
        மாறிலி முழு idx = i * WARP_SIZE + lane;
        எனில் (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <முழு n_experts>
நிலையான வெறுமை launch_topk_moe(queue_ptr stream, மாறிலி தசம * logits, தசம * weights, int32_t * ids, முழு n_rows,
                            முழு n_expert_used, தசம clamp_val, தசம scale_val, மாறிலி topk_moe_config & config) {
    மாறிலி sycl::range<1> block_dims(WARP_SIZE);
    மாறிலி sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                        scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

நிலையான வெறுமை ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  மாறிலி ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  மாறிலி ggml_tensor *             clamp,
                                  மாறிலி ggml_tensor *             scale,
                                  மாறிலி ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    மாறிலி முழு n_experts     = logits->ne[0];
    மாறிலி முழு n_rows        = logits->ne[1];
    மாறிலி முழு n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    மாறிலி தசம * logits_d  = (மாறிலி தசம *) logits->data;
    தசம *       weights_d = (தசம *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    மாறிலி தருக்க  with_norm = clamp != nullptr;
    மாறிலி தசம clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    மாறிலி தசம scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    தேர்வு (n_experts) {
        நிலை 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            நிறுத்து;
        நிலை 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            நிறுத்து;
        நிலை 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            நிறுத்து;
        நிலை 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            நிறுத்து;
        நிலை 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            நிறுத்து;
        நிலை 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            நிறுத்து;
        நிலை 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            நிறுத்து;
        நிலை 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            நிறுத்து;
        நிலை 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            நிறுத்து;
        நிலை 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            நிறுத்து;
        இயல்பு:
            GGML_ASSERT(பொய் && "fatal error");
    }
}

நிலையான தருக்க ggml_sycl_should_use_topk_moe(மாறிலி ggml_tensor * gating_op, மாறிலி ggml_tensor * weights,
                                          மாறிலி ggml_tensor * logits, மாறிலி ggml_tensor * ids) {
    மாறிலி முழு n_expert = ids->nb[1] / ids->nb[0];
    எனில் ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        திரும்பு பொய்;
    }

    எனில் (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        திரும்பு பொய்;
    }

    எனில் (gating_op->op == GGML_OP_SOFT_MAX) {
        தசம scale    = 1.0f;
        தசம max_bias = 0.0f;

        memcpy(&scale, (மாறிலி தசம *) gating_op->op_params + 0, அளவு(தசம));
        memcpy(&max_bias, (மாறிலி தசம *) gating_op->op_params + 1, அளவு(தசம));

        எனில் (!ggml_is_contiguous(gating_op->src[0])) {
            திரும்பு பொய்;
        }
        எனில் (scale != 1.0f || max_bias != 0.0f) {
            திரும்பு பொய்;
        }
        // don't fuse when masks or sinks are present
        எனில் (gating_op->src[1] || gating_op->src[2]) {
            திரும்பு பொய்;
        }
    } இல்லையெனில் எனில் (gating_op->op == GGML_OP_UNARY) {
        எனில் (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            திரும்பு பொய்;
        }
    }

    திரும்பு உண்மை;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
நிலையான தருக்க ggml_sycl_topk_moe_fusion(மாறிலி ggml_cgraph * cgraph, முழு முனை_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    மாறிலி முழு      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    எனில் (nodes[முனை_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = உண்மை;
    }

    எனில் (nodes[முனை_idx]->op == GGML_OP_UNARY) {
        எனில் (ggml_get_unary_op(nodes[முனை_idx]) != GGML_UNARY_OP_SIGMOID) {
            திரும்பு பொய்;
        }
        args.sigmoid = உண்மை;
    }

    எனில் (nodes[முனை_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = உண்மை;
    }

    முனை_idx++;

    எனில் (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        எனில் (முனை_idx >= n_nodes || nodes[முனை_idx]->op != GGML_OP_RESHAPE ||
            nodes[முனை_idx]->src[0] != nodes[முனை_idx - 1]) {
            திரும்பு பொய்;
        }
        ggml_tensor * probs_reshaped = nodes[முனை_idx];
        முனை_idx++;

        எனில் (முனை_idx >= n_nodes) {
            திரும்பு பொய்;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        எனில் (nodes[முனை_idx]->op == GGML_OP_ADD && nodes[முனை_idx]->src[0] == nodes[முனை_idx - 2]) {
            args.prob_bias = உண்மை;
            முனை_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        எனில் (முனை_idx >= n_nodes || nodes[முனை_idx]->op != GGML_OP_ARGSORT) {
            திரும்பு பொய்;
        }

        எனில் (args.prob_bias && nodes[முனை_idx]->src[0] != nodes[முனை_idx - 1]) {
            திரும்பு பொய்;
        } இல்லையெனில் எனில் (!args.prob_bias && nodes[முனை_idx]->src[0] != nodes[முனை_idx - 2]) {
            திரும்பு பொய்;
        }

        முனை_idx++;

        // ARGSORT -> VIEW
        எனில் (முனை_idx >= n_nodes || nodes[முனை_idx]->op != GGML_OP_VIEW ||
            nodes[முனை_idx]->src[0] != nodes[முனை_idx - 1]) {
            திரும்பு பொய்;
        }
        முனை_idx++;

        எனில் (முனை_idx >= n_nodes || nodes[முனை_idx]->op != GGML_OP_GET_ROWS) {
            திரும்பு பொய்;
        }

        // GET_ROWS
        எனில் (nodes[முனை_idx]->src[0] != probs_reshaped || nodes[முனை_idx]->src[1] != nodes[முனை_idx - 1]) {
            திரும்பு பொய்;
        }
        முனை_idx++;
    } இல்லையெனில் எனில் (args.delayed_softmax) {
        எனில் (முனை_idx - 2 < 0) {
            திரும்பு பொய்;
        }
        ggml_tensor * probs_reshaped = nodes[முனை_idx - 2];

        // VIEW -> ARGSORT
        எனில் (முனை_idx >= n_nodes || nodes[முனை_idx]->op != GGML_OP_VIEW ||
            nodes[முனை_idx]->src[0] != nodes[முனை_idx - 1]) {
            திரும்பு பொய்;
        }
        முனை_idx++;

        // GET_ROWS
        எனில் (முனை_idx >= n_nodes || nodes[முனை_idx]->src[1] != nodes[முனை_idx - 1] ||
            nodes[முனை_idx]->src[0] != probs_reshaped) {
            திரும்பு பொய்;
        }
        முனை_idx++;

        நிலையான மாறிலி std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        க்கு (மாறிலி ggml_op op : remaining_ops) {
            எனில் (முனை_idx >= n_nodes || nodes[முனை_idx]->op != op || nodes[முனை_idx]->src[0] != nodes[முனை_idx - 1]) {
                திரும்பு பொய்;
            }
            முனை_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    எனில் (முனை_idx >= n_nodes) {
        திரும்பு உண்மை;
    }

    எனில் (nodes[முனை_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        நிலையான மாறிலி std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = உண்மை;
        க்கு (மாறிலி ggml_op op : norm_ops) {
            எனில் (nodes[முனை_idx]->op == op && nodes[முனை_idx]->src[0] == nodes[முனை_idx - 1]) {
                முனை_idx++;
            } இல்லையெனில் {
                args.norm = பொய்;
                திரும்பு உண்மை;
            }
        }

        // DIV <- CLAMP, RESHAPE
        எனில் (nodes[முனை_idx]->op != GGML_OP_DIV || nodes[முனை_idx]->src[1] != nodes[முனை_idx - 1] ||
            nodes[முனை_idx]->src[0] != nodes[முனை_idx - 3]) {
            args.norm = பொய்;
            திரும்பு உண்மை;
        }
        முனை_idx++;

        எனில் (nodes[முனை_idx]->op != GGML_OP_RESHAPE || nodes[முனை_idx]->src[0] != nodes[முனை_idx - 1]) {
            args.norm = பொய்;
            திரும்பு உண்மை;
        }
        முனை_idx++;
    }

    எனில் (nodes[முனை_idx]->op == GGML_OP_SCALE && nodes[முனை_idx]->src[0] == nodes[முனை_idx - 1]) {
        args.scale = உண்மை;
    }

    திரும்பு உண்மை;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
நிலையான தருக்க ggml_sycl_check_fusion_memory_ranges(மாறிலி ggml_cgraph * cgraph, மாறிலி முழு முனை_idx,
                                                 மாறிலி முழு முனை_மொத்தம், மாறிலி முழு * out_nodes, மாறிலி முழு out_மொத்தம்,
                                                 மாறிலி தருக்க is_topk_moe = பொய்) {
    auto nodes_overlap = [&](மாறிலி ggml_tensor * a, மாறிலி ggml_tensor * b) {
        மாறிலி int64_t a_start = (int64_t) a->data;
        மாறிலி int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        மாறிலி int64_t b_start = (int64_t) b->data;
        மாறிலி int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        எனில் ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            திரும்பு உண்மை;
        }

        திரும்பு பொய்;
    };

    தருக்க is_ok = உண்மை;
    // exception for topk-moe, as each row is read entirely before writing
    எனில் (ggml_nrows(cgraph->nodes[முனை_idx]) == 1 && is_topk_moe) {
        திரும்பு உண்மை;
    }

    க்கு (முழு i = 0; i < out_மொத்தம்; ++i) {
        மாறிலி ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        க்கு (முழு j = முனை_idx; j < முனை_idx + முனை_மொத்தம்; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            க்கு (முழு src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                மாறிலி ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                எனில் (!src || src->op == GGML_OP_NONE) {
                    தொடர்;
                }

                எனில் (nodes_overlap(dst, src)) {
                    தருக்க found = பொய்;

                    க்கு (முழு k = முனை_idx; k < j; ++k) {
                        எனில் (cgraph->nodes[k] == src) {
                            found = உண்மை;
                            நிறுத்து;
                        }
                    }

                    எனில் (!found) {
                        is_ok = பொய்;
                        நிறுத்து;
                    }
                }
            }
        }
    }

    திரும்பு is_ok;
}

முழு ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, முழு i) {
    எனில் (!g_ggml_sycl_enable_fusion) {
        திரும்பு 0;
    }

    திரும்பு ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

முழு ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, முழு i) {
    ggml_tensor * முனை = cgraph->nodes[i];

    எனில் (முனை->op != GGML_OP_UNARY && முனை->op != GGML_OP_SOFT_MAX && முனை->op != GGML_OP_ARGSORT) {
        திரும்பு 0;
    }

    ggml_sycl_topk_moe_args args;
    எனில் (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        திரும்பு 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    எனில் (args.prob_bias) {
        திரும்பு 0;
    }

    மாறிலி ggml_tensor * logits  = முனை->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    மாறிலி ggml_tensor * clamp   = nullptr;
    மாறிலி ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    முழு                  out_nodes[2];

    எனில் (!args.delayed_softmax) {
        மாறிலி ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        எனில் (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (முழு) ops.size() - 3];
        }
        எனில் (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (முழு) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (முழு) ops.size() - 1];
        out_nodes[1] = i + (முழு) ops.size() - 1;

        எனில் (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(முனை, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (முழு) ops.size(), out_nodes, 2, /*is_topk_moe=*/உண்மை)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            திரும்பு (முழு) ops.size() - 1;
        }
    } இல்லையெனில் எனில் (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        மாறிலி ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        எனில் (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (முழு) ops.size(), out_nodes, 2, /*is_topk_moe=*/உண்மை)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            திரும்பு (முழு) ops.size() - 1;
        }
    }

    திரும்பு 0;
}
