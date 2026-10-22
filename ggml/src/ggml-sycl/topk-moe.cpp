#અન્તર્ભાવ <cfloat>
#અન્તર્ભાવ <initializer_list>
#અન્તર્ભાવ <vector>

#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

સંરચના ggml_sycl_topk_moe_args {
    તાર્કિક sigmoid{};
    તાર્કિક softmax{};
    તાર્કિક delayed_softmax{};
    તાર્કિક prob_bias{};
    તાર્કિક norm{};
    તાર્કિક scale{};
};

સંરચના topk_moe_config {
    તાર્કિક use_sigmoid;
    તાર્કિક with_norm;
    તાર્કિક delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
template <પૂર્ણાંક experts_per_thread, તાર્કિક use_limit>
સ્થિર અંતઃરેખ શૂન્ય softmax_warp_inplace(દશાંશ (&vals)[experts_per_thread], અચલ પૂર્ણાંક limit, અચલ પૂર્ણાંક lane) {
    દશાંશ max_val = -INFINITY;
#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < experts_per_thread; i++) {
        અચલ પૂર્ણાંક  idx    = lane + i * WARP_SIZE;
        અચલ તાર્કિક active = !use_limit || (idx < limit);
        જો (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    દશાંશ sum = 0.f;
#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < experts_per_thread; i++) {
        અચલ પૂર્ણાંક  idx    = lane + i * WARP_SIZE;
        અચલ તાર્કિક active = !use_limit || (idx < limit);
        જો (active) {
            અચલ દશાંશ val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } નહીંતર {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    અચલ દશાંશ inv_sum = 1.0f / sum;
#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < experts_per_thread; i++) {
        અચલ પૂર્ણાંક idx = lane + i * WARP_SIZE;
        જો (!use_limit || idx < limit) {
            vals[i] *= inv_sum;
        }
    }
}

template <પૂર્ણાંક experts_per_thread, તાર્કિક use_limit>
સ્થિર અંતઃરેખ શૂન્ય sigmoid_warp_inplace(દશાંશ (&vals)[experts_per_thread], અચલ પૂર્ણાંક limit, અચલ પૂર્ણાંક lane) {
#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < experts_per_thread; i++) {
        અચલ પૂર્ણાંક  idx    = lane + i * WARP_SIZE;
        અચલ તાર્કિક active = !use_limit || (idx < limit);
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
template <પૂર્ણાંક n_experts>
સ્થિર શૂન્ય topk_moe_kernel(અચલ દશાંશ * __restrict__ logits,
                            દશાંશ * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            અચલ પૂર્ણાંક                  n_rows,
                            અચલ પૂર્ણાંક                  n_expert_used,
                            અચલ દશાંશ                clamp_val,
                            અચલ દશાંશ                scale_val,
                            અચલ topk_moe_config       config) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    અચલ પૂર્ણાંક row       = item_ct1.get_group(0);
    જો (row >= n_rows) {
        પાછા;
    }
    અચલ પૂર્ણાંક lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr પૂર્ણાંક experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    દશાંશ wt[experts_per_thread];
#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < n_experts; i += WARP_SIZE) {
        અચલ પૂર્ણાંક expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    જો (!config.delayed_softmax) {
        જો (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, ખોટું>(wt, n_experts, lane);
        } નહીંતર {
            softmax_warp_inplace<experts_per_thread, ખોટું>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < experts_per_thread; i++) {
        જો (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    દશાંશ wt_sum = 0.f;
    દશાંશ output_weights[experts_per_thread];
#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    અચલ sycl::sub_group sg = item_ct1.get_sub_group();

    માટે (પૂર્ણાંક k = 0; k < n_expert_used; k++) {
        દશાંશ max_val    = wt[0];
        પૂર્ણાંક   max_expert = lane;
#pragma unroll
        માટે (પૂર્ણાંક i = 1; i < experts_per_thread; i++) {
            અચલ પૂર્ણાંક expert = lane + i * WARP_SIZE;
            જો ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            અચલ દશાંશ val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            અચલ પૂર્ણાંક   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            જો (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        જો ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        જો ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        જો ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            જો (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    જો (config.with_norm) {
        wt_sum          = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum          = sycl::fmax(wt_sum, clamp_val);
        અચલ દશાંશ inv = 1.0f / wt_sum;
#pragma unroll
        માટે (પૂર્ણાંક i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    જો (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, સાચું>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < experts_per_thread; i++) {
        અચલ પૂર્ણાંક idx = i * WARP_SIZE + lane;
        જો (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <પૂર્ણાંક n_experts>
સ્થિર શૂન્ય launch_topk_moe(queue_ptr stream, અચલ દશાંશ * logits, દશાંશ * weights, int32_t * ids, પૂર્ણાંક n_rows,
                            પૂર્ણાંક n_expert_used, દશાંશ clamp_val, દશાંશ scale_val, અચલ topk_moe_config & config) {
    અચલ sycl::range<1> block_dims(WARP_SIZE);
    અચલ sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                        scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

સ્થિર શૂન્ય ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  અચલ ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  અચલ ggml_tensor *             clamp,
                                  અચલ ggml_tensor *             scale,
                                  અચલ ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    અચલ પૂર્ણાંક n_experts     = logits->ne[0];
    અચલ પૂર્ણાંક n_rows        = logits->ne[1];
    અચલ પૂર્ણાંક n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    અચલ દશાંશ * logits_d  = (અચલ દશાંશ *) logits->data;
    દશાંશ *       weights_d = (દશાંશ *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    અચલ તાર્કિક  with_norm = clamp != nullptr;
    અચલ દશાંશ clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    અચલ દશાંશ scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    પસંદગી (n_experts) {
        સ્થિતિ 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            તોડો;
        સ્થિતિ 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            તોડો;
        સ્થિતિ 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            તોડો;
        સ્થિતિ 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            તોડો;
        સ્થિતિ 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            તોડો;
        સ્થિતિ 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            તોડો;
        સ્થિતિ 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            તોડો;
        સ્થિતિ 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            તોડો;
        સ્થિતિ 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            તોડો;
        સ્થિતિ 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            તોડો;
        મૂળભૂત:
            GGML_ASSERT(ખોટું && "fatal error");
    }
}

સ્થિર તાર્કિક ggml_sycl_should_use_topk_moe(અચલ ggml_tensor * gating_op, અચલ ggml_tensor * weights,
                                          અચલ ggml_tensor * logits, અચલ ggml_tensor * ids) {
    અચલ પૂર્ણાંક n_expert = ids->nb[1] / ids->nb[0];
    જો ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        પાછા ખોટું;
    }

    જો (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        પાછા ખોટું;
    }

    જો (gating_op->op == GGML_OP_SOFT_MAX) {
        દશાંશ scale    = 1.0f;
        દશાંશ max_bias = 0.0f;

        memcpy(&scale, (અચલ દશાંશ *) gating_op->op_params + 0, કદ(દશાંશ));
        memcpy(&max_bias, (અચલ દશાંશ *) gating_op->op_params + 1, કદ(દશાંશ));

        જો (!ggml_is_contiguous(gating_op->src[0])) {
            પાછા ખોટું;
        }
        જો (scale != 1.0f || max_bias != 0.0f) {
            પાછા ખોટું;
        }
        // don't fuse when masks or sinks are present
        જો (gating_op->src[1] || gating_op->src[2]) {
            પાછા ખોટું;
        }
    } નહીંતર જો (gating_op->op == GGML_OP_UNARY) {
        જો (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            પાછા ખોટું;
        }
    }

    પાછા સાચું;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
સ્થિર તાર્કિક ggml_sycl_topk_moe_fusion(અચલ ggml_cgraph * cgraph, પૂર્ણાંક ગાંઠ_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    અચલ પૂર્ણાંક      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    જો (nodes[ગાંઠ_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = સાચું;
    }

    જો (nodes[ગાંઠ_idx]->op == GGML_OP_UNARY) {
        જો (ggml_get_unary_op(nodes[ગાંઠ_idx]) != GGML_UNARY_OP_SIGMOID) {
            પાછા ખોટું;
        }
        args.sigmoid = સાચું;
    }

    જો (nodes[ગાંઠ_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = સાચું;
    }

    ગાંઠ_idx++;

    જો (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        જો (ગાંઠ_idx >= n_nodes || nodes[ગાંઠ_idx]->op != GGML_OP_RESHAPE ||
            nodes[ગાંઠ_idx]->src[0] != nodes[ગાંઠ_idx - 1]) {
            પાછા ખોટું;
        }
        ggml_tensor * probs_reshaped = nodes[ગાંઠ_idx];
        ગાંઠ_idx++;

        જો (ગાંઠ_idx >= n_nodes) {
            પાછા ખોટું;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        જો (nodes[ગાંઠ_idx]->op == GGML_OP_ADD && nodes[ગાંઠ_idx]->src[0] == nodes[ગાંઠ_idx - 2]) {
            args.prob_bias = સાચું;
            ગાંઠ_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        જો (ગાંઠ_idx >= n_nodes || nodes[ગાંઠ_idx]->op != GGML_OP_ARGSORT) {
            પાછા ખોટું;
        }

        જો (args.prob_bias && nodes[ગાંઠ_idx]->src[0] != nodes[ગાંઠ_idx - 1]) {
            પાછા ખોટું;
        } નહીંતર જો (!args.prob_bias && nodes[ગાંઠ_idx]->src[0] != nodes[ગાંઠ_idx - 2]) {
            પાછા ખોટું;
        }

        ગાંઠ_idx++;

        // ARGSORT -> VIEW
        જો (ગાંઠ_idx >= n_nodes || nodes[ગાંઠ_idx]->op != GGML_OP_VIEW ||
            nodes[ગાંઠ_idx]->src[0] != nodes[ગાંઠ_idx - 1]) {
            પાછા ખોટું;
        }
        ગાંઠ_idx++;

        જો (ગાંઠ_idx >= n_nodes || nodes[ગાંઠ_idx]->op != GGML_OP_GET_ROWS) {
            પાછા ખોટું;
        }

        // GET_ROWS
        જો (nodes[ગાંઠ_idx]->src[0] != probs_reshaped || nodes[ગાંઠ_idx]->src[1] != nodes[ગાંઠ_idx - 1]) {
            પાછા ખોટું;
        }
        ગાંઠ_idx++;
    } નહીંતર જો (args.delayed_softmax) {
        જો (ગાંઠ_idx - 2 < 0) {
            પાછા ખોટું;
        }
        ggml_tensor * probs_reshaped = nodes[ગાંઠ_idx - 2];

        // VIEW -> ARGSORT
        જો (ગાંઠ_idx >= n_nodes || nodes[ગાંઠ_idx]->op != GGML_OP_VIEW ||
            nodes[ગાંઠ_idx]->src[0] != nodes[ગાંઠ_idx - 1]) {
            પાછા ખોટું;
        }
        ગાંઠ_idx++;

        // GET_ROWS
        જો (ગાંઠ_idx >= n_nodes || nodes[ગાંઠ_idx]->src[1] != nodes[ગાંઠ_idx - 1] ||
            nodes[ગાંઠ_idx]->src[0] != probs_reshaped) {
            પાછા ખોટું;
        }
        ગાંઠ_idx++;

        સ્થિર અચલ std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        માટે (અચલ ggml_op op : remaining_ops) {
            જો (ગાંઠ_idx >= n_nodes || nodes[ગાંઠ_idx]->op != op || nodes[ગાંઠ_idx]->src[0] != nodes[ગાંઠ_idx - 1]) {
                પાછા ખોટું;
            }
            ગાંઠ_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    જો (ગાંઠ_idx >= n_nodes) {
        પાછા સાચું;
    }

    જો (nodes[ગાંઠ_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        સ્થિર અચલ std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = સાચું;
        માટે (અચલ ggml_op op : norm_ops) {
            જો (nodes[ગાંઠ_idx]->op == op && nodes[ગાંઠ_idx]->src[0] == nodes[ગાંઠ_idx - 1]) {
                ગાંઠ_idx++;
            } નહીંતર {
                args.norm = ખોટું;
                પાછા સાચું;
            }
        }

        // DIV <- CLAMP, RESHAPE
        જો (nodes[ગાંઠ_idx]->op != GGML_OP_DIV || nodes[ગાંઠ_idx]->src[1] != nodes[ગાંઠ_idx - 1] ||
            nodes[ગાંઠ_idx]->src[0] != nodes[ગાંઠ_idx - 3]) {
            args.norm = ખોટું;
            પાછા સાચું;
        }
        ગાંઠ_idx++;

        જો (nodes[ગાંઠ_idx]->op != GGML_OP_RESHAPE || nodes[ગાંઠ_idx]->src[0] != nodes[ગાંઠ_idx - 1]) {
            args.norm = ખોટું;
            પાછા સાચું;
        }
        ગાંઠ_idx++;
    }

    જો (nodes[ગાંઠ_idx]->op == GGML_OP_SCALE && nodes[ગાંઠ_idx]->src[0] == nodes[ગાંઠ_idx - 1]) {
        args.scale = સાચું;
    }

    પાછા સાચું;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
સ્થિર તાર્કિક ggml_sycl_check_fusion_memory_ranges(અચલ ggml_cgraph * cgraph, અચલ પૂર્ણાંક ગાંઠ_idx,
                                                 અચલ પૂર્ણાંક ગાંઠ_ગણતરી, અચલ પૂર્ણાંક * out_nodes, અચલ પૂર્ણાંક out_ગણતરી,
                                                 અચલ તાર્કિક is_topk_moe = ખોટું) {
    auto nodes_overlap = [&](અચલ ggml_tensor * a, અચલ ggml_tensor * b) {
        અચલ int64_t a_start = (int64_t) a->data;
        અચલ int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        અચલ int64_t b_start = (int64_t) b->data;
        અચલ int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        જો ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            પાછા સાચું;
        }

        પાછા ખોટું;
    };

    તાર્કિક is_ok = સાચું;
    // exception for topk-moe, as each row is read entirely before writing
    જો (ggml_nrows(cgraph->nodes[ગાંઠ_idx]) == 1 && is_topk_moe) {
        પાછા સાચું;
    }

    માટે (પૂર્ણાંક i = 0; i < out_ગણતરી; ++i) {
        અચલ ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        માટે (પૂર્ણાંક j = ગાંઠ_idx; j < ગાંઠ_idx + ગાંઠ_ગણતરી; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            માટે (પૂર્ણાંક src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                અચલ ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                જો (!src || src->op == GGML_OP_NONE) {
                    ચાલુ;
                }

                જો (nodes_overlap(dst, src)) {
                    તાર્કિક found = ખોટું;

                    માટે (પૂર્ણાંક k = ગાંઠ_idx; k < j; ++k) {
                        જો (cgraph->nodes[k] == src) {
                            found = સાચું;
                            તોડો;
                        }
                    }

                    જો (!found) {
                        is_ok = ખોટું;
                        તોડો;
                    }
                }
            }
        }
    }

    પાછા is_ok;
}

પૂર્ણાંક ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, પૂર્ણાંક i) {
    જો (!g_ggml_sycl_enable_fusion) {
        પાછા 0;
    }

    પાછા ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

પૂર્ણાંક ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, પૂર્ણાંક i) {
    ggml_tensor * ગાંઠ = cgraph->nodes[i];

    જો (ગાંઠ->op != GGML_OP_UNARY && ગાંઠ->op != GGML_OP_SOFT_MAX && ગાંઠ->op != GGML_OP_ARGSORT) {
        પાછા 0;
    }

    ggml_sycl_topk_moe_args args;
    જો (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        પાછા 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    જો (args.prob_bias) {
        પાછા 0;
    }

    અચલ ggml_tensor * logits  = ગાંઠ->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    અચલ ggml_tensor * clamp   = nullptr;
    અચલ ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    પૂર્ણાંક                  out_nodes[2];

    જો (!args.delayed_softmax) {
        અચલ ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        જો (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (પૂર્ણાંક) ops.size() - 3];
        }
        જો (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (પૂર્ણાંક) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (પૂર્ણાંક) ops.size() - 1];
        out_nodes[1] = i + (પૂર્ણાંક) ops.size() - 1;

        જો (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(ગાંઠ, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (પૂર્ણાંક) ops.size(), out_nodes, 2, /*is_topk_moe=*/સાચું)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            પાછા (પૂર્ણાંક) ops.size() - 1;
        }
    } નહીંતર જો (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        અચલ ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        જો (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (પૂર્ણાંક) ops.size(), out_nodes, 2, /*is_topk_moe=*/સાચું)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            પાછા (પૂર્ણાંક) ops.size() - 1;
        }
    }

    પાછા 0;
}
