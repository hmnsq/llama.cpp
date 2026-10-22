#অন্তর্ভাব <cfloat>
#অন্তর্ভাব <initializer_list>
#অন্তর্ভাব <vector>

#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

সংরচনা ggml_sycl_topk_moe_args {
    তার্কিক sigmoid{};
    তার্কিক softmax{};
    তার্কিক delayed_softmax{};
    তার্কিক prob_bias{};
    তার্কিক norm{};
    তার্কিক scale{};
};

সংরচনা topk_moe_config {
    তার্কিক use_sigmoid;
    তার্কিক with_norm;
    তার্কিক delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
template <পূর্ণাঙ্ক experts_per_thread, তার্কিক use_limit>
স্থির অন্তঃস্থ শূন্য softmax_warp_inplace(ভিন্ন (&vals)[experts_per_thread], ধ্রুব পূর্ণাঙ্ক limit, ধ্রুব পূর্ণাঙ্ক lane) {
    ভিন্ন max_val = -INFINITY;
#pragma unroll
    প্রতি (পূর্ণাঙ্ক i = 0; i < experts_per_thread; i++) {
        ধ্রুব পূর্ণাঙ্ক  idx    = lane + i * WARP_SIZE;
        ধ্রুব তার্কিক active = !use_limit || (idx < limit);
        যদি (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    ভিন্ন sum = 0.f;
#pragma unroll
    প্রতি (পূর্ণাঙ্ক i = 0; i < experts_per_thread; i++) {
        ধ্রুব পূর্ণাঙ্ক  idx    = lane + i * WARP_SIZE;
        ধ্রুব তার্কিক active = !use_limit || (idx < limit);
        যদি (active) {
            ধ্রুব ভিন্ন val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } অন্যথা {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    ধ্রুব ভিন্ন inv_sum = 1.0f / sum;
#pragma unroll
    প্রতি (পূর্ণাঙ্ক i = 0; i < experts_per_thread; i++) {
        ধ্রুব পূর্ণাঙ্ক idx = lane + i * WARP_SIZE;
        যদি (!use_limit || idx < limit) {
            vals[i] *= inv_sum;
        }
    }
}

template <পূর্ণাঙ্ক experts_per_thread, তার্কিক use_limit>
স্থির অন্তঃস্থ শূন্য sigmoid_warp_inplace(ভিন্ন (&vals)[experts_per_thread], ধ্রুব পূর্ণাঙ্ক limit, ধ্রুব পূর্ণাঙ্ক lane) {
#pragma unroll
    প্রতি (পূর্ণাঙ্ক i = 0; i < experts_per_thread; i++) {
        ধ্রুব পূর্ণাঙ্ক  idx    = lane + i * WARP_SIZE;
        ধ্রুব তার্কিক active = !use_limit || (idx < limit);
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
template <পূর্ণাঙ্ক n_experts>
স্থির শূন্য topk_moe_kernel(ধ্রুব ভিন্ন * __restrict__ logits,
                            ভিন্ন * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ধ্রুব পূর্ণাঙ্ক                  n_rows,
                            ধ্রুব পূর্ণাঙ্ক                  n_expert_used,
                            ধ্রুব ভিন্ন                clamp_val,
                            ধ্রুব ভিন্ন                scale_val,
                            ধ্রুব topk_moe_config       config) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    ধ্রুব পূর্ণাঙ্ক row       = item_ct1.get_group(0);
    যদি (row >= n_rows) {
        প্রত্যাগম;
    }
    ধ্রুব পূর্ণাঙ্ক lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr পূর্ণাঙ্ক experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    ভিন্ন wt[experts_per_thread];
#pragma unroll
    প্রতি (পূর্ণাঙ্ক i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    প্রতি (পূর্ণাঙ্ক i = 0; i < n_experts; i += WARP_SIZE) {
        ধ্রুব পূর্ণাঙ্ক expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    যদি (!config.delayed_softmax) {
        যদি (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, অসত্য>(wt, n_experts, lane);
        } অন্যথা {
            softmax_warp_inplace<experts_per_thread, অসত্য>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#pragma unroll
    প্রতি (পূর্ণাঙ্ক i = 0; i < experts_per_thread; i++) {
        যদি (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    ভিন্ন wt_sum = 0.f;
    ভিন্ন output_weights[experts_per_thread];
#pragma unroll
    প্রতি (পূর্ণাঙ্ক i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ধ্রুব sycl::sub_group sg = item_ct1.get_sub_group();

    প্রতি (পূর্ণাঙ্ক k = 0; k < n_expert_used; k++) {
        ভিন্ন max_val    = wt[0];
        পূর্ণাঙ্ক   max_expert = lane;
#pragma unroll
        প্রতি (পূর্ণাঙ্ক i = 1; i < experts_per_thread; i++) {
            ধ্রুব পূর্ণাঙ্ক expert = lane + i * WARP_SIZE;
            যদি ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        প্রতি (পূর্ণাঙ্ক mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            ধ্রুব ভিন্ন val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            ধ্রুব পূর্ণাঙ্ক   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            যদি (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        যদি ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
        }
        যদি ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        যদি ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            যদি (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    যদি (config.with_norm) {
        wt_sum          = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum          = sycl::fmax(wt_sum, clamp_val);
        ধ্রুব ভিন্ন inv = 1.0f / wt_sum;
#pragma unroll
        প্রতি (পূর্ণাঙ্ক i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    যদি (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, সত্য>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    প্রতি (পূর্ণাঙ্ক i = 0; i < experts_per_thread; i++) {
        ধ্রুব পূর্ণাঙ্ক idx = i * WARP_SIZE + lane;
        যদি (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <পূর্ণাঙ্ক n_experts>
স্থির শূন্য launch_topk_moe(queue_ptr stream, ধ্রুব ভিন্ন * logits, ভিন্ন * weights, int32_t * ids, পূর্ণাঙ্ক n_rows,
                            পূর্ণাঙ্ক n_expert_used, ভিন্ন clamp_val, ভিন্ন scale_val, ধ্রুব topk_moe_config & config) {
    ধ্রুব sycl::range<1> block_dims(WARP_SIZE);
    ধ্রুব sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                        scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

স্থির শূন্য ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ধ্রুব ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ধ্রুব ggml_tensor *             clamp,
                                  ধ্রুব ggml_tensor *             scale,
                                  ধ্রুব ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ধ্রুব পূর্ণাঙ্ক n_experts     = logits->ne[0];
    ধ্রুব পূর্ণাঙ্ক n_rows        = logits->ne[1];
    ধ্রুব পূর্ণাঙ্ক n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    ধ্রুব ভিন্ন * logits_d  = (ধ্রুব ভিন্ন *) logits->data;
    ভিন্ন *       weights_d = (ভিন্ন *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    ধ্রুব তার্কিক  with_norm = clamp != nullptr;
    ধ্রুব ভিন্ন clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ধ্রুব ভিন্ন scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    বিকল্প (n_experts) {
        পক্ষ 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            ভঙ্গ;
        পক্ষ 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            ভঙ্গ;
        পক্ষ 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            ভঙ্গ;
        পক্ষ 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            ভঙ্গ;
        পক্ষ 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            ভঙ্গ;
        পক্ষ 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            ভঙ্গ;
        পক্ষ 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            ভঙ্গ;
        পক্ষ 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            ভঙ্গ;
        পক্ষ 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            ভঙ্গ;
        পক্ষ 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            ভঙ্গ;
        স্বভাব:
            GGML_ASSERT(অসত্য && "fatal error");
    }
}

স্থির তার্কিক ggml_sycl_should_use_topk_moe(ধ্রুব ggml_tensor * gating_op, ধ্রুব ggml_tensor * weights,
                                          ধ্রুব ggml_tensor * logits, ধ্রুব ggml_tensor * ids) {
    ধ্রুব পূর্ণাঙ্ক n_expert = ids->nb[1] / ids->nb[0];
    যদি ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        প্রত্যাগম অসত্য;
    }

    যদি (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        প্রত্যাগম অসত্য;
    }

    যদি (gating_op->op == GGML_OP_SOFT_MAX) {
        ভিন্ন scale    = 1.0f;
        ভিন্ন max_bias = 0.0f;

        memcpy(&scale, (ধ্রুব ভিন্ন *) gating_op->op_params + 0, পরিমাণ(ভিন্ন));
        memcpy(&max_bias, (ধ্রুব ভিন্ন *) gating_op->op_params + 1, পরিমাণ(ভিন্ন));

        যদি (!ggml_is_contiguous(gating_op->src[0])) {
            প্রত্যাগম অসত্য;
        }
        যদি (scale != 1.0f || max_bias != 0.0f) {
            প্রত্যাগম অসত্য;
        }
        // don't fuse when masks or sinks are present
        যদি (gating_op->src[1] || gating_op->src[2]) {
            প্রত্যাগম অসত্য;
        }
    } অন্যথা যদি (gating_op->op == GGML_OP_UNARY) {
        যদি (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            প্রত্যাগম অসত্য;
        }
    }

    প্রত্যাগম সত্য;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
স্থির তার্কিক ggml_sycl_topk_moe_fusion(ধ্রুব ggml_cgraph * cgraph, পূর্ণাঙ্ক গ্রন্থি_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    ধ্রুব পূর্ণাঙ্ক      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    যদি (nodes[গ্রন্থি_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = সত্য;
    }

    যদি (nodes[গ্রন্থি_idx]->op == GGML_OP_UNARY) {
        যদি (ggml_get_unary_op(nodes[গ্রন্থি_idx]) != GGML_UNARY_OP_SIGMOID) {
            প্রত্যাগম অসত্য;
        }
        args.sigmoid = সত্য;
    }

    যদি (nodes[গ্রন্থি_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = সত্য;
    }

    গ্রন্থি_idx++;

    যদি (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        যদি (গ্রন্থি_idx >= n_nodes || nodes[গ্রন্থি_idx]->op != GGML_OP_RESHAPE ||
            nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 1]) {
            প্রত্যাগম অসত্য;
        }
        ggml_tensor * probs_reshaped = nodes[গ্রন্থি_idx];
        গ্রন্থি_idx++;

        যদি (গ্রন্থি_idx >= n_nodes) {
            প্রত্যাগম অসত্য;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        যদি (nodes[গ্রন্থি_idx]->op == GGML_OP_ADD && nodes[গ্রন্থি_idx]->src[0] == nodes[গ্রন্থি_idx - 2]) {
            args.prob_bias = সত্য;
            গ্রন্থি_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        যদি (গ্রন্থি_idx >= n_nodes || nodes[গ্রন্থি_idx]->op != GGML_OP_ARGSORT) {
            প্রত্যাগম অসত্য;
        }

        যদি (args.prob_bias && nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 1]) {
            প্রত্যাগম অসত্য;
        } অন্যথা যদি (!args.prob_bias && nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 2]) {
            প্রত্যাগম অসত্য;
        }

        গ্রন্থি_idx++;

        // ARGSORT -> VIEW
        যদি (গ্রন্থি_idx >= n_nodes || nodes[গ্রন্থি_idx]->op != GGML_OP_VIEW ||
            nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 1]) {
            প্রত্যাগম অসত্য;
        }
        গ্রন্থি_idx++;

        যদি (গ্রন্থি_idx >= n_nodes || nodes[গ্রন্থি_idx]->op != GGML_OP_GET_ROWS) {
            প্রত্যাগম অসত্য;
        }

        // GET_ROWS
        যদি (nodes[গ্রন্থি_idx]->src[0] != probs_reshaped || nodes[গ্রন্থি_idx]->src[1] != nodes[গ্রন্থি_idx - 1]) {
            প্রত্যাগম অসত্য;
        }
        গ্রন্থি_idx++;
    } অন্যথা যদি (args.delayed_softmax) {
        যদি (গ্রন্থি_idx - 2 < 0) {
            প্রত্যাগম অসত্য;
        }
        ggml_tensor * probs_reshaped = nodes[গ্রন্থি_idx - 2];

        // VIEW -> ARGSORT
        যদি (গ্রন্থি_idx >= n_nodes || nodes[গ্রন্থি_idx]->op != GGML_OP_VIEW ||
            nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 1]) {
            প্রত্যাগম অসত্য;
        }
        গ্রন্থি_idx++;

        // GET_ROWS
        যদি (গ্রন্থি_idx >= n_nodes || nodes[গ্রন্থি_idx]->src[1] != nodes[গ্রন্থি_idx - 1] ||
            nodes[গ্রন্থি_idx]->src[0] != probs_reshaped) {
            প্রত্যাগম অসত্য;
        }
        গ্রন্থি_idx++;

        স্থির ধ্রুব std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        প্রতি (ধ্রুব ggml_op op : remaining_ops) {
            যদি (গ্রন্থি_idx >= n_nodes || nodes[গ্রন্থি_idx]->op != op || nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 1]) {
                প্রত্যাগম অসত্য;
            }
            গ্রন্থি_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    যদি (গ্রন্থি_idx >= n_nodes) {
        প্রত্যাগম সত্য;
    }

    যদি (nodes[গ্রন্থি_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        স্থির ধ্রুব std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = সত্য;
        প্রতি (ধ্রুব ggml_op op : norm_ops) {
            যদি (nodes[গ্রন্থি_idx]->op == op && nodes[গ্রন্থি_idx]->src[0] == nodes[গ্রন্থি_idx - 1]) {
                গ্রন্থি_idx++;
            } অন্যথা {
                args.norm = অসত্য;
                প্রত্যাগম সত্য;
            }
        }

        // DIV <- CLAMP, RESHAPE
        যদি (nodes[গ্রন্থি_idx]->op != GGML_OP_DIV || nodes[গ্রন্থি_idx]->src[1] != nodes[গ্রন্থি_idx - 1] ||
            nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 3]) {
            args.norm = অসত্য;
            প্রত্যাগম সত্য;
        }
        গ্রন্থি_idx++;

        যদি (nodes[গ্রন্থি_idx]->op != GGML_OP_RESHAPE || nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 1]) {
            args.norm = অসত্য;
            প্রত্যাগম সত্য;
        }
        গ্রন্থি_idx++;
    }

    যদি (nodes[গ্রন্থি_idx]->op == GGML_OP_SCALE && nodes[গ্রন্থি_idx]->src[0] == nodes[গ্রন্থি_idx - 1]) {
        args.scale = সত্য;
    }

    প্রত্যাগম সত্য;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
স্থির তার্কিক ggml_sycl_check_fusion_memory_ranges(ধ্রুব ggml_cgraph * cgraph, ধ্রুব পূর্ণাঙ্ক গ্রন্থি_idx,
                                                 ধ্রুব পূর্ণাঙ্ক গ্রন্থি_গুনতি, ধ্রুব পূর্ণাঙ্ক * out_nodes, ধ্রুব পূর্ণাঙ্ক out_গুনতি,
                                                 ধ্রুব তার্কিক is_topk_moe = অসত্য) {
    auto nodes_overlap = [&](ধ্রুব ggml_tensor * a, ধ্রুব ggml_tensor * b) {
        ধ্রুব int64_t a_start = (int64_t) a->data;
        ধ্রুব int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        ধ্রুব int64_t b_start = (int64_t) b->data;
        ধ্রুব int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        যদি ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            প্রত্যাগম সত্য;
        }

        প্রত্যাগম অসত্য;
    };

    তার্কিক is_ok = সত্য;
    // exception for topk-moe, as each row is read entirely before writing
    যদি (ggml_nrows(cgraph->nodes[গ্রন্থি_idx]) == 1 && is_topk_moe) {
        প্রত্যাগম সত্য;
    }

    প্রতি (পূর্ণাঙ্ক i = 0; i < out_গুনতি; ++i) {
        ধ্রুব ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        প্রতি (পূর্ণাঙ্ক j = গ্রন্থি_idx; j < গ্রন্থি_idx + গ্রন্থি_গুনতি; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            প্রতি (পূর্ণাঙ্ক src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ধ্রুব ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                যদি (!src || src->op == GGML_OP_NONE) {
                    অনুবর্ত;
                }

                যদি (nodes_overlap(dst, src)) {
                    তার্কিক found = অসত্য;

                    প্রতি (পূর্ণাঙ্ক k = গ্রন্থি_idx; k < j; ++k) {
                        যদি (cgraph->nodes[k] == src) {
                            found = সত্য;
                            ভঙ্গ;
                        }
                    }

                    যদি (!found) {
                        is_ok = অসত্য;
                        ভঙ্গ;
                    }
                }
            }
        }
    }

    প্রত্যাগম is_ok;
}

পূর্ণাঙ্ক ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, পূর্ণাঙ্ক i) {
    যদি (!g_ggml_sycl_enable_fusion) {
        প্রত্যাগম 0;
    }

    প্রত্যাগম ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

পূর্ণাঙ্ক ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, পূর্ণাঙ্ক i) {
    ggml_tensor * গ্রন্থি = cgraph->nodes[i];

    যদি (গ্রন্থি->op != GGML_OP_UNARY && গ্রন্থি->op != GGML_OP_SOFT_MAX && গ্রন্থি->op != GGML_OP_ARGSORT) {
        প্রত্যাগম 0;
    }

    ggml_sycl_topk_moe_args args;
    যদি (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        প্রত্যাগম 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    যদি (args.prob_bias) {
        প্রত্যাগম 0;
    }

    ধ্রুব ggml_tensor * logits  = গ্রন্থি->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    ধ্রুব ggml_tensor * clamp   = nullptr;
    ধ্রুব ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    পূর্ণাঙ্ক                  out_nodes[2];

    যদি (!args.delayed_softmax) {
        ধ্রুব ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        যদি (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (পূর্ণাঙ্ক) ops.size() - 3];
        }
        যদি (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (পূর্ণাঙ্ক) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (পূর্ণাঙ্ক) ops.size() - 1];
        out_nodes[1] = i + (পূর্ণাঙ্ক) ops.size() - 1;

        যদি (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(গ্রন্থি, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (পূর্ণাঙ্ক) ops.size(), out_nodes, 2, /*is_topk_moe=*/সত্য)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            প্রত্যাগম (পূর্ণাঙ্ক) ops.size() - 1;
        }
    } অন্যথা যদি (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ধ্রুব ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        যদি (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (পূর্ণাঙ্ক) ops.size(), out_nodes, 2, /*is_topk_moe=*/সত্য)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            প্রত্যাগম (পূর্ণাঙ্ক) ops.size() - 1;
        }
    }

    প্রত্যাগম 0;
}
