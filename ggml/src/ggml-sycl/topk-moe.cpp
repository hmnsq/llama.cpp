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

কাঠামো ggml_sycl_topk_moe_args {
    যৌক্তিক sigmoid{};
    যৌক্তিক softmax{};
    যৌক্তিক delayed_softmax{};
    যৌক্তিক prob_bias{};
    যৌক্তিক norm{};
    যৌক্তিক scale{};
};

কাঠামো topk_moe_config {
    যৌক্তিক use_sigmoid;
    যৌক্তিক with_norm;
    যৌক্তিক delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
template <পূর্ণসংখ্যা experts_per_thread, যৌক্তিক use_limit>
স্থির অন্তঃরেখ শূন্য softmax_warp_inplace(দশমিক (&vals)[experts_per_thread], ধ্রুবক পূর্ণসংখ্যা limit, ধ্রুবক পূর্ণসংখ্যা lane) {
    দশমিক max_val = -INFINITY;
#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < experts_per_thread; i++) {
        ধ্রুবক পূর্ণসংখ্যা  idx    = lane + i * WARP_আকার;
        ধ্রুবক যৌক্তিক active = !use_limit || (idx < limit);
        যদি (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_আকার>(max_val);

    দশমিক যোগ_2 = 0.f;
#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < experts_per_thread; i++) {
        ধ্রুবক পূর্ণসংখ্যা  idx    = lane + i * WARP_আকার;
        ধ্রুবক যৌক্তিক active = !use_limit || (idx < limit);
        যদি (active) {
            ধ্রুবক দশমিক val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            যোগ_2 += val;
        } অন্যথা {
            vals[i] = 0.f;
        }
    }
    যোগ_2 = warp_reduce_যোগ<WARP_আকার>(যোগ_2);

    ধ্রুবক দশমিক inv_যোগ = 1.0f / যোগ_2;
#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < experts_per_thread; i++) {
        ধ্রুবক পূর্ণসংখ্যা idx = lane + i * WARP_আকার;
        যদি (!use_limit || idx < limit) {
            vals[i] *= inv_যোগ;
        }
    }
}

template <পূর্ণসংখ্যা experts_per_thread, যৌক্তিক use_limit>
স্থির অন্তঃরেখ শূন্য sigmoid_warp_inplace(দশমিক (&vals)[experts_per_thread], ধ্রুবক পূর্ণসংখ্যা limit, ধ্রুবক পূর্ণসংখ্যা lane) {
#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < experts_per_thread; i++) {
        ধ্রুবক পূর্ণসংখ্যা  idx    = lane + i * WARP_আকার;
        ধ্রুবক যৌক্তিক active = !use_limit || (idx < limit);
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
template <পূর্ণসংখ্যা n_experts>
স্থির শূন্য topk_moe_kernel(ধ্রুবক দশমিক * __restrict__ logits,
                            দশমিক * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ধ্রুবক পূর্ণসংখ্যা                  n_rows,
                            ধ্রুবক পূর্ণসংখ্যা                  n_expert_used,
                            ধ্রুবক দশমিক                clamp_val,
                            ধ্রুবক দশমিক                scale_val,
                            ধ্রুবক topk_moe_config       config) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    ধ্রুবক পূর্ণসংখ্যা row       = item_ct1.get_group(0);
    যদি (row >= n_rows) {
        ফেরত;
    }
    ধ্রুবক পূর্ণসংখ্যা lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_used * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr পূর্ণসংখ্যা experts_per_thread = (n_experts > WARP_আকার) ? n_experts / WARP_আকার : 1;

    দশমিক wt[experts_per_thread];
#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < n_experts; i += WARP_আকার) {
        ধ্রুবক পূর্ণসংখ্যা expert  = i + lane;
        wt[i / WARP_আকার] = (n_experts % WARP_আকার == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    যদি (!config.delayed_softmax) {
        যদি (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, মিথ্যা>(wt, n_experts, lane);
        } অন্যথা {
            softmax_warp_inplace<experts_per_thread, মিথ্যা>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < experts_per_thread; i++) {
        যদি (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    দশমিক wt_যোগ = 0.f;
    দশমিক output_weights[experts_per_thread];
#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ধ্রুবক sycl::sub_group sg = item_ct1.get_sub_group();

    জন্য (পূর্ণসংখ্যা k = 0; k < n_expert_used; k++) {
        দশমিক max_val    = wt[0];
        পূর্ণসংখ্যা   max_expert = lane;
#pragma unroll
        জন্য (পূর্ণসংখ্যা i = 1; i < experts_per_thread; i++) {
            ধ্রুবক পূর্ণসংখ্যা expert = lane + i * WARP_আকার;
            যদি ((n_experts % WARP_আকার == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        জন্য (পূর্ণসংখ্যা mask = WARP_আকার / 2; mask > 0; mask >>= 1) {
            ধ্রুবক দশমিক val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            ধ্রুবক পূর্ণসংখ্যা   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            যদি (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        যদি ((max_expert & (WARP_আকার - 1)) == lane) {
            wt[max_expert / WARP_আকার] = -INFINITY;
        }
        যদি ((k & (WARP_আকার - 1)) == lane) {
            output_weights[k / WARP_আকার] = max_val;
        }
        যদি ((max_expert & (WARP_আকার - 1)) == lane) {
            ids[k] = max_expert;
            যদি (config.with_norm) {
                wt_যোগ += max_val;
            }
        }
    }

    যদি (config.with_norm) {
        wt_যোগ          = warp_reduce_যোগ<WARP_আকার>(wt_যোগ);
        wt_যোগ          = sycl::fmax(wt_যোগ, clamp_val);
        ধ্রুবক দশমিক inv = 1.0f / wt_যোগ;
#pragma unroll
        জন্য (পূর্ণসংখ্যা i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    যদি (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, সত্য>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < experts_per_thread; i++) {
        ধ্রুবক পূর্ণসংখ্যা idx = i * WARP_আকার + lane;
        যদি (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template <পূর্ণসংখ্যা n_experts>
স্থির শূন্য launch_topk_moe(queue_ptr stream, ধ্রুবক দশমিক * logits, দশমিক * weights, int32_t * ids, পূর্ণসংখ্যা n_rows,
                            পূর্ণসংখ্যা n_expert_used, দশমিক clamp_val, দশমিক scale_val, ধ্রুবক topk_moe_config & config) {
    ধ্রুবক sycl::range<1> block_dims(WARP_আকার);
    ধ্রুবক sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                        scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

স্থির শূন্য ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ধ্রুবক ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ধ্রুবক ggml_tensor *             clamp,
                                  ধ্রুবক ggml_tensor *             scale,
                                  ধ্রুবক ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ধ্রুবক পূর্ণসংখ্যা n_experts     = logits->ne[0];
    ধ্রুবক পূর্ণসংখ্যা n_rows        = logits->ne[1];
    ধ্রুবক পূর্ণসংখ্যা n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_আকার(ids->type) == (size_t) n_experts);

    ধ্রুবক দশমিক * logits_d  = (ধ্রুবক দশমিক *) logits->তথ্য;
    দশমিক *       weights_d = (দশমিক *) weights->তথ্য;
    int32_t *     ids_d     = (int32_t *) ids->তথ্য;

    ধ্রুবক যৌক্তিক  with_norm = clamp != nullptr;
    ধ্রুবক দশমিক clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ধ্রুবক দশমিক scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    নির্বাচন (n_experts) {
        ক্ষেত্র 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            বিরতি;
        ক্ষেত্র 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            বিরতি;
        ক্ষেত্র 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            বিরতি;
        ক্ষেত্র 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            বিরতি;
        ক্ষেত্র 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            বিরতি;
        ক্ষেত্র 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            বিরতি;
        ক্ষেত্র 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            বিরতি;
        ক্ষেত্র 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            বিরতি;
        ক্ষেত্র 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            বিরতি;
        ক্ষেত্র 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            বিরতি;
        পূর্বনির্ধারিত:
            GGML_ASSERT(মিথ্যা && "fatal error");
    }
}

স্থির যৌক্তিক ggml_sycl_should_use_topk_moe(ধ্রুবক ggml_tensor * gating_op, ধ্রুবক ggml_tensor * weights,
                                          ধ্রুবক ggml_tensor * logits, ধ্রুবক ggml_tensor * ids) {
    ধ্রুবক পূর্ণসংখ্যা n_expert = ids->nb[1] / ids->nb[0];
    যদি ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        ফেরত মিথ্যা;
    }

    যদি (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        ফেরত মিথ্যা;
    }

    যদি (gating_op->op == GGML_OP_SOFT_MAX) {
        দশমিক scale    = 1.0f;
        দশমিক max_bias = 0.0f;

        memcpy(&scale, (ধ্রুবক দশমিক *) gating_op->op_params + 0, আকার(দশমিক));
        memcpy(&max_bias, (ধ্রুবক দশমিক *) gating_op->op_params + 1, আকার(দশমিক));

        যদি (!ggml_is_contiguous(gating_op->src[0])) {
            ফেরত মিথ্যা;
        }
        যদি (scale != 1.0f || max_bias != 0.0f) {
            ফেরত মিথ্যা;
        }
        // don't fuse when masks or sinks are present
        যদি (gating_op->src[1] || gating_op->src[2]) {
            ফেরত মিথ্যা;
        }
    } অন্যথা যদি (gating_op->op == GGML_OP_UNARY) {
        যদি (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            ফেরত মিথ্যা;
        }
    }

    ফেরত সত্য;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
স্থির যৌক্তিক ggml_sycl_topk_moe_fusion(ধ্রুবক ggml_cgraph * cgraph, পূর্ণসংখ্যা গ্রন্থি_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    ধ্রুবক পূর্ণসংখ্যা      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    যদি (nodes[গ্রন্থি_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = সত্য;
    }

    যদি (nodes[গ্রন্থি_idx]->op == GGML_OP_UNARY) {
        যদি (ggml_get_unary_op(nodes[গ্রন্থি_idx]) != GGML_UNARY_OP_SIGMOID) {
            ফেরত মিথ্যা;
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
            ফেরত মিথ্যা;
        }
        ggml_tensor * probs_reshaped = nodes[গ্রন্থি_idx];
        গ্রন্থি_idx++;

        যদি (গ্রন্থি_idx >= n_nodes) {
            ফেরত মিথ্যা;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        যদি (nodes[গ্রন্থি_idx]->op == GGML_OP_ADD && nodes[গ্রন্থি_idx]->src[0] == nodes[গ্রন্থি_idx - 2]) {
            args.prob_bias = সত্য;
            গ্রন্থি_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        যদি (গ্রন্থি_idx >= n_nodes || nodes[গ্রন্থি_idx]->op != GGML_OP_ARGSORT) {
            ফেরত মিথ্যা;
        }

        যদি (args.prob_bias && nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 1]) {
            ফেরত মিথ্যা;
        } অন্যথা যদি (!args.prob_bias && nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 2]) {
            ফেরত মিথ্যা;
        }

        গ্রন্থি_idx++;

        // ARGSORT -> VIEW
        যদি (গ্রন্থি_idx >= n_nodes || nodes[গ্রন্থি_idx]->op != GGML_OP_VIEW ||
            nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 1]) {
            ফেরত মিথ্যা;
        }
        গ্রন্থি_idx++;

        যদি (গ্রন্থি_idx >= n_nodes || nodes[গ্রন্থি_idx]->op != GGML_OP_GET_ROWS) {
            ফেরত মিথ্যা;
        }

        // GET_ROWS
        যদি (nodes[গ্রন্থি_idx]->src[0] != probs_reshaped || nodes[গ্রন্থি_idx]->src[1] != nodes[গ্রন্থি_idx - 1]) {
            ফেরত মিথ্যা;
        }
        গ্রন্থি_idx++;
    } অন্যথা যদি (args.delayed_softmax) {
        যদি (গ্রন্থি_idx - 2 < 0) {
            ফেরত মিথ্যা;
        }
        ggml_tensor * probs_reshaped = nodes[গ্রন্থি_idx - 2];

        // VIEW -> ARGSORT
        যদি (গ্রন্থি_idx >= n_nodes || nodes[গ্রন্থি_idx]->op != GGML_OP_VIEW ||
            nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 1]) {
            ফেরত মিথ্যা;
        }
        গ্রন্থি_idx++;

        // GET_ROWS
        যদি (গ্রন্থি_idx >= n_nodes || nodes[গ্রন্থি_idx]->src[1] != nodes[গ্রন্থি_idx - 1] ||
            nodes[গ্রন্থি_idx]->src[0] != probs_reshaped) {
            ফেরত মিথ্যা;
        }
        গ্রন্থি_idx++;

        স্থির ধ্রুবক std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        জন্য (ধ্রুবক ggml_op op : remaining_ops) {
            যদি (গ্রন্থি_idx >= n_nodes || nodes[গ্রন্থি_idx]->op != op || nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 1]) {
                ফেরত মিথ্যা;
            }
            গ্রন্থি_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    যদি (গ্রন্থি_idx >= n_nodes) {
        ফেরত সত্য;
    }

    যদি (nodes[গ্রন্থি_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        স্থির ধ্রুবক std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_যোগ_ROWS, GGML_OP_CLAMP };

        args.norm = সত্য;
        জন্য (ধ্রুবক ggml_op op : norm_ops) {
            যদি (nodes[গ্রন্থি_idx]->op == op && nodes[গ্রন্থি_idx]->src[0] == nodes[গ্রন্থি_idx - 1]) {
                গ্রন্থি_idx++;
            } অন্যথা {
                args.norm = মিথ্যা;
                ফেরত সত্য;
            }
        }

        // DIV <- CLAMP, RESHAPE
        যদি (nodes[গ্রন্থি_idx]->op != GGML_OP_DIV || nodes[গ্রন্থি_idx]->src[1] != nodes[গ্রন্থি_idx - 1] ||
            nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 3]) {
            args.norm = মিথ্যা;
            ফেরত সত্য;
        }
        গ্রন্থি_idx++;

        যদি (nodes[গ্রন্থি_idx]->op != GGML_OP_RESHAPE || nodes[গ্রন্থি_idx]->src[0] != nodes[গ্রন্থি_idx - 1]) {
            args.norm = মিথ্যা;
            ফেরত সত্য;
        }
        গ্রন্থি_idx++;
    }

    যদি (nodes[গ্রন্থি_idx]->op == GGML_OP_SCALE && nodes[গ্রন্থি_idx]->src[0] == nodes[গ্রন্থি_idx - 1]) {
        args.scale = সত্য;
    }

    ফেরত সত্য;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
স্থির যৌক্তিক ggml_sycl_check_fusion_memory_ranges(ধ্রুবক ggml_cgraph * cgraph, ধ্রুবক পূর্ণসংখ্যা গ্রন্থি_idx,
                                                 ধ্রুবক পূর্ণসংখ্যা গ্রন্থি_গুনতি, ধ্রুবক পূর্ণসংখ্যা * out_nodes, ধ্রুবক পূর্ণসংখ্যা out_গুনতি,
                                                 ধ্রুবক যৌক্তিক is_topk_moe = মিথ্যা) {
    auto nodes_overlap = [&](ধ্রুবক ggml_tensor * a, ধ্রুবক ggml_tensor * b) {
        ধ্রুবক int64_t a_start = (int64_t) a->তথ্য;
        ধ্রুবক int64_t a_end   = a_start + ggml_backend_buft_get_alloc_আকার(a->বাফার->buft, a);

        ধ্রুবক int64_t b_start = (int64_t) b->তথ্য;
        ধ্রুবক int64_t b_end   = b_start + ggml_backend_buft_get_alloc_আকার(b->বাফার->buft, b);

        যদি ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            ফেরত সত্য;
        }

        ফেরত মিথ্যা;
    };

    যৌক্তিক is_ok = সত্য;
    // exception for topk-moe, as each row is read entirely before writing
    যদি (ggml_nrows(cgraph->nodes[গ্রন্থি_idx]) == 1 && is_topk_moe) {
        ফেরত সত্য;
    }

    জন্য (পূর্ণসংখ্যা i = 0; i < out_গুনতি; ++i) {
        ধ্রুবক ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        জন্য (পূর্ণসংখ্যা j = গ্রন্থি_idx; j < গ্রন্থি_idx + গ্রন্থি_গুনতি; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            জন্য (পূর্ণসংখ্যা src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ধ্রুবক ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                যদি (!src || src->op == GGML_OP_NONE) {
                    চালু;
                }

                যদি (nodes_overlap(dst, src)) {
                    যৌক্তিক found = মিথ্যা;

                    জন্য (পূর্ণসংখ্যা k = গ্রন্থি_idx; k < j; ++k) {
                        যদি (cgraph->nodes[k] == src) {
                            found = সত্য;
                            বিরতি;
                        }
                    }

                    যদি (!found) {
                        is_ok = মিথ্যা;
                        বিরতি;
                    }
                }
            }
        }
    }

    ফেরত is_ok;
}

পূর্ণসংখ্যা ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, পূর্ণসংখ্যা i) {
    যদি (!g_ggml_sycl_enable_fusion) {
        ফেরত 0;
    }

    ফেরত ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

পূর্ণসংখ্যা ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, পূর্ণসংখ্যা i) {
    ggml_tensor * গ্রন্থি = cgraph->nodes[i];

    যদি (গ্রন্থি->op != GGML_OP_UNARY && গ্রন্থি->op != GGML_OP_SOFT_MAX && গ্রন্থি->op != GGML_OP_ARGSORT) {
        ফেরত 0;
    }

    ggml_sycl_topk_moe_args args;
    যদি (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        ফেরত 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    যদি (args.prob_bias) {
        ফেরত 0;
    }

    ধ্রুবক ggml_tensor * logits  = গ্রন্থি->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    ধ্রুবক ggml_tensor * clamp   = nullptr;
    ধ্রুবক ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    পূর্ণসংখ্যা                  out_nodes[2];

    যদি (!args.delayed_softmax) {
        ধ্রুবক ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        যদি (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_যোগ_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (পূর্ণসংখ্যা) ops.size() - 3];
        }
        যদি (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (পূর্ণসংখ্যা) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (পূর্ণসংখ্যা) ops.size() - 1];
        out_nodes[1] = i + (পূর্ণসংখ্যা) ops.size() - 1;

        যদি (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.তথ্য(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(গ্রন্থি, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (পূর্ণসংখ্যা) ops.size(), out_nodes, 2, /*is_topk_moe=*/সত্য)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            ফেরত (পূর্ণসংখ্যা) ops.size() - 1;
        }
    } অন্যথা যদি (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ধ্রুবক ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        যদি (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.তথ্য(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (পূর্ণসংখ্যা) ops.size(), out_nodes, 2, /*is_topk_moe=*/সত্য)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            ফেরত (পূর্ণসংখ্যা) ops.size() - 1;
        }
    }

    ফেরত 0;
}
