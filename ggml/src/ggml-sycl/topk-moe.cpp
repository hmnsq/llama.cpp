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

ساخت ggml_sycl_topk_moe_دلائل {
    منطقی sigmoid{};
    منطقی softmax{};
    منطقی delayed_softmax{};
    منطقی prob_bias{};
    منطقی norm{};
    منطقی scale{};
};

ساخت topk_moe_config {
    منطقی use_sigmoid;
    منطقی with_norm;
    منطقی delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
سانچہ <صحیح experts_فی_thread, منطقی use_حد_بندی>
ساکن اندرونی خالی softmax_warp_inplace(اعشاری (&vals)[experts_فی_thread], ثابت صحیح حد_بندی, ثابت صحیح lane) {
    اعشاری max_val = -INFINITY;
#ہدایت unroll
    برائے (صحیح i = 0; i < experts_فی_thread; i++) {
        ثابت صحیح  idx    = lane + i * WARP_سائز;
        ثابت منطقی فعال = !use_حد_بندی || (idx < حد_بندی);
        اگر (فعال) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_سائز>(max_val);

    اعشاری مجموع = 0.f;
#ہدایت unroll
    برائے (صحیح i = 0; i < experts_فی_thread; i++) {
        ثابت صحیح  idx    = lane + i * WARP_سائز;
        ثابت منطقی فعال = !use_حد_بندی || (idx < حد_بندی);
        اگر (فعال) {
            ثابت اعشاری val = sycl::قوت_نما(vals[i] - max_val);
            vals[i]         = val;
            مجموع += val;
        } ورنہ {
            vals[i] = 0.f;
        }
    }
    مجموع = warp_reduce_مجموع<WARP_سائز>(مجموع);

    ثابت اعشاری inv_مجموع = 1.0f / مجموع;
#ہدایت unroll
    برائے (صحیح i = 0; i < experts_فی_thread; i++) {
        ثابت صحیح idx = lane + i * WARP_سائز;
        اگر (!use_حد_بندی || idx < حد_بندی) {
            vals[i] *= inv_مجموع;
        }
    }
}

سانچہ <صحیح experts_فی_thread, منطقی use_حد_بندی>
ساکن اندرونی خالی sigmoid_warp_inplace(اعشاری (&vals)[experts_فی_thread], ثابت صحیح حد_بندی, ثابت صحیح lane) {
#ہدایت unroll
    برائے (صحیح i = 0; i < experts_فی_thread; i++) {
        ثابت صحیح  idx    = lane + i * WARP_سائز;
        ثابت منطقی فعال = !use_حد_بندی || (idx < حد_بندی);
        vals[i]           = فعال ? 1.f / (1.f + sycl::قوت_نما(-vals[i])) : -INFINITY;
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
سانچہ <صحیح n_experts>
ساکن خالی topk_moe_kernel(ثابت اعشاری * __restrict__ logits,
                            اعشاری * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ثابت صحیح                  n_صفیں,
                            ثابت صحیح                  n_expert_مستعمل,
                            ثابت اعشاری                clamp_val,
                            ثابت اعشاری                scale_val,
                            ثابت topk_moe_config       config) {
    خودکار      شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<1>();
    ثابت صحیح صف       = شے_ct1.get_گروہ(0);
    اگر (صف >= n_صفیں) {
        واپس;
    }
    ثابت صحیح lane = شے_ct1.get_local_id(0);

    logits  += (size_t) n_experts * صف;
    weights += (size_t) n_expert_مستعمل * صف;
    ids     += (size_t) n_experts * صف;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr صحیح experts_فی_thread = (n_experts > WARP_سائز) ? n_experts / WARP_سائز : 1;

    اعشاری wt[experts_فی_thread];
#ہدایت unroll
    برائے (صحیح i = 0; i < experts_فی_thread; i++) {
        wt[i] = -INFINITY;
    }
#ہدایت unroll
    برائے (صحیح i = 0; i < n_experts; i += WARP_سائز) {
        ثابت صحیح expert  = i + lane;
        wt[i / WARP_سائز] = (n_experts % WARP_سائز == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    اگر (!config.delayed_softmax) {
        اگر (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_فی_thread, جھوٹ>(wt, n_experts, lane);
        } ورنہ {
            softmax_warp_inplace<experts_فی_thread, جھوٹ>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#ہدایت unroll
    برائے (صحیح i = 0; i < experts_فی_thread; i++) {
        اگر (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    اعشاری wt_مجموع = 0.f;
    اعشاری output_weights[experts_فی_thread];
#ہدایت unroll
    برائے (صحیح i = 0; i < experts_فی_thread; i++) {
        output_weights[i] = 0.f;
    }

    ثابت sycl::sub_گروہ sg = شے_ct1.get_sub_گروہ();

    برائے (صحیح k = 0; k < n_expert_مستعمل; k++) {
        اعشاری max_val    = wt[0];
        صحیح   max_expert = lane;
#ہدایت unroll
        برائے (صحیح i = 1; i < experts_فی_thread; i++) {
            ثابت صحیح expert = lane + i * WARP_سائز;
            اگر ((n_experts % WARP_سائز == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#ہدایت unroll
        برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
            ثابت اعشاری val    = dpct::ترتیب_بدلو_sub_گروہ_by_xor(sg, max_val, نقاب);
            ثابت صحیح   expert = dpct::ترتیب_بدلو_sub_گروہ_by_xor(sg, max_expert, نقاب);
            اگر (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        اگر ((max_expert & (WARP_سائز - 1)) == lane) {
            wt[max_expert / WARP_سائز] = -INFINITY;
        }
        اگر ((k & (WARP_سائز - 1)) == lane) {
            output_weights[k / WARP_سائز] = max_val;
        }
        اگر ((max_expert & (WARP_سائز - 1)) == lane) {
            ids[k] = max_expert;
            اگر (config.with_norm) {
                wt_مجموع += max_val;
            }
        }
    }

    اگر (config.with_norm) {
        wt_مجموع          = warp_reduce_مجموع<WARP_سائز>(wt_مجموع);
        wt_مجموع          = sycl::fmax(wt_مجموع, clamp_val);
        ثابت اعشاری inv = 1.0f / wt_مجموع;
#ہدایت unroll
        برائے (صحیح i = 0; i < experts_فی_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    اگر (config.delayed_softmax) {
        softmax_warp_inplace<experts_فی_thread, سچ>(output_weights, n_expert_مستعمل, lane);
    }

#ہدایت unroll
    برائے (صحیح i = 0; i < experts_فی_thread; i++) {
        ثابت صحیح idx = i * WARP_سائز + lane;
        اگر (idx < n_expert_مستعمل) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

سانچہ <صحیح n_experts>
ساکن خالی launch_topk_moe(قطار_ptr stream, ثابت اعشاری * logits, اعشاری * weights, int32_t * ids, صحیح n_صفیں,
                            صحیح n_expert_مستعمل, اعشاری clamp_val, اعشاری scale_val, ثابت topk_moe_config & config) {
    ثابت sycl::range<1> بلاک_dims(WARP_سائز);
    ثابت sycl::range<1> بلاک_nums(n_صفیں);
    stream->parallel_for(sycl::nd_range<1>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_صفیں, n_expert_مستعمل, clamp_val,
                                                        scale_val, config);
                             GGML_غیر_مستعمل(شے_ct1);
                         });
}

ساکن خالی ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ثابت ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ثابت ggml_tensor *             clamp,
                                  ثابت ggml_tensor *             scale,
                                  ثابت ggml_sycl_topk_moe_دلائل & دلائل_2) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ثابت صحیح n_experts     = logits->ne[0];
    ثابت صحیح n_صفیں        = logits->ne[1];
    ثابت صحیح n_expert_مستعمل = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_سائز(ids->type) == (size_t) n_experts);

    ثابت اعشاری * logits_d  = (ثابت اعشاری *) logits->ڈیٹا;
    اعشاری *       weights_d = (اعشاری *) weights->ڈیٹا;
    int32_t *     ids_d     = (int32_t *) ids->ڈیٹا;

    ثابت منطقی  with_norm = clamp != خالی_اشاریہ;
    ثابت اعشاری clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ثابت اعشاری scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = دلائل_2.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = دلائل_2.delayed_softmax;

    قطار_ptr stream = ctx.stream();
    ggml_sycl_سیٹ_device(ctx.device);

    انتخاب (n_experts) {
        حالت 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_صفیں, n_expert_مستعمل, clamp_val, scale_val,
                               config);
            توڑو;
        حالت 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_صفیں, n_expert_مستعمل, clamp_val, scale_val,
                               config);
            توڑو;
        حالت 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_صفیں, n_expert_مستعمل, clamp_val, scale_val,
                               config);
            توڑو;
        حالت 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_صفیں, n_expert_مستعمل, clamp_val, scale_val,
                               config);
            توڑو;
        حالت 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_صفیں, n_expert_مستعمل, clamp_val, scale_val,
                                config);
            توڑو;
        حالت 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_صفیں, n_expert_مستعمل, clamp_val, scale_val,
                                config);
            توڑو;
        حالت 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_صفیں, n_expert_مستعمل, clamp_val, scale_val,
                                config);
            توڑو;
        حالت 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_صفیں, n_expert_مستعمل, clamp_val, scale_val,
                                 config);
            توڑو;
        حالت 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_صفیں, n_expert_مستعمل, clamp_val, scale_val,
                                 config);
            توڑو;
        حالت 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_صفیں, n_expert_مستعمل, clamp_val, scale_val,
                                 config);
            توڑو;
        طےشدہ:
            GGML_ASSERT(جھوٹ && "fatal error");
    }
}

ساکن منطقی ggml_sycl_should_use_topk_moe(ثابت ggml_tensor * gating_op, ثابت ggml_tensor * weights,
                                          ثابت ggml_tensor * logits, ثابت ggml_tensor * ids) {
    ثابت صحیح n_expert = ids->nb[1] / ids->nb[0];
    اگر ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        واپس جھوٹ;
    }

    اگر (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        واپس جھوٹ;
    }

    اگر (gating_op->op == GGML_OP_SOFT_MAX) {
        اعشاری scale    = 1.0f;
        اعشاری max_bias = 0.0f;

        حافظہ_نقل(&scale, (ثابت اعشاری *) gating_op->op_params + 0, حجم(اعشاری));
        حافظہ_نقل(&max_bias, (ثابت اعشاری *) gating_op->op_params + 1, حجم(اعشاری));

        اگر (!ggml_is_contiguous(gating_op->src[0])) {
            واپس جھوٹ;
        }
        اگر (scale != 1.0f || max_bias != 0.0f) {
            واپس جھوٹ;
        }
        // don't fuse when masks or sinks are present
        اگر (gating_op->src[1] || gating_op->src[2]) {
            واپس جھوٹ;
        }
    } ورنہ اگر (gating_op->op == GGML_OP_UNARY) {
        اگر (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            واپس جھوٹ;
        }
    }

    واپس سچ;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
ساکن منطقی ggml_sycl_topk_moe_fusion(ثابت ggml_cgraph * cgraph, صحیح گرہ_idx, ggml_sycl_topk_moe_دلائل & دلائل_2) {
    دلائل_2 = ggml_sycl_topk_moe_دلائل{};

    ثابت صحیح      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    اگر (nodes[گرہ_idx]->op == GGML_OP_SOFT_MAX) {
        دلائل_2.softmax = سچ;
    }

    اگر (nodes[گرہ_idx]->op == GGML_OP_UNARY) {
        اگر (ggml_get_unary_op(nodes[گرہ_idx]) != GGML_UNARY_OP_SIGMOID) {
            واپس جھوٹ;
        }
        دلائل_2.sigmoid = سچ;
    }

    اگر (nodes[گرہ_idx]->op == GGML_OP_ARGSORT) {
        دلائل_2.delayed_softmax = سچ;
    }

    گرہ_idx++;

    اگر (دلائل_2.sigmoid || دلائل_2.softmax) {
        // SOFTMAX -> RESHAPE
        اگر (گرہ_idx >= n_nodes || nodes[گرہ_idx]->op != GGML_OP_RESHAPE ||
            nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 1]) {
            واپس جھوٹ;
        }
        ggml_tensor * probs_reshaped = nodes[گرہ_idx];
        گرہ_idx++;

        اگر (گرہ_idx >= n_nodes) {
            واپس جھوٹ;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        اگر (nodes[گرہ_idx]->op == GGML_OP_جوڑو && nodes[گرہ_idx]->src[0] == nodes[گرہ_idx - 2]) {
            دلائل_2.prob_bias = سچ;
            گرہ_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        اگر (گرہ_idx >= n_nodes || nodes[گرہ_idx]->op != GGML_OP_ARGSORT) {
            واپس جھوٹ;
        }

        اگر (دلائل_2.prob_bias && nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 1]) {
            واپس جھوٹ;
        } ورنہ اگر (!دلائل_2.prob_bias && nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 2]) {
            واپس جھوٹ;
        }

        گرہ_idx++;

        // ARGSORT -> VIEW
        اگر (گرہ_idx >= n_nodes || nodes[گرہ_idx]->op != GGML_OP_VIEW ||
            nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 1]) {
            واپس جھوٹ;
        }
        گرہ_idx++;

        اگر (گرہ_idx >= n_nodes || nodes[گرہ_idx]->op != GGML_OP_GET_صفیں) {
            واپس جھوٹ;
        }

        // GET_ROWS
        اگر (nodes[گرہ_idx]->src[0] != probs_reshaped || nodes[گرہ_idx]->src[1] != nodes[گرہ_idx - 1]) {
            واپس جھوٹ;
        }
        گرہ_idx++;
    } ورنہ اگر (دلائل_2.delayed_softmax) {
        اگر (گرہ_idx - 2 < 0) {
            واپس جھوٹ;
        }
        ggml_tensor * probs_reshaped = nodes[گرہ_idx - 2];

        // VIEW -> ARGSORT
        اگر (گرہ_idx >= n_nodes || nodes[گرہ_idx]->op != GGML_OP_VIEW ||
            nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 1]) {
            واپس جھوٹ;
        }
        گرہ_idx++;

        // GET_ROWS
        اگر (گرہ_idx >= n_nodes || nodes[گرہ_idx]->src[1] != nodes[گرہ_idx - 1] ||
            nodes[گرہ_idx]->src[0] != probs_reshaped) {
            واپس جھوٹ;
        }
        گرہ_idx++;

        ساکن ثابت std::vector<ggml_op> بقایا_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        برائے (ثابت ggml_op op : بقایا_ops) {
            اگر (گرہ_idx >= n_nodes || nodes[گرہ_idx]->op != op || nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 1]) {
                واپس جھوٹ;
            }
            گرہ_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    اگر (گرہ_idx >= n_nodes) {
        واپس سچ;
    }

    اگر (nodes[گرہ_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        ساکن ثابت std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_مجموع_صفیں, GGML_OP_CLAMP };

        دلائل_2.norm = سچ;
        برائے (ثابت ggml_op op : norm_ops) {
            اگر (nodes[گرہ_idx]->op == op && nodes[گرہ_idx]->src[0] == nodes[گرہ_idx - 1]) {
                گرہ_idx++;
            } ورنہ {
                دلائل_2.norm = جھوٹ;
                واپس سچ;
            }
        }

        // DIV <- CLAMP, RESHAPE
        اگر (nodes[گرہ_idx]->op != GGML_OP_DIV || nodes[گرہ_idx]->src[1] != nodes[گرہ_idx - 1] ||
            nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 3]) {
            دلائل_2.norm = جھوٹ;
            واپس سچ;
        }
        گرہ_idx++;

        اگر (nodes[گرہ_idx]->op != GGML_OP_RESHAPE || nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 1]) {
            دلائل_2.norm = جھوٹ;
            واپس سچ;
        }
        گرہ_idx++;
    }

    اگر (nodes[گرہ_idx]->op == GGML_OP_SCALE && nodes[گرہ_idx]->src[0] == nodes[گرہ_idx - 1]) {
        دلائل_2.scale = سچ;
    }

    واپس سچ;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
ساکن منطقی ggml_sycl_check_fusion_حافظہ_ranges(ثابت ggml_cgraph * cgraph, ثابت صحیح گرہ_idx,
                                                 ثابت صحیح گرہ_گنتی, ثابت صحیح * برآمد_nodes, ثابت صحیح برآمد_گنتی,
                                                 ثابت منطقی is_topk_moe = جھوٹ) {
    خودکار nodes_overlap = [&](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        ثابت int64_t a_آغاز = (int64_t) a->ڈیٹا;
        ثابت int64_t a_انتہا   = a_آغاز + ggml_backend_buft_get_alloc_سائز(a->بفر->buft, a);

        ثابت int64_t b_آغاز = (int64_t) b->ڈیٹا;
        ثابت int64_t b_انتہا   = b_آغاز + ggml_backend_buft_get_alloc_سائز(b->بفر->buft, b);

        اگر ((b_آغاز <= a_آغاز && a_آغاز < b_انتہا) || (a_آغاز <= b_آغاز && b_آغاز < a_انتہا)) {
            واپس سچ;
        }

        واپس جھوٹ;
    };

    منطقی is_ok = سچ;
    // exception for topk-moe, as each row is read entirely before writing
    اگر (ggml_nrows(cgraph->nodes[گرہ_idx]) == 1 && is_topk_moe) {
        واپس سچ;
    }

    برائے (صحیح i = 0; i < برآمد_گنتی; ++i) {
        ثابت ggml_tensor * dst = cgraph->nodes[برآمد_nodes[i]];

        برائے (صحیح j = گرہ_idx; j < گرہ_idx + گرہ_گنتی; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            برائے (صحیح src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ثابت ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                اگر (!src || src->op == GGML_OP_NONE) {
                    جاری;
                }

                اگر (nodes_overlap(dst, src)) {
                    منطقی ملا = جھوٹ;

                    برائے (صحیح k = گرہ_idx; k < j; ++k) {
                        اگر (cgraph->nodes[k] == src) {
                            ملا = سچ;
                            توڑو;
                        }
                    }

                    اگر (!ملا) {
                        is_ok = جھوٹ;
                        توڑو;
                    }
                }
            }
        }
    }

    واپس is_ok;
}

صحیح ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحیح i) {
    اگر (!g_ggml_sycl_enable_fusion) {
        واپس 0;
    }

    واپس ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

صحیح ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحیح i) {
    ggml_tensor * گرہ = cgraph->nodes[i];

    اگر (گرہ->op != GGML_OP_UNARY && گرہ->op != GGML_OP_SOFT_MAX && گرہ->op != GGML_OP_ARGSORT) {
        واپس 0;
    }

    ggml_sycl_topk_moe_دلائل دلائل_2;
    اگر (!ggml_sycl_topk_moe_fusion(cgraph, i, دلائل_2)) {
        واپس 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    اگر (دلائل_2.prob_bias) {
        واپس 0;
    }

    ثابت ggml_tensor * logits  = گرہ->src[0];
    ggml_tensor *       weights = خالی_اشاریہ;
    ggml_tensor *       ids     = خالی_اشاریہ;
    ثابت ggml_tensor * clamp   = خالی_اشاریہ;
    ثابت ggml_tensor * scale   = خالی_اشاریہ;

    std::vector<ggml_op> ops;
    صحیح                  برآمد_nodes[2];

    اگر (!دلائل_2.delayed_softmax) {
        ثابت ggml_op gating_op = دلائل_2.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_صفیں });
        برآمد_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        اگر (دلائل_2.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_مجموع_صفیں, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (صحیح) ops.size() - 3];
        }
        اگر (دلائل_2.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (صحیح) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (صحیح) ops.size() - 1];
        برآمد_nodes[1] = i + (صحیح) ops.size() - 1;

        اگر (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.ڈیٹا(), برآمد_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(گرہ, weights, logits, ids) &&
            ggml_sycl_check_fusion_حافظہ_ranges(cgraph, i, (صحیح) ops.size(), برآمد_nodes, 2, /*is_topk_moe=*/سچ)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, دلائل_2);
            واپس (صحیح) ops.size() - 1;
        }
    } ورنہ اگر (!دلائل_2.norm && !دلائل_2.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_صفیں, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ثابت ggml_tensor * softmax = cgraph->nodes[i + 4];
        برآمد_nodes[0]                = i + 1;
        برآمد_nodes[1]                = i + 5;

        اگر (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.ڈیٹا(), برآمد_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_حافظہ_ranges(cgraph, i, (صحیح) ops.size(), برآمد_nodes, 2, /*is_topk_moe=*/سچ)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, دلائل_2);
            واپس (صحیح) ops.size() - 1;
        }
    }

    واپس 0;
}
