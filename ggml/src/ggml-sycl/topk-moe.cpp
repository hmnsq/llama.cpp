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

بنية ggml_sycl_topk_moe_دليلون {
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
سانچو <صحيح experts_في_thread, منطقي use_حد_بندي>
ساكن مضمن فراغ softmax_warp_inplace(عائم (&vals)[experts_في_thread], ثابت صحيح حد_بندي, ثابت صحيح lane) {
    عائم max_val = -INFINITY;
#هدايت unroll
    لكل (صحيح i = 0; i < experts_في_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_سائيز;
        ثابت منطقي سرگرم = !use_حد_بندي || (idx < حد_بندي);
        إذا (سرگرم) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_سائيز>(max_val);

    عائم جوڙ = 0.f;
#هدايت unroll
    لكل (صحيح i = 0; i < experts_في_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_سائيز;
        ثابت منطقي سرگرم = !use_حد_بندي || (idx < حد_بندي);
        إذا (سرگرم) {
            ثابت عائم val = sycl::قوت_نما(vals[i] - max_val);
            vals[i]         = val;
            جوڙ += val;
        } وإلا {
            vals[i] = 0.f;
        }
    }
    جوڙ = warp_reduce_جوڙ<WARP_سائيز>(جوڙ);

    ثابت عائم inv_جوڙ = 1.0f / جوڙ;
#هدايت unroll
    لكل (صحيح i = 0; i < experts_في_thread; i++) {
        ثابت صحيح idx = lane + i * WARP_سائيز;
        إذا (!use_حد_بندي || idx < حد_بندي) {
            vals[i] *= inv_جوڙ;
        }
    }
}

سانچو <صحيح experts_في_thread, منطقي use_حد_بندي>
ساكن مضمن فراغ sigmoid_warp_inplace(عائم (&vals)[experts_في_thread], ثابت صحيح حد_بندي, ثابت صحيح lane) {
#هدايت unroll
    لكل (صحيح i = 0; i < experts_في_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_سائيز;
        ثابت منطقي سرگرم = !use_حد_بندي || (idx < حد_بندي);
        vals[i]           = سرگرم ? 1.f / (1.f + sycl::قوت_نما(-vals[i])) : -INFINITY;
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
سانچو <صحيح n_experts>
ساكن فراغ topk_moe_kernel(ثابت عائم * __restrict__ logits,
                            عائم * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ثابت صحيح                  n_صفون,
                            ثابت صحيح                  n_expert_استعمال_ٿيل,
                            ثابت عائم                clamp_val,
                            ثابت عائم                scale_val,
                            ثابت topk_moe_config       config) {
    خودڪار      شي_ct1 = sycl::ext::oneapi::this_work_شي::get_nd_شي<1>();
    ثابت صحيح صف       = شي_ct1.get_گروهه(0);
    إذا (صف >= n_صفون) {
        أرجع;
    }
    ثابت صحيح lane = شي_ct1.get_local_id(0);

    logits  += (size_t) n_experts * صف;
    weights += (size_t) n_expert_استعمال_ٿيل * صف;
    ids     += (size_t) n_experts * صف;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr صحيح experts_في_thread = (n_experts > WARP_سائيز) ? n_experts / WARP_سائيز : 1;

    عائم wt[experts_في_thread];
#هدايت unroll
    لكل (صحيح i = 0; i < experts_في_thread; i++) {
        wt[i] = -INFINITY;
    }
#هدايت unroll
    لكل (صحيح i = 0; i < n_experts; i += WARP_سائيز) {
        ثابت صحيح expert  = i + lane;
        wt[i / WARP_سائيز] = (n_experts % WARP_سائيز == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    إذا (!config.delayed_softmax) {
        إذا (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_في_thread, خطأ>(wt, n_experts, lane);
        } وإلا {
            softmax_warp_inplace<experts_في_thread, خطأ>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#هدايت unroll
    لكل (صحيح i = 0; i < experts_في_thread; i++) {
        إذا (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    عائم wt_جوڙ = 0.f;
    عائم output_weights[experts_في_thread];
#هدايت unroll
    لكل (صحيح i = 0; i < experts_في_thread; i++) {
        output_weights[i] = 0.f;
    }

    ثابت sycl::sub_گروهه sg = شي_ct1.get_sub_گروهه();

    لكل (صحيح k = 0; k < n_expert_استعمال_ٿيل; k++) {
        عائم max_val    = wt[0];
        صحيح   max_expert = lane;
#هدايت unroll
        لكل (صحيح i = 1; i < experts_في_thread; i++) {
            ثابت صحيح expert = lane + i * WARP_سائيز;
            إذا ((n_experts % WARP_سائيز == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#هدايت unroll
        لكل (صحيح نقاب = WARP_سائيز / 2; نقاب > 0; نقاب >>= 1) {
            ثابت عائم val    = dpct::ترتيب_بدلايو_sub_گروهه_by_xor(sg, max_val, نقاب);
            ثابت صحيح   expert = dpct::ترتيب_بدلايو_sub_گروهه_by_xor(sg, max_expert, نقاب);
            إذا (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        إذا ((max_expert & (WARP_سائيز - 1)) == lane) {
            wt[max_expert / WARP_سائيز] = -INFINITY;
        }
        إذا ((k & (WARP_سائيز - 1)) == lane) {
            output_weights[k / WARP_سائيز] = max_val;
        }
        إذا ((max_expert & (WARP_سائيز - 1)) == lane) {
            ids[k] = max_expert;
            إذا (config.with_norm) {
                wt_جوڙ += max_val;
            }
        }
    }

    إذا (config.with_norm) {
        wt_جوڙ          = warp_reduce_جوڙ<WARP_سائيز>(wt_جوڙ);
        wt_جوڙ          = sycl::fmax(wt_جوڙ, clamp_val);
        ثابت عائم inv = 1.0f / wt_جوڙ;
#هدايت unroll
        لكل (صحيح i = 0; i < experts_في_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    إذا (config.delayed_softmax) {
        softmax_warp_inplace<experts_في_thread, سچ>(output_weights, n_expert_استعمال_ٿيل, lane);
    }

#هدايت unroll
    لكل (صحيح i = 0; i < experts_في_thread; i++) {
        ثابت صحيح idx = i * WARP_سائيز + lane;
        إذا (idx < n_expert_استعمال_ٿيل) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

سانچو <صحيح n_experts>
ساكن فراغ launch_topk_moe(قطار_ptr stream, ثابت عائم * logits, عائم * weights, int32_t * ids, صحيح n_صفون,
                            صحيح n_expert_استعمال_ٿيل, عائم clamp_val, عائم scale_val, ثابت topk_moe_config & config) {
    ثابت sycl::range<1> بلاڪ_dims(WARP_سائيز);
    ثابت sycl::range<1> بلاڪ_nums(n_صفون);
    stream->parallel_for(sycl::nd_range<1>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
                         [=](sycl::nd_شي<1> شي_ct1) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_صفون, n_expert_استعمال_ٿيل, clamp_val,
                                                        scale_val, config);
                             GGML_اڻ_استعمال_ٿيل(شي_ct1);
                         });
}

ساكن فراغ ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ثابت ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ثابت ggml_tensor *             clamp,
                                  ثابت ggml_tensor *             scale,
                                  ثابت ggml_sycl_topk_moe_دليلون & دليلون_2) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ثابت صحيح n_experts     = logits->ne[0];
    ثابت صحيح n_صفون        = logits->ne[1];
    ثابت صحيح n_expert_استعمال_ٿيل = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_سائيز(ids->type) == (size_t) n_experts);

    ثابت عائم * logits_d  = (ثابت عائم *) logits->معلومات;
    عائم *       weights_d = (عائم *) weights->معلومات;
    int32_t *     ids_d     = (int32_t *) ids->معلومات;

    ثابت منطقي  with_norm = clamp != خالي_اشارو;
    ثابت عائم clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ثابت عائم scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = دليلون_2.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = دليلون_2.delayed_softmax;

    قطار_ptr stream = ctx.stream();
    ggml_sycl_سيٽ_device(ctx.device);

    بدل (n_experts) {
        حالة 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_صفون, n_expert_استعمال_ٿيل, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_صفون, n_expert_استعمال_ٿيل, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_صفون, n_expert_استعمال_ٿيل, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_صفون, n_expert_استعمال_ٿيل, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_صفون, n_expert_استعمال_ٿيل, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_صفون, n_expert_استعمال_ٿيل, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_صفون, n_expert_استعمال_ٿيل, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_صفون, n_expert_استعمال_ٿيل, clamp_val, scale_val,
                                 config);
            اكسر;
        حالة 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_صفون, n_expert_استعمال_ٿيل, clamp_val, scale_val,
                                 config);
            اكسر;
        حالة 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_صفون, n_expert_استعمال_ٿيل, clamp_val, scale_val,
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

        يادگيري_نقل(&scale, (ثابت عائم *) gating_op->op_params + 0, حجم(عائم));
        يادگيري_نقل(&max_bias, (ثابت عائم *) gating_op->op_params + 1, حجم(عائم));

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

    أرجع سچ;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
ساكن منطقي ggml_sycl_topk_moe_fusion(ثابت ggml_cgraph * cgraph, صحيح ڳنڍ_idx, ggml_sycl_topk_moe_دليلون & دليلون_2) {
    دليلون_2 = ggml_sycl_topk_moe_دليلون{};

    ثابت صحيح      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    إذا (nodes[ڳنڍ_idx]->op == GGML_OP_SOFT_MAX) {
        دليلون_2.softmax = سچ;
    }

    إذا (nodes[ڳنڍ_idx]->op == GGML_OP_UNARY) {
        إذا (ggml_get_unary_op(nodes[ڳنڍ_idx]) != GGML_UNARY_OP_SIGMOID) {
            أرجع خطأ;
        }
        دليلون_2.sigmoid = سچ;
    }

    إذا (nodes[ڳنڍ_idx]->op == GGML_OP_ARGSORT) {
        دليلون_2.delayed_softmax = سچ;
    }

    ڳنڍ_idx++;

    إذا (دليلون_2.sigmoid || دليلون_2.softmax) {
        // SOFTMAX -> RESHAPE
        إذا (ڳنڍ_idx >= n_nodes || nodes[ڳنڍ_idx]->op != GGML_OP_RESHAPE ||
            nodes[ڳنڍ_idx]->src[0] != nodes[ڳنڍ_idx - 1]) {
            أرجع خطأ;
        }
        ggml_tensor * probs_reshaped = nodes[ڳنڍ_idx];
        ڳنڍ_idx++;

        إذا (ڳنڍ_idx >= n_nodes) {
            أرجع خطأ;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        إذا (nodes[ڳنڍ_idx]->op == GGML_OP_جوڙيو && nodes[ڳنڍ_idx]->src[0] == nodes[ڳنڍ_idx - 2]) {
            دليلون_2.prob_bias = سچ;
            ڳنڍ_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        إذا (ڳنڍ_idx >= n_nodes || nodes[ڳنڍ_idx]->op != GGML_OP_ARGSORT) {
            أرجع خطأ;
        }

        إذا (دليلون_2.prob_bias && nodes[ڳنڍ_idx]->src[0] != nodes[ڳنڍ_idx - 1]) {
            أرجع خطأ;
        } وإلا إذا (!دليلون_2.prob_bias && nodes[ڳنڍ_idx]->src[0] != nodes[ڳنڍ_idx - 2]) {
            أرجع خطأ;
        }

        ڳنڍ_idx++;

        // ARGSORT -> VIEW
        إذا (ڳنڍ_idx >= n_nodes || nodes[ڳنڍ_idx]->op != GGML_OP_VIEW ||
            nodes[ڳنڍ_idx]->src[0] != nodes[ڳنڍ_idx - 1]) {
            أرجع خطأ;
        }
        ڳنڍ_idx++;

        إذا (ڳنڍ_idx >= n_nodes || nodes[ڳنڍ_idx]->op != GGML_OP_GET_صفون) {
            أرجع خطأ;
        }

        // GET_ROWS
        إذا (nodes[ڳنڍ_idx]->src[0] != probs_reshaped || nodes[ڳنڍ_idx]->src[1] != nodes[ڳنڍ_idx - 1]) {
            أرجع خطأ;
        }
        ڳنڍ_idx++;
    } وإلا إذا (دليلون_2.delayed_softmax) {
        إذا (ڳنڍ_idx - 2 < 0) {
            أرجع خطأ;
        }
        ggml_tensor * probs_reshaped = nodes[ڳنڍ_idx - 2];

        // VIEW -> ARGSORT
        إذا (ڳنڍ_idx >= n_nodes || nodes[ڳنڍ_idx]->op != GGML_OP_VIEW ||
            nodes[ڳنڍ_idx]->src[0] != nodes[ڳنڍ_idx - 1]) {
            أرجع خطأ;
        }
        ڳنڍ_idx++;

        // GET_ROWS
        إذا (ڳنڍ_idx >= n_nodes || nodes[ڳنڍ_idx]->src[1] != nodes[ڳنڍ_idx - 1] ||
            nodes[ڳنڍ_idx]->src[0] != probs_reshaped) {
            أرجع خطأ;
        }
        ڳنڍ_idx++;

        ساكن ثابت std::vector<ggml_op> بچيل_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        لكل (ثابت ggml_op op : بچيل_ops) {
            إذا (ڳنڍ_idx >= n_nodes || nodes[ڳنڍ_idx]->op != op || nodes[ڳنڍ_idx]->src[0] != nodes[ڳنڍ_idx - 1]) {
                أرجع خطأ;
            }
            ڳنڍ_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    إذا (ڳنڍ_idx >= n_nodes) {
        أرجع سچ;
    }

    إذا (nodes[ڳنڍ_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        ساكن ثابت std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_جوڙ_صفون, GGML_OP_CLAMP };

        دليلون_2.norm = سچ;
        لكل (ثابت ggml_op op : norm_ops) {
            إذا (nodes[ڳنڍ_idx]->op == op && nodes[ڳنڍ_idx]->src[0] == nodes[ڳنڍ_idx - 1]) {
                ڳنڍ_idx++;
            } وإلا {
                دليلون_2.norm = خطأ;
                أرجع سچ;
            }
        }

        // DIV <- CLAMP, RESHAPE
        إذا (nodes[ڳنڍ_idx]->op != GGML_OP_DIV || nodes[ڳنڍ_idx]->src[1] != nodes[ڳنڍ_idx - 1] ||
            nodes[ڳنڍ_idx]->src[0] != nodes[ڳنڍ_idx - 3]) {
            دليلون_2.norm = خطأ;
            أرجع سچ;
        }
        ڳنڍ_idx++;

        إذا (nodes[ڳنڍ_idx]->op != GGML_OP_RESHAPE || nodes[ڳنڍ_idx]->src[0] != nodes[ڳنڍ_idx - 1]) {
            دليلون_2.norm = خطأ;
            أرجع سچ;
        }
        ڳنڍ_idx++;
    }

    إذا (nodes[ڳنڍ_idx]->op == GGML_OP_SCALE && nodes[ڳنڍ_idx]->src[0] == nodes[ڳنڍ_idx - 1]) {
        دليلون_2.scale = سچ;
    }

    أرجع سچ;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
ساكن منطقي ggml_sycl_check_fusion_يادگيري_ranges(ثابت ggml_cgraph * cgraph, ثابت صحيح ڳنڍ_idx,
                                                 ثابت صحيح ڳنڍ_ڳڻپ, ثابت صحيح * ٻاهريون_nodes, ثابت صحيح ٻاهريون_ڳڻپ,
                                                 ثابت منطقي is_topk_moe = خطأ) {
    خودڪار nodes_overlap = [&](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        ثابت int64_t a_شروعات = (int64_t) a->معلومات;
        ثابت int64_t a_پڇاڙي   = a_شروعات + ggml_backend_buft_get_alloc_سائيز(a->بفر->buft, a);

        ثابت int64_t b_شروعات = (int64_t) b->معلومات;
        ثابت int64_t b_پڇاڙي   = b_شروعات + ggml_backend_buft_get_alloc_سائيز(b->بفر->buft, b);

        إذا ((b_شروعات <= a_شروعات && a_شروعات < b_پڇاڙي) || (a_شروعات <= b_شروعات && b_شروعات < a_پڇاڙي)) {
            أرجع سچ;
        }

        أرجع خطأ;
    };

    منطقي is_ok = سچ;
    // exception for topk-moe, as each row is read entirely before writing
    إذا (ggml_nrows(cgraph->nodes[ڳنڍ_idx]) == 1 && is_topk_moe) {
        أرجع سچ;
    }

    لكل (صحيح i = 0; i < ٻاهريون_ڳڻپ; ++i) {
        ثابت ggml_tensor * dst = cgraph->nodes[ٻاهريون_nodes[i]];

        لكل (صحيح j = ڳنڍ_idx; j < ڳنڍ_idx + ڳنڍ_ڳڻپ; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            لكل (صحيح src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ثابت ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                إذا (!src || src->op == GGML_OP_NONE) {
                    استمر;
                }

                إذا (nodes_overlap(dst, src)) {
                    منطقي مليو = خطأ;

                    لكل (صحيح k = ڳنڍ_idx; k < j; ++k) {
                        إذا (cgraph->nodes[k] == src) {
                            مليو = سچ;
                            اكسر;
                        }
                    }

                    إذا (!مليو) {
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
    ggml_tensor * ڳنڍ = cgraph->nodes[i];

    إذا (ڳنڍ->op != GGML_OP_UNARY && ڳنڍ->op != GGML_OP_SOFT_MAX && ڳنڍ->op != GGML_OP_ARGSORT) {
        أرجع 0;
    }

    ggml_sycl_topk_moe_دليلون دليلون_2;
    إذا (!ggml_sycl_topk_moe_fusion(cgraph, i, دليلون_2)) {
        أرجع 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    إذا (دليلون_2.prob_bias) {
        أرجع 0;
    }

    ثابت ggml_tensor * logits  = ڳنڍ->src[0];
    ggml_tensor *       weights = خالي_اشارو;
    ggml_tensor *       ids     = خالي_اشارو;
    ثابت ggml_tensor * clamp   = خالي_اشارو;
    ثابت ggml_tensor * scale   = خالي_اشارو;

    std::vector<ggml_op> ops;
    صحيح                  ٻاهريون_nodes[2];

    إذا (!دليلون_2.delayed_softmax) {
        ثابت ggml_op gating_op = دليلون_2.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_صفون });
        ٻاهريون_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        إذا (دليلون_2.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_جوڙ_صفون, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (صحيح) ops.size() - 3];
        }
        إذا (دليلون_2.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (صحيح) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (صحيح) ops.size() - 1];
        ٻاهريون_nodes[1] = i + (صحيح) ops.size() - 1;

        إذا (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.معلومات(), ٻاهريون_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(ڳنڍ, weights, logits, ids) &&
            ggml_sycl_check_fusion_يادگيري_ranges(cgraph, i, (صحيح) ops.size(), ٻاهريون_nodes, 2, /*is_topk_moe=*/سچ)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, دليلون_2);
            أرجع (صحيح) ops.size() - 1;
        }
    } وإلا إذا (!دليلون_2.norm && !دليلون_2.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_صفون, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ثابت ggml_tensor * softmax = cgraph->nodes[i + 4];
        ٻاهريون_nodes[0]                = i + 1;
        ٻاهريون_nodes[1]                = i + 5;

        إذا (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.معلومات(), ٻاهريون_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_يادگيري_ranges(cgraph, i, (صحيح) ops.size(), ٻاهريون_nodes, 2, /*is_topk_moe=*/سچ)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, دليلون_2);
            أرجع (صحيح) ops.size() - 1;
        }
    }

    أرجع 0;
}
