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

بنية ggml_sycl_topk_moe_دلیلاں {
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
سانچا <صحيح experts_فی_thread, منطقي use_حد_بندی>
ساكن مضمن فراغ softmax_warp_inplace(عائم (&vals)[experts_فی_thread], ثابت صحيح حد_بندی, ثابت صحيح lane) {
    عائم max_val = -INFINITY;
#ہدایت unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_ناپ;
        ثابت منطقي سرگرم = !use_حد_بندی || (idx < حد_بندی);
        إذا (سرگرم) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_ناپ>(max_val);

    عائم جوڑ = 0.f;
#ہدایت unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_ناپ;
        ثابت منطقي سرگرم = !use_حد_بندی || (idx < حد_بندی);
        إذا (سرگرم) {
            ثابت عائم val = sycl::قوت_نما(vals[i] - max_val);
            vals[i]         = val;
            جوڑ += val;
        } وإلا {
            vals[i] = 0.f;
        }
    }
    جوڑ = warp_reduce_جوڑ<WARP_ناپ>(جوڑ);

    ثابت عائم inv_جوڑ = 1.0f / جوڑ;
#ہدایت unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        ثابت صحيح idx = lane + i * WARP_ناپ;
        إذا (!use_حد_بندی || idx < حد_بندی) {
            vals[i] *= inv_جوڑ;
        }
    }
}

سانچا <صحيح experts_فی_thread, منطقي use_حد_بندی>
ساكن مضمن فراغ sigmoid_warp_inplace(عائم (&vals)[experts_فی_thread], ثابت صحيح حد_بندی, ثابت صحيح lane) {
#ہدایت unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_ناپ;
        ثابت منطقي سرگرم = !use_حد_بندی || (idx < حد_بندی);
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
سانچا <صحيح n_experts>
ساكن فراغ topk_moe_kernel(ثابت عائم * __restrict__ logits,
                            عائم * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ثابت صحيح                  n_صفاں,
                            ثابت صحيح                  n_expert_ورتیا,
                            ثابت عائم                clamp_val,
                            ثابت عائم                scale_val,
                            ثابت topk_moe_config       config) {
    خودکار      شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<1>();
    ثابت صحيح صف       = شے_ct1.get_گروہ(0);
    إذا (صف >= n_صفاں) {
        أرجع;
    }
    ثابت صحيح lane = شے_ct1.get_local_id(0);

    logits  += (size_t) n_experts * صف;
    weights += (size_t) n_expert_ورتیا * صف;
    ids     += (size_t) n_experts * صف;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr صحيح experts_فی_thread = (n_experts > WARP_ناپ) ? n_experts / WARP_ناپ : 1;

    عائم wt[experts_فی_thread];
#ہدایت unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        wt[i] = -INFINITY;
    }
#ہدایت unroll
    لكل (صحيح i = 0; i < n_experts; i += WARP_ناپ) {
        ثابت صحيح expert  = i + lane;
        wt[i / WARP_ناپ] = (n_experts % WARP_ناپ == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    إذا (!config.delayed_softmax) {
        إذا (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_فی_thread, خطأ>(wt, n_experts, lane);
        } وإلا {
            softmax_warp_inplace<experts_فی_thread, خطأ>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#ہدایت unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        إذا (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    عائم wt_جوڑ = 0.f;
    عائم output_weights[experts_فی_thread];
#ہدایت unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        output_weights[i] = 0.f;
    }

    ثابت sycl::sub_گروہ sg = شے_ct1.get_sub_گروہ();

    لكل (صحيح k = 0; k < n_expert_ورتیا; k++) {
        عائم max_val    = wt[0];
        صحيح   max_expert = lane;
#ہدایت unroll
        لكل (صحيح i = 1; i < experts_فی_thread; i++) {
            ثابت صحيح expert = lane + i * WARP_ناپ;
            إذا ((n_experts % WARP_ناپ == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#ہدایت unroll
        لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
            ثابت عائم val    = dpct::ترتیب_بدلو_sub_گروہ_by_xor(sg, max_val, نقاب);
            ثابت صحيح   expert = dpct::ترتیب_بدلو_sub_گروہ_by_xor(sg, max_expert, نقاب);
            إذا (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        إذا ((max_expert & (WARP_ناپ - 1)) == lane) {
            wt[max_expert / WARP_ناپ] = -INFINITY;
        }
        إذا ((k & (WARP_ناپ - 1)) == lane) {
            output_weights[k / WARP_ناپ] = max_val;
        }
        إذا ((max_expert & (WARP_ناپ - 1)) == lane) {
            ids[k] = max_expert;
            إذا (config.with_norm) {
                wt_جوڑ += max_val;
            }
        }
    }

    إذا (config.with_norm) {
        wt_جوڑ          = warp_reduce_جوڑ<WARP_ناپ>(wt_جوڑ);
        wt_جوڑ          = sycl::fmax(wt_جوڑ, clamp_val);
        ثابت عائم inv = 1.0f / wt_جوڑ;
#ہدایت unroll
        لكل (صحيح i = 0; i < experts_فی_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    إذا (config.delayed_softmax) {
        softmax_warp_inplace<experts_فی_thread, سچ>(output_weights, n_expert_ورتیا, lane);
    }

#ہدایت unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        ثابت صحيح idx = i * WARP_ناپ + lane;
        إذا (idx < n_expert_ورتیا) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

سانچا <صحيح n_experts>
ساكن فراغ launch_topk_moe(قطار_ptr stream, ثابت عائم * logits, عائم * weights, int32_t * ids, صحيح n_صفاں,
                            صحيح n_expert_ورتیا, عائم clamp_val, عائم scale_val, ثابت topk_moe_config & config) {
    ثابت sycl::range<1> بلاک_dims(WARP_ناپ);
    ثابت sycl::range<1> بلاک_nums(n_صفاں);
    stream->parallel_for(sycl::nd_range<1>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<1> شے_ct1) [[sycl::reqd_sub_گروہ_ناپ(WARP_ناپ)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_صفاں, n_expert_ورتیا, clamp_val,
                                                        scale_val, config);
                             GGML_بن_ورتیا(شے_ct1);
                         });
}

ساكن فراغ ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ثابت ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ثابت ggml_tensor *             clamp,
                                  ثابت ggml_tensor *             scale,
                                  ثابت ggml_sycl_topk_moe_دلیلاں & دلیلاں_2) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ثابت صحيح n_experts     = logits->ne[0];
    ثابت صحيح n_صفاں        = logits->ne[1];
    ثابت صحيح n_expert_ورتیا = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_ناپ(ids->type) == (size_t) n_experts);

    ثابت عائم * logits_d  = (ثابت عائم *) logits->انکڑے;
    عائم *       weights_d = (عائم *) weights->انکڑے;
    int32_t *     ids_d     = (int32_t *) ids->انکڑے;

    ثابت منطقي  with_norm = clamp != خالی_سوچک;
    ثابت عائم clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ثابت عائم scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = دلیلاں_2.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = دلیلاں_2.delayed_softmax;

    قطار_ptr stream = ctx.stream();
    ggml_sycl_سیٹ_device(ctx.device);

    بدل (n_experts) {
        حالة 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_صفاں, n_expert_ورتیا, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_صفاں, n_expert_ورتیا, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_صفاں, n_expert_ورتیا, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_صفاں, n_expert_ورتیا, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_صفاں, n_expert_ورتیا, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_صفاں, n_expert_ورتیا, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_صفاں, n_expert_ورتیا, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_صفاں, n_expert_ورتیا, clamp_val, scale_val,
                                 config);
            اكسر;
        حالة 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_صفاں, n_expert_ورتیا, clamp_val, scale_val,
                                 config);
            اكسر;
        حالة 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_صفاں, n_expert_ورتیا, clamp_val, scale_val,
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

        حافظہ_نقل(&scale, (ثابت عائم *) gating_op->op_params + 0, حجم(عائم));
        حافظہ_نقل(&max_bias, (ثابت عائم *) gating_op->op_params + 1, حجم(عائم));

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
ساكن منطقي ggml_sycl_topk_moe_fusion(ثابت ggml_cgraph * cgraph, صحيح گرہ_idx, ggml_sycl_topk_moe_دلیلاں & دلیلاں_2) {
    دلیلاں_2 = ggml_sycl_topk_moe_دلیلاں{};

    ثابت صحيح      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    إذا (nodes[گرہ_idx]->op == GGML_OP_SOFT_MAX) {
        دلیلاں_2.softmax = سچ;
    }

    إذا (nodes[گرہ_idx]->op == GGML_OP_UNARY) {
        إذا (ggml_get_unary_op(nodes[گرہ_idx]) != GGML_UNARY_OP_SIGMOID) {
            أرجع خطأ;
        }
        دلیلاں_2.sigmoid = سچ;
    }

    إذا (nodes[گرہ_idx]->op == GGML_OP_ARGSORT) {
        دلیلاں_2.delayed_softmax = سچ;
    }

    گرہ_idx++;

    إذا (دلیلاں_2.sigmoid || دلیلاں_2.softmax) {
        // SOFTMAX -> RESHAPE
        إذا (گرہ_idx >= n_nodes || nodes[گرہ_idx]->op != GGML_OP_RESHAPE ||
            nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 1]) {
            أرجع خطأ;
        }
        ggml_tensor * probs_reshaped = nodes[گرہ_idx];
        گرہ_idx++;

        إذا (گرہ_idx >= n_nodes) {
            أرجع خطأ;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        إذا (nodes[گرہ_idx]->op == GGML_OP_جوڑو && nodes[گرہ_idx]->src[0] == nodes[گرہ_idx - 2]) {
            دلیلاں_2.prob_bias = سچ;
            گرہ_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        إذا (گرہ_idx >= n_nodes || nodes[گرہ_idx]->op != GGML_OP_ARGSORT) {
            أرجع خطأ;
        }

        إذا (دلیلاں_2.prob_bias && nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 1]) {
            أرجع خطأ;
        } وإلا إذا (!دلیلاں_2.prob_bias && nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 2]) {
            أرجع خطأ;
        }

        گرہ_idx++;

        // ARGSORT -> VIEW
        إذا (گرہ_idx >= n_nodes || nodes[گرہ_idx]->op != GGML_OP_VIEW ||
            nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 1]) {
            أرجع خطأ;
        }
        گرہ_idx++;

        إذا (گرہ_idx >= n_nodes || nodes[گرہ_idx]->op != GGML_OP_GET_صفاں) {
            أرجع خطأ;
        }

        // GET_ROWS
        إذا (nodes[گرہ_idx]->src[0] != probs_reshaped || nodes[گرہ_idx]->src[1] != nodes[گرہ_idx - 1]) {
            أرجع خطأ;
        }
        گرہ_idx++;
    } وإلا إذا (دلیلاں_2.delayed_softmax) {
        إذا (گرہ_idx - 2 < 0) {
            أرجع خطأ;
        }
        ggml_tensor * probs_reshaped = nodes[گرہ_idx - 2];

        // VIEW -> ARGSORT
        إذا (گرہ_idx >= n_nodes || nodes[گرہ_idx]->op != GGML_OP_VIEW ||
            nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 1]) {
            أرجع خطأ;
        }
        گرہ_idx++;

        // GET_ROWS
        إذا (گرہ_idx >= n_nodes || nodes[گرہ_idx]->src[1] != nodes[گرہ_idx - 1] ||
            nodes[گرہ_idx]->src[0] != probs_reshaped) {
            أرجع خطأ;
        }
        گرہ_idx++;

        ساكن ثابت std::vector<ggml_op> بچیا_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        لكل (ثابت ggml_op op : بچیا_ops) {
            إذا (گرہ_idx >= n_nodes || nodes[گرہ_idx]->op != op || nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 1]) {
                أرجع خطأ;
            }
            گرہ_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    إذا (گرہ_idx >= n_nodes) {
        أرجع سچ;
    }

    إذا (nodes[گرہ_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        ساكن ثابت std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_جوڑ_صفاں, GGML_OP_CLAMP };

        دلیلاں_2.norm = سچ;
        لكل (ثابت ggml_op op : norm_ops) {
            إذا (nodes[گرہ_idx]->op == op && nodes[گرہ_idx]->src[0] == nodes[گرہ_idx - 1]) {
                گرہ_idx++;
            } وإلا {
                دلیلاں_2.norm = خطأ;
                أرجع سچ;
            }
        }

        // DIV <- CLAMP, RESHAPE
        إذا (nodes[گرہ_idx]->op != GGML_OP_DIV || nodes[گرہ_idx]->src[1] != nodes[گرہ_idx - 1] ||
            nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 3]) {
            دلیلاں_2.norm = خطأ;
            أرجع سچ;
        }
        گرہ_idx++;

        إذا (nodes[گرہ_idx]->op != GGML_OP_RESHAPE || nodes[گرہ_idx]->src[0] != nodes[گرہ_idx - 1]) {
            دلیلاں_2.norm = خطأ;
            أرجع سچ;
        }
        گرہ_idx++;
    }

    إذا (nodes[گرہ_idx]->op == GGML_OP_SCALE && nodes[گرہ_idx]->src[0] == nodes[گرہ_idx - 1]) {
        دلیلاں_2.scale = سچ;
    }

    أرجع سچ;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
ساكن منطقي ggml_sycl_check_fusion_حافظہ_ranges(ثابت ggml_cgraph * cgraph, ثابت صحيح گرہ_idx,
                                                 ثابت صحيح گرہ_گنتی, ثابت صحيح * باہرلا_nodes, ثابت صحيح باہرلا_گنتی,
                                                 ثابت منطقي is_topk_moe = خطأ) {
    خودکار nodes_overlap = [&](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        ثابت int64_t a_شروع = (int64_t) a->انکڑے;
        ثابت int64_t a_سرا   = a_شروع + ggml_backend_buft_get_alloc_ناپ(a->بفر->buft, a);

        ثابت int64_t b_شروع = (int64_t) b->انکڑے;
        ثابت int64_t b_سرا   = b_شروع + ggml_backend_buft_get_alloc_ناپ(b->بفر->buft, b);

        إذا ((b_شروع <= a_شروع && a_شروع < b_سرا) || (a_شروع <= b_شروع && b_شروع < a_سرا)) {
            أرجع سچ;
        }

        أرجع خطأ;
    };

    منطقي is_ok = سچ;
    // exception for topk-moe, as each row is read entirely before writing
    إذا (ggml_nrows(cgraph->nodes[گرہ_idx]) == 1 && is_topk_moe) {
        أرجع سچ;
    }

    لكل (صحيح i = 0; i < باہرلا_گنتی; ++i) {
        ثابت ggml_tensor * dst = cgraph->nodes[باہرلا_nodes[i]];

        لكل (صحيح j = گرہ_idx; j < گرہ_idx + گرہ_گنتی; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            لكل (صحيح src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ثابت ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                إذا (!src || src->op == GGML_OP_NONE) {
                    استمر;
                }

                إذا (nodes_overlap(dst, src)) {
                    منطقي لبھیا = خطأ;

                    لكل (صحيح k = گرہ_idx; k < j; ++k) {
                        إذا (cgraph->nodes[k] == src) {
                            لبھیا = سچ;
                            اكسر;
                        }
                    }

                    إذا (!لبھیا) {
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
    ggml_tensor * گرہ = cgraph->nodes[i];

    إذا (گرہ->op != GGML_OP_UNARY && گرہ->op != GGML_OP_SOFT_MAX && گرہ->op != GGML_OP_ARGSORT) {
        أرجع 0;
    }

    ggml_sycl_topk_moe_دلیلاں دلیلاں_2;
    إذا (!ggml_sycl_topk_moe_fusion(cgraph, i, دلیلاں_2)) {
        أرجع 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    إذا (دلیلاں_2.prob_bias) {
        أرجع 0;
    }

    ثابت ggml_tensor * logits  = گرہ->src[0];
    ggml_tensor *       weights = خالی_سوچک;
    ggml_tensor *       ids     = خالی_سوچک;
    ثابت ggml_tensor * clamp   = خالی_سوچک;
    ثابت ggml_tensor * scale   = خالی_سوچک;

    std::vector<ggml_op> ops;
    صحيح                  باہرلا_nodes[2];

    إذا (!دلیلاں_2.delayed_softmax) {
        ثابت ggml_op gating_op = دلیلاں_2.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_صفاں });
        باہرلا_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        إذا (دلیلاں_2.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_جوڑ_صفاں, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (صحيح) ops.size() - 3];
        }
        إذا (دلیلاں_2.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (صحيح) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (صحيح) ops.size() - 1];
        باہرلا_nodes[1] = i + (صحيح) ops.size() - 1;

        إذا (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.انکڑے(), باہرلا_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(گرہ, weights, logits, ids) &&
            ggml_sycl_check_fusion_حافظہ_ranges(cgraph, i, (صحيح) ops.size(), باہرلا_nodes, 2, /*is_topk_moe=*/سچ)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, دلیلاں_2);
            أرجع (صحيح) ops.size() - 1;
        }
    } وإلا إذا (!دلیلاں_2.norm && !دلیلاں_2.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_صفاں, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ثابت ggml_tensor * softmax = cgraph->nodes[i + 4];
        باہرلا_nodes[0]                = i + 1;
        باہرلا_nodes[1]                = i + 5;

        إذا (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.انکڑے(), باہرلا_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_حافظہ_ranges(cgraph, i, (صحيح) ops.size(), باہرلا_nodes, 2, /*is_topk_moe=*/سچ)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, دلیلاں_2);
            أرجع (صحيح) ops.size() - 1;
        }
    }

    أرجع 0;
}
