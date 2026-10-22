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

بنية ggml_sycl_topk_moe_دلیلونه {
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
کينډۍ <صحيح experts_فی_thread, منطقي use_حد>
ساكن مضمن فراغ softmax_warp_inplace(عائم (&vals)[experts_فی_thread], ثابت صحيح حد, ثابت صحيح lane) {
    عائم max_val = -INFINITY;
#لارښوونه unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_اندازه;
        ثابت منطقي فعال = !use_حد || (idx < حد);
        إذا (فعال) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_اندازه>(max_val);

    عائم مجموعه = 0.f;
#لارښوونه unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_اندازه;
        ثابت منطقي فعال = !use_حد || (idx < حد);
        إذا (فعال) {
            ثابت عائم val = sycl::توانیز_تابع(vals[i] - max_val);
            vals[i]         = val;
            مجموعه += val;
        } وإلا {
            vals[i] = 0.f;
        }
    }
    مجموعه = warp_reduce_مجموعه<WARP_اندازه>(مجموعه);

    ثابت عائم inv_مجموعه = 1.0f / مجموعه;
#لارښوونه unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        ثابت صحيح idx = lane + i * WARP_اندازه;
        إذا (!use_حد || idx < حد) {
            vals[i] *= inv_مجموعه;
        }
    }
}

کينډۍ <صحيح experts_فی_thread, منطقي use_حد>
ساكن مضمن فراغ sigmoid_warp_inplace(عائم (&vals)[experts_فی_thread], ثابت صحيح حد, ثابت صحيح lane) {
#لارښوونه unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_اندازه;
        ثابت منطقي فعال = !use_حد || (idx < حد);
        vals[i]           = فعال ? 1.f / (1.f + sycl::توانیز_تابع(-vals[i])) : -INFINITY;
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
کينډۍ <صحيح n_experts>
ساكن فراغ topk_moe_kernel(ثابت عائم * __restrict__ logits,
                            عائم * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ثابت صحيح                  n_لیکې,
                            ثابت صحيح                  n_expert_کارول_شوی,
                            ثابت عائم                clamp_val,
                            ثابت عائم                scale_val,
                            ثابت topk_moe_config       config) {
    خپلکاری      توکی_ct1 = sycl::ext::oneapi::this_work_توکی::get_nd_توکی<1>();
    ثابت صحيح لیکه       = توکی_ct1.get_ډله(0);
    إذا (لیکه >= n_لیکې) {
        أرجع;
    }
    ثابت صحيح lane = توکی_ct1.get_local_id(0);

    logits  += (size_t) n_experts * لیکه;
    weights += (size_t) n_expert_کارول_شوی * لیکه;
    ids     += (size_t) n_experts * لیکه;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr صحيح experts_فی_thread = (n_experts > WARP_اندازه) ? n_experts / WARP_اندازه : 1;

    عائم wt[experts_فی_thread];
#لارښوونه unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        wt[i] = -INFINITY;
    }
#لارښوونه unroll
    لكل (صحيح i = 0; i < n_experts; i += WARP_اندازه) {
        ثابت صحيح expert  = i + lane;
        wt[i / WARP_اندازه] = (n_experts % WARP_اندازه == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
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
#لارښوونه unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        إذا (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    عائم wt_مجموعه = 0.f;
    عائم output_weights[experts_فی_thread];
#لارښوونه unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        output_weights[i] = 0.f;
    }

    ثابت sycl::sub_ډله sg = توکی_ct1.get_sub_ډله();

    لكل (صحيح k = 0; k < n_expert_کارول_شوی; k++) {
        عائم max_val    = wt[0];
        صحيح   max_expert = lane;
#لارښوونه unroll
        لكل (صحيح i = 1; i < experts_فی_thread; i++) {
            ثابت صحيح expert = lane + i * WARP_اندازه;
            إذا ((n_experts % WARP_اندازه == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#لارښوونه unroll
        لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
            ثابت عائم val    = dpct::ترتیب_بدلول_sub_ډله_by_xor(sg, max_val, ماسک);
            ثابت صحيح   expert = dpct::ترتیب_بدلول_sub_ډله_by_xor(sg, max_expert, ماسک);
            إذا (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        إذا ((max_expert & (WARP_اندازه - 1)) == lane) {
            wt[max_expert / WARP_اندازه] = -INFINITY;
        }
        إذا ((k & (WARP_اندازه - 1)) == lane) {
            output_weights[k / WARP_اندازه] = max_val;
        }
        إذا ((max_expert & (WARP_اندازه - 1)) == lane) {
            ids[k] = max_expert;
            إذا (config.with_norm) {
                wt_مجموعه += max_val;
            }
        }
    }

    إذا (config.with_norm) {
        wt_مجموعه          = warp_reduce_مجموعه<WARP_اندازه>(wt_مجموعه);
        wt_مجموعه          = sycl::fmax(wt_مجموعه, clamp_val);
        ثابت عائم inv = 1.0f / wt_مجموعه;
#لارښوونه unroll
        لكل (صحيح i = 0; i < experts_فی_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    إذا (config.delayed_softmax) {
        softmax_warp_inplace<experts_فی_thread, ریښتیا>(output_weights, n_expert_کارول_شوی, lane);
    }

#لارښوونه unroll
    لكل (صحيح i = 0; i < experts_فی_thread; i++) {
        ثابت صحيح idx = i * WARP_اندازه + lane;
        إذا (idx < n_expert_کارول_شوی) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

کينډۍ <صحيح n_experts>
ساكن فراغ launch_topk_moe(کتار_ptr stream, ثابت عائم * logits, عائم * weights, int32_t * ids, صحيح n_لیکې,
                            صحيح n_expert_کارول_شوی, عائم clamp_val, عائم scale_val, ثابت topk_moe_config & config) {
    ثابت sycl::range<1> بلاک_dims(WARP_اندازه);
    ثابت sycl::range<1> بلاک_nums(n_لیکې);
    stream->parallel_for(sycl::nd_range<1>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_توکی<1> توکی_ct1) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_لیکې, n_expert_کارول_شوی, clamp_val,
                                                        scale_val, config);
                             GGML_ناکارول_شوی(توکی_ct1);
                         });
}

ساكن فراغ ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ثابت ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ثابت ggml_tensor *             clamp,
                                  ثابت ggml_tensor *             scale,
                                  ثابت ggml_sycl_topk_moe_دلیلونه & دلیلونه_2) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ثابت صحيح n_experts     = logits->ne[0];
    ثابت صحيح n_لیکې        = logits->ne[1];
    ثابت صحيح n_expert_کارول_شوی = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_اندازه(ids->type) == (size_t) n_experts);

    ثابت عائم * logits_d  = (ثابت عائم *) logits->معلومات;
    عائم *       weights_d = (عائم *) weights->معلومات;
    int32_t *     ids_d     = (int32_t *) ids->معلومات;

    ثابت منطقي  with_norm = clamp != تش_اشاره;
    ثابت عائم clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ثابت عائم scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = دلیلونه_2.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = دلیلونه_2.delayed_softmax;

    کتار_ptr stream = ctx.stream();
    ggml_sycl_سېټ_device(ctx.device);

    بدل (n_experts) {
        حالة 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_لیکې, n_expert_کارول_شوی, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_لیکې, n_expert_کارول_شوی, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_لیکې, n_expert_کارول_شوی, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_لیکې, n_expert_کارول_شوی, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_لیکې, n_expert_کارول_شوی, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_لیکې, n_expert_کارول_شوی, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_لیکې, n_expert_کارول_شوی, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_لیکې, n_expert_کارول_شوی, clamp_val, scale_val,
                                 config);
            اكسر;
        حالة 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_لیکې, n_expert_کارول_شوی, clamp_val, scale_val,
                                 config);
            اكسر;
        حالة 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_لیکې, n_expert_کارول_شوی, clamp_val, scale_val,
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

        حافظه_نقل(&scale, (ثابت عائم *) gating_op->op_params + 0, حجم(عائم));
        حافظه_نقل(&max_bias, (ثابت عائم *) gating_op->op_params + 1, حجم(عائم));

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

    أرجع ریښتیا;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
ساكن منطقي ggml_sycl_topk_moe_fusion(ثابت ggml_cgraph * cgraph, صحيح غوټه_idx, ggml_sycl_topk_moe_دلیلونه & دلیلونه_2) {
    دلیلونه_2 = ggml_sycl_topk_moe_دلیلونه{};

    ثابت صحيح      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    إذا (nodes[غوټه_idx]->op == GGML_OP_SOFT_MAX) {
        دلیلونه_2.softmax = ریښتیا;
    }

    إذا (nodes[غوټه_idx]->op == GGML_OP_UNARY) {
        إذا (ggml_get_unary_op(nodes[غوټه_idx]) != GGML_UNARY_OP_SIGMOID) {
            أرجع خطأ;
        }
        دلیلونه_2.sigmoid = ریښتیا;
    }

    إذا (nodes[غوټه_idx]->op == GGML_OP_ARGSORT) {
        دلیلونه_2.delayed_softmax = ریښتیا;
    }

    غوټه_idx++;

    إذا (دلیلونه_2.sigmoid || دلیلونه_2.softmax) {
        // SOFTMAX -> RESHAPE
        إذا (غوټه_idx >= n_nodes || nodes[غوټه_idx]->op != GGML_OP_RESHAPE ||
            nodes[غوټه_idx]->src[0] != nodes[غوټه_idx - 1]) {
            أرجع خطأ;
        }
        ggml_tensor * probs_reshaped = nodes[غوټه_idx];
        غوټه_idx++;

        إذا (غوټه_idx >= n_nodes) {
            أرجع خطأ;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        إذا (nodes[غوټه_idx]->op == GGML_OP_جمع_کړه && nodes[غوټه_idx]->src[0] == nodes[غوټه_idx - 2]) {
            دلیلونه_2.prob_bias = ریښتیا;
            غوټه_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        إذا (غوټه_idx >= n_nodes || nodes[غوټه_idx]->op != GGML_OP_ARGSORT) {
            أرجع خطأ;
        }

        إذا (دلیلونه_2.prob_bias && nodes[غوټه_idx]->src[0] != nodes[غوټه_idx - 1]) {
            أرجع خطأ;
        } وإلا إذا (!دلیلونه_2.prob_bias && nodes[غوټه_idx]->src[0] != nodes[غوټه_idx - 2]) {
            أرجع خطأ;
        }

        غوټه_idx++;

        // ARGSORT -> VIEW
        إذا (غوټه_idx >= n_nodes || nodes[غوټه_idx]->op != GGML_OP_VIEW ||
            nodes[غوټه_idx]->src[0] != nodes[غوټه_idx - 1]) {
            أرجع خطأ;
        }
        غوټه_idx++;

        إذا (غوټه_idx >= n_nodes || nodes[غوټه_idx]->op != GGML_OP_GET_لیکې) {
            أرجع خطأ;
        }

        // GET_ROWS
        إذا (nodes[غوټه_idx]->src[0] != probs_reshaped || nodes[غوټه_idx]->src[1] != nodes[غوټه_idx - 1]) {
            أرجع خطأ;
        }
        غوټه_idx++;
    } وإلا إذا (دلیلونه_2.delayed_softmax) {
        إذا (غوټه_idx - 2 < 0) {
            أرجع خطأ;
        }
        ggml_tensor * probs_reshaped = nodes[غوټه_idx - 2];

        // VIEW -> ARGSORT
        إذا (غوټه_idx >= n_nodes || nodes[غوټه_idx]->op != GGML_OP_VIEW ||
            nodes[غوټه_idx]->src[0] != nodes[غوټه_idx - 1]) {
            أرجع خطأ;
        }
        غوټه_idx++;

        // GET_ROWS
        إذا (غوټه_idx >= n_nodes || nodes[غوټه_idx]->src[1] != nodes[غوټه_idx - 1] ||
            nodes[غوټه_idx]->src[0] != probs_reshaped) {
            أرجع خطأ;
        }
        غوټه_idx++;

        ساكن ثابت std::vector<ggml_op> پاتې_شوی_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        لكل (ثابت ggml_op op : پاتې_شوی_ops) {
            إذا (غوټه_idx >= n_nodes || nodes[غوټه_idx]->op != op || nodes[غوټه_idx]->src[0] != nodes[غوټه_idx - 1]) {
                أرجع خطأ;
            }
            غوټه_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    إذا (غوټه_idx >= n_nodes) {
        أرجع ریښتیا;
    }

    إذا (nodes[غوټه_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        ساكن ثابت std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_مجموعه_لیکې, GGML_OP_CLAMP };

        دلیلونه_2.norm = ریښتیا;
        لكل (ثابت ggml_op op : norm_ops) {
            إذا (nodes[غوټه_idx]->op == op && nodes[غوټه_idx]->src[0] == nodes[غوټه_idx - 1]) {
                غوټه_idx++;
            } وإلا {
                دلیلونه_2.norm = خطأ;
                أرجع ریښتیا;
            }
        }

        // DIV <- CLAMP, RESHAPE
        إذا (nodes[غوټه_idx]->op != GGML_OP_DIV || nodes[غوټه_idx]->src[1] != nodes[غوټه_idx - 1] ||
            nodes[غوټه_idx]->src[0] != nodes[غوټه_idx - 3]) {
            دلیلونه_2.norm = خطأ;
            أرجع ریښتیا;
        }
        غوټه_idx++;

        إذا (nodes[غوټه_idx]->op != GGML_OP_RESHAPE || nodes[غوټه_idx]->src[0] != nodes[غوټه_idx - 1]) {
            دلیلونه_2.norm = خطأ;
            أرجع ریښتیا;
        }
        غوټه_idx++;
    }

    إذا (nodes[غوټه_idx]->op == GGML_OP_SCALE && nodes[غوټه_idx]->src[0] == nodes[غوټه_idx - 1]) {
        دلیلونه_2.scale = ریښتیا;
    }

    أرجع ریښتیا;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
ساكن منطقي ggml_sycl_check_fusion_حافظه_ranges(ثابت ggml_cgraph * cgraph, ثابت صحيح غوټه_idx,
                                                 ثابت صحيح غوټه_شمېر, ثابت صحيح * بهر_nodes, ثابت صحيح بهر_شمېر,
                                                 ثابت منطقي is_topk_moe = خطأ) {
    خپلکاری nodes_overlap = [&](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        ثابت int64_t a_پیل = (int64_t) a->معلومات;
        ثابت int64_t a_پای_ټکی   = a_پیل + ggml_backend_buft_get_alloc_اندازه(a->بفر->buft, a);

        ثابت int64_t b_پیل = (int64_t) b->معلومات;
        ثابت int64_t b_پای_ټکی   = b_پیل + ggml_backend_buft_get_alloc_اندازه(b->بفر->buft, b);

        إذا ((b_پیل <= a_پیل && a_پیل < b_پای_ټکی) || (a_پیل <= b_پیل && b_پیل < a_پای_ټکی)) {
            أرجع ریښتیا;
        }

        أرجع خطأ;
    };

    منطقي is_ok = ریښتیا;
    // exception for topk-moe, as each row is read entirely before writing
    إذا (ggml_nrows(cgraph->nodes[غوټه_idx]) == 1 && is_topk_moe) {
        أرجع ریښتیا;
    }

    لكل (صحيح i = 0; i < بهر_شمېر; ++i) {
        ثابت ggml_tensor * dst = cgraph->nodes[بهر_nodes[i]];

        لكل (صحيح j = غوټه_idx; j < غوټه_idx + غوټه_شمېر; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            لكل (صحيح src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ثابت ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                إذا (!src || src->op == GGML_OP_NONE) {
                    استمر;
                }

                إذا (nodes_overlap(dst, src)) {
                    منطقي وموندل = خطأ;

                    لكل (صحيح k = غوټه_idx; k < j; ++k) {
                        إذا (cgraph->nodes[k] == src) {
                            وموندل = ریښتیا;
                            اكسر;
                        }
                    }

                    إذا (!وموندل) {
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
    ggml_tensor * غوټه = cgraph->nodes[i];

    إذا (غوټه->op != GGML_OP_UNARY && غوټه->op != GGML_OP_SOFT_MAX && غوټه->op != GGML_OP_ARGSORT) {
        أرجع 0;
    }

    ggml_sycl_topk_moe_دلیلونه دلیلونه_2;
    إذا (!ggml_sycl_topk_moe_fusion(cgraph, i, دلیلونه_2)) {
        أرجع 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    إذا (دلیلونه_2.prob_bias) {
        أرجع 0;
    }

    ثابت ggml_tensor * logits  = غوټه->src[0];
    ggml_tensor *       weights = تش_اشاره;
    ggml_tensor *       ids     = تش_اشاره;
    ثابت ggml_tensor * clamp   = تش_اشاره;
    ثابت ggml_tensor * scale   = تش_اشاره;

    std::vector<ggml_op> ops;
    صحيح                  بهر_nodes[2];

    إذا (!دلیلونه_2.delayed_softmax) {
        ثابت ggml_op gating_op = دلیلونه_2.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_لیکې });
        بهر_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        إذا (دلیلونه_2.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_مجموعه_لیکې, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (صحيح) ops.size() - 3];
        }
        إذا (دلیلونه_2.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (صحيح) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (صحيح) ops.size() - 1];
        بهر_nodes[1] = i + (صحيح) ops.size() - 1;

        إذا (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.معلومات(), بهر_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(غوټه, weights, logits, ids) &&
            ggml_sycl_check_fusion_حافظه_ranges(cgraph, i, (صحيح) ops.size(), بهر_nodes, 2, /*is_topk_moe=*/ریښتیا)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, دلیلونه_2);
            أرجع (صحيح) ops.size() - 1;
        }
    } وإلا إذا (!دلیلونه_2.norm && !دلیلونه_2.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_لیکې, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ثابت ggml_tensor * softmax = cgraph->nodes[i + 4];
        بهر_nodes[0]                = i + 1;
        بهر_nodes[1]                = i + 5;

        إذا (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.معلومات(), بهر_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_حافظه_ranges(cgraph, i, (صحيح) ops.size(), بهر_nodes, 2, /*is_topk_moe=*/ریښتیا)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, دلیلونه_2);
            أرجع (صحيح) ops.size() - 1;
        }
    }

    أرجع 0;
}
