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

بنية ggml_sycl_topk_moe_آرگومانها {
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
قالب <صحيح experts_در_هر_thread, منطقي use_حد>
ساكن مضمن فراغ softmax_warp_inplace(عائم (&vals)[experts_در_هر_thread], ثابت صحيح حد, ثابت صحيح lane) {
    عائم max_val = -INFINITY;
#رهنمود unroll
    لكل (صحيح i = 0; i < experts_در_هر_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_اندازه;
        ثابت منطقي فعال = !use_حد || (idx < حد);
        إذا (فعال) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_اندازه>(max_val);

    عائم جمع = 0.f;
#رهنمود unroll
    لكل (صحيح i = 0; i < experts_در_هر_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_اندازه;
        ثابت منطقي فعال = !use_حد || (idx < حد);
        إذا (فعال) {
            ثابت عائم val = sycl::نمایی_تابع(vals[i] - max_val);
            vals[i]         = val;
            جمع += val;
        } وإلا {
            vals[i] = 0.f;
        }
    }
    جمع = warp_reduce_جمع<WARP_اندازه>(جمع);

    ثابت عائم inv_جمع = 1.0f / جمع;
#رهنمود unroll
    لكل (صحيح i = 0; i < experts_در_هر_thread; i++) {
        ثابت صحيح idx = lane + i * WARP_اندازه;
        إذا (!use_حد || idx < حد) {
            vals[i] *= inv_جمع;
        }
    }
}

قالب <صحيح experts_در_هر_thread, منطقي use_حد>
ساكن مضمن فراغ sigmoid_warp_inplace(عائم (&vals)[experts_در_هر_thread], ثابت صحيح حد, ثابت صحيح lane) {
#رهنمود unroll
    لكل (صحيح i = 0; i < experts_در_هر_thread; i++) {
        ثابت صحيح  idx    = lane + i * WARP_اندازه;
        ثابت منطقي فعال = !use_حد || (idx < حد);
        vals[i]           = فعال ? 1.f / (1.f + sycl::نمایی_تابع(-vals[i])) : -INFINITY;
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
قالب <صحيح n_experts>
ساكن فراغ topk_moe_kernel(ثابت عائم * __restrict__ logits,
                            عائم * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ثابت صحيح                  n_ردیفها,
                            ثابت صحيح                  n_expert_استفاده_شده,
                            ثابت عائم                clamp_val,
                            ثابت عائم                scale_val,
                            ثابت topk_moe_config       config) {
    خودکار      قلم_ct1 = sycl::ext::oneapi::this_work_قلم::get_nd_قلم<1>();
    ثابت صحيح ردیف       = قلم_ct1.get_گروه(0);
    إذا (ردیف >= n_ردیفها) {
        أرجع;
    }
    ثابت صحيح lane = قلم_ct1.get_local_id(0);

    logits  += (size_t) n_experts * ردیف;
    weights += (size_t) n_expert_استفاده_شده * ردیف;
    ids     += (size_t) n_experts * ردیف;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr صحيح experts_در_هر_thread = (n_experts > WARP_اندازه) ? n_experts / WARP_اندازه : 1;

    عائم wt[experts_در_هر_thread];
#رهنمود unroll
    لكل (صحيح i = 0; i < experts_در_هر_thread; i++) {
        wt[i] = -INFINITY;
    }
#رهنمود unroll
    لكل (صحيح i = 0; i < n_experts; i += WARP_اندازه) {
        ثابت صحيح expert  = i + lane;
        wt[i / WARP_اندازه] = (n_experts % WARP_اندازه == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    إذا (!config.delayed_softmax) {
        إذا (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_در_هر_thread, خطأ>(wt, n_experts, lane);
        } وإلا {
            softmax_warp_inplace<experts_در_هر_thread, خطأ>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#رهنمود unroll
    لكل (صحيح i = 0; i < experts_در_هر_thread; i++) {
        إذا (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    عائم wt_جمع = 0.f;
    عائم output_weights[experts_در_هر_thread];
#رهنمود unroll
    لكل (صحيح i = 0; i < experts_در_هر_thread; i++) {
        output_weights[i] = 0.f;
    }

    ثابت sycl::sub_گروه sg = قلم_ct1.get_sub_گروه();

    لكل (صحيح k = 0; k < n_expert_استفاده_شده; k++) {
        عائم max_val    = wt[0];
        صحيح   max_expert = lane;
#رهنمود unroll
        لكل (صحيح i = 1; i < experts_در_هر_thread; i++) {
            ثابت صحيح expert = lane + i * WARP_اندازه;
            إذا ((n_experts % WARP_اندازه == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#رهنمود unroll
        لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
            ثابت عائم val    = dpct::جایگشت_sub_گروه_by_xor(sg, max_val, ماسک);
            ثابت صحيح   expert = dpct::جایگشت_sub_گروه_by_xor(sg, max_expert, ماسک);
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
                wt_جمع += max_val;
            }
        }
    }

    إذا (config.with_norm) {
        wt_جمع          = warp_reduce_جمع<WARP_اندازه>(wt_جمع);
        wt_جمع          = sycl::fmax(wt_جمع, clamp_val);
        ثابت عائم inv = 1.0f / wt_جمع;
#رهنمود unroll
        لكل (صحيح i = 0; i < experts_در_هر_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    إذا (config.delayed_softmax) {
        softmax_warp_inplace<experts_در_هر_thread, درست>(output_weights, n_expert_استفاده_شده, lane);
    }

#رهنمود unroll
    لكل (صحيح i = 0; i < experts_در_هر_thread; i++) {
        ثابت صحيح idx = i * WARP_اندازه + lane;
        إذا (idx < n_expert_استفاده_شده) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

قالب <صحيح n_experts>
ساكن فراغ launch_topk_moe(صف_ptr stream, ثابت عائم * logits, عائم * weights, int32_t * ids, صحيح n_ردیفها,
                            صحيح n_expert_استفاده_شده, عائم clamp_val, عائم scale_val, ثابت topk_moe_config & config) {
    ثابت sycl::range<1> بلوک_dims(WARP_اندازه);
    ثابت sycl::range<1> بلوک_nums(n_ردیفها);
    stream->parallel_for(sycl::nd_range<1>(بلوک_nums * بلوک_dims, بلوک_dims),
                         [=](sycl::nd_قلم<1> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_ردیفها, n_expert_استفاده_شده, clamp_val,
                                                        scale_val, config);
                             GGML_استفادهنشده(قلم_ct1);
                         });
}

ساكن فراغ ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ثابت ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ثابت ggml_tensor *             clamp,
                                  ثابت ggml_tensor *             scale,
                                  ثابت ggml_sycl_topk_moe_آرگومانها & آرگومانها_2) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ثابت صحيح n_experts     = logits->ne[0];
    ثابت صحيح n_ردیفها        = logits->ne[1];
    ثابت صحيح n_expert_استفاده_شده = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_اندازه(ids->type) == (size_t) n_experts);

    ثابت عائم * logits_d  = (ثابت عائم *) logits->داده;
    عائم *       weights_d = (عائم *) weights->داده;
    int32_t *     ids_d     = (int32_t *) ids->داده;

    ثابت منطقي  with_norm = clamp != اشاره_تهی;
    ثابت عائم clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ثابت عائم scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = آرگومانها_2.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = آرگومانها_2.delayed_softmax;

    صف_ptr stream = ctx.stream();
    ggml_sycl_مجموعه_device(ctx.device);

    بدل (n_experts) {
        حالة 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_ردیفها, n_expert_استفاده_شده, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_ردیفها, n_expert_استفاده_شده, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_ردیفها, n_expert_استفاده_شده, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_ردیفها, n_expert_استفاده_شده, clamp_val, scale_val,
                               config);
            اكسر;
        حالة 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_ردیفها, n_expert_استفاده_شده, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_ردیفها, n_expert_استفاده_شده, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_ردیفها, n_expert_استفاده_شده, clamp_val, scale_val,
                                config);
            اكسر;
        حالة 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_ردیفها, n_expert_استفاده_شده, clamp_val, scale_val,
                                 config);
            اكسر;
        حالة 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_ردیفها, n_expert_استفاده_شده, clamp_val, scale_val,
                                 config);
            اكسر;
        حالة 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_ردیفها, n_expert_استفاده_شده, clamp_val, scale_val,
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

        کپی_حافظه(&scale, (ثابت عائم *) gating_op->op_params + 0, حجم(عائم));
        کپی_حافظه(&max_bias, (ثابت عائم *) gating_op->op_params + 1, حجم(عائم));

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

    أرجع درست;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
ساكن منطقي ggml_sycl_topk_moe_fusion(ثابت ggml_cgraph * cgraph, صحيح گره_idx, ggml_sycl_topk_moe_آرگومانها & آرگومانها_2) {
    آرگومانها_2 = ggml_sycl_topk_moe_آرگومانها{};

    ثابت صحيح      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    إذا (nodes[گره_idx]->op == GGML_OP_SOFT_MAX) {
        آرگومانها_2.softmax = درست;
    }

    إذا (nodes[گره_idx]->op == GGML_OP_UNARY) {
        إذا (ggml_get_unary_op(nodes[گره_idx]) != GGML_UNARY_OP_SIGMOID) {
            أرجع خطأ;
        }
        آرگومانها_2.sigmoid = درست;
    }

    إذا (nodes[گره_idx]->op == GGML_OP_ARGSORT) {
        آرگومانها_2.delayed_softmax = درست;
    }

    گره_idx++;

    إذا (آرگومانها_2.sigmoid || آرگومانها_2.softmax) {
        // SOFTMAX -> RESHAPE
        إذا (گره_idx >= n_nodes || nodes[گره_idx]->op != GGML_OP_RESHAPE ||
            nodes[گره_idx]->src[0] != nodes[گره_idx - 1]) {
            أرجع خطأ;
        }
        ggml_tensor * probs_reshaped = nodes[گره_idx];
        گره_idx++;

        إذا (گره_idx >= n_nodes) {
            أرجع خطأ;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        إذا (nodes[گره_idx]->op == GGML_OP_جمع_کن && nodes[گره_idx]->src[0] == nodes[گره_idx - 2]) {
            آرگومانها_2.prob_bias = درست;
            گره_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        إذا (گره_idx >= n_nodes || nodes[گره_idx]->op != GGML_OP_ARGSORT) {
            أرجع خطأ;
        }

        إذا (آرگومانها_2.prob_bias && nodes[گره_idx]->src[0] != nodes[گره_idx - 1]) {
            أرجع خطأ;
        } وإلا إذا (!آرگومانها_2.prob_bias && nodes[گره_idx]->src[0] != nodes[گره_idx - 2]) {
            أرجع خطأ;
        }

        گره_idx++;

        // ARGSORT -> VIEW
        إذا (گره_idx >= n_nodes || nodes[گره_idx]->op != GGML_OP_VIEW ||
            nodes[گره_idx]->src[0] != nodes[گره_idx - 1]) {
            أرجع خطأ;
        }
        گره_idx++;

        إذا (گره_idx >= n_nodes || nodes[گره_idx]->op != GGML_OP_GET_ردیفها) {
            أرجع خطأ;
        }

        // GET_ROWS
        إذا (nodes[گره_idx]->src[0] != probs_reshaped || nodes[گره_idx]->src[1] != nodes[گره_idx - 1]) {
            أرجع خطأ;
        }
        گره_idx++;
    } وإلا إذا (آرگومانها_2.delayed_softmax) {
        إذا (گره_idx - 2 < 0) {
            أرجع خطأ;
        }
        ggml_tensor * probs_reshaped = nodes[گره_idx - 2];

        // VIEW -> ARGSORT
        إذا (گره_idx >= n_nodes || nodes[گره_idx]->op != GGML_OP_VIEW ||
            nodes[گره_idx]->src[0] != nodes[گره_idx - 1]) {
            أرجع خطأ;
        }
        گره_idx++;

        // GET_ROWS
        إذا (گره_idx >= n_nodes || nodes[گره_idx]->src[1] != nodes[گره_idx - 1] ||
            nodes[گره_idx]->src[0] != probs_reshaped) {
            أرجع خطأ;
        }
        گره_idx++;

        ساكن ثابت std::vector<ggml_op> مانده_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        لكل (ثابت ggml_op op : مانده_ops) {
            إذا (گره_idx >= n_nodes || nodes[گره_idx]->op != op || nodes[گره_idx]->src[0] != nodes[گره_idx - 1]) {
                أرجع خطأ;
            }
            گره_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    إذا (گره_idx >= n_nodes) {
        أرجع درست;
    }

    إذا (nodes[گره_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        ساكن ثابت std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_جمع_ردیفها, GGML_OP_CLAMP };

        آرگومانها_2.norm = درست;
        لكل (ثابت ggml_op op : norm_ops) {
            إذا (nodes[گره_idx]->op == op && nodes[گره_idx]->src[0] == nodes[گره_idx - 1]) {
                گره_idx++;
            } وإلا {
                آرگومانها_2.norm = خطأ;
                أرجع درست;
            }
        }

        // DIV <- CLAMP, RESHAPE
        إذا (nodes[گره_idx]->op != GGML_OP_DIV || nodes[گره_idx]->src[1] != nodes[گره_idx - 1] ||
            nodes[گره_idx]->src[0] != nodes[گره_idx - 3]) {
            آرگومانها_2.norm = خطأ;
            أرجع درست;
        }
        گره_idx++;

        إذا (nodes[گره_idx]->op != GGML_OP_RESHAPE || nodes[گره_idx]->src[0] != nodes[گره_idx - 1]) {
            آرگومانها_2.norm = خطأ;
            أرجع درست;
        }
        گره_idx++;
    }

    إذا (nodes[گره_idx]->op == GGML_OP_SCALE && nodes[گره_idx]->src[0] == nodes[گره_idx - 1]) {
        آرگومانها_2.scale = درست;
    }

    أرجع درست;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
ساكن منطقي ggml_sycl_check_fusion_حافظه_ranges(ثابت ggml_cgraph * cgraph, ثابت صحيح گره_idx,
                                                 ثابت صحيح گره_شمار, ثابت صحيح * بیرون_nodes, ثابت صحيح بیرون_شمار,
                                                 ثابت منطقي is_topk_moe = خطأ) {
    خودکار nodes_overlap = [&](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        ثابت int64_t a_شروع = (int64_t) a->داده;
        ثابت int64_t a_انتها   = a_شروع + ggml_backend_buft_get_alloc_اندازه(a->میانگیر->buft, a);

        ثابت int64_t b_شروع = (int64_t) b->داده;
        ثابت int64_t b_انتها   = b_شروع + ggml_backend_buft_get_alloc_اندازه(b->میانگیر->buft, b);

        إذا ((b_شروع <= a_شروع && a_شروع < b_انتها) || (a_شروع <= b_شروع && b_شروع < a_انتها)) {
            أرجع درست;
        }

        أرجع خطأ;
    };

    منطقي is_ok = درست;
    // exception for topk-moe, as each row is read entirely before writing
    إذا (ggml_nrows(cgraph->nodes[گره_idx]) == 1 && is_topk_moe) {
        أرجع درست;
    }

    لكل (صحيح i = 0; i < بیرون_شمار; ++i) {
        ثابت ggml_tensor * dst = cgraph->nodes[بیرون_nodes[i]];

        لكل (صحيح j = گره_idx; j < گره_idx + گره_شمار; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            لكل (صحيح src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ثابت ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                إذا (!src || src->op == GGML_OP_NONE) {
                    استمر;
                }

                إذا (nodes_overlap(dst, src)) {
                    منطقي یافت = خطأ;

                    لكل (صحيح k = گره_idx; k < j; ++k) {
                        إذا (cgraph->nodes[k] == src) {
                            یافت = درست;
                            اكسر;
                        }
                    }

                    إذا (!یافت) {
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
    ggml_tensor * گره = cgraph->nodes[i];

    إذا (گره->op != GGML_OP_UNARY && گره->op != GGML_OP_SOFT_MAX && گره->op != GGML_OP_ARGSORT) {
        أرجع 0;
    }

    ggml_sycl_topk_moe_آرگومانها آرگومانها_2;
    إذا (!ggml_sycl_topk_moe_fusion(cgraph, i, آرگومانها_2)) {
        أرجع 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    إذا (آرگومانها_2.prob_bias) {
        أرجع 0;
    }

    ثابت ggml_tensor * logits  = گره->src[0];
    ggml_tensor *       weights = اشاره_تهی;
    ggml_tensor *       ids     = اشاره_تهی;
    ثابت ggml_tensor * clamp   = اشاره_تهی;
    ثابت ggml_tensor * scale   = اشاره_تهی;

    std::vector<ggml_op> ops;
    صحيح                  بیرون_nodes[2];

    إذا (!آرگومانها_2.delayed_softmax) {
        ثابت ggml_op gating_op = آرگومانها_2.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ردیفها });
        بیرون_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        إذا (آرگومانها_2.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_جمع_ردیفها, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (صحيح) ops.size() - 3];
        }
        إذا (آرگومانها_2.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (صحيح) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (صحيح) ops.size() - 1];
        بیرون_nodes[1] = i + (صحيح) ops.size() - 1;

        إذا (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.داده(), بیرون_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(گره, weights, logits, ids) &&
            ggml_sycl_check_fusion_حافظه_ranges(cgraph, i, (صحيح) ops.size(), بیرون_nodes, 2, /*is_topk_moe=*/درست)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, آرگومانها_2);
            أرجع (صحيح) ops.size() - 1;
        }
    } وإلا إذا (!آرگومانها_2.norm && !آرگومانها_2.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ردیفها, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ثابت ggml_tensor * softmax = cgraph->nodes[i + 4];
        بیرون_nodes[0]                = i + 1;
        بیرون_nodes[1]                = i + 5;

        إذا (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.داده(), بیرون_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_حافظه_ranges(cgraph, i, (صحيح) ops.size(), بیرون_nodes, 2, /*is_topk_moe=*/درست)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, آرگومانها_2);
            أرجع (صحيح) ops.size() - 1;
        }
    }

    أرجع 0;
}
