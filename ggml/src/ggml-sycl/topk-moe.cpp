#అన్తర్భావ <cfloat>
#అన్తర్భావ <initializer_list>
#అన్తర్భావ <vector>

#అన్తర్భావ "ggml.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

నిర్మాణం ggml_sycl_topk_moe_వాదనలు {
    తార్కిక sigmoid{};
    తార్కిక softmax{};
    తార్కిక delayed_softmax{};
    తార్కిక prob_bias{};
    తార్కిక norm{};
    తార్కిక scale{};
};

నిర్మాణం topk_moe_config {
    తార్కిక use_sigmoid;
    తార్కిక with_norm;
    తార్కిక delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
మూస <పూర్ణాంకం experts_చొప్పున_thread, తార్కిక use_హద్దు>
స్థిర అంతఃవరుస శూన్య softmax_warp_inplace(దశాంశం (&vals)[experts_చొప్పున_thread], స్థిరరాశి పూర్ణాంకం హద్దు, స్థిరరాశి పూర్ణాంకం lane) {
    దశాంశం max_val = -INFINITY;
#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < experts_చొప్పున_thread; i++) {
        స్థిరరాశి పూర్ణాంకం  idx    = lane + i * WARP_కొలత;
        స్థిరరాశి తార్కిక చురుకైన = !use_హద్దు || (idx < హద్దు);
        ఒకవేళ (చురుకైన) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_కొలత>(max_val);

    దశాంశం కూడిక = 0.f;
#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < experts_చొప్పున_thread; i++) {
        స్థిరరాశి పూర్ణాంకం  idx    = lane + i * WARP_కొలత;
        స్థిరరాశి తార్కిక చురుకైన = !use_హద్దు || (idx < హద్దు);
        ఒకవేళ (చురుకైన) {
            స్థిరరాశి దశాంశం val = sycl::ఘాతాంకం(vals[i] - max_val);
            vals[i]         = val;
            కూడిక += val;
        } లేకపోతే {
            vals[i] = 0.f;
        }
    }
    కూడిక = warp_reduce_కూడిక<WARP_కొలత>(కూడిక);

    స్థిరరాశి దశాంశం inv_కూడిక = 1.0f / కూడిక;
#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < experts_చొప్పున_thread; i++) {
        స్థిరరాశి పూర్ణాంకం idx = lane + i * WARP_కొలత;
        ఒకవేళ (!use_హద్దు || idx < హద్దు) {
            vals[i] *= inv_కూడిక;
        }
    }
}

మూస <పూర్ణాంకం experts_చొప్పున_thread, తార్కిక use_హద్దు>
స్థిర అంతఃవరుస శూన్య sigmoid_warp_inplace(దశాంశం (&vals)[experts_చొప్పున_thread], స్థిరరాశి పూర్ణాంకం హద్దు, స్థిరరాశి పూర్ణాంకం lane) {
#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < experts_చొప్పున_thread; i++) {
        స్థిరరాశి పూర్ణాంకం  idx    = lane + i * WARP_కొలత;
        స్థిరరాశి తార్కిక చురుకైన = !use_హద్దు || (idx < హద్దు);
        vals[i]           = చురుకైన ? 1.f / (1.f + sycl::ఘాతాంకం(-vals[i])) : -INFINITY;
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
మూస <పూర్ణాంకం n_experts>
స్థిర శూన్య topk_moe_kernel(స్థిరరాశి దశాంశం * __restrict__ logits,
                            దశాంశం * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            స్థిరరాశి పూర్ణాంకం                  n_అడ్డువరుసలు,
                            స్థిరరాశి పూర్ణాంకం                  n_expert_వాడిన,
                            స్థిరరాశి దశాంశం                clamp_val,
                            స్థిరరాశి దశాంశం                scale_val,
                            స్థిరరాశి topk_moe_config       config) {
    స్వయంచాలక      అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<1>();
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస       = అంశం_ct1.get_సమూహం(0);
    ఒకవేళ (అడ్డువరుస >= n_అడ్డువరుసలు) {
        తిరిగి;
    }
    స్థిరరాశి పూర్ణాంకం lane = అంశం_ct1.get_local_id(0);

    logits  += (size_t) n_experts * అడ్డువరుస;
    weights += (size_t) n_expert_వాడిన * అడ్డువరుస;
    ids     += (size_t) n_experts * అడ్డువరుస;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr పూర్ణాంకం experts_చొప్పున_thread = (n_experts > WARP_కొలత) ? n_experts / WARP_కొలత : 1;

    దశాంశం wt[experts_చొప్పున_thread];
#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < experts_చొప్పున_thread; i++) {
        wt[i] = -INFINITY;
    }
#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < n_experts; i += WARP_కొలత) {
        స్థిరరాశి పూర్ణాంకం expert  = i + lane;
        wt[i / WARP_కొలత] = (n_experts % WARP_కొలత == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    ఒకవేళ (!config.delayed_softmax) {
        ఒకవేళ (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_చొప్పున_thread, అబద్ధం>(wt, n_experts, lane);
        } లేకపోతే {
            softmax_warp_inplace<experts_చొప్పున_thread, అబద్ధం>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < experts_చొప్పున_thread; i++) {
        ఒకవేళ (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    దశాంశం wt_కూడిక = 0.f;
    దశాంశం output_weights[experts_చొప్పున_thread];
#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < experts_చొప్పున_thread; i++) {
        output_weights[i] = 0.f;
    }

    స్థిరరాశి sycl::sub_సమూహం sg = అంశం_ct1.get_sub_సమూహం();

    కొరకు (పూర్ణాంకం k = 0; k < n_expert_వాడిన; k++) {
        దశాంశం max_val    = wt[0];
        పూర్ణాంకం   max_expert = lane;
#సూచన unroll
        కొరకు (పూర్ణాంకం i = 1; i < experts_చొప్పున_thread; i++) {
            స్థిరరాశి పూర్ణాంకం expert = lane + i * WARP_కొలత;
            ఒకవేళ ((n_experts % WARP_కొలత == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#సూచన unroll
        కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
            స్థిరరాశి దశాంశం val    = dpct::క్రమమార్పు_sub_సమూహం_by_xor(sg, max_val, ముసుగు);
            స్థిరరాశి పూర్ణాంకం   expert = dpct::క్రమమార్పు_sub_సమూహం_by_xor(sg, max_expert, ముసుగు);
            ఒకవేళ (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        ఒకవేళ ((max_expert & (WARP_కొలత - 1)) == lane) {
            wt[max_expert / WARP_కొలత] = -INFINITY;
        }
        ఒకవేళ ((k & (WARP_కొలత - 1)) == lane) {
            output_weights[k / WARP_కొలత] = max_val;
        }
        ఒకవేళ ((max_expert & (WARP_కొలత - 1)) == lane) {
            ids[k] = max_expert;
            ఒకవేళ (config.with_norm) {
                wt_కూడిక += max_val;
            }
        }
    }

    ఒకవేళ (config.with_norm) {
        wt_కూడిక          = warp_reduce_కూడిక<WARP_కొలత>(wt_కూడిక);
        wt_కూడిక          = sycl::fmax(wt_కూడిక, clamp_val);
        స్థిరరాశి దశాంశం inv = 1.0f / wt_కూడిక;
#సూచన unroll
        కొరకు (పూర్ణాంకం i = 0; i < experts_చొప్పున_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    ఒకవేళ (config.delayed_softmax) {
        softmax_warp_inplace<experts_చొప్పున_thread, నిజం>(output_weights, n_expert_వాడిన, lane);
    }

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < experts_చొప్పున_thread; i++) {
        స్థిరరాశి పూర్ణాంకం idx = i * WARP_కొలత + lane;
        ఒకవేళ (idx < n_expert_వాడిన) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

మూస <పూర్ణాంకం n_experts>
స్థిర శూన్య launch_topk_moe(వరుస_ptr stream, స్థిరరాశి దశాంశం * logits, దశాంశం * weights, int32_t * ids, పూర్ణాంకం n_అడ్డువరుసలు,
                            పూర్ణాంకం n_expert_వాడిన, దశాంశం clamp_val, దశాంశం scale_val, స్థిరరాశి topk_moe_config & config) {
    స్థిరరాశి sycl::range<1> ఖండం_dims(WARP_కొలత);
    స్థిరరాశి sycl::range<1> ఖండం_nums(n_అడ్డువరుసలు);
    stream->parallel_for(sycl::nd_range<1>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_అడ్డువరుసలు, n_expert_వాడిన, clamp_val,
                                                        scale_val, config);
                             GGML_వాడని(అంశం_ct1);
                         });
}

స్థిర శూన్య ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  స్థిరరాశి ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  స్థిరరాశి ggml_tensor *             clamp,
                                  స్థిరరాశి ggml_tensor *             scale,
                                  స్థిరరాశి ggml_sycl_topk_moe_వాదనలు & వాదనలు_2) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    స్థిరరాశి పూర్ణాంకం n_experts     = logits->ne[0];
    స్థిరరాశి పూర్ణాంకం n_అడ్డువరుసలు        = logits->ne[1];
    స్థిరరాశి పూర్ణాంకం n_expert_వాడిన = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_కొలత(ids->type) == (size_t) n_experts);

    స్థిరరాశి దశాంశం * logits_d  = (స్థిరరాశి దశాంశం *) logits->దత్తాంశం;
    దశాంశం *       weights_d = (దశాంశం *) weights->దత్తాంశం;
    int32_t *     ids_d     = (int32_t *) ids->దత్తాంశం;

    స్థిరరాశి తార్కిక  with_norm = clamp != శూన్యసూచిక;
    స్థిరరాశి దశాంశం clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    స్థిరరాశి దశాంశం scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = వాదనలు_2.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = వాదనలు_2.delayed_softmax;

    వరుస_ptr stream = ctx.stream();
    ggml_sycl_సమితి_device(ctx.device);

    ఎంపిక (n_experts) {
        స్థితి 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_అడ్డువరుసలు, n_expert_వాడిన, clamp_val, scale_val,
                               config);
            ఆపు;
        స్థితి 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_అడ్డువరుసలు, n_expert_వాడిన, clamp_val, scale_val,
                               config);
            ఆపు;
        స్థితి 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_అడ్డువరుసలు, n_expert_వాడిన, clamp_val, scale_val,
                               config);
            ఆపు;
        స్థితి 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_అడ్డువరుసలు, n_expert_వాడిన, clamp_val, scale_val,
                               config);
            ఆపు;
        స్థితి 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_అడ్డువరుసలు, n_expert_వాడిన, clamp_val, scale_val,
                                config);
            ఆపు;
        స్థితి 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_అడ్డువరుసలు, n_expert_వాడిన, clamp_val, scale_val,
                                config);
            ఆపు;
        స్థితి 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_అడ్డువరుసలు, n_expert_వాడిన, clamp_val, scale_val,
                                config);
            ఆపు;
        స్థితి 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_అడ్డువరుసలు, n_expert_వాడిన, clamp_val, scale_val,
                                 config);
            ఆపు;
        స్థితి 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_అడ్డువరుసలు, n_expert_వాడిన, clamp_val, scale_val,
                                 config);
            ఆపు;
        స్థితి 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_అడ్డువరుసలు, n_expert_వాడిన, clamp_val, scale_val,
                                 config);
            ఆపు;
        అప్రమేయ:
            GGML_ASSERT(అబద్ధం && "fatal error");
    }
}

స్థిర తార్కిక ggml_sycl_should_use_topk_moe(స్థిరరాశి ggml_tensor * gating_op, స్థిరరాశి ggml_tensor * weights,
                                          స్థిరరాశి ggml_tensor * logits, స్థిరరాశి ggml_tensor * ids) {
    స్థిరరాశి పూర్ణాంకం n_expert = ids->nb[1] / ids->nb[0];
    ఒకవేళ ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (gating_op->op == GGML_OP_SOFT_MAX) {
        దశాంశం scale    = 1.0f;
        దశాంశం max_bias = 0.0f;

        జ్ఞాపకం_నకలు(&scale, (స్థిరరాశి దశాంశం *) gating_op->op_params + 0, పరిమాణం(దశాంశం));
        జ్ఞాపకం_నకలు(&max_bias, (స్థిరరాశి దశాంశం *) gating_op->op_params + 1, పరిమాణం(దశాంశం));

        ఒకవేళ (!ggml_is_contiguous(gating_op->src[0])) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (scale != 1.0f || max_bias != 0.0f) {
            తిరిగి అబద్ధం;
        }
        // don't fuse when masks or sinks are present
        ఒకవేళ (gating_op->src[1] || gating_op->src[2]) {
            తిరిగి అబద్ధం;
        }
    } లేకపోతే ఒకవేళ (gating_op->op == GGML_OP_UNARY) {
        ఒకవేళ (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            తిరిగి అబద్ధం;
        }
    }

    తిరిగి నిజం;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
స్థిర తార్కిక ggml_sycl_topk_moe_fusion(స్థిరరాశి ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, ggml_sycl_topk_moe_వాదనలు & వాదనలు_2) {
    వాదనలు_2 = ggml_sycl_topk_moe_వాదనలు{};

    స్థిరరాశి పూర్ణాంకం      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    ఒకవేళ (nodes[కణుపు_idx]->op == GGML_OP_SOFT_MAX) {
        వాదనలు_2.softmax = నిజం;
    }

    ఒకవేళ (nodes[కణుపు_idx]->op == GGML_OP_UNARY) {
        ఒకవేళ (ggml_get_unary_op(nodes[కణుపు_idx]) != GGML_UNARY_OP_SIGMOID) {
            తిరిగి అబద్ధం;
        }
        వాదనలు_2.sigmoid = నిజం;
    }

    ఒకవేళ (nodes[కణుపు_idx]->op == GGML_OP_ARGSORT) {
        వాదనలు_2.delayed_softmax = నిజం;
    }

    కణుపు_idx++;

    ఒకవేళ (వాదనలు_2.sigmoid || వాదనలు_2.softmax) {
        // SOFTMAX -> RESHAPE
        ఒకవేళ (కణుపు_idx >= n_nodes || nodes[కణుపు_idx]->op != GGML_OP_RESHAPE ||
            nodes[కణుపు_idx]->src[0] != nodes[కణుపు_idx - 1]) {
            తిరిగి అబద్ధం;
        }
        ggml_tensor * probs_reshaped = nodes[కణుపు_idx];
        కణుపు_idx++;

        ఒకవేళ (కణుపు_idx >= n_nodes) {
            తిరిగి అబద్ధం;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        ఒకవేళ (nodes[కణుపు_idx]->op == GGML_OP_కూడు && nodes[కణుపు_idx]->src[0] == nodes[కణుపు_idx - 2]) {
            వాదనలు_2.prob_bias = నిజం;
            కణుపు_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        ఒకవేళ (కణుపు_idx >= n_nodes || nodes[కణుపు_idx]->op != GGML_OP_ARGSORT) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (వాదనలు_2.prob_bias && nodes[కణుపు_idx]->src[0] != nodes[కణుపు_idx - 1]) {
            తిరిగి అబద్ధం;
        } లేకపోతే ఒకవేళ (!వాదనలు_2.prob_bias && nodes[కణుపు_idx]->src[0] != nodes[కణుపు_idx - 2]) {
            తిరిగి అబద్ధం;
        }

        కణుపు_idx++;

        // ARGSORT -> VIEW
        ఒకవేళ (కణుపు_idx >= n_nodes || nodes[కణుపు_idx]->op != GGML_OP_VIEW ||
            nodes[కణుపు_idx]->src[0] != nodes[కణుపు_idx - 1]) {
            తిరిగి అబద్ధం;
        }
        కణుపు_idx++;

        ఒకవేళ (కణుపు_idx >= n_nodes || nodes[కణుపు_idx]->op != GGML_OP_GET_అడ్డువరుసలు) {
            తిరిగి అబద్ధం;
        }

        // GET_ROWS
        ఒకవేళ (nodes[కణుపు_idx]->src[0] != probs_reshaped || nodes[కణుపు_idx]->src[1] != nodes[కణుపు_idx - 1]) {
            తిరిగి అబద్ధం;
        }
        కణుపు_idx++;
    } లేకపోతే ఒకవేళ (వాదనలు_2.delayed_softmax) {
        ఒకవేళ (కణుపు_idx - 2 < 0) {
            తిరిగి అబద్ధం;
        }
        ggml_tensor * probs_reshaped = nodes[కణుపు_idx - 2];

        // VIEW -> ARGSORT
        ఒకవేళ (కణుపు_idx >= n_nodes || nodes[కణుపు_idx]->op != GGML_OP_VIEW ||
            nodes[కణుపు_idx]->src[0] != nodes[కణుపు_idx - 1]) {
            తిరిగి అబద్ధం;
        }
        కణుపు_idx++;

        // GET_ROWS
        ఒకవేళ (కణుపు_idx >= n_nodes || nodes[కణుపు_idx]->src[1] != nodes[కణుపు_idx - 1] ||
            nodes[కణుపు_idx]->src[0] != probs_reshaped) {
            తిరిగి అబద్ధం;
        }
        కణుపు_idx++;

        స్థిర స్థిరరాశి std::vector<ggml_op> మిగిలిన_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        కొరకు (స్థిరరాశి ggml_op op : మిగిలిన_ops) {
            ఒకవేళ (కణుపు_idx >= n_nodes || nodes[కణుపు_idx]->op != op || nodes[కణుపు_idx]->src[0] != nodes[కణుపు_idx - 1]) {
                తిరిగి అబద్ధం;
            }
            కణుపు_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    ఒకవేళ (కణుపు_idx >= n_nodes) {
        తిరిగి నిజం;
    }

    ఒకవేళ (nodes[కణుపు_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        స్థిర స్థిరరాశి std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_కూడిక_అడ్డువరుసలు, GGML_OP_CLAMP };

        వాదనలు_2.norm = నిజం;
        కొరకు (స్థిరరాశి ggml_op op : norm_ops) {
            ఒకవేళ (nodes[కణుపు_idx]->op == op && nodes[కణుపు_idx]->src[0] == nodes[కణుపు_idx - 1]) {
                కణుపు_idx++;
            } లేకపోతే {
                వాదనలు_2.norm = అబద్ధం;
                తిరిగి నిజం;
            }
        }

        // DIV <- CLAMP, RESHAPE
        ఒకవేళ (nodes[కణుపు_idx]->op != GGML_OP_DIV || nodes[కణుపు_idx]->src[1] != nodes[కణుపు_idx - 1] ||
            nodes[కణుపు_idx]->src[0] != nodes[కణుపు_idx - 3]) {
            వాదనలు_2.norm = అబద్ధం;
            తిరిగి నిజం;
        }
        కణుపు_idx++;

        ఒకవేళ (nodes[కణుపు_idx]->op != GGML_OP_RESHAPE || nodes[కణుపు_idx]->src[0] != nodes[కణుపు_idx - 1]) {
            వాదనలు_2.norm = అబద్ధం;
            తిరిగి నిజం;
        }
        కణుపు_idx++;
    }

    ఒకవేళ (nodes[కణుపు_idx]->op == GGML_OP_SCALE && nodes[కణుపు_idx]->src[0] == nodes[కణుపు_idx - 1]) {
        వాదనలు_2.scale = నిజం;
    }

    తిరిగి నిజం;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
స్థిర తార్కిక ggml_sycl_check_fusion_జ్ఞాపకం_ranges(స్థిరరాశి ggml_cgraph * cgraph, స్థిరరాశి పూర్ణాంకం కణుపు_idx,
                                                 స్థిరరాశి పూర్ణాంకం కణుపు_లెక్క, స్థిరరాశి పూర్ణాంకం * బహిర్గతం_nodes, స్థిరరాశి పూర్ణాంకం బహిర్గతం_లెక్క,
                                                 స్థిరరాశి తార్కిక is_topk_moe = అబద్ధం) {
    స్వయంచాలక nodes_overlap = [&](స్థిరరాశి ggml_tensor * a, స్థిరరాశి ggml_tensor * b) {
        స్థిరరాశి int64_t a_ప్రారంభం = (int64_t) a->దత్తాంశం;
        స్థిరరాశి int64_t a_చివర   = a_ప్రారంభం + ggml_backend_buft_get_alloc_కొలత(a->బఫరు->buft, a);

        స్థిరరాశి int64_t b_ప్రారంభం = (int64_t) b->దత్తాంశం;
        స్థిరరాశి int64_t b_చివర   = b_ప్రారంభం + ggml_backend_buft_get_alloc_కొలత(b->బఫరు->buft, b);

        ఒకవేళ ((b_ప్రారంభం <= a_ప్రారంభం && a_ప్రారంభం < b_చివర) || (a_ప్రారంభం <= b_ప్రారంభం && b_ప్రారంభం < a_చివర)) {
            తిరిగి నిజం;
        }

        తిరిగి అబద్ధం;
    };

    తార్కిక is_ok = నిజం;
    // exception for topk-moe, as each row is read entirely before writing
    ఒకవేళ (ggml_nrows(cgraph->nodes[కణుపు_idx]) == 1 && is_topk_moe) {
        తిరిగి నిజం;
    }

    కొరకు (పూర్ణాంకం i = 0; i < బహిర్గతం_లెక్క; ++i) {
        స్థిరరాశి ggml_tensor * dst = cgraph->nodes[బహిర్గతం_nodes[i]];

        కొరకు (పూర్ణాంకం j = కణుపు_idx; j < కణుపు_idx + కణుపు_లెక్క; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            కొరకు (పూర్ణాంకం src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                స్థిరరాశి ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                ఒకవేళ (!src || src->op == GGML_OP_NONE) {
                    కొనసాగు;
                }

                ఒకవేళ (nodes_overlap(dst, src)) {
                    తార్కిక దొరికింది = అబద్ధం;

                    కొరకు (పూర్ణాంకం k = కణుపు_idx; k < j; ++k) {
                        ఒకవేళ (cgraph->nodes[k] == src) {
                            దొరికింది = నిజం;
                            ఆపు;
                        }
                    }

                    ఒకవేళ (!దొరికింది) {
                        is_ok = అబద్ధం;
                        ఆపు;
                    }
                }
            }
        }
    }

    తిరిగి is_ok;
}

పూర్ణాంకం ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, పూర్ణాంకం i) {
    ఒకవేళ (!g_ggml_sycl_enable_fusion) {
        తిరిగి 0;
    }

    తిరిగి ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

పూర్ణాంకం ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, పూర్ణాంకం i) {
    ggml_tensor * కణుపు = cgraph->nodes[i];

    ఒకవేళ (కణుపు->op != GGML_OP_UNARY && కణుపు->op != GGML_OP_SOFT_MAX && కణుపు->op != GGML_OP_ARGSORT) {
        తిరిగి 0;
    }

    ggml_sycl_topk_moe_వాదనలు వాదనలు_2;
    ఒకవేళ (!ggml_sycl_topk_moe_fusion(cgraph, i, వాదనలు_2)) {
        తిరిగి 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    ఒకవేళ (వాదనలు_2.prob_bias) {
        తిరిగి 0;
    }

    స్థిరరాశి ggml_tensor * logits  = కణుపు->src[0];
    ggml_tensor *       weights = శూన్యసూచిక;
    ggml_tensor *       ids     = శూన్యసూచిక;
    స్థిరరాశి ggml_tensor * clamp   = శూన్యసూచిక;
    స్థిరరాశి ggml_tensor * scale   = శూన్యసూచిక;

    std::vector<ggml_op> ops;
    పూర్ణాంకం                  బహిర్గతం_nodes[2];

    ఒకవేళ (!వాదనలు_2.delayed_softmax) {
        స్థిరరాశి ggml_op gating_op = వాదనలు_2.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_అడ్డువరుసలు });
        బహిర్గతం_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        ఒకవేళ (వాదనలు_2.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_కూడిక_అడ్డువరుసలు, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (పూర్ణాంకం) ops.size() - 3];
        }
        ఒకవేళ (వాదనలు_2.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (పూర్ణాంకం) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (పూర్ణాంకం) ops.size() - 1];
        బహిర్గతం_nodes[1] = i + (పూర్ణాంకం) ops.size() - 1;

        ఒకవేళ (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.దత్తాంశం(), బహిర్గతం_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(కణుపు, weights, logits, ids) &&
            ggml_sycl_check_fusion_జ్ఞాపకం_ranges(cgraph, i, (పూర్ణాంకం) ops.size(), బహిర్గతం_nodes, 2, /*is_topk_moe=*/నిజం)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, వాదనలు_2);
            తిరిగి (పూర్ణాంకం) ops.size() - 1;
        }
    } లేకపోతే ఒకవేళ (!వాదనలు_2.norm && !వాదనలు_2.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_అడ్డువరుసలు, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        స్థిరరాశి ggml_tensor * softmax = cgraph->nodes[i + 4];
        బహిర్గతం_nodes[0]                = i + 1;
        బహిర్గతం_nodes[1]                = i + 5;

        ఒకవేళ (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.దత్తాంశం(), బహిర్గతం_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_జ్ఞాపకం_ranges(cgraph, i, (పూర్ణాంకం) ops.size(), బహిర్గతం_nodes, 2, /*is_topk_moe=*/నిజం)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, వాదనలు_2);
            తిరిగి (పూర్ణాంకం) ops.size() - 1;
        }
    }

    తిరిగి 0;
}
