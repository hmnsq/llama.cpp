#अन्तर्भाव <cfloat>
#अन्तर्भाव <initializer_list>
#अन्तर्भाव <vector>

#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

संरचना ggml_sycl_topk_moe_तर्काः {
    तार्किक sigmoid{};
    तार्किक softmax{};
    तार्किक delayed_softmax{};
    तार्किक prob_bias{};
    तार्किक norm{};
    तार्किक scale{};
};

संरचना topk_moe_config {
    तार्किक use_sigmoid;
    तार्किक with_norm;
    तार्किक delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
प्रतिरूपम् <पूर्णाङ्क experts_प्रत्येकम्_thread, तार्किक use_सीमा>
स्थिर अन्तःस्थ शून्य softmax_warp_inplace(भिन्न (&vals)[experts_प्रत्येकम्_thread], ध्रुव पूर्णाङ्क सीमा, ध्रुव पूर्णाङ्क lane) {
    भिन्न max_val = -INFINITY;
#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < experts_प्रत्येकम्_thread; i++) {
        ध्रुव पूर्णाङ्क  idx    = lane + i * WARP_आकार;
        ध्रुव तार्किक सक्रिय = !use_सीमा || (idx < सीमा);
        यदि (सक्रिय) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_आकार>(max_val);

    भिन्न योग_2 = 0.f;
#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < experts_प्रत्येकम्_thread; i++) {
        ध्रुव पूर्णाङ्क  idx    = lane + i * WARP_आकार;
        ध्रुव तार्किक सक्रिय = !use_सीमा || (idx < सीमा);
        यदि (सक्रिय) {
            ध्रुव भिन्न val = sycl::घातांकः(vals[i] - max_val);
            vals[i]         = val;
            योग_2 += val;
        } अन्यथा {
            vals[i] = 0.f;
        }
    }
    योग_2 = warp_reduce_योग<WARP_आकार>(योग_2);

    ध्रुव भिन्न inv_योग = 1.0f / योग_2;
#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < experts_प्रत्येकम्_thread; i++) {
        ध्रुव पूर्णाङ्क idx = lane + i * WARP_आकार;
        यदि (!use_सीमा || idx < सीमा) {
            vals[i] *= inv_योग;
        }
    }
}

प्रतिरूपम् <पूर्णाङ्क experts_प्रत्येकम्_thread, तार्किक use_सीमा>
स्थिर अन्तःस्थ शून्य sigmoid_warp_inplace(भिन्न (&vals)[experts_प्रत्येकम्_thread], ध्रुव पूर्णाङ्क सीमा, ध्रुव पूर्णाङ्क lane) {
#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < experts_प्रत्येकम्_thread; i++) {
        ध्रुव पूर्णाङ्क  idx    = lane + i * WARP_आकार;
        ध्रुव तार्किक सक्रिय = !use_सीमा || (idx < सीमा);
        vals[i]           = सक्रिय ? 1.f / (1.f + sycl::घातांकः(-vals[i])) : -INFINITY;
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
प्रतिरूपम् <पूर्णाङ्क n_experts>
स्थिर शून्य topk_moe_kernel(ध्रुव भिन्न * __restrict__ logits,
                            भिन्न * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            ध्रुव पूर्णाङ्क                  n_पङ्क्तिकाः,
                            ध्रुव पूर्णाङ्क                  n_expert_प्रयुक्तम्,
                            ध्रुव भिन्न                clamp_val,
                            ध्रुव भिन्न                scale_val,
                            ध्रुव topk_moe_config       config) {
    स्वचालित      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    ध्रुव पूर्णाङ्क row       = item_ct1.get_गणः(0);
    यदि (row >= n_पङ्क्तिकाः) {
        प्रत्यागम;
    }
    ध्रुव पूर्णाङ्क lane = item_ct1.get_local_id(0);

    logits  += (size_t) n_experts * row;
    weights += (size_t) n_expert_प्रयुक्तम् * row;
    ids     += (size_t) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr पूर्णाङ्क experts_प्रत्येकम्_thread = (n_experts > WARP_आकार) ? n_experts / WARP_आकार : 1;

    भिन्न wt[experts_प्रत्येकम्_thread];
#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < experts_प्रत्येकम्_thread; i++) {
        wt[i] = -INFINITY;
    }
#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < n_experts; i += WARP_आकार) {
        ध्रुव पूर्णाङ्क expert  = i + lane;
        wt[i / WARP_आकार] = (n_experts % WARP_आकार == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    यदि (!config.delayed_softmax) {
        यदि (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_प्रत्येकम्_thread, असत्य>(wt, n_experts, lane);
        } अन्यथा {
            softmax_warp_inplace<experts_प्रत्येकम्_thread, असत्य>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < experts_प्रत्येकम्_thread; i++) {
        यदि (sycl::isnan(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    भिन्न wt_योग = 0.f;
    भिन्न output_weights[experts_प्रत्येकम्_thread];
#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < experts_प्रत्येकम्_thread; i++) {
        output_weights[i] = 0.f;
    }

    ध्रुव sycl::sub_गणः sg = item_ct1.get_sub_गणः();

    प्रति (पूर्णाङ्क k = 0; k < n_expert_प्रयुक्तम्; k++) {
        भिन्न max_val    = wt[0];
        पूर्णाङ्क   max_expert = lane;
#निर्देशः unroll
        प्रति (पूर्णाङ्क i = 1; i < experts_प्रत्येकम्_thread; i++) {
            ध्रुव पूर्णाङ्क expert = lane + i * WARP_आकार;
            यदि ((n_experts % WARP_आकार == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#निर्देशः unroll
        प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
            ध्रुव भिन्न val    = dpct::permute_sub_गणः_by_xor(sg, max_val, आवरणम्);
            ध्रुव पूर्णाङ्क   expert = dpct::permute_sub_गणः_by_xor(sg, max_expert, आवरणम्);
            यदि (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        यदि ((max_expert & (WARP_आकार - 1)) == lane) {
            wt[max_expert / WARP_आकार] = -INFINITY;
        }
        यदि ((k & (WARP_आकार - 1)) == lane) {
            output_weights[k / WARP_आकार] = max_val;
        }
        यदि ((max_expert & (WARP_आकार - 1)) == lane) {
            ids[k] = max_expert;
            यदि (config.with_norm) {
                wt_योग += max_val;
            }
        }
    }

    यदि (config.with_norm) {
        wt_योग          = warp_reduce_योग<WARP_आकार>(wt_योग);
        wt_योग          = sycl::fmax(wt_योग, clamp_val);
        ध्रुव भिन्न inv = 1.0f / wt_योग;
#निर्देशः unroll
        प्रति (पूर्णाङ्क i = 0; i < experts_प्रत्येकम्_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    यदि (config.delayed_softmax) {
        softmax_warp_inplace<experts_प्रत्येकम्_thread, सत्य>(output_weights, n_expert_प्रयुक्तम्, lane);
    }

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < experts_प्रत्येकम्_thread; i++) {
        ध्रुव पूर्णाङ्क idx = i * WARP_आकार + lane;
        यदि (idx < n_expert_प्रयुक्तम्) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

प्रतिरूपम् <पूर्णाङ्क n_experts>
स्थिर शून्य launch_topk_moe(queue_ptr stream, ध्रुव भिन्न * logits, भिन्न * weights, int32_t * ids, पूर्णाङ्क n_पङ्क्तिकाः,
                            पूर्णाङ्क n_expert_प्रयुक्तम्, भिन्न clamp_val, भिन्न scale_val, ध्रुव topk_moe_config & config) {
    ध्रुव sycl::range<1> खण्डम्_dims(WARP_आकार);
    ध्रुव sycl::range<1> खण्डम्_nums(n_पङ्क्तिकाः);
    stream->parallel_for(sycl::nd_range<1>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_पङ्क्तिकाः, n_expert_प्रयुक्तम्, clamp_val,
                                                        scale_val, config);
                             GGML_अप्रयुक्तम्(item_ct1);
                         });
}

स्थिर शून्य ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  ध्रुव ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  ध्रुव ggml_tensor *             clamp,
                                  ध्रुव ggml_tensor *             scale,
                                  ध्रुव ggml_sycl_topk_moe_तर्काः & तर्काः_2) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    ध्रुव पूर्णाङ्क n_experts     = logits->ne[0];
    ध्रुव पूर्णाङ्क n_पङ्क्तिकाः        = logits->ne[1];
    ध्रुव पूर्णाङ्क n_expert_प्रयुक्तम् = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_आकार(ids->type) == (size_t) n_experts);

    ध्रुव भिन्न * logits_d  = (ध्रुव भिन्न *) logits->दत्त;
    भिन्न *       weights_d = (भिन्न *) weights->दत्त;
    int32_t *     ids_d     = (int32_t *) ids->दत्त;

    ध्रुव तार्किक  with_norm = clamp != शून्यसूचकः;
    ध्रुव भिन्न clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -INFINITY;
    ध्रुव भिन्न scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = तर्काः_2.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = तर्काः_2.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_समुच्चयः_device(ctx.device);

    विकल्प (n_experts) {
        पक्ष 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_पङ्क्तिकाः, n_expert_प्रयुक्तम्, clamp_val, scale_val,
                               config);
            भङ्ग;
        पक्ष 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_पङ्क्तिकाः, n_expert_प्रयुक्तम्, clamp_val, scale_val,
                               config);
            भङ्ग;
        पक्ष 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_पङ्क्तिकाः, n_expert_प्रयुक्तम्, clamp_val, scale_val,
                               config);
            भङ्ग;
        पक्ष 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_पङ्क्तिकाः, n_expert_प्रयुक्तम्, clamp_val, scale_val,
                               config);
            भङ्ग;
        पक्ष 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_पङ्क्तिकाः, n_expert_प्रयुक्तम्, clamp_val, scale_val,
                                config);
            भङ्ग;
        पक्ष 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_पङ्क्तिकाः, n_expert_प्रयुक्तम्, clamp_val, scale_val,
                                config);
            भङ्ग;
        पक्ष 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_पङ्क्तिकाः, n_expert_प्रयुक्तम्, clamp_val, scale_val,
                                config);
            भङ्ग;
        पक्ष 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_पङ्क्तिकाः, n_expert_प्रयुक्तम्, clamp_val, scale_val,
                                 config);
            भङ्ग;
        पक्ष 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_पङ्क्तिकाः, n_expert_प्रयुक्तम्, clamp_val, scale_val,
                                 config);
            भङ्ग;
        पक्ष 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_पङ्क्तिकाः, n_expert_प्रयुक्तम्, clamp_val, scale_val,
                                 config);
            भङ्ग;
        स्वभाव:
            GGML_ASSERT(असत्य && "fatal error");
    }
}

स्थिर तार्किक ggml_sycl_should_use_topk_moe(ध्रुव ggml_tensor * gating_op, ध्रुव ggml_tensor * weights,
                                          ध्रुव ggml_tensor * logits, ध्रुव ggml_tensor * ids) {
    ध्रुव पूर्णाङ्क n_expert = ids->nb[1] / ids->nb[0];
    यदि ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        प्रत्यागम असत्य;
    }

    यदि (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        प्रत्यागम असत्य;
    }

    यदि (gating_op->op == GGML_OP_SOFT_MAX) {
        भिन्न scale    = 1.0f;
        भिन्न max_bias = 0.0f;

        स्मृतिं_प्रतिलिख(&scale, (ध्रुव भिन्न *) gating_op->op_params + 0, परिमाण(भिन्न));
        स्मृतिं_प्रतिलिख(&max_bias, (ध्रुव भिन्न *) gating_op->op_params + 1, परिमाण(भिन्न));

        यदि (!ggml_is_contiguous(gating_op->src[0])) {
            प्रत्यागम असत्य;
        }
        यदि (scale != 1.0f || max_bias != 0.0f) {
            प्रत्यागम असत्य;
        }
        // don't fuse when masks or sinks are present
        यदि (gating_op->src[1] || gating_op->src[2]) {
            प्रत्यागम असत्य;
        }
    } अन्यथा यदि (gating_op->op == GGML_OP_UNARY) {
        यदि (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            प्रत्यागम असत्य;
        }
    }

    प्रत्यागम सत्य;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
स्थिर तार्किक ggml_sycl_topk_moe_fusion(ध्रुव ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, ggml_sycl_topk_moe_तर्काः & तर्काः_2) {
    तर्काः_2 = ggml_sycl_topk_moe_तर्काः{};

    ध्रुव पूर्णाङ्क      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    यदि (nodes[ग्रन्थिः_idx]->op == GGML_OP_SOFT_MAX) {
        तर्काः_2.softmax = सत्य;
    }

    यदि (nodes[ग्रन्थिः_idx]->op == GGML_OP_UNARY) {
        यदि (ggml_get_unary_op(nodes[ग्रन्थिः_idx]) != GGML_UNARY_OP_SIGMOID) {
            प्रत्यागम असत्य;
        }
        तर्काः_2.sigmoid = सत्य;
    }

    यदि (nodes[ग्रन्थिः_idx]->op == GGML_OP_ARGSORT) {
        तर्काः_2.delayed_softmax = सत्य;
    }

    ग्रन्थिः_idx++;

    यदि (तर्काः_2.sigmoid || तर्काः_2.softmax) {
        // SOFTMAX -> RESHAPE
        यदि (ग्रन्थिः_idx >= n_nodes || nodes[ग्रन्थिः_idx]->op != GGML_OP_RESHAPE ||
            nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 1]) {
            प्रत्यागम असत्य;
        }
        ggml_tensor * probs_reshaped = nodes[ग्रन्थिः_idx];
        ग्रन्थिः_idx++;

        यदि (ग्रन्थिः_idx >= n_nodes) {
            प्रत्यागम असत्य;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        यदि (nodes[ग्रन्थिः_idx]->op == GGML_OP_सङ्कलय && nodes[ग्रन्थिः_idx]->src[0] == nodes[ग्रन्थिः_idx - 2]) {
            तर्काः_2.prob_bias = सत्य;
            ग्रन्थिः_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        यदि (ग्रन्थिः_idx >= n_nodes || nodes[ग्रन्थिः_idx]->op != GGML_OP_ARGSORT) {
            प्रत्यागम असत्य;
        }

        यदि (तर्काः_2.prob_bias && nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 1]) {
            प्रत्यागम असत्य;
        } अन्यथा यदि (!तर्काः_2.prob_bias && nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 2]) {
            प्रत्यागम असत्य;
        }

        ग्रन्थिः_idx++;

        // ARGSORT -> VIEW
        यदि (ग्रन्थिः_idx >= n_nodes || nodes[ग्रन्थिः_idx]->op != GGML_OP_VIEW ||
            nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 1]) {
            प्रत्यागम असत्य;
        }
        ग्रन्थिः_idx++;

        यदि (ग्रन्थिः_idx >= n_nodes || nodes[ग्रन्थिः_idx]->op != GGML_OP_GET_पङ्क्तिकाः) {
            प्रत्यागम असत्य;
        }

        // GET_ROWS
        यदि (nodes[ग्रन्थिः_idx]->src[0] != probs_reshaped || nodes[ग्रन्थिः_idx]->src[1] != nodes[ग्रन्थिः_idx - 1]) {
            प्रत्यागम असत्य;
        }
        ग्रन्थिः_idx++;
    } अन्यथा यदि (तर्काः_2.delayed_softmax) {
        यदि (ग्रन्थिः_idx - 2 < 0) {
            प्रत्यागम असत्य;
        }
        ggml_tensor * probs_reshaped = nodes[ग्रन्थिः_idx - 2];

        // VIEW -> ARGSORT
        यदि (ग्रन्थिः_idx >= n_nodes || nodes[ग्रन्थिः_idx]->op != GGML_OP_VIEW ||
            nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 1]) {
            प्रत्यागम असत्य;
        }
        ग्रन्थिः_idx++;

        // GET_ROWS
        यदि (ग्रन्थिः_idx >= n_nodes || nodes[ग्रन्थिः_idx]->src[1] != nodes[ग्रन्थिः_idx - 1] ||
            nodes[ग्रन्थिः_idx]->src[0] != probs_reshaped) {
            प्रत्यागम असत्य;
        }
        ग्रन्थिः_idx++;

        स्थिर ध्रुव std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        प्रति (ध्रुव ggml_op op : remaining_ops) {
            यदि (ग्रन्थिः_idx >= n_nodes || nodes[ग्रन्थिः_idx]->op != op || nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 1]) {
                प्रत्यागम असत्य;
            }
            ग्रन्थिः_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    यदि (ग्रन्थिः_idx >= n_nodes) {
        प्रत्यागम सत्य;
    }

    यदि (nodes[ग्रन्थिः_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        स्थिर ध्रुव std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_योग_पङ्क्तिकाः, GGML_OP_CLAMP };

        तर्काः_2.norm = सत्य;
        प्रति (ध्रुव ggml_op op : norm_ops) {
            यदि (nodes[ग्रन्थिः_idx]->op == op && nodes[ग्रन्थिः_idx]->src[0] == nodes[ग्रन्थिः_idx - 1]) {
                ग्रन्थिः_idx++;
            } अन्यथा {
                तर्काः_2.norm = असत्य;
                प्रत्यागम सत्य;
            }
        }

        // DIV <- CLAMP, RESHAPE
        यदि (nodes[ग्रन्थिः_idx]->op != GGML_OP_DIV || nodes[ग्रन्थिः_idx]->src[1] != nodes[ग्रन्थिः_idx - 1] ||
            nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 3]) {
            तर्काः_2.norm = असत्य;
            प्रत्यागम सत्य;
        }
        ग्रन्थिः_idx++;

        यदि (nodes[ग्रन्थिः_idx]->op != GGML_OP_RESHAPE || nodes[ग्रन्थिः_idx]->src[0] != nodes[ग्रन्थिः_idx - 1]) {
            तर्काः_2.norm = असत्य;
            प्रत्यागम सत्य;
        }
        ग्रन्थिः_idx++;
    }

    यदि (nodes[ग्रन्थिः_idx]->op == GGML_OP_SCALE && nodes[ग्रन्थिः_idx]->src[0] == nodes[ग्रन्थिः_idx - 1]) {
        तर्काः_2.scale = सत्य;
    }

    प्रत्यागम सत्य;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
स्थिर तार्किक ggml_sycl_check_fusion_स्मृतिः_ranges(ध्रुव ggml_cgraph * cgraph, ध्रुव पूर्णाङ्क ग्रन्थिः_idx,
                                                 ध्रुव पूर्णाङ्क ग्रन्थिः_गणनम्, ध्रुव पूर्णाङ्क * बहिर्गमनम्_nodes, ध्रुव पूर्णाङ्क बहिर्गमनम्_गणनम्,
                                                 ध्रुव तार्किक is_topk_moe = असत्य) {
    स्वचालित nodes_overlap = [&](ध्रुव ggml_tensor * a, ध्रुव ggml_tensor * b) {
        ध्रुव int64_t a_start = (int64_t) a->दत्त;
        ध्रुव int64_t a_अन्तः   = a_start + ggml_backend_buft_get_alloc_आकार(a->बफर->buft, a);

        ध्रुव int64_t b_start = (int64_t) b->दत्त;
        ध्रुव int64_t b_अन्तः   = b_start + ggml_backend_buft_get_alloc_आकार(b->बफर->buft, b);

        यदि ((b_start <= a_start && a_start < b_अन्तः) || (a_start <= b_start && b_start < a_अन्तः)) {
            प्रत्यागम सत्य;
        }

        प्रत्यागम असत्य;
    };

    तार्किक is_ok = सत्य;
    // exception for topk-moe, as each row is read entirely before writing
    यदि (ggml_nrows(cgraph->nodes[ग्रन्थिः_idx]) == 1 && is_topk_moe) {
        प्रत्यागम सत्य;
    }

    प्रति (पूर्णाङ्क i = 0; i < बहिर्गमनम्_गणनम्; ++i) {
        ध्रुव ggml_tensor * dst = cgraph->nodes[बहिर्गमनम्_nodes[i]];

        प्रति (पूर्णाङ्क j = ग्रन्थिः_idx; j < ग्रन्थिः_idx + ग्रन्थिः_गणनम्; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            प्रति (पूर्णाङ्क src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                ध्रुव ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                यदि (!src || src->op == GGML_OP_NONE) {
                    अनुवर्त;
                }

                यदि (nodes_overlap(dst, src)) {
                    तार्किक प्राप्तम् = असत्य;

                    प्रति (पूर्णाङ्क k = ग्रन्थिः_idx; k < j; ++k) {
                        यदि (cgraph->nodes[k] == src) {
                            प्राप्तम् = सत्य;
                            भङ्ग;
                        }
                    }

                    यदि (!प्राप्तम्) {
                        is_ok = असत्य;
                        भङ्ग;
                    }
                }
            }
        }
    }

    प्रत्यागम is_ok;
}

पूर्णाङ्क ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णाङ्क i) {
    यदि (!g_ggml_sycl_enable_fusion) {
        प्रत्यागम 0;
    }

    प्रत्यागम ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

पूर्णाङ्क ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णाङ्क i) {
    ggml_tensor * ग्रन्थिः = cgraph->nodes[i];

    यदि (ग्रन्थिः->op != GGML_OP_UNARY && ग्रन्थिः->op != GGML_OP_SOFT_MAX && ग्रन्थिः->op != GGML_OP_ARGSORT) {
        प्रत्यागम 0;
    }

    ggml_sycl_topk_moe_तर्काः तर्काः_2;
    यदि (!ggml_sycl_topk_moe_fusion(cgraph, i, तर्काः_2)) {
        प्रत्यागम 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    यदि (तर्काः_2.prob_bias) {
        प्रत्यागम 0;
    }

    ध्रुव ggml_tensor * logits  = ग्रन्थिः->src[0];
    ggml_tensor *       weights = शून्यसूचकः;
    ggml_tensor *       ids     = शून्यसूचकः;
    ध्रुव ggml_tensor * clamp   = शून्यसूचकः;
    ध्रुव ggml_tensor * scale   = शून्यसूचकः;

    std::vector<ggml_op> ops;
    पूर्णाङ्क                  बहिर्गमनम्_nodes[2];

    यदि (!तर्काः_2.delayed_softmax) {
        ध्रुव ggml_op gating_op = तर्काः_2.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_पङ्क्तिकाः });
        बहिर्गमनम्_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        यदि (तर्काः_2.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_योग_पङ्क्तिकाः, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (पूर्णाङ्क) ops.size() - 3];
        }
        यदि (तर्काः_2.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (पूर्णाङ्क) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (पूर्णाङ्क) ops.size() - 1];
        बहिर्गमनम्_nodes[1] = i + (पूर्णाङ्क) ops.size() - 1;

        यदि (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.दत्त(), बहिर्गमनम्_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(ग्रन्थिः, weights, logits, ids) &&
            ggml_sycl_check_fusion_स्मृतिः_ranges(cgraph, i, (पूर्णाङ्क) ops.size(), बहिर्गमनम्_nodes, 2, /*is_topk_moe=*/सत्य)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, तर्काः_2);
            प्रत्यागम (पूर्णाङ्क) ops.size() - 1;
        }
    } अन्यथा यदि (!तर्काः_2.norm && !तर्काः_2.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_पङ्क्तिकाः, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        ध्रुव ggml_tensor * softmax = cgraph->nodes[i + 4];
        बहिर्गमनम्_nodes[0]                = i + 1;
        बहिर्गमनम्_nodes[1]                = i + 5;

        यदि (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.दत्त(), बहिर्गमनम्_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_स्मृतिः_ranges(cgraph, i, (पूर्णाङ्क) ops.size(), बहिर्गमनम्_nodes, 2, /*is_topk_moe=*/सत्य)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, तर्काः_2);
            प्रत्यागम (पूर्णाङ्क) ops.size() - 1;
        }
    }

    प्रत्यागम 0;
}
