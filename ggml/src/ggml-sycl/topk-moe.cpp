#समावेश <cfloat>
#समावेश <initializer_list>
#समावेश <vector>

#समावेश "ggml.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-backend-impl.h"
#समावेश "topk-moe.hpp"

// SYCL port of ggml-cuda/topk-moe.cu. The kernel is a translation of the CUDA no-bias, no-PDL
// path of topk_moe_cuda; the fusion-detection helpers below are ported near-verbatim from
// ggml-cuda.cu (pure graph / pointer inspection, backend-agnostic). Bias is not implemented here:
// if a routing bias is detected, the fusion is declined and the eager path runs unchanged.

काष्ठा ggml_sycl_topk_moe_args {
    bool sigmoid{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

काष्ठा topk_moe_config {
    bool use_sigmoid;
    bool with_norm;
    bool delayed_softmax;
};

// warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path
ढाँचा <पूर्णांक experts_per_thread, bool use_limit>
अटल अंतरभूत व्योम softmax_warp_inplace(भग्न (&vals)[experts_per_thread], स्थिर पूर्णांक limit, स्थिर पूर्णांक lane) {
    भग्न max_val = -अनन्त;
#आशय unroll
    क्रम (पूर्णांक i = 0; i < experts_per_thread; i++) {
        स्थिर पूर्णांक  idx    = lane + i * WARP_आकार;
        स्थिर bool active = !use_limit || (idx < limit);
        अगर (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_आकार>(max_val);

    भग्न योग_2 = 0.f;
#आशय unroll
    क्रम (पूर्णांक i = 0; i < experts_per_thread; i++) {
        स्थिर पूर्णांक  idx    = lane + i * WARP_आकार;
        स्थिर bool active = !use_limit || (idx < limit);
        अगर (active) {
            स्थिर भग्न val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            योग_2 += val;
        } अन्यथा {
            vals[i] = 0.f;
        }
    }
    योग_2 = warp_reduce_योग<WARP_आकार>(योग_2);

    स्थिर भग्न inv_योग = 1.0f / योग_2;
#आशय unroll
    क्रम (पूर्णांक i = 0; i < experts_per_thread; i++) {
        स्थिर पूर्णांक idx = lane + i * WARP_आकार;
        अगर (!use_limit || idx < limit) {
            vals[i] *= inv_योग;
        }
    }
}

ढाँचा <पूर्णांक experts_per_thread, bool use_limit>
अटल अंतरभूत व्योम sigmoid_warp_inplace(भग्न (&vals)[experts_per_thread], स्थिर पूर्णांक limit, स्थिर पूर्णांक lane) {
#आशय unroll
    क्रम (पूर्णांक i = 0; i < experts_per_thread; i++) {
        स्थिर पूर्णांक  idx    = lane + i * WARP_आकार;
        स्थिर bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + sycl::exp(-vals[i])) : -अनन्त;
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
ढाँचा <पूर्णांक n_experts>
अटल व्योम topk_moe_kernel(स्थिर भग्न * __restrict__ logits,
                            भग्न * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            स्थिर पूर्णांक                  n_rows,
                            स्थिर पूर्णांक                  n_expert_used,
                            स्थिर भग्न                clamp_val,
                            स्थिर भग्न                scale_val,
                            स्थिर topk_moe_config       config) {
    स्वतः      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    स्थिर पूर्णांक row       = item_ct1.get_group(0);
    अगर (row >= n_rows) {
        वापस;
    }
    स्थिर पूर्णांक lane = item_ct1.get_local_id(0);

    logits  += (माप_प्रकार) n_experts * row;
    weights += (माप_प्रकार) n_expert_used * row;
    ids     += (माप_प्रकार) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr पूर्णांक experts_per_thread = (n_experts > WARP_आकार) ? n_experts / WARP_आकार : 1;

    भग्न wt[experts_per_thread];
#आशय unroll
    क्रम (पूर्णांक i = 0; i < experts_per_thread; i++) {
        wt[i] = -अनन्त;
    }
#आशय unroll
    क्रम (पूर्णांक i = 0; i < n_experts; i += WARP_आकार) {
        स्थिर पूर्णांक expert  = i + lane;
        wt[i / WARP_आकार] = (n_experts % WARP_आकार == 0 || expert < n_experts) ? logits[expert] : -अनन्त;
    }

    अगर (!config.delayed_softmax) {
        अगर (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        } अन्यथा {
            softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#आशय unroll
    क्रम (पूर्णांक i = 0; i < experts_per_thread; i++) {
        अगर (sycl::isnan(wt[i])) {
            wt[i] = -भग्न_उच्च;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    भग्न wt_योग = 0.f;
    भग्न output_weights[experts_per_thread];
#आशय unroll
    क्रम (पूर्णांक i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    स्थिर sycl::sub_group sg = item_ct1.get_sub_group();

    क्रम (पूर्णांक k = 0; k < n_expert_used; k++) {
        भग्न max_val    = wt[0];
        पूर्णांक   max_expert = lane;
#आशय unroll
        क्रम (पूर्णांक i = 1; i < experts_per_thread; i++) {
            स्थिर पूर्णांक expert = lane + i * WARP_आकार;
            अगर ((n_experts % WARP_आकार == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#आशय unroll
        क्रम (पूर्णांक mask = WARP_आकार / 2; mask > 0; mask >>= 1) {
            स्थिर भग्न val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            स्थिर पूर्णांक   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            अगर (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        अगर ((max_expert & (WARP_आकार - 1)) == lane) {
            wt[max_expert / WARP_आकार] = -अनन्त;
        }
        अगर ((k & (WARP_आकार - 1)) == lane) {
            output_weights[k / WARP_आकार] = max_val;
        }
        अगर ((max_expert & (WARP_आकार - 1)) == lane) {
            ids[k] = max_expert;
            अगर (config.with_norm) {
                wt_योग += max_val;
            }
        }
    }

    अगर (config.with_norm) {
        wt_योग          = warp_reduce_योग<WARP_आकार>(wt_योग);
        wt_योग          = sycl::fmax(wt_योग, clamp_val);
        स्थिर भग्न inv = 1.0f / wt_योग;
#आशय unroll
        क्रम (पूर्णांक i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    अगर (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, lane);
    }

#आशय unroll
    क्रम (पूर्णांक i = 0; i < experts_per_thread; i++) {
        स्थिर पूर्णांक idx = i * WARP_आकार + lane;
        अगर (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

ढाँचा <पूर्णांक n_experts>
अटल व्योम launch_topk_moe(queue_ptr stream, स्थिर भग्न * logits, भग्न * weights, int32_t * ids, पूर्णांक n_rows,
                            पूर्णांक n_expert_used, भग्न clamp_val, भग्न scale_val, स्थिर topk_moe_config & config) {
    स्थिर sycl::range<1> block_dims(WARP_आकार);
    स्थिर sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                        scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

अटल व्योम ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  स्थिर ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  स्थिर ggml_tensor *             clamp,
                                  स्थिर ggml_tensor *             scale,
                                  स्थिर ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    स्थिर पूर्णांक n_experts     = logits->ne[0];
    स्थिर पूर्णांक n_rows        = logits->ne[1];
    स्थिर पूर्णांक n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_आकार(ids->type) == (माप_प्रकार) n_experts);

    स्थिर भग्न * logits_d  = (स्थिर भग्न *) logits->तथ्य;
    भग्न *       weights_d = (भग्न *) weights->तथ्य;
    int32_t *     ids_d     = (int32_t *) ids->तथ्य;

    स्थिर bool  with_norm = clamp != nullptr;
    स्थिर भग्न clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -अनन्त;
    स्थिर भग्न scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    चयन (n_experts) {
        हाल 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            अवरोध;
        हाल 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            अवरोध;
        हाल 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            अवरोध;
        हाल 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            अवरोध;
        हाल 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            अवरोध;
        हाल 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            अवरोध;
        हाल 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            अवरोध;
        हाल 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            अवरोध;
        हाल 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            अवरोध;
        हाल 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            अवरोध;
        शेष:
            GGML_ASSERT(false && "fatal error");
    }
}

अटल bool ggml_sycl_should_use_topk_moe(स्थिर ggml_tensor * gating_op, स्थिर ggml_tensor * weights,
                                          स्थिर ggml_tensor * logits, स्थिर ggml_tensor * ids) {
    स्थिर पूर्णांक n_expert = ids->nb[1] / ids->nb[0];
    अगर ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        वापस false;
    }

    अगर (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        वापस false;
    }

    अगर (gating_op->op == GGML_OP_SOFT_MAX) {
        भग्न scale    = 1.0f;
        भग्न max_bias = 0.0f;

        स_नकल(&scale, (स्थिर भग्न *) gating_op->op_params + 0, माप(भग्न));
        स_नकल(&max_bias, (स्थिर भग्न *) gating_op->op_params + 1, माप(भग्न));

        अगर (!ggml_is_contiguous(gating_op->src[0])) {
            वापस false;
        }
        अगर (scale != 1.0f || max_bias != 0.0f) {
            वापस false;
        }
        // don't fuse when masks or sinks are present
        अगर (gating_op->src[1] || gating_op->src[2]) {
            वापस false;
        }
    } अन्यथा अगर (gating_op->op == GGML_OP_UNARY) {
        अगर (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            वापस false;
        }
    }

    वापस true;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
अटल bool ggml_sycl_topk_moe_fusion(स्थिर ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    स्थिर पूर्णांक      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    अगर (nodes[गाँठ_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    अगर (nodes[गाँठ_idx]->op == GGML_OP_UNARY) {
        अगर (ggml_get_unary_op(nodes[गाँठ_idx]) != GGML_UNARY_OP_SIGMOID) {
            वापस false;
        }
        args.sigmoid = true;
    }

    अगर (nodes[गाँठ_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    गाँठ_idx++;

    अगर (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        अगर (गाँठ_idx >= n_nodes || nodes[गाँठ_idx]->op != GGML_OP_RESHAPE ||
            nodes[गाँठ_idx]->src[0] != nodes[गाँठ_idx - 1]) {
            वापस false;
        }
        ggml_tensor * probs_reshaped = nodes[गाँठ_idx];
        गाँठ_idx++;

        अगर (गाँठ_idx >= n_nodes) {
            वापस false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        अगर (nodes[गाँठ_idx]->op == GGML_OP_ADD && nodes[गाँठ_idx]->src[0] == nodes[गाँठ_idx - 2]) {
            args.prob_bias = true;
            गाँठ_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        अगर (गाँठ_idx >= n_nodes || nodes[गाँठ_idx]->op != GGML_OP_ARGSORT) {
            वापस false;
        }

        अगर (args.prob_bias && nodes[गाँठ_idx]->src[0] != nodes[गाँठ_idx - 1]) {
            वापस false;
        } अन्यथा अगर (!args.prob_bias && nodes[गाँठ_idx]->src[0] != nodes[गाँठ_idx - 2]) {
            वापस false;
        }

        गाँठ_idx++;

        // ARGSORT -> VIEW
        अगर (गाँठ_idx >= n_nodes || nodes[गाँठ_idx]->op != GGML_OP_VIEW ||
            nodes[गाँठ_idx]->src[0] != nodes[गाँठ_idx - 1]) {
            वापस false;
        }
        गाँठ_idx++;

        अगर (गाँठ_idx >= n_nodes || nodes[गाँठ_idx]->op != GGML_OP_GET_ROWS) {
            वापस false;
        }

        // GET_ROWS
        अगर (nodes[गाँठ_idx]->src[0] != probs_reshaped || nodes[गाँठ_idx]->src[1] != nodes[गाँठ_idx - 1]) {
            वापस false;
        }
        गाँठ_idx++;
    } अन्यथा अगर (args.delayed_softmax) {
        अगर (गाँठ_idx - 2 < 0) {
            वापस false;
        }
        ggml_tensor * probs_reshaped = nodes[गाँठ_idx - 2];

        // VIEW -> ARGSORT
        अगर (गाँठ_idx >= n_nodes || nodes[गाँठ_idx]->op != GGML_OP_VIEW ||
            nodes[गाँठ_idx]->src[0] != nodes[गाँठ_idx - 1]) {
            वापस false;
        }
        गाँठ_idx++;

        // GET_ROWS
        अगर (गाँठ_idx >= n_nodes || nodes[गाँठ_idx]->src[1] != nodes[गाँठ_idx - 1] ||
            nodes[गाँठ_idx]->src[0] != probs_reshaped) {
            वापस false;
        }
        गाँठ_idx++;

        अटल स्थिर std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        क्रम (स्थिर ggml_op op : remaining_ops) {
            अगर (गाँठ_idx >= n_nodes || nodes[गाँठ_idx]->op != op || nodes[गाँठ_idx]->src[0] != nodes[गाँठ_idx - 1]) {
                वापस false;
            }
            गाँठ_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    अगर (गाँठ_idx >= n_nodes) {
        वापस true;
    }

    अगर (nodes[गाँठ_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        अटल स्थिर std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_योग_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        क्रम (स्थिर ggml_op op : norm_ops) {
            अगर (nodes[गाँठ_idx]->op == op && nodes[गाँठ_idx]->src[0] == nodes[गाँठ_idx - 1]) {
                गाँठ_idx++;
            } अन्यथा {
                args.norm = false;
                वापस true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        अगर (nodes[गाँठ_idx]->op != GGML_OP_DIV || nodes[गाँठ_idx]->src[1] != nodes[गाँठ_idx - 1] ||
            nodes[गाँठ_idx]->src[0] != nodes[गाँठ_idx - 3]) {
            args.norm = false;
            वापस true;
        }
        गाँठ_idx++;

        अगर (nodes[गाँठ_idx]->op != GGML_OP_RESHAPE || nodes[गाँठ_idx]->src[0] != nodes[गाँठ_idx - 1]) {
            args.norm = false;
            वापस true;
        }
        गाँठ_idx++;
    }

    अगर (nodes[गाँठ_idx]->op == GGML_OP_SCALE && nodes[गाँठ_idx]->src[0] == nodes[गाँठ_idx - 1]) {
        args.scale = true;
    }

    वापस true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
अटल bool ggml_sycl_check_fusion_memory_ranges(स्थिर ggml_cgraph * cgraph, स्थिर पूर्णांक गाँठ_idx,
                                                 स्थिर पूर्णांक गाँठ_गिनती, स्थिर पूर्णांक * out_nodes, स्थिर पूर्णांक out_गिनती,
                                                 स्थिर bool is_topk_moe = false) {
    स्वतः nodes_overlap = [&](स्थिर ggml_tensor * a, स्थिर ggml_tensor * b) {
        स्थिर int64_t a_start = (int64_t) a->तथ्य;
        स्थिर int64_t a_end   = a_start + ggml_backend_buft_get_alloc_आकार(a->बफर->buft, a);

        स्थिर int64_t b_start = (int64_t) b->तथ्य;
        स्थिर int64_t b_end   = b_start + ggml_backend_buft_get_alloc_आकार(b->बफर->buft, b);

        अगर ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            वापस true;
        }

        वापस false;
    };

    bool is_ok = true;
    // exception for topk-moe, as each row is read entirely before writing
    अगर (ggml_nrows(cgraph->nodes[गाँठ_idx]) == 1 && is_topk_moe) {
        वापस true;
    }

    क्रम (पूर्णांक i = 0; i < out_गिनती; ++i) {
        स्थिर ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        क्रम (पूर्णांक j = गाँठ_idx; j < गाँठ_idx + गाँठ_गिनती; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            क्रम (पूर्णांक src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                स्थिर ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                अगर (!src || src->op == GGML_OP_NONE) {
                    जारी;
                }

                अगर (nodes_overlap(dst, src)) {
                    bool found = false;

                    क्रम (पूर्णांक k = गाँठ_idx; k < j; ++k) {
                        अगर (cgraph->nodes[k] == src) {
                            found = true;
                            अवरोध;
                        }
                    }

                    अगर (!found) {
                        is_ok = false;
                        अवरोध;
                    }
                }
            }
        }
    }

    वापस is_ok;
}

पूर्णांक ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णांक i) {
    अगर (!g_ggml_sycl_enable_fusion) {
        वापस 0;
    }

    वापस ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

पूर्णांक ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णांक i) {
    ggml_tensor * गाँठ = cgraph->nodes[i];

    अगर (गाँठ->op != GGML_OP_UNARY && गाँठ->op != GGML_OP_SOFT_MAX && गाँठ->op != GGML_OP_ARGSORT) {
        वापस 0;
    }

    ggml_sycl_topk_moe_args args;
    अगर (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        वापस 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    अगर (args.prob_bias) {
        वापस 0;
    }

    स्थिर ggml_tensor * logits  = गाँठ->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    स्थिर ggml_tensor * clamp   = nullptr;
    स्थिर ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    पूर्णांक                  out_nodes[2];

    अगर (!args.delayed_softmax) {
        स्थिर ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        अगर (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_योग_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (पूर्णांक) ops.size() - 3];
        }
        अगर (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (पूर्णांक) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (पूर्णांक) ops.size() - 1];
        out_nodes[1] = i + (पूर्णांक) ops.size() - 1;

        अगर (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.तथ्य(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(गाँठ, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (पूर्णांक) ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            वापस (पूर्णांक) ops.size() - 1;
        }
    } अन्यथा अगर (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        स्थिर ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        अगर (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.तथ्य(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (पूर्णांक) ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            वापस (पूर्णांक) ops.size() - 1;
        }
    }

    वापस 0;
}
