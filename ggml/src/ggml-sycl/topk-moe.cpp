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

संरचना ggml_sycl_topk_moe_args {
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
ढाँचा <पूर्णांक experts_per_thread, तार्किक use_limit>
स्थिर ओळीत शून्य softmax_warp_inplace(दशांश (&vals)[experts_per_thread], स्थिरांक पूर्णांक limit, स्थिरांक पूर्णांक lane) {
    दशांश max_val = -अनन्त;
#pragma unroll
    साठी (पूर्णांक i = 0; i < experts_per_thread; i++) {
        स्थिरांक पूर्णांक  idx    = lane + i * WARP_SIZE;
        स्थिरांक तार्किक active = !use_limit || (idx < limit);
        जर (active) {
            max_val = sycl::fmax(max_val, vals[i]);
        }
    }
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    दशांश sum = 0.f;
#pragma unroll
    साठी (पूर्णांक i = 0; i < experts_per_thread; i++) {
        स्थिरांक पूर्णांक  idx    = lane + i * WARP_SIZE;
        स्थिरांक तार्किक active = !use_limit || (idx < limit);
        जर (active) {
            स्थिरांक दशांश val = sycl::exp(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } अन्यथा {
            vals[i] = 0.f;
        }
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);

    स्थिरांक दशांश inv_sum = 1.0f / sum;
#pragma unroll
    साठी (पूर्णांक i = 0; i < experts_per_thread; i++) {
        स्थिरांक पूर्णांक idx = lane + i * WARP_SIZE;
        जर (!use_limit || idx < limit) {
            vals[i] *= inv_sum;
        }
    }
}

ढाँचा <पूर्णांक experts_per_thread, तार्किक use_limit>
स्थिर ओळीत शून्य sigmoid_warp_inplace(दशांश (&vals)[experts_per_thread], स्थिरांक पूर्णांक limit, स्थिरांक पूर्णांक lane) {
#pragma unroll
    साठी (पूर्णांक i = 0; i < experts_per_thread; i++) {
        स्थिरांक पूर्णांक  idx    = lane + i * WARP_SIZE;
        स्थिरांक तार्किक active = !use_limit || (idx < limit);
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
स्थिर शून्य topk_moe_kernel(स्थिरांक दशांश * __restrict__ logits,
                            दशांश * __restrict__       weights,
                            int32_t * __restrict__     ids,
                            स्थिरांक पूर्णांक                  n_rows,
                            स्थिरांक पूर्णांक                  n_expert_used,
                            स्थिरांक दशांश                clamp_val,
                            स्थिरांक दशांश                scale_val,
                            स्थिरांक topk_moe_config       config) {
    स्वतः      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    स्थिरांक पूर्णांक row       = item_ct1.get_group(0);
    जर (row >= n_rows) {
        परत;
    }
    स्थिरांक पूर्णांक lane = item_ct1.get_local_id(0);

    logits  += (माप_प्रकार) n_experts * row;
    weights += (माप_प्रकार) n_expert_used * row;
    ids     += (माप_प्रकार) n_experts * row;  // ids row stride is n_experts (matches the argsort tensor)

    constexpr पूर्णांक experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    दशांश wt[experts_per_thread];
#pragma unroll
    साठी (पूर्णांक i = 0; i < experts_per_thread; i++) {
        wt[i] = -अनन्त;
    }
#pragma unroll
    साठी (पूर्णांक i = 0; i < n_experts; i += WARP_SIZE) {
        स्थिरांक पूर्णांक expert  = i + lane;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -अनन्त;
    }

    जर (!config.delayed_softmax) {
        जर (config.use_sigmoid) {
            sigmoid_warp_inplace<experts_per_thread, असत्य>(wt, n_experts, lane);
        } अन्यथा {
            softmax_warp_inplace<experts_per_thread, असत्य>(wt, n_experts, lane);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs. NaN comparisons
    // always return false, which would cause the same expert to be selected repeatedly.
#pragma unroll
    साठी (पूर्णांक i = 0; i < experts_per_thread; i++) {
        जर (sycl::isnan(wt[i])) {
            wt[i] = -भग्न_उच्च;
        }
    }

    // each thread now holds either a portion of the softmax distribution or the raw logits. Do the
    // argmax reduce over n_expert_used, each time marking the selected expert as -inf to exclude it
    // from the next iteration.

    दशांश wt_sum = 0.f;
    दशांश output_weights[experts_per_thread];
#pragma unroll
    साठी (पूर्णांक i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    स्थिरांक sycl::sub_group sg = item_ct1.get_sub_group();

    साठी (पूर्णांक k = 0; k < n_expert_used; k++) {
        दशांश max_val    = wt[0];
        पूर्णांक   max_expert = lane;
#pragma unroll
        साठी (पूर्णांक i = 1; i < experts_per_thread; i++) {
            स्थिरांक पूर्णांक expert = lane + i * WARP_SIZE;
            जर ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
#pragma unroll
        साठी (पूर्णांक mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            स्थिरांक दशांश val    = dpct::permute_sub_group_by_xor(sg, max_val, mask);
            स्थिरांक पूर्णांक   expert = dpct::permute_sub_group_by_xor(sg, max_expert, mask);
            जर (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        जर ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -अनन्त;
        }
        जर ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }
        जर ((max_expert & (WARP_SIZE - 1)) == lane) {
            ids[k] = max_expert;
            जर (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    जर (config.with_norm) {
        wt_sum          = warp_reduce_sum<WARP_SIZE>(wt_sum);
        wt_sum          = sycl::fmax(wt_sum, clamp_val);
        स्थिरांक दशांश inv = 1.0f / wt_sum;
#pragma unroll
        साठी (पूर्णांक i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv;
        }
    }

    जर (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, सत्य>(output_weights, n_expert_used, lane);
    }

#pragma unroll
    साठी (पूर्णांक i = 0; i < experts_per_thread; i++) {
        स्थिरांक पूर्णांक idx = i * WARP_SIZE + lane;
        जर (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

ढाँचा <पूर्णांक n_experts>
स्थिर शून्य launch_topk_moe(queue_ptr stream, स्थिरांक दशांश * logits, दशांश * weights, int32_t * ids, पूर्णांक n_rows,
                            पूर्णांक n_expert_used, दशांश clamp_val, दशांश scale_val, स्थिरांक topk_moe_config & config) {
    स्थिरांक sycl::range<1> block_dims(WARP_SIZE);
    स्थिरांक sycl::range<1> block_nums(n_rows);
    stream->parallel_for(sycl::nd_range<1>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             topk_moe_kernel<n_experts>(logits, weights, ids, n_rows, n_expert_used, clamp_val,
                                                        scale_val, config);
                             GGML_UNUSED(item_ct1);
                         });
}

स्थिर शून्य ggml_sycl_op_topk_moe(ggml_backend_sycl_context &     ctx,
                                  स्थिरांक ggml_tensor *             logits,
                                  ggml_tensor *                   weights,
                                  ggml_tensor *                   ids,
                                  स्थिरांक ggml_tensor *             clamp,
                                  स्थिरांक ggml_tensor *             scale,
                                  स्थिरांक ggml_sycl_topk_moe_args & args) {
    GGML_ASSERT(logits->type  == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type     == GGML_TYPE_I32);

    स्थिरांक पूर्णांक n_experts     = logits->ne[0];
    स्थिरांक पूर्णांक n_rows        = logits->ne[1];
    स्थिरांक पूर्णांक n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (माप_प्रकार) n_experts);

    स्थिरांक दशांश * logits_d  = (स्थिरांक दशांश *) logits->data;
    दशांश *       weights_d = (दशांश *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    स्थिरांक तार्किक  with_norm = clamp != nullptr;
    स्थिरांक दशांश clamp_val = clamp ? ggml_get_op_params_f32(clamp, 0) : -अनन्त;
    स्थिरांक दशांश scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    topk_moe_config config;
    config.use_sigmoid     = args.sigmoid;
    config.with_norm       = with_norm;
    config.delayed_softmax = args.delayed_softmax;

    queue_ptr stream = ctx.stream();
    ggml_sycl_set_device(ctx.device);

    निवड (n_experts) {
        स्थिती 1:
            launch_topk_moe<1>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            थांबा;
        स्थिती 2:
            launch_topk_moe<2>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            थांबा;
        स्थिती 4:
            launch_topk_moe<4>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            थांबा;
        स्थिती 8:
            launch_topk_moe<8>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                               config);
            थांबा;
        स्थिती 16:
            launch_topk_moe<16>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            थांबा;
        स्थिती 32:
            launch_topk_moe<32>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            थांबा;
        स्थिती 64:
            launch_topk_moe<64>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                config);
            थांबा;
        स्थिती 128:
            launch_topk_moe<128>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            थांबा;
        स्थिती 256:
            launch_topk_moe<256>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            थांबा;
        स्थिती 512:
            launch_topk_moe<512>(stream, logits_d, weights_d, ids_d, n_rows, n_expert_used, clamp_val, scale_val,
                                 config);
            थांबा;
        पूर्वनिर्धारित:
            GGML_ASSERT(असत्य && "fatal error");
    }
}

स्थिर तार्किक ggml_sycl_should_use_topk_moe(स्थिरांक ggml_tensor * gating_op, स्थिरांक ggml_tensor * weights,
                                          स्थिरांक ggml_tensor * logits, स्थिरांक ggml_tensor * ids) {
    स्थिरांक पूर्णांक n_expert = ids->nb[1] / ids->nb[0];
    जर ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        परत असत्य;
    }

    जर (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        परत असत्य;
    }

    जर (gating_op->op == GGML_OP_SOFT_MAX) {
        दशांश scale    = 1.0f;
        दशांश max_bias = 0.0f;

        स_नकल(&scale, (स्थिरांक दशांश *) gating_op->op_params + 0, आकार(दशांश));
        स_नकल(&max_bias, (स्थिरांक दशांश *) gating_op->op_params + 1, आकार(दशांश));

        जर (!ggml_is_contiguous(gating_op->src[0])) {
            परत असत्य;
        }
        जर (scale != 1.0f || max_bias != 0.0f) {
            परत असत्य;
        }
        // don't fuse when masks or sinks are present
        जर (gating_op->src[1] || gating_op->src[2]) {
            परत असत्य;
        }
    } अन्यथा जर (gating_op->op == GGML_OP_UNARY) {
        जर (ggml_get_unary_op(gating_op) != GGML_UNARY_OP_SIGMOID) {
            परत असत्य;
        }
    }

    परत सत्य;
}

// ported from ggml_cuda_topk_moe_fusion - pure graph inspection, backend-agnostic
स्थिर तार्किक ggml_sycl_topk_moe_fusion(स्थिरांक ggml_cgraph * cgraph, पूर्णांक गाठ_idx, ggml_sycl_topk_moe_args & args) {
    args = ggml_sycl_topk_moe_args{};

    स्थिरांक पूर्णांक      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    जर (nodes[गाठ_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = सत्य;
    }

    जर (nodes[गाठ_idx]->op == GGML_OP_UNARY) {
        जर (ggml_get_unary_op(nodes[गाठ_idx]) != GGML_UNARY_OP_SIGMOID) {
            परत असत्य;
        }
        args.sigmoid = सत्य;
    }

    जर (nodes[गाठ_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = सत्य;
    }

    गाठ_idx++;

    जर (args.sigmoid || args.softmax) {
        // SOFTMAX -> RESHAPE
        जर (गाठ_idx >= n_nodes || nodes[गाठ_idx]->op != GGML_OP_RESHAPE ||
            nodes[गाठ_idx]->src[0] != nodes[गाठ_idx - 1]) {
            परत असत्य;
        }
        ggml_tensor * probs_reshaped = nodes[गाठ_idx];
        गाठ_idx++;

        जर (गाठ_idx >= n_nodes) {
            परत असत्य;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        जर (nodes[गाठ_idx]->op == GGML_OP_ADD && nodes[गाठ_idx]->src[0] == nodes[गाठ_idx - 2]) {
            args.prob_bias = सत्य;
            गाठ_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        जर (गाठ_idx >= n_nodes || nodes[गाठ_idx]->op != GGML_OP_ARGSORT) {
            परत असत्य;
        }

        जर (args.prob_bias && nodes[गाठ_idx]->src[0] != nodes[गाठ_idx - 1]) {
            परत असत्य;
        } अन्यथा जर (!args.prob_bias && nodes[गाठ_idx]->src[0] != nodes[गाठ_idx - 2]) {
            परत असत्य;
        }

        गाठ_idx++;

        // ARGSORT -> VIEW
        जर (गाठ_idx >= n_nodes || nodes[गाठ_idx]->op != GGML_OP_VIEW ||
            nodes[गाठ_idx]->src[0] != nodes[गाठ_idx - 1]) {
            परत असत्य;
        }
        गाठ_idx++;

        जर (गाठ_idx >= n_nodes || nodes[गाठ_idx]->op != GGML_OP_GET_ROWS) {
            परत असत्य;
        }

        // GET_ROWS
        जर (nodes[गाठ_idx]->src[0] != probs_reshaped || nodes[गाठ_idx]->src[1] != nodes[गाठ_idx - 1]) {
            परत असत्य;
        }
        गाठ_idx++;
    } अन्यथा जर (args.delayed_softmax) {
        जर (गाठ_idx - 2 < 0) {
            परत असत्य;
        }
        ggml_tensor * probs_reshaped = nodes[गाठ_idx - 2];

        // VIEW -> ARGSORT
        जर (गाठ_idx >= n_nodes || nodes[गाठ_idx]->op != GGML_OP_VIEW ||
            nodes[गाठ_idx]->src[0] != nodes[गाठ_idx - 1]) {
            परत असत्य;
        }
        गाठ_idx++;

        // GET_ROWS
        जर (गाठ_idx >= n_nodes || nodes[गाठ_idx]->src[1] != nodes[गाठ_idx - 1] ||
            nodes[गाठ_idx]->src[0] != probs_reshaped) {
            परत असत्य;
        }
        गाठ_idx++;

        स्थिर स्थिरांक std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        साठी (स्थिरांक ggml_op op : remaining_ops) {
            जर (गाठ_idx >= n_nodes || nodes[गाठ_idx]->op != op || nodes[गाठ_idx]->src[0] != nodes[गाठ_idx - 1]) {
                परत असत्य;
            }
            गाठ_idx++;
        }
    }

    // at this point we can check for norm + scale; everything is now at least valid up to the norm
    जर (गाठ_idx >= n_nodes) {
        परत सत्य;
    }

    जर (nodes[गाठ_idx]->op == GGML_OP_RESHAPE) {
        // check RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE
        स्थिर स्थिरांक std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = सत्य;
        साठी (स्थिरांक ggml_op op : norm_ops) {
            जर (nodes[गाठ_idx]->op == op && nodes[गाठ_idx]->src[0] == nodes[गाठ_idx - 1]) {
                गाठ_idx++;
            } अन्यथा {
                args.norm = असत्य;
                परत सत्य;
            }
        }

        // DIV <- CLAMP, RESHAPE
        जर (nodes[गाठ_idx]->op != GGML_OP_DIV || nodes[गाठ_idx]->src[1] != nodes[गाठ_idx - 1] ||
            nodes[गाठ_idx]->src[0] != nodes[गाठ_idx - 3]) {
            args.norm = असत्य;
            परत सत्य;
        }
        गाठ_idx++;

        जर (nodes[गाठ_idx]->op != GGML_OP_RESHAPE || nodes[गाठ_idx]->src[0] != nodes[गाठ_idx - 1]) {
            args.norm = असत्य;
            परत सत्य;
        }
        गाठ_idx++;
    }

    जर (nodes[गाठ_idx]->op == GGML_OP_SCALE && nodes[गाठ_idx]->src[0] == nodes[गाठ_idx - 1]) {
        args.scale = सत्य;
    }

    परत सत्य;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
// ported from ggml_cuda_check_fusion_memory_ranges - pure pointer/range inspection
स्थिर तार्किक ggml_sycl_check_fusion_memory_ranges(स्थिरांक ggml_cgraph * cgraph, स्थिरांक पूर्णांक गाठ_idx,
                                                 स्थिरांक पूर्णांक गाठ_मोजणी, स्थिरांक पूर्णांक * out_nodes, स्थिरांक पूर्णांक out_मोजणी,
                                                 स्थिरांक तार्किक is_topk_moe = असत्य) {
    स्वतः nodes_overlap = [&](स्थिरांक ggml_tensor * a, स्थिरांक ggml_tensor * b) {
        स्थिरांक int64_t a_start = (int64_t) a->data;
        स्थिरांक int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        स्थिरांक int64_t b_start = (int64_t) b->data;
        स्थिरांक int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        जर ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            परत सत्य;
        }

        परत असत्य;
    };

    तार्किक is_ok = सत्य;
    // exception for topk-moe, as each row is read entirely before writing
    जर (ggml_nrows(cgraph->nodes[गाठ_idx]) == 1 && is_topk_moe) {
        परत सत्य;
    }

    साठी (पूर्णांक i = 0; i < out_मोजणी; ++i) {
        स्थिरांक ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        साठी (पूर्णांक j = गाठ_idx; j < गाठ_idx + गाठ_मोजणी; ++j) {
            // loop over all srcs of all nodes in the fusion. If the src overlaps the destination and
            // the src is not an intermediate node that's being elided, then disable fusion.
            साठी (पूर्णांक src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                स्थिरांक ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                जर (!src || src->op == GGML_OP_NONE) {
                    सुरूठेवा;
                }

                जर (nodes_overlap(dst, src)) {
                    तार्किक found = असत्य;

                    साठी (पूर्णांक k = गाठ_idx; k < j; ++k) {
                        जर (cgraph->nodes[k] == src) {
                            found = सत्य;
                            थांबा;
                        }
                    }

                    जर (!found) {
                        is_ok = असत्य;
                        थांबा;
                    }
                }
            }
        }
    }

    परत is_ok;
}

पूर्णांक ggml_sycl_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णांक i) {
    जर (!g_ggml_sycl_enable_fusion) {
        परत 0;
    }

    परत ggml_sycl_fuse_topk_moe(ctx, cgraph, i);
}

पूर्णांक ggml_sycl_fuse_topk_moe(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णांक i) {
    ggml_tensor * गाठ = cgraph->nodes[i];

    जर (गाठ->op != GGML_OP_UNARY && गाठ->op != GGML_OP_SOFT_MAX && गाठ->op != GGML_OP_ARGSORT) {
        परत 0;
    }

    ggml_sycl_topk_moe_args args;
    जर (!ggml_sycl_topk_moe_fusion(cgraph, i, args)) {
        परत 0;
    }

    // this kernel implements the no-bias path only; decline anything with a routing bias
    जर (args.prob_bias) {
        परत 0;
    }

    स्थिरांक ggml_tensor * logits  = गाठ->src[0];
    ggml_tensor *       weights = nullptr;
    ggml_tensor *       ids     = nullptr;
    स्थिरांक ggml_tensor * clamp   = nullptr;
    स्थिरांक ggml_tensor * scale   = nullptr;

    std::vector<ggml_op> ops;
    पूर्णांक                  out_nodes[2];

    जर (!args.delayed_softmax) {
        स्थिरांक ggml_op gating_op = args.sigmoid ? GGML_OP_UNARY : GGML_OP_SOFT_MAX;
        ops.insert(ops.end(), { gating_op, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
        out_nodes[0] = i + 3;
        ids          = cgraph->nodes[i + 3];

        जर (args.norm) {
            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
            clamp = cgraph->nodes[i + (पूर्णांक) ops.size() - 3];
        }
        जर (args.scale) {
            ops.insert(ops.end(), { GGML_OP_SCALE });
            scale = cgraph->nodes[i + (पूर्णांक) ops.size() - 1];
        }

        weights      = cgraph->nodes[i + (पूर्णांक) ops.size() - 1];
        out_nodes[1] = i + (पूर्णांक) ops.size() - 1;

        जर (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(गाठ, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (पूर्णांक) ops.size(), out_nodes, 2, /*is_topk_moe=*/सत्य)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            परत (पूर्णांक) ops.size() - 1;
        }
    } अन्यथा जर (!args.norm && !args.prob_bias) {
        // gpt-oss style: argsort -> view -> get_rows -> reshape -> softmax -> reshape, no norm/bias
        ops.insert(ops.end(),
                   { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                     GGML_OP_RESHAPE });
        weights                     = cgraph->nodes[i + 5];
        ids                         = cgraph->nodes[i + 1];
        स्थिरांक ggml_tensor * softmax = cgraph->nodes[i + 4];
        out_nodes[0]                = i + 1;
        out_nodes[1]                = i + 5;

        जर (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
            ggml_sycl_should_use_topk_moe(softmax, weights, logits, ids) &&
            ggml_sycl_check_fusion_memory_ranges(cgraph, i, (पूर्णांक) ops.size(), out_nodes, 2, /*is_topk_moe=*/सत्य)) {
            ggml_sycl_op_topk_moe(ctx, logits, weights, ids, clamp, scale, args);
            परत (पूर्णांक) ops.size() - 1;
        }
    }

    परत 0;
}
