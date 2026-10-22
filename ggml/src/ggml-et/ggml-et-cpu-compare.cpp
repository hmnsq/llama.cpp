#ଅନ୍ତର୍ଭାଵ "ggml-et-cpu-compare.h"

#ଅନ୍ତର୍ଭାଵ "ggml-cpu/ggml-cpu-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpu/ops.h"

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <cstdlib>
#ଅନ୍ତର୍ଭାଵ <cstring>

ତାର୍କିକ ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ଅଚଳ ggml_tensor * ଗଣ୍ଠି, ggml_op op) {
    ଯଦି (!ctx || !ଗଣ୍ଠି) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Clear context
    memset(ctx, 0, ଆକାର(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    auto get_tensor_buffer_size = [](ଅଚଳ ggml_tensor * tensor) -> size_t {
        ଯଦି (!tensor) {
            ଫେରନ୍ତୁ 0;
        }

        ଯଦି (tensor->buffer) {
            // Get actual backend buffer size
            size_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);

            // Use the full buffer size to avoid any truncation issues
            ଫେରନ୍ତୁ buffer_size;
        } ଅନ୍ୟଥା {
            // Fallback to logical size if no buffer
            ଫେରନ୍ତୁ ggml_nbytes(tensor);
        }
    };

    ctx->src0_size = get_tensor_buffer_size(ଗଣ୍ଠି->src[0]);
    ctx->src1_size = get_tensor_buffer_size(ଗଣ୍ଠି->src[1]);
    ctx->src2_size = get_tensor_buffer_size(ଗଣ୍ଠି->src[2]);
    ctx->dst_size  = get_tensor_buffer_size(ଗଣ୍ଠି);

    // Allocate CPU buffers for all tensors
    ଯଦି (ctx->src0_size > 0) {
        ctx->cpu_src0_data = malloc(ctx->src0_size);
        ଯଦି (!ctx->cpu_src0_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            ଯାଅ cleanup;
        }
    }

    ଯଦି (ctx->src1_size > 0) {
        ctx->cpu_src1_data = malloc(ctx->src1_size);
        ଯଦି (!ctx->cpu_src1_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            ଯାଅ cleanup;
        }
    }

    ଯଦି (ctx->src2_size > 0) {
        ctx->cpu_src2_data = malloc(ctx->src2_size);
        ଯଦି (!ctx->cpu_src2_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            ଯାଅ cleanup;
        }
    }

    ctx->cpu_dst_data = malloc(ctx->dst_size);
    ଯଦି (!ctx->cpu_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        ଯାଅ cleanup;
    }

    ctx->et_dst_data = malloc(ctx->dst_size);
    ଯଦି (!ctx->et_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        ଯାଅ cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    ଯଦି (ctx->src0_size > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_size = ggml_nbytes(ଗଣ୍ଠି->src[0]);
        ggml_backend_tensor_get(ଗଣ୍ଠି->src[0], ctx->cpu_src0_data, 0, logical_size);
    }
    ଯଦି (ctx->src1_size > 0) {
        size_t logical_size = ggml_nbytes(ଗଣ୍ଠି->src[1]);
        ggml_backend_tensor_get(ଗଣ୍ଠି->src[1], ctx->cpu_src1_data, 0, logical_size);
    }
    ଯଦି (ctx->src2_size > 0) {
        size_t logical_size = ggml_nbytes(ଗଣ୍ଠି->src[2]);
        ggml_backend_tensor_get(ଗଣ୍ଠି->src[2], ctx->cpu_src2_data, 0, logical_size);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_size = ggml_nbytes(ଗଣ୍ଠି);
        ggml_backend_tensor_get(ଗଣ୍ଠି, ctx->cpu_dst_data, 0, logical_size);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    ଯଦି (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        ଯାଅ cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_size   = ggml_tensor_overhead() * 4 + ggml_graph_overhead();  // up to 4 tensors + graph
    ctx_params.mem_buffer = nullptr;
    ctx_params.no_alloc   = ସତ୍ୟ;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    ଯଦି (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        ଯାଅ cleanup;
    }

    // Create CPU tensors with proper context
    ଯଦି (ଗଣ୍ଠି->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, ଗଣ୍ଠି->src[0]->type, GGML_MAX_DIMS, ଗଣ୍ଠି->src[0]->ne);
        ଯଦି (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            ଯାଅ cleanup;
        }
        ctx->cpu_src0->data = ctx->cpu_src0_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src0->nb, ଗଣ୍ଠି->src[0]->nb, ଆକାର(ଗଣ୍ଠି->src[0]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src0->op_params, ଗଣ୍ଠି->src[0]->op_params, ଆକାର(ଗଣ୍ଠି->src[0]->op_params));
    }

    ଯଦି (ଗଣ୍ଠି->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, ଗଣ୍ଠି->src[1]->type, GGML_MAX_DIMS, ଗଣ୍ଠି->src[1]->ne);
        ଯଦି (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            ଯାଅ cleanup;
        }
        ctx->cpu_src1->data = ctx->cpu_src1_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src1->nb, ଗଣ୍ଠି->src[1]->nb, ଆକାର(ଗଣ୍ଠି->src[1]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src1->op_params, ଗଣ୍ଠି->src[1]->op_params, ଆକାର(ଗଣ୍ଠି->src[1]->op_params));
    }

    ଯଦି (ଗଣ୍ଠି->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, ଗଣ୍ଠି->src[2]->type, GGML_MAX_DIMS, ଗଣ୍ଠି->src[2]->ne);
        ଯଦି (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            ଯାଅ cleanup;
        }
        ctx->cpu_src2->data = ctx->cpu_src2_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src2->nb, ଗଣ୍ଠି->src[2]->nb, ଆକାର(ଗଣ୍ଠି->src[2]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src2->op_params, ଗଣ୍ଠି->src[2]->op_params, ଆକାର(ଗଣ୍ଠି->src[2]->op_params));
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ତାର୍କିକ ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ଅଚଳ ggml_tensor *                ଗଣ୍ଠି,
                                           ଅଚଳ ggml_et_cpu_compare_config * config) {
    ଯଦି (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !ଗଣ୍ଠି || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = ଗଣ୍ଠି->op;
    ବାଛ (op) {
        ସ୍ଥିତି GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ADD:
            ctx->cpu_dst = ggml_add(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ROPE:
            {
                ଅଚଳ int32_t * op_params  = (ଅଚଳ int32_t *) ଗଣ୍ଠି->op_params;
                ଅଚଳ int32_t   n_dims     = op_params[1];
                ଅଚଳ int32_t   mode       = op_params[2];
                ଅଚଳ int32_t   n_ctx_orig = op_params[4];
                ଅଚଳ ଦଶମିକ     freq_base  = *((ଅଚଳ ଦଶମିକ *) (op_params + 5));
                ଅଚଳ ଦଶମିକ     freq_scale = *((ଅଚଳ ଦଶମିକ *) (op_params + 6));
                ଅଚଳ ଦଶମିକ     ext_factor = *((ଅଚଳ ଦଶମିକ *) (op_params + 7));
                ଅଚଳ ଦଶମିକ     attn_factor = *((ଅଚଳ ଦଶମିକ *) (op_params + 8));
                ଅଚଳ ଦଶମିକ     beta_fast  = *((ଅଚଳ ଦଶମିକ *) (op_params + 9));
                ଅଚଳ ଦଶମିକ     beta_slow  = *((ଅଚଳ ଦଶମିକ *) (op_params + 10));

                ଯଦି (mode & GGML_ROPE_TYPE_MROPE) {
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା sections[GGML_MROPE_SECTIONS];
                    memcpy(sections, op_params + 11, ଆକାର(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
                                                   ext_factor, attn_factor, beta_fast, beta_slow);
                } ଅନ୍ୟଥା {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
                                                 attn_factor, beta_fast, beta_slow);
                }
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                ଦଶମିକ eps;
                memcpy(&eps, ଗଣ୍ଠି->op_params, ଆକାର(ଦଶମିକ));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(ଗଣ୍ଠି, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SUM_ROWS:
            ctx->cpu_dst = ggml_sum_rows(ctx->ggml_ctx, ctx->cpu_src0);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MEAN:
            ctx->cpu_dst = ggml_mean(ctx->ggml_ctx, ctx->cpu_src0);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CLAMP:
            {
                ଦଶମିକ clamp_min, clamp_max;
                memcpy(&clamp_min, (ଅଚଳ ଦଶମିକ *) ଗଣ୍ଠି->op_params + 0, ଆକାର(ଦଶମିକ));
                memcpy(&clamp_max, (ଅଚଳ ଦଶମିକ *) ଗଣ୍ଠି->op_params + 1, ଆକାର(ଦଶମିକ));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(ଗଣ୍ଠି, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                ଯଦି (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
                ଯଦି (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    ଅଚଳ ଦଶମିକ limit = ggml_get_op_params_f32(ଗଣ୍ଠି, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, limit);
                } ଅନ୍ୟଥା {
                    ctx->cpu_dst = ggml_glu_split(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                ଦଶମିକ scale    = 1.0f;
                ଦଶମିକ max_bias = 0.0f;
                memcpy(&scale, (ଅଚଳ ଦଶମିକ *) ଗଣ୍ଠି->op_params + 0, ଆକାର(ଦଶମିକ));
                memcpy(&max_bias, (ଅଚଳ ଦଶମିକ *) ଗଣ୍ଠି->op_params + 1, ଆକାର(ଦଶମିକ));

                ଯଦି (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } ଅନ୍ୟଥା {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                ଯଦି (ctx->cpu_src2) {
                    ggml_soft_max_add_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GET_ROWS:
            ctx->cpu_dst = ggml_get_rows(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SET_ROWS:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_base = ggml_new_tensor(ctx->ggml_ctx, ଗଣ୍ଠି->type, GGML_MAX_DIMS, ଗଣ୍ଠି->ne);
                ଯଦି (!cpu_dst_base) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
                cpu_dst_base->data = ctx->cpu_dst_data;
                memcpy(cpu_dst_base->nb, ଗଣ୍ଠି->nb, ଆକାର(ଗଣ୍ଠି->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_set_rows(ctx->ggml_ctx, cpu_dst_base, ctx->cpu_src0, ctx->cpu_src1);
            }
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ctx->cpu_dst->data = ctx->cpu_dst_data;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    ଯଦି (op != GGML_OP_CONT) {
        memcpy(ctx->cpu_dst->nb, ଗଣ୍ଠି->nb, ଆକାର(ଗଣ୍ଠି->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_graph = ggml_new_graph_custom(ctx->ggml_ctx, 1, ମିଥ୍ୟା);
    ଯଦି (!ctx->cpu_graph) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ctx->cpu_graph->nodes[0] = ctx->cpu_dst;
    ctx->cpu_graph->n_nodes  = 1;

    // Log input data for debugging if enabled
    ଯଦି (config && config->log_differences) {
        ଯଦି (ctx->cpu_src0_data && ctx->src0_size >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_data)[0],
                           ((uint8_t *) ctx->cpu_src0_data)[1], ((uint8_t *) ctx->cpu_src0_data)[2],
                           ((uint8_t *) ctx->cpu_src0_data)[3]);
        }
        ଯଦି (ctx->cpu_src1_data && ctx->src1_size >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((ଦଶମିକ *) ctx->cpu_src1_data)[0],
                           ((ଦଶମିକ *) ctx->cpu_src1_data)[1], ((ଦଶମିକ *) ctx->cpu_src1_data)[2],
                           ((ଦଶମିକ *) ctx->cpu_src1_data)[3]);
        }
    }

    // Compute using CPU backend
    ggml_status cpu_result = ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph);

    ଯଦି (cpu_result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_result);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // Log output data for debugging if enabled
    ଯଦି (config && config->log_differences && ctx->dst_size >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((ଦଶମିକ *) ctx->cpu_dst_data)[0], ((ଦଶମିକ *) ctx->cpu_dst_data)[1],
                       ((ଦଶମିକ *) ctx->cpu_dst_data)[2], ((ଦଶମିକ *) ctx->cpu_dst_data)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_size = ggml_nbytes(ଗଣ୍ଠି);
    ggml_backend_tensor_get(ଗଣ୍ଠି, ctx->et_dst_data, 0, dst_logical_size);

    ଯଦି (config->log_differences) {
        size_t ଅଙ୍କ_elements = ggml_nelements(ଗଣ୍ଠି);
        size_t max_log      = std::min(ଅଙ୍କ_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        ତାର୍କିକ    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ADD || op == GGML_OP_GLU);
        ଦଶମିକ * cpu_src0_float = is_elementwise ? (ଦଶମିକ *) ctx->cpu_src0_data : nullptr;
        ଦଶମିକ * cpu_src1_float = is_elementwise ? (ଦଶମିକ *) ctx->cpu_src1_data : nullptr;

        // Helper to get float value from tensor data (handles f16 and f32)
        auto get_float = [](ଅଚଳ ଶୂନ୍ୟ * data, size_t idx, ggml_type type) -> ଦଶମିକ {
            ଯଦି (type == GGML_TYPE_F16) {
                ଅଚଳ ggml_fp16_t * fp16_data = (ଅଚଳ ggml_fp16_t *) data;
                ଫେରନ୍ତୁ ggml_fp16_to_fp32(fp16_data[idx]);
            }

            ଅଚଳ ଦଶମିକ * float_data = (ଅଚଳ ଦଶମିକ *) data;
            ଫେରନ୍ତୁ float_data[idx];
        };

        // Compare all elements but log only the first max_log_elements
        ତାର୍କିକ   matches          = ସତ୍ୟ;
        size_t total_mismatches = 0;

        // First pass: check all elements for mismatches
        ପାଇଁ (size_t i = 0; i < ଅଙ୍କ_elements; i++) {
            ଦଶମିକ cpu_val  = get_float(ctx->cpu_dst_data, i, ଗଣ୍ଠି->type);
            ଦଶମିକ et_val   = get_float(ctx->et_dst_data, i, ଗଣ୍ଠି->type);
            ଦଶମିକ diff     = fabsf(cpu_val - et_val);
            ଦଶମିକ rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            ଯଦି (rel_diff > config->tolerance) {
                matches = ମିଥ୍ୟା;
                total_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        ପାଇଁ (size_t i = 0; i < max_log; i++) {
            ଦଶମିକ cpu_val = get_float(ctx->cpu_dst_data, i, ଗଣ୍ଠି->type);
            ଦଶମିକ et_val  = get_float(ctx->et_dst_data, i, ଗଣ୍ଠି->type);
            ଦଶମିକ diff    = fabsf(cpu_val - et_val);

            ଯଦି (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } ଅନ୍ୟଥା ଯଦି (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } ଅନ୍ୟଥା {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        ଯଦି (ଅଙ୍କ_elements > max_log) {
            size_t mid     = ଅଙ୍କ_elements / 2;
            size_t end     = ଅଙ୍କ_elements - 1;
            ଦଶମିକ  cpu_mid = get_float(ctx->cpu_dst_data, mid, ଗଣ୍ଠି->type);
            ଦଶମିକ  et_mid  = get_float(ctx->et_dst_data, mid, ଗଣ୍ଠି->type);
            ଦଶମିକ  cpu_end = get_float(ctx->cpu_dst_data, end, ଗଣ୍ଠି->type);
            ଦଶମିକ  et_end  = get_float(ctx->et_dst_data, end, ଗଣ୍ଠି->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_end, et_end);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       ଅଙ୍କ_elements - total_mismatches, ଅଙ୍କ_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    ଯଦି (config->use_cpu_result) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_size = ggml_nbytes(ଗଣ୍ଠି);
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(ଗଣ୍ଠି), ctx->cpu_dst_data, 0, dst_logical_size);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ଶୂନ୍ୟ ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    ଯଦି (!ctx) {
        ଫେରନ୍ତୁ;
    }

    ଯଦି (ctx->cpu_src0_data) {
        ମୁକ୍ତ(ctx->cpu_src0_data);
        ctx->cpu_src0_data = nullptr;
    }
    ଯଦି (ctx->cpu_src1_data) {
        ମୁକ୍ତ(ctx->cpu_src1_data);
        ctx->cpu_src1_data = nullptr;
    }
    ଯଦି (ctx->cpu_src2_data) {
        ମୁକ୍ତ(ctx->cpu_src2_data);
        ctx->cpu_src2_data = nullptr;
    }
    ଯଦି (ctx->cpu_dst_data) {
        ମୁକ୍ତ(ctx->cpu_dst_data);
        ctx->cpu_dst_data = nullptr;
    }
    ଯଦି (ctx->et_dst_data) {
        ମୁକ୍ତ(ctx->et_dst_data);
        ctx->et_dst_data = nullptr;
    }

    ଯଦି (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }

    ଯଦି (ctx->cpu_backend) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = nullptr;
    }

    // Clear pointers
    ctx->cpu_src0  = nullptr;
    ctx->cpu_src1  = nullptr;
    ctx->cpu_src2  = nullptr;
    ctx->cpu_dst   = nullptr;
    ctx->cpu_graph = nullptr;
}
