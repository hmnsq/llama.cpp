#ಅನ್ತರ್ಭಾವ "ggml-et-cpu-compare.h"

#ಅನ್ತರ್ಭಾವ "ggml-cpu/ggml-cpu-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu/ops.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cstring>

ತಾರ್ಕಿಕ ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ಧ್ರುವ ggml_tensor * ಗಂಟು, ggml_op op) {
    ಯದಿ (!ctx || !ಗಂಟು) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // Clear context
    memset(ctx, 0, ಪರಿಮಾಣ(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    auto get_tensor_buffer_size = [](ಧ್ರುವ ggml_tensor * tensor) -> size_t {
        ಯದಿ (!tensor) {
            ಪ್ರತ್ಯಾಗಮ 0;
        }

        ಯದಿ (tensor->buffer) {
            // Get actual backend buffer size
            size_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);

            // Use the full buffer size to avoid any truncation issues
            ಪ್ರತ್ಯಾಗಮ buffer_size;
        } ಅನ್ಯಥಾ {
            // Fallback to logical size if no buffer
            ಪ್ರತ್ಯಾಗಮ ggml_nbytes(tensor);
        }
    };

    ctx->src0_size = get_tensor_buffer_size(ಗಂಟು->src[0]);
    ctx->src1_size = get_tensor_buffer_size(ಗಂಟು->src[1]);
    ctx->src2_size = get_tensor_buffer_size(ಗಂಟು->src[2]);
    ctx->dst_size  = get_tensor_buffer_size(ಗಂಟು);

    // Allocate CPU buffers for all tensors
    ಯದಿ (ctx->src0_size > 0) {
        ctx->cpu_src0_data = malloc(ctx->src0_size);
        ಯದಿ (!ctx->cpu_src0_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            ಗಚ್ಛ cleanup;
        }
    }

    ಯದಿ (ctx->src1_size > 0) {
        ctx->cpu_src1_data = malloc(ctx->src1_size);
        ಯದಿ (!ctx->cpu_src1_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            ಗಚ್ಛ cleanup;
        }
    }

    ಯದಿ (ctx->src2_size > 0) {
        ctx->cpu_src2_data = malloc(ctx->src2_size);
        ಯದಿ (!ctx->cpu_src2_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            ಗಚ್ಛ cleanup;
        }
    }

    ctx->cpu_dst_data = malloc(ctx->dst_size);
    ಯದಿ (!ctx->cpu_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        ಗಚ್ಛ cleanup;
    }

    ctx->et_dst_data = malloc(ctx->dst_size);
    ಯದಿ (!ctx->et_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        ಗಚ್ಛ cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    ಯದಿ (ctx->src0_size > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_size = ggml_nbytes(ಗಂಟು->src[0]);
        ggml_backend_tensor_get(ಗಂಟು->src[0], ctx->cpu_src0_data, 0, logical_size);
    }
    ಯದಿ (ctx->src1_size > 0) {
        size_t logical_size = ggml_nbytes(ಗಂಟು->src[1]);
        ggml_backend_tensor_get(ಗಂಟು->src[1], ctx->cpu_src1_data, 0, logical_size);
    }
    ಯದಿ (ctx->src2_size > 0) {
        size_t logical_size = ggml_nbytes(ಗಂಟು->src[2]);
        ggml_backend_tensor_get(ಗಂಟು->src[2], ctx->cpu_src2_data, 0, logical_size);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_size = ggml_nbytes(ಗಂಟು);
        ggml_backend_tensor_get(ಗಂಟು, ctx->cpu_dst_data, 0, logical_size);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    ಯದಿ (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        ಗಚ್ಛ cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_size   = ggml_tensor_overhead() * 4 + ggml_graph_overhead();  // up to 4 tensors + graph
    ctx_params.mem_buffer = nullptr;
    ctx_params.no_alloc   = ಸತ್ಯ;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    ಯದಿ (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        ಗಚ್ಛ cleanup;
    }

    // Create CPU tensors with proper context
    ಯದಿ (ಗಂಟು->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, ಗಂಟು->src[0]->type, GGML_MAX_DIMS, ಗಂಟು->src[0]->ne);
        ಯದಿ (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            ಗಚ್ಛ cleanup;
        }
        ctx->cpu_src0->data = ctx->cpu_src0_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src0->nb, ಗಂಟು->src[0]->nb, ಪರಿಮಾಣ(ಗಂಟು->src[0]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src0->op_params, ಗಂಟು->src[0]->op_params, ಪರಿಮಾಣ(ಗಂಟು->src[0]->op_params));
    }

    ಯದಿ (ಗಂಟು->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, ಗಂಟು->src[1]->type, GGML_MAX_DIMS, ಗಂಟು->src[1]->ne);
        ಯದಿ (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            ಗಚ್ಛ cleanup;
        }
        ctx->cpu_src1->data = ctx->cpu_src1_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src1->nb, ಗಂಟು->src[1]->nb, ಪರಿಮಾಣ(ಗಂಟು->src[1]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src1->op_params, ಗಂಟು->src[1]->op_params, ಪರಿಮಾಣ(ಗಂಟು->src[1]->op_params));
    }

    ಯದಿ (ಗಂಟು->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, ಗಂಟು->src[2]->type, GGML_MAX_DIMS, ಗಂಟು->src[2]->ne);
        ಯದಿ (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            ಗಚ್ಛ cleanup;
        }
        ctx->cpu_src2->data = ctx->cpu_src2_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src2->nb, ಗಂಟು->src[2]->nb, ಪರಿಮಾಣ(ಗಂಟು->src[2]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src2->op_params, ಗಂಟು->src[2]->op_params, ಪರಿಮಾಣ(ಗಂಟು->src[2]->op_params));
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ಧ್ರುವ ggml_tensor *                ಗಂಟು,
                                           ಧ್ರುವ ggml_et_cpu_compare_config * config) {
    ಯದಿ (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !ಗಂಟು || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = ಗಂಟು->op;
    ವಿಕಲ್ಪ (op) {
        ಪಕ್ಷ GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ADD:
            ctx->cpu_dst = ggml_add(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_ROPE:
            {
                ಧ್ರುವ int32_t * op_params  = (ಧ್ರುವ int32_t *) ಗಂಟು->op_params;
                ಧ್ರುವ int32_t   n_dims     = op_params[1];
                ಧ್ರುವ int32_t   mode       = op_params[2];
                ಧ್ರುವ int32_t   n_ctx_orig = op_params[4];
                ಧ್ರುವ ಭಿನ್ನ     freq_base  = *((ಧ್ರುವ ಭಿನ್ನ *) (op_params + 5));
                ಧ್ರುವ ಭಿನ್ನ     freq_scale = *((ಧ್ರುವ ಭಿನ್ನ *) (op_params + 6));
                ಧ್ರುವ ಭಿನ್ನ     ext_factor = *((ಧ್ರುವ ಭಿನ್ನ *) (op_params + 7));
                ಧ್ರುವ ಭಿನ್ನ     attn_factor = *((ಧ್ರುವ ಭಿನ್ನ *) (op_params + 8));
                ಧ್ರುವ ಭಿನ್ನ     beta_fast  = *((ಧ್ರುವ ಭಿನ್ನ *) (op_params + 9));
                ಧ್ರುವ ಭಿನ್ನ     beta_slow  = *((ಧ್ರುವ ಭಿನ್ನ *) (op_params + 10));

                ಯದಿ (mode & GGML_ROPE_TYPE_MROPE) {
                    ಪೂರ್ಣಾಙ್ಕ sections[GGML_MROPE_SECTIONS];
                    memcpy(sections, op_params + 11, ಪರಿಮಾಣ(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
                                                   ext_factor, attn_factor, beta_fast, beta_slow);
                } ಅನ್ಯಥಾ {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
                                                 attn_factor, beta_fast, beta_slow);
                }
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                ಭಿನ್ನ eps;
                memcpy(&eps, ಗಂಟು->op_params, ಪರಿಮಾಣ(ಭಿನ್ನ));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(ಗಂಟು, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SUM_ROWS:
            ctx->cpu_dst = ggml_sum_rows(ctx->ggml_ctx, ctx->cpu_src0);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_MEAN:
            ctx->cpu_dst = ggml_mean(ctx->ggml_ctx, ctx->cpu_src0);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_CLAMP:
            {
                ಭಿನ್ನ clamp_min, clamp_max;
                memcpy(&clamp_min, (ಧ್ರುವ ಭಿನ್ನ *) ಗಂಟು->op_params + 0, ಪರಿಮಾಣ(ಭಿನ್ನ));
                memcpy(&clamp_max, (ಧ್ರುವ ಭಿನ್ನ *) ಗಂಟು->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(ಗಂಟು, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                ಯದಿ (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                ಯದಿ (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    ಧ್ರುವ ಭಿನ್ನ limit = ggml_get_op_params_f32(ಗಂಟು, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, limit);
                } ಅನ್ಯಥಾ {
                    ctx->cpu_dst = ggml_glu_split(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                ಭಿನ್ನ scale    = 1.0f;
                ಭಿನ್ನ max_bias = 0.0f;
                memcpy(&scale, (ಧ್ರುವ ಭಿನ್ನ *) ಗಂಟು->op_params + 0, ಪರಿಮಾಣ(ಭಿನ್ನ));
                memcpy(&max_bias, (ಧ್ರುವ ಭಿನ್ನ *) ಗಂಟು->op_params + 1, ಪರಿಮಾಣ(ಭಿನ್ನ));

                ಯದಿ (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } ಅನ್ಯಥಾ {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                ಯದಿ (ctx->cpu_src2) {
                    ggml_soft_max_add_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_GET_ROWS:
            ctx->cpu_dst = ggml_get_rows(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_OP_SET_ROWS:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_base = ggml_new_tensor(ctx->ggml_ctx, ಗಂಟು->type, GGML_MAX_DIMS, ಗಂಟು->ne);
                ಯದಿ (!cpu_dst_base) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
                }
                cpu_dst_base->data = ctx->cpu_dst_data;
                memcpy(cpu_dst_base->nb, ಗಂಟು->nb, ಪರಿಮಾಣ(ಗಂಟು->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_set_rows(ctx->ggml_ctx, cpu_dst_base, ctx->cpu_src0, ctx->cpu_src1);
            }
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ctx->cpu_dst->data = ctx->cpu_dst_data;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    ಯದಿ (op != GGML_OP_CONT) {
        memcpy(ctx->cpu_dst->nb, ಗಂಟು->nb, ಪರಿಮಾಣ(ಗಂಟು->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_graph = ggml_new_graph_custom(ctx->ggml_ctx, 1, ಅಸತ್ಯ);
    ಯದಿ (!ctx->cpu_graph) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ctx->cpu_graph->nodes[0] = ctx->cpu_dst;
    ctx->cpu_graph->n_nodes  = 1;

    // Log input data for debugging if enabled
    ಯದಿ (config && config->log_differences) {
        ಯದಿ (ctx->cpu_src0_data && ctx->src0_size >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_data)[0],
                           ((uint8_t *) ctx->cpu_src0_data)[1], ((uint8_t *) ctx->cpu_src0_data)[2],
                           ((uint8_t *) ctx->cpu_src0_data)[3]);
        }
        ಯದಿ (ctx->cpu_src1_data && ctx->src1_size >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((ಭಿನ್ನ *) ctx->cpu_src1_data)[0],
                           ((ಭಿನ್ನ *) ctx->cpu_src1_data)[1], ((ಭಿನ್ನ *) ctx->cpu_src1_data)[2],
                           ((ಭಿನ್ನ *) ctx->cpu_src1_data)[3]);
        }
    }

    // Compute using CPU backend
    ggml_status cpu_result = ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph);

    ಯದಿ (cpu_result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_result);
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // Log output data for debugging if enabled
    ಯದಿ (config && config->log_differences && ctx->dst_size >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((ಭಿನ್ನ *) ctx->cpu_dst_data)[0], ((ಭಿನ್ನ *) ctx->cpu_dst_data)[1],
                       ((ಭಿನ್ನ *) ctx->cpu_dst_data)[2], ((ಭಿನ್ನ *) ctx->cpu_dst_data)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_size = ggml_nbytes(ಗಂಟು);
    ggml_backend_tensor_get(ಗಂಟು, ctx->et_dst_data, 0, dst_logical_size);

    ಯದಿ (config->log_differences) {
        size_t ಅಂಕೆ_elements = ggml_nelements(ಗಂಟು);
        size_t max_log      = std::min(ಅಂಕೆ_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        ತಾರ್ಕಿಕ    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ADD || op == GGML_OP_GLU);
        ಭಿನ್ನ * cpu_src0_float = is_elementwise ? (ಭಿನ್ನ *) ctx->cpu_src0_data : nullptr;
        ಭಿನ್ನ * cpu_src1_float = is_elementwise ? (ಭಿನ್ನ *) ctx->cpu_src1_data : nullptr;

        // Helper to get float value from tensor data (handles f16 and f32)
        auto get_float = [](ಧ್ರುವ ಶೂನ್ಯ * data, size_t idx, ggml_type type) -> ಭಿನ್ನ {
            ಯದಿ (type == GGML_TYPE_F16) {
                ಧ್ರುವ ggml_fp16_t * fp16_data = (ಧ್ರುವ ggml_fp16_t *) data;
                ಪ್ರತ್ಯಾಗಮ ggml_fp16_to_fp32(fp16_data[idx]);
            }

            ಧ್ರುವ ಭಿನ್ನ * float_data = (ಧ್ರುವ ಭಿನ್ನ *) data;
            ಪ್ರತ್ಯಾಗಮ float_data[idx];
        };

        // Compare all elements but log only the first max_log_elements
        ತಾರ್ಕಿಕ   matches          = ಸತ್ಯ;
        size_t total_mismatches = 0;

        // First pass: check all elements for mismatches
        ಪ್ರತಿ (size_t i = 0; i < ಅಂಕೆ_elements; i++) {
            ಭಿನ್ನ cpu_val  = get_float(ctx->cpu_dst_data, i, ಗಂಟು->type);
            ಭಿನ್ನ et_val   = get_float(ctx->et_dst_data, i, ಗಂಟು->type);
            ಭಿನ್ನ diff     = fabsf(cpu_val - et_val);
            ಭಿನ್ನ rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            ಯದಿ (rel_diff > config->tolerance) {
                matches = ಅಸತ್ಯ;
                total_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        ಪ್ರತಿ (size_t i = 0; i < max_log; i++) {
            ಭಿನ್ನ cpu_val = get_float(ctx->cpu_dst_data, i, ಗಂಟು->type);
            ಭಿನ್ನ et_val  = get_float(ctx->et_dst_data, i, ಗಂಟು->type);
            ಭಿನ್ನ diff    = fabsf(cpu_val - et_val);

            ಯದಿ (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } ಅನ್ಯಥಾ ಯದಿ (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } ಅನ್ಯಥಾ {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        ಯದಿ (ಅಂಕೆ_elements > max_log) {
            size_t mid     = ಅಂಕೆ_elements / 2;
            size_t end     = ಅಂಕೆ_elements - 1;
            ಭಿನ್ನ  cpu_mid = get_float(ctx->cpu_dst_data, mid, ಗಂಟು->type);
            ಭಿನ್ನ  et_mid  = get_float(ctx->et_dst_data, mid, ಗಂಟು->type);
            ಭಿನ್ನ  cpu_end = get_float(ctx->cpu_dst_data, end, ಗಂಟು->type);
            ಭಿನ್ನ  et_end  = get_float(ctx->et_dst_data, end, ಗಂಟು->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_end, et_end);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       ಅಂಕೆ_elements - total_mismatches, ಅಂಕೆ_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    ಯದಿ (config->use_cpu_result) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_size = ggml_nbytes(ಗಂಟು);
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(ಗಂಟು), ctx->cpu_dst_data, 0, dst_logical_size);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

ಶೂನ್ಯ ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    ಯದಿ (!ctx) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಯದಿ (ctx->cpu_src0_data) {
        ಮುಕ್ತ(ctx->cpu_src0_data);
        ctx->cpu_src0_data = nullptr;
    }
    ಯದಿ (ctx->cpu_src1_data) {
        ಮುಕ್ತ(ctx->cpu_src1_data);
        ctx->cpu_src1_data = nullptr;
    }
    ಯದಿ (ctx->cpu_src2_data) {
        ಮುಕ್ತ(ctx->cpu_src2_data);
        ctx->cpu_src2_data = nullptr;
    }
    ಯದಿ (ctx->cpu_dst_data) {
        ಮುಕ್ತ(ctx->cpu_dst_data);
        ctx->cpu_dst_data = nullptr;
    }
    ಯದಿ (ctx->et_dst_data) {
        ಮುಕ್ತ(ctx->et_dst_data);
        ctx->et_dst_data = nullptr;
    }

    ಯದಿ (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }

    ಯದಿ (ctx->cpu_backend) {
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
