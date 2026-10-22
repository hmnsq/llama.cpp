#અન્તર્ભાવ "ggml-et-cpu-compare.h"

#અન્તર્ભાવ "ggml-cpu/ggml-cpu-impl.h"
#અન્તર્ભાવ "ggml-cpu/ops.h"

#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <cmath>
#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <cstring>

તાર્કિક ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, અચલ ggml_tensor * ગાંઠ, ggml_op op) {
    જો (!ctx || !ગાંઠ) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        પાછા ખોટું;
    }

    // Clear context
    memset(ctx, 0, કદ(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    auto get_tensor_buffer_size = [](અચલ ggml_tensor * tensor) -> size_t {
        જો (!tensor) {
            પાછા 0;
        }

        જો (tensor->buffer) {
            // Get actual backend buffer size
            size_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);

            // Use the full buffer size to avoid any truncation issues
            પાછા buffer_size;
        } નહીંતર {
            // Fallback to logical size if no buffer
            પાછા ggml_nbytes(tensor);
        }
    };

    ctx->src0_size = get_tensor_buffer_size(ગાંઠ->src[0]);
    ctx->src1_size = get_tensor_buffer_size(ગાંઠ->src[1]);
    ctx->src2_size = get_tensor_buffer_size(ગાંઠ->src[2]);
    ctx->dst_size  = get_tensor_buffer_size(ગાંઠ);

    // Allocate CPU buffers for all tensors
    જો (ctx->src0_size > 0) {
        ctx->cpu_src0_data = malloc(ctx->src0_size);
        જો (!ctx->cpu_src0_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            જાઓ cleanup;
        }
    }

    જો (ctx->src1_size > 0) {
        ctx->cpu_src1_data = malloc(ctx->src1_size);
        જો (!ctx->cpu_src1_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            જાઓ cleanup;
        }
    }

    જો (ctx->src2_size > 0) {
        ctx->cpu_src2_data = malloc(ctx->src2_size);
        જો (!ctx->cpu_src2_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            જાઓ cleanup;
        }
    }

    ctx->cpu_dst_data = malloc(ctx->dst_size);
    જો (!ctx->cpu_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        જાઓ cleanup;
    }

    ctx->et_dst_data = malloc(ctx->dst_size);
    જો (!ctx->et_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        જાઓ cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    જો (ctx->src0_size > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_size = ggml_nbytes(ગાંઠ->src[0]);
        ggml_backend_tensor_get(ગાંઠ->src[0], ctx->cpu_src0_data, 0, logical_size);
    }
    જો (ctx->src1_size > 0) {
        size_t logical_size = ggml_nbytes(ગાંઠ->src[1]);
        ggml_backend_tensor_get(ગાંઠ->src[1], ctx->cpu_src1_data, 0, logical_size);
    }
    જો (ctx->src2_size > 0) {
        size_t logical_size = ggml_nbytes(ગાંઠ->src[2]);
        ggml_backend_tensor_get(ગાંઠ->src[2], ctx->cpu_src2_data, 0, logical_size);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_size = ggml_nbytes(ગાંઠ);
        ggml_backend_tensor_get(ગાંઠ, ctx->cpu_dst_data, 0, logical_size);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    જો (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        જાઓ cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_size   = ggml_tensor_overhead() * 4 + ggml_graph_overhead();  // up to 4 tensors + graph
    ctx_params.mem_buffer = nullptr;
    ctx_params.no_alloc   = સાચું;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    જો (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        જાઓ cleanup;
    }

    // Create CPU tensors with proper context
    જો (ગાંઠ->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, ગાંઠ->src[0]->type, GGML_MAX_DIMS, ગાંઠ->src[0]->ne);
        જો (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            જાઓ cleanup;
        }
        ctx->cpu_src0->data = ctx->cpu_src0_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src0->nb, ગાંઠ->src[0]->nb, કદ(ગાંઠ->src[0]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src0->op_params, ગાંઠ->src[0]->op_params, કદ(ગાંઠ->src[0]->op_params));
    }

    જો (ગાંઠ->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, ગાંઠ->src[1]->type, GGML_MAX_DIMS, ગાંઠ->src[1]->ne);
        જો (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            જાઓ cleanup;
        }
        ctx->cpu_src1->data = ctx->cpu_src1_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src1->nb, ગાંઠ->src[1]->nb, કદ(ગાંઠ->src[1]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src1->op_params, ગાંઠ->src[1]->op_params, કદ(ગાંઠ->src[1]->op_params));
    }

    જો (ગાંઠ->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, ગાંઠ->src[2]->type, GGML_MAX_DIMS, ગાંઠ->src[2]->ne);
        જો (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            જાઓ cleanup;
        }
        ctx->cpu_src2->data = ctx->cpu_src2_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src2->nb, ગાંઠ->src[2]->nb, કદ(ગાંઠ->src[2]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src2->op_params, ગાંઠ->src[2]->op_params, કદ(ગાંઠ->src[2]->op_params));
    }

    પાછા સાચું;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    પાછા ખોટું;
}

તાર્કિક ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           અચલ ggml_tensor *                ગાંઠ,
                                           અચલ ggml_et_cpu_compare_config * config) {
    જો (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !ગાંઠ || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        પાછા ખોટું;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = ગાંઠ->op;
    પસંદગી (op) {
        સ્થિતિ GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            તોડો;
        સ્થિતિ GGML_OP_ADD:
            ctx->cpu_dst = ggml_add(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            તોડો;
        સ્થિતિ GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            તોડો;
        સ્થિતિ GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            તોડો;
        સ્થિતિ GGML_OP_ROPE:
            {
                અચલ int32_t * op_params  = (અચલ int32_t *) ગાંઠ->op_params;
                અચલ int32_t   n_dims     = op_params[1];
                અચલ int32_t   mode       = op_params[2];
                અચલ int32_t   n_ctx_orig = op_params[4];
                અચલ દશાંશ     freq_base  = *((અચલ દશાંશ *) (op_params + 5));
                અચલ દશાંશ     freq_scale = *((અચલ દશાંશ *) (op_params + 6));
                અચલ દશાંશ     ext_factor = *((અચલ દશાંશ *) (op_params + 7));
                અચલ દશાંશ     attn_factor = *((અચલ દશાંશ *) (op_params + 8));
                અચલ દશાંશ     beta_fast  = *((અચલ દશાંશ *) (op_params + 9));
                અચલ દશાંશ     beta_slow  = *((અચલ દશાંશ *) (op_params + 10));

                જો (mode & GGML_ROPE_TYPE_MROPE) {
                    પૂર્ણાંક sections[GGML_MROPE_SECTIONS];
                    memcpy(sections, op_params + 11, કદ(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
                                                   ext_factor, attn_factor, beta_fast, beta_slow);
                } નહીંતર {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
                                                 attn_factor, beta_fast, beta_slow);
                }
            }
            તોડો;
        સ્થિતિ GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                દશાંશ eps;
                memcpy(&eps, ગાંઠ->op_params, કદ(દશાંશ));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            તોડો;
        સ્થિતિ GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            તોડો;
        સ્થિતિ GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(ગાંઠ, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            તોડો;
        સ્થિતિ GGML_OP_SUM_ROWS:
            ctx->cpu_dst = ggml_sum_rows(ctx->ggml_ctx, ctx->cpu_src0);
            તોડો;
        સ્થિતિ GGML_OP_MEAN:
            ctx->cpu_dst = ggml_mean(ctx->ggml_ctx, ctx->cpu_src0);
            તોડો;
        સ્થિતિ GGML_OP_CLAMP:
            {
                દશાંશ clamp_min, clamp_max;
                memcpy(&clamp_min, (અચલ દશાંશ *) ગાંઠ->op_params + 0, કદ(દશાંશ));
                memcpy(&clamp_max, (અચલ દશાંશ *) ગાંઠ->op_params + 1, કદ(દશાંશ));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            તોડો;
        સ્થિતિ GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(ગાંઠ, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                જો (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    પાછા ખોટું;
                }
                જો (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    અચલ દશાંશ limit = ggml_get_op_params_f32(ગાંઠ, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, limit);
                } નહીંતર {
                    ctx->cpu_dst = ggml_glu_split(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            તોડો;
        સ્થિતિ GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                દશાંશ scale    = 1.0f;
                દશાંશ max_bias = 0.0f;
                memcpy(&scale, (અચલ દશાંશ *) ગાંઠ->op_params + 0, કદ(દશાંશ));
                memcpy(&max_bias, (અચલ દશાંશ *) ગાંઠ->op_params + 1, કદ(દશાંશ));

                જો (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } નહીંતર {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                જો (ctx->cpu_src2) {
                    ggml_soft_max_add_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            તોડો;
        સ્થિતિ GGML_OP_GET_ROWS:
            ctx->cpu_dst = ggml_get_rows(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            તોડો;
        સ્થિતિ GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            તોડો;
        સ્થિતિ GGML_OP_SET_ROWS:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_base = ggml_new_tensor(ctx->ggml_ctx, ગાંઠ->type, GGML_MAX_DIMS, ગાંઠ->ne);
                જો (!cpu_dst_base) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    પાછા ખોટું;
                }
                cpu_dst_base->data = ctx->cpu_dst_data;
                memcpy(cpu_dst_base->nb, ગાંઠ->nb, કદ(ગાંઠ->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_set_rows(ctx->ggml_ctx, cpu_dst_base, ctx->cpu_src0, ctx->cpu_src1);
            }
            તોડો;
        મૂળભૂત:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            પાછા ખોટું;
    }

    જો (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        પાછા ખોટું;
    }

    ctx->cpu_dst->data = ctx->cpu_dst_data;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    જો (op != GGML_OP_CONT) {
        memcpy(ctx->cpu_dst->nb, ગાંઠ->nb, કદ(ગાંઠ->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_graph = ggml_new_graph_custom(ctx->ggml_ctx, 1, ખોટું);
    જો (!ctx->cpu_graph) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        પાછા ખોટું;
    }
    ctx->cpu_graph->nodes[0] = ctx->cpu_dst;
    ctx->cpu_graph->n_nodes  = 1;

    // Log input data for debugging if enabled
    જો (config && config->log_differences) {
        જો (ctx->cpu_src0_data && ctx->src0_size >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_data)[0],
                           ((uint8_t *) ctx->cpu_src0_data)[1], ((uint8_t *) ctx->cpu_src0_data)[2],
                           ((uint8_t *) ctx->cpu_src0_data)[3]);
        }
        જો (ctx->cpu_src1_data && ctx->src1_size >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((દશાંશ *) ctx->cpu_src1_data)[0],
                           ((દશાંશ *) ctx->cpu_src1_data)[1], ((દશાંશ *) ctx->cpu_src1_data)[2],
                           ((દશાંશ *) ctx->cpu_src1_data)[3]);
        }
    }

    // Compute using CPU backend
    ggml_status cpu_result = ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph);

    જો (cpu_result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_result);
        પાછા ખોટું;
    }

    // Log output data for debugging if enabled
    જો (config && config->log_differences && ctx->dst_size >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((દશાંશ *) ctx->cpu_dst_data)[0], ((દશાંશ *) ctx->cpu_dst_data)[1],
                       ((દશાંશ *) ctx->cpu_dst_data)[2], ((દશાંશ *) ctx->cpu_dst_data)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_size = ggml_nbytes(ગાંઠ);
    ggml_backend_tensor_get(ગાંઠ, ctx->et_dst_data, 0, dst_logical_size);

    જો (config->log_differences) {
        size_t અંક_elements = ggml_nelements(ગાંઠ);
        size_t max_log      = std::min(અંક_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        તાર્કિક    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ADD || op == GGML_OP_GLU);
        દશાંશ * cpu_src0_float = is_elementwise ? (દશાંશ *) ctx->cpu_src0_data : nullptr;
        દશાંશ * cpu_src1_float = is_elementwise ? (દશાંશ *) ctx->cpu_src1_data : nullptr;

        // Helper to get float value from tensor data (handles f16 and f32)
        auto get_float = [](અચલ શૂન્ય * data, size_t idx, ggml_type type) -> દશાંશ {
            જો (type == GGML_TYPE_F16) {
                અચલ ggml_fp16_t * fp16_data = (અચલ ggml_fp16_t *) data;
                પાછા ggml_fp16_to_fp32(fp16_data[idx]);
            }

            અચલ દશાંશ * float_data = (અચલ દશાંશ *) data;
            પાછા float_data[idx];
        };

        // Compare all elements but log only the first max_log_elements
        તાર્કિક   matches          = સાચું;
        size_t total_mismatches = 0;

        // First pass: check all elements for mismatches
        માટે (size_t i = 0; i < અંક_elements; i++) {
            દશાંશ cpu_val  = get_float(ctx->cpu_dst_data, i, ગાંઠ->type);
            દશાંશ et_val   = get_float(ctx->et_dst_data, i, ગાંઠ->type);
            દશાંશ diff     = fabsf(cpu_val - et_val);
            દશાંશ rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            જો (rel_diff > config->tolerance) {
                matches = ખોટું;
                total_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        માટે (size_t i = 0; i < max_log; i++) {
            દશાંશ cpu_val = get_float(ctx->cpu_dst_data, i, ગાંઠ->type);
            દશાંશ et_val  = get_float(ctx->et_dst_data, i, ગાંઠ->type);
            દશાંશ diff    = fabsf(cpu_val - et_val);

            જો (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } નહીંતર જો (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } નહીંતર {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        જો (અંક_elements > max_log) {
            size_t mid     = અંક_elements / 2;
            size_t end     = અંક_elements - 1;
            દશાંશ  cpu_mid = get_float(ctx->cpu_dst_data, mid, ગાંઠ->type);
            દશાંશ  et_mid  = get_float(ctx->et_dst_data, mid, ગાંઠ->type);
            દશાંશ  cpu_end = get_float(ctx->cpu_dst_data, end, ગાંઠ->type);
            દશાંશ  et_end  = get_float(ctx->et_dst_data, end, ગાંઠ->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_end, et_end);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       અંક_elements - total_mismatches, અંક_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    જો (config->use_cpu_result) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_size = ggml_nbytes(ગાંઠ);
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(ગાંઠ), ctx->cpu_dst_data, 0, dst_logical_size);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    પાછા સાચું;
}

શૂન્ય ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    જો (!ctx) {
        પાછા;
    }

    જો (ctx->cpu_src0_data) {
        મુક્ત(ctx->cpu_src0_data);
        ctx->cpu_src0_data = nullptr;
    }
    જો (ctx->cpu_src1_data) {
        મુક્ત(ctx->cpu_src1_data);
        ctx->cpu_src1_data = nullptr;
    }
    જો (ctx->cpu_src2_data) {
        મુક્ત(ctx->cpu_src2_data);
        ctx->cpu_src2_data = nullptr;
    }
    જો (ctx->cpu_dst_data) {
        મુક્ત(ctx->cpu_dst_data);
        ctx->cpu_dst_data = nullptr;
    }
    જો (ctx->et_dst_data) {
        મુક્ત(ctx->et_dst_data);
        ctx->et_dst_data = nullptr;
    }

    જો (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }

    જો (ctx->cpu_backend) {
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
