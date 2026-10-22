#അന്തര്ഭാവ "ggml-et-cpu-compare.h"

#അന്തര്ഭാവ "ggml-cpu/ggml-cpu-impl.h"
#അന്തര്ഭാവ "ggml-cpu/ops.h"

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cmath>
#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <cstring>

തർക്കം ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, സ്ഥിരം ggml_tensor * കണ്ണി, ggml_op op) {
    എങ്കിൽ (!ctx || !കണ്ണി) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        തിരികെ അസത്യം;
    }

    // Clear context
    memset(ctx, 0, വലുപ്പം(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    auto get_tensor_buffer_size = [](സ്ഥിരം ggml_tensor * tensor) -> size_t {
        എങ്കിൽ (!tensor) {
            തിരികെ 0;
        }

        എങ്കിൽ (tensor->buffer) {
            // Get actual backend buffer size
            size_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);

            // Use the full buffer size to avoid any truncation issues
            തിരികെ buffer_size;
        } അല്ലെങ്കിൽ {
            // Fallback to logical size if no buffer
            തിരികെ ggml_nbytes(tensor);
        }
    };

    ctx->src0_size = get_tensor_buffer_size(കണ്ണി->src[0]);
    ctx->src1_size = get_tensor_buffer_size(കണ്ണി->src[1]);
    ctx->src2_size = get_tensor_buffer_size(കണ്ണി->src[2]);
    ctx->dst_size  = get_tensor_buffer_size(കണ്ണി);

    // Allocate CPU buffers for all tensors
    എങ്കിൽ (ctx->src0_size > 0) {
        ctx->cpu_src0_data = malloc(ctx->src0_size);
        എങ്കിൽ (!ctx->cpu_src0_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            പോകുക cleanup;
        }
    }

    എങ്കിൽ (ctx->src1_size > 0) {
        ctx->cpu_src1_data = malloc(ctx->src1_size);
        എങ്കിൽ (!ctx->cpu_src1_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            പോകുക cleanup;
        }
    }

    എങ്കിൽ (ctx->src2_size > 0) {
        ctx->cpu_src2_data = malloc(ctx->src2_size);
        എങ്കിൽ (!ctx->cpu_src2_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            പോകുക cleanup;
        }
    }

    ctx->cpu_dst_data = malloc(ctx->dst_size);
    എങ്കിൽ (!ctx->cpu_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        പോകുക cleanup;
    }

    ctx->et_dst_data = malloc(ctx->dst_size);
    എങ്കിൽ (!ctx->et_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        പോകുക cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    എങ്കിൽ (ctx->src0_size > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_size = ggml_nbytes(കണ്ണി->src[0]);
        ggml_backend_tensor_get(കണ്ണി->src[0], ctx->cpu_src0_data, 0, logical_size);
    }
    എങ്കിൽ (ctx->src1_size > 0) {
        size_t logical_size = ggml_nbytes(കണ്ണി->src[1]);
        ggml_backend_tensor_get(കണ്ണി->src[1], ctx->cpu_src1_data, 0, logical_size);
    }
    എങ്കിൽ (ctx->src2_size > 0) {
        size_t logical_size = ggml_nbytes(കണ്ണി->src[2]);
        ggml_backend_tensor_get(കണ്ണി->src[2], ctx->cpu_src2_data, 0, logical_size);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_size = ggml_nbytes(കണ്ണി);
        ggml_backend_tensor_get(കണ്ണി, ctx->cpu_dst_data, 0, logical_size);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    എങ്കിൽ (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        പോകുക cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_size   = ggml_tensor_overhead() * 4 + ggml_graph_overhead();  // up to 4 tensors + graph
    ctx_params.mem_buffer = nullptr;
    ctx_params.no_alloc   = സത്യം;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    എങ്കിൽ (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        പോകുക cleanup;
    }

    // Create CPU tensors with proper context
    എങ്കിൽ (കണ്ണി->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, കണ്ണി->src[0]->type, GGML_MAX_DIMS, കണ്ണി->src[0]->ne);
        എങ്കിൽ (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            പോകുക cleanup;
        }
        ctx->cpu_src0->data = ctx->cpu_src0_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src0->nb, കണ്ണി->src[0]->nb, വലുപ്പം(കണ്ണി->src[0]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src0->op_params, കണ്ണി->src[0]->op_params, വലുപ്പം(കണ്ണി->src[0]->op_params));
    }

    എങ്കിൽ (കണ്ണി->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, കണ്ണി->src[1]->type, GGML_MAX_DIMS, കണ്ണി->src[1]->ne);
        എങ്കിൽ (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            പോകുക cleanup;
        }
        ctx->cpu_src1->data = ctx->cpu_src1_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src1->nb, കണ്ണി->src[1]->nb, വലുപ്പം(കണ്ണി->src[1]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src1->op_params, കണ്ണി->src[1]->op_params, വലുപ്പം(കണ്ണി->src[1]->op_params));
    }

    എങ്കിൽ (കണ്ണി->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, കണ്ണി->src[2]->type, GGML_MAX_DIMS, കണ്ണി->src[2]->ne);
        എങ്കിൽ (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            പോകുക cleanup;
        }
        ctx->cpu_src2->data = ctx->cpu_src2_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src2->nb, കണ്ണി->src[2]->nb, വലുപ്പം(കണ്ണി->src[2]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src2->op_params, കണ്ണി->src[2]->op_params, വലുപ്പം(കണ്ണി->src[2]->op_params));
    }

    തിരികെ സത്യം;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    തിരികെ അസത്യം;
}

തർക്കം ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           സ്ഥിരം ggml_tensor *                കണ്ണി,
                                           സ്ഥിരം ggml_et_cpu_compare_config * config) {
    എങ്കിൽ (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !കണ്ണി || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        തിരികെ അസത്യം;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = കണ്ണി->op;
    തിരഞ്ഞെടുപ്പ് (op) {
        അവസ്ഥ GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_ADD:
            ctx->cpu_dst = ggml_add(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_ROPE:
            {
                സ്ഥിരം int32_t * op_params  = (സ്ഥിരം int32_t *) കണ്ണി->op_params;
                സ്ഥിരം int32_t   n_dims     = op_params[1];
                സ്ഥിരം int32_t   mode       = op_params[2];
                സ്ഥിരം int32_t   n_ctx_orig = op_params[4];
                സ്ഥിരം ദശാംശം     freq_base  = *((സ്ഥിരം ദശാംശം *) (op_params + 5));
                സ്ഥിരം ദശാംശം     freq_scale = *((സ്ഥിരം ദശാംശം *) (op_params + 6));
                സ്ഥിരം ദശാംശം     ext_factor = *((സ്ഥിരം ദശാംശം *) (op_params + 7));
                സ്ഥിരം ദശാംശം     attn_factor = *((സ്ഥിരം ദശാംശം *) (op_params + 8));
                സ്ഥിരം ദശാംശം     beta_fast  = *((സ്ഥിരം ദശാംശം *) (op_params + 9));
                സ്ഥിരം ദശാംശം     beta_slow  = *((സ്ഥിരം ദശാംശം *) (op_params + 10));

                എങ്കിൽ (mode & GGML_ROPE_TYPE_MROPE) {
                    പൂർണ്ണസംഖ്യ sections[GGML_MROPE_SECTIONS];
                    memcpy(sections, op_params + 11, വലുപ്പം(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
                                                   ext_factor, attn_factor, beta_fast, beta_slow);
                } അല്ലെങ്കിൽ {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
                                                 attn_factor, beta_fast, beta_slow);
                }
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                ദശാംശം eps;
                memcpy(&eps, കണ്ണി->op_params, വലുപ്പം(ദശാംശം));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(കണ്ണി, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SUM_ROWS:
            ctx->cpu_dst = ggml_sum_rows(ctx->ggml_ctx, ctx->cpu_src0);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_MEAN:
            ctx->cpu_dst = ggml_mean(ctx->ggml_ctx, ctx->cpu_src0);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CLAMP:
            {
                ദശാംശം clamp_min, clamp_max;
                memcpy(&clamp_min, (സ്ഥിരം ദശാംശം *) കണ്ണി->op_params + 0, വലുപ്പം(ദശാംശം));
                memcpy(&clamp_max, (സ്ഥിരം ദശാംശം *) കണ്ണി->op_params + 1, വലുപ്പം(ദശാംശം));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(കണ്ണി, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                എങ്കിൽ (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    തിരികെ അസത്യം;
                }
                എങ്കിൽ (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    സ്ഥിരം ദശാംശം limit = ggml_get_op_params_f32(കണ്ണി, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, limit);
                } അല്ലെങ്കിൽ {
                    ctx->cpu_dst = ggml_glu_split(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                ദശാംശം scale    = 1.0f;
                ദശാംശം max_bias = 0.0f;
                memcpy(&scale, (സ്ഥിരം ദശാംശം *) കണ്ണി->op_params + 0, വലുപ്പം(ദശാംശം));
                memcpy(&max_bias, (സ്ഥിരം ദശാംശം *) കണ്ണി->op_params + 1, വലുപ്പം(ദശാംശം));

                എങ്കിൽ (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } അല്ലെങ്കിൽ {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                എങ്കിൽ (ctx->cpu_src2) {
                    ggml_soft_max_add_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_GET_ROWS:
            ctx->cpu_dst = ggml_get_rows(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SET_ROWS:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_base = ggml_new_tensor(ctx->ggml_ctx, കണ്ണി->type, GGML_MAX_DIMS, കണ്ണി->ne);
                എങ്കിൽ (!cpu_dst_base) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    തിരികെ അസത്യം;
                }
                cpu_dst_base->data = ctx->cpu_dst_data;
                memcpy(cpu_dst_base->nb, കണ്ണി->nb, വലുപ്പം(കണ്ണി->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_set_rows(ctx->ggml_ctx, cpu_dst_base, ctx->cpu_src0, ctx->cpu_src1);
            }
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            തിരികെ അസത്യം;
    }

    എങ്കിൽ (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        തിരികെ അസത്യം;
    }

    ctx->cpu_dst->data = ctx->cpu_dst_data;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    എങ്കിൽ (op != GGML_OP_CONT) {
        memcpy(ctx->cpu_dst->nb, കണ്ണി->nb, വലുപ്പം(കണ്ണി->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_graph = ggml_new_graph_custom(ctx->ggml_ctx, 1, അസത്യം);
    എങ്കിൽ (!ctx->cpu_graph) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        തിരികെ അസത്യം;
    }
    ctx->cpu_graph->nodes[0] = ctx->cpu_dst;
    ctx->cpu_graph->n_nodes  = 1;

    // Log input data for debugging if enabled
    എങ്കിൽ (config && config->log_differences) {
        എങ്കിൽ (ctx->cpu_src0_data && ctx->src0_size >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_data)[0],
                           ((uint8_t *) ctx->cpu_src0_data)[1], ((uint8_t *) ctx->cpu_src0_data)[2],
                           ((uint8_t *) ctx->cpu_src0_data)[3]);
        }
        എങ്കിൽ (ctx->cpu_src1_data && ctx->src1_size >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((ദശാംശം *) ctx->cpu_src1_data)[0],
                           ((ദശാംശം *) ctx->cpu_src1_data)[1], ((ദശാംശം *) ctx->cpu_src1_data)[2],
                           ((ദശാംശം *) ctx->cpu_src1_data)[3]);
        }
    }

    // Compute using CPU backend
    ggml_status cpu_result = ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph);

    എങ്കിൽ (cpu_result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_result);
        തിരികെ അസത്യം;
    }

    // Log output data for debugging if enabled
    എങ്കിൽ (config && config->log_differences && ctx->dst_size >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((ദശാംശം *) ctx->cpu_dst_data)[0], ((ദശാംശം *) ctx->cpu_dst_data)[1],
                       ((ദശാംശം *) ctx->cpu_dst_data)[2], ((ദശാംശം *) ctx->cpu_dst_data)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_size = ggml_nbytes(കണ്ണി);
    ggml_backend_tensor_get(കണ്ണി, ctx->et_dst_data, 0, dst_logical_size);

    എങ്കിൽ (config->log_differences) {
        size_t അക്കം_elements = ggml_nelements(കണ്ണി);
        size_t max_log      = std::min(അക്കം_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        തർക്കം    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ADD || op == GGML_OP_GLU);
        ദശാംശം * cpu_src0_float = is_elementwise ? (ദശാംശം *) ctx->cpu_src0_data : nullptr;
        ദശാംശം * cpu_src1_float = is_elementwise ? (ദശാംശം *) ctx->cpu_src1_data : nullptr;

        // Helper to get float value from tensor data (handles f16 and f32)
        auto get_float = [](സ്ഥിരം ശൂന്യം * data, size_t idx, ggml_type type) -> ദശാംശം {
            എങ്കിൽ (type == GGML_TYPE_F16) {
                സ്ഥിരം ggml_fp16_t * fp16_data = (സ്ഥിരം ggml_fp16_t *) data;
                തിരികെ ggml_fp16_to_fp32(fp16_data[idx]);
            }

            സ്ഥിരം ദശാംശം * float_data = (സ്ഥിരം ദശാംശം *) data;
            തിരികെ float_data[idx];
        };

        // Compare all elements but log only the first max_log_elements
        തർക്കം   matches          = സത്യം;
        size_t total_mismatches = 0;

        // First pass: check all elements for mismatches
        വേണ്ടി (size_t i = 0; i < അക്കം_elements; i++) {
            ദശാംശം cpu_val  = get_float(ctx->cpu_dst_data, i, കണ്ണി->type);
            ദശാംശം et_val   = get_float(ctx->et_dst_data, i, കണ്ണി->type);
            ദശാംശം diff     = fabsf(cpu_val - et_val);
            ദശാംശം rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            എങ്കിൽ (rel_diff > config->tolerance) {
                matches = അസത്യം;
                total_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        വേണ്ടി (size_t i = 0; i < max_log; i++) {
            ദശാംശം cpu_val = get_float(ctx->cpu_dst_data, i, കണ്ണി->type);
            ദശാംശം et_val  = get_float(ctx->et_dst_data, i, കണ്ണി->type);
            ദശാംശം diff    = fabsf(cpu_val - et_val);

            എങ്കിൽ (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } അല്ലെങ്കിൽ എങ്കിൽ (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } അല്ലെങ്കിൽ {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        എങ്കിൽ (അക്കം_elements > max_log) {
            size_t mid     = അക്കം_elements / 2;
            size_t end     = അക്കം_elements - 1;
            ദശാംശം  cpu_mid = get_float(ctx->cpu_dst_data, mid, കണ്ണി->type);
            ദശാംശം  et_mid  = get_float(ctx->et_dst_data, mid, കണ്ണി->type);
            ദശാംശം  cpu_end = get_float(ctx->cpu_dst_data, end, കണ്ണി->type);
            ദശാംശം  et_end  = get_float(ctx->et_dst_data, end, കണ്ണി->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_end, et_end);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       അക്കം_elements - total_mismatches, അക്കം_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    എങ്കിൽ (config->use_cpu_result) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_size = ggml_nbytes(കണ്ണി);
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(കണ്ണി), ctx->cpu_dst_data, 0, dst_logical_size);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    തിരികെ സത്യം;
}

ശൂന്യം ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    എങ്കിൽ (!ctx) {
        തിരികെ;
    }

    എങ്കിൽ (ctx->cpu_src0_data) {
        സ്വതന്ത്രം(ctx->cpu_src0_data);
        ctx->cpu_src0_data = nullptr;
    }
    എങ്കിൽ (ctx->cpu_src1_data) {
        സ്വതന്ത്രം(ctx->cpu_src1_data);
        ctx->cpu_src1_data = nullptr;
    }
    എങ്കിൽ (ctx->cpu_src2_data) {
        സ്വതന്ത്രം(ctx->cpu_src2_data);
        ctx->cpu_src2_data = nullptr;
    }
    എങ്കിൽ (ctx->cpu_dst_data) {
        സ്വതന്ത്രം(ctx->cpu_dst_data);
        ctx->cpu_dst_data = nullptr;
    }
    എങ്കിൽ (ctx->et_dst_data) {
        സ്വതന്ത്രം(ctx->et_dst_data);
        ctx->et_dst_data = nullptr;
    }

    എങ്കിൽ (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }

    എങ്കിൽ (ctx->cpu_backend) {
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
