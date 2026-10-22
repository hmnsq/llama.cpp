#अन्तर्भाव "ggml-et-cpu-compare.h"

#अन्तर्भाव "ggml-cpu/ggml-cpu-impl.h"
#अन्तर्भाव "ggml-cpu/ops.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cmath>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>

तार्किक ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ध्रुव ggml_tensor * ग्रन्थिः, ggml_op op) {
    जइ (!ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        प्रत्यागम मिच्छ;
    }

    // Clear context
    memset(ctx, 0, परिमाण(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    auto get_tensor_buffer_size = [](ध्रुव ggml_tensor * tensor) -> size_t {
        जइ (!tensor) {
            प्रत्यागम 0;
        }

        जइ (tensor->buffer) {
            // Get actual backend buffer size
            size_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);

            // Use the full buffer size to avoid any truncation issues
            प्रत्यागम buffer_size;
        } अण्णहा {
            // Fallback to logical size if no buffer
            प्रत्यागम ggml_nbytes(tensor);
        }
    };

    ctx->src0_size = get_tensor_buffer_size(ग्रन्थिः->src[0]);
    ctx->src1_size = get_tensor_buffer_size(ग्रन्थिः->src[1]);
    ctx->src2_size = get_tensor_buffer_size(ग्रन्थिः->src[2]);
    ctx->dst_size  = get_tensor_buffer_size(ग्रन्थिः);

    // Allocate CPU buffers for all tensors
    जइ (ctx->src0_size > 0) {
        ctx->cpu_src0_data = malloc(ctx->src0_size);
        जइ (!ctx->cpu_src0_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            गच्छ cleanup;
        }
    }

    जइ (ctx->src1_size > 0) {
        ctx->cpu_src1_data = malloc(ctx->src1_size);
        जइ (!ctx->cpu_src1_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            गच्छ cleanup;
        }
    }

    जइ (ctx->src2_size > 0) {
        ctx->cpu_src2_data = malloc(ctx->src2_size);
        जइ (!ctx->cpu_src2_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            गच्छ cleanup;
        }
    }

    ctx->cpu_dst_data = malloc(ctx->dst_size);
    जइ (!ctx->cpu_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        गच्छ cleanup;
    }

    ctx->et_dst_data = malloc(ctx->dst_size);
    जइ (!ctx->et_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        गच्छ cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    जइ (ctx->src0_size > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_size = ggml_nbytes(ग्रन्थिः->src[0]);
        ggml_backend_tensor_get(ग्रन्थिः->src[0], ctx->cpu_src0_data, 0, logical_size);
    }
    जइ (ctx->src1_size > 0) {
        size_t logical_size = ggml_nbytes(ग्रन्थिः->src[1]);
        ggml_backend_tensor_get(ग्रन्थिः->src[1], ctx->cpu_src1_data, 0, logical_size);
    }
    जइ (ctx->src2_size > 0) {
        size_t logical_size = ggml_nbytes(ग्रन्थिः->src[2]);
        ggml_backend_tensor_get(ग्रन्थिः->src[2], ctx->cpu_src2_data, 0, logical_size);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_size = ggml_nbytes(ग्रन्थिः);
        ggml_backend_tensor_get(ग्रन्थिः, ctx->cpu_dst_data, 0, logical_size);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    जइ (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        गच्छ cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_size   = ggml_tensor_overhead() * 4 + ggml_graph_overhead();  // up to 4 tensors + graph
    ctx_params.mem_buffer = nullptr;
    ctx_params.no_alloc   = सच्च;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    जइ (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        गच्छ cleanup;
    }

    // Create CPU tensors with proper context
    जइ (ग्रन्थिः->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, ग्रन्थिः->src[0]->type, GGML_MAX_DIMS, ग्रन्थिः->src[0]->ne);
        जइ (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            गच्छ cleanup;
        }
        ctx->cpu_src0->data = ctx->cpu_src0_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src0->nb, ग्रन्थिः->src[0]->nb, परिमाण(ग्रन्थिः->src[0]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src0->op_params, ग्रन्थिः->src[0]->op_params, परिमाण(ग्रन्थिः->src[0]->op_params));
    }

    जइ (ग्रन्थिः->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, ग्रन्थिः->src[1]->type, GGML_MAX_DIMS, ग्रन्थिः->src[1]->ne);
        जइ (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            गच्छ cleanup;
        }
        ctx->cpu_src1->data = ctx->cpu_src1_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src1->nb, ग्रन्थिः->src[1]->nb, परिमाण(ग्रन्थिः->src[1]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src1->op_params, ग्रन्थिः->src[1]->op_params, परिमाण(ग्रन्थिः->src[1]->op_params));
    }

    जइ (ग्रन्थिः->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, ग्रन्थिः->src[2]->type, GGML_MAX_DIMS, ग्रन्थिः->src[2]->ne);
        जइ (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            गच्छ cleanup;
        }
        ctx->cpu_src2->data = ctx->cpu_src2_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src2->nb, ग्रन्थिः->src[2]->nb, परिमाण(ग्रन्थिः->src[2]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src2->op_params, ग्रन्थिः->src[2]->op_params, परिमाण(ग्रन्थिः->src[2]->op_params));
    }

    प्रत्यागम सच्च;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    प्रत्यागम मिच्छ;
}

तार्किक ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ध्रुव ggml_tensor *                ग्रन्थिः,
                                           ध्रुव ggml_et_cpu_compare_config * config) {
    जइ (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !ग्रन्थिः || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        प्रत्यागम मिच्छ;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = ग्रन्थिः->op;
    विकल्प (op) {
        पक्ष GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            भङ्ग;
        पक्ष GGML_OP_ADD:
            ctx->cpu_dst = ggml_add(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            भङ्ग;
        पक्ष GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            भङ्ग;
        पक्ष GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            भङ्ग;
        पक्ष GGML_OP_ROPE:
            {
                ध्रुव int32_t * op_params  = (ध्रुव int32_t *) ग्रन्थिः->op_params;
                ध्रुव int32_t   n_dims     = op_params[1];
                ध्रुव int32_t   mode       = op_params[2];
                ध्रुव int32_t   n_ctx_orig = op_params[4];
                ध्रुव भिन्न     freq_base  = *((ध्रुव भिन्न *) (op_params + 5));
                ध्रुव भिन्न     freq_scale = *((ध्रुव भिन्न *) (op_params + 6));
                ध्रुव भिन्न     ext_factor = *((ध्रुव भिन्न *) (op_params + 7));
                ध्रुव भिन्न     attn_factor = *((ध्रुव भिन्न *) (op_params + 8));
                ध्रुव भिन्न     beta_fast  = *((ध्रुव भिन्न *) (op_params + 9));
                ध्रुव भिन्न     beta_slow  = *((ध्रुव भिन्न *) (op_params + 10));

                जइ (mode & GGML_ROPE_TYPE_MROPE) {
                    पूर्णाङ्क sections[GGML_MROPE_SECTIONS];
                    memcpy(sections, op_params + 11, परिमाण(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
                                                   ext_factor, attn_factor, beta_fast, beta_slow);
                } अण्णहा {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
                                                 attn_factor, beta_fast, beta_slow);
                }
            }
            भङ्ग;
        पक्ष GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                भिन्न eps;
                memcpy(&eps, ग्रन्थिः->op_params, परिमाण(भिन्न));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            भङ्ग;
        पक्ष GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            भङ्ग;
        पक्ष GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(ग्रन्थिः, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            भङ्ग;
        पक्ष GGML_OP_SUM_ROWS:
            ctx->cpu_dst = ggml_sum_rows(ctx->ggml_ctx, ctx->cpu_src0);
            भङ्ग;
        पक्ष GGML_OP_MEAN:
            ctx->cpu_dst = ggml_mean(ctx->ggml_ctx, ctx->cpu_src0);
            भङ्ग;
        पक्ष GGML_OP_CLAMP:
            {
                भिन्न clamp_min, clamp_max;
                memcpy(&clamp_min, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 0, परिमाण(भिन्न));
                memcpy(&clamp_max, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 1, परिमाण(भिन्न));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            भङ्ग;
        पक्ष GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(ग्रन्थिः, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                जइ (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    प्रत्यागम मिच्छ;
                }
                जइ (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    ध्रुव भिन्न limit = ggml_get_op_params_f32(ग्रन्थिः, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, limit);
                } अण्णहा {
                    ctx->cpu_dst = ggml_glu_split(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            भङ्ग;
        पक्ष GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                भिन्न scale    = 1.0f;
                भिन्न max_bias = 0.0f;
                memcpy(&scale, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 0, परिमाण(भिन्न));
                memcpy(&max_bias, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 1, परिमाण(भिन्न));

                जइ (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } अण्णहा {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                जइ (ctx->cpu_src2) {
                    ggml_soft_max_add_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            भङ्ग;
        पक्ष GGML_OP_GET_ROWS:
            ctx->cpu_dst = ggml_get_rows(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            भङ्ग;
        पक्ष GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            भङ्ग;
        पक्ष GGML_OP_SET_ROWS:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_base = ggml_new_tensor(ctx->ggml_ctx, ग्रन्थिः->type, GGML_MAX_DIMS, ग्रन्थिः->ne);
                जइ (!cpu_dst_base) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    प्रत्यागम मिच्छ;
                }
                cpu_dst_base->data = ctx->cpu_dst_data;
                memcpy(cpu_dst_base->nb, ग्रन्थिः->nb, परिमाण(ग्रन्थिः->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_set_rows(ctx->ggml_ctx, cpu_dst_base, ctx->cpu_src0, ctx->cpu_src1);
            }
            भङ्ग;
        स्वभाव:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            प्रत्यागम मिच्छ;
    }

    जइ (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        प्रत्यागम मिच्छ;
    }

    ctx->cpu_dst->data = ctx->cpu_dst_data;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    जइ (op != GGML_OP_CONT) {
        memcpy(ctx->cpu_dst->nb, ग्रन्थिः->nb, परिमाण(ग्रन्थिः->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_graph = ggml_new_graph_custom(ctx->ggml_ctx, 1, मिच्छ);
    जइ (!ctx->cpu_graph) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        प्रत्यागम मिच्छ;
    }
    ctx->cpu_graph->nodes[0] = ctx->cpu_dst;
    ctx->cpu_graph->n_nodes  = 1;

    // Log input data for debugging if enabled
    जइ (config && config->log_differences) {
        जइ (ctx->cpu_src0_data && ctx->src0_size >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_data)[0],
                           ((uint8_t *) ctx->cpu_src0_data)[1], ((uint8_t *) ctx->cpu_src0_data)[2],
                           ((uint8_t *) ctx->cpu_src0_data)[3]);
        }
        जइ (ctx->cpu_src1_data && ctx->src1_size >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((भिन्न *) ctx->cpu_src1_data)[0],
                           ((भिन्न *) ctx->cpu_src1_data)[1], ((भिन्न *) ctx->cpu_src1_data)[2],
                           ((भिन्न *) ctx->cpu_src1_data)[3]);
        }
    }

    // Compute using CPU backend
    ggml_status cpu_result = ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph);

    जइ (cpu_result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_result);
        प्रत्यागम मिच्छ;
    }

    // Log output data for debugging if enabled
    जइ (config && config->log_differences && ctx->dst_size >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((भिन्न *) ctx->cpu_dst_data)[0], ((भिन्न *) ctx->cpu_dst_data)[1],
                       ((भिन्न *) ctx->cpu_dst_data)[2], ((भिन्न *) ctx->cpu_dst_data)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_size = ggml_nbytes(ग्रन्थिः);
    ggml_backend_tensor_get(ग्रन्थिः, ctx->et_dst_data, 0, dst_logical_size);

    जइ (config->log_differences) {
        size_t अङ्कः_elements = ggml_nelements(ग्रन्थिः);
        size_t max_log      = std::min(अङ्कः_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        तार्किक    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ADD || op == GGML_OP_GLU);
        भिन्न * cpu_src0_float = is_elementwise ? (भिन्न *) ctx->cpu_src0_data : nullptr;
        भिन्न * cpu_src1_float = is_elementwise ? (भिन्न *) ctx->cpu_src1_data : nullptr;

        // Helper to get float value from tensor data (handles f16 and f32)
        auto get_float = [](ध्रुव शून्य * data, size_t idx, ggml_type type) -> भिन्न {
            जइ (type == GGML_TYPE_F16) {
                ध्रुव ggml_fp16_t * fp16_data = (ध्रुव ggml_fp16_t *) data;
                प्रत्यागम ggml_fp16_to_fp32(fp16_data[idx]);
            }

            ध्रुव भिन्न * float_data = (ध्रुव भिन्न *) data;
            प्रत्यागम float_data[idx];
        };

        // Compare all elements but log only the first max_log_elements
        तार्किक   matches          = सच्च;
        size_t total_mismatches = 0;

        // First pass: check all elements for mismatches
        प्रति (size_t i = 0; i < अङ्कः_elements; i++) {
            भिन्न cpu_val  = get_float(ctx->cpu_dst_data, i, ग्रन्थिः->type);
            भिन्न et_val   = get_float(ctx->et_dst_data, i, ग्रन्थिः->type);
            भिन्न diff     = fabsf(cpu_val - et_val);
            भिन्न rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            जइ (rel_diff > config->tolerance) {
                matches = मिच्छ;
                total_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        प्रति (size_t i = 0; i < max_log; i++) {
            भिन्न cpu_val = get_float(ctx->cpu_dst_data, i, ग्रन्थिः->type);
            भिन्न et_val  = get_float(ctx->et_dst_data, i, ग्रन्थिः->type);
            भिन्न diff    = fabsf(cpu_val - et_val);

            जइ (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } अण्णहा जइ (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } अण्णहा {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        जइ (अङ्कः_elements > max_log) {
            size_t mid     = अङ्कः_elements / 2;
            size_t end     = अङ्कः_elements - 1;
            भिन्न  cpu_mid = get_float(ctx->cpu_dst_data, mid, ग्रन्थिः->type);
            भिन्न  et_mid  = get_float(ctx->et_dst_data, mid, ग्रन्थिः->type);
            भिन्न  cpu_end = get_float(ctx->cpu_dst_data, end, ग्रन्थिः->type);
            भिन्न  et_end  = get_float(ctx->et_dst_data, end, ग्रन्थिः->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_end, et_end);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       अङ्कः_elements - total_mismatches, अङ्कः_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    जइ (config->use_cpu_result) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_size = ggml_nbytes(ग्रन्थिः);
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(ग्रन्थिः), ctx->cpu_dst_data, 0, dst_logical_size);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    प्रत्यागम सच्च;
}

शून्य ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    जइ (!ctx) {
        प्रत्यागम;
    }

    जइ (ctx->cpu_src0_data) {
        मुक्त(ctx->cpu_src0_data);
        ctx->cpu_src0_data = nullptr;
    }
    जइ (ctx->cpu_src1_data) {
        मुक्त(ctx->cpu_src1_data);
        ctx->cpu_src1_data = nullptr;
    }
    जइ (ctx->cpu_src2_data) {
        मुक्त(ctx->cpu_src2_data);
        ctx->cpu_src2_data = nullptr;
    }
    जइ (ctx->cpu_dst_data) {
        मुक्त(ctx->cpu_dst_data);
        ctx->cpu_dst_data = nullptr;
    }
    जइ (ctx->et_dst_data) {
        मुक्त(ctx->et_dst_data);
        ctx->et_dst_data = nullptr;
    }

    जइ (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }

    जइ (ctx->cpu_backend) {
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
