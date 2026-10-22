#अन्तर्भाव "ggml-et-cpu-compare.h"

#अन्तर्भाव "ggml-cpu/ggml-cpu-impl.h"
#अन्तर्भाव "ggml-cpu/ops.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cmath>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>

तार्किक ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ध्रुव ggml_tensor * गाठ, ggml_op op) {
    यदि (!ctx || !गाठ) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        प्रत्यागम असत्य;
    }

    // Clear context
    memset(ctx, 0, परिमाण(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    auto get_tensor_buffer_size = [](ध्रुव ggml_tensor * tensor) -> size_t {
        यदि (!tensor) {
            प्रत्यागम 0;
        }

        यदि (tensor->buffer) {
            // Get actual backend buffer size
            size_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);

            // Use the full buffer size to avoid any truncation issues
            प्रत्यागम buffer_size;
        } अन्यथा {
            // Fallback to logical size if no buffer
            प्रत्यागम ggml_nbytes(tensor);
        }
    };

    ctx->src0_size = get_tensor_buffer_size(गाठ->src[0]);
    ctx->src1_size = get_tensor_buffer_size(गाठ->src[1]);
    ctx->src2_size = get_tensor_buffer_size(गाठ->src[2]);
    ctx->dst_size  = get_tensor_buffer_size(गाठ);

    // Allocate CPU buffers for all tensors
    यदि (ctx->src0_size > 0) {
        ctx->cpu_src0_data = malloc(ctx->src0_size);
        यदि (!ctx->cpu_src0_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            गच्छ cleanup;
        }
    }

    यदि (ctx->src1_size > 0) {
        ctx->cpu_src1_data = malloc(ctx->src1_size);
        यदि (!ctx->cpu_src1_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            गच्छ cleanup;
        }
    }

    यदि (ctx->src2_size > 0) {
        ctx->cpu_src2_data = malloc(ctx->src2_size);
        यदि (!ctx->cpu_src2_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            गच्छ cleanup;
        }
    }

    ctx->cpu_dst_data = malloc(ctx->dst_size);
    यदि (!ctx->cpu_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        गच्छ cleanup;
    }

    ctx->et_dst_data = malloc(ctx->dst_size);
    यदि (!ctx->et_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        गच्छ cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    यदि (ctx->src0_size > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_size = ggml_nbytes(गाठ->src[0]);
        ggml_backend_tensor_get(गाठ->src[0], ctx->cpu_src0_data, 0, logical_size);
    }
    यदि (ctx->src1_size > 0) {
        size_t logical_size = ggml_nbytes(गाठ->src[1]);
        ggml_backend_tensor_get(गाठ->src[1], ctx->cpu_src1_data, 0, logical_size);
    }
    यदि (ctx->src2_size > 0) {
        size_t logical_size = ggml_nbytes(गाठ->src[2]);
        ggml_backend_tensor_get(गाठ->src[2], ctx->cpu_src2_data, 0, logical_size);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_size = ggml_nbytes(गाठ);
        ggml_backend_tensor_get(गाठ, ctx->cpu_dst_data, 0, logical_size);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    यदि (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        गच्छ cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_size   = ggml_tensor_overhead() * 4 + ggml_graph_overhead();  // up to 4 tensors + graph
    ctx_params.mem_buffer = nullptr;
    ctx_params.no_alloc   = सत्य;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    यदि (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        गच्छ cleanup;
    }

    // Create CPU tensors with proper context
    यदि (गाठ->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, गाठ->src[0]->type, GGML_MAX_DIMS, गाठ->src[0]->ne);
        यदि (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            गच्छ cleanup;
        }
        ctx->cpu_src0->data = ctx->cpu_src0_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src0->nb, गाठ->src[0]->nb, परिमाण(गाठ->src[0]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src0->op_params, गाठ->src[0]->op_params, परिमाण(गाठ->src[0]->op_params));
    }

    यदि (गाठ->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, गाठ->src[1]->type, GGML_MAX_DIMS, गाठ->src[1]->ne);
        यदि (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            गच्छ cleanup;
        }
        ctx->cpu_src1->data = ctx->cpu_src1_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src1->nb, गाठ->src[1]->nb, परिमाण(गाठ->src[1]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src1->op_params, गाठ->src[1]->op_params, परिमाण(गाठ->src[1]->op_params));
    }

    यदि (गाठ->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, गाठ->src[2]->type, GGML_MAX_DIMS, गाठ->src[2]->ne);
        यदि (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            गच्छ cleanup;
        }
        ctx->cpu_src2->data = ctx->cpu_src2_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src2->nb, गाठ->src[2]->nb, परिमाण(गाठ->src[2]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src2->op_params, गाठ->src[2]->op_params, परिमाण(गाठ->src[2]->op_params));
    }

    प्रत्यागम सत्य;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    प्रत्यागम असत्य;
}

तार्किक ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ध्रुव ggml_tensor *                गाठ,
                                           ध्रुव ggml_et_cpu_compare_config * config) {
    यदि (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !गाठ || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        प्रत्यागम असत्य;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = गाठ->op;
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
                ध्रुव int32_t * op_params  = (ध्रुव int32_t *) गाठ->op_params;
                ध्रुव int32_t   n_dims     = op_params[1];
                ध्रुव int32_t   mode       = op_params[2];
                ध्रुव int32_t   n_ctx_orig = op_params[4];
                ध्रुव भिन्न     freq_base  = *((ध्रुव भिन्न *) (op_params + 5));
                ध्रुव भिन्न     freq_scale = *((ध्रुव भिन्न *) (op_params + 6));
                ध्रुव भिन्न     ext_factor = *((ध्रुव भिन्न *) (op_params + 7));
                ध्रुव भिन्न     attn_factor = *((ध्रुव भिन्न *) (op_params + 8));
                ध्रुव भिन्न     beta_fast  = *((ध्रुव भिन्न *) (op_params + 9));
                ध्रुव भिन्न     beta_slow  = *((ध्रुव भिन्न *) (op_params + 10));

                यदि (mode & GGML_ROPE_TYPE_MROPE) {
                    पूर्णाङ्क sections[GGML_MROPE_SECTIONS];
                    memcpy(sections, op_params + 11, परिमाण(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
                                                   ext_factor, attn_factor, beta_fast, beta_slow);
                } अन्यथा {
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
                memcpy(&eps, गाठ->op_params, परिमाण(भिन्न));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            भङ्ग;
        पक्ष GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            भङ्ग;
        पक्ष GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(गाठ, 0);
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
                memcpy(&clamp_min, (ध्रुव भिन्न *) गाठ->op_params + 0, परिमाण(भिन्न));
                memcpy(&clamp_max, (ध्रुव भिन्न *) गाठ->op_params + 1, परिमाण(भिन्न));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            भङ्ग;
        पक्ष GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(गाठ, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                यदि (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    प्रत्यागम असत्य;
                }
                यदि (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    ध्रुव भिन्न limit = ggml_get_op_params_f32(गाठ, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, limit);
                } अन्यथा {
                    ctx->cpu_dst = ggml_glu_split(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            भङ्ग;
        पक्ष GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                भिन्न scale    = 1.0f;
                भिन्न max_bias = 0.0f;
                memcpy(&scale, (ध्रुव भिन्न *) गाठ->op_params + 0, परिमाण(भिन्न));
                memcpy(&max_bias, (ध्रुव भिन्न *) गाठ->op_params + 1, परिमाण(भिन्न));

                यदि (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } अन्यथा {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                यदि (ctx->cpu_src2) {
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
                ggml_tensor * cpu_dst_base = ggml_new_tensor(ctx->ggml_ctx, गाठ->type, GGML_MAX_DIMS, गाठ->ne);
                यदि (!cpu_dst_base) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    प्रत्यागम असत्य;
                }
                cpu_dst_base->data = ctx->cpu_dst_data;
                memcpy(cpu_dst_base->nb, गाठ->nb, परिमाण(गाठ->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_set_rows(ctx->ggml_ctx, cpu_dst_base, ctx->cpu_src0, ctx->cpu_src1);
            }
            भङ्ग;
        स्वभाव:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            प्रत्यागम असत्य;
    }

    यदि (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        प्रत्यागम असत्य;
    }

    ctx->cpu_dst->data = ctx->cpu_dst_data;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    यदि (op != GGML_OP_CONT) {
        memcpy(ctx->cpu_dst->nb, गाठ->nb, परिमाण(गाठ->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_graph = ggml_new_graph_custom(ctx->ggml_ctx, 1, असत्य);
    यदि (!ctx->cpu_graph) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        प्रत्यागम असत्य;
    }
    ctx->cpu_graph->nodes[0] = ctx->cpu_dst;
    ctx->cpu_graph->n_nodes  = 1;

    // Log input data for debugging if enabled
    यदि (config && config->log_differences) {
        यदि (ctx->cpu_src0_data && ctx->src0_size >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_data)[0],
                           ((uint8_t *) ctx->cpu_src0_data)[1], ((uint8_t *) ctx->cpu_src0_data)[2],
                           ((uint8_t *) ctx->cpu_src0_data)[3]);
        }
        यदि (ctx->cpu_src1_data && ctx->src1_size >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((भिन्न *) ctx->cpu_src1_data)[0],
                           ((भिन्न *) ctx->cpu_src1_data)[1], ((भिन्न *) ctx->cpu_src1_data)[2],
                           ((भिन्न *) ctx->cpu_src1_data)[3]);
        }
    }

    // Compute using CPU backend
    ggml_status cpu_result = ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph);

    यदि (cpu_result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_result);
        प्रत्यागम असत्य;
    }

    // Log output data for debugging if enabled
    यदि (config && config->log_differences && ctx->dst_size >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((भिन्न *) ctx->cpu_dst_data)[0], ((भिन्न *) ctx->cpu_dst_data)[1],
                       ((भिन्न *) ctx->cpu_dst_data)[2], ((भिन्न *) ctx->cpu_dst_data)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_size = ggml_nbytes(गाठ);
    ggml_backend_tensor_get(गाठ, ctx->et_dst_data, 0, dst_logical_size);

    यदि (config->log_differences) {
        size_t अंक_elements = ggml_nelements(गाठ);
        size_t max_log      = std::min(अंक_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        तार्किक    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ADD || op == GGML_OP_GLU);
        भिन्न * cpu_src0_float = is_elementwise ? (भिन्न *) ctx->cpu_src0_data : nullptr;
        भिन्न * cpu_src1_float = is_elementwise ? (भिन्न *) ctx->cpu_src1_data : nullptr;

        // Helper to get float value from tensor data (handles f16 and f32)
        auto get_float = [](ध्रुव शून्य * data, size_t idx, ggml_type type) -> भिन्न {
            यदि (type == GGML_TYPE_F16) {
                ध्रुव ggml_fp16_t * fp16_data = (ध्रुव ggml_fp16_t *) data;
                प्रत्यागम ggml_fp16_to_fp32(fp16_data[idx]);
            }

            ध्रुव भिन्न * float_data = (ध्रुव भिन्न *) data;
            प्रत्यागम float_data[idx];
        };

        // Compare all elements but log only the first max_log_elements
        तार्किक   matches          = सत्य;
        size_t total_mismatches = 0;

        // First pass: check all elements for mismatches
        प्रति (size_t i = 0; i < अंक_elements; i++) {
            भिन्न cpu_val  = get_float(ctx->cpu_dst_data, i, गाठ->type);
            भिन्न et_val   = get_float(ctx->et_dst_data, i, गाठ->type);
            भिन्न diff     = fabsf(cpu_val - et_val);
            भिन्न rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            यदि (rel_diff > config->tolerance) {
                matches = असत्य;
                total_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        प्रति (size_t i = 0; i < max_log; i++) {
            भिन्न cpu_val = get_float(ctx->cpu_dst_data, i, गाठ->type);
            भिन्न et_val  = get_float(ctx->et_dst_data, i, गाठ->type);
            भिन्न diff    = fabsf(cpu_val - et_val);

            यदि (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } अन्यथा यदि (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } अन्यथा {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        यदि (अंक_elements > max_log) {
            size_t mid     = अंक_elements / 2;
            size_t end     = अंक_elements - 1;
            भिन्न  cpu_mid = get_float(ctx->cpu_dst_data, mid, गाठ->type);
            भिन्न  et_mid  = get_float(ctx->et_dst_data, mid, गाठ->type);
            भिन्न  cpu_end = get_float(ctx->cpu_dst_data, end, गाठ->type);
            भिन्न  et_end  = get_float(ctx->et_dst_data, end, गाठ->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_end, et_end);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       अंक_elements - total_mismatches, अंक_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    यदि (config->use_cpu_result) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_size = ggml_nbytes(गाठ);
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(गाठ), ctx->cpu_dst_data, 0, dst_logical_size);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    प्रत्यागम सत्य;
}

शून्य ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    यदि (!ctx) {
        प्रत्यागम;
    }

    यदि (ctx->cpu_src0_data) {
        मुक्त(ctx->cpu_src0_data);
        ctx->cpu_src0_data = nullptr;
    }
    यदि (ctx->cpu_src1_data) {
        मुक्त(ctx->cpu_src1_data);
        ctx->cpu_src1_data = nullptr;
    }
    यदि (ctx->cpu_src2_data) {
        मुक्त(ctx->cpu_src2_data);
        ctx->cpu_src2_data = nullptr;
    }
    यदि (ctx->cpu_dst_data) {
        मुक्त(ctx->cpu_dst_data);
        ctx->cpu_dst_data = nullptr;
    }
    यदि (ctx->et_dst_data) {
        मुक्त(ctx->et_dst_data);
        ctx->et_dst_data = nullptr;
    }

    यदि (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }

    यदि (ctx->cpu_backend) {
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
