#अन्तर्भाव "ggml-et-cpu-compare.h"

#अन्तर्भाव "ggml-cpu/ggml-cpu-impl.h"
#अन्तर्भाव "ggml-cpu/ops.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cmath>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>

तार्किक ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, स्थिरांक ggml_tensor * गाठ, ggml_op op) {
    जर (!ctx || !गाठ) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        परत असत्य;
    }

    // Clear context
    स_रखो(ctx, 0, आकार(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    स्वतः get_tensor_buffer_size = [](स्थिरांक ggml_tensor * tensor) -> माप_प्रकार {
        जर (!tensor) {
            परत 0;
        }

        जर (tensor->buffer) {
            // Get actual backend buffer size
            माप_प्रकार buffer_size = ggml_backend_buffer_get_size(tensor->buffer);

            // Use the full buffer size to avoid any truncation issues
            परत buffer_size;
        } अन्यथा {
            // Fallback to logical size if no buffer
            परत ggml_nbytes(tensor);
        }
    };

    ctx->src0_size = get_tensor_buffer_size(गाठ->src[0]);
    ctx->src1_size = get_tensor_buffer_size(गाठ->src[1]);
    ctx->src2_size = get_tensor_buffer_size(गाठ->src[2]);
    ctx->dst_size  = get_tensor_buffer_size(गाठ);

    // Allocate CPU buffers for all tensors
    जर (ctx->src0_size > 0) {
        ctx->cpu_src0_data = दो_स्मृति(ctx->src0_size);
        जर (!ctx->cpu_src0_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            जा cleanup;
        }
    }

    जर (ctx->src1_size > 0) {
        ctx->cpu_src1_data = दो_स्मृति(ctx->src1_size);
        जर (!ctx->cpu_src1_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            जा cleanup;
        }
    }

    जर (ctx->src2_size > 0) {
        ctx->cpu_src2_data = दो_स्मृति(ctx->src2_size);
        जर (!ctx->cpu_src2_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            जा cleanup;
        }
    }

    ctx->cpu_dst_data = दो_स्मृति(ctx->dst_size);
    जर (!ctx->cpu_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        जा cleanup;
    }

    ctx->et_dst_data = दो_स्मृति(ctx->dst_size);
    जर (!ctx->et_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        जा cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    जर (ctx->src0_size > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        माप_प्रकार logical_size = ggml_nbytes(गाठ->src[0]);
        ggml_backend_tensor_get(गाठ->src[0], ctx->cpu_src0_data, 0, logical_size);
    }
    जर (ctx->src1_size > 0) {
        माप_प्रकार logical_size = ggml_nbytes(गाठ->src[1]);
        ggml_backend_tensor_get(गाठ->src[1], ctx->cpu_src1_data, 0, logical_size);
    }
    जर (ctx->src2_size > 0) {
        माप_प्रकार logical_size = ggml_nbytes(गाठ->src[2]);
        ggml_backend_tensor_get(गाठ->src[2], ctx->cpu_src2_data, 0, logical_size);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        माप_प्रकार logical_size = ggml_nbytes(गाठ);
        ggml_backend_tensor_get(गाठ, ctx->cpu_dst_data, 0, logical_size);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    जर (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        जा cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_size   = ggml_tensor_overhead() * 4 + ggml_graph_overhead();  // up to 4 tensors + graph
    ctx_params.mem_buffer = nullptr;
    ctx_params.no_alloc   = सत्य;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    जर (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        जा cleanup;
    }

    // Create CPU tensors with proper context
    जर (गाठ->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, गाठ->src[0]->type, GGML_MAX_DIMS, गाठ->src[0]->ne);
        जर (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            जा cleanup;
        }
        ctx->cpu_src0->data = ctx->cpu_src0_data;
        // Copy stride array (nb) for correct memory layout
        स_नकल(ctx->cpu_src0->nb, गाठ->src[0]->nb, आकार(गाठ->src[0]->nb));
        // Copy op_params if present
        स_नकल(ctx->cpu_src0->op_params, गाठ->src[0]->op_params, आकार(गाठ->src[0]->op_params));
    }

    जर (गाठ->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, गाठ->src[1]->type, GGML_MAX_DIMS, गाठ->src[1]->ne);
        जर (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            जा cleanup;
        }
        ctx->cpu_src1->data = ctx->cpu_src1_data;
        // Copy stride array (nb) for correct memory layout
        स_नकल(ctx->cpu_src1->nb, गाठ->src[1]->nb, आकार(गाठ->src[1]->nb));
        // Copy op_params if present
        स_नकल(ctx->cpu_src1->op_params, गाठ->src[1]->op_params, आकार(गाठ->src[1]->op_params));
    }

    जर (गाठ->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, गाठ->src[2]->type, GGML_MAX_DIMS, गाठ->src[2]->ne);
        जर (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            जा cleanup;
        }
        ctx->cpu_src2->data = ctx->cpu_src2_data;
        // Copy stride array (nb) for correct memory layout
        स_नकल(ctx->cpu_src2->nb, गाठ->src[2]->nb, आकार(गाठ->src[2]->nb));
        // Copy op_params if present
        स_नकल(ctx->cpu_src2->op_params, गाठ->src[2]->op_params, आकार(गाठ->src[2]->op_params));
    }

    परत सत्य;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    परत असत्य;
}

तार्किक ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           स्थिरांक ggml_tensor *                गाठ,
                                           स्थिरांक ggml_et_cpu_compare_config * config) {
    जर (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !गाठ || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        परत असत्य;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = गाठ->op;
    निवड (op) {
        स्थिती GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            थांबा;
        स्थिती GGML_OP_ADD:
            ctx->cpu_dst = ggml_add(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            थांबा;
        स्थिती GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            थांबा;
        स्थिती GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            थांबा;
        स्थिती GGML_OP_ROPE:
            {
                स्थिरांक int32_t * op_params  = (स्थिरांक int32_t *) गाठ->op_params;
                स्थिरांक int32_t   n_dims     = op_params[1];
                स्थिरांक int32_t   mode       = op_params[2];
                स्थिरांक int32_t   n_ctx_orig = op_params[4];
                स्थिरांक दशांश     freq_base  = *((स्थिरांक दशांश *) (op_params + 5));
                स्थिरांक दशांश     freq_scale = *((स्थिरांक दशांश *) (op_params + 6));
                स्थिरांक दशांश     ext_factor = *((स्थिरांक दशांश *) (op_params + 7));
                स्थिरांक दशांश     attn_factor = *((स्थिरांक दशांश *) (op_params + 8));
                स्थिरांक दशांश     beta_fast  = *((स्थिरांक दशांश *) (op_params + 9));
                स्थिरांक दशांश     beta_slow  = *((स्थिरांक दशांश *) (op_params + 10));

                जर (mode & GGML_ROPE_TYPE_MROPE) {
                    पूर्णांक sections[GGML_MROPE_SECTIONS];
                    स_नकल(sections, op_params + 11, आकार(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
                                                   ext_factor, attn_factor, beta_fast, beta_slow);
                } अन्यथा {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
                                                 attn_factor, beta_fast, beta_slow);
                }
            }
            थांबा;
        स्थिती GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                दशांश eps;
                स_नकल(&eps, गाठ->op_params, आकार(दशांश));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            थांबा;
        स्थिती GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            थांबा;
        स्थिती GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(गाठ, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            थांबा;
        स्थिती GGML_OP_SUM_ROWS:
            ctx->cpu_dst = ggml_sum_rows(ctx->ggml_ctx, ctx->cpu_src0);
            थांबा;
        स्थिती GGML_OP_MEAN:
            ctx->cpu_dst = ggml_mean(ctx->ggml_ctx, ctx->cpu_src0);
            थांबा;
        स्थिती GGML_OP_CLAMP:
            {
                दशांश clamp_min, clamp_max;
                स_नकल(&clamp_min, (स्थिरांक दशांश *) गाठ->op_params + 0, आकार(दशांश));
                स_नकल(&clamp_max, (स्थिरांक दशांश *) गाठ->op_params + 1, आकार(दशांश));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            थांबा;
        स्थिती GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(गाठ, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                जर (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    परत असत्य;
                }
                जर (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    स्थिरांक दशांश limit = ggml_get_op_params_f32(गाठ, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, limit);
                } अन्यथा {
                    ctx->cpu_dst = ggml_glu_split(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            थांबा;
        स्थिती GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                दशांश scale    = 1.0f;
                दशांश max_bias = 0.0f;
                स_नकल(&scale, (स्थिरांक दशांश *) गाठ->op_params + 0, आकार(दशांश));
                स_नकल(&max_bias, (स्थिरांक दशांश *) गाठ->op_params + 1, आकार(दशांश));

                जर (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } अन्यथा {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                जर (ctx->cpu_src2) {
                    ggml_soft_max_add_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            थांबा;
        स्थिती GGML_OP_GET_ROWS:
            ctx->cpu_dst = ggml_get_rows(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            थांबा;
        स्थिती GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            थांबा;
        स्थिती GGML_OP_SET_ROWS:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_base = ggml_new_tensor(ctx->ggml_ctx, गाठ->type, GGML_MAX_DIMS, गाठ->ne);
                जर (!cpu_dst_base) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    परत असत्य;
                }
                cpu_dst_base->data = ctx->cpu_dst_data;
                स_नकल(cpu_dst_base->nb, गाठ->nb, आकार(गाठ->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_set_rows(ctx->ggml_ctx, cpu_dst_base, ctx->cpu_src0, ctx->cpu_src1);
            }
            थांबा;
        पूर्वनिर्धारित:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            परत असत्य;
    }

    जर (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        परत असत्य;
    }

    ctx->cpu_dst->data = ctx->cpu_dst_data;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    जर (op != GGML_OP_CONT) {
        स_नकल(ctx->cpu_dst->nb, गाठ->nb, आकार(गाठ->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_graph = ggml_new_graph_custom(ctx->ggml_ctx, 1, असत्य);
    जर (!ctx->cpu_graph) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        परत असत्य;
    }
    ctx->cpu_graph->nodes[0] = ctx->cpu_dst;
    ctx->cpu_graph->n_nodes  = 1;

    // Log input data for debugging if enabled
    जर (config && config->log_differences) {
        जर (ctx->cpu_src0_data && ctx->src0_size >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_data)[0],
                           ((uint8_t *) ctx->cpu_src0_data)[1], ((uint8_t *) ctx->cpu_src0_data)[2],
                           ((uint8_t *) ctx->cpu_src0_data)[3]);
        }
        जर (ctx->cpu_src1_data && ctx->src1_size >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((दशांश *) ctx->cpu_src1_data)[0],
                           ((दशांश *) ctx->cpu_src1_data)[1], ((दशांश *) ctx->cpu_src1_data)[2],
                           ((दशांश *) ctx->cpu_src1_data)[3]);
        }
    }

    // Compute using CPU backend
    ggml_status cpu_result = ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph);

    जर (cpu_result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_result);
        परत असत्य;
    }

    // Log output data for debugging if enabled
    जर (config && config->log_differences && ctx->dst_size >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((दशांश *) ctx->cpu_dst_data)[0], ((दशांश *) ctx->cpu_dst_data)[1],
                       ((दशांश *) ctx->cpu_dst_data)[2], ((दशांश *) ctx->cpu_dst_data)[3]);
    }

    // Now copy ET device destination to host for comparison
    माप_प्रकार dst_logical_size = ggml_nbytes(गाठ);
    ggml_backend_tensor_get(गाठ, ctx->et_dst_data, 0, dst_logical_size);

    जर (config->log_differences) {
        माप_प्रकार अंक_elements = ggml_nelements(गाठ);
        माप_प्रकार max_log      = std::min(अंक_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        तार्किक    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ADD || op == GGML_OP_GLU);
        दशांश * cpu_src0_float = is_elementwise ? (दशांश *) ctx->cpu_src0_data : nullptr;
        दशांश * cpu_src1_float = is_elementwise ? (दशांश *) ctx->cpu_src1_data : nullptr;

        // Helper to get float value from tensor data (handles f16 and f32)
        स्वतः get_float = [](स्थिरांक शून्य * data, माप_प्रकार idx, ggml_type type) -> दशांश {
            जर (type == GGML_TYPE_F16) {
                स्थिरांक ggml_fp16_t * fp16_data = (स्थिरांक ggml_fp16_t *) data;
                परत ggml_fp16_to_fp32(fp16_data[idx]);
            }

            स्थिरांक दशांश * float_data = (स्थिरांक दशांश *) data;
            परत float_data[idx];
        };

        // Compare all elements but log only the first max_log_elements
        तार्किक   matches          = सत्य;
        माप_प्रकार total_mismatches = 0;

        // First pass: check all elements for mismatches
        साठी (माप_प्रकार i = 0; i < अंक_elements; i++) {
            दशांश cpu_val  = get_float(ctx->cpu_dst_data, i, गाठ->type);
            दशांश et_val   = get_float(ctx->et_dst_data, i, गाठ->type);
            दशांश diff     = fabsf(cpu_val - et_val);
            दशांश rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            जर (rel_diff > config->tolerance) {
                matches = असत्य;
                total_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        साठी (माप_प्रकार i = 0; i < max_log; i++) {
            दशांश cpu_val = get_float(ctx->cpu_dst_data, i, गाठ->type);
            दशांश et_val  = get_float(ctx->et_dst_data, i, गाठ->type);
            दशांश diff    = fabsf(cpu_val - et_val);

            जर (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } अन्यथा जर (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } अन्यथा {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        जर (अंक_elements > max_log) {
            माप_प्रकार mid     = अंक_elements / 2;
            माप_प्रकार end     = अंक_elements - 1;
            दशांश  cpu_mid = get_float(ctx->cpu_dst_data, mid, गाठ->type);
            दशांश  et_mid  = get_float(ctx->et_dst_data, mid, गाठ->type);
            दशांश  cpu_end = get_float(ctx->cpu_dst_data, end, गाठ->type);
            दशांश  et_end  = get_float(ctx->et_dst_data, end, गाठ->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_end, et_end);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       अंक_elements - total_mismatches, अंक_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    जर (config->use_cpu_result) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        माप_प्रकार dst_logical_size = ggml_nbytes(गाठ);
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(गाठ), ctx->cpu_dst_data, 0, dst_logical_size);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    परत सत्य;
}

शून्य ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    जर (!ctx) {
        परत;
    }

    जर (ctx->cpu_src0_data) {
        मुक्त(ctx->cpu_src0_data);
        ctx->cpu_src0_data = nullptr;
    }
    जर (ctx->cpu_src1_data) {
        मुक्त(ctx->cpu_src1_data);
        ctx->cpu_src1_data = nullptr;
    }
    जर (ctx->cpu_src2_data) {
        मुक्त(ctx->cpu_src2_data);
        ctx->cpu_src2_data = nullptr;
    }
    जर (ctx->cpu_dst_data) {
        मुक्त(ctx->cpu_dst_data);
        ctx->cpu_dst_data = nullptr;
    }
    जर (ctx->et_dst_data) {
        मुक्त(ctx->et_dst_data);
        ctx->et_dst_data = nullptr;
    }

    जर (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }

    जर (ctx->cpu_backend) {
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
