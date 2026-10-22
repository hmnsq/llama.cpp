#समावेश "ggml-et-cpu-compare.h"

#समावेश "ggml-cpu/ggml-cpu-impl.h"
#समावेश "ggml-cpu/ops.h"

#समावेश <algorithm>
#समावेश <cmath>
#समावेश <cstdlib>
#समावेश <cstring>

bool ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, स्थिर ggml_tensor * गाँठ, ggml_op op) {
    अगर (!ctx || !गाँठ) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        वापस false;
    }

    // Clear context
    स_रखो(ctx, 0, माप(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    स्वतः get_tensor_बफर_आकार = [](स्थिर ggml_tensor * tensor) -> माप_प्रकार {
        अगर (!tensor) {
            वापस 0;
        }

        अगर (tensor->बफर) {
            // Get actual backend buffer size
            माप_प्रकार बफर_आकार = ggml_backend_बफर_get_आकार(tensor->बफर);

            // Use the full buffer size to avoid any truncation issues
            वापस बफर_आकार;
        } अन्यथा {
            // Fallback to logical size if no buffer
            वापस ggml_nbytes(tensor);
        }
    };

    ctx->src0_आकार = get_tensor_बफर_आकार(गाँठ->src[0]);
    ctx->src1_आकार = get_tensor_बफर_आकार(गाँठ->src[1]);
    ctx->src2_आकार = get_tensor_बफर_आकार(गाँठ->src[2]);
    ctx->dst_आकार  = get_tensor_बफर_आकार(गाँठ);

    // Allocate CPU buffers for all tensors
    अगर (ctx->src0_आकार > 0) {
        ctx->cpu_src0_तथ्य = दो_स्मृति(ctx->src0_आकार);
        अगर (!ctx->cpu_src0_तथ्य) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            जाओ cleanup;
        }
    }

    अगर (ctx->src1_आकार > 0) {
        ctx->cpu_src1_तथ्य = दो_स्मृति(ctx->src1_आकार);
        अगर (!ctx->cpu_src1_तथ्य) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            जाओ cleanup;
        }
    }

    अगर (ctx->src2_आकार > 0) {
        ctx->cpu_src2_तथ्य = दो_स्मृति(ctx->src2_आकार);
        अगर (!ctx->cpu_src2_तथ्य) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            जाओ cleanup;
        }
    }

    ctx->cpu_dst_तथ्य = दो_स्मृति(ctx->dst_आकार);
    अगर (!ctx->cpu_dst_तथ्य) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        जाओ cleanup;
    }

    ctx->et_dst_तथ्य = दो_स्मृति(ctx->dst_आकार);
    अगर (!ctx->et_dst_तथ्य) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        जाओ cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    अगर (ctx->src0_आकार > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        माप_प्रकार logical_आकार = ggml_nbytes(गाँठ->src[0]);
        ggml_backend_tensor_get(गाँठ->src[0], ctx->cpu_src0_तथ्य, 0, logical_आकार);
    }
    अगर (ctx->src1_आकार > 0) {
        माप_प्रकार logical_आकार = ggml_nbytes(गाँठ->src[1]);
        ggml_backend_tensor_get(गाँठ->src[1], ctx->cpu_src1_तथ्य, 0, logical_आकार);
    }
    अगर (ctx->src2_आकार > 0) {
        माप_प्रकार logical_आकार = ggml_nbytes(गाँठ->src[2]);
        ggml_backend_tensor_get(गाँठ->src[2], ctx->cpu_src2_तथ्य, 0, logical_आकार);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        माप_प्रकार logical_आकार = ggml_nbytes(गाँठ);
        ggml_backend_tensor_get(गाँठ, ctx->cpu_dst_तथ्य, 0, logical_आकार);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    अगर (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        जाओ cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_आकार   = ggml_tensor_overhead() * 4 + ggml_graph_overhead();  // up to 4 tensors + graph
    ctx_params.mem_बफर = nullptr;
    ctx_params.no_alloc   = true;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    अगर (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        जाओ cleanup;
    }

    // Create CPU tensors with proper context
    अगर (गाँठ->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, गाँठ->src[0]->type, GGML_MAX_DIMS, गाँठ->src[0]->ne);
        अगर (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            जाओ cleanup;
        }
        ctx->cpu_src0->तथ्य = ctx->cpu_src0_तथ्य;
        // Copy stride array (nb) for correct memory layout
        स_नकल(ctx->cpu_src0->nb, गाँठ->src[0]->nb, माप(गाँठ->src[0]->nb));
        // Copy op_params if present
        स_नकल(ctx->cpu_src0->op_params, गाँठ->src[0]->op_params, माप(गाँठ->src[0]->op_params));
    }

    अगर (गाँठ->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, गाँठ->src[1]->type, GGML_MAX_DIMS, गाँठ->src[1]->ne);
        अगर (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            जाओ cleanup;
        }
        ctx->cpu_src1->तथ्य = ctx->cpu_src1_तथ्य;
        // Copy stride array (nb) for correct memory layout
        स_नकल(ctx->cpu_src1->nb, गाँठ->src[1]->nb, माप(गाँठ->src[1]->nb));
        // Copy op_params if present
        स_नकल(ctx->cpu_src1->op_params, गाँठ->src[1]->op_params, माप(गाँठ->src[1]->op_params));
    }

    अगर (गाँठ->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, गाँठ->src[2]->type, GGML_MAX_DIMS, गाँठ->src[2]->ne);
        अगर (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            जाओ cleanup;
        }
        ctx->cpu_src2->तथ्य = ctx->cpu_src2_तथ्य;
        // Copy stride array (nb) for correct memory layout
        स_नकल(ctx->cpu_src2->nb, गाँठ->src[2]->nb, माप(गाँठ->src[2]->nb));
        // Copy op_params if present
        स_नकल(ctx->cpu_src2->op_params, गाँठ->src[2]->op_params, माप(गाँठ->src[2]->op_params));
    }

    वापस true;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    वापस false;
}

bool ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           स्थिर ggml_tensor *                गाँठ,
                                           स्थिर ggml_et_cpu_compare_config * config) {
    अगर (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !गाँठ || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        वापस false;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = गाँठ->op;
    चयन (op) {
        हाल GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            अवरोध;
        हाल GGML_OP_ADD:
            ctx->cpu_dst = ggml_add(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            अवरोध;
        हाल GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            अवरोध;
        हाल GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            अवरोध;
        हाल GGML_OP_ROPE:
            {
                स्थिर int32_t * op_params  = (स्थिर int32_t *) गाँठ->op_params;
                स्थिर int32_t   n_dims     = op_params[1];
                स्थिर int32_t   mode       = op_params[2];
                स्थिर int32_t   n_ctx_orig = op_params[4];
                स्थिर भग्न     freq_base  = *((स्थिर भग्न *) (op_params + 5));
                स्थिर भग्न     freq_scale = *((स्थिर भग्न *) (op_params + 6));
                स्थिर भग्न     ext_factor = *((स्थिर भग्न *) (op_params + 7));
                स्थिर भग्न     attn_factor = *((स्थिर भग्न *) (op_params + 8));
                स्थिर भग्न     beta_fast  = *((स्थिर भग्न *) (op_params + 9));
                स्थिर भग्न     beta_slow  = *((स्थिर भग्न *) (op_params + 10));

                अगर (mode & GGML_ROPE_TYPE_MROPE) {
                    पूर्णांक sections[GGML_MROPE_SECTIONS];
                    स_नकल(sections, op_params + 11, माप(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
                                                   ext_factor, attn_factor, beta_fast, beta_slow);
                } अन्यथा {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
                                                 attn_factor, beta_fast, beta_slow);
                }
            }
            अवरोध;
        हाल GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                भग्न eps;
                स_नकल(&eps, गाँठ->op_params, माप(भग्न));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            अवरोध;
        हाल GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            अवरोध;
        हाल GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(गाँठ, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            अवरोध;
        हाल GGML_OP_योग_ROWS:
            ctx->cpu_dst = ggml_योग_rows(ctx->ggml_ctx, ctx->cpu_src0);
            अवरोध;
        हाल GGML_OP_MEAN:
            ctx->cpu_dst = ggml_mean(ctx->ggml_ctx, ctx->cpu_src0);
            अवरोध;
        हाल GGML_OP_CLAMP:
            {
                भग्न clamp_min, clamp_max;
                स_नकल(&clamp_min, (स्थिर भग्न *) गाँठ->op_params + 0, माप(भग्न));
                स_नकल(&clamp_max, (स्थिर भग्न *) गाँठ->op_params + 1, माप(भग्न));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            अवरोध;
        हाल GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(गाँठ, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                अगर (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    वापस false;
                }
                अगर (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    स्थिर भग्न limit = ggml_get_op_params_f32(गाँठ, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, limit);
                } अन्यथा {
                    ctx->cpu_dst = ggml_glu_split(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            अवरोध;
        हाल GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                भग्न scale    = 1.0f;
                भग्न max_bias = 0.0f;
                स_नकल(&scale, (स्थिर भग्न *) गाँठ->op_params + 0, माप(भग्न));
                स_नकल(&max_bias, (स्थिर भग्न *) गाँठ->op_params + 1, माप(भग्न));

                अगर (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } अन्यथा {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                अगर (ctx->cpu_src2) {
                    ggml_soft_max_add_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            अवरोध;
        हाल GGML_OP_GET_ROWS:
            ctx->cpu_dst = ggml_get_rows(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            अवरोध;
        हाल GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            अवरोध;
        हाल GGML_OP_SET_ROWS:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_base = ggml_new_tensor(ctx->ggml_ctx, गाँठ->type, GGML_MAX_DIMS, गाँठ->ne);
                अगर (!cpu_dst_base) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    वापस false;
                }
                cpu_dst_base->तथ्य = ctx->cpu_dst_तथ्य;
                स_नकल(cpu_dst_base->nb, गाँठ->nb, माप(गाँठ->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_set_rows(ctx->ggml_ctx, cpu_dst_base, ctx->cpu_src0, ctx->cpu_src1);
            }
            अवरोध;
        शेष:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            वापस false;
    }

    अगर (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        वापस false;
    }

    ctx->cpu_dst->तथ्य = ctx->cpu_dst_तथ्य;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    अगर (op != GGML_OP_CONT) {
        स_नकल(ctx->cpu_dst->nb, गाँठ->nb, माप(गाँठ->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_graph = ggml_new_graph_custom(ctx->ggml_ctx, 1, false);
    अगर (!ctx->cpu_graph) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        वापस false;
    }
    ctx->cpu_graph->nodes[0] = ctx->cpu_dst;
    ctx->cpu_graph->n_nodes  = 1;

    // Log input data for debugging if enabled
    अगर (config && config->log_differences) {
        अगर (ctx->cpu_src0_तथ्य && ctx->src0_आकार >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_तथ्य)[0],
                           ((uint8_t *) ctx->cpu_src0_तथ्य)[1], ((uint8_t *) ctx->cpu_src0_तथ्य)[2],
                           ((uint8_t *) ctx->cpu_src0_तथ्य)[3]);
        }
        अगर (ctx->cpu_src1_तथ्य && ctx->src1_आकार >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((भग्न *) ctx->cpu_src1_तथ्य)[0],
                           ((भग्न *) ctx->cpu_src1_तथ्य)[1], ((भग्न *) ctx->cpu_src1_तथ्य)[2],
                           ((भग्न *) ctx->cpu_src1_तथ्य)[3]);
        }
    }

    // Compute using CPU backend
    ggml_status cpu_परिणाम = ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph);

    अगर (cpu_परिणाम != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_परिणाम);
        वापस false;
    }

    // Log output data for debugging if enabled
    अगर (config && config->log_differences && ctx->dst_आकार >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((भग्न *) ctx->cpu_dst_तथ्य)[0], ((भग्न *) ctx->cpu_dst_तथ्य)[1],
                       ((भग्न *) ctx->cpu_dst_तथ्य)[2], ((भग्न *) ctx->cpu_dst_तथ्य)[3]);
    }

    // Now copy ET device destination to host for comparison
    माप_प्रकार dst_logical_आकार = ggml_nbytes(गाँठ);
    ggml_backend_tensor_get(गाँठ, ctx->et_dst_तथ्य, 0, dst_logical_आकार);

    अगर (config->log_differences) {
        माप_प्रकार अंक_elements = ggml_nelements(गाँठ);
        माप_प्रकार max_log      = std::min(अंक_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        bool    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ADD || op == GGML_OP_GLU);
        भग्न * cpu_src0_float = is_elementwise ? (भग्न *) ctx->cpu_src0_तथ्य : nullptr;
        भग्न * cpu_src1_float = is_elementwise ? (भग्न *) ctx->cpu_src1_तथ्य : nullptr;

        // Helper to get float value from tensor data (handles f16 and f32)
        स्वतः get_float = [](स्थिर व्योम * तथ्य, माप_प्रकार idx, ggml_type type) -> भग्न {
            अगर (type == GGML_TYPE_F16) {
                स्थिर ggml_fp16_t * fp16_तथ्य = (स्थिर ggml_fp16_t *) तथ्य;
                वापस ggml_fp16_to_fp32(fp16_तथ्य[idx]);
            }

            स्थिर भग्न * float_तथ्य = (स्थिर भग्न *) तथ्य;
            वापस float_तथ्य[idx];
        };

        // Compare all elements but log only the first max_log_elements
        bool   matches          = true;
        माप_प्रकार योग_mismatches = 0;

        // First pass: check all elements for mismatches
        क्रम (माप_प्रकार i = 0; i < अंक_elements; i++) {
            भग्न cpu_val  = get_float(ctx->cpu_dst_तथ्य, i, गाँठ->type);
            भग्न et_val   = get_float(ctx->et_dst_तथ्य, i, गाँठ->type);
            भग्न diff     = fabsf(cpu_val - et_val);
            भग्न rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            अगर (rel_diff > config->tolerance) {
                matches = false;
                योग_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        क्रम (माप_प्रकार i = 0; i < max_log; i++) {
            भग्न cpu_val = get_float(ctx->cpu_dst_तथ्य, i, गाँठ->type);
            भग्न et_val  = get_float(ctx->et_dst_तथ्य, i, गाँठ->type);
            भग्न diff    = fabsf(cpu_val - et_val);

            अगर (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } अन्यथा अगर (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } अन्यथा {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        अगर (अंक_elements > max_log) {
            माप_प्रकार mid     = अंक_elements / 2;
            माप_प्रकार end     = अंक_elements - 1;
            भग्न  cpu_mid = get_float(ctx->cpu_dst_तथ्य, mid, गाँठ->type);
            भग्न  et_mid  = get_float(ctx->et_dst_तथ्य, mid, गाँठ->type);
            भग्न  cpu_end = get_float(ctx->cpu_dst_तथ्य, end, गाँठ->type);
            भग्न  et_end  = get_float(ctx->et_dst_तथ्य, end, गाँठ->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_end, et_end);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       अंक_elements - योग_mismatches, अंक_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    अगर (config->use_cpu_परिणाम) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        माप_प्रकार dst_logical_आकार = ggml_nbytes(गाँठ);
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(गाँठ), ctx->cpu_dst_तथ्य, 0, dst_logical_आकार);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    वापस true;
}

व्योम ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    अगर (!ctx) {
        वापस;
    }

    अगर (ctx->cpu_src0_तथ्य) {
        मुक्त(ctx->cpu_src0_तथ्य);
        ctx->cpu_src0_तथ्य = nullptr;
    }
    अगर (ctx->cpu_src1_तथ्य) {
        मुक्त(ctx->cpu_src1_तथ्य);
        ctx->cpu_src1_तथ्य = nullptr;
    }
    अगर (ctx->cpu_src2_तथ्य) {
        मुक्त(ctx->cpu_src2_तथ्य);
        ctx->cpu_src2_तथ्य = nullptr;
    }
    अगर (ctx->cpu_dst_तथ्य) {
        मुक्त(ctx->cpu_dst_तथ्य);
        ctx->cpu_dst_तथ्य = nullptr;
    }
    अगर (ctx->et_dst_तथ्य) {
        मुक्त(ctx->et_dst_तथ्य);
        ctx->et_dst_तथ्य = nullptr;
    }

    अगर (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }

    अगर (ctx->cpu_backend) {
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
