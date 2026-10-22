#अन्तर्भाव "ggml-et-cpu-compare.h"

#अन्तर्भाव "ggml-cpu/ggml-cpu-impl.h"
#अन्तर्भाव "ggml-cpu/ops.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cmath>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>

तार्किक ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ध्रुव ggml_tensor * ग्रन्थिः, ggml_op op) {
    यदि (!ctx || !ग्रन्थिः) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        प्रत्यागम असत्य;
    }

    // Clear context
    स्मृतिं_पूरय(ctx, 0, परिमाण(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    स्वचालित get_tensor_बफर_आकार = [](ध्रुव ggml_tensor * tensor) -> size_t {
        यदि (!tensor) {
            प्रत्यागम 0;
        }

        यदि (tensor->बफर) {
            // Get actual backend buffer size
            size_t बफर_आकार = ggml_backend_बफर_get_आकार(tensor->बफर);

            // Use the full buffer size to avoid any truncation issues
            प्रत्यागम बफर_आकार;
        } अन्यथा {
            // Fallback to logical size if no buffer
            प्रत्यागम ggml_nbytes(tensor);
        }
    };

    ctx->src0_आकार = get_tensor_बफर_आकार(ग्रन्थिः->src[0]);
    ctx->src1_आकार = get_tensor_बफर_आकार(ग्रन्थिः->src[1]);
    ctx->src2_आकार = get_tensor_बफर_आकार(ग्रन्थिः->src[2]);
    ctx->dst_आकार  = get_tensor_बफर_आकार(ग्रन्थिः);

    // Allocate CPU buffers for all tensors
    यदि (ctx->src0_आकार > 0) {
        ctx->cpu_src0_दत्त = स्मृतिं_देहि(ctx->src0_आकार);
        यदि (!ctx->cpu_src0_दत्त) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            गच्छ cleanup;
        }
    }

    यदि (ctx->src1_आकार > 0) {
        ctx->cpu_src1_दत्त = स्मृतिं_देहि(ctx->src1_आकार);
        यदि (!ctx->cpu_src1_दत्त) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            गच्छ cleanup;
        }
    }

    यदि (ctx->src2_आकार > 0) {
        ctx->cpu_src2_दत्त = स्मृतिं_देहि(ctx->src2_आकार);
        यदि (!ctx->cpu_src2_दत्त) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            गच्छ cleanup;
        }
    }

    ctx->cpu_dst_दत्त = स्मृतिं_देहि(ctx->dst_आकार);
    यदि (!ctx->cpu_dst_दत्त) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        गच्छ cleanup;
    }

    ctx->et_dst_दत्त = स्मृतिं_देहि(ctx->dst_आकार);
    यदि (!ctx->et_dst_दत्त) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        गच्छ cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    यदि (ctx->src0_आकार > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_आकार = ggml_nbytes(ग्रन्थिः->src[0]);
        ggml_backend_tensor_get(ग्रन्थिः->src[0], ctx->cpu_src0_दत्त, 0, logical_आकार);
    }
    यदि (ctx->src1_आकार > 0) {
        size_t logical_आकार = ggml_nbytes(ग्रन्थिः->src[1]);
        ggml_backend_tensor_get(ग्रन्थिः->src[1], ctx->cpu_src1_दत्त, 0, logical_आकार);
    }
    यदि (ctx->src2_आकार > 0) {
        size_t logical_आकार = ggml_nbytes(ग्रन्थिः->src[2]);
        ggml_backend_tensor_get(ग्रन्थिः->src[2], ctx->cpu_src2_दत्त, 0, logical_आकार);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_आकार = ggml_nbytes(ग्रन्थिः);
        ggml_backend_tensor_get(ग्रन्थिः, ctx->cpu_dst_दत्त, 0, logical_आकार);
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
    ctx_params.mem_आकार   = ggml_tensor_overhead() * 4 + ggml_graph_overhead();  // up to 4 tensors + graph
    ctx_params.mem_बफर = शून्यसूचकः;
    ctx_params.no_alloc   = सत्य;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    यदि (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        गच्छ cleanup;
    }

    // Create CPU tensors with proper context
    यदि (ग्रन्थिः->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, ग्रन्थिः->src[0]->type, GGML_MAX_DIMS, ग्रन्थिः->src[0]->ne);
        यदि (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            गच्छ cleanup;
        }
        ctx->cpu_src0->दत्त = ctx->cpu_src0_दत्त;
        // Copy stride array (nb) for correct memory layout
        स्मृतिं_प्रतिलिख(ctx->cpu_src0->nb, ग्रन्थिः->src[0]->nb, परिमाण(ग्रन्थिः->src[0]->nb));
        // Copy op_params if present
        स्मृतिं_प्रतिलिख(ctx->cpu_src0->op_params, ग्रन्थिः->src[0]->op_params, परिमाण(ग्रन्थिः->src[0]->op_params));
    }

    यदि (ग्रन्थिः->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, ग्रन्थिः->src[1]->type, GGML_MAX_DIMS, ग्रन्थिः->src[1]->ne);
        यदि (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            गच्छ cleanup;
        }
        ctx->cpu_src1->दत्त = ctx->cpu_src1_दत्त;
        // Copy stride array (nb) for correct memory layout
        स्मृतिं_प्रतिलिख(ctx->cpu_src1->nb, ग्रन्थिः->src[1]->nb, परिमाण(ग्रन्थिः->src[1]->nb));
        // Copy op_params if present
        स्मृतिं_प्रतिलिख(ctx->cpu_src1->op_params, ग्रन्थिः->src[1]->op_params, परिमाण(ग्रन्थिः->src[1]->op_params));
    }

    यदि (ग्रन्थिः->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, ग्रन्थिः->src[2]->type, GGML_MAX_DIMS, ग्रन्थिः->src[2]->ne);
        यदि (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            गच्छ cleanup;
        }
        ctx->cpu_src2->दत्त = ctx->cpu_src2_दत्त;
        // Copy stride array (nb) for correct memory layout
        स्मृतिं_प्रतिलिख(ctx->cpu_src2->nb, ग्रन्थिः->src[2]->nb, परिमाण(ग्रन्थिः->src[2]->nb));
        // Copy op_params if present
        स्मृतिं_प्रतिलिख(ctx->cpu_src2->op_params, ग्रन्थिः->src[2]->op_params, परिमाण(ग्रन्थिः->src[2]->op_params));
    }

    प्रत्यागम सत्य;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    प्रत्यागम असत्य;
}

तार्किक ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ध्रुव ggml_tensor *                ग्रन्थिः,
                                           ध्रुव ggml_et_cpu_compare_config * config) {
    यदि (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !ग्रन्थिः || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        प्रत्यागम असत्य;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = ग्रन्थिः->op;
    विकल्प (op) {
        पक्ष GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            भङ्ग;
        पक्ष GGML_OP_सङ्कलय:
            ctx->cpu_dst = ggml_सङ्कलय(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
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
                ध्रुव भिन्न     freq_आधारः  = *((ध्रुव भिन्न *) (op_params + 5));
                ध्रुव भिन्न     freq_scale = *((ध्रुव भिन्न *) (op_params + 6));
                ध्रुव भिन्न     ext_गुणकः = *((ध्रुव भिन्न *) (op_params + 7));
                ध्रुव भिन्न     attn_गुणकः = *((ध्रुव भिन्न *) (op_params + 8));
                ध्रुव भिन्न     beta_fast  = *((ध्रुव भिन्न *) (op_params + 9));
                ध्रुव भिन्न     beta_slow  = *((ध्रुव भिन्न *) (op_params + 10));

                यदि (mode & GGML_ROPE_TYPE_MROPE) {
                    पूर्णाङ्क sections[GGML_MROPE_SECTIONS];
                    स्मृतिं_प्रतिलिख(sections, op_params + 11, परिमाण(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_आधारः, freq_scale,
                                                   ext_गुणकः, attn_गुणकः, beta_fast, beta_slow);
                } अन्यथा {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_आधारः, freq_scale, ext_गुणकः,
                                                 attn_गुणकः, beta_fast, beta_slow);
                }
            }
            भङ्ग;
        पक्ष GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                भिन्न eps;
                स्मृतिं_प्रतिलिख(&eps, ग्रन्थिः->op_params, परिमाण(भिन्न));
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
        पक्ष GGML_OP_योग_पङ्क्तिकाः:
            ctx->cpu_dst = ggml_योग_पङ्क्तिकाः(ctx->ggml_ctx, ctx->cpu_src0);
            भङ्ग;
        पक्ष GGML_OP_माध्यम्:
            ctx->cpu_dst = ggml_माध्यम्(ctx->ggml_ctx, ctx->cpu_src0);
            भङ्ग;
        पक्ष GGML_OP_CLAMP:
            {
                भिन्न clamp_min, clamp_max;
                स्मृतिं_प्रतिलिख(&clamp_min, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 0, परिमाण(भिन्न));
                स्मृतिं_प्रतिलिख(&clamp_max, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 1, परिमाण(भिन्न));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            भङ्ग;
        पक्ष GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(ग्रन्थिः, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                यदि (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    प्रत्यागम असत्य;
                }
                यदि (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    ध्रुव भिन्न सीमा = ggml_get_op_params_f32(ग्रन्थिः, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, सीमा);
                } अन्यथा {
                    ctx->cpu_dst = ggml_glu_विभज(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            भङ्ग;
        पक्ष GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                भिन्न scale    = 1.0f;
                भिन्न max_bias = 0.0f;
                स्मृतिं_प्रतिलिख(&scale, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 0, परिमाण(भिन्न));
                स्मृतिं_प्रतिलिख(&max_bias, (ध्रुव भिन्न *) ग्रन्थिः->op_params + 1, परिमाण(भिन्न));

                यदि (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } अन्यथा {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                यदि (ctx->cpu_src2) {
                    ggml_soft_max_सङ्कलय_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            भङ्ग;
        पक्ष GGML_OP_GET_पङ्क्तिकाः:
            ctx->cpu_dst = ggml_get_पङ्क्तिकाः(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            भङ्ग;
        पक्ष GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            भङ्ग;
        पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_आधारः = ggml_new_tensor(ctx->ggml_ctx, ग्रन्थिः->type, GGML_MAX_DIMS, ग्रन्थिः->ne);
                यदि (!cpu_dst_आधारः) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    प्रत्यागम असत्य;
                }
                cpu_dst_आधारः->दत्त = ctx->cpu_dst_दत्त;
                स्मृतिं_प्रतिलिख(cpu_dst_आधारः->nb, ग्रन्थिः->nb, परिमाण(ग्रन्थिः->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_समुच्चयः_पङ्क्तिकाः(ctx->ggml_ctx, cpu_dst_आधारः, ctx->cpu_src0, ctx->cpu_src1);
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

    ctx->cpu_dst->दत्त = ctx->cpu_dst_दत्त;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    यदि (op != GGML_OP_CONT) {
        स्मृतिं_प्रतिलिख(ctx->cpu_dst->nb, ग्रन्थिः->nb, परिमाण(ग्रन्थिः->nb));
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
        यदि (ctx->cpu_src0_दत्त && ctx->src0_आकार >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_दत्त)[0],
                           ((uint8_t *) ctx->cpu_src0_दत्त)[1], ((uint8_t *) ctx->cpu_src0_दत्त)[2],
                           ((uint8_t *) ctx->cpu_src0_दत्त)[3]);
        }
        यदि (ctx->cpu_src1_दत्त && ctx->src1_आकार >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((भिन्न *) ctx->cpu_src1_दत्त)[0],
                           ((भिन्न *) ctx->cpu_src1_दत्त)[1], ((भिन्न *) ctx->cpu_src1_दत्त)[2],
                           ((भिन्न *) ctx->cpu_src1_दत्त)[3]);
        }
    }

    // Compute using CPU backend
    ggml_स्थितिः cpu_परिणाम = ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph);

    यदि (cpu_परिणाम != GGML_स्थितिः_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_परिणाम);
        प्रत्यागम असत्य;
    }

    // Log output data for debugging if enabled
    यदि (config && config->log_differences && ctx->dst_आकार >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((भिन्न *) ctx->cpu_dst_दत्त)[0], ((भिन्न *) ctx->cpu_dst_दत्त)[1],
                       ((भिन्न *) ctx->cpu_dst_दत्त)[2], ((भिन्न *) ctx->cpu_dst_दत्त)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_आकार = ggml_nbytes(ग्रन्थिः);
    ggml_backend_tensor_get(ग्रन्थिः, ctx->et_dst_दत्त, 0, dst_logical_आकार);

    यदि (config->log_differences) {
        size_t अङ्कः_elements = ggml_nelements(ग्रन्थिः);
        size_t max_log      = std::min(अङ्कः_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        तार्किक    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_सङ्कलय || op == GGML_OP_GLU);
        भिन्न * cpu_src0_float = is_elementwise ? (भिन्न *) ctx->cpu_src0_दत्त : शून्यसूचकः;
        भिन्न * cpu_src1_float = is_elementwise ? (भिन्न *) ctx->cpu_src1_दत्त : शून्यसूचकः;

        // Helper to get float value from tensor data (handles f16 and f32)
        स्वचालित get_float = [](ध्रुव शून्य * दत्त, size_t idx, ggml_type type) -> भिन्न {
            यदि (type == GGML_TYPE_F16) {
                ध्रुव ggml_fp16_t * fp16_दत्त = (ध्रुव ggml_fp16_t *) दत्त;
                प्रत्यागम ggml_fp16_गन्तव्यम्_fp32(fp16_दत्त[idx]);
            }

            ध्रुव भिन्न * float_दत्त = (ध्रुव भिन्न *) दत्त;
            प्रत्यागम float_दत्त[idx];
        };

        // Compare all elements but log only the first max_log_elements
        तार्किक   matches          = सत्य;
        size_t योग_mismatches = 0;

        // First pass: check all elements for mismatches
        प्रति (size_t i = 0; i < अङ्कः_elements; i++) {
            भिन्न cpu_val  = get_float(ctx->cpu_dst_दत्त, i, ग्रन्थिः->type);
            भिन्न et_val   = get_float(ctx->et_dst_दत्त, i, ग्रन्थिः->type);
            भिन्न diff     = fabsf(cpu_val - et_val);
            भिन्न rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            यदि (rel_diff > config->tolerance) {
                matches = असत्य;
                योग_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        प्रति (size_t i = 0; i < max_log; i++) {
            भिन्न cpu_val = get_float(ctx->cpu_dst_दत्त, i, ग्रन्थिः->type);
            भिन्न et_val  = get_float(ctx->et_dst_दत्त, i, ग्रन्थिः->type);
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
        यदि (अङ्कः_elements > max_log) {
            size_t mid     = अङ्कः_elements / 2;
            size_t end     = अङ्कः_elements - 1;
            भिन्न  cpu_mid = get_float(ctx->cpu_dst_दत्त, mid, ग्रन्थिः->type);
            भिन्न  et_mid  = get_float(ctx->et_dst_दत्त, mid, ग्रन्थिः->type);
            भिन्न  cpu_अन्तः = get_float(ctx->cpu_dst_दत्त, end, ग्रन्थिः->type);
            भिन्न  et_अन्तः  = get_float(ctx->et_dst_दत्त, end, ग्रन्थिः->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_अन्तः, et_अन्तः);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       अङ्कः_elements - योग_mismatches, अङ्कः_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    यदि (config->use_cpu_परिणाम) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_आकार = ggml_nbytes(ग्रन्थिः);
        ggml_backend_tensor_समुच्चयः(const_cast<ggml_tensor *>(ग्रन्थिः), ctx->cpu_dst_दत्त, 0, dst_logical_आकार);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    प्रत्यागम सत्य;
}

शून्य ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    यदि (!ctx) {
        प्रत्यागम;
    }

    यदि (ctx->cpu_src0_दत्त) {
        मुक्त(ctx->cpu_src0_दत्त);
        ctx->cpu_src0_दत्त = शून्यसूचकः;
    }
    यदि (ctx->cpu_src1_दत्त) {
        मुक्त(ctx->cpu_src1_दत्त);
        ctx->cpu_src1_दत्त = शून्यसूचकः;
    }
    यदि (ctx->cpu_src2_दत्त) {
        मुक्त(ctx->cpu_src2_दत्त);
        ctx->cpu_src2_दत्त = शून्यसूचकः;
    }
    यदि (ctx->cpu_dst_दत्त) {
        मुक्त(ctx->cpu_dst_दत्त);
        ctx->cpu_dst_दत्त = शून्यसूचकः;
    }
    यदि (ctx->et_dst_दत्त) {
        मुक्त(ctx->et_dst_दत्त);
        ctx->et_dst_दत्त = शून्यसूचकः;
    }

    यदि (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = शून्यसूचकः;
    }

    यदि (ctx->cpu_backend) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = शून्यसूचकः;
    }

    // Clear pointers
    ctx->cpu_src0  = शून्यसूचकः;
    ctx->cpu_src1  = शून्यसूचकः;
    ctx->cpu_src2  = शून्यसूचकः;
    ctx->cpu_dst   = शून्यसूचकः;
    ctx->cpu_graph = शून्यसूचकः;
}
