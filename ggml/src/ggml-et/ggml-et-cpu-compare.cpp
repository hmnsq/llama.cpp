#அந்தர்பாவ "ggml-et-cpu-compare.h"

#அந்தர்பாவ "ggml-cpu/ggml-cpu-impl.h"
#அந்தர்பாவ "ggml-cpu/ops.h"

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cmath>
#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <cstring>

தருக்க ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, மாறிலி ggml_tensor * முனை, ggml_op op) {
    எனில் (!ctx || !முனை) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        திரும்பு பொய்;
    }

    // Clear context
    memset(ctx, 0, அளவு(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    auto get_tensor_buffer_size = [](மாறிலி ggml_tensor * tensor) -> size_t {
        எனில் (!tensor) {
            திரும்பு 0;
        }

        எனில் (tensor->buffer) {
            // Get actual backend buffer size
            size_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);

            // Use the full buffer size to avoid any truncation issues
            திரும்பு buffer_size;
        } இல்லையெனில் {
            // Fallback to logical size if no buffer
            திரும்பு ggml_nbytes(tensor);
        }
    };

    ctx->src0_size = get_tensor_buffer_size(முனை->src[0]);
    ctx->src1_size = get_tensor_buffer_size(முனை->src[1]);
    ctx->src2_size = get_tensor_buffer_size(முனை->src[2]);
    ctx->dst_size  = get_tensor_buffer_size(முனை);

    // Allocate CPU buffers for all tensors
    எனில் (ctx->src0_size > 0) {
        ctx->cpu_src0_data = malloc(ctx->src0_size);
        எனில் (!ctx->cpu_src0_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            செல் cleanup;
        }
    }

    எனில் (ctx->src1_size > 0) {
        ctx->cpu_src1_data = malloc(ctx->src1_size);
        எனில் (!ctx->cpu_src1_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            செல் cleanup;
        }
    }

    எனில் (ctx->src2_size > 0) {
        ctx->cpu_src2_data = malloc(ctx->src2_size);
        எனில் (!ctx->cpu_src2_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            செல் cleanup;
        }
    }

    ctx->cpu_dst_data = malloc(ctx->dst_size);
    எனில் (!ctx->cpu_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        செல் cleanup;
    }

    ctx->et_dst_data = malloc(ctx->dst_size);
    எனில் (!ctx->et_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        செல் cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    எனில் (ctx->src0_size > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_size = ggml_nbytes(முனை->src[0]);
        ggml_backend_tensor_get(முனை->src[0], ctx->cpu_src0_data, 0, logical_size);
    }
    எனில் (ctx->src1_size > 0) {
        size_t logical_size = ggml_nbytes(முனை->src[1]);
        ggml_backend_tensor_get(முனை->src[1], ctx->cpu_src1_data, 0, logical_size);
    }
    எனில் (ctx->src2_size > 0) {
        size_t logical_size = ggml_nbytes(முனை->src[2]);
        ggml_backend_tensor_get(முனை->src[2], ctx->cpu_src2_data, 0, logical_size);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_size = ggml_nbytes(முனை);
        ggml_backend_tensor_get(முனை, ctx->cpu_dst_data, 0, logical_size);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    எனில் (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        செல் cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_size   = ggml_tensor_overhead() * 4 + ggml_graph_overhead();  // up to 4 tensors + graph
    ctx_params.mem_buffer = nullptr;
    ctx_params.no_alloc   = உண்மை;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    எனில் (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        செல் cleanup;
    }

    // Create CPU tensors with proper context
    எனில் (முனை->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, முனை->src[0]->type, GGML_MAX_DIMS, முனை->src[0]->ne);
        எனில் (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            செல் cleanup;
        }
        ctx->cpu_src0->data = ctx->cpu_src0_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src0->nb, முனை->src[0]->nb, அளவு(முனை->src[0]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src0->op_params, முனை->src[0]->op_params, அளவு(முனை->src[0]->op_params));
    }

    எனில் (முனை->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, முனை->src[1]->type, GGML_MAX_DIMS, முனை->src[1]->ne);
        எனில் (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            செல் cleanup;
        }
        ctx->cpu_src1->data = ctx->cpu_src1_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src1->nb, முனை->src[1]->nb, அளவு(முனை->src[1]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src1->op_params, முனை->src[1]->op_params, அளவு(முனை->src[1]->op_params));
    }

    எனில் (முனை->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, முனை->src[2]->type, GGML_MAX_DIMS, முனை->src[2]->ne);
        எனில் (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            செல் cleanup;
        }
        ctx->cpu_src2->data = ctx->cpu_src2_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src2->nb, முனை->src[2]->nb, அளவு(முனை->src[2]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src2->op_params, முனை->src[2]->op_params, அளவு(முனை->src[2]->op_params));
    }

    திரும்பு உண்மை;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    திரும்பு பொய்;
}

தருக்க ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           மாறிலி ggml_tensor *                முனை,
                                           மாறிலி ggml_et_cpu_compare_config * config) {
    எனில் (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !முனை || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        திரும்பு பொய்;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = முனை->op;
    தேர்வு (op) {
        நிலை GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            நிறுத்து;
        நிலை GGML_OP_ADD:
            ctx->cpu_dst = ggml_add(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            நிறுத்து;
        நிலை GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            நிறுத்து;
        நிலை GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            நிறுத்து;
        நிலை GGML_OP_ROPE:
            {
                மாறிலி int32_t * op_params  = (மாறிலி int32_t *) முனை->op_params;
                மாறிலி int32_t   n_dims     = op_params[1];
                மாறிலி int32_t   mode       = op_params[2];
                மாறிலி int32_t   n_ctx_orig = op_params[4];
                மாறிலி தசம     freq_base  = *((மாறிலி தசம *) (op_params + 5));
                மாறிலி தசம     freq_scale = *((மாறிலி தசம *) (op_params + 6));
                மாறிலி தசம     ext_factor = *((மாறிலி தசம *) (op_params + 7));
                மாறிலி தசம     attn_factor = *((மாறிலி தசம *) (op_params + 8));
                மாறிலி தசம     beta_fast  = *((மாறிலி தசம *) (op_params + 9));
                மாறிலி தசம     beta_slow  = *((மாறிலி தசம *) (op_params + 10));

                எனில் (mode & GGML_ROPE_TYPE_MROPE) {
                    முழு sections[GGML_MROPE_SECTIONS];
                    memcpy(sections, op_params + 11, அளவு(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
                                                   ext_factor, attn_factor, beta_fast, beta_slow);
                } இல்லையெனில் {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
                                                 attn_factor, beta_fast, beta_slow);
                }
            }
            நிறுத்து;
        நிலை GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                தசம eps;
                memcpy(&eps, முனை->op_params, அளவு(தசம));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            நிறுத்து;
        நிலை GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            நிறுத்து;
        நிலை GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(முனை, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            நிறுத்து;
        நிலை GGML_OP_SUM_ROWS:
            ctx->cpu_dst = ggml_sum_rows(ctx->ggml_ctx, ctx->cpu_src0);
            நிறுத்து;
        நிலை GGML_OP_MEAN:
            ctx->cpu_dst = ggml_mean(ctx->ggml_ctx, ctx->cpu_src0);
            நிறுத்து;
        நிலை GGML_OP_CLAMP:
            {
                தசம clamp_min, clamp_max;
                memcpy(&clamp_min, (மாறிலி தசம *) முனை->op_params + 0, அளவு(தசம));
                memcpy(&clamp_max, (மாறிலி தசம *) முனை->op_params + 1, அளவு(தசம));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            நிறுத்து;
        நிலை GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(முனை, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                எனில் (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    திரும்பு பொய்;
                }
                எனில் (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    மாறிலி தசம limit = ggml_get_op_params_f32(முனை, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, limit);
                } இல்லையெனில் {
                    ctx->cpu_dst = ggml_glu_split(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            நிறுத்து;
        நிலை GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                தசம scale    = 1.0f;
                தசம max_bias = 0.0f;
                memcpy(&scale, (மாறிலி தசம *) முனை->op_params + 0, அளவு(தசம));
                memcpy(&max_bias, (மாறிலி தசம *) முனை->op_params + 1, அளவு(தசம));

                எனில் (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } இல்லையெனில் {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                எனில் (ctx->cpu_src2) {
                    ggml_soft_max_add_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            நிறுத்து;
        நிலை GGML_OP_GET_ROWS:
            ctx->cpu_dst = ggml_get_rows(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            நிறுத்து;
        நிலை GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            நிறுத்து;
        நிலை GGML_OP_SET_ROWS:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_base = ggml_new_tensor(ctx->ggml_ctx, முனை->type, GGML_MAX_DIMS, முனை->ne);
                எனில் (!cpu_dst_base) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    திரும்பு பொய்;
                }
                cpu_dst_base->data = ctx->cpu_dst_data;
                memcpy(cpu_dst_base->nb, முனை->nb, அளவு(முனை->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_set_rows(ctx->ggml_ctx, cpu_dst_base, ctx->cpu_src0, ctx->cpu_src1);
            }
            நிறுத்து;
        இயல்பு:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            திரும்பு பொய்;
    }

    எனில் (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        திரும்பு பொய்;
    }

    ctx->cpu_dst->data = ctx->cpu_dst_data;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    எனில் (op != GGML_OP_CONT) {
        memcpy(ctx->cpu_dst->nb, முனை->nb, அளவு(முனை->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_graph = ggml_new_graph_custom(ctx->ggml_ctx, 1, பொய்);
    எனில் (!ctx->cpu_graph) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        திரும்பு பொய்;
    }
    ctx->cpu_graph->nodes[0] = ctx->cpu_dst;
    ctx->cpu_graph->n_nodes  = 1;

    // Log input data for debugging if enabled
    எனில் (config && config->log_differences) {
        எனில் (ctx->cpu_src0_data && ctx->src0_size >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_data)[0],
                           ((uint8_t *) ctx->cpu_src0_data)[1], ((uint8_t *) ctx->cpu_src0_data)[2],
                           ((uint8_t *) ctx->cpu_src0_data)[3]);
        }
        எனில் (ctx->cpu_src1_data && ctx->src1_size >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((தசம *) ctx->cpu_src1_data)[0],
                           ((தசம *) ctx->cpu_src1_data)[1], ((தசம *) ctx->cpu_src1_data)[2],
                           ((தசம *) ctx->cpu_src1_data)[3]);
        }
    }

    // Compute using CPU backend
    ggml_status cpu_result = ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph);

    எனில் (cpu_result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_result);
        திரும்பு பொய்;
    }

    // Log output data for debugging if enabled
    எனில் (config && config->log_differences && ctx->dst_size >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((தசம *) ctx->cpu_dst_data)[0], ((தசம *) ctx->cpu_dst_data)[1],
                       ((தசம *) ctx->cpu_dst_data)[2], ((தசம *) ctx->cpu_dst_data)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_size = ggml_nbytes(முனை);
    ggml_backend_tensor_get(முனை, ctx->et_dst_data, 0, dst_logical_size);

    எனில் (config->log_differences) {
        size_t இலக்கம்_elements = ggml_nelements(முனை);
        size_t max_log      = std::min(இலக்கம்_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        தருக்க    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ADD || op == GGML_OP_GLU);
        தசம * cpu_src0_float = is_elementwise ? (தசம *) ctx->cpu_src0_data : nullptr;
        தசம * cpu_src1_float = is_elementwise ? (தசம *) ctx->cpu_src1_data : nullptr;

        // Helper to get float value from tensor data (handles f16 and f32)
        auto get_float = [](மாறிலி வெறுமை * data, size_t idx, ggml_type type) -> தசம {
            எனில் (type == GGML_TYPE_F16) {
                மாறிலி ggml_fp16_t * fp16_data = (மாறிலி ggml_fp16_t *) data;
                திரும்பு ggml_fp16_to_fp32(fp16_data[idx]);
            }

            மாறிலி தசம * float_data = (மாறிலி தசம *) data;
            திரும்பு float_data[idx];
        };

        // Compare all elements but log only the first max_log_elements
        தருக்க   matches          = உண்மை;
        size_t total_mismatches = 0;

        // First pass: check all elements for mismatches
        க்கு (size_t i = 0; i < இலக்கம்_elements; i++) {
            தசம cpu_val  = get_float(ctx->cpu_dst_data, i, முனை->type);
            தசம et_val   = get_float(ctx->et_dst_data, i, முனை->type);
            தசம diff     = fabsf(cpu_val - et_val);
            தசம rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            எனில் (rel_diff > config->tolerance) {
                matches = பொய்;
                total_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        க்கு (size_t i = 0; i < max_log; i++) {
            தசம cpu_val = get_float(ctx->cpu_dst_data, i, முனை->type);
            தசம et_val  = get_float(ctx->et_dst_data, i, முனை->type);
            தசம diff    = fabsf(cpu_val - et_val);

            எனில் (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } இல்லையெனில் எனில் (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } இல்லையெனில் {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        எனில் (இலக்கம்_elements > max_log) {
            size_t mid     = இலக்கம்_elements / 2;
            size_t end     = இலக்கம்_elements - 1;
            தசம  cpu_mid = get_float(ctx->cpu_dst_data, mid, முனை->type);
            தசம  et_mid  = get_float(ctx->et_dst_data, mid, முனை->type);
            தசம  cpu_end = get_float(ctx->cpu_dst_data, end, முனை->type);
            தசம  et_end  = get_float(ctx->et_dst_data, end, முனை->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_end, et_end);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       இலக்கம்_elements - total_mismatches, இலக்கம்_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    எனில் (config->use_cpu_result) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_size = ggml_nbytes(முனை);
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(முனை), ctx->cpu_dst_data, 0, dst_logical_size);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    திரும்பு உண்மை;
}

வெறுமை ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    எனில் (!ctx) {
        திரும்பு;
    }

    எனில் (ctx->cpu_src0_data) {
        விடுவி(ctx->cpu_src0_data);
        ctx->cpu_src0_data = nullptr;
    }
    எனில் (ctx->cpu_src1_data) {
        விடுவி(ctx->cpu_src1_data);
        ctx->cpu_src1_data = nullptr;
    }
    எனில் (ctx->cpu_src2_data) {
        விடுவி(ctx->cpu_src2_data);
        ctx->cpu_src2_data = nullptr;
    }
    எனில் (ctx->cpu_dst_data) {
        விடுவி(ctx->cpu_dst_data);
        ctx->cpu_dst_data = nullptr;
    }
    எனில் (ctx->et_dst_data) {
        விடுவி(ctx->et_dst_data);
        ctx->et_dst_data = nullptr;
    }

    எனில் (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }

    எனில் (ctx->cpu_backend) {
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
