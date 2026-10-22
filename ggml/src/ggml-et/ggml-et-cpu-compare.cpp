#شامل "ggml-et-cpu-compare.h"

#شامل "ggml-cpu/ggml-cpu-impl.h"
#شامل "ggml-cpu/ops.h"

#شامل <algorithm>
#شامل <cmath>
#شامل <cstdlib>
#شامل <cstring>

منطقی ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ثابت ggml_tensor * گره, ggml_op op) {
    اگر (!ctx || !گره) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        بازگشت نادرست;
    }

    // Clear context
    memset(ctx, 0, اندازه(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    auto get_tensor_buffer_size = [](ثابت ggml_tensor * tensor) -> size_t {
        اگر (!tensor) {
            بازگشت 0;
        }

        اگر (tensor->buffer) {
            // Get actual backend buffer size
            size_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);

            // Use the full buffer size to avoid any truncation issues
            بازگشت buffer_size;
        } وگرنه {
            // Fallback to logical size if no buffer
            بازگشت ggml_nbytes(tensor);
        }
    };

    ctx->src0_size = get_tensor_buffer_size(گره->src[0]);
    ctx->src1_size = get_tensor_buffer_size(گره->src[1]);
    ctx->src2_size = get_tensor_buffer_size(گره->src[2]);
    ctx->dst_size  = get_tensor_buffer_size(گره);

    // Allocate CPU buffers for all tensors
    اگر (ctx->src0_size > 0) {
        ctx->cpu_src0_data = malloc(ctx->src0_size);
        اگر (!ctx->cpu_src0_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            برو cleanup;
        }
    }

    اگر (ctx->src1_size > 0) {
        ctx->cpu_src1_data = malloc(ctx->src1_size);
        اگر (!ctx->cpu_src1_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            برو cleanup;
        }
    }

    اگر (ctx->src2_size > 0) {
        ctx->cpu_src2_data = malloc(ctx->src2_size);
        اگر (!ctx->cpu_src2_data) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            برو cleanup;
        }
    }

    ctx->cpu_dst_data = malloc(ctx->dst_size);
    اگر (!ctx->cpu_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        برو cleanup;
    }

    ctx->et_dst_data = malloc(ctx->dst_size);
    اگر (!ctx->et_dst_data) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        برو cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    اگر (ctx->src0_size > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_size = ggml_nbytes(گره->src[0]);
        ggml_backend_tensor_get(گره->src[0], ctx->cpu_src0_data, 0, logical_size);
    }
    اگر (ctx->src1_size > 0) {
        size_t logical_size = ggml_nbytes(گره->src[1]);
        ggml_backend_tensor_get(گره->src[1], ctx->cpu_src1_data, 0, logical_size);
    }
    اگر (ctx->src2_size > 0) {
        size_t logical_size = ggml_nbytes(گره->src[2]);
        ggml_backend_tensor_get(گره->src[2], ctx->cpu_src2_data, 0, logical_size);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_size = ggml_nbytes(گره);
        ggml_backend_tensor_get(گره, ctx->cpu_dst_data, 0, logical_size);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    اگر (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        برو cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_size   = ggml_tensor_overhead() * 4 + ggml_graph_overhead();  // up to 4 tensors + graph
    ctx_params.mem_buffer = nullptr;
    ctx_params.no_alloc   = درست;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    اگر (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        برو cleanup;
    }

    // Create CPU tensors with proper context
    اگر (گره->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, گره->src[0]->type, GGML_MAX_DIMS, گره->src[0]->ne);
        اگر (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            برو cleanup;
        }
        ctx->cpu_src0->data = ctx->cpu_src0_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src0->nb, گره->src[0]->nb, اندازه(گره->src[0]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src0->op_params, گره->src[0]->op_params, اندازه(گره->src[0]->op_params));
    }

    اگر (گره->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, گره->src[1]->type, GGML_MAX_DIMS, گره->src[1]->ne);
        اگر (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            برو cleanup;
        }
        ctx->cpu_src1->data = ctx->cpu_src1_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src1->nb, گره->src[1]->nb, اندازه(گره->src[1]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src1->op_params, گره->src[1]->op_params, اندازه(گره->src[1]->op_params));
    }

    اگر (گره->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, گره->src[2]->type, GGML_MAX_DIMS, گره->src[2]->ne);
        اگر (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            برو cleanup;
        }
        ctx->cpu_src2->data = ctx->cpu_src2_data;
        // Copy stride array (nb) for correct memory layout
        memcpy(ctx->cpu_src2->nb, گره->src[2]->nb, اندازه(گره->src[2]->nb));
        // Copy op_params if present
        memcpy(ctx->cpu_src2->op_params, گره->src[2]->op_params, اندازه(گره->src[2]->op_params));
    }

    بازگشت درست;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    بازگشت نادرست;
}

منطقی ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ثابت ggml_tensor *                گره,
                                           ثابت ggml_et_cpu_compare_config * config) {
    اگر (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !گره || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        بازگشت نادرست;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = گره->op;
    انتخاب (op) {
        حالت GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            بشکن;
        حالت GGML_OP_ADD:
            ctx->cpu_dst = ggml_add(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            بشکن;
        حالت GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            بشکن;
        حالت GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            بشکن;
        حالت GGML_OP_ROPE:
            {
                ثابت int32_t * op_params  = (ثابت int32_t *) گره->op_params;
                ثابت int32_t   n_dims     = op_params[1];
                ثابت int32_t   mode       = op_params[2];
                ثابت int32_t   n_ctx_orig = op_params[4];
                ثابت اعشاری     freq_base  = *((ثابت اعشاری *) (op_params + 5));
                ثابت اعشاری     freq_scale = *((ثابت اعشاری *) (op_params + 6));
                ثابت اعشاری     ext_factor = *((ثابت اعشاری *) (op_params + 7));
                ثابت اعشاری     attn_factor = *((ثابت اعشاری *) (op_params + 8));
                ثابت اعشاری     beta_fast  = *((ثابت اعشاری *) (op_params + 9));
                ثابت اعشاری     beta_slow  = *((ثابت اعشاری *) (op_params + 10));

                اگر (mode & GGML_ROPE_TYPE_MROPE) {
                    صحیح sections[GGML_MROPE_SECTIONS];
                    memcpy(sections, op_params + 11, اندازه(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_base, freq_scale,
                                                   ext_factor, attn_factor, beta_fast, beta_slow);
                } وگرنه {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
                                                 attn_factor, beta_fast, beta_slow);
                }
            }
            بشکن;
        حالت GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                اعشاری eps;
                memcpy(&eps, گره->op_params, اندازه(اعشاری));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            بشکن;
        حالت GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            بشکن;
        حالت GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(گره, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            بشکن;
        حالت GGML_OP_SUM_ROWS:
            ctx->cpu_dst = ggml_sum_rows(ctx->ggml_ctx, ctx->cpu_src0);
            بشکن;
        حالت GGML_OP_MEAN:
            ctx->cpu_dst = ggml_mean(ctx->ggml_ctx, ctx->cpu_src0);
            بشکن;
        حالت GGML_OP_CLAMP:
            {
                اعشاری clamp_min, clamp_max;
                memcpy(&clamp_min, (ثابت اعشاری *) گره->op_params + 0, اندازه(اعشاری));
                memcpy(&clamp_max, (ثابت اعشاری *) گره->op_params + 1, اندازه(اعشاری));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            بشکن;
        حالت GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(گره, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                اگر (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    بازگشت نادرست;
                }
                اگر (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    ثابت اعشاری limit = ggml_get_op_params_f32(گره, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, limit);
                } وگرنه {
                    ctx->cpu_dst = ggml_glu_split(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            بشکن;
        حالت GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                اعشاری scale    = 1.0f;
                اعشاری max_bias = 0.0f;
                memcpy(&scale, (ثابت اعشاری *) گره->op_params + 0, اندازه(اعشاری));
                memcpy(&max_bias, (ثابت اعشاری *) گره->op_params + 1, اندازه(اعشاری));

                اگر (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } وگرنه {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                اگر (ctx->cpu_src2) {
                    ggml_soft_max_add_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            بشکن;
        حالت GGML_OP_GET_ROWS:
            ctx->cpu_dst = ggml_get_rows(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            بشکن;
        حالت GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            بشکن;
        حالت GGML_OP_SET_ROWS:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_base = ggml_new_tensor(ctx->ggml_ctx, گره->type, GGML_MAX_DIMS, گره->ne);
                اگر (!cpu_dst_base) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    بازگشت نادرست;
                }
                cpu_dst_base->data = ctx->cpu_dst_data;
                memcpy(cpu_dst_base->nb, گره->nb, اندازه(گره->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_set_rows(ctx->ggml_ctx, cpu_dst_base, ctx->cpu_src0, ctx->cpu_src1);
            }
            بشکن;
        پیشفرض:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            بازگشت نادرست;
    }

    اگر (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        بازگشت نادرست;
    }

    ctx->cpu_dst->data = ctx->cpu_dst_data;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    اگر (op != GGML_OP_CONT) {
        memcpy(ctx->cpu_dst->nb, گره->nb, اندازه(گره->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_graph = ggml_new_graph_custom(ctx->ggml_ctx, 1, نادرست);
    اگر (!ctx->cpu_graph) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        بازگشت نادرست;
    }
    ctx->cpu_graph->nodes[0] = ctx->cpu_dst;
    ctx->cpu_graph->n_nodes  = 1;

    // Log input data for debugging if enabled
    اگر (config && config->log_differences) {
        اگر (ctx->cpu_src0_data && ctx->src0_size >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_data)[0],
                           ((uint8_t *) ctx->cpu_src0_data)[1], ((uint8_t *) ctx->cpu_src0_data)[2],
                           ((uint8_t *) ctx->cpu_src0_data)[3]);
        }
        اگر (ctx->cpu_src1_data && ctx->src1_size >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((اعشاری *) ctx->cpu_src1_data)[0],
                           ((اعشاری *) ctx->cpu_src1_data)[1], ((اعشاری *) ctx->cpu_src1_data)[2],
                           ((اعشاری *) ctx->cpu_src1_data)[3]);
        }
    }

    // Compute using CPU backend
    ggml_status cpu_result = ggml_backend_graph_compute(ctx->cpu_backend, ctx->cpu_graph);

    اگر (cpu_result != GGML_STATUS_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_result);
        بازگشت نادرست;
    }

    // Log output data for debugging if enabled
    اگر (config && config->log_differences && ctx->dst_size >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((اعشاری *) ctx->cpu_dst_data)[0], ((اعشاری *) ctx->cpu_dst_data)[1],
                       ((اعشاری *) ctx->cpu_dst_data)[2], ((اعشاری *) ctx->cpu_dst_data)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_size = ggml_nbytes(گره);
    ggml_backend_tensor_get(گره, ctx->et_dst_data, 0, dst_logical_size);

    اگر (config->log_differences) {
        size_t رقم_elements = ggml_nelements(گره);
        size_t max_log      = std::min(رقم_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        منطقی    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_ADD || op == GGML_OP_GLU);
        اعشاری * cpu_src0_float = is_elementwise ? (اعشاری *) ctx->cpu_src0_data : nullptr;
        اعشاری * cpu_src1_float = is_elementwise ? (اعشاری *) ctx->cpu_src1_data : nullptr;

        // Helper to get float value from tensor data (handles f16 and f32)
        auto get_float = [](ثابت تهی * data, size_t idx, ggml_type type) -> اعشاری {
            اگر (type == GGML_TYPE_F16) {
                ثابت ggml_fp16_t * fp16_data = (ثابت ggml_fp16_t *) data;
                بازگشت ggml_fp16_to_fp32(fp16_data[idx]);
            }

            ثابت اعشاری * float_data = (ثابت اعشاری *) data;
            بازگشت float_data[idx];
        };

        // Compare all elements but log only the first max_log_elements
        منطقی   matches          = درست;
        size_t total_mismatches = 0;

        // First pass: check all elements for mismatches
        برای (size_t i = 0; i < رقم_elements; i++) {
            اعشاری cpu_val  = get_float(ctx->cpu_dst_data, i, گره->type);
            اعشاری et_val   = get_float(ctx->et_dst_data, i, گره->type);
            اعشاری diff     = fabsf(cpu_val - et_val);
            اعشاری rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            اگر (rel_diff > config->tolerance) {
                matches = نادرست;
                total_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        برای (size_t i = 0; i < max_log; i++) {
            اعشاری cpu_val = get_float(ctx->cpu_dst_data, i, گره->type);
            اعشاری et_val  = get_float(ctx->et_dst_data, i, گره->type);
            اعشاری diff    = fabsf(cpu_val - et_val);

            اگر (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } وگرنه اگر (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } وگرنه {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        اگر (رقم_elements > max_log) {
            size_t mid     = رقم_elements / 2;
            size_t end     = رقم_elements - 1;
            اعشاری  cpu_mid = get_float(ctx->cpu_dst_data, mid, گره->type);
            اعشاری  et_mid  = get_float(ctx->et_dst_data, mid, گره->type);
            اعشاری  cpu_end = get_float(ctx->cpu_dst_data, end, گره->type);
            اعشاری  et_end  = get_float(ctx->et_dst_data, end, گره->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_end, et_end);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       رقم_elements - total_mismatches, رقم_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    اگر (config->use_cpu_result) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_size = ggml_nbytes(گره);
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(گره), ctx->cpu_dst_data, 0, dst_logical_size);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    بازگشت درست;
}

تهی ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    اگر (!ctx) {
        بازگشت;
    }

    اگر (ctx->cpu_src0_data) {
        آزاد(ctx->cpu_src0_data);
        ctx->cpu_src0_data = nullptr;
    }
    اگر (ctx->cpu_src1_data) {
        آزاد(ctx->cpu_src1_data);
        ctx->cpu_src1_data = nullptr;
    }
    اگر (ctx->cpu_src2_data) {
        آزاد(ctx->cpu_src2_data);
        ctx->cpu_src2_data = nullptr;
    }
    اگر (ctx->cpu_dst_data) {
        آزاد(ctx->cpu_dst_data);
        ctx->cpu_dst_data = nullptr;
    }
    اگر (ctx->et_dst_data) {
        آزاد(ctx->et_dst_data);
        ctx->et_dst_data = nullptr;
    }

    اگر (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }

    اگر (ctx->cpu_backend) {
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
