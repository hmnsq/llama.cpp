#شامل "ggml-et-cpu-compare.h"

#شامل "ggml-cpu/ggml-cpu-impl.h"
#شامل "ggml-cpu/ops.h"

#شامل <algorithm>
#شامل <cmath>
#شامل <cstdlib>
#شامل <cstring>

منطقی ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ثابت ggml_tensor * گرہ, ggml_op op) {
    اگر (!ctx || !گرہ) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        واپس جھوٹ;
    }

    // Clear context
    memset(ctx, 0, حجم(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    خودکار get_tensor_بفر_سائز = [](ثابت ggml_tensor * tensor) -> size_t {
        اگر (!tensor) {
            واپس 0;
        }

        اگر (tensor->بفر) {
            // Get actual backend buffer size
            size_t بفر_سائز = ggml_backend_بفر_get_سائز(tensor->بفر);

            // Use the full buffer size to avoid any truncation issues
            واپس بفر_سائز;
        } ورنہ {
            // Fallback to logical size if no buffer
            واپس ggml_nbytes(tensor);
        }
    };

    ctx->src0_سائز = get_tensor_بفر_سائز(گرہ->src[0]);
    ctx->src1_سائز = get_tensor_بفر_سائز(گرہ->src[1]);
    ctx->src2_سائز = get_tensor_بفر_سائز(گرہ->src[2]);
    ctx->dst_سائز  = get_tensor_بفر_سائز(گرہ);

    // Allocate CPU buffers for all tensors
    اگر (ctx->src0_سائز > 0) {
        ctx->cpu_src0_ڈیٹا = حافظہ_دو(ctx->src0_سائز);
        اگر (!ctx->cpu_src0_ڈیٹا) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            جاؤ cleanup;
        }
    }

    اگر (ctx->src1_سائز > 0) {
        ctx->cpu_src1_ڈیٹا = حافظہ_دو(ctx->src1_سائز);
        اگر (!ctx->cpu_src1_ڈیٹا) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            جاؤ cleanup;
        }
    }

    اگر (ctx->src2_سائز > 0) {
        ctx->cpu_src2_ڈیٹا = حافظہ_دو(ctx->src2_سائز);
        اگر (!ctx->cpu_src2_ڈیٹا) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            جاؤ cleanup;
        }
    }

    ctx->cpu_dst_ڈیٹا = حافظہ_دو(ctx->dst_سائز);
    اگر (!ctx->cpu_dst_ڈیٹا) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        جاؤ cleanup;
    }

    ctx->et_dst_ڈیٹا = حافظہ_دو(ctx->dst_سائز);
    اگر (!ctx->et_dst_ڈیٹا) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        جاؤ cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    اگر (ctx->src0_سائز > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_سائز = ggml_nbytes(گرہ->src[0]);
        ggml_backend_tensor_get(گرہ->src[0], ctx->cpu_src0_ڈیٹا, 0, logical_سائز);
    }
    اگر (ctx->src1_سائز > 0) {
        size_t logical_سائز = ggml_nbytes(گرہ->src[1]);
        ggml_backend_tensor_get(گرہ->src[1], ctx->cpu_src1_ڈیٹا, 0, logical_سائز);
    }
    اگر (ctx->src2_سائز > 0) {
        size_t logical_سائز = ggml_nbytes(گرہ->src[2]);
        ggml_backend_tensor_get(گرہ->src[2], ctx->cpu_src2_ڈیٹا, 0, logical_سائز);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_سائز = ggml_nbytes(گرہ);
        ggml_backend_tensor_get(گرہ, ctx->cpu_dst_ڈیٹا, 0, logical_سائز);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    اگر (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        جاؤ cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_سائز   = ggml_tensor_overhead() * 4 + ggml_گراف_overhead();  // up to 4 tensors + graph
    ctx_params.mem_بفر = خالی_اشاریہ;
    ctx_params.no_alloc   = سچ;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    اگر (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        جاؤ cleanup;
    }

    // Create CPU tensors with proper context
    اگر (گرہ->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, گرہ->src[0]->type, GGML_MAX_DIMS, گرہ->src[0]->ne);
        اگر (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            جاؤ cleanup;
        }
        ctx->cpu_src0->ڈیٹا = ctx->cpu_src0_ڈیٹا;
        // Copy stride array (nb) for correct memory layout
        حافظہ_نقل(ctx->cpu_src0->nb, گرہ->src[0]->nb, حجم(گرہ->src[0]->nb));
        // Copy op_params if present
        حافظہ_نقل(ctx->cpu_src0->op_params, گرہ->src[0]->op_params, حجم(گرہ->src[0]->op_params));
    }

    اگر (گرہ->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, گرہ->src[1]->type, GGML_MAX_DIMS, گرہ->src[1]->ne);
        اگر (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            جاؤ cleanup;
        }
        ctx->cpu_src1->ڈیٹا = ctx->cpu_src1_ڈیٹا;
        // Copy stride array (nb) for correct memory layout
        حافظہ_نقل(ctx->cpu_src1->nb, گرہ->src[1]->nb, حجم(گرہ->src[1]->nb));
        // Copy op_params if present
        حافظہ_نقل(ctx->cpu_src1->op_params, گرہ->src[1]->op_params, حجم(گرہ->src[1]->op_params));
    }

    اگر (گرہ->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, گرہ->src[2]->type, GGML_MAX_DIMS, گرہ->src[2]->ne);
        اگر (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            جاؤ cleanup;
        }
        ctx->cpu_src2->ڈیٹا = ctx->cpu_src2_ڈیٹا;
        // Copy stride array (nb) for correct memory layout
        حافظہ_نقل(ctx->cpu_src2->nb, گرہ->src[2]->nb, حجم(گرہ->src[2]->nb));
        // Copy op_params if present
        حافظہ_نقل(ctx->cpu_src2->op_params, گرہ->src[2]->op_params, حجم(گرہ->src[2]->op_params));
    }

    واپس سچ;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    واپس جھوٹ;
}

منطقی ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ثابت ggml_tensor *                گرہ,
                                           ثابت ggml_et_cpu_compare_config * config) {
    اگر (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !گرہ || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        واپس جھوٹ;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = گرہ->op;
    انتخاب (op) {
        حالت GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            توڑو;
        حالت GGML_OP_جوڑو:
            ctx->cpu_dst = ggml_جوڑو(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            توڑو;
        حالت GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            توڑو;
        حالت GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            توڑو;
        حالت GGML_OP_ROPE:
            {
                ثابت int32_t * op_params  = (ثابت int32_t *) گرہ->op_params;
                ثابت int32_t   n_dims     = op_params[1];
                ثابت int32_t   mode       = op_params[2];
                ثابت int32_t   n_ctx_orig = op_params[4];
                ثابت اعشاری     freq_بنیاد  = *((ثابت اعشاری *) (op_params + 5));
                ثابت اعشاری     freq_scale = *((ثابت اعشاری *) (op_params + 6));
                ثابت اعشاری     ext_ضربی_عدد = *((ثابت اعشاری *) (op_params + 7));
                ثابت اعشاری     attn_ضربی_عدد = *((ثابت اعشاری *) (op_params + 8));
                ثابت اعشاری     beta_fast  = *((ثابت اعشاری *) (op_params + 9));
                ثابت اعشاری     beta_slow  = *((ثابت اعشاری *) (op_params + 10));

                اگر (mode & GGML_ROPE_TYPE_MROPE) {
                    صحیح sections[GGML_MROPE_SECTIONS];
                    حافظہ_نقل(sections, op_params + 11, حجم(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_بنیاد, freq_scale,
                                                   ext_ضربی_عدد, attn_ضربی_عدد, beta_fast, beta_slow);
                } ورنہ {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_بنیاد, freq_scale, ext_ضربی_عدد,
                                                 attn_ضربی_عدد, beta_fast, beta_slow);
                }
            }
            توڑو;
        حالت GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                اعشاری eps;
                حافظہ_نقل(&eps, گرہ->op_params, حجم(اعشاری));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            توڑو;
        حالت GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            توڑو;
        حالت GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(گرہ, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            توڑو;
        حالت GGML_OP_مجموع_صفیں:
            ctx->cpu_dst = ggml_مجموع_صفیں(ctx->ggml_ctx, ctx->cpu_src0);
            توڑو;
        حالت GGML_OP_اوسط:
            ctx->cpu_dst = ggml_اوسط(ctx->ggml_ctx, ctx->cpu_src0);
            توڑو;
        حالت GGML_OP_CLAMP:
            {
                اعشاری clamp_min, clamp_max;
                حافظہ_نقل(&clamp_min, (ثابت اعشاری *) گرہ->op_params + 0, حجم(اعشاری));
                حافظہ_نقل(&clamp_max, (ثابت اعشاری *) گرہ->op_params + 1, حجم(اعشاری));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            توڑو;
        حالت GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(گرہ, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                اگر (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    واپس جھوٹ;
                }
                اگر (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    ثابت اعشاری حد_بندی = ggml_get_op_params_f32(گرہ, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, حد_بندی);
                } ورنہ {
                    ctx->cpu_dst = ggml_glu_بانٹو(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            توڑو;
        حالت GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                اعشاری scale    = 1.0f;
                اعشاری max_bias = 0.0f;
                حافظہ_نقل(&scale, (ثابت اعشاری *) گرہ->op_params + 0, حجم(اعشاری));
                حافظہ_نقل(&max_bias, (ثابت اعشاری *) گرہ->op_params + 1, حجم(اعشاری));

                اگر (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } ورنہ {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                اگر (ctx->cpu_src2) {
                    ggml_soft_max_جوڑو_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            توڑو;
        حالت GGML_OP_GET_صفیں:
            ctx->cpu_dst = ggml_get_صفیں(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            توڑو;
        حالت GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            توڑو;
        حالت GGML_OP_سیٹ_صفیں:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_بنیاد = ggml_new_tensor(ctx->ggml_ctx, گرہ->type, GGML_MAX_DIMS, گرہ->ne);
                اگر (!cpu_dst_بنیاد) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    واپس جھوٹ;
                }
                cpu_dst_بنیاد->ڈیٹا = ctx->cpu_dst_ڈیٹا;
                حافظہ_نقل(cpu_dst_بنیاد->nb, گرہ->nb, حجم(گرہ->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_سیٹ_صفیں(ctx->ggml_ctx, cpu_dst_بنیاد, ctx->cpu_src0, ctx->cpu_src1);
            }
            توڑو;
        طےشدہ:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            واپس جھوٹ;
    }

    اگر (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        واپس جھوٹ;
    }

    ctx->cpu_dst->ڈیٹا = ctx->cpu_dst_ڈیٹا;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    اگر (op != GGML_OP_CONT) {
        حافظہ_نقل(ctx->cpu_dst->nb, گرہ->nb, حجم(گرہ->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_گراف = ggml_new_گراف_custom(ctx->ggml_ctx, 1, جھوٹ);
    اگر (!ctx->cpu_گراف) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        واپس جھوٹ;
    }
    ctx->cpu_گراف->nodes[0] = ctx->cpu_dst;
    ctx->cpu_گراف->n_nodes  = 1;

    // Log input data for debugging if enabled
    اگر (config && config->log_differences) {
        اگر (ctx->cpu_src0_ڈیٹا && ctx->src0_سائز >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_ڈیٹا)[0],
                           ((uint8_t *) ctx->cpu_src0_ڈیٹا)[1], ((uint8_t *) ctx->cpu_src0_ڈیٹا)[2],
                           ((uint8_t *) ctx->cpu_src0_ڈیٹا)[3]);
        }
        اگر (ctx->cpu_src1_ڈیٹا && ctx->src1_سائز >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((اعشاری *) ctx->cpu_src1_ڈیٹا)[0],
                           ((اعشاری *) ctx->cpu_src1_ڈیٹا)[1], ((اعشاری *) ctx->cpu_src1_ڈیٹا)[2],
                           ((اعشاری *) ctx->cpu_src1_ڈیٹا)[3]);
        }
    }

    // Compute using CPU backend
    ggml_حالت cpu_نتیجہ = ggml_backend_گراف_compute(ctx->cpu_backend, ctx->cpu_گراف);

    اگر (cpu_نتیجہ != GGML_حالت_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_نتیجہ);
        واپس جھوٹ;
    }

    // Log output data for debugging if enabled
    اگر (config && config->log_differences && ctx->dst_سائز >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((اعشاری *) ctx->cpu_dst_ڈیٹا)[0], ((اعشاری *) ctx->cpu_dst_ڈیٹا)[1],
                       ((اعشاری *) ctx->cpu_dst_ڈیٹا)[2], ((اعشاری *) ctx->cpu_dst_ڈیٹا)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_سائز = ggml_nbytes(گرہ);
    ggml_backend_tensor_get(گرہ, ctx->et_dst_ڈیٹا, 0, dst_logical_سائز);

    اگر (config->log_differences) {
        size_t عدد_elements = ggml_nelements(گرہ);
        size_t max_log      = std::min(عدد_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        منطقی    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_جوڑو || op == GGML_OP_GLU);
        اعشاری * cpu_src0_float = is_elementwise ? (اعشاری *) ctx->cpu_src0_ڈیٹا : خالی_اشاریہ;
        اعشاری * cpu_src1_float = is_elementwise ? (اعشاری *) ctx->cpu_src1_ڈیٹا : خالی_اشاریہ;

        // Helper to get float value from tensor data (handles f16 and f32)
        خودکار get_float = [](ثابت خالی * ڈیٹا, size_t idx, ggml_type type) -> اعشاری {
            اگر (type == GGML_TYPE_F16) {
                ثابت ggml_fp16_t * fp16_ڈیٹا = (ثابت ggml_fp16_t *) ڈیٹا;
                واپس ggml_fp16_مقصد_fp32(fp16_ڈیٹا[idx]);
            }

            ثابت اعشاری * float_ڈیٹا = (ثابت اعشاری *) ڈیٹا;
            واپس float_ڈیٹا[idx];
        };

        // Compare all elements but log only the first max_log_elements
        منطقی   matches          = سچ;
        size_t کل_mismatches = 0;

        // First pass: check all elements for mismatches
        برائے (size_t i = 0; i < عدد_elements; i++) {
            اعشاری cpu_val  = get_float(ctx->cpu_dst_ڈیٹا, i, گرہ->type);
            اعشاری et_val   = get_float(ctx->et_dst_ڈیٹا, i, گرہ->type);
            اعشاری diff     = fabsf(cpu_val - et_val);
            اعشاری rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            اگر (rel_diff > config->tolerance) {
                matches = جھوٹ;
                کل_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        برائے (size_t i = 0; i < max_log; i++) {
            اعشاری cpu_val = get_float(ctx->cpu_dst_ڈیٹا, i, گرہ->type);
            اعشاری et_val  = get_float(ctx->et_dst_ڈیٹا, i, گرہ->type);
            اعشاری diff    = fabsf(cpu_val - et_val);

            اگر (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } ورنہ اگر (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } ورنہ {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        اگر (عدد_elements > max_log) {
            size_t mid     = عدد_elements / 2;
            size_t end     = عدد_elements - 1;
            اعشاری  cpu_mid = get_float(ctx->cpu_dst_ڈیٹا, mid, گرہ->type);
            اعشاری  et_mid  = get_float(ctx->et_dst_ڈیٹا, mid, گرہ->type);
            اعشاری  cpu_انتہا = get_float(ctx->cpu_dst_ڈیٹا, end, گرہ->type);
            اعشاری  et_انتہا  = get_float(ctx->et_dst_ڈیٹا, end, گرہ->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_انتہا, et_انتہا);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       عدد_elements - کل_mismatches, عدد_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    اگر (config->use_cpu_نتیجہ) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_سائز = ggml_nbytes(گرہ);
        ggml_backend_tensor_سیٹ(const_cast<ggml_tensor *>(گرہ), ctx->cpu_dst_ڈیٹا, 0, dst_logical_سائز);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    واپس سچ;
}

خالی ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    اگر (!ctx) {
        واپس;
    }

    اگر (ctx->cpu_src0_ڈیٹا) {
        آزاد(ctx->cpu_src0_ڈیٹا);
        ctx->cpu_src0_ڈیٹا = خالی_اشاریہ;
    }
    اگر (ctx->cpu_src1_ڈیٹا) {
        آزاد(ctx->cpu_src1_ڈیٹا);
        ctx->cpu_src1_ڈیٹا = خالی_اشاریہ;
    }
    اگر (ctx->cpu_src2_ڈیٹا) {
        آزاد(ctx->cpu_src2_ڈیٹا);
        ctx->cpu_src2_ڈیٹا = خالی_اشاریہ;
    }
    اگر (ctx->cpu_dst_ڈیٹا) {
        آزاد(ctx->cpu_dst_ڈیٹا);
        ctx->cpu_dst_ڈیٹا = خالی_اشاریہ;
    }
    اگر (ctx->et_dst_ڈیٹا) {
        آزاد(ctx->et_dst_ڈیٹا);
        ctx->et_dst_ڈیٹا = خالی_اشاریہ;
    }

    اگر (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = خالی_اشاریہ;
    }

    اگر (ctx->cpu_backend) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = خالی_اشاریہ;
    }

    // Clear pointers
    ctx->cpu_src0  = خالی_اشاریہ;
    ctx->cpu_src1  = خالی_اشاریہ;
    ctx->cpu_src2  = خالی_اشاریہ;
    ctx->cpu_dst   = خالی_اشاریہ;
    ctx->cpu_گراف = خالی_اشاریہ;
}
