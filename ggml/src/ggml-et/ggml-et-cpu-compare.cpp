#ضمن "ggml-et-cpu-compare.h"

#ضمن "ggml-cpu/ggml-cpu-impl.h"
#ضمن "ggml-cpu/ops.h"

#ضمن <algorithm>
#ضمن <cmath>
#ضمن <cstdlib>
#ضمن <cstring>

منطقي ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ثابت ggml_tensor * ڳنڍ, ggml_op op) {
    إذا (!ctx || !ڳنڍ) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        أرجع خطأ;
    }

    // Clear context
    memset(ctx, 0, حجم(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    خودڪار get_tensor_بفر_سائيز = [](ثابت ggml_tensor * tensor) -> size_t {
        إذا (!tensor) {
            أرجع 0;
        }

        إذا (tensor->بفر) {
            // Get actual backend buffer size
            size_t بفر_سائيز = ggml_backend_بفر_get_سائيز(tensor->بفر);

            // Use the full buffer size to avoid any truncation issues
            أرجع بفر_سائيز;
        } وإلا {
            // Fallback to logical size if no buffer
            أرجع ggml_nbytes(tensor);
        }
    };

    ctx->src0_سائيز = get_tensor_بفر_سائيز(ڳنڍ->src[0]);
    ctx->src1_سائيز = get_tensor_بفر_سائيز(ڳنڍ->src[1]);
    ctx->src2_سائيز = get_tensor_بفر_سائيز(ڳنڍ->src[2]);
    ctx->dst_سائيز  = get_tensor_بفر_سائيز(ڳنڍ);

    // Allocate CPU buffers for all tensors
    إذا (ctx->src0_سائيز > 0) {
        ctx->cpu_src0_معلومات = يادگيري_ڏيو(ctx->src0_سائيز);
        إذا (!ctx->cpu_src0_معلومات) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            اذهب cleanup;
        }
    }

    إذا (ctx->src1_سائيز > 0) {
        ctx->cpu_src1_معلومات = يادگيري_ڏيو(ctx->src1_سائيز);
        إذا (!ctx->cpu_src1_معلومات) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            اذهب cleanup;
        }
    }

    إذا (ctx->src2_سائيز > 0) {
        ctx->cpu_src2_معلومات = يادگيري_ڏيو(ctx->src2_سائيز);
        إذا (!ctx->cpu_src2_معلومات) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            اذهب cleanup;
        }
    }

    ctx->cpu_dst_معلومات = يادگيري_ڏيو(ctx->dst_سائيز);
    إذا (!ctx->cpu_dst_معلومات) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        اذهب cleanup;
    }

    ctx->et_dst_معلومات = يادگيري_ڏيو(ctx->dst_سائيز);
    إذا (!ctx->et_dst_معلومات) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        اذهب cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    إذا (ctx->src0_سائيز > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_سائيز = ggml_nbytes(ڳنڍ->src[0]);
        ggml_backend_tensor_get(ڳنڍ->src[0], ctx->cpu_src0_معلومات, 0, logical_سائيز);
    }
    إذا (ctx->src1_سائيز > 0) {
        size_t logical_سائيز = ggml_nbytes(ڳنڍ->src[1]);
        ggml_backend_tensor_get(ڳنڍ->src[1], ctx->cpu_src1_معلومات, 0, logical_سائيز);
    }
    إذا (ctx->src2_سائيز > 0) {
        size_t logical_سائيز = ggml_nbytes(ڳنڍ->src[2]);
        ggml_backend_tensor_get(ڳنڍ->src[2], ctx->cpu_src2_معلومات, 0, logical_سائيز);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_سائيز = ggml_nbytes(ڳنڍ);
        ggml_backend_tensor_get(ڳنڍ, ctx->cpu_dst_معلومات, 0, logical_سائيز);
    }

    // Create CPU backend for reference computation
    GGML_LOG_DEBUG("ET: Creating CPU backend for reference computation\n");
    ctx->cpu_backend = ggml_backend_cpu_init();
    إذا (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ET: Failed to create CPU backend\n");
        اذهب cleanup;
    }

    // Create GGML context for CPU tensors
    GGML_LOG_DEBUG("ET: Creating GGML context for CPU computation\n");
    ggml_init_params ctx_params;
    ctx_params.mem_سائيز   = ggml_tensor_overhead() * 4 + ggml_گراف_overhead();  // up to 4 tensors + graph
    ctx_params.mem_بفر = خالي_اشارو;
    ctx_params.no_alloc   = سچ;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    إذا (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        اذهب cleanup;
    }

    // Create CPU tensors with proper context
    إذا (ڳنڍ->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, ڳنڍ->src[0]->type, GGML_MAX_DIMS, ڳنڍ->src[0]->ne);
        إذا (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            اذهب cleanup;
        }
        ctx->cpu_src0->معلومات = ctx->cpu_src0_معلومات;
        // Copy stride array (nb) for correct memory layout
        يادگيري_نقل(ctx->cpu_src0->nb, ڳنڍ->src[0]->nb, حجم(ڳنڍ->src[0]->nb));
        // Copy op_params if present
        يادگيري_نقل(ctx->cpu_src0->op_params, ڳنڍ->src[0]->op_params, حجم(ڳنڍ->src[0]->op_params));
    }

    إذا (ڳنڍ->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, ڳنڍ->src[1]->type, GGML_MAX_DIMS, ڳنڍ->src[1]->ne);
        إذا (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            اذهب cleanup;
        }
        ctx->cpu_src1->معلومات = ctx->cpu_src1_معلومات;
        // Copy stride array (nb) for correct memory layout
        يادگيري_نقل(ctx->cpu_src1->nb, ڳنڍ->src[1]->nb, حجم(ڳنڍ->src[1]->nb));
        // Copy op_params if present
        يادگيري_نقل(ctx->cpu_src1->op_params, ڳنڍ->src[1]->op_params, حجم(ڳنڍ->src[1]->op_params));
    }

    إذا (ڳنڍ->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, ڳنڍ->src[2]->type, GGML_MAX_DIMS, ڳنڍ->src[2]->ne);
        إذا (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            اذهب cleanup;
        }
        ctx->cpu_src2->معلومات = ctx->cpu_src2_معلومات;
        // Copy stride array (nb) for correct memory layout
        يادگيري_نقل(ctx->cpu_src2->nb, ڳنڍ->src[2]->nb, حجم(ڳنڍ->src[2]->nb));
        // Copy op_params if present
        يادگيري_نقل(ctx->cpu_src2->op_params, ڳنڍ->src[2]->op_params, حجم(ڳنڍ->src[2]->op_params));
    }

    أرجع سچ;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    أرجع خطأ;
}

منطقي ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ثابت ggml_tensor *                ڳنڍ,
                                           ثابت ggml_et_cpu_compare_config * config) {
    إذا (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !ڳنڍ || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        أرجع خطأ;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = ڳنڍ->op;
    بدل (op) {
        حالة GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            اكسر;
        حالة GGML_OP_جوڙيو:
            ctx->cpu_dst = ggml_جوڙيو(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            اكسر;
        حالة GGML_OP_MUL_MAT:
            ctx->cpu_dst = ggml_mul_mat(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            اكسر;
        حالة GGML_OP_MUL_MAT_ID:
            // MUL_MAT_ID: Mixture of Experts matrix multiplication
            // src0 (as): expert weight matrices [K, M, n_expert]
            // src1 (b):  activations [K, n_expert_used, batch]
            // src2 (ids): expert selection indices [n_expert_used, batch]
            ctx->cpu_dst = ggml_mul_mat_id(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2);
            اكسر;
        حالة GGML_OP_ROPE:
            {
                ثابت int32_t * op_params  = (ثابت int32_t *) ڳنڍ->op_params;
                ثابت int32_t   n_dims     = op_params[1];
                ثابت int32_t   mode       = op_params[2];
                ثابت int32_t   n_ctx_orig = op_params[4];
                ثابت عائم     freq_بنياد  = *((ثابت عائم *) (op_params + 5));
                ثابت عائم     freq_scale = *((ثابت عائم *) (op_params + 6));
                ثابت عائم     ext_ضرب_ڪندڙ = *((ثابت عائم *) (op_params + 7));
                ثابت عائم     attn_ضرب_ڪندڙ = *((ثابت عائم *) (op_params + 8));
                ثابت عائم     beta_fast  = *((ثابت عائم *) (op_params + 9));
                ثابت عائم     beta_slow  = *((ثابت عائم *) (op_params + 10));

                إذا (mode & GGML_ROPE_TYPE_MROPE) {
                    صحيح sections[GGML_MROPE_SECTIONS];
                    يادگيري_نقل(sections, op_params + 11, حجم(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_بنياد, freq_scale,
                                                   ext_ضرب_ڪندڙ, attn_ضرب_ڪندڙ, beta_fast, beta_slow);
                } وإلا {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_بنياد, freq_scale, ext_ضرب_ڪندڙ,
                                                 attn_ضرب_ڪندڙ, beta_fast, beta_slow);
                }
            }
            اكسر;
        حالة GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                عائم eps;
                يادگيري_نقل(&eps, ڳنڍ->op_params, حجم(عائم));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            اكسر;
        حالة GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            اكسر;
        حالة GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(ڳنڍ, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            اكسر;
        حالة GGML_OP_جوڙ_صفون:
            ctx->cpu_dst = ggml_جوڙ_صفون(ctx->ggml_ctx, ctx->cpu_src0);
            اكسر;
        حالة GGML_OP_سراسري:
            ctx->cpu_dst = ggml_سراسري(ctx->ggml_ctx, ctx->cpu_src0);
            اكسر;
        حالة GGML_OP_CLAMP:
            {
                عائم clamp_min, clamp_max;
                يادگيري_نقل(&clamp_min, (ثابت عائم *) ڳنڍ->op_params + 0, حجم(عائم));
                يادگيري_نقل(&clamp_max, (ثابت عائم *) ڳنڍ->op_params + 1, حجم(عائم));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            اكسر;
        حالة GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(ڳنڍ, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                إذا (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    أرجع خطأ;
                }
                إذا (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    ثابت عائم حد_بندي = ggml_get_op_params_f32(ڳنڍ, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, حد_بندي);
                } وإلا {
                    ctx->cpu_dst = ggml_glu_ورهايو(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            اكسر;
        حالة GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                عائم scale    = 1.0f;
                عائم max_bias = 0.0f;
                يادگيري_نقل(&scale, (ثابت عائم *) ڳنڍ->op_params + 0, حجم(عائم));
                يادگيري_نقل(&max_bias, (ثابت عائم *) ڳنڍ->op_params + 1, حجم(عائم));

                إذا (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } وإلا {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                إذا (ctx->cpu_src2) {
                    ggml_soft_max_جوڙيو_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            اكسر;
        حالة GGML_OP_GET_صفون:
            ctx->cpu_dst = ggml_get_صفون(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            اكسر;
        حالة GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            اكسر;
        حالة GGML_OP_سيٽ_صفون:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_بنياد = ggml_new_tensor(ctx->ggml_ctx, ڳنڍ->type, GGML_MAX_DIMS, ڳنڍ->ne);
                إذا (!cpu_dst_بنياد) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    أرجع خطأ;
                }
                cpu_dst_بنياد->معلومات = ctx->cpu_dst_معلومات;
                يادگيري_نقل(cpu_dst_بنياد->nb, ڳنڍ->nb, حجم(ڳنڍ->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_سيٽ_صفون(ctx->ggml_ctx, cpu_dst_بنياد, ctx->cpu_src0, ctx->cpu_src1);
            }
            اكسر;
        افتراضي:
            GGML_LOG_ERROR("ET: Unsupported operation %s for CPU comparison\n", ggml_op_name(op));
            أرجع خطأ;
    }

    إذا (!ctx->cpu_dst) {
        GGML_LOG_ERROR("ET: Failed to create CPU destination tensor for operation %s\n", ggml_op_name(op));
        أرجع خطأ;
    }

    ctx->cpu_dst->معلومات = ctx->cpu_dst_معلومات;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    إذا (op != GGML_OP_CONT) {
        يادگيري_نقل(ctx->cpu_dst->nb, ڳنڍ->nb, حجم(ڳنڍ->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_گراف = ggml_new_گراف_custom(ctx->ggml_ctx, 1, خطأ);
    إذا (!ctx->cpu_گراف) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        أرجع خطأ;
    }
    ctx->cpu_گراف->nodes[0] = ctx->cpu_dst;
    ctx->cpu_گراف->n_nodes  = 1;

    // Log input data for debugging if enabled
    إذا (config && config->log_differences) {
        إذا (ctx->cpu_src0_معلومات && ctx->src0_سائيز >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_معلومات)[0],
                           ((uint8_t *) ctx->cpu_src0_معلومات)[1], ((uint8_t *) ctx->cpu_src0_معلومات)[2],
                           ((uint8_t *) ctx->cpu_src0_معلومات)[3]);
        }
        إذا (ctx->cpu_src1_معلومات && ctx->src1_سائيز >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((عائم *) ctx->cpu_src1_معلومات)[0],
                           ((عائم *) ctx->cpu_src1_معلومات)[1], ((عائم *) ctx->cpu_src1_معلومات)[2],
                           ((عائم *) ctx->cpu_src1_معلومات)[3]);
        }
    }

    // Compute using CPU backend
    ggml_حالت cpu_نتيجو = ggml_backend_گراف_compute(ctx->cpu_backend, ctx->cpu_گراف);

    إذا (cpu_نتيجو != GGML_حالت_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_نتيجو);
        أرجع خطأ;
    }

    // Log output data for debugging if enabled
    إذا (config && config->log_differences && ctx->dst_سائيز >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((عائم *) ctx->cpu_dst_معلومات)[0], ((عائم *) ctx->cpu_dst_معلومات)[1],
                       ((عائم *) ctx->cpu_dst_معلومات)[2], ((عائم *) ctx->cpu_dst_معلومات)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_سائيز = ggml_nbytes(ڳنڍ);
    ggml_backend_tensor_get(ڳنڍ, ctx->et_dst_معلومات, 0, dst_logical_سائيز);

    إذا (config->log_differences) {
        size_t انگ_elements = ggml_nelements(ڳنڍ);
        size_t max_log      = std::min(انگ_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        منطقي    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_جوڙيو || op == GGML_OP_GLU);
        عائم * cpu_src0_float = is_elementwise ? (عائم *) ctx->cpu_src0_معلومات : خالي_اشارو;
        عائم * cpu_src1_float = is_elementwise ? (عائم *) ctx->cpu_src1_معلومات : خالي_اشارو;

        // Helper to get float value from tensor data (handles f16 and f32)
        خودڪار get_float = [](ثابت فراغ * معلومات, size_t idx, ggml_type type) -> عائم {
            إذا (type == GGML_TYPE_F16) {
                ثابت ggml_fp16_t * fp16_معلومات = (ثابت ggml_fp16_t *) معلومات;
                أرجع ggml_fp16_منزل_fp32(fp16_معلومات[idx]);
            }

            ثابت عائم * float_معلومات = (ثابت عائم *) معلومات;
            أرجع float_معلومات[idx];
        };

        // Compare all elements but log only the first max_log_elements
        منطقي   matches          = سچ;
        size_t ڪل_mismatches = 0;

        // First pass: check all elements for mismatches
        لكل (size_t i = 0; i < انگ_elements; i++) {
            عائم cpu_val  = get_float(ctx->cpu_dst_معلومات, i, ڳنڍ->type);
            عائم et_val   = get_float(ctx->et_dst_معلومات, i, ڳنڍ->type);
            عائم diff     = fabsf(cpu_val - et_val);
            عائم rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            إذا (rel_diff > config->tolerance) {
                matches = خطأ;
                ڪل_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        لكل (size_t i = 0; i < max_log; i++) {
            عائم cpu_val = get_float(ctx->cpu_dst_معلومات, i, ڳنڍ->type);
            عائم et_val  = get_float(ctx->et_dst_معلومات, i, ڳنڍ->type);
            عائم diff    = fabsf(cpu_val - et_val);

            إذا (is_elementwise && cpu_src0_float && cpu_src1_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f, src1=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i],
                               cpu_src1_float[i], cpu_val, et_val, diff);
            } وإلا إذا (is_elementwise && cpu_src0_float) {
                GGML_LOG_DEBUG("ET: [%zu] src0=%.6f -> CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_src0_float[i], cpu_val,
                               et_val, diff);
            } وإلا {
                GGML_LOG_DEBUG("ET: [%zu] CPU=%.6f, ET=%.6f, diff=%.6f\n", i, cpu_val, et_val, diff);
            }
        }

        // Check some elements from the middle and end for full coverage
        إذا (انگ_elements > max_log) {
            size_t mid     = انگ_elements / 2;
            size_t end     = انگ_elements - 1;
            عائم  cpu_mid = get_float(ctx->cpu_dst_معلومات, mid, ڳنڍ->type);
            عائم  et_mid  = get_float(ctx->et_dst_معلومات, mid, ڳنڍ->type);
            عائم  cpu_پڇاڙي = get_float(ctx->cpu_dst_معلومات, end, ڳنڍ->type);
            عائم  et_پڇاڙي  = get_float(ctx->et_dst_معلومات, end, ڳنڍ->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_پڇاڙي, et_پڇاڙي);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       انگ_elements - ڪل_mismatches, انگ_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    إذا (config->use_cpu_نتيجو) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_سائيز = ggml_nbytes(ڳنڍ);
        ggml_backend_tensor_سيٽ(const_cast<ggml_tensor *>(ڳنڍ), ctx->cpu_dst_معلومات, 0, dst_logical_سائيز);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    أرجع سچ;
}

فراغ ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    إذا (!ctx) {
        أرجع;
    }

    إذا (ctx->cpu_src0_معلومات) {
        حرر(ctx->cpu_src0_معلومات);
        ctx->cpu_src0_معلومات = خالي_اشارو;
    }
    إذا (ctx->cpu_src1_معلومات) {
        حرر(ctx->cpu_src1_معلومات);
        ctx->cpu_src1_معلومات = خالي_اشارو;
    }
    إذا (ctx->cpu_src2_معلومات) {
        حرر(ctx->cpu_src2_معلومات);
        ctx->cpu_src2_معلومات = خالي_اشارو;
    }
    إذا (ctx->cpu_dst_معلومات) {
        حرر(ctx->cpu_dst_معلومات);
        ctx->cpu_dst_معلومات = خالي_اشارو;
    }
    إذا (ctx->et_dst_معلومات) {
        حرر(ctx->et_dst_معلومات);
        ctx->et_dst_معلومات = خالي_اشارو;
    }

    إذا (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = خالي_اشارو;
    }

    إذا (ctx->cpu_backend) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = خالي_اشارو;
    }

    // Clear pointers
    ctx->cpu_src0  = خالي_اشارو;
    ctx->cpu_src1  = خالي_اشارو;
    ctx->cpu_src2  = خالي_اشارو;
    ctx->cpu_dst   = خالي_اشارو;
    ctx->cpu_گراف = خالي_اشارو;
}
