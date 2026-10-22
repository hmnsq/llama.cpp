#ضمن "ggml-et-cpu-compare.h"

#ضمن "ggml-cpu/ggml-cpu-impl.h"
#ضمن "ggml-cpu/ops.h"

#ضمن <algorithm>
#ضمن <cmath>
#ضمن <cstdlib>
#ضمن <cstring>

منطقي ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ثابت ggml_tensor * غوټه, ggml_op op) {
    إذا (!ctx || !غوټه) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        أرجع خطأ;
    }

    // Clear context
    memset(ctx, 0, حجم(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    خپلکاری get_tensor_بفر_اندازه = [](ثابت ggml_tensor * tensor) -> size_t {
        إذا (!tensor) {
            أرجع 0;
        }

        إذا (tensor->بفر) {
            // Get actual backend buffer size
            size_t بفر_اندازه = ggml_backend_بفر_get_اندازه(tensor->بفر);

            // Use the full buffer size to avoid any truncation issues
            أرجع بفر_اندازه;
        } وإلا {
            // Fallback to logical size if no buffer
            أرجع ggml_nbytes(tensor);
        }
    };

    ctx->src0_اندازه = get_tensor_بفر_اندازه(غوټه->src[0]);
    ctx->src1_اندازه = get_tensor_بفر_اندازه(غوټه->src[1]);
    ctx->src2_اندازه = get_tensor_بفر_اندازه(غوټه->src[2]);
    ctx->dst_اندازه  = get_tensor_بفر_اندازه(غوټه);

    // Allocate CPU buffers for all tensors
    إذا (ctx->src0_اندازه > 0) {
        ctx->cpu_src0_معلومات = حافظه_ورکړه(ctx->src0_اندازه);
        إذا (!ctx->cpu_src0_معلومات) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            اذهب cleanup;
        }
    }

    إذا (ctx->src1_اندازه > 0) {
        ctx->cpu_src1_معلومات = حافظه_ورکړه(ctx->src1_اندازه);
        إذا (!ctx->cpu_src1_معلومات) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            اذهب cleanup;
        }
    }

    إذا (ctx->src2_اندازه > 0) {
        ctx->cpu_src2_معلومات = حافظه_ورکړه(ctx->src2_اندازه);
        إذا (!ctx->cpu_src2_معلومات) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            اذهب cleanup;
        }
    }

    ctx->cpu_dst_معلومات = حافظه_ورکړه(ctx->dst_اندازه);
    إذا (!ctx->cpu_dst_معلومات) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        اذهب cleanup;
    }

    ctx->et_dst_معلومات = حافظه_ورکړه(ctx->dst_اندازه);
    إذا (!ctx->et_dst_معلومات) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        اذهب cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    إذا (ctx->src0_اندازه > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_اندازه = ggml_nbytes(غوټه->src[0]);
        ggml_backend_tensor_get(غوټه->src[0], ctx->cpu_src0_معلومات, 0, logical_اندازه);
    }
    إذا (ctx->src1_اندازه > 0) {
        size_t logical_اندازه = ggml_nbytes(غوټه->src[1]);
        ggml_backend_tensor_get(غوټه->src[1], ctx->cpu_src1_معلومات, 0, logical_اندازه);
    }
    إذا (ctx->src2_اندازه > 0) {
        size_t logical_اندازه = ggml_nbytes(غوټه->src[2]);
        ggml_backend_tensor_get(غوټه->src[2], ctx->cpu_src2_معلومات, 0, logical_اندازه);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_اندازه = ggml_nbytes(غوټه);
        ggml_backend_tensor_get(غوټه, ctx->cpu_dst_معلومات, 0, logical_اندازه);
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
    ctx_params.mem_اندازه   = ggml_tensor_overhead() * 4 + ggml_ګراف_overhead();  // up to 4 tensors + graph
    ctx_params.mem_بفر = تش_اشاره;
    ctx_params.no_alloc   = ریښتیا;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    إذا (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        اذهب cleanup;
    }

    // Create CPU tensors with proper context
    إذا (غوټه->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, غوټه->src[0]->type, GGML_MAX_DIMS, غوټه->src[0]->ne);
        إذا (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            اذهب cleanup;
        }
        ctx->cpu_src0->معلومات = ctx->cpu_src0_معلومات;
        // Copy stride array (nb) for correct memory layout
        حافظه_نقل(ctx->cpu_src0->nb, غوټه->src[0]->nb, حجم(غوټه->src[0]->nb));
        // Copy op_params if present
        حافظه_نقل(ctx->cpu_src0->op_params, غوټه->src[0]->op_params, حجم(غوټه->src[0]->op_params));
    }

    إذا (غوټه->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, غوټه->src[1]->type, GGML_MAX_DIMS, غوټه->src[1]->ne);
        إذا (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            اذهب cleanup;
        }
        ctx->cpu_src1->معلومات = ctx->cpu_src1_معلومات;
        // Copy stride array (nb) for correct memory layout
        حافظه_نقل(ctx->cpu_src1->nb, غوټه->src[1]->nb, حجم(غوټه->src[1]->nb));
        // Copy op_params if present
        حافظه_نقل(ctx->cpu_src1->op_params, غوټه->src[1]->op_params, حجم(غوټه->src[1]->op_params));
    }

    إذا (غوټه->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, غوټه->src[2]->type, GGML_MAX_DIMS, غوټه->src[2]->ne);
        إذا (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            اذهب cleanup;
        }
        ctx->cpu_src2->معلومات = ctx->cpu_src2_معلومات;
        // Copy stride array (nb) for correct memory layout
        حافظه_نقل(ctx->cpu_src2->nb, غوټه->src[2]->nb, حجم(غوټه->src[2]->nb));
        // Copy op_params if present
        حافظه_نقل(ctx->cpu_src2->op_params, غوټه->src[2]->op_params, حجم(غوټه->src[2]->op_params));
    }

    أرجع ریښتیا;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    أرجع خطأ;
}

منطقي ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ثابت ggml_tensor *                غوټه,
                                           ثابت ggml_et_cpu_compare_config * config) {
    إذا (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !غوټه || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        أرجع خطأ;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = غوټه->op;
    بدل (op) {
        حالة GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            اكسر;
        حالة GGML_OP_جمع_کړه:
            ctx->cpu_dst = ggml_جمع_کړه(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
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
                ثابت int32_t * op_params  = (ثابت int32_t *) غوټه->op_params;
                ثابت int32_t   n_dims     = op_params[1];
                ثابت int32_t   mode       = op_params[2];
                ثابت int32_t   n_ctx_orig = op_params[4];
                ثابت عائم     freq_بنسټ  = *((ثابت عائم *) (op_params + 5));
                ثابت عائم     freq_scale = *((ثابت عائم *) (op_params + 6));
                ثابت عائم     ext_ضریب = *((ثابت عائم *) (op_params + 7));
                ثابت عائم     attn_ضریب = *((ثابت عائم *) (op_params + 8));
                ثابت عائم     beta_fast  = *((ثابت عائم *) (op_params + 9));
                ثابت عائم     beta_slow  = *((ثابت عائم *) (op_params + 10));

                إذا (mode & GGML_ROPE_TYPE_MROPE) {
                    صحيح sections[GGML_MROPE_SECTIONS];
                    حافظه_نقل(sections, op_params + 11, حجم(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_بنسټ, freq_scale,
                                                   ext_ضریب, attn_ضریب, beta_fast, beta_slow);
                } وإلا {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_بنسټ, freq_scale, ext_ضریب,
                                                 attn_ضریب, beta_fast, beta_slow);
                }
            }
            اكسر;
        حالة GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                عائم eps;
                حافظه_نقل(&eps, غوټه->op_params, حجم(عائم));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            اكسر;
        حالة GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            اكسر;
        حالة GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(غوټه, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            اكسر;
        حالة GGML_OP_مجموعه_لیکې:
            ctx->cpu_dst = ggml_مجموعه_لیکې(ctx->ggml_ctx, ctx->cpu_src0);
            اكسر;
        حالة GGML_OP_منځنۍ:
            ctx->cpu_dst = ggml_منځنۍ(ctx->ggml_ctx, ctx->cpu_src0);
            اكسر;
        حالة GGML_OP_CLAMP:
            {
                عائم clamp_min, clamp_max;
                حافظه_نقل(&clamp_min, (ثابت عائم *) غوټه->op_params + 0, حجم(عائم));
                حافظه_نقل(&clamp_max, (ثابت عائم *) غوټه->op_params + 1, حجم(عائم));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            اكسر;
        حالة GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(غوټه, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                إذا (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    أرجع خطأ;
                }
                إذا (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    ثابت عائم حد = ggml_get_op_params_f32(غوټه, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, حد);
                } وإلا {
                    ctx->cpu_dst = ggml_glu_وېشل(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            اكسر;
        حالة GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                عائم scale    = 1.0f;
                عائم max_bias = 0.0f;
                حافظه_نقل(&scale, (ثابت عائم *) غوټه->op_params + 0, حجم(عائم));
                حافظه_نقل(&max_bias, (ثابت عائم *) غوټه->op_params + 1, حجم(عائم));

                إذا (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } وإلا {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                إذا (ctx->cpu_src2) {
                    ggml_soft_max_جمع_کړه_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            اكسر;
        حالة GGML_OP_GET_لیکې:
            ctx->cpu_dst = ggml_get_لیکې(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            اكسر;
        حالة GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            اكسر;
        حالة GGML_OP_سېټ_لیکې:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_بنسټ = ggml_new_tensor(ctx->ggml_ctx, غوټه->type, GGML_MAX_DIMS, غوټه->ne);
                إذا (!cpu_dst_بنسټ) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    أرجع خطأ;
                }
                cpu_dst_بنسټ->معلومات = ctx->cpu_dst_معلومات;
                حافظه_نقل(cpu_dst_بنسټ->nb, غوټه->nb, حجم(غوټه->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_سېټ_لیکې(ctx->ggml_ctx, cpu_dst_بنسټ, ctx->cpu_src0, ctx->cpu_src1);
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
        حافظه_نقل(ctx->cpu_dst->nb, غوټه->nb, حجم(غوټه->nb));
    }
    // For CONT operations, keep the contiguous strides created by ggml_cont()

    // Create minimal computation graph
    ctx->cpu_ګراف = ggml_new_ګراف_custom(ctx->ggml_ctx, 1, خطأ);
    إذا (!ctx->cpu_ګراف) {
        GGML_LOG_ERROR("ET: Failed to create CPU computation graph\n");
        أرجع خطأ;
    }
    ctx->cpu_ګراف->nodes[0] = ctx->cpu_dst;
    ctx->cpu_ګراف->n_nodes  = 1;

    // Log input data for debugging if enabled
    إذا (config && config->log_differences) {
        إذا (ctx->cpu_src0_معلومات && ctx->src0_اندازه >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_معلومات)[0],
                           ((uint8_t *) ctx->cpu_src0_معلومات)[1], ((uint8_t *) ctx->cpu_src0_معلومات)[2],
                           ((uint8_t *) ctx->cpu_src0_معلومات)[3]);
        }
        إذا (ctx->cpu_src1_معلومات && ctx->src1_اندازه >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((عائم *) ctx->cpu_src1_معلومات)[0],
                           ((عائم *) ctx->cpu_src1_معلومات)[1], ((عائم *) ctx->cpu_src1_معلومات)[2],
                           ((عائم *) ctx->cpu_src1_معلومات)[3]);
        }
    }

    // Compute using CPU backend
    ggml_حالت cpu_پایله = ggml_backend_ګراف_compute(ctx->cpu_backend, ctx->cpu_ګراف);

    إذا (cpu_پایله != GGML_حالت_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_پایله);
        أرجع خطأ;
    }

    // Log output data for debugging if enabled
    إذا (config && config->log_differences && ctx->dst_اندازه >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((عائم *) ctx->cpu_dst_معلومات)[0], ((عائم *) ctx->cpu_dst_معلومات)[1],
                       ((عائم *) ctx->cpu_dst_معلومات)[2], ((عائم *) ctx->cpu_dst_معلومات)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_اندازه = ggml_nbytes(غوټه);
    ggml_backend_tensor_get(غوټه, ctx->et_dst_معلومات, 0, dst_logical_اندازه);

    إذا (config->log_differences) {
        size_t عدد_elements = ggml_nelements(غوټه);
        size_t max_log      = std::min(عدد_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        منطقي    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_جمع_کړه || op == GGML_OP_GLU);
        عائم * cpu_src0_float = is_elementwise ? (عائم *) ctx->cpu_src0_معلومات : تش_اشاره;
        عائم * cpu_src1_float = is_elementwise ? (عائم *) ctx->cpu_src1_معلومات : تش_اشاره;

        // Helper to get float value from tensor data (handles f16 and f32)
        خپلکاری get_float = [](ثابت فراغ * معلومات, size_t idx, ggml_type type) -> عائم {
            إذا (type == GGML_TYPE_F16) {
                ثابت ggml_fp16_t * fp16_معلومات = (ثابت ggml_fp16_t *) معلومات;
                أرجع ggml_fp16_منزل_fp32(fp16_معلومات[idx]);
            }

            ثابت عائم * float_معلومات = (ثابت عائم *) معلومات;
            أرجع float_معلومات[idx];
        };

        // Compare all elements but log only the first max_log_elements
        منطقي   matches          = ریښتیا;
        size_t ټول_mismatches = 0;

        // First pass: check all elements for mismatches
        لكل (size_t i = 0; i < عدد_elements; i++) {
            عائم cpu_val  = get_float(ctx->cpu_dst_معلومات, i, غوټه->type);
            عائم et_val   = get_float(ctx->et_dst_معلومات, i, غوټه->type);
            عائم diff     = fabsf(cpu_val - et_val);
            عائم rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            إذا (rel_diff > config->tolerance) {
                matches = خطأ;
                ټول_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        لكل (size_t i = 0; i < max_log; i++) {
            عائم cpu_val = get_float(ctx->cpu_dst_معلومات, i, غوټه->type);
            عائم et_val  = get_float(ctx->et_dst_معلومات, i, غوټه->type);
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
        إذا (عدد_elements > max_log) {
            size_t mid     = عدد_elements / 2;
            size_t end     = عدد_elements - 1;
            عائم  cpu_mid = get_float(ctx->cpu_dst_معلومات, mid, غوټه->type);
            عائم  et_mid  = get_float(ctx->et_dst_معلومات, mid, غوټه->type);
            عائم  cpu_پای_ټکی = get_float(ctx->cpu_dst_معلومات, end, غوټه->type);
            عائم  et_پای_ټکی  = get_float(ctx->et_dst_معلومات, end, غوټه->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_پای_ټکی, et_پای_ټکی);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       عدد_elements - ټول_mismatches, عدد_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    إذا (config->use_cpu_پایله) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_اندازه = ggml_nbytes(غوټه);
        ggml_backend_tensor_سېټ(const_cast<ggml_tensor *>(غوټه), ctx->cpu_dst_معلومات, 0, dst_logical_اندازه);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    أرجع ریښتیا;
}

فراغ ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    إذا (!ctx) {
        أرجع;
    }

    إذا (ctx->cpu_src0_معلومات) {
        حرر(ctx->cpu_src0_معلومات);
        ctx->cpu_src0_معلومات = تش_اشاره;
    }
    إذا (ctx->cpu_src1_معلومات) {
        حرر(ctx->cpu_src1_معلومات);
        ctx->cpu_src1_معلومات = تش_اشاره;
    }
    إذا (ctx->cpu_src2_معلومات) {
        حرر(ctx->cpu_src2_معلومات);
        ctx->cpu_src2_معلومات = تش_اشاره;
    }
    إذا (ctx->cpu_dst_معلومات) {
        حرر(ctx->cpu_dst_معلومات);
        ctx->cpu_dst_معلومات = تش_اشاره;
    }
    إذا (ctx->et_dst_معلومات) {
        حرر(ctx->et_dst_معلومات);
        ctx->et_dst_معلومات = تش_اشاره;
    }

    إذا (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = تش_اشاره;
    }

    إذا (ctx->cpu_backend) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = تش_اشاره;
    }

    // Clear pointers
    ctx->cpu_src0  = تش_اشاره;
    ctx->cpu_src1  = تش_اشاره;
    ctx->cpu_src2  = تش_اشاره;
    ctx->cpu_dst   = تش_اشاره;
    ctx->cpu_ګراف = تش_اشاره;
}
