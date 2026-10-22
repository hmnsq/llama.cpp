#ضمن "ggml-et-cpu-compare.h"

#ضمن "ggml-cpu/ggml-cpu-impl.h"
#ضمن "ggml-cpu/ops.h"

#ضمن <algorithm>
#ضمن <cmath>
#ضمن <cstdlib>
#ضمن <cstring>

منطقي ggml_et_cpu_compare_init_pre(ggml_et_cpu_compare_ctx * ctx, ثابت ggml_tensor * گره, ggml_op op) {
    إذا (!ctx || !گره) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compare init\n");
        أرجع خطأ;
    }

    // Clear context
    memset(ctx, 0, حجم(*ctx));

    // Calculate actual buffer sizes - use backend buffer size for accurate copy
    خودکار get_tensor_میانگیر_اندازه = [](ثابت ggml_tensor * tensor) -> size_t {
        إذا (!tensor) {
            أرجع 0;
        }

        إذا (tensor->میانگیر) {
            // Get actual backend buffer size
            size_t میانگیر_اندازه = ggml_backend_میانگیر_get_اندازه(tensor->میانگیر);

            // Use the full buffer size to avoid any truncation issues
            أرجع میانگیر_اندازه;
        } وإلا {
            // Fallback to logical size if no buffer
            أرجع ggml_nbytes(tensor);
        }
    };

    ctx->src0_اندازه = get_tensor_میانگیر_اندازه(گره->src[0]);
    ctx->src1_اندازه = get_tensor_میانگیر_اندازه(گره->src[1]);
    ctx->src2_اندازه = get_tensor_میانگیر_اندازه(گره->src[2]);
    ctx->dst_اندازه  = get_tensor_میانگیر_اندازه(گره);

    // Allocate CPU buffers for all tensors
    إذا (ctx->src0_اندازه > 0) {
        ctx->cpu_src0_داده = حافظه_بده(ctx->src0_اندازه);
        إذا (!ctx->cpu_src0_داده) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src0 buffer\n");
            اذهب cleanup;
        }
    }

    إذا (ctx->src1_اندازه > 0) {
        ctx->cpu_src1_داده = حافظه_بده(ctx->src1_اندازه);
        إذا (!ctx->cpu_src1_داده) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src1 buffer\n");
            اذهب cleanup;
        }
    }

    إذا (ctx->src2_اندازه > 0) {
        ctx->cpu_src2_داده = حافظه_بده(ctx->src2_اندازه);
        إذا (!ctx->cpu_src2_داده) {
            GGML_LOG_ERROR("ET: Failed to allocate CPU src2 buffer\n");
            اذهب cleanup;
        }
    }

    ctx->cpu_dst_داده = حافظه_بده(ctx->dst_اندازه);
    إذا (!ctx->cpu_dst_داده) {
        GGML_LOG_ERROR("ET: Failed to allocate CPU dst buffer\n");
        اذهب cleanup;
    }

    ctx->et_dst_داده = حافظه_بده(ctx->dst_اندازه);
    إذا (!ctx->et_dst_داده) {
        GGML_LOG_ERROR("ET: Failed to allocate ET dst buffer\n");
        اذهب cleanup;
    }

    // Copy data from ET device buffers to CPU host buffers
    إذا (ctx->src0_اندازه > 0) {
        // Copy logical tensor size - ggml_backend_tensor_get handles stride layout internally
        size_t logical_اندازه = ggml_nbytes(گره->src[0]);
        ggml_backend_tensor_get(گره->src[0], ctx->cpu_src0_داده, 0, logical_اندازه);
    }
    إذا (ctx->src1_اندازه > 0) {
        size_t logical_اندازه = ggml_nbytes(گره->src[1]);
        ggml_backend_tensor_get(گره->src[1], ctx->cpu_src1_داده, 0, logical_اندازه);
    }
    إذا (ctx->src2_اندازه > 0) {
        size_t logical_اندازه = ggml_nbytes(گره->src[2]);
        ggml_backend_tensor_get(گره->src[2], ctx->cpu_src2_داده, 0, logical_اندازه);
    }

    // Copy destination data from device (for operations like SET_ROWS that modify existing data)
    // Most ops create new tensors so this is unused, but SET_ROWS requires existing dst data
    {
        size_t logical_اندازه = ggml_nbytes(گره);
        ggml_backend_tensor_get(گره, ctx->cpu_dst_داده, 0, logical_اندازه);
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
    ctx_params.mem_اندازه   = ggml_tensor_overhead() * 4 + ggml_گراف_overhead();  // up to 4 tensors + graph
    ctx_params.mem_میانگیر = اشاره_تهی;
    ctx_params.no_alloc   = درست;                                                // We'll manage data ourselves
    ctx->ggml_ctx         = ggml_init(ctx_params);
    إذا (!ctx->ggml_ctx) {
        GGML_LOG_ERROR("ET: Failed to create GGML context\n");
        اذهب cleanup;
    }

    // Create CPU tensors with proper context
    إذا (گره->src[0]) {
        ctx->cpu_src0 = ggml_new_tensor(ctx->ggml_ctx, گره->src[0]->type, GGML_MAX_DIMS, گره->src[0]->ne);
        إذا (!ctx->cpu_src0) {
            GGML_LOG_ERROR("ET: Failed to create CPU src0 tensor\n");
            اذهب cleanup;
        }
        ctx->cpu_src0->داده = ctx->cpu_src0_داده;
        // Copy stride array (nb) for correct memory layout
        کپی_حافظه(ctx->cpu_src0->nb, گره->src[0]->nb, حجم(گره->src[0]->nb));
        // Copy op_params if present
        کپی_حافظه(ctx->cpu_src0->op_params, گره->src[0]->op_params, حجم(گره->src[0]->op_params));
    }

    إذا (گره->src[1]) {
        ctx->cpu_src1 = ggml_new_tensor(ctx->ggml_ctx, گره->src[1]->type, GGML_MAX_DIMS, گره->src[1]->ne);
        إذا (!ctx->cpu_src1) {
            GGML_LOG_ERROR("ET: Failed to create CPU src1 tensor\n");
            اذهب cleanup;
        }
        ctx->cpu_src1->داده = ctx->cpu_src1_داده;
        // Copy stride array (nb) for correct memory layout
        کپی_حافظه(ctx->cpu_src1->nb, گره->src[1]->nb, حجم(گره->src[1]->nb));
        // Copy op_params if present
        کپی_حافظه(ctx->cpu_src1->op_params, گره->src[1]->op_params, حجم(گره->src[1]->op_params));
    }

    إذا (گره->src[2]) {
        ctx->cpu_src2 = ggml_new_tensor(ctx->ggml_ctx, گره->src[2]->type, GGML_MAX_DIMS, گره->src[2]->ne);
        إذا (!ctx->cpu_src2) {
            GGML_LOG_ERROR("ET: Failed to create CPU src2 tensor\n");
            اذهب cleanup;
        }
        ctx->cpu_src2->داده = ctx->cpu_src2_داده;
        // Copy stride array (nb) for correct memory layout
        کپی_حافظه(ctx->cpu_src2->nb, گره->src[2]->nb, حجم(گره->src[2]->nb));
        // Copy op_params if present
        کپی_حافظه(ctx->cpu_src2->op_params, گره->src[2]->op_params, حجم(گره->src[2]->op_params));
    }

    أرجع درست;

cleanup:
    ggml_et_cpu_compare_free(ctx);
    أرجع خطأ;
}

منطقي ggml_et_cpu_compare_compute_and_check(ggml_et_cpu_compare_ctx *          ctx,
                                           ثابت ggml_tensor *                گره,
                                           ثابت ggml_et_cpu_compare_config * config) {
    إذا (!ctx || !ctx->cpu_backend || !ctx->ggml_ctx || !گره || !config) {
        GGML_LOG_ERROR("ET: Invalid parameters for CPU compute and check\n");
        أرجع خطأ;
    }

    // Create operation-specific CPU destination tensor based on the node's operation
    ggml_op op = گره->op;
    بدل (op) {
        حالة GGML_OP_MUL:
            ctx->cpu_dst = ggml_mul(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            اكسر;
        حالة GGML_OP_جمع_کن:
            ctx->cpu_dst = ggml_جمع_کن(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
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
                ثابت int32_t * op_params  = (ثابت int32_t *) گره->op_params;
                ثابت int32_t   n_dims     = op_params[1];
                ثابت int32_t   mode       = op_params[2];
                ثابت int32_t   n_ctx_orig = op_params[4];
                ثابت عائم     freq_پایه  = *((ثابت عائم *) (op_params + 5));
                ثابت عائم     freq_scale = *((ثابت عائم *) (op_params + 6));
                ثابت عائم     ext_ضریب = *((ثابت عائم *) (op_params + 7));
                ثابت عائم     attn_ضریب = *((ثابت عائم *) (op_params + 8));
                ثابت عائم     beta_fast  = *((ثابت عائم *) (op_params + 9));
                ثابت عائم     beta_slow  = *((ثابت عائم *) (op_params + 10));

                إذا (mode & GGML_ROPE_TYPE_MROPE) {
                    صحيح sections[GGML_MROPE_SECTIONS];
                    کپی_حافظه(sections, op_params + 11, حجم(sections));
                    ctx->cpu_dst = ggml_rope_multi(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                   n_dims, sections, mode, n_ctx_orig, freq_پایه, freq_scale,
                                                   ext_ضریب, attn_ضریب, beta_fast, beta_slow);
                } وإلا {
                    ctx->cpu_dst = ggml_rope_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, ctx->cpu_src2,
                                                 n_dims, mode, n_ctx_orig, freq_پایه, freq_scale, ext_ضریب,
                                                 attn_ضریب, beta_fast, beta_slow);
                }
            }
            اكسر;
        حالة GGML_OP_RMS_NORM:
            // Extract epsilon parameter from op_params (stored as float)
            {
                عائم eps;
                کپی_حافظه(&eps, گره->op_params, حجم(عائم));
                ctx->cpu_dst = ggml_rms_norm(ctx->ggml_ctx, ctx->cpu_src0, eps);
            }
            اكسر;
        حالة GGML_OP_SQR:
            ctx->cpu_dst = ggml_sqr(ctx->ggml_ctx, ctx->cpu_src0);
            اكسر;
        حالة GGML_OP_UNARY:
            {
                ggml_unary_op uop = (ggml_unary_op) ggml_get_op_params_i32(گره, 0);
                ctx->cpu_dst      = ggml_unary(ctx->ggml_ctx, ctx->cpu_src0, uop);
            }
            اكسر;
        حالة GGML_OP_جمع_ردیفها:
            ctx->cpu_dst = ggml_جمع_ردیفها(ctx->ggml_ctx, ctx->cpu_src0);
            اكسر;
        حالة GGML_OP_میانگین:
            ctx->cpu_dst = ggml_میانگین(ctx->ggml_ctx, ctx->cpu_src0);
            اكسر;
        حالة GGML_OP_CLAMP:
            {
                عائم clamp_min, clamp_max;
                کپی_حافظه(&clamp_min, (ثابت عائم *) گره->op_params + 0, حجم(عائم));
                کپی_حافظه(&clamp_max, (ثابت عائم *) گره->op_params + 1, حجم(عائم));
                ctx->cpu_dst = ggml_clamp(ctx->ggml_ctx, ctx->cpu_src0, clamp_min, clamp_max);
            }
            اكسر;
        حالة GGML_OP_GLU:
            // Extract GLU parameters from op_params (split mode only)
            {
                int32_t     glu_op_type = ggml_get_op_params_i32(گره, 0);  // GLU variant
                ggml_glu_op glu_op      = (ggml_glu_op) glu_op_type;

                // Only support split tensor mode
                إذا (!ctx->cpu_src1) {
                    GGML_LOG_ERROR("ET: GLU CPU comparison requires split tensor mode\n");
                    أرجع خطأ;
                }
                إذا (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                    ثابت عائم حد = ggml_get_op_params_f32(گره, 3);
                    ctx->cpu_dst      = ggml_swiglu_clamp(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, حد);
                } وإلا {
                    ctx->cpu_dst = ggml_glu_تقسیم(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, glu_op);
                }
            }
            اكسر;
        حالة GGML_OP_SOFT_MAX:
            {
                // Extract scale and max_bias from op_params
                عائم scale    = 1.0f;
                عائم max_bias = 0.0f;
                کپی_حافظه(&scale, (ثابت عائم *) گره->op_params + 0, حجم(عائم));
                کپی_حافظه(&max_bias, (ثابت عائم *) گره->op_params + 1, حجم(عائم));

                إذا (ctx->cpu_src1 || scale != 1.0f || max_bias != 0.0f) {
                    // Use extended softmax when mask or non-default parameters are present
                    ctx->cpu_dst = ggml_soft_max_ext(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1, scale, max_bias);
                } وإلا {
                    // Use simple softmax when no mask and default parameters
                    ctx->cpu_dst = ggml_soft_max(ctx->ggml_ctx, ctx->cpu_src0);
                }

                // Add sinks if present
                إذا (ctx->cpu_src2) {
                    ggml_soft_max_جمع_کن_sinks(ctx->cpu_dst, ctx->cpu_src2);
                }
            }
            اكسر;
        حالة GGML_OP_GET_ردیفها:
            ctx->cpu_dst = ggml_get_ردیفها(ctx->ggml_ctx, ctx->cpu_src0, ctx->cpu_src1);
            اكسر;
        حالة GGML_OP_CONT:
            ctx->cpu_dst = ggml_cont(ctx->ggml_ctx, ctx->cpu_src0);
            اكسر;
        حالة GGML_OP_مجموعه_ردیفها:
            {
                // SET_ROWS operation scatters src0 rows to dst[src1] positions
                // Create destination tensor (this is the "view" that SET_ROWS returns)
                ggml_tensor * cpu_dst_پایه = ggml_new_tensor(ctx->ggml_ctx, گره->type, GGML_MAX_DIMS, گره->ne);
                إذا (!cpu_dst_پایه) {
                    GGML_LOG_ERROR("ET: Failed to create CPU destination base tensor for SET_ROWS\n");
                    أرجع خطأ;
                }
                cpu_dst_پایه->داده = ctx->cpu_dst_داده;
                کپی_حافظه(cpu_dst_پایه->nb, گره->nb, حجم(گره->nb));

                // Note: cpu_dst_data already contains the pre-existing destination data from device
                // SET_ROWS will update specific rows, leaving others unchanged

                // Perform SET_ROWS operation: returns a view that scatters src0 rows to dst[src1] positions
                ctx->cpu_dst = ggml_مجموعه_ردیفها(ctx->ggml_ctx, cpu_dst_پایه, ctx->cpu_src0, ctx->cpu_src1);
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

    ctx->cpu_dst->داده = ctx->cpu_dst_داده;
    // Copy stride array (nb) for correct memory layout - except for CONT which should keep contiguous strides
    إذا (op != GGML_OP_CONT) {
        کپی_حافظه(ctx->cpu_dst->nb, گره->nb, حجم(گره->nb));
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
        إذا (ctx->cpu_src0_داده && ctx->src0_اندازه >= 4) {
            GGML_LOG_DEBUG("ET: CPU src0 first few bytes: %02x %02x %02x %02x\n", ((uint8_t *) ctx->cpu_src0_داده)[0],
                           ((uint8_t *) ctx->cpu_src0_داده)[1], ((uint8_t *) ctx->cpu_src0_داده)[2],
                           ((uint8_t *) ctx->cpu_src0_داده)[3]);
        }
        إذا (ctx->cpu_src1_داده && ctx->src1_اندازه >= 16) {
            GGML_LOG_DEBUG("ET: CPU src1 first few floats: %.6f %.6f %.6f %.6f\n", ((عائم *) ctx->cpu_src1_داده)[0],
                           ((عائم *) ctx->cpu_src1_داده)[1], ((عائم *) ctx->cpu_src1_داده)[2],
                           ((عائم *) ctx->cpu_src1_داده)[3]);
        }
    }

    // Compute using CPU backend
    ggml_وضعیت cpu_نتیجه = ggml_backend_گراف_compute(ctx->cpu_backend, ctx->cpu_گراف);

    إذا (cpu_نتیجه != GGML_وضعیت_SUCCESS) {
        GGML_LOG_ERROR("ET: CPU reference computation failed with status %d\n", cpu_نتیجه);
        أرجع خطأ;
    }

    // Log output data for debugging if enabled
    إذا (config && config->log_differences && ctx->dst_اندازه >= 16) {
        GGML_LOG_DEBUG("ET: CPU dst first few floats after computation: %.6f %.6f %.6f %.6f\n",
                       ((عائم *) ctx->cpu_dst_داده)[0], ((عائم *) ctx->cpu_dst_داده)[1],
                       ((عائم *) ctx->cpu_dst_داده)[2], ((عائم *) ctx->cpu_dst_داده)[3]);
    }

    // Now copy ET device destination to host for comparison
    size_t dst_logical_اندازه = ggml_nbytes(گره);
    ggml_backend_tensor_get(گره, ctx->et_dst_داده, 0, dst_logical_اندازه);

    إذا (config->log_differences) {
        size_t رقم_elements = ggml_nelements(گره);
        size_t max_log      = std::min(رقم_elements, config->max_log_elements);

        // Check if this is an elementwise operation that can show src inputs
        منطقي    is_elementwise = (op == GGML_OP_MUL || op == GGML_OP_جمع_کن || op == GGML_OP_GLU);
        عائم * cpu_src0_float = is_elementwise ? (عائم *) ctx->cpu_src0_داده : اشاره_تهی;
        عائم * cpu_src1_float = is_elementwise ? (عائم *) ctx->cpu_src1_داده : اشاره_تهی;

        // Helper to get float value from tensor data (handles f16 and f32)
        خودکار get_float = [](ثابت فراغ * داده, size_t idx, ggml_type type) -> عائم {
            إذا (type == GGML_TYPE_F16) {
                ثابت ggml_fp16_t * fp16_داده = (ثابت ggml_fp16_t *) داده;
                أرجع ggml_fp16_مقصد_fp32(fp16_داده[idx]);
            }

            ثابت عائم * float_داده = (ثابت عائم *) داده;
            أرجع float_داده[idx];
        };

        // Compare all elements but log only the first max_log_elements
        منطقي   matches          = درست;
        size_t مجموع_mismatches = 0;

        // First pass: check all elements for mismatches
        لكل (size_t i = 0; i < رقم_elements; i++) {
            عائم cpu_val  = get_float(ctx->cpu_dst_داده, i, گره->type);
            عائم et_val   = get_float(ctx->et_dst_داده, i, گره->type);
            عائم diff     = fabsf(cpu_val - et_val);
            عائم rel_diff = diff / (fabsf(cpu_val) + 1e-8f);

            إذا (rel_diff > config->tolerance) {
                matches = خطأ;
                مجموع_mismatches++;
            }
        }

        // Second pass: log detailed info for first max_log elements only
        لكل (size_t i = 0; i < max_log; i++) {
            عائم cpu_val = get_float(ctx->cpu_dst_داده, i, گره->type);
            عائم et_val  = get_float(ctx->et_dst_داده, i, گره->type);
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
        إذا (رقم_elements > max_log) {
            size_t mid     = رقم_elements / 2;
            size_t end     = رقم_elements - 1;
            عائم  cpu_mid = get_float(ctx->cpu_dst_داده, mid, گره->type);
            عائم  et_mid  = get_float(ctx->et_dst_داده, mid, گره->type);
            عائم  cpu_انتها = get_float(ctx->cpu_dst_داده, end, گره->type);
            عائم  et_انتها  = get_float(ctx->et_dst_داده, end, گره->type);

            GGML_LOG_DEBUG("ET: Middle element [%zu]: CPU=%.6f, ET=%.6f\n", mid, cpu_mid, et_mid);
            GGML_LOG_DEBUG("ET: Last element [%zu]: CPU=%.6f, ET=%.6f\n", end, cpu_انتها, et_انتها);
        }

        GGML_LOG_DEBUG("ET: Results %s (%zu/%zu elements match within tolerance %.6f)\n", matches ? "MATCH" : "DIFFER",
                       رقم_elements - مجموع_mismatches, رقم_elements, config->tolerance);
    }

    // Copy CPU result to device if flag is set
    إذا (config->use_cpu_نتیجه) {
        GGML_LOG_DEBUG("ET: Overwriting ET device result with CPU result for correct inference\n");
        size_t dst_logical_اندازه = ggml_nbytes(گره);
        ggml_backend_tensor_مجموعه(const_cast<ggml_tensor *>(گره), ctx->cpu_dst_داده, 0, dst_logical_اندازه);
        GGML_LOG_DEBUG("ET: CPU result copied to ET device buffer\n");
    }

    أرجع درست;
}

فراغ ggml_et_cpu_compare_free(ggml_et_cpu_compare_ctx * ctx) {
    إذا (!ctx) {
        أرجع;
    }

    إذا (ctx->cpu_src0_داده) {
        حرر(ctx->cpu_src0_داده);
        ctx->cpu_src0_داده = اشاره_تهی;
    }
    إذا (ctx->cpu_src1_داده) {
        حرر(ctx->cpu_src1_داده);
        ctx->cpu_src1_داده = اشاره_تهی;
    }
    إذا (ctx->cpu_src2_داده) {
        حرر(ctx->cpu_src2_داده);
        ctx->cpu_src2_داده = اشاره_تهی;
    }
    إذا (ctx->cpu_dst_داده) {
        حرر(ctx->cpu_dst_داده);
        ctx->cpu_dst_داده = اشاره_تهی;
    }
    إذا (ctx->et_dst_داده) {
        حرر(ctx->et_dst_داده);
        ctx->et_dst_داده = اشاره_تهی;
    }

    إذا (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = اشاره_تهی;
    }

    إذا (ctx->cpu_backend) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = اشاره_تهی;
    }

    // Clear pointers
    ctx->cpu_src0  = اشاره_تهی;
    ctx->cpu_src1  = اشاره_تهی;
    ctx->cpu_src2  = اشاره_تهی;
    ctx->cpu_dst   = اشاره_تهی;
    ctx->cpu_گراف = اشاره_تهی;
}
